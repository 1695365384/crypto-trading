"""加密货币交易环境 - 优化版本

优化点:
1. 预计算flatten后的特征数据
2. 预分配观察数组
3. 减少不必要的数组复制
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from config.config import EnvConfig

warnings.filterwarnings("ignore")


class CryptoTradingEnv:
    """加密货币交易环境 - 优化版本"""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: EnvConfig,
        price_data: np.ndarray,
        feature_data: np.ndarray,
        feature_columns: Optional[List[str]] = None,
        render_mode: Optional[str] = None,
    ):
        """
        初始化交易环境

        Args:
            config: 环境配置
            price_data: 价格数据 [timesteps, assets] 或 [timesteps, 4] (OHLC)
            feature_data: 特征数据 [timesteps, assets, features] 或 [timesteps, features]
            feature_columns: 特征列名
            render_mode: 渲染模式
        """
        self.config = config
        self.feature_columns = feature_columns or []

        # 维度
        self.n_steps = price_data.shape[0]
        self.n_assets = price_data.shape[1] if len(price_data.shape) == 2 else 1

        # 处理特征数据
        if len(feature_data.shape) == 3:
            # [timesteps, assets, features]
            self.n_features = feature_data.shape[2]
            # 预计算所有时间步的flatten特征
            self._flat_features = feature_data.reshape(self.n_steps, -1).astype(np.float32)
        else:
            # [timesteps, features]
            self.n_features = feature_data.shape[1]
            # 复制到多个资产
            self._flat_features = np.tile(feature_data.astype(np.float32), (1, self.n_assets))

        # 存储价格数据 (只保留close价格)
        if len(price_data.shape) == 2 and price_data.shape[1] > 2:
            # OHLC格式，取close
            self.price_data = price_data[:, -1:None] if price_data.shape[1] == 4 else price_data
            if price_data.shape[1] == 4:
                self.price_data = np.column_stack([price_data[:, 3]] * self.n_assets)
            else:
                self.price_data = price_data
        else:
            self.price_data = price_data.astype(np.float32)

        # 计算观察维度
        self._window_size = config.lookback_window * self._flat_features.shape[1]
        self._obs_dim = self._window_size + self.n_assets + 1

        # 预分配观察数组
        self._obs_buffer = np.zeros(self._obs_dim, dtype=np.float32)

        # 账户状态
        self.initial_amount = config.initial_amount
        self.balance = config.initial_amount
        self.positions = np.zeros(self.n_assets, dtype=np.float32)
        self.portfolio_value = config.initial_amount

        # 历史记录
        self.portfolio_history: List[float] = []
        self.returns_history: List[float] = []
        self.trade_history: List[Dict] = []
        self.action_history: List[np.ndarray] = []

        # 当前步数
        self.current_step = 0

        # 渲染模式
        self.render_mode = render_mode

        # 定义空间
        self._define_spaces()

    def _define_spaces(self):
        """定义观察和动作空间"""
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.config.lookback_window
        self.balance = self.initial_amount
        self.positions.fill(0)
        self.portfolio_value = self.initial_amount

        self.portfolio_history = [self.portfolio_value]
        self.returns_history = []
        self.trade_history = []
        self.action_history = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步交易"""
        # 1. Clip动作
        action = np.clip(action, -1, 1)

        # 2. 执行交易
        self._execute_trades(action)

        # 3. 更新步数
        self.current_step += 1

        # 4. 计算新的组合价值
        new_portfolio_value = self._calculate_portfolio_value()

        # 5. 计算奖励
        reward = self._calculate_reward(new_portfolio_value)

        # 6. 记录历史
        returns = (new_portfolio_value - self.portfolio_value) / self.portfolio_value
        self.returns_history.append(returns)
        self.portfolio_value = new_portfolio_value
        self.portfolio_history.append(self.portfolio_value)

        # 7. 检查终止条件
        terminated = self._is_terminated()
        truncated = False

        # 8. 获取新观察
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """获取当前观察 - 优化版本"""
        window_start = self.current_step - self.config.lookback_window

        # 使用预计算的flatten特征
        window_flat = self._flat_features[window_start : self.current_step].flatten()

        # 计算持仓比例
        current_prices = self.price_data[self.current_step]
        position_values = self.positions * current_prices
        position_pct = position_values / max(self.portfolio_value, 1e-8)

        # 余额比例
        balance_pct = self.balance / self.initial_amount

        # 直接填充预分配的buffer
        self._obs_buffer[: len(window_flat)] = window_flat
        self._obs_buffer[len(window_flat) : len(window_flat) + self.n_assets] = position_pct
        self._obs_buffer[-1] = balance_pct

        return self._obs_buffer

    def _execute_trades(self, actions: np.ndarray):
        """执行交易 - 简化版本"""
        current_prices = self.price_data[self.current_step]

        for i, action in enumerate(actions):
            price = current_prices[i]

            if action > 0.01:  # 买入
                buy_amount = self.balance * action * self.config.max_position_pct
                actual_price = price * (1 + self.config.slippage_pct)
                cost = buy_amount * (1 + self.config.transaction_cost_pct)

                if cost <= self.balance:
                    self.balance -= cost
                    self.positions[i] += buy_amount / actual_price

            elif action < -0.01:  # 卖出
                sell_ratio = -action
                sell_shares = self.positions[i] * sell_ratio

                if sell_shares > 0:
                    actual_price = price * (1 - self.config.slippage_pct)
                    revenue = sell_shares * actual_price * (1 - self.config.transaction_cost_pct)
                    self.balance += revenue
                    self.positions[i] -= sell_shares

    def _calculate_portfolio_value(self) -> float:
        """计算当前组合价值"""
        current_prices = self.price_data[min(self.current_step, self.n_steps - 1)]
        positions_value = np.dot(self.positions, current_prices)
        return self.balance + positions_value

    def _calculate_reward(self, new_value: float) -> float:
        """计算奖励"""
        returns = (new_value - self.portfolio_value) / max(self.portfolio_value, 1e-8)
        reward = returns

        # 简单的风险惩罚
        if len(self.returns_history) > 10:
            reward -= 0.1 * np.std(self.returns_history[-10:])

        return reward * self.config.reward_scaling

    def _is_terminated(self) -> bool:
        """检查是否终止"""
        return bool(
            self.current_step >= self.n_steps - 1
            or self.portfolio_value < self.initial_amount * 0.1
        )

    def _get_info(self) -> Dict[str, Any]:
        """获取当前信息"""
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "positions": self.positions.copy(),
            "returns": self.returns_history[-1] if self.returns_history else 0,
            "total_trades": len(self.trade_history),
        }

    def render(self):
        """渲染环境状态"""
        if self.render_mode == "human":
            info = self._get_info()
            print(
                f"Step: {info['step']}, "
                f"Portfolio: ${info['portfolio_value']:.2f}, "
                f"Return: {info['returns']*100:.2f}%"
            )

    def close(self):
        """关闭环境"""

    def get_portfolio_stats(self) -> Dict[str, float]:
        """获取组合统计信息"""
        if len(self.portfolio_history) < 2:
            return {
                "total_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "final_value": self.initial_amount,
                "total_trades": 0,
            }

        values = np.array(self.portfolio_history)
        returns = np.diff(values) / values[:-1]
        total_return = (values[-1] - values[0]) / values[0]

        sharpe = 0
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)

        peak = np.maximum.accumulate(values)
        max_drawdown = np.max((peak - values) / peak)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "final_value": values[-1],
            "total_trades": len(self.trade_history),
        }

    def get_obs_structure(self) -> Dict[str, int]:
        """
        获取观察空间的结构信息 (用于 LSTM 网络)

        Returns:
            包含 lookback_window, feature_dim, obs_dim 的字典
        """
        # 计算每步的特征维度
        # 观察 = [历史窗口特征 (flatten) + 持仓比例 (n_assets) + 余额比例 (1)]
        account_state_dim = self.n_assets + 1
        window_features_dim = self._obs_dim - account_state_dim
        feature_dim_per_step = window_features_dim // self.config.lookback_window

        return {
            "lookback_window": self.config.lookback_window,
            "feature_dim": feature_dim_per_step,
            "obs_dim": self._obs_dim,
            "n_assets": self.n_assets,
            "account_state_dim": account_state_dim,
        }
