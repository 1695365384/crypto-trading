"""加密货币交易环境"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces

from config.config import EnvConfig

warnings.filterwarnings("ignore")


class CryptoTradingEnv:
    """加密货币交易环境 (Gymnasium 兼容)"""

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
            price_data: 价格数据 [timesteps, assets]
            feature_data: 特征数据 [timesteps, assets, features]
            feature_columns: 特征列名
            render_mode: 渲染模式
        """
        self.config = config
        self.price_data = price_data
        self.feature_data = feature_data
        self.feature_columns = feature_columns or []

        # 维度
        self.n_steps = price_data.shape[0]
        self.n_assets = price_data.shape[1]
        self.n_features = (
            feature_data.shape[2] if len(feature_data.shape) == 3 else feature_data.shape[1]
        )

        # 账户状态
        self.initial_amount = config.initial_amount
        self.balance = config.initial_amount
        self.positions = np.zeros(self.n_assets)  # 各币种持仓数量
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
        # 观察空间维度
        # [lookback_window * features * assets + positions + balance]
        obs_dim = (
            self.config.lookback_window * self.n_features * self.n_assets
            + self.n_assets  # 当前持仓比例
            + 1  # 当前余额比例
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 动作空间: 连续，范围 [-1, 1] for each asset
        # 正值表示买入比例，负值表示卖出比例
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        重置环境

        Args:
            seed: 随机种子
            options: 额外选项

        Returns:
            (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = self.config.lookback_window
        self.balance = self.initial_amount
        self.positions = np.zeros(self.n_assets)
        self.portfolio_value = self.initial_amount

        self.portfolio_history = [self.portfolio_value]
        self.returns_history = []
        self.trade_history = []
        self.action_history = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步交易

        Args:
            action: 交易动作 [-1, 1] for each asset

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # 1. 记录动作
        action = np.clip(action, -1, 1)
        self.action_history.append(action.copy())

        # 2. 执行交易
        trade_info = self._execute_trades(action)
        if trade_info:
            self.trade_history.append(trade_info)

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

        # 8. 获取新观察和信息
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """获取当前观察"""
        # 获取历史窗口数据
        window_start = self.current_step - self.config.lookback_window

        if len(self.feature_data.shape) == 3:
            # [timesteps, assets, features]
            window_data = self.feature_data[window_start : self.current_step]
            # 展平为 [lookback * assets * features]
            window_flat = window_data.flatten()
        else:
            # [timesteps, features]
            window_data = self.feature_data[window_start : self.current_step]
            window_flat = np.tile(window_data.flatten(), self.n_assets)

        # 计算持仓比例
        current_prices = self.price_data[self.current_step]
        position_values = self.positions * current_prices
        position_pct = position_values / max(self.portfolio_value, 1e-8)

        # 余额比例
        balance_pct = self.balance / self.initial_amount

        # 组合观察
        obs = np.concatenate([window_flat, position_pct, [balance_pct]])

        return obs.astype(np.float32)

    def _execute_trades(self, actions: np.ndarray) -> Optional[Dict]:
        """
        执行交易

        Args:
            actions: 交易动作

        Returns:
            交易信息字典
        """
        current_prices = self.price_data[self.current_step]
        trades = []

        for i, action in enumerate(actions):
            price = current_prices[i]

            if action > 0.01:  # 买入
                # 计算买入金额
                buy_amount = self.balance * action * self.config.max_position_pct

                # 应用滑点
                actual_price = price * (1 + self.config.slippage_pct)

                # 计算手续费
                cost = buy_amount * (1 + self.config.transaction_cost_pct)

                if cost <= self.balance:
                    self.balance -= cost
                    shares = buy_amount / actual_price
                    self.positions[i] += shares

                    trades.append(
                        {
                            "type": "buy",
                            "asset": i,
                            "price": actual_price,
                            "shares": shares,
                            "cost": cost,
                        }
                    )

            elif action < -0.01:  # 卖出
                sell_ratio = -action
                sell_shares = self.positions[i] * sell_ratio

                if sell_shares > 0:
                    # 应用滑点
                    actual_price = price * (1 - self.config.slippage_pct)

                    # 计算收入
                    revenue = sell_shares * actual_price * (1 - self.config.transaction_cost_pct)

                    self.balance += revenue
                    self.positions[i] -= sell_shares

                    trades.append(
                        {
                            "type": "sell",
                            "asset": i,
                            "price": actual_price,
                            "shares": sell_shares,
                            "revenue": revenue,
                        }
                    )

        return {"step": self.current_step, "trades": trades} if trades else None

    def _calculate_portfolio_value(self) -> float:
        """计算当前组合价值"""
        current_prices = self.price_data[min(self.current_step, self.n_steps - 1)]
        positions_value = np.sum(self.positions * current_prices)
        return self.balance + positions_value

    def _calculate_reward(self, new_value: float) -> float:
        """
        计算奖励

        Args:
            new_value: 新的组合价值

        Returns:
            奖励值
        """
        # 收益率
        returns = (new_value - self.portfolio_value) / max(self.portfolio_value, 1e-8)

        # 基础奖励
        reward = returns

        # 风险惩罚 (可选)
        if len(self.returns_history) > 10:
            volatility = np.std(self.returns_history[-10:])
            reward -= 0.1 * volatility

        # 缩放
        reward = reward * self.config.reward_scaling

        return reward

    def _is_terminated(self) -> bool:
        """检查是否终止"""
        # 数据结束
        if self.current_step >= self.n_steps - 1:
            return True

        # 破产
        if self.portfolio_value < self.initial_amount * 0.1:
            return True

        return False

    def _get_info(self) -> Dict[str, Any]:
        """获取当前信息"""
        current_prices = self.price_data[min(self.current_step, self.n_steps - 1)]

        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "positions": self.positions.copy(),
            "position_values": self.positions * current_prices,
            "prices": current_prices.copy(),
            "total_trades": len(self.trade_history),
            "returns": self.returns_history[-1] if self.returns_history else 0,
        }

    def render(self):
        """渲染环境状态"""
        if self.render_mode == "human":
            info = self._get_info()
            print(
                f"Step: {info['step']}, "
                f"Portfolio: ${info['portfolio_value']:.2f}, "
                f"Balance: ${info['balance']:.2f}, "
                f"Return: {info['returns']*100:.2f}%"
            )

    def close(self):
        """关闭环境"""

    def get_portfolio_stats(self) -> Dict[str, float]:
        """获取组合统计信息"""
        if len(self.portfolio_history) < 2:
            return {}

        values = np.array(self.portfolio_history)
        returns = np.diff(values) / values[:-1]

        total_return = (values[-1] - values[0]) / values[0]

        # 夏普比率 (年化)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60)
        else:
            sharpe = 0

        # 最大回撤
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = np.max(drawdown)

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "final_value": values[-1],
            "total_trades": len(self.trade_history),
        }
