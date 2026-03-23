"""回测模块"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np

from agents.ppo_agent import PPOAgent
from config.config import Config
from envs.crypto_env import CryptoTradingEnv


@dataclass
class BacktestResult:
    """回测结果"""

    # 收益指标
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # 风险指标
    max_drawdown: float
    volatility: float
    var_95: float
    cvar_95: float

    # 交易统计
    total_trades: int
    win_rate: float
    avg_trade_return: float
    profit_factor: float

    # 时序数据
    portfolio_values: List[float]
    returns: List[float]
    trades: List[Dict]

    # 时间信息
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_days: Optional[float] = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)

    def save(self, path: str):
        """保存结果"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    def summary(self) -> str:
        """生成摘要字符串"""
        return f"""
=== Backtest Results ===
Total Return: {self.total_return:.2%}
Annual Return: {self.annual_return:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Sortino Ratio: {self.sortino_ratio:.2f}
Calmar Ratio: {self.calmar_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}
Volatility: {self.volatility:.2%}
VaR (95%): {self.var_95:.2%}
CVaR (95%): {self.cvar_95:.2%}
Total Trades: {self.total_trades}
Win Rate: {self.win_rate:.2%}
Profit Factor: {self.profit_factor:.2f}
========================
"""


class Backtester:
    """回测引擎"""

    def __init__(self, config: Config):
        self.config = config

    def run(
        self, agent: PPOAgent, env: CryptoTradingEnv, deterministic: bool = True
    ) -> BacktestResult:
        """
        运行回测

        Args:
            agent: 训练好的智能体
            env: 测试环境
            deterministic: 是否使用确定性策略

        Returns:
            回测结果
        """
        # 重置环境
        obs, _ = env.reset()
        done = False

        portfolio_values = [env.portfolio_value]
        returns = []
        trades = []

        while not done:
            # 获取动作
            action, _ = agent.get_action(obs, deterministic=deterministic)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 记录
            portfolio_values.append(info["portfolio_value"])
            returns.append(info["returns"])

            # 记录交易
            if info["total_trades"] > len(trades):
                trades.append(
                    {
                        "step": info["step"],
                        "action": action.tolist(),
                        "portfolio_value": info["portfolio_value"],
                        "positions": info["positions"].tolist(),
                    }
                )

        # 计算指标
        return self._calculate_metrics(portfolio_values, returns, trades)

    def run_comparison(
        self,
        agent: PPOAgent,
        env: CryptoTradingEnv,
        benchmarks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, BacktestResult]:
        """
        运行对比回测

        Args:
            agent: 智能体
            env: 环境
            benchmarks: 基准策略 {'name': prices}

        Returns:
            各策略的回测结果
        """
        results = {}

        # 运行智能体
        results["agent"] = self.run(agent, env)

        # 运行基准
        if benchmarks:
            for name, prices in benchmarks.items():
                bench_env = CryptoTradingEnv(
                    self.config.env, prices.reshape(-1, 1), np.zeros((len(prices), 1, 1))
                )
                # 买入持有策略
                results[name] = self._buy_and_hold(bench_env)

        return results

    def _buy_and_hold(self, env: CryptoTradingEnv) -> BacktestResult:
        """买入持有策略"""
        obs, _ = env.reset()
        done = False

        portfolio_values = [env.portfolio_value]
        returns = []
        trades = []

        # 初始买入
        action = np.array([1.0])
        obs, _, _, _, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])

        # 持有
        while not done:
            action = np.array([0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            portfolio_values.append(info["portfolio_value"])
            returns.append(info["returns"])

        return self._calculate_metrics(portfolio_values, returns, trades)

    def _calculate_metrics(
        self, portfolio_values: List[float], returns: List[float], trades: List[Dict]
    ) -> BacktestResult:
        """计算回测指标"""
        values = np.array(portfolio_values)
        returns_arr = np.array(returns) if returns else np.array([0])

        # 收益率
        total_return = (values[-1] - values[0]) / values[0]

        # 年化收益率 (分钟级数据)
        n_minutes = len(values)
        annual_return = (1 + total_return) ** (252 * 24 * 60 / n_minutes) - 1

        # 夏普比率
        risk_free_rate = 0.02 / (252 * 24 * 60)  # 分钟级无风险利率
        excess_returns = returns_arr - risk_free_rate
        if np.std(excess_returns) > 0:
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252 * 24 * 60)
        else:
            sharpe_ratio = 0

        # Sortino 比率
        downside_returns = returns_arr[returns_arr < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = np.mean(returns_arr) / np.std(downside_returns) * np.sqrt(252 * 24 * 60)
        else:
            sortino_ratio = 0

        # 最大回撤
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak
        max_drawdown = np.max(drawdown)

        # Calmar 比率
        if max_drawdown > 0:
            calmar_ratio = annual_return / max_drawdown
        else:
            calmar_ratio = 0

        # 波动率
        volatility = np.std(returns_arr) * np.sqrt(252 * 24 * 60)

        # VaR 和 CVaR
        var_95 = np.percentile(returns_arr, 5)
        cvar_95 = (
            np.mean(returns_arr[returns_arr <= var_95])
            if len(returns_arr[returns_arr <= var_95]) > 0
            else var_95
        )

        # 胜率
        if len(trades) > 1:
            trade_values = [t["portfolio_value"] for t in trades]
            trade_rets = np.diff(trade_values) / trade_values[:-1]
            win_rate = np.mean(trade_rets > 0)
            avg_trade_return = np.mean(trade_rets)

            # 盈亏比
            profits = trade_rets[trade_rets > 0]
            losses = np.abs(trade_rets[trade_rets < 0])
            profit_factor = (
                np.sum(profits) / np.sum(losses) if len(losses) > 0 and np.sum(losses) > 0 else 0
            )
        else:
            win_rate = 0
            avg_trade_return = 0
            profit_factor = 0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            var_95=var_95,
            cvar_95=cvar_95,
            total_trades=len(trades),
            win_rate=win_rate,
            avg_trade_return=avg_trade_return,
            profit_factor=profit_factor,
            portfolio_values=portfolio_values,
            returns=returns,
            trades=trades,
        )
