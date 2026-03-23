"""风险管理模块"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RiskMetrics:
    """风险指标"""

    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    max_daily_loss: float = 0.0
    current_drawdown: float = 0.0
    position_concentration: float = 0.0
    leverage: float = 1.0


class RiskManager:
    """风险管理器"""

    def __init__(
        self,
        initial_capital: float,
        max_position_pct: float = 0.5,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.2,
        max_leverage: float = 1.0,
        risk_free_rate: float = 0.02,
    ):
        """
        初始化风险管理器

        Args:
            initial_capital: 初始资金
            max_position_pct: 单币种最大仓位比例
            max_daily_loss_pct: 最大日损失比例
            max_drawdown_pct: 最大回撤比例
            max_leverage: 最大杠杆
            risk_free_rate: 无风险利率
        """
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_leverage = max_leverage
        self.risk_free_rate = risk_free_rate

        # 状态追踪
        self.peak_value = initial_capital
        self.daily_start_value = initial_capital
        self.weekly_start_value = initial_capital
        self.daily_trades: List[Dict] = []
        self.position_history: List[np.ndarray] = []

        # 日期追踪
        self.last_reset_date = datetime.now().date()

    def check_trade(
        self,
        action: np.ndarray,
        current_positions: np.ndarray,
        current_prices: np.ndarray,
        current_balance: float,
    ) -> Tuple[bool, str]:
        """
        检查交易是否合规

        Args:
            action: 交易动作
            current_positions: 当前持仓
            current_prices: 当前价格
            current_balance: 当前余额

        Returns:
            (is_valid, reason)
        """
        # 检查动作范围
        if np.any(np.abs(action) > 1):
            return False, "Action out of range [-1, 1]"

        # 计算新仓位
        new_positions = current_positions + action * self.max_position_pct

        # 检查仓位限制
        if np.any(np.abs(new_positions) > self.max_position_pct):
            return False, f"Position exceeds max {self.max_position_pct:.0%}"

        # 计算新组合价值
        current_value = current_balance + np.sum(current_positions * current_prices)

        # 检查日损失限制
        daily_loss = (current_value - self.daily_start_value) / self.daily_start_value
        if daily_loss < -self.max_daily_loss_pct:
            return False, f"Daily loss limit reached: {daily_loss:.2%}"

        # 检查最大回撤
        if current_value > self.peak_value:
            self.peak_value = current_value

        current_drawdown = (self.peak_value - current_value) / self.peak_value
        if current_drawdown > self.max_drawdown_pct:
            return False, f"Max drawdown exceeded: {current_drawdown:.2%}"

        # 检查杠杆
        position_value = np.sum(np.abs(new_positions) * current_prices)
        leverage = position_value / current_value if current_value > 0 else 0
        if leverage > self.max_leverage:
            return False, f"Leverage exceeded: {leverage:.2f}x"

        return True, "Trade approved"

    def adjust_action(
        self,
        action: np.ndarray,
        current_positions: np.ndarray,
        current_prices: np.ndarray,
        current_balance: float,
    ) -> np.ndarray:
        """
        调整动作以符合风险限制

        Args:
            action: 原始动作
            current_positions: 当前持仓
            current_prices: 当前价格
            current_balance: 当前余额

        Returns:
            调整后的动作
        """
        adjusted = action.copy()

        # 计算新仓位
        new_positions = current_positions + adjusted * self.max_position_pct

        # 限制仓位
        for i in range(len(new_positions)):
            if np.abs(new_positions[i]) > self.max_position_pct:
                sign = np.sign(new_positions[i])
                new_positions[i] = sign * self.max_position_pct
                adjusted[i] = (new_positions[i] - current_positions[i]) / self.max_position_pct

        return adjusted

    def update(self, portfolio_value: float, positions: np.ndarray, trade: Optional[Dict] = None):
        """
        更新风险管理状态

        Args:
            portfolio_value: 当前组合价值
            positions: 当前持仓
            trade: 交易信息
        """
        # 更新峰值
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # 记录交易
        if trade:
            self.daily_trades.append(trade)

        # 记录持仓
        self.position_history.append(positions.copy())

        # 检查日期重置
        today = datetime.now().date()
        if today != self.last_reset_date:
            self._reset_daily()

    def _reset_daily(self):
        """重置每日统计"""
        self.daily_start_value = self.peak_value
        self.daily_trades = []

        # 检查是否需要周重置
        today = datetime.now().date()
        if today.weekday() == 0:  # Monday
            self.weekly_start_value = self.peak_value

        self.last_reset_date = today

    def get_metrics(
        self, portfolio_value: float, positions: np.ndarray, current_prices: np.ndarray
    ) -> RiskMetrics:
        """
        获取当前风险指标

        Args:
            portfolio_value: 组合价值
            positions: 持仓
            current_prices: 当前价格

        Returns:
            风险指标
        """
        # 日/周 PnL
        daily_pnl = (portfolio_value - self.daily_start_value) / self.daily_start_value
        weekly_pnl = (portfolio_value - self.weekly_start_value) / self.weekly_start_value

        # 当前回撤
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value

        # 仓位集中度
        position_values = np.abs(positions * current_prices)
        total_position = np.sum(position_values)
        if total_position > 0:
            position_concentration = np.max(position_values) / total_position
        else:
            position_concentration = 0

        # 杠杆
        leverage = total_position / portfolio_value if portfolio_value > 0 else 0

        return RiskMetrics(
            daily_pnl=daily_pnl,
            weekly_pnl=weekly_pnl,
            max_daily_loss=self.max_daily_loss_pct,
            current_drawdown=current_drawdown,
            position_concentration=position_concentration,
            leverage=leverage,
        )

    def should_stop_trading(self, portfolio_value: float) -> Tuple[bool, str]:
        """
        检查是否应该停止交易

        Args:
            portfolio_value: 当前组合价值

        Returns:
            (should_stop, reason)
        """
        # 检查日损失
        daily_loss = (portfolio_value - self.daily_start_value) / self.daily_start_value
        if daily_loss < -self.max_daily_loss_pct:
            return True, f"Daily loss limit reached: {daily_loss:.2%}"

        # 检查回撤
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        if current_drawdown > self.max_drawdown_pct:
            return True, f"Max drawdown exceeded: {current_drawdown:.2%}"

        return False, ""


class PositionSizer:
    """仓位计算器"""

    def __init__(
        self, method: str = "kelly", max_position: float = 0.5, risk_per_trade: float = 0.02
    ):
        """
        初始化仓位计算器

        Args:
            method: 计算方法 ('kelly', 'fixed', 'volatility')
            max_position: 最大仓位
            risk_per_trade: 每笔交易风险
        """
        self.method = method
        self.max_position = max_position
        self.risk_per_trade = risk_per_trade

    def calculate(
        self,
        confidence: float,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
    ) -> float:
        """
        计算仓位大小

        Args:
            confidence: 策略置信度
            volatility: 波动率
            win_rate: 胜率

        Returns:
            仓位比例
        """
        if self.method == "kelly" and win_rate is not None:
            # Kelly Criterion
            # f = (p * b - q) / b
            # p = win_rate, q = 1 - p, b = win/loss ratio
            kelly = 2 * win_rate - 1  # 简化假设
            position = kelly * confidence

        elif self.method == "volatility" and volatility is not None:
            # 波动率倒数
            position = self.risk_per_trade / volatility * confidence

        else:
            # 固定比例
            position = self.risk_per_trade * confidence

        return float(np.clip(position, 0, self.max_position))
