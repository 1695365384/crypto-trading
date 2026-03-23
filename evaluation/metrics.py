"""评估指标模块"""

from typing import Dict, List

import numpy as np
from scipy import stats


def calculate_metrics(returns: np.ndarray, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    计算常用评估指标

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率 (年化)

    Returns:
        指标字典
    """
    if len(returns) == 0:
        return {}

    # 年化因子 (分钟级)
    annual_factor = 252 * 24 * 60
    daily_rf = risk_free_rate / annual_factor

    # 基本统计
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    # 年化收益和波动
    annual_return = mean_return * annual_factor
    annual_volatility = std_return * np.sqrt(annual_factor)

    # 夏普比率
    excess_returns = returns - daily_rf
    sharpe_ratio = (
        np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(annual_factor)
        if np.std(excess_returns) > 0
        else 0
    )

    # Sortino 比率
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
    sortino_ratio = (
        np.mean(excess_returns) / downside_std * np.sqrt(annual_factor) if downside_std > 0 else 0
    )

    # VaR
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)

    # CVaR (Expected Shortfall)
    cvar_95 = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else var_95
    cvar_99 = np.mean(returns[returns <= var_99]) if len(returns[returns <= var_99]) > 0 else var_99

    # 偏度和峰度
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)

    # 最大连续上涨/下跌
    max_consecutive_up = _max_consecutive(returns > 0)
    max_consecutive_down = _max_consecutive(returns < 0)

    return {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "var_95": var_95,
        "var_99": var_99,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "max_consecutive_up": max_consecutive_up,
        "max_consecutive_down": max_consecutive_down,
    }


def calculate_drawdown(values: np.ndarray) -> Dict[str, float]:
    """
    计算回撤相关指标

    Args:
        values: 组合价值序列

    Returns:
        回撤指标字典
    """
    if len(values) == 0:
        return {}

    # 计算峰值
    peak = np.maximum.accumulate(values)

    # 计算回撤
    drawdown = (peak - values) / peak

    # 最大回撤
    max_drawdown = np.max(drawdown)

    # 最大回撤持续时间
    max_dd_idx = np.argmax(drawdown)
    peak_idx = np.argmax(values[: max_dd_idx + 1])
    max_dd_duration = max_dd_idx - peak_idx

    # 平均回撤
    avg_drawdown = np.mean(drawdown[drawdown > 0]) if np.any(drawdown > 0) else 0

    # 回撤频率
    dd_count = np.sum(np.diff(drawdown > 0) > 0)

    return {
        "max_drawdown": max_drawdown,
        "max_dd_duration": max_dd_duration,
        "avg_drawdown": avg_drawdown,
        "dd_count": dd_count,
    }


def calculate_trade_metrics(trades: List[Dict]) -> Dict[str, float]:
    """
    计算交易相关指标

    Args:
        trades: 交易列表

    Returns:
        交易指标字典
    """
    if len(trades) < 2:
        return {
            "total_trades": len(trades),
            "win_rate": 0,
            "avg_trade_return": 0,
            "profit_factor": 0,
            "avg_winning_trade": 0,
            "avg_losing_trade": 0,
        }

    # 计算每笔交易的收益
    trade_values = [t.get("portfolio_value", 0) for t in trades]
    trade_returns = np.diff(trade_values) / np.array(trade_values[:-1])

    # 胜率
    winning_trades = trade_returns[trade_returns > 0]
    losing_trades = trade_returns[trade_returns < 0]
    win_rate = len(winning_trades) / len(trade_returns)

    # 平均收益
    avg_trade_return = np.mean(trade_returns)

    # 盈亏比
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
    avg_loss = np.mean(np.abs(losing_trades)) if len(losing_trades) > 0 else 0
    profit_factor = (
        np.sum(winning_trades) / np.sum(np.abs(losing_trades))
        if len(losing_trades) > 0
        else float("inf")
    )

    return {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "avg_trade_return": avg_trade_return,
        "profit_factor": profit_factor,
        "avg_winning_trade": avg_win,
        "avg_losing_trade": avg_loss,
    }


def calculate_benchmark_comparison(
    returns: np.ndarray, benchmark_returns: np.ndarray
) -> Dict[str, float]:
    """
    计算与基准的比较指标

    Args:
        returns: 策略收益
        benchmark_returns: 基准收益

    Returns:
        比较指标字典
    """
    if len(returns) != len(benchmark_returns):
        return {}

    # Alpha 和 Beta
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
    alpha = np.mean(returns) - beta * np.mean(benchmark_returns)

    # 信息比率
    active_return = returns - benchmark_returns
    tracking_error = np.std(active_return)
    information_ratio = (
        np.mean(active_return) / tracking_error * np.sqrt(252 * 24 * 60)
        if tracking_error > 0
        else 0
    )

    # 胜率 (相对于基准)
    outperformance = np.mean(returns > benchmark_returns)

    # 最大相对回撤
    cumulative_diff = np.cumprod(1 + active_return)
    peak = np.maximum.accumulate(cumulative_diff)
    relative_drawdown = np.max((peak - cumulative_diff) / peak)

    return {
        "alpha": alpha,
        "beta": beta,
        "information_ratio": information_ratio,
        "outperformance_rate": outperformance,
        "relative_drawdown": relative_drawdown,
    }


def _max_consecutive(condition: np.ndarray) -> int:
    """计算最大连续满足条件的次数"""
    max_count = 0
    current_count = 0

    for c in condition:
        if c:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count
