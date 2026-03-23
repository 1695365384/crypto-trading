"""可视化模块"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from evaluation.backtest import BacktestResult


class Visualizer:
    """回测可视化工具"""

    def __init__(self, figsize: Tuple[int, int] = (14, 10), style: str = "seaborn-v0_8-darkgrid"):
        """初始化可视化器"""
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use("seaborn-v0_8-whitegrid")
        self.figsize = figsize

    def plot_portfolio_value(
        self,
        result: BacktestResult,
        title: str = "Portfolio Value",
        save_path: Optional[str] = None,
    ):
        """绘制组合价值曲线"""
        fig, ax = plt.subplots(figsize=self.figsize)

        values = np.array(result.portfolio_values)
        ax.plot(values, linewidth=2, label="Portfolio Value")
        ax.axhline(y=values[0], color="gray", linestyle="--", alpha=0.5, label="Initial Value")

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Value ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 添加统计信息
        stats_text = f"Total Return: {result.total_return:.2%}\nMax DD: {result.max_drawdown:.2%}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def plot_returns_distribution(
        self,
        result: BacktestResult,
        title: str = "Returns Distribution",
        save_path: Optional[str] = None,
    ):
        """绘制收益分布"""
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)

        returns = np.array(result.returns)

        # 直方图
        axes[0].hist(returns, bins=50, density=True, alpha=0.7, edgecolor="black")
        axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.5)
        axes[0].axvline(
            x=np.mean(returns), color="green", linestyle="-", label=f"Mean: {np.mean(returns):.4f}"
        )
        axes[0].set_title("Returns Histogram")
        axes[0].set_xlabel("Returns")
        axes[0].set_ylabel("Density")
        axes[0].legend()

        # 累积收益
        cumulative = np.cumprod(1 + returns)
        axes[1].plot(cumulative, linewidth=2)
        axes[1].set_title("Cumulative Returns")
        axes[1].set_xlabel("Time Steps")
        axes[1].set_ylabel("Cumulative Return")
        axes[1].axhline(y=1, color="gray", linestyle="--", alpha=0.5)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def plot_drawdown(
        self,
        result: BacktestResult,
        title: str = "Drawdown Analysis",
        save_path: Optional[str] = None,
    ):
        """绘制回撤分析"""
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        values = np.array(result.portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdown = (peak - values) / peak

        # 组合价值
        axes[0].plot(values, linewidth=2)
        axes[0].plot(peak, "--", alpha=0.5, label="Peak")
        axes[0].set_title("Portfolio Value")
        axes[0].set_ylabel("Value ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 回撤
        axes[1].fill_between(range(len(drawdown)), 0, -drawdown, alpha=0.5, color="red")
        axes[1].set_title("Drawdown")
        axes[1].set_xlabel("Time Steps")
        axes[1].set_ylabel("Drawdown")
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def plot_comparison(
        self,
        results: Dict[str, BacktestResult],
        title: str = "Strategy Comparison",
        save_path: Optional[str] = None,
    ):
        """绘制策略对比"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

        # 组合价值对比
        for (name, result), color in zip(results.items(), colors):
            axes[0, 0].plot(result.portfolio_values, label=name, linewidth=2, color=color)
        axes[0, 0].set_title("Portfolio Value")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 收益率对比
        for (name, result), color in zip(results.items(), colors):
            cumulative = np.cumprod(1 + np.array(result.returns))
            axes[0, 1].plot(cumulative, label=name, linewidth=2, color=color)
        axes[0, 1].set_title("Cumulative Returns")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 指标柱状图
        metrics = ["sharpe_ratio", "max_drawdown", "win_rate"]
        x = np.arange(len(metrics))
        width = 0.8 / len(results)

        for i, (name, result) in enumerate(results.items()):
            values = [
                getattr(result, metrics[0], 0),
                -getattr(result, metrics[1], 0),  # 负值因为回撤是负指标
                getattr(result, metrics[2], 0),
            ]
            axes[1, 0].bar(x + i * width, values, width, label=name, color=colors[i])

        axes[1, 0].set_xticks(x + width * (len(results) - 1) / 2)
        axes[1, 0].set_xticklabels(["Sharpe", "-Max DD", "Win Rate"])
        axes[1, 0].set_title("Key Metrics Comparison")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 总收益对比
        names = list(results.keys())
        returns = [results[n].total_return * 100 for n in names]
        axes[1, 1].bar(names, returns, color=colors)
        axes[1, 1].set_title("Total Return (%)")
        axes[1, 1].set_ylabel("Return (%)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def plot_training_progress(
        self,
        train_stats: List[Dict],
        val_stats: List[Dict],
        title: str = "Training Progress",
        save_path: Optional[str] = None,
    ):
        """绘制训练进度"""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # Actor Loss
        if train_stats and "actor_loss" in train_stats[0]:
            timesteps = [s["timestep"] for s in train_stats]
            actor_losses = [s["actor_loss"] for s in train_stats]
            axes[0, 0].plot(timesteps, actor_losses)
            axes[0, 0].set_title("Actor Loss")
            axes[0, 0].set_xlabel("Timesteps")
            axes[0, 0].grid(True, alpha=0.3)

        # Critic Loss
        if train_stats and "critic_loss" in train_stats[0]:
            timesteps = [s["timestep"] for s in train_stats]
            critic_losses = [s["critic_loss"] for s in train_stats]
            axes[0, 1].plot(timesteps, critic_losses)
            axes[0, 1].set_title("Critic Loss")
            axes[0, 1].set_xlabel("Timesteps")
            axes[0, 1].grid(True, alpha=0.3)

        # Validation Return
        if val_stats and "total_return" in val_stats[0]:
            val_returns = [s["total_return"] for s in val_stats]
            axes[1, 0].plot(val_returns)
            axes[1, 0].set_title("Validation Return")
            axes[1, 0].set_xlabel("Evaluation")
            axes[1, 0].grid(True, alpha=0.3)

        # Validation Sharpe
        if val_stats and "sharpe_ratio" in val_stats[0]:
            val_sharpes = [s["sharpe_ratio"] for s in val_stats]
            axes[1, 1].plot(val_sharpes)
            axes[1, 1].set_title("Validation Sharpe Ratio")
            axes[1, 1].set_xlabel("Evaluation")
            axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def generate_report(self, result: BacktestResult, output_dir: str, prefix: str = "backtest"):
        """生成完整的回测报告"""
        os.makedirs(output_dir, exist_ok=True)

        # 绘制所有图表
        self.plot_portfolio_value(
            result, save_path=os.path.join(output_dir, f"{prefix}_portfolio.png")
        )

        self.plot_returns_distribution(
            result, save_path=os.path.join(output_dir, f"{prefix}_returns.png")
        )

        self.plot_drawdown(result, save_path=os.path.join(output_dir, f"{prefix}_drawdown.png"))

        # 保存文本报告
        report_path = os.path.join(output_dir, f"{prefix}_report.txt")
        with open(report_path, "w") as f:
            f.write(result.summary())

        print(f"Report generated in: {output_dir}")
