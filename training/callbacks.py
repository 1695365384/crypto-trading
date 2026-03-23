"""训练回调模块"""

import os
from typing import Any, Dict

import numpy as np


class TrainingCallback:
    """训练回调基类"""

    def on_step(self, step: int, info: Dict[str, Any]):
        """每步调用"""

    def on_epoch(self, epoch: int, info: Dict[str, Any]):
        """每个 epoch 调用"""

    def on_train_end(self, info: Dict[str, Any]):
        """训练结束时调用"""


class EarlyStoppingCallback(TrainingCallback):
    """早停回调"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001, metric: str = "val_reward"):
        """
        初始化早停回调

        Args:
            patience: 容忍次数
            min_delta: 最小改进阈值
            metric: 监控指标
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_value = -np.inf
        self.wait = 0
        self.should_stop = False

    def on_epoch(self, epoch: int, info: Dict[str, Any]):
        """检查是否应该停止"""
        val_stats = info.get("val_stats")
        if val_stats is None:
            return  # 跳过没有验证数据的 epoch

        current_value = val_stats.get(self.metric, -np.inf)

        if current_value - self.best_value > self.min_delta:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.should_stop = True
            print(
                f"Early stopping triggered: {self.metric} hasn't improved for {self.patience} epochs"
            )


class CheckpointCallback(TrainingCallback):
    """检查点回调"""

    def __init__(self, save_freq: int, save_path: str, save_best: bool = True):
        """
        初始化检查点回调

        Args:
            save_freq: 保存频率 (步数)
            save_path: 保存路径
            save_best: 是否保存最佳
        """
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_best = save_best
        self.best_value = -np.inf

        os.makedirs(save_path, exist_ok=True)

    def on_step(self, step: int, info: Dict[str, Any]):
        """定期保存"""
        if step % self.save_freq == 0:
            agent = info["agent"]
            path = os.path.join(self.save_path, f"checkpoint_{step}.pt")
            agent.save(path)
            print(f"Saved checkpoint to: {path}")


class TensorBoardCallback(TrainingCallback):
    """TensorBoard 回调"""

    def __init__(self, log_dir: str):
        """
        初始化 TensorBoard 回调

        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        self.writer = None

    def on_step(self, step: int, info: Dict[str, Any]):
        """记录步数指标"""
        if self.writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(self.log_dir)
            except ImportError:
                print("TensorBoard not available, skipping logging")
                return

        self.writer.add_scalar("train/reward", info.get("reward", 0), step)

    def on_epoch(self, epoch: int, info: Dict[str, Any]):
        """记录 epoch 指标"""
        if self.writer is None:
            return

        train_stats = info.get("train_stats", {})
        val_stats = info.get("val_stats")

        for key, value in train_stats.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{key}", value, epoch)

        if val_stats:  # 只有在有验证数据时才记录
            for key, value in val_stats.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"val/{key}", value, epoch)

    def on_train_end(self, info: Dict[str, Any]):
        """关闭 writer"""
        if self.writer:
            self.writer.close()


class ProgressCallback(TrainingCallback):
    """进度打印回调"""

    def __init__(self, print_freq: int = 1000):
        self.print_freq = print_freq

    def on_step(self, step: int, info: Dict[str, Any]):
        """打印进度"""
        if step % self.print_freq == 0:
            reward = info.get("reward", 0)
            print(f"Step {step}: reward={reward:.4f}")


class MetricsLoggerCallback(TrainingCallback):
    """指标记录回调"""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.metrics = []

    def on_epoch(self, epoch: int, info: Dict[str, Any]):
        """记录指标"""
        train_stats = info.get("train_stats", {})
        val_stats = info.get("val_stats", {})

        metric = {"epoch": epoch, "train": train_stats, "val": val_stats}
        self.metrics.append(metric)

    def on_train_end(self, info: Dict[str, Any]):
        """保存指标"""
        import json

        with open(self.log_path, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)
