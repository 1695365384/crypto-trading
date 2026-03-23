"""全局配置模块"""

import os
from dataclasses import dataclass, field
from typing import List

import yaml


@dataclass
class DataConfig:
    """数据配置"""

    # 交易对
    tickers: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])

    # 时间范围
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"

    # K线周期 (分钟)
    timeframe: int = 1  # 1分钟

    # 技术指标
    indicators: List[str] = field(
        default_factory=lambda: [
            "macd",
            "rsi_14",
            "boll_ub",
            "boll_lb",
            "ema_20",
            "ema_50",
            "atr",
            "obv",
        ]
    )

    # 数据分割比例
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class EnvConfig:
    """环境配置"""

    # 初始资金
    initial_amount: float = 10000.0  # USDT

    # 交易限制
    max_position_pct: float = 0.5  # 单币种最大仓位 50%

    # 手续费
    transaction_cost_pct: float = 0.001  # 0.1%

    # 滑点
    slippage_pct: float = 0.0005  # 0.05%

    # 奖励缩放 (调整后让奖励信号更明显)
    reward_scaling: float = 1.0

    # 观察窗口
    lookback_window: int = 60  # 60分钟

    # 币种数量
    num_assets: int = 2

    # 止损止盈
    stop_loss_pct: float = 0.1  # 10% 止损
    take_profit_pct: float = 0.2  # 20% 止盈


@dataclass
class ModelConfig:
    """模型配置"""

    # 算法
    algorithm: str = "PPO"

    # 网络结构
    actor_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])
    critic_hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])

    # 超参数
    learning_rate: float = 3e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.02  # 增加探索，防止过早收敛
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # 训练
    batch_size: int = 64
    buffer_size: int = 2048
    n_epochs: int = 10
    total_timesteps: int = 1_000_000

    # 保存频率
    save_freq: int = 50000
    eval_freq: int = 10000


@dataclass
class Config:
    """总配置"""

    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # 路径
    data_dir: str = "./data"
    model_dir: str = "./models"
    log_dir: str = "./logs"

    # 设备 (auto 会自动选择 MPS > CUDA > CPU)
    device: str = "auto"

    # 随机种子
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """从 YAML 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        data_config = DataConfig(**config_dict.get("data", {}))
        env_config = EnvConfig(**config_dict.get("env", {}))
        model_config = ModelConfig(**config_dict.get("model", {}))

        return cls(
            data=data_config,
            env=env_config,
            model=model_config,
            data_dir=config_dict.get("data_dir", "./data"),
            model_dir=config_dict.get("model_dir", "./models"),
            log_dir=config_dict.get("log_dir", "./logs"),
            device=config_dict.get("device", "cuda:0"),
            seed=config_dict.get("seed", 42),
        )

    def to_yaml(self, path: str):
        """保存配置到 YAML 文件"""
        config_dict = {
            "data": self.data.__dict__,
            "env": self.env.__dict__,
            "model": self.model.__dict__,
            "data_dir": self.data_dir,
            "model_dir": self.model_dir,
            "log_dir": self.log_dir,
            "device": self.device,
            "seed": self.seed,
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
