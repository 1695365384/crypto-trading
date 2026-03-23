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

    # 并行环境数量
    n_envs: int = 1


@dataclass
class NetworkConfig:
    """网络架构配置 - LSTM + MLP 混合架构"""

    # LSTM 配置
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.1

    # MLP 配置
    mlp_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128, 64])
    mlp_dropout: float = 0.2

    # 特征维度（自动计算，不需要手动设置）
    feature_dim_per_step: int = 0  # 0 表示自动计算


@dataclass
class ModelConfig:
    """模型配置"""

    # 算法
    algorithm: str = "PPO"

    # 网络架构配置
    network: NetworkConfig = field(default_factory=NetworkConfig)

    # 网络结构 (保留用于配置兼容性)
    actor_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])
    critic_hidden_sizes: List[int] = field(default_factory=lambda: [256, 128])

    # 超参数
    learning_rate: float = 3e-5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.02  # 增加探索，防止过早收敛
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # 训练
    batch_size: int = 256
    buffer_size: int = 8192
    n_epochs: int = 10
    total_timesteps: int = 1_000_000

    # 保存频率
    save_freq: int = 50000
    eval_freq: int = 10000

    # 性能优化参数
    n_envs: int = 4  # 并行环境数量
    use_amp: bool = True  # 混合精度训练
    pin_memory: bool = True  # 加速CPU-GPU数据传输


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

        # 处理嵌套的 network 配置
        model_dict = config_dict.get("model", {})
        network_dict = model_dict.pop("network", {})
        network_config = NetworkConfig(**network_dict)
        model_config = ModelConfig(network=network_config, **model_dict)

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
        model_dict = {
            "algorithm": self.model.algorithm,
            "actor_hidden_sizes": self.model.actor_hidden_sizes,
            "critic_hidden_sizes": self.model.critic_hidden_sizes,
            "learning_rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "clip_ratio": self.model.clip_ratio,
            "entropy_coef": self.model.entropy_coef,
            "value_coef": self.model.value_coef,
            "max_grad_norm": self.model.max_grad_norm,
            "batch_size": self.model.batch_size,
            "buffer_size": self.model.buffer_size,
            "n_epochs": self.model.n_epochs,
            "total_timesteps": self.model.total_timesteps,
            "save_freq": self.model.save_freq,
            "eval_freq": self.model.eval_freq,
            "n_envs": self.model.n_envs,
            "use_amp": self.model.use_amp,
            "pin_memory": self.model.pin_memory,
            "network": {
                "lstm_hidden_size": self.model.network.lstm_hidden_size,
                "lstm_num_layers": self.model.network.lstm_num_layers,
                "lstm_dropout": self.model.network.lstm_dropout,
                "mlp_hidden_sizes": self.model.network.mlp_hidden_sizes,
                "mlp_dropout": self.model.network.mlp_dropout,
                "feature_dim_per_step": self.model.network.feature_dim_per_step,
            },
        }

        config_dict = {
            "data": self.data.__dict__,
            "env": self.env.__dict__,
            "model": model_dict,
            "data_dir": self.data_dir,
            "model_dir": self.model_dir,
            "log_dir": self.log_dir,
            "device": self.device,
            "seed": self.seed,
        }

        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
