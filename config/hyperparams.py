"""超参数配置"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class HyperParams:
    """超参数管理"""

    # PPO 超参数搜索空间
    PPO_SEARCH_SPACE: Dict[str, Any] = field(default_factory=dict)

    # 预设配置
    PRESETS: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        self.PPO_SEARCH_SPACE = {
            "learning_rate": [1e-5, 3e-5, 1e-4, 3e-4],
            "gamma": [0.95, 0.99, 0.995],
            "gae_lambda": [0.9, 0.95, 0.99],
            "clip_ratio": [0.1, 0.2, 0.3],
            "entropy_coef": [0.001, 0.01, 0.1],
            "value_coef": [0.5, 1.0],
            "batch_size": [32, 64, 128],
            "n_epochs": [5, 10, 20],
        }

        self.PRESETS = {
            "conservative": {
                "learning_rate": 1e-5,
                "gamma": 0.995,
                "gae_lambda": 0.99,
                "clip_ratio": 0.1,
                "entropy_coef": 0.001,
                "value_coef": 0.5,
                "batch_size": 64,
                "n_epochs": 10,
            },
            "balanced": {
                "learning_rate": 3e-5,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_ratio": 0.2,
                "entropy_coef": 0.01,
                "value_coef": 0.5,
                "batch_size": 64,
                "n_epochs": 10,
            },
            "aggressive": {
                "learning_rate": 1e-4,
                "gamma": 0.95,
                "gae_lambda": 0.9,
                "clip_ratio": 0.3,
                "entropy_coef": 0.1,
                "value_coef": 1.0,
                "batch_size": 128,
                "n_epochs": 20,
            },
        }

    def get_preset(self, name: str) -> Dict[str, Any]:
        """获取预设配置"""
        if name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {name}. Available: {list(self.PRESETS.keys())}")
        return self.PRESETS[name].copy()

    def get_search_space(self, algorithm: str = "PPO") -> Dict[str, Any]:
        """获取超参数搜索空间"""
        if algorithm == "PPO":
            return self.PPO_SEARCH_SPACE.copy()
        raise ValueError(f"Unknown algorithm: {algorithm}")
