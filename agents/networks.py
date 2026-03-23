"""网络结构模块"""

from typing import List, Tuple

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """策略网络"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()

        layers = []
        prev_size = obs_dim
        for size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, size),
                    nn.LayerNorm(size),
                    nn.ReLU(),
                ]
            )
            prev_size = size

        self.shared = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(obs)
        mean = torch.tanh(self.mean_head(x))
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std


class CriticNetwork(nn.Module):
    """价值网络"""

    def __init__(self, obs_dim: int, hidden_sizes: List[int]):
        super().__init__()

        layers = []
        prev_size = obs_dim
        for size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, size),
                    nn.LayerNorm(size),
                    nn.ReLU(),
                ]
            )
            prev_size = size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
