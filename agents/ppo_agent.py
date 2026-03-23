"""PPO 智能体模块"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from config.config import ModelConfig


class ActorNetwork(nn.Module):
    """策略网络 (Actor)"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()

        # 构建隐藏层
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

        # 输出均值和标准差
        self.mean_head = nn.Linear(prev_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        # 输出层使用更小的初始化
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            obs: 观察张量 [batch, obs_dim]

        Returns:
            (mean, std) 动作分布参数
        """
        x = self.shared(obs)
        mean = torch.tanh(self.mean_head(x))  # 限制在 [-1, 1]
        std = torch.exp(self.log_std.clamp(-20, 2))
        return mean, std

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        获取动作

        Args:
            obs: 观察张量
            deterministic: 是否确定性策略

        Returns:
            (action, log_prob)
        """
        mean, std = self.forward(obs)

        if deterministic:
            return mean, None

        dist = Normal(mean, std)
        action = dist.rsample()  # 重参数化采样
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作

        Args:
            obs: 观察张量
            actions: 动作张量

        Returns:
            (log_prob, entropy)
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class CriticNetwork(nn.Module):
    """价值网络 (Critic)"""

    def __init__(self, obs_dim: int, hidden_sizes: List[int]):
        super().__init__()

        # 构建隐藏层
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

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            obs: 观察张量 [batch, obs_dim]

        Returns:
            价值估计 [batch, 1]
        """
        return self.network(obs)


class PPOAgent:
    """PPO 智能体"""

    def __init__(self, config: ModelConfig, device: str = "cuda:0"):
        self.config = config
        # 正确检测设备：支持 CUDA、MPS (Apple Silicon) 和 CPU
        if device == "auto" or device == "mps":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        elif device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        # 网络初始化 (延迟到第一次看到环境)
        self.actor: Optional[ActorNetwork] = None
        self.critic: Optional[CriticNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None

    def init_networks(self, obs_dim: int, action_dim: int):
        """
        初始化网络

        Args:
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
        """
        self.actor = ActorNetwork(obs_dim, action_dim, self.config.actor_hidden_sizes).to(
            self.device
        )

        self.critic = CriticNetwork(obs_dim, self.config.critic_hidden_sizes).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config.learning_rate,
        )

    def compute_gae(
        self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, next_value: float
    ) -> torch.Tensor:
        """
        计算广义优势估计 (GAE)

        Args:
            rewards: 奖励数组
            values: 价值估计数组
            dones: 终止标志数组
            next_value: 最后状态的下一价值

        Returns:
            优势张量
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32, device=self.device)

    def update(self, batch: Tuple) -> Dict[str, float]:
        """
        PPO 更新

        Args:
            batch: (obs, actions, old_log_probs, returns, advantages)

        Returns:
            训练指标字典
        """
        obs, actions, old_log_probs, returns, advantages = batch

        # 确保所有张量在正确设备上
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0

        for _ in range(self.config.n_epochs):
            # 计算新的 log_prob 和熵
            new_log_probs, entropy = self.actor.evaluate_actions(obs, actions)

            # 重要性采样比率
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO 目标 (Clipped Surrogate Objective)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            value_pred = self.critic(obs).squeeze()
            critic_loss = nn.MSELoss()(value_pred, returns)

            # 总损失
            loss = (
                actor_loss
                + self.config.value_coef * critic_loss
                - self.config.entropy_coef * entropy.mean()
            )

            # 梯度更新
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()

        return {
            "actor_loss": total_actor_loss / self.config.n_epochs,
            "critic_loss": total_critic_loss / self.config.n_epochs,
            "entropy": total_entropy / self.config.n_epochs,
        }

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        获取动作

        Args:
            obs: 观察数组
            deterministic: 是否确定性策略

        Returns:
            (action, log_prob)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob = self.actor.get_action(obs_tensor, deterministic)

            action = action.cpu().numpy()[0]
            log_prob = log_prob.item() if log_prob is not None else 0.0

        return action, log_prob

    def get_value(self, obs: np.ndarray) -> float:
        """
        获取价值估计

        Args:
            obs: 观察数组

        Returns:
            价值估计
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            value = self.critic(obs_tensor).item()

        return value

    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
