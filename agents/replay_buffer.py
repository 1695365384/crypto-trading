"""经验回放缓冲区"""

from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """PPO 经验回放缓冲区"""

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, device: str = "cuda:0"):
        """
        初始化缓冲区

        Args:
            buffer_size: 缓冲区大小
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
            device: 设备
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 初始化缓冲区
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """
        添加一条经验

        Args:
            obs: 观察
            action: 动作
            reward: 奖励
            value: 价值估计
            log_prob: 对数概率
            done: 是否终止
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def get(self) -> Tuple[torch.Tensor, ...]:
        """
        获取所有数据

        Returns:
            (obs, actions, rewards, values, log_probs, dones)
        """
        indices = np.arange(self.size)

        return (
            torch.FloatTensor(self.observations[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.values[indices]).to(self.device),
            torch.FloatTensor(self.log_probs[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device),
        )

    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0

    def is_full(self) -> bool:
        """检查是否已满"""
        return self.size >= self.buffer_size

    def __len__(self) -> int:
        return self.size


class RolloutBuffer:
    """用于收集完整轨迹的缓冲区"""

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ):
        """添加经验"""
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def get(self) -> Tuple[torch.Tensor, ...]:
        """获取所有数据"""
        return (
            torch.FloatTensor(np.array(self.observations)).to(self.device),
            torch.FloatTensor(np.array(self.actions)).to(self.device),
            torch.FloatTensor(np.array(self.rewards)).to(self.device),
            torch.FloatTensor(np.array(self.values)).to(self.device),
            torch.FloatTensor(np.array(self.log_probs)).to(self.device),
            torch.FloatTensor(np.array(self.dones)).to(self.device),
        )

    def clear(self):
        """清空缓冲区"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def __len__(self) -> int:
        return len(self.observations)
