"""经验回放缓冲区 - 优化版本"""

from typing import Optional, Tuple

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
    """
    用于收集完整轨迹的缓冲区 - 优化版本

    优化点:
    1. 预分配固定大小数组，避免动态扩容
    2. 支持 pin_memory 加速 CPU-GPU 传输
    3. 批量数据传输，减少同步开销
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda:0",
        pin_memory: bool = True,
    ):
        """
        初始化缓冲区

        Args:
            buffer_size: 缓冲区大小
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
            device: 设备
            pin_memory: 是否使用 pin memory 加速传输
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pin_memory = pin_memory and self.device.type == "cuda"

        # 预分配 numpy 数组 (比 Python list 快 10x+)
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        # GPU 上的缓存张量 (预分配，避免重复创建)
        self._gpu_obs: Optional[torch.Tensor] = None
        self._gpu_actions: Optional[torch.Tensor] = None
        self._gpu_rewards: Optional[torch.Tensor] = None
        self._gpu_values: Optional[torch.Tensor] = None
        self._gpu_log_probs: Optional[torch.Tensor] = None
        self._gpu_dones: Optional[torch.Tensor] = None

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
        """添加经验"""
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
        获取所有数据 - 优化版本

        使用 pin_memory 和非阻塞传输加速
        """
        indices = np.arange(self.size)

        # 使用 pin_memory 加速传输
        if self.pin_memory:
            # 创建 pinned tensors
            obs_tensor = torch.as_tensor(self.observations[indices], pin_memory=True)
            actions_tensor = torch.as_tensor(self.actions[indices], pin_memory=True)
            rewards_tensor = torch.as_tensor(self.rewards[indices], pin_memory=True)
            values_tensor = torch.as_tensor(self.values[indices], pin_memory=True)
            log_probs_tensor = torch.as_tensor(self.log_probs[indices], pin_memory=True)
            dones_tensor = torch.as_tensor(self.dones[indices], pin_memory=True)

            # 非阻塞传输到 GPU
            return (
                obs_tensor.to(self.device, non_blocking=True),
                actions_tensor.to(self.device, non_blocking=True),
                rewards_tensor.to(self.device, non_blocking=True),
                values_tensor.to(self.device, non_blocking=True),
                log_probs_tensor.to(self.device, non_blocking=True),
                dones_tensor.to(self.device, non_blocking=True),
            )
        else:
            return (
                torch.as_tensor(self.observations[indices]).to(self.device),
                torch.as_tensor(self.actions[indices]).to(self.device),
                torch.as_tensor(self.rewards[indices]).to(self.device),
                torch.as_tensor(self.values[indices]).to(self.device),
                torch.as_tensor(self.log_probs[indices]).to(self.device),
                torch.as_tensor(self.dones[indices]).to(self.device),
            )

    def get_minibatch(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        获取一个 mini-batch 用于训练

        Args:
            batch_size: mini-batch 大小

        Returns:
            (obs, actions, rewards, values, log_probs, dones)
        """
        indices = np.random.choice(self.size, batch_size, replace=False)

        if self.pin_memory:
            return (
                torch.as_tensor(self.observations[indices], pin_memory=True).to(
                    self.device, non_blocking=True
                ),
                torch.as_tensor(self.actions[indices], pin_memory=True).to(
                    self.device, non_blocking=True
                ),
                torch.as_tensor(self.rewards[indices], pin_memory=True).to(
                    self.device, non_blocking=True
                ),
                torch.as_tensor(self.values[indices], pin_memory=True).to(
                    self.device, non_blocking=True
                ),
                torch.as_tensor(self.log_probs[indices], pin_memory=True).to(
                    self.device, non_blocking=True
                ),
                torch.as_tensor(self.dones[indices], pin_memory=True).to(
                    self.device, non_blocking=True
                ),
            )
        else:
            return (
                torch.as_tensor(self.observations[indices]).to(self.device),
                torch.as_tensor(self.actions[indices]).to(self.device),
                torch.as_tensor(self.rewards[indices]).to(self.device),
                torch.as_tensor(self.values[indices]).to(self.device),
                torch.as_tensor(self.log_probs[indices]).to(self.device),
                torch.as_tensor(self.dones[indices]).to(self.device),
            )

    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size


class ParallelRolloutBuffer:
    """
    并行环境专用的 Rollout Buffer

    支持同时收集多个环境的经验
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        action_dim: int,
        n_envs: int,
        device: str = "cuda:0",
        pin_memory: bool = True,
    ):
        """
        初始化并行缓冲区

        Args:
            buffer_size: 每个环境的缓冲区大小
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
            n_envs: 并行环境数量
            device: 设备
            pin_memory: 是否使用 pin memory
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_envs = n_envs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.pin_memory = pin_memory and self.device.type == "cuda"

        # 预分配: [n_envs, buffer_size, dim]
        self.observations = np.zeros((n_envs, buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_envs, buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((n_envs, buffer_size), dtype=np.float32)
        self.values = np.zeros((n_envs, buffer_size), dtype=np.float32)
        self.log_probs = np.zeros((n_envs, buffer_size), dtype=np.float32)
        self.dones = np.zeros((n_envs, buffer_size), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs: np.ndarray,  # [n_envs, obs_dim]
        actions: np.ndarray,  # [n_envs, action_dim]
        rewards: np.ndarray,  # [n_envs]
        values: np.ndarray,  # [n_envs]
        log_probs: np.ndarray,  # [n_envs]
        dones: np.ndarray,  # [n_envs]
    ):
        """批量添加多个环境的经验"""
        self.observations[:, self.ptr] = obs
        self.actions[:, self.ptr] = actions
        self.rewards[:, self.ptr] = rewards
        self.values[:, self.ptr] = values
        self.log_probs[:, self.ptr] = log_probs
        self.dones[:, self.ptr] = dones

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def get(self) -> Tuple[torch.Tensor, ...]:
        """获取所有数据，展平为 [n_envs * buffer_size, ...]"""
        # 展平所有环境的数据
        obs_flat = self.observations[:, : self.size].reshape(-1, self.obs_dim)
        actions_flat = self.actions[:, : self.size].reshape(-1, self.action_dim)
        rewards_flat = self.rewards[:, : self.size].reshape(-1)
        values_flat = self.values[:, : self.size].reshape(-1)
        log_probs_flat = self.log_probs[:, : self.size].reshape(-1)
        dones_flat = self.dones[:, : self.size].reshape(-1)

        if self.pin_memory:
            return (
                torch.as_tensor(obs_flat, pin_memory=True).to(self.device, non_blocking=True),
                torch.as_tensor(actions_flat, pin_memory=True).to(self.device, non_blocking=True),
                torch.as_tensor(rewards_flat, pin_memory=True).to(self.device, non_blocking=True),
                torch.as_tensor(values_flat, pin_memory=True).to(self.device, non_blocking=True),
                torch.as_tensor(log_probs_flat, pin_memory=True).to(self.device, non_blocking=True),
                torch.as_tensor(dones_flat, pin_memory=True).to(self.device, non_blocking=True),
            )
        else:
            return (
                torch.as_tensor(obs_flat).to(self.device),
                torch.as_tensor(actions_flat).to(self.device),
                torch.as_tensor(rewards_flat).to(self.device),
                torch.as_tensor(values_flat).to(self.device),
                torch.as_tensor(log_probs_flat).to(self.device),
                torch.as_tensor(dones_flat).to(self.device),
            )

    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size * self.n_envs
