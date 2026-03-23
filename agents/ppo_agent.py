"""PPO 智能体模块 - LSTM + MLP 架构

优化点:
1. 混合精度训练 (AMP)
2. 批量推理优化
3. 预分配观察张量
4. 共享 LSTM 编码器
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.networks import ActorNetwork, CriticNetwork, LSTMSharedEncoder
from config.config import ModelConfig, NetworkConfig

# 尝试导入混合精度训练
try:
    from torch.cuda.amp import GradScaler, autocast

    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False


class PPOAgent:
    """PPO 智能体 - LSTM + MLP 架构"""

    def __init__(
        self,
        config: ModelConfig,
        network_config: Optional[NetworkConfig] = None,
        device: str = "cuda:0",
        use_amp: bool = True,
    ):
        self.config = config
        self.network_config = network_config or NetworkConfig()
        self.use_amp = use_amp and AMP_AVAILABLE and torch.cuda.is_available()

        # 设备检测
        if device == "auto" or device == "mps":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.use_amp = False  # MPS 不支持 AMP
            else:
                self.device = torch.device("cpu")
                self.use_amp = False
        elif device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            self.use_amp = False

        # 网络初始化 (延迟到第一次看到环境)
        self.actor: Optional[ActorNetwork] = None
        self.critic: Optional[CriticNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None

        # LSTM 共享编码器
        self._lstm_shared_encoder: Optional[LSTMSharedEncoder] = None
        self._lookback_window: int = 0
        self._feature_dim: int = 0

        # 混合精度训练
        self.scaler = GradScaler() if self.use_amp else None

        # 预分配观察张量用于推理 (避免重复分配)
        self._obs_buffer: Optional[torch.Tensor] = None

    def init_networks(
        self,
        obs_dim: int,
        action_dim: int,
        lookback_window: int = 30,
        feature_dim: int = 0,
    ):
        """
        初始化网络

        Args:
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
            lookback_window: 回看窗口大小
            feature_dim: 每步特征维度 (0 表示自动计算)
        """
        self._lookback_window = lookback_window

        # 自动计算 feature_dim
        if feature_dim <= 0:
            # 排除账户状态 (最后 n_assets + 1 维)
            account_state_dim = action_dim + 1  # n_assets + balance
            feature_dim = (obs_dim - account_state_dim) // lookback_window

        self._feature_dim = feature_dim

        # 创建共享 LSTM 编码器
        self._lstm_shared_encoder = LSTMSharedEncoder(feature_dim, self.network_config)

        # 创建 Actor 和 Critic (共享 LSTM)
        self.actor = ActorNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            lookback_window=lookback_window,
            feature_dim=feature_dim,
            config=self.network_config,
            shared_encoder=self._lstm_shared_encoder,
        ).to(self.device)

        self.critic = CriticNetwork(
            obs_dim=obs_dim,
            lookback_window=lookback_window,
            feature_dim=feature_dim,
            config=self.network_config,
            shared_encoder=self._lstm_shared_encoder,
        ).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config.learning_rate,
        )

        # 预分配观察缓冲区
        self._obs_buffer = torch.empty(1, obs_dim, dtype=torch.float32, device=self.device)

    def compute_gae(
        self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, next_value: float
    ) -> torch.Tensor:
        """
        计算广义优势估计 (GAE) - 优化版本

        使用向量化计算替代循环
        """
        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)

        # 向量化计算
        next_values = np.concatenate([values[1:], [next_value]])
        deltas = rewards + self.config.gamma * next_values * (1 - dones) - values

        # 反向累积 GAE
        gae = 0
        for t in reversed(range(n)):
            gae = deltas[t] + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return torch.as_tensor(advantages, device=self.device)

    def update(self, batch: Tuple) -> Dict[str, float]:
        """
        PPO 更新 - 支持混合精度和 mini-batch

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

        # 使用 mini-batch 训练
        batch_size = obs.shape[0]
        mini_batch_size = min(self.config.batch_size, batch_size)
        n_mini_batches = max(1, batch_size // mini_batch_size)

        for _ in range(self.config.n_epochs):
            # 打乱数据
            indices = torch.randperm(batch_size, device=self.device)

            for i in range(n_mini_batches):
                start = i * mini_batch_size
                end = min(start + mini_batch_size, batch_size)
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # 使用混合精度训练
                if self.use_amp:
                    with autocast():
                        new_log_probs, entropy, _ = self.actor.evaluate_actions(mb_obs, mb_actions)
                        ratio = torch.exp(new_log_probs - mb_old_log_probs)

                        surr1 = ratio * mb_advantages
                        surr2 = (
                            torch.clamp(
                                ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                            )
                            * mb_advantages
                        )
                        actor_loss = -torch.min(surr1, surr2).mean()

                        value_pred, _ = self.critic(mb_obs)
                        value_pred = value_pred.squeeze()
                        critic_loss = nn.MSELoss()(value_pred, mb_returns)

                        loss = (
                            actor_loss
                            + self.config.value_coef * critic_loss
                            - self.config.entropy_coef * entropy.mean()
                        )

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.actor.parameters()) + list(self.critic.parameters()),
                        self.config.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    new_log_probs, entropy, _ = self.actor.evaluate_actions(mb_obs, mb_actions)
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)

                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                        * mb_advantages
                    )
                    actor_loss = -torch.min(surr1, surr2).mean()

                    value_pred, _ = self.critic(mb_obs)
                    value_pred = value_pred.squeeze()
                    critic_loss = nn.MSELoss()(value_pred, mb_returns)

                    loss = (
                        actor_loss
                        + self.config.value_coef * critic_loss
                        - self.config.entropy_coef * entropy.mean()
                    )

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

        n_updates = self.config.n_epochs * n_mini_batches
        return {
            "actor_loss": total_actor_loss / n_updates,
            "critic_loss": total_critic_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """
        获取动作 - 优化版本

        使用预分配的缓冲区避免重复内存分配

        Args:
            obs: 观察数组
            deterministic: 是否确定性策略

        Returns:
            (action, log_prob)
        """
        with torch.no_grad():
            # 直接创建tensor并传输，避免copy开销
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, _ = self.actor.get_action(obs_tensor, deterministic)

            action = action[0].cpu().numpy()
            log_prob = log_prob[0].item() if log_prob is not None else 0.0

        return action, log_prob

    def get_action_and_value(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        同时获取动作和价值 - 减少GPU调用次数

        Args:
            obs: 观察数组
            deterministic: 是否确定性策略

        Returns:
            (action, log_prob, value)
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, _ = self.actor.get_action(obs_tensor, deterministic)
            value, _ = self.critic(obs_tensor)

            action = action[0].cpu().numpy()
            log_prob = log_prob[0].item() if log_prob is not None else 0.0
            value = value[0].item()

        return action, log_prob, value

    def get_action_batch(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量获取动作 - 用于并行环境

        Args:
            obs: 观察数组 [batch, obs_dim]
            deterministic: 是否确定性策略

        Returns:
            (actions, log_probs)  [batch, action_dim], [batch]
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            actions, log_probs, _ = self.actor.get_action(obs_tensor, deterministic)

            actions = actions.cpu().numpy()
            log_probs = log_probs.cpu().numpy() if log_probs is not None else np.zeros(len(obs))

        return actions, log_probs

    def get_value(self, obs: np.ndarray) -> float:
        """
        获取价值估计

        Args:
            obs: 观察数组

        Returns:
            价值估计
        """
        with torch.no_grad():
            self._obs_buffer.copy_(torch.as_tensor(obs, dtype=torch.float32))
            value, _ = self.critic(self._obs_buffer)
            value = value.item()

        return value

    def get_value_batch(self, obs: np.ndarray) -> np.ndarray:
        """
        批量获取价值估计 - 用于并行环境

        Args:
            obs: 观察数组 [batch, obs_dim]

        Returns:
            价值估计 [batch]
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            values, _ = self.critic(obs_tensor)
            values = values.squeeze(-1).cpu().numpy()

        return values

    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "network_config": self.network_config.__dict__,
            "lookback_window": self._lookback_window,
            "feature_dim": self._feature_dim,
        }
        if self.scaler is not None:
            save_dict["scaler"] = self.scaler.state_dict()
        torch.save(save_dict, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
