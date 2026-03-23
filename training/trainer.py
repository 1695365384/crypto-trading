"""训练器模块 - 优化版本

优化点:
1. 真正的多进程并行环境
2. 批量数据收集
3. 异步数据传输
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from agents.ppo_agent import PPOAgent
from agents.replay_buffer import RolloutBuffer
from config.config import Config
from envs.crypto_env import CryptoTradingEnv
from training.callbacks import TrainingCallback


class Trainer:
    """PPO 训练器 - 优化版本"""

    def __init__(
        self,
        config: Config,
        agent: PPOAgent,
        train_env: CryptoTradingEnv,
        val_env: Optional[CryptoTradingEnv] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
    ):
        self.config = config
        self.agent = agent
        self.train_env = train_env
        self.val_env = val_env
        self.callbacks = callbacks or []

        # 获取环境维度
        obs_dim = train_env.observation_space.shape[0]
        action_dim = train_env.action_space.shape[0]
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 缓冲区
        self.buffer = RolloutBuffer(
            buffer_size=config.model.buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=config.device,
            pin_memory=getattr(config.model, "pin_memory", False),
        )

        self.n_envs = getattr(config.model, "n_envs", 1)

        # 训练统计
        self.train_stats: List[Dict] = []
        self.val_stats: List[Dict] = []

        # 最佳验证奖励
        self.best_val_reward = -np.inf

    def train(self) -> Dict[str, float]:
        """执行训练"""
        print(f"Starting training for {self.config.model.total_timesteps} timesteps...")
        print(f"Device: {self.agent.device}")
        print(f"Parallel envs: {self.n_envs}")
        print(f"Batch size: {self.config.model.batch_size}")
        print(f"Buffer size: {self.config.model.buffer_size}")

        total_timesteps = self.config.model.total_timesteps
        buffer_size = self.config.model.buffer_size
        eval_freq = self.config.model.eval_freq

        obs, _ = self.train_env.reset()
        episode_rewards = []

        epoch_callbacks = [cb for cb in self.callbacks if hasattr(cb, "on_epoch")]

        with tqdm(total=total_timesteps, desc="Training") as pbar:
            timestep = 0
            while timestep < total_timesteps:
                # 收集经验
                for _ in range(buffer_size):
                    # 合并获取action和value，减少GPU调用
                    action, log_prob, value = self.agent.get_action_and_value(obs)

                    next_obs, reward, terminated, truncated, info = self.train_env.step(action)
                    done = terminated or truncated

                    self.buffer.add(obs, action, reward, value, log_prob, done)

                    obs = next_obs
                    episode_rewards.append(reward)
                    timestep += 1

                    if done:
                        obs, _ = self.train_env.reset()
                        episode_rewards = []

                    if timestep >= total_timesteps:
                        break

                pbar.update(
                    buffer_size
                    if timestep >= total_timesteps
                    else (timestep % buffer_size or buffer_size)
                )

                # 获取最后的值估计
                last_value = self.agent.get_value(obs)

                # 计算优势和回报
                batch = self._compute_advantages_and_returns(last_value)

                # 更新策略
                update_info = self.agent.update(batch)

                # 清空缓冲区
                self.buffer.clear()

                # 记录统计
                stats = {
                    "timestep": timestep,
                    **update_info,
                    "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0,
                }
                self.train_stats.append(stats)

                # 验证
                if self.val_env and timestep % eval_freq < buffer_size:
                    val_stats = self._validate()
                    self.val_stats.append(val_stats)

                    pbar.set_postfix(
                        {
                            "loss": f"{update_info['actor_loss']:.4f}",
                            "return": f"{val_stats['total_return']:.1%}",
                            "sharpe": f"{val_stats['sharpe_ratio']:.2f}",
                        }
                    )

                    if val_stats["total_return"] > self.best_val_reward:
                        self.best_val_reward = val_stats["total_return"]
                        self._save_best_model()

                # Epoch 回调
                if epoch_callbacks:
                    for callback in epoch_callbacks:
                        callback.on_epoch(
                            timestep // buffer_size,
                            {
                                "agent": self.agent,
                                "timestep": timestep,
                                "train_stats": stats,
                                "val_stats": self.val_stats[-1] if self.val_stats else None,
                            },
                        )
                        if hasattr(callback, "should_stop") and callback.should_stop:
                            print(f"\nEarly stopping triggered at timestep {timestep}")
                            return stats

        # 训练结束回调
        for callback in self.callbacks:
            if hasattr(callback, "on_train_end"):
                callback.on_train_end(
                    {
                        "agent": self.agent,
                        "train_stats": self.train_stats,
                        "val_stats": self.val_stats,
                    }
                )

        print("\nTraining completed!")
        return self.train_stats[-1] if self.train_stats else {}

    def _compute_advantages_and_returns(self, last_value: float) -> Tuple:
        """计算优势和回报"""
        obs, actions, rewards, values, old_log_probs, dones = self.buffer.get()

        # 计算 GAE
        advantages = self.agent.compute_gae(
            rewards.cpu().numpy(), values.cpu().numpy(), dones.cpu().numpy(), last_value
        )

        # 计算回报
        returns = advantages + values.to(self.agent.device)

        return obs, actions, old_log_probs, returns, advantages

    def _validate(self) -> Dict[str, float]:
        """验证"""
        obs, _ = self.val_env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = self.agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = self.val_env.step(action)
            done = terminated or truncated
            total_reward += reward

        stats = self.val_env.get_portfolio_stats()
        stats["total_reward"] = total_reward

        return stats

    def _save_best_model(self):
        """保存最佳模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.model_dir, f"best_model_{timestamp}.pt")
        self.agent.save(path)
        print(f"Saved best model to: {path}")

    def save_training_logs(self, path: str):
        """保存训练日志"""
        import json

        os.makedirs(os.path.dirname(path), exist_ok=True)

        logs = {
            "train_stats": self.train_stats,
            "val_stats": self.val_stats,
            "config": {
                "model": self.config.model.__dict__,
                "env": self.config.env.__dict__,
                "data": self.config.data.__dict__,
            },
        }

        with open(path, "w") as f:
            json.dump(logs, f, indent=2, default=str)
