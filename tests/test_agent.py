"""智能体测试"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import ActorNetwork, CriticNetwork, PPOAgent
from config.config import ModelConfig


class TestPPOAgent:

    @pytest.fixture
    def model_config(self):
        return ModelConfig(
            actor_hidden_sizes=[64, 32],
            critic_hidden_sizes=[64, 32],
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            n_epochs=4,
            batch_size=32,
        )

    @pytest.fixture
    def agent(self, model_config):
        agent = PPOAgent(model_config, device="cpu")
        agent.init_networks(obs_dim=100, action_dim=2)
        return agent

    def test_actor_network(self, model_config):
        """测试 Actor 网络"""
        actor = ActorNetwork(obs_dim=100, action_dim=2, hidden_sizes=[64, 32])

        obs = torch.randn(10, 100)
        mean, std = actor(obs)

        assert mean.shape == (10, 2)
        assert std.shape == (2,)
        assert torch.all(mean >= -1) and torch.all(mean <= 1)
        assert torch.all(std > 0)

    def test_critic_network(self, model_config):
        """测试 Critic 网络"""
        critic = CriticNetwork(obs_dim=100, hidden_sizes=[64, 32])

        obs = torch.randn(10, 100)
        value = critic(obs)

        assert value.shape == (10, 1)

    def test_agent_initialization(self, agent):
        """测试智能体初始化"""
        assert agent.actor is not None
        assert agent.critic is not None
        assert agent.optimizer is not None

    def test_get_action(self, agent):
        """测试获取动作"""
        obs = np.random.randn(100)

        action, log_prob = agent.get_action(obs, deterministic=False)

        assert action.shape == (2,)
        assert np.all(action >= -1) and np.all(action <= 1)
        assert isinstance(log_prob, float)

    def test_get_deterministic_action(self, agent):
        """测试确定性动作"""
        obs = np.random.randn(100)

        action1, _ = agent.get_action(obs, deterministic=True)
        action2, _ = agent.get_action(obs, deterministic=True)

        np.testing.assert_array_almost_equal(action1, action2)

    def test_get_value(self, agent):
        """测试获取价值"""
        obs = np.random.randn(100)

        value = agent.get_value(obs)

        assert isinstance(value, float)

    def test_compute_gae(self, agent):
        """测试 GAE 计算"""
        rewards = np.array([0.1, 0.2, -0.1, 0.3, 0.0])
        values = np.array([1.0, 1.1, 1.3, 1.2, 1.5])
        dones = np.array([0, 0, 0, 0, 0])
        next_value = 1.4

        advantages = agent.compute_gae(rewards, values, dones, next_value)

        assert advantages.shape == (5,)
        assert torch.is_tensor(advantages)

    def test_update(self, agent):
        """测试更新"""
        batch_size = 32
        obs_dim = 100
        action_dim = 2

        obs = torch.randn(batch_size, obs_dim)
        actions = torch.randn(batch_size, action_dim)
        old_log_probs = torch.randn(batch_size)
        returns = torch.randn(batch_size)
        advantages = torch.randn(batch_size)

        batch = (obs, actions, old_log_probs, returns, advantages)

        update_info = agent.update(batch)

        assert "actor_loss" in update_info
        assert "critic_loss" in update_info
        assert "entropy" in update_info

    def test_save_load(self, agent, tmp_path):
        """测试保存和加载"""
        save_path = tmp_path / "test_model.pt"

        # 保存
        agent.save(str(save_path))
        assert save_path.exists()

        # 创建新智能体并加载
        new_agent = PPOAgent(agent.config, device="cpu")
        new_agent.init_networks(obs_dim=100, action_dim=2)
        new_agent.load(str(save_path))

        # 验证动作一致
        obs = np.random.randn(100)
        action1, _ = agent.get_action(obs, deterministic=True)
        action2, _ = new_agent.get_action(obs, deterministic=True)

        np.testing.assert_array_almost_equal(action1, action2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
