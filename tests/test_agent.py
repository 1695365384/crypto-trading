"""智能体测试 - LSTM + MLP 架构"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.networks import ActorNetwork, CriticNetwork, LSTMSharedEncoder
from agents.ppo_agent import PPOAgent
from config.config import ModelConfig, NetworkConfig

# 测试常量
LOOKBACK_WINDOW = 30
FEATURE_DIM = 51
N_ASSETS = 2
ACCOUNT_STATE_DIM = N_ASSETS + 1  # n_assets + balance
OBS_DIM = LOOKBACK_WINDOW * FEATURE_DIM + ACCOUNT_STATE_DIM  # 30 * 51 + 2 = 1552
ACTION_DIM = N_ASSETS


class TestNetworks:
    """测试网络结构"""

    @pytest.fixture
    def network_config(self):
        return NetworkConfig(
            lstm_hidden_size=64,  # 测试用小模型
            lstm_num_layers=1,
            lstm_dropout=0.0,  # 测试时禁用 dropout
            mlp_hidden_sizes=[32, 16],
            mlp_dropout=0.0,
        )

    def test_lstm_shared_encoder(self, network_config):
        """测试共享 LSTM 编码器"""
        feature_dim = 10
        encoder = LSTMSharedEncoder(feature_dim, network_config)

        # 输入: [batch, seq_len, feature_dim]
        x = torch.randn(4, 30, feature_dim)
        output, hidden = encoder(x)

        assert output.shape == (4, network_config.lstm_hidden_size)
        assert hidden is not None

    def test_actor_network(self, network_config):
        """测试 Actor 网络"""
        actor = ActorNetwork(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            lookback_window=LOOKBACK_WINDOW,
            feature_dim=FEATURE_DIM,
            config=network_config,
        )

        # 测试前向传播
        obs = torch.randn(4, OBS_DIM)
        mean, std, hidden = actor(obs)

        assert mean.shape == (4, ACTION_DIM)
        assert std.shape == (ACTION_DIM,)
        assert torch.all(mean >= -1) and torch.all(mean <= 1)
        assert torch.all(std > 0)

    def test_actor_get_action(self, network_config):
        """测试 Actor 获取动作"""
        actor = ActorNetwork(
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            lookback_window=LOOKBACK_WINDOW,
            feature_dim=FEATURE_DIM,
            config=network_config,
        )

        obs = torch.randn(4, OBS_DIM)
        action, log_prob, hidden = actor.get_action(obs, deterministic=False)

        assert action.shape == (4, ACTION_DIM)
        assert torch.all(action >= -1) and torch.all(action <= 1)
        assert log_prob is not None
        assert log_prob.shape == (4,)

    def test_critic_network(self, network_config):
        """测试 Critic 网络"""
        critic = CriticNetwork(
            obs_dim=OBS_DIM,
            lookback_window=LOOKBACK_WINDOW,
            feature_dim=FEATURE_DIM,
            config=network_config,
        )

        obs = torch.randn(4, OBS_DIM)
        value, hidden = critic(obs)

        assert value.shape == (4, 1)


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
    def network_config(self):
        return NetworkConfig(
            lstm_hidden_size=64,
            lstm_num_layers=1,
            lstm_dropout=0.0,
            mlp_hidden_sizes=[32, 16],
            mlp_dropout=0.0,
        )

    @pytest.fixture
    def agent(self, model_config, network_config):
        agent = PPOAgent(model_config, network_config, device="cpu")
        agent.init_networks(obs_dim=OBS_DIM, action_dim=ACTION_DIM, lookback_window=LOOKBACK_WINDOW)
        return agent

    def test_agent_initialization(self, agent):
        """测试智能体初始化"""
        assert agent.actor is not None
        assert agent.critic is not None
        assert agent.optimizer is not None
        assert agent._lstm_shared_encoder is not None

    def test_get_action(self, agent):
        """测试获取动作"""
        obs = np.random.randn(OBS_DIM)

        action, log_prob = agent.get_action(obs, deterministic=False)

        assert action.shape == (ACTION_DIM,)
        assert np.all(action >= -1) and np.all(action <= 1)
        assert isinstance(log_prob, float)

    def test_get_deterministic_action(self, agent):
        """测试确定性动作"""
        obs = np.random.randn(OBS_DIM)

        action1, _ = agent.get_action(obs, deterministic=True)
        action2, _ = agent.get_action(obs, deterministic=True)

        np.testing.assert_array_almost_equal(action1, action2)

    def test_get_value(self, agent):
        """测试获取价值"""
        obs = np.random.randn(OBS_DIM)

        value = agent.get_value(obs)

        assert isinstance(value, float)

    def test_get_action_and_value(self, agent):
        """测试同时获取动作和价值"""
        obs = np.random.randn(OBS_DIM)

        action, log_prob, value = agent.get_action_and_value(obs, deterministic=False)

        assert action.shape == (ACTION_DIM,)
        assert isinstance(log_prob, float)
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

        obs = torch.randn(batch_size, OBS_DIM)
        actions = torch.randn(batch_size, ACTION_DIM)
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
        new_agent = PPOAgent(agent.config, agent.network_config, device="cpu")
        new_agent.init_networks(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM, lookback_window=LOOKBACK_WINDOW
        )
        new_agent.load(str(save_path))

        # 验证动作一致
        obs = np.random.randn(OBS_DIM)
        action1, _ = agent.get_action(obs, deterministic=True)
        action2, _ = new_agent.get_action(obs, deterministic=True)

        np.testing.assert_array_almost_equal(action1, action2)

    def test_shared_encoder(self, model_config, network_config):
        """测试共享 LSTM 编码器"""
        agent = PPOAgent(model_config, network_config, device="cpu")
        agent.init_networks(obs_dim=OBS_DIM, action_dim=ACTION_DIM, lookback_window=LOOKBACK_WINDOW)

        # Actor 和 Critic 应该共享同一个 LSTM 编码器
        assert agent.actor.shared_encoder is agent._lstm_shared_encoder
        assert agent.critic.shared_encoder is agent._lstm_shared_encoder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
