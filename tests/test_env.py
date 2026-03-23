"""环境测试"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import EnvConfig
from envs.crypto_env import CryptoTradingEnv


class TestCryptoTradingEnv:

    @pytest.fixture
    def env_config(self):
        return EnvConfig(
            initial_amount=10000,
            max_position_pct=0.5,
            transaction_cost_pct=0.001,
            lookback_window=60,
            num_assets=2,
        )

    @pytest.fixture
    def sample_data(self):
        # 创建样本数据
        n_steps = 1000
        n_assets = 2
        n_features = 10

        price_data = np.random.uniform(90, 110, (n_steps, n_assets))
        feature_data = np.random.randn(n_steps, n_assets, n_features)

        return price_data, feature_data

    def test_env_initialization(self, env_config, sample_data):
        """测试环境初始化"""
        price_data, feature_data = sample_data
        env = CryptoTradingEnv(env_config, price_data, feature_data)

        assert env.n_assets == 2
        assert env.n_steps == 1000
        assert env.portfolio_value == env_config.initial_amount

    def test_env_reset(self, env_config, sample_data):
        """测试环境重置"""
        price_data, feature_data = sample_data
        env = CryptoTradingEnv(env_config, price_data, feature_data)

        obs, info = env.reset()

        assert obs.shape == env.observation_space.shape
        assert env.portfolio_value == env_config.initial_amount
        assert env.current_step == env_config.lookback_window

    def test_env_step(self, env_config, sample_data):
        """测试环境步骤"""
        price_data, feature_data = sample_data
        env = CryptoTradingEnv(env_config, price_data, feature_data)

        obs, _ = env.reset()
        action = np.array([0.5, -0.3])

        next_obs, reward, terminated, truncated, info = env.step(action)

        assert next_obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(info, dict)

    def test_env_episode(self, env_config, sample_data):
        """测试完整回合"""
        price_data, feature_data = sample_data
        env = CryptoTradingEnv(env_config, price_data, feature_data)

        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            action = np.random.uniform(-1, 1, env.n_assets)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            if steps > len(price_data):
                break

        assert steps > 0
        assert "portfolio_value" in info

    def test_buy_action(self, env_config, sample_data):
        """测试买入动作"""
        price_data, feature_data = sample_data
        env = CryptoTradingEnv(env_config, price_data, feature_data)

        obs, _ = env.reset()
        initial_balance = env.balance

        # 买入
        action = np.array([1.0, 0.0])
        obs, _, _, _, info = env.step(action)

        assert env.balance < initial_balance
        assert env.positions[0] > 0

    def test_sell_action(self, env_config, sample_data):
        """测试卖出动作"""
        price_data, feature_data = sample_data
        env = CryptoTradingEnv(env_config, price_data, feature_data)

        obs, _ = env.reset()

        # 先买入
        action = np.array([1.0, 0.0])
        env.step(action)
        initial_position = env.positions[0].copy()

        # 再卖出
        action = np.array([-1.0, 0.0])
        env.step(action)

        assert env.positions[0] < initial_position

    def test_portfolio_stats(self, env_config, sample_data):
        """测试组合统计"""
        price_data, feature_data = sample_data
        env = CryptoTradingEnv(env_config, price_data, feature_data)

        obs, _ = env.reset()
        done = False

        while not done:
            action = np.random.uniform(-1, 1, env.n_assets)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        stats = env.get_portfolio_stats()

        assert "total_return" in stats
        assert "sharpe_ratio" in stats
        assert "max_drawdown" in stats

    def test_observation_space(self, env_config, sample_data):
        """测试观察空间"""
        price_data, feature_data = sample_data
        env = CryptoTradingEnv(env_config, price_data, feature_data)

        assert env.observation_space.shape[0] > 0
        assert env.action_space.shape[0] == env.n_assets
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
