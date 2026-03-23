"""回测测试"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOAgent
from config.config import Config
from envs.crypto_env import CryptoTradingEnv
from evaluation.backtest import Backtester, BacktestResult
from evaluation.metrics import calculate_drawdown, calculate_metrics


class TestBacktest:

    @pytest.fixture
    def config(self):
        config = Config()
        config.device = "cpu"
        return config

    @pytest.fixture
    def env_and_agent(self, config):
        # 创建样本数据
        n_steps = 1000
        n_assets = 2
        n_features = 20

        price_data = np.random.uniform(90, 110, (n_steps, n_assets))
        feature_data = np.random.randn(n_steps, n_assets, n_features)

        # 创建环境
        env = CryptoTradingEnv(config.env, price_data, feature_data)

        # 创建智能体
        agent = PPOAgent(config.model, device="cpu")
        agent.init_networks(env.observation_space.shape[0], env.action_space.shape[0])

        return env, agent

    def test_backtester_initialization(self, config):
        """测试回测器初始化"""
        backtester = Backtester(config)
        assert backtester.config is not None

    def test_backtest_run(self, config, env_and_agent):
        """测试回测运行"""
        env, agent = env_and_agent
        backtester = Backtester(config)

        result = backtester.run(agent, env, deterministic=True)

        assert isinstance(result, BacktestResult)
        assert len(result.portfolio_values) > 0
        assert result.total_return != 0

    def test_backtest_result_properties(self, config, env_and_agent):
        """测试回测结果属性"""
        env, agent = env_and_agent
        backtester = Backtester(config)

        result = backtester.run(agent, env, deterministic=True)

        # 检查所有属性都存在
        assert hasattr(result, "total_return")
        assert hasattr(result, "annual_return")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown")
        assert hasattr(result, "volatility")
        assert hasattr(result, "total_trades")
        assert hasattr(result, "win_rate")

    def test_backtest_result_to_dict(self, config, env_and_agent):
        """测试结果转字典"""
        env, agent = env_and_agent
        backtester = Backtester(config)

        result = backtester.run(agent, env, deterministic=True)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "total_return" in result_dict

    def test_backtest_result_summary(self, config, env_and_agent):
        """测试结果摘要"""
        env, agent = env_and_agent
        backtester = Backtester(config)

        result = backtester.run(agent, env, deterministic=True)
        summary = result.summary()

        assert isinstance(summary, str)
        assert "Total Return" in summary

    def test_calculate_metrics(self):
        """测试指标计算"""
        returns = np.random.randn(100) * 0.01

        metrics = calculate_metrics(returns)

        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "var_95" in metrics

    def test_calculate_drawdown(self):
        """测试回撤计算"""
        values = np.array([100, 105, 110, 105, 100, 95, 100, 110])

        dd_metrics = calculate_drawdown(values)

        assert "max_drawdown" in dd_metrics
        assert dd_metrics["max_drawdown"] >= 0

    def test_deterministic_vs_stochastic(self, config, env_and_agent):
        """测试确定性 vs 随机策略"""
        env, agent = env_and_agent
        backtester = Backtester(config)

        # 运行两次确定性
        result1 = backtester.run(agent, env, deterministic=True)
        result2 = backtester.run(agent, env, deterministic=True)

        # 确定性结果应该完全一致
        np.testing.assert_array_almost_equal(result1.portfolio_values, result2.portfolio_values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
