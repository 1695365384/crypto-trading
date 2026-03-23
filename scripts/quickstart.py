"""
快速启动脚本 - 用于验证系统安装
用法: python scripts/quickstart.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_dependencies():
    """检查依赖"""
    print("Checking dependencies...")

    required = ["torch", "numpy", "pandas", "gymnasium", "sklearn"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg}")
            missing.append(pkg)

    if missing:
        print(f"\nMissing packages: {missing}")
        print("Run: pip install -r requirements.txt")
        return False

    return True


def test_environment():
    """测试环境"""
    print("\nTesting environment...")

    import numpy as np

    from config.config import EnvConfig
    from envs.crypto_env import CryptoTradingEnv

    # 创建测试数据
    n_steps = 100
    n_assets = 2
    n_features = 10

    price_data = np.random.uniform(90, 110, (n_steps, n_assets))
    feature_data = np.random.randn(n_steps, n_assets, n_features)

    # 创建环境
    config = EnvConfig(lookback_window=10)
    env = CryptoTradingEnv(config, price_data, feature_data)

    # 测试重置和步骤
    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space.shape}")

    action = np.random.uniform(-1, 1, n_assets)
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Reward: {reward:.6f}")

    print("  ✓ Environment test passed")
    return True


def test_agent():
    """测试智能体"""
    print("\nTesting agent...")

    import numpy as np

    from agents.ppo_agent import PPOAgent
    from config.config import ModelConfig

    config = ModelConfig()
    agent = PPOAgent(config, device="cpu")
    agent.init_networks(obs_dim=100, action_dim=2)

    # 测试动作生成
    obs = np.random.randn(100)
    action, log_prob = agent.get_action(obs)
    print(f"  Action: {action}")
    print(f"  Log prob: {log_prob:.4f}")

    # 测试价值估计
    value = agent.get_value(obs)
    print(f"  Value: {value:.4f}")

    print("  ✓ Agent test passed")
    return True


def test_backtest():
    """测试回测"""
    print("\nTesting backtest...")

    import numpy as np

    from agents.ppo_agent import PPOAgent
    from config.config import Config
    from envs.crypto_env import CryptoTradingEnv
    from evaluation.backtest import Backtester

    config = Config()
    config.device = "cpu"

    # 创建测试数据
    n_steps = 200
    n_assets = 2
    n_features = 20

    price_data = np.random.uniform(90, 110, (n_steps, n_assets))
    feature_data = np.random.randn(n_steps, n_assets, n_features)

    # 创建环境和智能体
    env = CryptoTradingEnv(config.env, price_data, feature_data)
    agent = PPOAgent(config.model, device="cpu")
    agent.init_networks(env.observation_space.shape[0], env.action_space.shape[0])

    # 运行回测
    backtester = Backtester(config)
    results = backtester.run(agent, env, deterministic=True)

    print(f"  Total return: {results.total_return:.2%}")
    print(f"  Max drawdown: {results.max_drawdown:.2%}")

    print("  ✓ Backtest test passed")
    return True


def main():
    print("=" * 60)
    print("Crypto Trading Agent - Quick Start")
    print("=" * 60)

    all_passed = True

    # 检查依赖
    if not check_dependencies():
        all_passed = False
        return

    # 测试环境
    try:
        test_environment()
    except Exception as e:
        print(f"  ✗ Environment test failed: {e}")
        all_passed = False

    # 测试智能体
    try:
        test_agent()
    except Exception as e:
        print(f"  ✗ Agent test failed: {e}")
        all_passed = False

    # 测试回测
    try:
        test_backtest()
    except Exception as e:
        print(f"  ✗ Backtest test failed: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! System is ready.")
        print("\nNext steps:")
        print("  1. Prepare your data in ./data/")
        print("  2. Run: python scripts/train.py --config config/dev.yaml")
    else:
        print("Some tests failed. Please check the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
