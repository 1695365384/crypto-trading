"""
评估脚本
用法: python scripts/evaluate.py --model models/best_model.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOAgent
from config.config import Config
from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer
from data.preprocessor import Preprocessor
from envs.crypto_env import CryptoTradingEnv
from evaluation.backtest import Backtester


def main():
    parser = argparse.ArgumentParser(description="Evaluate trading agent")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--config", type=str, default=None, help="Config path")
    parser.add_argument("--data", type=str, default=None, help="Data path")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    config.device = args.device

    print("=" * 60)
    print("Evaluating Crypto Trading Agent")
    print("=" * 60)

    # 加载数据
    data_loader = DataLoader(config.data)
    raw_data = data_loader.load(args.data or config.data_dir)

    # 特征工程
    engineer = FeatureEngineer(config.data.indicators)
    featured_data = engineer.process(raw_data)

    # 预处理
    preprocessor = Preprocessor()
    train_data, val_data, test_data = preprocessor.split(featured_data)

    # 在各数据集上评估
    datasets = {"train": train_data, "val": val_data, "test": test_data}

    # 加载模型
    agent = PPOAgent(config.model, config.model.network, config.device)

    results = {}
    backtester = Backtester(config)

    for name, data in datasets.items():
        print(f"\nEvaluating on {name} data...")

        # 标准化
        if name == "train":
            data = preprocessor.fit_transform(data)
        else:
            data = preprocessor.transform(data)

        # 创建环境
        features, prices, feature_cols = preprocessor.create_env_data(data)
        env = CryptoTradingEnv(config.env, prices, features, feature_cols)

        # 初始化网络
        if not hasattr(agent, "actor") or agent.actor is None:
            agent.init_networks(
                obs_dim=env.observation_space.shape[0],
                action_dim=env.action_space.shape[0],
                lookback_window=config.env.lookback_window,
            )
            agent.load(args.model)

        # 回测
        result = backtester.run(agent, env, deterministic=True)
        results[name] = result

        if args.verbose:
            print(result.summary())

    # 汇总结果
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'Dataset':<10} {'Return':<12} {'Sharpe':<10} {'MaxDD':<10} {'WinRate':<10}")
    print("-" * 60)
    for name, result in results.items():
        print(
            f"{name:<10} {result.total_return:>10.2%} {result.sharpe_ratio:>10.2f} {result.max_drawdown:>10.2%} {result.win_rate:>10.2%}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
