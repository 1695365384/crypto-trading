"""
回测脚本
用法: python scripts/backtest.py --model models/best_model.pt --data data/test.csv
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
from evaluation.visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(description="Backtest trading agent")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--data", type=str, default=None, help="Data path")
    parser.add_argument("--config", type=str, default=None, help="Config path")
    parser.add_argument("--output", type=str, default="./backtest_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    config.device = args.device

    print("=" * 60)
    print("Backtesting Crypto Trading Agent")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # 加载数据
    print("\nLoading data...")
    data_loader = DataLoader(config.data)
    if args.data:
        raw_data = data_loader.load(args.data)
    else:
        raw_data = data_loader.load(config.data_dir)

    # 特征工程
    print("Engineering features...")
    engineer = FeatureEngineer(config.data.indicators)
    featured_data = engineer.process(raw_data)

    # 预处理
    print("Preprocessing...")
    preprocessor = Preprocessor()
    featured_data = preprocessor.fit_transform(featured_data)

    # 创建环境
    features, prices, feature_cols = preprocessor.create_env_data(featured_data)
    env = CryptoTradingEnv(config.env, prices, features, feature_cols)

    # 加载模型
    print(f"\nLoading model from {args.model}...")
    agent = PPOAgent(config.model, config.device)
    agent.init_networks(env.observation_space.shape[0], env.action_space.shape[0])
    agent.load(args.model)

    # 运行回测
    print("\nRunning backtest...")
    backtester = Backtester(config)
    results = backtester.run(agent, env, deterministic=True)

    # 打印结果
    print(results.summary())

    # 生成可视化
    print("\nGenerating visualizations...")
    visualizer = Visualizer()
    os.makedirs(args.output, exist_ok=True)
    visualizer.generate_report(results, args.output, "backtest")

    # 保存结果
    results.save(os.path.join(args.output, "backtest_results.json"))

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
