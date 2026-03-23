"""
训练脚本
用法: python scripts/train.py --config config/prod.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOAgent
from config.config import Config
from data.data_loader import DataLoader
from data.feature_engineer import FeatureEngineer
from data.preprocessor import Preprocessor
from envs.crypto_env import CryptoTradingEnv
from evaluation.backtest import Backtester
from evaluation.visualizer import Visualizer
from training.callbacks import CheckpointCallback, EarlyStoppingCallback, TensorBoardCallback
from training.trainer import Trainer


def find_latest_model(model_dir: str) -> str | None:
    """查找最新的模型文件"""
    import glob

    patterns = [
        os.path.join(model_dir, "model_*.pt"),
        os.path.join(model_dir, "checkpoint_*.pt"),
        os.path.join(model_dir, "best_model_*.pt"),
    ]
    all_models = []
    for pattern in patterns:
        all_models.extend(glob.glob(pattern))
    if not all_models:
        return None
    # 按修改时间排序，返回最新的
    return max(all_models, key=os.path.getmtime)


def main():
    # 1. 解析参数（配置文件里控制所有训练参数）
    parser = argparse.ArgumentParser(description="Train crypto trading agent")
    parser.add_argument("--config", type=str, default="config/okx.yaml", help="Config file path")
    args = parser.parse_args()

    # 2. 加载配置
    import torch

    if os.path.exists(args.config):
        config = Config.from_yaml(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default config...")
        config = Config()

    # 自动选择设备：配置文件指定 auto 时自动检测
    if config.device == "auto":
        if torch.cuda.is_available():
            config.device = "cuda:0"
            print(f"CUDA available, using GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            config.device = "mps"  # Apple Silicon GPU
            print("MPS available, using Apple Silicon GPU")
        else:
            config.device = "cpu"
            print("No GPU available, using CPU (training will be slower)")

    # 设置随机种子
    import numpy as np

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    print("=" * 60)
    print("Crypto Trading Agent Training")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Total Timesteps: {config.model.total_timesteps}")
    print(f"Tickers: {config.data.tickers}")
    print("=" * 60)

    # 3. 数据加载
    print("\n[1/6] Loading data...")
    data_loader = DataLoader(config.data)
    raw_data = data_loader.load(config.data_dir)

    for ticker, df in raw_data.items():
        print(f"  {ticker}: {len(df)} rows")

    # 4. 特征工程
    print("\n[2/6] Engineering features...")
    engineer = FeatureEngineer(config.data.indicators)
    featured_data = engineer.process(raw_data)

    for ticker, df in featured_data.items():
        print(f"  {ticker}: {df.shape[1]} features")

    # 5. 数据预处理
    print("\n[3/6] Preprocessing data...")
    preprocessor = Preprocessor()

    # 分割数据
    train_data, val_data, test_data = preprocessor.split(
        featured_data, config.data.train_ratio, config.data.val_ratio, config.data.test_ratio
    )

    for ticker in train_data:
        print(
            f"  {ticker} - Train: {len(train_data[ticker])}, Val: {len(val_data[ticker])}, Test: {len(test_data[ticker])}"
        )

    # 标准化
    train_data = preprocessor.fit_transform(train_data)
    val_data = preprocessor.transform(val_data)
    test_data = preprocessor.transform(test_data)

    # 保存预处理器
    preprocessor.save(os.path.join(config.model_dir, "preprocessor.pkl"))

    # 创建环境数据
    train_features, train_prices, feature_cols = preprocessor.create_env_data(train_data)
    val_features, val_prices, _ = preprocessor.create_env_data(val_data)
    test_features, test_prices, _ = preprocessor.create_env_data(test_data)

    # 6. 创建环境
    print("\n[4/6] Creating environments...")
    train_env = CryptoTradingEnv(config.env, train_prices, train_features, feature_cols)
    val_env = CryptoTradingEnv(config.env, val_prices, val_features, feature_cols)
    test_env = CryptoTradingEnv(config.env, test_prices, test_features, feature_cols)

    print(f"  Observation space: {train_env.observation_space.shape}")
    print(f"  Action space: {train_env.action_space.shape}")

    # 7. 创建智能体
    print("\n[5/6] Initializing agent...")
    agent = PPOAgent(config.model, config.model.network, config.device)
    agent.init_networks(
        obs_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.shape[0],
        lookback_window=config.env.lookback_window,
    )

    # 自动加载最新模型继续训练
    resume_path = find_latest_model(config.model_dir)
    if resume_path:
        print(f"\n  >>> Resuming from: {resume_path}")
        agent.load(resume_path)
        print("  >>> Model loaded! Continuing training...")
    else:
        print("\n  >>> No existing model found, starting from scratch...")

    print(f"  Actor params: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"  Critic params: {sum(p.numel() for p in agent.critic.parameters()):,}")

    # 8. 设置回调
    callbacks = [
        CheckpointCallback(save_freq=config.model.save_freq, save_path=config.model_dir),
        EarlyStoppingCallback(patience=50, min_delta=0.003, metric="total_return"),
        TensorBoardCallback(log_dir=os.path.join(config.log_dir, "tensorboard")),
    ]

    # 9. 训练
    print("\n[6/6] Training...")
    trainer = Trainer(config, agent, train_env, val_env, callbacks)
    trainer.train()

    # 保存训练日志
    trainer.save_training_logs(os.path.join(config.log_dir, "training_logs.json"))

    # 10. 测试评估
    print("\n" + "=" * 60)
    print("Testing on held-out data...")
    print("=" * 60)

    backtester = Backtester(config)
    test_results = backtester.run(agent, test_env)

    print(test_results.summary())

    # 11. 可视化
    print("\nGenerating visualizations...")
    visualizer = Visualizer()
    visualizer.generate_report(test_results, os.path.join(config.log_dir, "backtest"), "final")

    # 12. 保存最终模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(config.model_dir, f"model_{timestamp}.pt")
    agent.save(model_path)

    # 保存配置
    config.to_yaml(os.path.join(config.model_dir, f"config_{timestamp}.yaml"))

    print("\nTraining completed!")
    print(f"Model saved to: {model_path}")
    print(f"Logs saved to: {config.log_dir}")


if __name__ == "__main__":
    main()
