"""交易预测器模块"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from config.config import ModelConfig


class TradingPredictor:
    """交易预测器"""

    def __init__(
        self,
        model_path: str,
        obs_dim: int,
        action_dim: int,
        model_config: Optional[ModelConfig] = None,
        device: str = "cuda:0",
    ):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
            model_config: 模型配置
            device: 设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_config = model_config or ModelConfig()

        # 初始化智能体
        self.agent = PPOAgent(self.model_config, device)
        self.agent.init_networks(obs_dim, action_dim)

        # 加载模型
        self.load(model_path)

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        预测交易动作

        Args:
            observation: 当前市场观察
            deterministic: 是否使用确定性策略

        Returns:
            (action, confidence) 动作和置信度
        """
        with torch.no_grad():
            action, log_prob = self.agent.get_action(observation, deterministic)

            # 计算置信度 (基于策略熵)
            mean, std = self.agent.actor(
                torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            )
            entropy = -torch.log(std).mean().item()
            confidence = 1 / (1 + entropy)

        return action, confidence

    def predict_batch(
        self, observations: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量预测

        Args:
            observations: 观察数组 [batch, obs_dim]
            deterministic: 是否使用确定性策略

        Returns:
            (actions, confidences)
        """
        actions = []
        confidences = []

        for obs in observations:
            action, conf = self.predict(obs, deterministic)
            actions.append(action)
            confidences.append(conf)

        return np.array(actions), np.array(confidences)

    def should_trade(
        self,
        action: np.ndarray,
        confidence: float,
        action_threshold: float = 0.1,
        confidence_threshold: float = 0.5,
    ) -> List[bool]:
        """
        判断是否执行交易

        Args:
            action: 交易动作
            confidence: 置信度
            action_threshold: 动作阈值
            confidence_threshold: 置信度阈值

        Returns:
            是否执行各币种的交易
        """
        return (np.abs(action) > action_threshold) & (confidence > confidence_threshold)

    def get_position_adjustment(
        self, current_positions: np.ndarray, action: np.ndarray, max_position: float = 0.5
    ) -> np.ndarray:
        """
        获取仓位调整

        Args:
            current_positions: 当前仓位
            action: 交易动作
            max_position: 最大仓位

        Returns:
            调整后的目标仓位
        """
        # 将动作映射到目标仓位
        target_positions = current_positions + action * max_position

        # 限制仓位范围
        target_positions = np.clip(target_positions, -max_position, max_position)

        return target_positions

    def load(self, path: str):
        """加载模型"""
        if os.path.exists(path):
            self.agent.load(path)
            print(f"Model loaded from: {path}")
        else:
            raise FileNotFoundError(f"Model file not found: {path}")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "device": str(self.device),
            "actor_params": sum(p.numel() for p in self.agent.actor.parameters()),
            "critic_params": sum(p.numel() for p in self.agent.critic.parameters()),
            "config": self.model_config.__dict__,
        }


class EnsemblePredictor:
    """集成预测器 (多模型投票)"""

    def __init__(
        self,
        model_paths: List[str],
        obs_dim: int,
        action_dim: int,
        model_config: Optional[ModelConfig] = None,
        device: str = "cuda:0",
    ):
        """
        初始化集成预测器

        Args:
            model_paths: 模型路径列表
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
            model_config: 模型配置
            device: 设备
        """
        self.predictors = []

        for path in model_paths:
            predictor = TradingPredictor(path, obs_dim, action_dim, model_config, device)
            self.predictors.append(predictor)

    def predict(self, observation: np.ndarray, method: str = "mean") -> Tuple[np.ndarray, float]:
        """
        集成预测

        Args:
            observation: 观察
            method: 集成方法 ('mean', 'median', 'vote')

        Returns:
            (action, confidence)
        """
        actions = []
        confidences = []

        for predictor in self.predictors:
            action, conf = predictor.predict(observation)
            actions.append(action)
            confidences.append(conf)

        actions = np.array(actions)
        confidences = np.array(confidences)

        if method == "mean":
            final_action = np.mean(actions, axis=0)
            final_conf = np.mean(confidences)
        elif method == "median":
            final_action = np.median(actions, axis=0)
            final_conf = np.median(confidences)
        elif method == "vote":
            # 基于置信度的加权投票
            weights = confidences / np.sum(confidences)
            final_action = np.average(actions, axis=0, weights=weights)
            final_conf = np.mean(confidences)
        else:
            raise ValueError(f"Unknown method: {method}")

        return final_action, final_conf

    def get_disagreement(self, observation: np.ndarray) -> float:
        """
        计算模型间的分歧度

        Args:
            observation: 观察

        Returns:
            分歧度 (动作标准差)
        """
        actions = []
        for predictor in self.predictors:
            action, _ = predictor.predict(observation)
            actions.append(action)

        return np.std(actions)
