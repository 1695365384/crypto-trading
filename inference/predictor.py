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
        lookback_window: int = 30,
        model_config: Optional[ModelConfig] = None,
        device: str = "cuda:0",
    ):
        """
        初始化预测器

        Args:
            model_path: 模型文件路径
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
            lookback_window: 回看窗口大小 (LSTM 模式需要)
            model_config: 模型配置
            device: 设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_config = model_config or ModelConfig()

        # 初始化智能体
        self.agent = PPOAgent(self.model_config, self.model_config.network, device)
        self.agent.init_networks(
            obs_dim=obs_dim,
            action_dim=action_dim,
            lookback_window=lookback_window,
        )

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
            mean, std, _ = self.agent.actor(
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
    """多模型集成预测器

    支持多种集成方法:
    - mean: 均值集成
    - median: 中位数集成
    - vote: 加权投票
    """

    def __init__(
        self,
        model_paths: List[str],
        obs_dim: int,
        action_dim: int,
        lookback_window: int = 30,
        model_config: Optional[ModelConfig] = None,
        method: str = "mean",
        weights: Optional[List[float]] = None,
        device: str = "cuda:0",
    ):
        """
        初始化集成预测器

        Args:
            model_paths: 模型路径列表
            obs_dim: 观察空间维度
            action_dim: 动作空间维度
            lookback_window: 回看窗口大小
            model_config: 模型配置
            method: 集成方法 ('mean', 'median', 'vote')
            weights: 模型权重 (用于 vote 方法)
            device: 设备
        """
        # 验证集成方法
        valid_methods = ["mean", "median", "vote"]
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Must be one of {valid_methods}")

        self.method = method
        self.model_paths = model_paths
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # 加载所有模型
        self.predictors: List[TradingPredictor] = []
        for path in model_paths:
            predictor = TradingPredictor(
                path,
                obs_dim,
                action_dim,
                lookback_window=lookback_window,
                model_config=model_config,
                device=device,
            )
            self.predictors.append(predictor)

        # 设置权重
        if weights is None:
            # 均匀权重
            self.weights = [1.0 / len(model_paths)] * len(model_paths)
        else:
            if len(weights) != len(model_paths):
                raise ValueError(
                    f"Weights length ({len(weights)}) must match model count ({len(model_paths)})"
                )
            # 归一化权重
            total = sum(weights)
            self.weights = [w / total for w in weights]

        print(f"EnsemblePredictor initialized with {len(model_paths)} models, method={method}")

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        集成预测

        Args:
            observation: 观察
            deterministic: 是否使用确定性策略

        Returns:
            (action, confidence)
        """
        # 收集所有模型的预测
        actions = []
        confidences = []

        for predictor in self.predictors:
            action, conf = predictor.predict(observation, deterministic)
            actions.append(action)
            confidences.append(conf)

        actions = np.array(actions)  # [n_models, action_dim]
        confidences = np.array(confidences)  # [n_models]

        # 根据方法聚合
        if self.method == "mean":
            final_action, final_confidence = self._aggregate_mean(actions, confidences)
        elif self.method == "median":
            final_action, final_confidence = self._aggregate_median(actions, confidences)
        elif self.method == "vote":
            final_action, final_confidence = self._aggregate_vote(actions, confidences)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return final_action, final_confidence

    def _aggregate_mean(
        self, actions: np.ndarray, confidences: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """均值集成"""
        # 加权平均动作
        weights = np.array(self.weights)
        final_action = np.average(actions, axis=0, weights=weights)

        # 加权平均置信度
        final_confidence = float(np.average(confidences, weights=weights))

        return final_action, final_confidence

    def _aggregate_median(
        self, actions: np.ndarray, confidences: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """中位数集成"""
        # 动作取中位数
        final_action = np.median(actions, axis=0)

        # 置信度取中位数
        final_confidence = float(np.median(confidences))

        return final_action, final_confidence

    def _aggregate_vote(
        self, actions: np.ndarray, confidences: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """加权投票集成

        根据动作方向和置信度进行加权投票
        """
        n_models, action_dim = actions.shape
        weights = np.array(self.weights)

        final_action = np.zeros(action_dim)

        for i in range(action_dim):
            # 获取该维度的所有动作
            dim_actions = actions[:, i]

            # 分别计算买入和卖出的加权投票
            buy_mask = dim_actions > 0
            sell_mask = dim_actions < 0

            buy_vote = np.sum(weights[buy_mask] * dim_actions[buy_mask]) if np.any(buy_mask) else 0
            sell_vote = (
                np.sum(weights[sell_mask] * dim_actions[sell_mask]) if np.any(sell_mask) else 0
            )

            # 最终动作
            final_action[i] = buy_vote + sell_vote

        # 加权平均置信度
        final_confidence = float(np.average(confidences, weights=weights))

        return final_action, final_confidence

    def predict_batch(
        self, observations: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量集成预测

        Args:
            observations: 观察数组 [batch, obs_dim]
            deterministic: 是否使用确定性策略

        Returns:
            (actions, confidences) - 动作和置信度数组
        """
        batch_size = observations.shape[0]

        # 收集所有模型的预测
        all_actions_list: List[np.ndarray] = []
        all_confidences_list: List[np.ndarray] = []

        for predictor in self.predictors:
            actions, confidences = predictor.predict_batch(observations, deterministic)
            all_actions_list.append(actions)
            all_confidences_list.append(confidences)

        all_actions: np.ndarray = np.array(all_actions_list)  # [n_models, batch, action_dim]
        all_confidences: np.ndarray = np.array(all_confidences_list)  # [n_models, batch]

        # 聚合
        if self.method == "mean":
            weights = np.array(self.weights)
            final_actions = np.average(all_actions, axis=0, weights=weights)
            final_confidences = np.average(all_confidences, axis=0, weights=weights)
        elif self.method == "median":
            final_actions = np.median(all_actions, axis=0)
            final_confidences = np.median(all_confidences, axis=0)
        else:  # vote
            final_actions = np.zeros((batch_size, self.action_dim))
            final_confidences = np.zeros(batch_size)

            for b in range(batch_size):
                action, conf = self._aggregate_vote(all_actions[:, b, :], all_confidences[:, b])
                final_actions[b] = action
                final_confidences[b] = conf

        return final_actions, final_confidences

    def should_trade(
        self,
        action: np.ndarray,
        confidence: float,
        action_threshold: float = 0.1,
        confidence_threshold: float = 0.3,
    ) -> bool:
        """判断是否应该交易"""
        max_action = np.max(np.abs(action))
        return max_action > action_threshold and confidence > confidence_threshold

    def get_position_adjustment(
        self, current_positions: np.ndarray, action: np.ndarray, max_position: float = 0.5
    ) -> np.ndarray:
        """获取仓位调整建议"""
        target_positions = current_positions + action * max_position
        return np.clip(target_positions, -max_position, max_position)

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

        return float(np.std(actions))

    def get_ensemble_info(self) -> Dict:
        """
        获取集成模型信息

        Returns:
            集成模型信息字典
        """
        return {
            "n_models": len(self.predictors),
            "method": self.method,
            "weights": self.weights,
            "model_paths": self.model_paths,
            "predictors_info": [p.get_model_info() for p in self.predictors],
        }
