"""网络结构模块

LSTM + MLP 混合架构:
- LSTM 层: 处理时序输入，输出 128 维
- MLP 头: 128 -> 256 -> 128 -> 64 -> 输出
- Dropout 正则化
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from config.config import NetworkConfig


class LSTMSharedEncoder(nn.Module):
    """共享 LSTM 编码器 - Actor 和 Critic 共用"""

    def __init__(self, feature_dim: int, config: NetworkConfig):
        super().__init__()
        self.config = config

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
            batch_first=True,
        )

        # LSTM 输出 dropout
        self.output_dropout = nn.Dropout(config.mlp_dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播

        Args:
            x: 输入张量 [batch, seq_len, feature_dim]
            hidden_state: 可选的 LSTM 隐藏状态 (h, c)

        Returns:
            (output, new_hidden_state)
            output: [batch, lstm_hidden_size]
        """
        lstm_out, new_hidden = self.lstm(x, hidden_state)

        # 取最后一个时间步的输出
        output = lstm_out[:, -1, :]

        # 应用 dropout
        output = self.output_dropout(output)

        return output, new_hidden


class ActorNetwork(nn.Module):
    """LSTM + MLP 策略网络"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        lookback_window: int,
        feature_dim: int,
        config: NetworkConfig,
        shared_encoder: Optional[LSTMSharedEncoder] = None,
    ):
        super().__init__()
        self.config = config
        self.lookback_window = lookback_window
        self.feature_dim = feature_dim
        self.obs_dim = obs_dim

        # 共享 LSTM 编码器
        if shared_encoder is not None:
            self.shared_encoder = shared_encoder
        else:
            self.shared_encoder = LSTMSharedEncoder(feature_dim, config)

        # MLP 头
        layers = []
        prev_size = config.lstm_hidden_size
        for size in config.mlp_hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, size),
                    nn.LayerNorm(size),
                    nn.ReLU(),
                    nn.Dropout(config.mlp_dropout),
                ]
            )
            prev_size = size

        self.mlp = nn.Sequential(*layers)

        # 输出头
        self.mean_head = nn.Linear(prev_size, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        # 输出层使用更小的初始化
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播

        Args:
            obs: 观察张量
                - 如果是 flatten 的: [batch, obs_dim]
                - 如果是序列: [batch, seq_len, feature_dim]
            hidden_state: LSTM 隐藏状态

        Returns:
            (mean, std, new_hidden_state)
        """
        # 处理输入形状
        if obs.dim() == 2:
            # flatten 输入，提取窗口特征部分 (排除 account_state)
            batch_size = obs.size(0)
            window_features = obs[:, : self.lookback_window * self.feature_dim]
            obs = window_features.view(batch_size, self.lookback_window, self.feature_dim)

        # LSTM 编码
        lstm_out, new_hidden = self.shared_encoder(obs, hidden_state)

        # MLP 处理
        x = self.mlp(lstm_out)

        # 输出
        mean = torch.tanh(self.mean_head(x))
        std = torch.exp(self.log_std.clamp(-20, 2))

        return mean, std, new_hidden

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        获取动作

        Returns:
            (action, log_prob, new_hidden_state)
        """
        mean, std, new_hidden = self.forward(obs, hidden_state)

        if deterministic:
            return mean, None, new_hidden

        from torch.distributions import Normal

        dist = Normal(mean, std)
        action = dist.rsample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, new_hidden

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        评估动作

        Returns:
            (log_prob, entropy, new_hidden_state)
        """
        mean, std, new_hidden = self.forward(obs, hidden_state)

        from torch.distributions import Normal

        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy, new_hidden


class CriticNetwork(nn.Module):
    """LSTM + MLP 价值网络"""

    def __init__(
        self,
        obs_dim: int,
        lookback_window: int,
        feature_dim: int,
        config: NetworkConfig,
        shared_encoder: Optional[LSTMSharedEncoder] = None,
    ):
        super().__init__()
        self.config = config
        self.lookback_window = lookback_window
        self.feature_dim = feature_dim
        self.obs_dim = obs_dim

        # 共享 LSTM 编码器
        if shared_encoder is not None:
            self.shared_encoder = shared_encoder
        else:
            self.shared_encoder = LSTMSharedEncoder(feature_dim, config)

        # MLP 头
        layers = []
        prev_size = config.lstm_hidden_size
        for size in config.mlp_hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, size),
                    nn.LayerNorm(size),
                    nn.ReLU(),
                    nn.Dropout(config.mlp_dropout),
                ]
            )
            prev_size = size

        self.mlp = nn.Sequential(*layers)

        # 输出头
        self.value_head = nn.Linear(prev_size, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播

        Returns:
            (value, new_hidden_state)
        """
        # 处理输入形状
        if obs.dim() == 2:
            batch_size = obs.size(0)
            # 提取窗口特征部分 (排除 account_state)
            window_features = obs[:, : self.lookback_window * self.feature_dim]
            obs = window_features.view(batch_size, self.lookback_window, self.feature_dim)

        # LSTM 编码
        lstm_out, new_hidden = self.shared_encoder(obs, hidden_state)

        # MLP 处理
        x = self.mlp(lstm_out)

        # 输出
        value = self.value_head(x)

        return value, new_hidden
