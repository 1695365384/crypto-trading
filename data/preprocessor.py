"""数据预处理模块"""

import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

ScalerType = Union[StandardScaler, RobustScaler]


class Preprocessor:
    """数据预处理器"""

    def __init__(self, scaler_type: str = "robust"):
        """
        初始化预处理器

        Args:
            scaler_type: 标准化类型 ('standard' 或 'robust')
        """
        self.scaler_type = scaler_type
        self.scalers: Dict[str, ScalerType] = {}
        self.feature_columns: List[str] = []

    def split(
        self,
        data: Dict[str, pd.DataFrame],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[Dict[str, pd.DataFrame], ...]:
        """
        按时间顺序分割数据

        Args:
            data: 处理后的数据字典
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例

        Returns:
            (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        train_data = {}
        val_data = {}
        test_data = {}

        for ticker, df in data.items():
            n = len(df)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))

            train_data[ticker] = df.iloc[:train_end].copy()
            val_data[ticker] = df.iloc[train_end:val_end].copy()
            test_data[ticker] = df.iloc[val_end:].copy()

        return train_data, val_data, test_data

    def fit_transform(
        self, train_data: Dict[str, pd.DataFrame], feature_columns: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        拟合并转换训练数据

        Args:
            train_data: 训练数据
            feature_columns: 要标准化的特征列

        Returns:
            标准化后的数据
        """
        if feature_columns:
            self.feature_columns = feature_columns
        else:
            # 排除时间特征和二元特征
            first_df = list(train_data.values())[0]
            self.feature_columns = [
                col
                for col in first_df.columns
                if col
                not in [
                    "hour",
                    "day_of_week",
                    "minute",
                    "is_weekend",
                    "hour_sin",
                    "hour_cos",
                    "day_sin",
                    "day_cos",
                    "minute_sin",
                    "minute_cos",
                ]
            ]

        # 合并所有交易对的训练数据用于拟合 scaler
        all_train = pd.concat([df[self.feature_columns] for df in train_data.values()])

        # 创建 scaler
        if self.scaler_type == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        scaler.fit(all_train)
        self.scalers["main"] = scaler

        # 转换数据
        return self._transform_data(train_data)

    def transform(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        使用已拟合的 scaler 转换数据

        Args:
            data: 待转换数据

        Returns:
            标准化后的数据
        """
        return self._transform_data(data)

    def _transform_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """内部转换方法"""
        result = {}

        for ticker, df in data.items():
            df = df.copy()

            # 标准化特征列
            if self.feature_columns and "main" in self.scalers:
                df[self.feature_columns] = self.scalers["main"].transform(df[self.feature_columns])

            result[ticker] = df

        return result

    def create_sequences(
        self,
        data: Dict[str, pd.DataFrame],
        lookback: int = 60,
        feature_columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列样本

        Args:
            data: 数据字典
            lookback: 回看窗口
            feature_columns: 特征列

        Returns:
            (X, prices) 特征数组和价格数组
        """
        if feature_columns is None:
            feature_columns = list(list(data.values())[0].columns)

        X_list = []
        prices_list = []

        for ticker, df in data.items():
            values = df[feature_columns].values
            prices = df["close"].values

            for i in range(lookback, len(values)):
                X_list.append(values[i - lookback : i])
                prices_list.append(prices[i])

        X = np.array(X_list)
        prices = np.array(prices_list)

        return X, prices

    def create_env_data(
        self, data: Dict[str, pd.DataFrame], lookback: int = 60
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        创建环境所需的数据格式

        Args:
            data: 数据字典
            lookback: 回看窗口

        Returns:
            (feature_data, price_data, feature_columns)
        """
        # 获取特征列
        first_df = list(data.values())[0]
        feature_columns = list(first_df.columns)

        # 提取价格和特征
        tickers = list(data.keys())
        n_timesteps = len(first_df)
        n_features = len(feature_columns)

        # 价格数组 [timesteps, assets]
        prices = np.zeros((n_timesteps, len(tickers)))
        for i, ticker in enumerate(tickers):
            prices[:, i] = data[ticker]["close"].values

        # 特征数组 [timesteps, assets, features]
        features = np.zeros((n_timesteps, len(tickers), n_features))
        for i, ticker in enumerate(tickers):
            features[:, i, :] = data[ticker][feature_columns].values

        return features, prices, feature_columns

    def save(self, path: str):
        """保存预处理器状态"""
        state = {
            "scaler_type": self.scaler_type,
            "scalers": self.scalers,
            "feature_columns": self.feature_columns,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str):
        """加载预处理器状态"""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.scaler_type = state["scaler_type"]
        self.scalers = state["scalers"]
        self.feature_columns = state["feature_columns"]
