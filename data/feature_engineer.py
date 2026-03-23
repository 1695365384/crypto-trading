"""特征工程模块"""

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class FeatureEngineer:
    """技术指标特征工程"""

    def __init__(self, indicators: List[str]):
        """
        初始化特征工程器

        Args:
            indicators: 需要计算的技术指标列表
        """
        self.indicators = indicators
        self.feature_stats: Dict[str, Tuple[float, float]] = {}

    def process(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        处理所有交易对的数据

        Args:
            data: 原始价格数据字典

        Returns:
            添加特征后的数据字典
        """
        processed = {}
        for ticker, df in data.items():
            processed[ticker] = self.add_features(df.copy())
        return processed

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加所有特征

        Args:
            df: OHLCV 数据框

        Returns:
            添加特征后的数据框
        """
        # 添加技术指标
        df = self.add_technical_indicators(df)

        # 添加时间特征
        df = self.add_time_features(df)

        # 添加价格变化率
        df = self.add_returns(df)

        # 清理 NaN
        df = df.dropna()

        return df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        for indicator in self.indicators:
            if indicator == "macd":
                df = self._add_macd(df)
            elif indicator.startswith("rsi"):
                period = int(indicator.split("_")[1])
                df = self._add_rsi(df, period)
            elif indicator.startswith("ema"):
                period = int(indicator.split("_")[1])
                df = self._add_ema(df, period)
            elif indicator == "boll_ub":
                df = self._add_bollinger(df)
            elif indicator == "boll_lb":
                pass  # 已在 boll_ub 中添加
            elif indicator == "atr":
                df = self._add_atr(df)
            elif indicator == "obv":
                df = self._add_obv(df)
            elif indicator == "cci":
                df = self._add_cci(df)
            elif indicator == "adx":
                df = self._add_adx(df)

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        df = df.copy()

        # 基本时间特征
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["minute"] = df.index.minute
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

        # 周期性编码
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

        return df

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加收益率特征"""
        df = df.copy()

        # 简单收益率
        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_15"] = df["close"].pct_change(15)
        df["return_60"] = df["close"].pct_change(60)

        # 对数收益率
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # 波动率
        df["volatility_10"] = df["return_1"].rolling(10).std()
        df["volatility_30"] = df["return_1"].rolling(30).std()

        return df

    def _add_macd(
        self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """添加 MACD 指标"""
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """添加 RSI 指标"""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # 使用 EMA 方法
        avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

        rs = avg_gain / avg_loss
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        return df

    def _add_ema(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """添加 EMA 指标"""
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
        return df

    def _add_bollinger(
        self, df: pd.DataFrame, period: int = 20, std_dev: float = 2
    ) -> pd.DataFrame:
        """添加布林带指标"""
        sma = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()

        df["boll_mid"] = sma
        df["boll_ub"] = sma + std_dev * std
        df["boll_lb"] = sma - std_dev * std
        df["boll_width"] = (df["boll_ub"] - df["boll_lb"]) / sma
        df["boll_pct"] = (df["close"] - df["boll_lb"]) / (df["boll_ub"] - df["boll_lb"])
        return df

    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """添加 ATR 指标"""
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=period).mean()
        return df

    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加 OBV 指标"""
        direction = np.sign(df["close"].diff())
        direction.iloc[0] = 0
        df["obv"] = (direction * df["volume"]).cumsum()
        return df

    def _add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """添加 CCI 指标"""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        df["cci"] = (tp - sma) / (0.015 * mad)
        return df

    def _add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """添加 ADX 指标"""
        high = df["high"]
        low = df["low"]

        plus_dm = high.diff()
        minus_dm = low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr = self._true_range(df)
        atr = tr.rolling(window=period).mean()

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (abs(minus_dm).rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df["adx"] = dx.rolling(window=period).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        return df

    def _true_range(self, df: pd.DataFrame) -> pd.Series:
        """计算真实波幅"""
        high = df["high"]
        low = df["low"]
        close = df["close"].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    def get_feature_columns(self) -> List[str]:
        """获取所有特征列名"""
        base_cols = ["open", "high", "low", "close", "volume"]
        time_cols = [
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
        return_cols = [
            "return_1",
            "return_5",
            "return_15",
            "return_60",
            "log_return",
            "volatility_10",
            "volatility_30",
        ]

        indicator_cols = []
        for ind in self.indicators:
            if ind == "macd":
                indicator_cols.extend(["macd", "macd_signal", "macd_hist"])
            elif ind.startswith("rsi"):
                indicator_cols.append(ind)
            elif ind.startswith("ema"):
                indicator_cols.append(ind)
            elif ind == "boll_ub":
                indicator_cols.extend(["boll_mid", "boll_ub", "boll_lb", "boll_width", "boll_pct"])
            elif ind == "atr":
                indicator_cols.append("atr")
            elif ind == "obv":
                indicator_cols.append("obv")
            elif ind == "cci":
                indicator_cols.append("cci")
            elif ind == "adx":
                indicator_cols.extend(["adx", "plus_di", "minus_di"])

        return base_cols + indicator_cols + time_cols + return_cols
