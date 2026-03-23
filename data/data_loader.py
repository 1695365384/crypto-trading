"""数据加载模块"""

import glob
import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.config import DataConfig


class DataLoader:
    """加密货币数据加载器"""

    # OKX 数据列名映射
    OKX_COLUMN_MAP = {
        "open_time": "timestamp",
        "vol_ccy": "volCcy",
        "vol_quote": "volCcyQuote",
    }

    def __init__(self, config: DataConfig):
        self.config = config
        self.data_cache: Dict[str, pd.DataFrame] = {}

    def load(self, data_path: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        加载价格数据

        Args:
            data_path: 数据文件路径，如果为None则尝试从API获取

        Returns:
            各交易对的价格数据字典
        """
        if data_path and os.path.exists(data_path):
            return self._load_from_file(data_path)
        else:
            return self._load_from_api()

    def _load_from_file(self, path: str) -> Dict[str, pd.DataFrame]:
        """从文件加载数据"""
        data = {}

        if os.path.isdir(path):
            # 目录：尝试多种文件名格式
            for ticker in self.config.tickers:
                # 提取基础币种名称（支持 BTCUSDT 和 BTC-USDT 两种格式）
                base_currency = ticker.replace("-", "").replace("USDT", "").replace("USD", "")

                # 尝试多种可能的文件名格式
                possible_names = [
                    f"{ticker}.csv",  # BTCUSDT.csv 或 BTC-USDT.csv
                    f"{ticker.replace('-', '')}.csv",  # BTCUSDT.csv
                    f"{ticker.replace('-', '')}_1m.csv",  # BTCUSDT_1m.csv
                    f"{ticker}_1m.csv",  # BTC-USDT_1m.csv
                ]

                file_path = None
                for name in possible_names:
                    candidate = os.path.join(path, name)
                    if os.path.exists(candidate):
                        file_path = candidate
                        break

                # 如果还是找不到，尝试模糊匹配
                if file_path is None:
                    # 使用基础币种名进行模糊匹配
                    patterns = [
                        os.path.join(path, f"{base_currency}USDT*.csv"),
                        os.path.join(path, f"{base_currency}-USDT*.csv"),
                        os.path.join(path, f"{base_currency.lower()}usdt*.csv"),
                    ]
                    for pattern in patterns:
                        matches = glob.glob(pattern)
                        if matches:
                            file_path = matches[0]
                            break

                if file_path:
                    df = self._load_csv_file(file_path, ticker)
                    if df is not None and not df.empty:
                        data[ticker] = df
                        print(f"  Loaded {ticker} from {os.path.basename(file_path)}")
                else:
                    print(f"  Warning: No data file found for {ticker}")
        else:
            # 单文件：包含所有交易对
            df = pd.read_csv(path)
            if "ticker" in df.columns or "instrument_name" in df.columns:
                ticker_col = "ticker" if "ticker" in df.columns else "instrument_name"
                for ticker in self.config.tickers:
                    ticker_df = df[df[ticker_col] == ticker].copy()
                    if not ticker_df.empty:
                        ticker_df = self._normalize_columns(ticker_df)
                        ticker_df.set_index("timestamp", inplace=True)
                        data[ticker] = ticker_df

        self.data_cache = data
        return data

    def _load_csv_file(self, file_path: str, ticker: str) -> Optional[pd.DataFrame]:
        """加载单个 CSV 文件"""
        try:
            df = pd.read_csv(file_path)

            # 检测并处理 OKX 格式
            if "open_time" in df.columns:
                df = self._normalize_okx_data(df)
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
            else:
                print(f"Warning: Unknown data format in {file_path}")
                return None

            return df

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def _normalize_okx_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化 OKX 数据格式"""
        # 重命名列
        df = df.rename(columns=self.OKX_COLUMN_MAP)

        # 转换时间戳（毫秒 -> datetime）
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms")
        elif "open_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["open_time"].astype(np.int64), unit="ms")

        # 确保数值列是 float 类型
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)

        # 删除不需要的列
        drop_cols = ["instrument_name", "confirm", "volCcy", "volCcyQuote"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # 设置索引并排序
        df = df.set_index("timestamp").sort_index()

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        return df.rename(columns=self.OKX_COLUMN_MAP)

    def _load_from_api(self) -> Dict[str, pd.DataFrame]:
        """从 API 加载数据 (需要实现具体API调用)"""
        # 这里可以集成 Binance API 或其他数据源
        # 目前生成模拟数据用于测试
        return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> Dict[str, pd.DataFrame]:
        """生成合成数据用于测试"""
        data = {}

        # 生成时间索引
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)
        freq = f"{self.config.timeframe}min"
        timestamps = pd.date_range(start=start, end=end, freq=freq)

        for ticker in self.config.tickers:
            # 基础价格
            if ticker == "BTCUSDT":
                base_price = 30000
                volatility = 0.02
            elif ticker == "ETHUSDT":
                base_price = 2000
                volatility = 0.025
            else:
                base_price = 100
                volatility = 0.03

            # 生成随机游走价格
            n = len(timestamps)
            returns = np.random.randn(n) * volatility / np.sqrt(252 * 24 * 60)
            prices = base_price * np.exp(np.cumsum(returns))

            # 生成 OHLCV 数据
            df = pd.DataFrame(index=timestamps)
            df["open"] = prices
            df["high"] = prices * (1 + np.abs(np.random.randn(n)) * 0.001)
            df["low"] = prices * (1 - np.abs(np.random.randn(n)) * 0.001)
            df["close"] = prices * (1 + np.random.randn(n) * 0.0005)
            df["volume"] = np.random.exponential(1000, n)

            data[ticker] = df

        self.data_cache = data
        return data

    def get_price_array(self) -> np.ndarray:
        """获取价格数组 [时间步, 资产数]"""
        prices = []
        for ticker in self.config.tickers:
            if ticker in self.data_cache:
                prices.append(self.data_cache[ticker]["close"].values)

        return np.column_stack(prices)

    def get_combined_dataframe(self) -> pd.DataFrame:
        """获取合并的数据框"""
        dfs = []
        for ticker in self.config.tickers:
            if ticker in self.data_cache:
                df = self.data_cache[ticker].copy()
                df.columns = [f"{col}_{ticker}" for col in df.columns]
                dfs.append(df)

        if dfs:
            return pd.concat(dfs, axis=1)
        return pd.DataFrame()

    def save(self, path: str):
        """保存数据到文件"""
        os.makedirs(path, exist_ok=True)

        for ticker, df in self.data_cache.items():
            file_path = os.path.join(path, f"{ticker}.csv")
            df.to_csv(file_path)
