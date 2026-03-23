"""
OKX 数据提供者
支持 REST API 获取历史数据，WebSocket 获取实时行情
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# OKX SDK
from okx import MarketData
from okx.websocket import WsPublicAsync

logger = logging.getLogger(__name__)


@dataclass
class OKXConfig:
    """OKX 配置"""

    # API 凭证 (可选，公开数据不需要)
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""

    # 是否使用模拟环境
    demo: bool = False

    # WebSocket URL
    ws_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    ws_business_url: str = "wss://ws.okx.com:8443/ws/v5/business"


class OKXRestProvider:
    """OKX REST API 数据提供者"""

    # K 线周期映射
    BAR_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1H": "1H",
        "2H": "2H",
        "4H": "4H",
        "6H": "6H",
        "12H": "12H",
        "1D": "1D",
        "1W": "1W",
        "1M": "1Mutc",
    }

    def __init__(self, config: Optional[OKXConfig] = None):
        """
        初始化 REST API 客户端

        Args:
            config: OKX 配置
        """
        self.config = config or OKXConfig()
        self._client = None

    @property
    def client(self):
        """延迟初始化客户端"""
        if self._client is None:
            # 将空字符串转换为 None
            api_key = self.config.api_key if self.config.api_key else None
            secret_key = self.config.secret_key if self.config.secret_key else None
            passphrase = self.config.passphrase if self.config.passphrase else None

            self._client = MarketData.MarketAPI(
                api_key=api_key,
                api_secret_key=secret_key,
                passphrase=passphrase,
                flag="1" if self.config.demo else "0",
                debug=False,
            )
        return self._client

    def get_candlesticks(
        self,
        instId: str,
        bar: str = "1m",
        limit: int = 100,
        before: Optional[str] = None,
        after: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取 K 线数据

        Args:
            instId: 产品 ID，如 'BTC-USDT'
            bar: K 线周期，如 '1m', '5m', '1H', '1D'
            limit: 返回数量，最大 300
            before: 请求此时间戳之前的数据 (毫秒)
            after: 请求此时间戳之后的数据 (毫秒)

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume, volCcy, volCcyQuote, confirm]
        """
        okx_bar = self.BAR_MAP.get(bar, bar)

        params = {"instId": instId, "bar": okx_bar, "limit": str(min(limit, 300))}

        if before:
            params["before"] = str(before)
        if after:
            params["after"] = str(after)

        try:
            response = self.client.get_candlesticks(**params)

            if response.get("code") != "0":
                logger.error(f"OKX API error: {response.get('msg')}")
                return pd.DataFrame()

            data = response.get("data", [])

            if not data:
                return pd.DataFrame()

            # 解析数据
            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "volCcy",
                    "volCcyQuote",
                    "confirm",
                ],
            )

            # 转换类型
            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms")
            for col in ["open", "high", "low", "close", "volume", "volCcy", "volCcyQuote"]:
                df[col] = df[col].astype(float)
            df["confirm"] = df["confirm"].astype(int)

            # 按时间排序 (OKX 返回的是倒序)
            df = df.sort_values("timestamp").reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching candlesticks: {e}")
            return pd.DataFrame()

    def get_history_candlesticks(
        self,
        instId: str,
        bar: str = "1m",
        limit: int = 100,
        before: Optional[str] = None,
        after: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取历史 K 线数据 (时间范围可长达 2 年)

        Args:
            instId: 产品 ID
            bar: K 线周期
            limit: 返回数量
            before: 请求此时间戳之前的数据
            after: 请求此时间戳之后的数据

        Returns:
            DataFrame
        """
        okx_bar = self.BAR_MAP.get(bar, bar)

        params = {"instId": instId, "bar": okx_bar, "limit": str(min(limit, 100))}

        if before:
            params["before"] = str(before)
        if after:
            params["after"] = str(after)

        try:
            response = self.client.get_history_candlesticks(**params)

            if response.get("code") != "0":
                logger.error(f"OKX API error: {response.get('msg')}")
                return pd.DataFrame()

            data = response.get("data", [])

            if not data:
                return pd.DataFrame()

            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "volCcy",
                    "volCcyQuote",
                    "confirm",
                ],
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"].astype(np.int64), unit="ms")
            for col in ["open", "high", "low", "close", "volume", "volCcy", "volCcyQuote"]:
                df[col] = df[col].astype(float)
            df["confirm"] = df["confirm"].astype(int)

            df = df.sort_values("timestamp").reset_index(drop=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching history candlesticks: {e}")
            return pd.DataFrame()

    def get_ticker(self, instId: str) -> Dict:
        """
        获取行情数据

        Args:
            instId: 产品 ID

        Returns:
            行情字典
        """
        try:
            response = self.client.get_ticker(instId=instId)

            if response.get("code") != "0":
                logger.error(f"OKX API error: {response.get('msg')}")
                return {}

            data = response.get("data", [])
            return data[0] if data else {}

        except Exception as e:
            logger.error(f"Error fetching ticker: {e}")
            return {}

    def get_tickers(self, instType: str = "SPOT") -> List[Dict]:
        """
        获取所有行情

        Args:
            instType: 产品类型 'SPOT', 'MARGIN', 'SWAP', 'FUTURES', 'OPTION'

        Returns:
            行情列表
        """
        try:
            response = self.client.get_tickers(instType=instType)

            if response.get("code") != "0":
                logger.error(f"OKX API error: {response.get('msg')}")
                return []

            data: List[Dict] = response.get("data", [])
            return data

        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return []

    def fetch_historical_data(
        self,
        instId: str,
        bar: str = "1m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        total_limit: int = 10000,
    ) -> pd.DataFrame:
        """
        批量获取历史数据 (支持分页)

        Args:
            instId: 产品 ID
            bar: K 线周期
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            total_limit: 总数据量限制

        Returns:
            DataFrame
        """
        all_data = []

        # 转换日期为时间戳
        if end_date:
            after_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            after_ts = None

        if start_date:
            before_ts = int(
                (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=1)).timestamp() * 1000
            )
        else:
            before_ts = None

        current_after = after_ts
        fetched = 0
        batch_size = 300

        while fetched < total_limit:
            # 获取一批数据
            df = self.get_candlesticks(
                instId=instId,
                bar=bar,
                limit=min(batch_size, total_limit - fetched),
                before=str(before_ts) if before_ts else None,
                after=str(current_after) if current_after else None,
            )

            if df.empty:
                break

            all_data.append(df)
            fetched += len(df)

            # 更新 after 参数获取更早的数据
            # OKX 返回的是按时间倒序，最早的时间在最后
            oldest_ts = int(df["timestamp"].iloc[0].timestamp() * 1000)
            current_after = oldest_ts - 1

            # 检查是否到达开始日期
            if before_ts and oldest_ts >= before_ts:
                break

            # 避免请求过于频繁
            import time

            time.sleep(0.1)

        if not all_data:
            return pd.DataFrame()

        # 合并并去重
        result = pd.concat(all_data, ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"])
        result = result.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Fetched {len(result)} candles for {instId} {bar}")

        return result


class OKXWebSocketProvider:
    """OKX WebSocket 实时数据提供者"""

    # K 线频道映射
    CANDLE_CHANNELS = {
        "1s": "candle1s",
        "1m": "candle1m",
        "5m": "candle5m",
        "15m": "candle15m",
        "30m": "candle30m",
        "1H": "candle1H",
        "2H": "candle2H",
        "4H": "candle4H",
        "6H": "candle6H",
        "12H": "candle12H",
        "1D": "candle1D",
        "1W": "candle1W",
        "1M": "candle1Mutc",
    }

    def __init__(self, config: Optional[OKXConfig] = None):
        """
        初始化 WebSocket 客户端

        Args:
            config: OKX 配置
        """
        self.config = config or OKXConfig()
        self._ws_client: Optional[WsPublicAsync.WsPublicAsync] = None
        self._callbacks: Dict[str, List[Callable]] = {}
        self._running = False
        self._data_buffer: Dict[str, List[Dict]] = {}

    async def start(self):
        """启动 WebSocket 连接"""
        if self._running:
            return

        self._ws_client = WsPublicAsync.WsPublicAsync(url=self.config.ws_business_url)
        await self._ws_client.start()
        self._running = True
        logger.info("WebSocket connection started")

    async def stop(self):
        """停止 WebSocket 连接"""
        if self._ws_client and self._running:
            await self._ws_client.stop()
            self._running = False
            logger.info("WebSocket connection stopped")

    def _on_message(self, message: str):
        """处理消息"""
        try:
            data = json.loads(message)

            # 忽略订阅确认消息
            if "event" in data:
                return

            arg = data.get("arg", {})
            channel = arg.get("channel", "")
            instId = arg.get("instId", "")

            key = f"{channel}_{instId}"

            # 存储数据
            if key not in self._data_buffer:
                self._data_buffer[key] = []

            candle_data = data.get("data", [])
            if candle_data:
                self._data_buffer[key].extend(candle_data)

            # 调用回调
            if key in self._callbacks:
                for callback in self._callbacks[key]:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def subscribe_candlesticks(
        self, instIds: List[str], bar: str = "1m", callback: Optional[Callable] = None
    ):
        """
        订阅 K 线数据

        Args:
            instIds: 产品 ID 列表
            bar: K 线周期
            callback: 数据回调函数
        """
        if not self._running:
            await self.start()

        channel = self.CANDLE_CHANNELS.get(bar, f"candle{bar}")

        args = []
        for instId in instIds:
            key = f"{channel}_{instId}"
            if callback:
                if key not in self._callbacks:
                    self._callbacks[key] = []
                self._callbacks[key].append(callback)

            args.append({"channel": channel, "instId": instId})

        if self._ws_client is not None:
            await self._ws_client.subscribe(args, self._on_message)
        logger.info(f"Subscribed to candlesticks: {instIds} {bar}")

    async def subscribe_tickers(self, instIds: List[str], callback: Optional[Callable] = None):
        """
        订阅行情数据

        Args:
            instIds: 产品 ID 列表
            callback: 数据回调函数
        """
        if not self._running:
            await self.start()

        args = []
        for instId in instIds:
            key = f"tickers_{instId}"
            if callback:
                if key not in self._callbacks:
                    self._callbacks[key] = []
                self._callbacks[key].append(callback)

            args.append({"channel": "tickers", "instId": instId})

        if self._ws_client is not None:
            await self._ws_client.subscribe(args, self._on_message)
        logger.info(f"Subscribed to tickers: {instIds}")

    async def subscribe_orderbook(self, instIds: List[str], callback: Optional[Callable] = None):
        """
        订阅订单簿数据

        Args:
            instIds: 产品 ID 列表
            callback: 数据回调函数
        """
        if not self._running:
            await self.start()

        args = []
        for instId in instIds:
            key = f"books_{instId}"
            if callback:
                if key not in self._callbacks:
                    self._callbacks[key] = []
                self._callbacks[key].append(callback)

            args.append({"channel": "books", "instId": instId})

        if self._ws_client is not None:
            await self._ws_client.subscribe(args, self._on_message)
        logger.info(f"Subscribed to orderbook: {instIds}")

    async def unsubscribe_candlesticks(self, instIds: List[str], bar: str = "1m"):
        """取消订阅 K 线数据"""
        channel = self.CANDLE_CHANNELS.get(bar, f"candle{bar}")

        args = [{"channel": channel, "instId": instId} for instId in instIds]
        if self._ws_client is not None:
            await self._ws_client.unsubscribe(args, self._on_message)

    def get_buffered_data(self, instId: str, bar: str = "1m") -> List[Dict]:
        """获取缓冲的数据"""
        channel = self.CANDLE_CHANNELS.get(bar, f"candle{bar}")
        key = f"{channel}_{instId}"
        return self._data_buffer.get(key, [])

    def clear_buffer(self, instId: Optional[str] = None, bar: Optional[str] = None):
        """清空缓冲区"""
        if instId and bar:
            channel = self.CANDLE_CHANNELS.get(bar, f"candle{bar}")
            key = f"{channel}_{instId}"
            if key in self._data_buffer:
                self._data_buffer[key] = []
        else:
            self._data_buffer.clear()


class OKXDataProvider:
    """OKX 数据提供者 (整合 REST 和 WebSocket)"""

    def __init__(self, config: Optional[OKXConfig] = None):
        """
        初始化数据提供者

        Args:
            config: OKX 配置
        """
        self.config = config or OKXConfig()
        self.rest = OKXRestProvider(self.config)
        self.ws = OKXWebSocketProvider(self.config)

    async def get_realtime_candlesticks(
        self,
        instIds: List[str],
        bar: str = "1m",
        callback: Optional[Callable] = None,
        duration: int = 60,
    ):
        """
        获取实时 K 线数据

        Args:
            instIds: 产品 ID 列表
            bar: K 线周期
            callback: 数据回调函数
            duration: 运行时长 (秒)
        """
        await self.ws.subscribe_candlesticks(instIds, bar, callback)
        await asyncio.sleep(duration)
        await self.ws.stop()

    def fetch_and_prepare(
        self,
        instIds: List[str],
        bar: str = "1m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        获取并准备历史数据

        Args:
            instIds: 产品 ID 列表
            bar: K 线周期
            start_date: 开始日期
            end_date: 结束日期
            save_path: 保存路径

        Returns:
            数据字典 {instId: DataFrame}
        """
        data = {}

        for instId in instIds:
            logger.info(f"Fetching data for {instId}...")

            df = self.rest.fetch_historical_data(
                instId=instId, bar=bar, start_date=start_date, end_date=end_date
            )

            if not df.empty:
                data[instId] = df

                if save_path:
                    import os

                    os.makedirs(save_path, exist_ok=True)
                    file_path = os.path.join(save_path, f"{instId.replace('-', '')}_{bar}.csv")
                    df.to_csv(file_path, index=False)
                    logger.info(f"Saved to {file_path}")

        return data


# 使用示例
if __name__ == "__main__":
    # REST API 示例
    provider = OKXDataProvider()

    # 获取历史数据
    data = provider.fetch_and_prepare(
        instIds=["BTC-USDT", "ETH-USDT"],
        bar="1m",
        start_date="2024-01-01",
        end_date="2024-01-31",
        save_path="./data/okx/",
    )

    for instId, df in data.items():
        print(f"\n{instId}: {len(df)} rows")
        print(df.head())

    # WebSocket 实时数据示例
    async def realtime_example():
        provider = OKXDataProvider()

        def on_candle(message):
            data = json.loads(message)
            if "data" in data:
                for candle in data["data"]:
                    print(f"Candle: {candle}")

        await provider.get_realtime_candlesticks(
            instIds=["BTC-USDT"], bar="1m", callback=on_candle, duration=60
        )

    # 运行实时数据
    # asyncio.run(realtime_example())
