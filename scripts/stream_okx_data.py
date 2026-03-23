"""
OKX 实时数据流脚本
用法: python scripts/stream_okx_data.py --symbols BTC-USDT ETH-USDT --bar 1m
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.okx_provider import OKXDataProvider


class RealtimeDataCollector:
    """实时数据收集器"""

    def __init__(self, output_dir: str = "./data/okx/realtime"):
        self.output_dir = output_dir
        self.buffer = {}
        os.makedirs(output_dir, exist_ok=True)

    def on_candle(self, message: str):
        """处理 K 线消息"""
        try:
            data = json.loads(message)

            if "data" not in data:
                return

            arg = data.get("arg", {})
            instId = arg.get("instId", "")
            # channel = arg.get("channel", "")  # 可用于日志调试

            if not instId:
                return

            # 初始化缓冲
            if instId not in self.buffer:
                self.buffer[instId] = []

            # 存储数据
            for candle in data["data"]:
                # candle: [ts, o, h, l, c, vol, volCcy, volCcyQuote, confirm]
                record = {
                    "timestamp": datetime.fromtimestamp(int(candle[0]) / 1000),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                    "volCcy": float(candle[6]),
                    "volCcyQuote": float(candle[7]),
                    "confirm": int(candle[8]),
                }
                self.buffer[instId].append(record)

                # 打印实时数据
                confirm_str = "✓" if record["confirm"] else "○"
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{instId} {confirm_str} "
                    f"O:{record['open']:.2f} H:{record['high']:.2f} "
                    f"L:{record['low']:.2f} C:{record['close']:.2f} "
                    f"V:{record['volume']:.2f}"
                )

        except Exception as e:
            print(f"Error processing message: {e}")

    def save_buffer(self):
        """保存缓冲数据"""
        for instId, records in self.buffer.items():
            if not records:
                continue

            df = pd.DataFrame(records)
            filename = (
                f"{instId.replace('-', '')}_realtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved {len(records)} records to {filepath}")

        self.buffer.clear()


async def main():
    parser = argparse.ArgumentParser(description="Stream real-time data from OKX")
    parser.add_argument(
        "--symbols", nargs="+", default=["BTC-USDT", "ETH-USDT"], help="Trading symbols"
    )
    parser.add_argument(
        "--bar",
        type=str,
        default="1m",
        choices=["1s", "1m", "5m", "15m", "30m", "1H", "2H", "4H", "6H", "12H", "1D"],
        help="Kline interval",
    )
    parser.add_argument(
        "--duration", type=int, default=3600, help="Duration in seconds (0 for infinite)"
    )
    parser.add_argument(
        "--output", type=str, default="./data/okx/realtime", help="Output directory"
    )
    parser.add_argument("--save-interval", type=int, default=300, help="Save interval in seconds")

    args = parser.parse_args()

    print("=" * 60)
    print("OKX Real-time Data Streamer")
    print("=" * 60)
    print(f"Symbols: {args.symbols}")
    print(f"Interval: {args.bar}")
    print(f"Duration: {args.duration}s ({'infinite' if args.duration == 0 else ''})")
    print(f"Output: {args.output}")
    print("=" * 60)
    print("\nPress Ctrl+C to stop...\n")

    # 创建收集器
    collector = RealtimeDataCollector(output_dir=args.output)

    # 创建数据提供者
    provider = OKXDataProvider()

    # 启动 WebSocket
    await provider.ws.start()

    # 订阅 K 线
    await provider.ws.subscribe_candlesticks(
        instIds=args.symbols, bar=args.bar, callback=collector.on_candle
    )

    # 运行
    try:
        if args.duration > 0:
            elapsed = 0
            while elapsed < args.duration:
                await asyncio.sleep(min(args.save_interval, args.duration - elapsed))
                elapsed += args.save_interval
                collector.save_buffer()
        else:
            # 无限运行
            while True:
                await asyncio.sleep(args.save_interval)
                collector.save_buffer()

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        # 保存剩余数据
        collector.save_buffer()
        await provider.ws.stop()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
