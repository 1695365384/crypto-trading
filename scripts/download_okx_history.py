"""
OKX 历史数据下载脚本 - 使用 market-data-history API (月度聚合)
下载 1 分钟 K 线历史数据
用法: python scripts/download_okx_history.py --start 2025-03-01 --end 2026-03-23
"""

import argparse
import logging
import os
import sys
import time
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytz
import requests

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from okx import PublicData

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """加载配置"""
    config_file = Path(config_path)
    if not config_file.exists():
        config_file = Path(__file__).parent.parent / config_path
    if not config_file.exists():
        return {}
    with open(config_file, "rb") as f:
        return tomllib.load(f)


class OKXHistoryDownloader:
    """OKX 历史数据下载器"""

    def __init__(self, api_key: str, secret_key: str, passphrase: str, rate_limit: float = 2.5):
        self.client = PublicData.PublicAPI(
            api_key=api_key,
            api_secret_key=secret_key,
            passphrase=passphrase,
            flag="0",
            debug=False,
        )
        self.rate_limit = rate_limit
        self.last_request = 0
        self.tz = pytz.timezone("Asia/Shanghai")

    def _wait(self):
        """速率限制"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()

    def get_file_list(
        self, symbol: str, inst_type: str, start_date: str, end_date: str
    ) -> List[Dict]:
        """获取月度数据文件列表"""
        begin_ts = int(
            self.tz.localize(datetime.strptime(start_date, "%Y-%m-%d")).timestamp() * 1000
        )
        end_ts = int(self.tz.localize(datetime.strptime(end_date, "%Y-%m-%d")).timestamp() * 1000)

        self._wait()

        params = {
            "module": "2",  # 1分钟K线
            "instType": inst_type,
            "dateAggrType": "monthly",
            "begin": str(begin_ts),
            "end": str(end_ts),
        }

        if inst_type == "SPOT":
            params["instIdList"] = symbol
        else:
            params["instFamilyList"] = symbol

        try:
            r = self.client.get_market_data_history(**params)

            if r.get("code") != "0":
                logger.error(f"API error: {r.get('msg')}")
                return []

            files = []
            data = r.get("data", [])
            if data:
                for detail in data[0].get("details", []):
                    for gd in detail.get("groupDetails", []):
                        files.append(
                            {
                                "filename": gd["filename"],
                                "url": gd["url"],
                                "sizeMB": float(gd["sizeMB"]),
                            }
                        )

            logger.info(f"Found {len(files)} monthly files for {symbol}")
            return files

        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    def download_file(self, file_info: Dict) -> pd.DataFrame:
        """下载并解压单个文件"""
        url = file_info["url"]
        filename = file_info["filename"]

        try:
            logger.info(f"Downloading {filename} ({file_info['sizeMB']:.2f} MB)...")
            r = requests.get(url, timeout=120)
            r.raise_for_status()

            with zipfile.ZipFile(BytesIO(r.content)) as zf:
                csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
                if not csv_files:
                    return pd.DataFrame()
                with zf.open(csv_files[0]) as csv_file:
                    df = pd.read_csv(csv_file)
            return df

        except Exception as e:
            logger.error(f"Download error: {e}")
            return pd.DataFrame()

    def download_all(
        self, symbol: str, inst_type: str, start: str, end: str, output_dir: str
    ) -> pd.DataFrame:
        """下载所有数据并合并"""
        files = self.get_file_list(symbol, inst_type, start, end)
        if not files:
            return pd.DataFrame()

        all_data = []
        for i, f in enumerate(files):
            logger.info(f"[{i+1}/{len(files)}] {f['filename']}")
            df = self.download_file(f)
            if not df.empty:
                all_data.append(df)
            time.sleep(0.5)

        if not all_data:
            return pd.DataFrame()

        combined = pd.concat(all_data, ignore_index=True)

        # 标准化列名
        col_map = {
            "ts": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "vol": "volume",
            "volume": "volume",
        }
        for old, new in col_map.items():
            if old in combined.columns and new not in combined.columns:
                combined.rename(columns={old: new}, inplace=True)

        # 时间戳处理
        if "timestamp" in combined.columns:
            if combined["timestamp"].dtype != "datetime64[ns]":
                try:
                    combined["timestamp"] = pd.to_datetime(combined["timestamp"], unit="ms")
                except (ValueError, TypeError):
                    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
            combined = combined.drop_duplicates(subset=["timestamp"])
            combined = combined.sort_values("timestamp").reset_index(drop=True)

        # 保存
        output_path = Path(output_dir) / f"{symbol.replace('-', '')}_1m.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"Saved {len(combined):,} rows to {output_path}")

        return combined


def main():
    parser = argparse.ArgumentParser(description="Download OKX historical 1m candlesticks")
    parser.add_argument("--config", default="settings.toml")
    parser.add_argument("--symbols", nargs="+", help="e.g. BTC-USDT ETH-USDT")
    parser.add_argument("--start", default="2025-03-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2026-03-23", help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="./data/okx")
    parser.add_argument("--inst-type", default="SPOT", choices=["SPOT", "SWAP"])

    args = parser.parse_args()

    config = load_config(args.config)
    okx_cfg = config.get("okx", {})
    data_cfg = config.get("data", {})

    api_key = okx_cfg.get("api_key", "")
    secret_key = okx_cfg.get("secret_key", "")
    passphrase = okx_cfg.get("passphrase", "")

    symbols = args.symbols or data_cfg.get("symbols", ["BTC-USDT", "ETH-USDT"])
    output_dir = args.output or data_cfg.get("output_dir", "./data/okx")

    print("=" * 60)
    print("OKX Historical Data Downloader")
    print("=" * 60)
    print(f"Symbols: {symbols}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Type: {args.inst_type}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    if not all([api_key, secret_key, passphrase]):
        logger.error("Missing API credentials in settings.toml")
        sys.exit(1)

    downloader = OKXHistoryDownloader(api_key, secret_key, passphrase)
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    total_start = time.time()

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Downloading {symbol}...")
        print("=" * 60)

        df = downloader.download_all(symbol, args.inst_type, args.start, args.end, output_dir)

        if not df.empty:
            results[symbol] = df
            print(f"\n{symbol}: {len(df):,} rows")
            if "timestamp" in df.columns:
                print(f"  Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total = sum(len(df) for df in results.values())
    for s, df in results.items():
        print(f"{s}: {len(df):,} rows")
    print(f"\nTotal: {total:,} rows")
    print(f"Time: {time.time() - total_start:.1f}s")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
