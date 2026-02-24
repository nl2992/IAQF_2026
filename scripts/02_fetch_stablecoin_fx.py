"""
Fetch stablecoin FX pairs from Binance.US for March 1-21, 2023
"""
import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone

START_UTC = datetime(2023, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
END_UTC   = datetime(2023, 3, 21, 23, 59, 59, tzinfo=timezone.utc)
START_MS  = int(START_UTC.timestamp()) * 1000
END_MS    = int(END_UTC.timestamp()) * 1000

OUTPUT_DIR = "/home/ubuntu/iaqf_data/raw"
BINANCEUS_URL = "https://api.binance.us/api/v3/klines"
BINANCE_COLS = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_asset_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
]

def fetch_binanceus_klines(symbol: str) -> pd.DataFrame:
    all_rows = []
    cursor = START_MS
    print(f"  Fetching Binance.US {symbol} ...")
    while cursor < END_MS:
        params = {"symbol": symbol, "interval": "1m", "startTime": cursor,
                  "endTime": END_MS, "limit": 1000}
        for attempt in range(5):
            try:
                r = requests.get(BINANCEUS_URL, params=params, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    break
                elif r.status_code == 429:
                    time.sleep(60)
                    data = []
                else:
                    print(f"    HTTP {r.status_code}: {r.text[:200]}")
                    data = []
                    break
            except Exception as e:
                print(f"    Error: {e}")
                time.sleep(5)
                data = []
        if not data:
            break
        all_rows.extend(data)
        cursor = data[-1][0] + 60_000
        if len(data) < 1000:
            break
        time.sleep(0.12)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=BINANCE_COLS)
    df["timestamp_utc"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume"]:
        df[col] = pd.to_numeric(df[col])
    df["num_trades"] = pd.to_numeric(df["num_trades"])
    df = df[df["timestamp_utc"] <= pd.Timestamp(END_UTC)]
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    print(f"    → {len(df)} rows, {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
    return df

pairs = {
    "USDTUSD":  "binanceus_USDTUSD_1m.parquet",
    "USDCUSD":  "binanceus_USDCUSD_1m.parquet",
    "USDCUSDT": "binanceus_USDCUSDT_1m.parquet",
    "BUSDUSDT": "binanceus_BUSDUSDT_1m.parquet",
}

for symbol, fname in pairs.items():
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        df_tmp = pd.read_parquet(fpath)
        print(f"  {symbol}: cached ({len(df_tmp)} rows)")
        continue
    df = fetch_binanceus_klines(symbol)
    if not df.empty:
        df.to_parquet(fpath, index=False)
        print(f"  Saved: {fname}")

print("\nDone. Files:")
import glob
for f in sorted(glob.glob(OUTPUT_DIR + "/*.parquet")):
    df_tmp = pd.read_parquet(f)
    print(f"  {os.path.basename(f):45s}: {len(df_tmp):6d} rows")
