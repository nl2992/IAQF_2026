from pathlib import Path
"""
IAQF 2026 Data Retrieval Script
Fetches 1-minute OHLCV candle data for BTC pairs across Binance.US and Coinbase
Period: March 1–21, 2023 (UTC)
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timezone

# ── Time window ──────────────────────────────────────────────────────────────
START_UTC = datetime(2023, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
END_UTC   = datetime(2023, 3, 21, 23, 59, 59, tzinfo=timezone.utc)
START_TS  = int(START_UTC.timestamp())   # seconds
END_TS    = int(END_UTC.timestamp())     # seconds
START_MS  = START_TS * 1000              # milliseconds
END_MS    = END_TS * 1000               # milliseconds

OUTPUT_DIR = str(Path(__file__).parent.parent / "data" / "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Binance.US 1-minute klines ────────────────────────────────────────────────
BINANCEUS_URL = "https://api.binance.us/api/v3/klines"
BINANCE_COLS = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_asset_volume", "num_trades",
    "taker_buy_base_vol", "taker_buy_quote_vol", "ignore"
]

def fetch_binanceus_klines(symbol: str, interval: str = "1m") -> pd.DataFrame:
    """Fetch all 1-minute klines from Binance.US for the study window."""
    all_rows = []
    cursor = START_MS
    print(f"  Fetching Binance.US {symbol} ...")
    while cursor < END_MS:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": END_MS,
            "limit": 1000,
        }
        for attempt in range(5):
            try:
                r = requests.get(BINANCEUS_URL, params=params, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    break
                elif r.status_code == 429:
                    print("    Rate limited, sleeping 60s...")
                    time.sleep(60)
                    data = []
                else:
                    print(f"    HTTP {r.status_code}: {r.text[:200]}")
                    data = []
                    break
            except Exception as e:
                print(f"    Error: {e}, retry {attempt+1}")
                time.sleep(5)
                data = []
        if not data:
            break
        all_rows.extend(data)
        last_ts = data[-1][0]
        cursor = last_ts + 60_000  # advance by 1 minute
        if len(data) < 1000:
            break
        time.sleep(0.12)  # polite rate limiting

    if not all_rows:
        print(f"  WARNING: No data for {symbol}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=BINANCE_COLS)
    df["timestamp_utc"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_asset_volume",
                "taker_buy_base_vol", "taker_buy_quote_vol"]:
        df[col] = pd.to_numeric(df[col])
    df["num_trades"] = pd.to_numeric(df["num_trades"])
    df = df[df["timestamp_utc"] <= pd.Timestamp(END_UTC)]
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    print(f"    → {len(df)} rows, {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
    return df

# ── Coinbase Advanced Trade API ───────────────────────────────────────────────
CB_URL_TMPL = "https://api.coinbase.com/api/v3/brokerage/market/products/{product_id}/candles"
CB_MAX_CANDLES = 300  # Coinbase returns max 300 candles per request

def fetch_coinbase_candles(product_id: str, granularity: str = "ONE_MINUTE") -> pd.DataFrame:
    """Fetch all 1-minute candles from Coinbase for the study window."""
    all_rows = []
    cursor_start = START_TS
    step = CB_MAX_CANDLES * 60  # 300 minutes per chunk
    url = CB_URL_TMPL.format(product_id=product_id)
    print(f"  Fetching Coinbase {product_id} ...")
    while cursor_start < END_TS:
        chunk_end = min(cursor_start + step, END_TS)
        params = {
            "start": str(cursor_start),
            "end": str(chunk_end),
            "granularity": granularity,
        }
        for attempt in range(5):
            try:
                r = requests.get(url, params=params, timeout=20)
                if r.status_code == 200:
                    data = r.json().get("candles", [])
                    break
                elif r.status_code == 429:
                    print("    Rate limited, sleeping 30s...")
                    time.sleep(30)
                    data = []
                else:
                    print(f"    HTTP {r.status_code}: {r.text[:200]}")
                    data = []
                    break
            except Exception as e:
                print(f"    Error: {e}, retry {attempt+1}")
                time.sleep(5)
                data = []
        all_rows.extend(data)
        cursor_start = chunk_end + 60
        time.sleep(0.15)

    if not all_rows:
        print(f"  WARNING: No data for {product_id}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["timestamp_utc"] = pd.to_datetime(df["start"].astype(int), unit="s", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df = df[(df["timestamp_utc"] >= pd.Timestamp(START_UTC)) &
            (df["timestamp_utc"] <= pd.Timestamp(END_UTC))]
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    print(f"    → {len(df)} rows, {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
    return df

# ── Kraken OHLC (stablecoin FX) ───────────────────────────────────────────────
KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"

def fetch_kraken_ohlc(pair: str) -> pd.DataFrame:
    """Fetch 1-minute OHLC from Kraken for the study window."""
    all_rows = []
    cursor = START_TS
    print(f"  Fetching Kraken {pair} ...")
    while cursor < END_TS:
        params = {"pair": pair, "interval": 1, "since": cursor}
        try:
            r = requests.get(KRAKEN_URL, params=params, timeout=20)
            if r.status_code == 200:
                result = r.json().get("result", {})
                pair_keys = [k for k in result.keys() if k != "last"]
                if not pair_keys:
                    break
                rows = result[pair_keys[0]]
                if not rows:
                    break
                all_rows.extend(rows)
                cursor = rows[-1][0] + 60
                if len(rows) < 720:
                    break
            else:
                print(f"    Kraken HTTP {r.status_code}")
                break
        except Exception as e:
            print(f"    Kraken error: {e}")
            break
        time.sleep(0.2)

    if not all_rows:
        print(f"  WARNING: No data for {pair}")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["time","open","high","low","close","vwap","volume","count"])
    df["timestamp_utc"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    for col in ["open","high","low","close","vwap","volume"]:
        df[col] = pd.to_numeric(df[col])
    df = df[(df["timestamp_utc"] >= pd.Timestamp(START_UTC)) &
            (df["timestamp_utc"] <= pd.Timestamp(END_UTC))]
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    print(f"    → {len(df)} rows, {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
    return df

# ── Main retrieval ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("IAQF Data Retrieval: March 1–21, 2023 (1-minute candles)")
    print("=" * 60)

    # ── Binance.US pairs ──
    print("\n[Binance.US]")
    binanceus_pairs = {
        "BTCUSDT": "binanceus_BTCUSDT_1m.parquet",
        "BTCUSDC": "binanceus_BTCUSDC_1m.parquet",
        "BTCUSD":  "binanceus_BTCUSD_1m.parquet",   # USD-quoted BTC (Binance.US)
        "BTCBUSD": "binanceus_BTCBUSD_1m.parquet",   # BUSD proxy
    }
    for symbol, fname in binanceus_pairs.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            df_tmp = pd.read_parquet(fpath)
            print(f"  {symbol}: already cached ({len(df_tmp)} rows), skipping.")
            continue
        df = fetch_binanceus_klines(symbol)
        if not df.empty:
            df.to_parquet(fpath, index=False)
            print(f"  Saved: {fname}")

    # ── Coinbase pairs ──
    print("\n[Coinbase]")
    coinbase_pairs = {
        "BTC-USD":  "coinbase_BTCUSD_1m.parquet",
        "BTC-USDC": "coinbase_BTCUSDC_1m.parquet",
        "BTC-USDT": "coinbase_BTCUSDT_1m.parquet",
    }
    for product_id, fname in coinbase_pairs.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            df_tmp = pd.read_parquet(fpath)
            print(f"  {product_id}: already cached ({len(df_tmp)} rows), skipping.")
            continue
        df = fetch_coinbase_candles(product_id)
        if not df.empty:
            df.to_parquet(fpath, index=False)
            print(f"  Saved: {fname}")

    # ── Stablecoin FX via Kraken ──
    print("\n[Kraken — Stablecoin FX]")
    kraken_pairs = {
        "USDTUSD": "kraken_USDTUSD_1m.parquet",
        "USDCUSD": "kraken_USDCUSD_1m.parquet",
    }
    for pair, fname in kraken_pairs.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            df_tmp = pd.read_parquet(fpath)
            print(f"  {pair}: already cached ({len(df_tmp)} rows), skipping.")
            continue
        df = fetch_kraken_ohlc(pair)
        if not df.empty:
            df.to_parquet(fpath, index=False)
            print(f"  Saved: {fname}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("✓ Data retrieval complete.")
    print("Files saved in:", OUTPUT_DIR)
    import glob
    files = sorted(glob.glob(OUTPUT_DIR + "/*.parquet"))
    total_rows = 0
    for f in files:
        df_tmp = pd.read_parquet(f)
        total_rows += len(df_tmp)
        print(f"  {os.path.basename(f):45s}: {len(df_tmp):6d} rows")
    print(f"\n  TOTAL: {total_rows:,} rows across {len(files)} files")
