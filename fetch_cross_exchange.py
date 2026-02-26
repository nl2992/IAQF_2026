"""
fetch_cross_exchange.py
Fetches:
  1. Kraken tick trades -> 1-min OHLC for XBTUSD, USDCUSD, USDTUSD, USDCUSDT (Mar 1-21 2023)
  2. Bybit public trade archive -> 1-min OHLC + microstructure for BTCUSDT (Mar 2023)
Saves to: data/cross_exchange/
"""
import requests, time, datetime, gzip, io, os
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent / "data" / "cross_exchange"
OUT.mkdir(parents=True, exist_ok=True)

START_DT = datetime.datetime(2023, 3, 1, 0, 0, 0)
END_DT   = datetime.datetime(2023, 3, 22, 0, 0, 0)  # exclusive
START_TS = int(START_DT.timestamp())
END_TS   = int(END_DT.timestamp())

# ─────────────────────────────────────────────────────────────────────────────
# 1. KRAKEN — paginated trades -> 1-min OHLC
# ─────────────────────────────────────────────────────────────────────────────
KRAKEN_PAIRS = {
    "XBTUSD":   "kraken_btcusd",
    "USDCUSD":  "kraken_usdcusd",
    "USDTUSD":  "kraken_usdtusd",
    "USDCUSDT": "kraken_usdcusdt",
}

def fetch_kraken_trades(pair: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    """Paginate Kraken Trades endpoint and return all trades in [start_ts, end_ts)."""
    BASE = "https://api.kraken.com/0/public"
    # Kraken 'since' is in nanoseconds for Trades endpoint
    since_ns = start_ts * 1_000_000_000
    end_ns   = end_ts   * 1_000_000_000
    all_trades = []
    calls = 0
    while True:
        try:
            r = requests.get(f"{BASE}/Trades",
                             params={"pair": pair, "since": since_ns, "count": 1000},
                             timeout=20)
            data = r.json()
        except Exception as e:
            print(f"  Request error: {e}, retrying in 5s")
            time.sleep(5)
            continue

        if data.get("error"):
            print(f"  Kraken error for {pair}: {data['error']}")
            time.sleep(2)
            continue

        result_key = [k for k in data["result"] if k != "last"][0]
        trades = data["result"][result_key]
        last_ns = int(data["result"]["last"])
        calls += 1

        # Filter to our window
        batch = [(float(t[2]), float(t[0]), float(t[1]), t[3]) for t in trades
                 if float(t[2]) * 1e9 < end_ns]
        all_trades.extend(batch)

        # Progress
        if batch:
            latest = datetime.datetime.utcfromtimestamp(batch[-1][0])
            print(f"  {pair}: call {calls}, {len(all_trades)} trades, latest={latest}", flush=True)

        # Stop conditions
        if last_ns >= end_ns:
            break
        if not trades:
            break
        since_ns = last_ns
        time.sleep(0.5)  # respect rate limit

    if not all_trades:
        return pd.DataFrame()

    df = pd.DataFrame(all_trades, columns=["timestamp", "price", "volume", "side"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df[df["timestamp"] < pd.Timestamp(END_DT, tz="UTC")]
    return df

def trades_to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Resample tick trades to 1-minute OHLCV."""
    df = df.set_index("timestamp").sort_index()
    ohlcv = df["price"].resample("1min").ohlc()
    ohlcv["volume"]   = df["volume"].resample("1min").sum()
    ohlcv["n_trades"] = df["price"].resample("1min").count()
    # Signed volume (buy - sell)
    buy_vol  = df.loc[df["side"] == "b", "volume"].resample("1min").sum()
    sell_vol = df.loc[df["side"] == "s", "volume"].resample("1min").sum()
    ohlcv["buy_vol"]  = buy_vol.reindex(ohlcv.index, fill_value=0)
    ohlcv["sell_vol"] = sell_vol.reindex(ohlcv.index, fill_value=0)
    ohlcv["obi"] = (ohlcv["buy_vol"] - ohlcv["sell_vol"]) / (ohlcv["buy_vol"] + ohlcv["sell_vol"] + 1e-12)
    ohlcv = ohlcv.dropna(subset=["open"])
    return ohlcv

print("=" * 60)
print("FETCHING KRAKEN TRADES")
print("=" * 60)
for pair, name in KRAKEN_PAIRS.items():
    out_path = OUT / f"{name}_1min.parquet"
    if out_path.exists():
        print(f"  {name}: already exists, skipping")
        continue
    print(f"\nFetching {pair} ({name})...")
    df_trades = fetch_kraken_trades(pair, START_TS, END_TS)
    if df_trades.empty:
        print(f"  WARNING: no data for {pair}")
        continue
    df_ohlcv = trades_to_ohlcv(df_trades)
    df_ohlcv.to_parquet(out_path)
    print(f"  Saved {len(df_ohlcv)} rows to {out_path.name}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. BYBIT — public trade archive -> 1-min OHLC + microstructure
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FETCHING BYBIT TRADE ARCHIVE")
print("=" * 60)

BYBIT_PAIRS = {
    "BTCUSDT": "bybit_btcusdt",
    "BTCUSDC": "bybit_btcusdc",
}

def fetch_bybit_archive(symbol: str, year_month: str) -> pd.DataFrame:
    """Download and parse Bybit public trade archive for a given month."""
    url = f"https://public.bybit.com/spot/{symbol}/{symbol}-{year_month}.csv.gz"
    print(f"  Downloading {url}...")
    r = requests.get(url, timeout=120, stream=True)
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} for {url}")
        return pd.DataFrame()
    content = r.content
    with gzip.open(io.BytesIO(content)) as f:
        df = pd.read_csv(f)
    print(f"  Raw rows: {len(df):,}  columns: {list(df.columns)}")
    return df

for symbol, name in BYBIT_PAIRS.items():
    out_path = OUT / f"{name}_1min.parquet"
    if out_path.exists():
        print(f"\n{name}: already exists, skipping")
        continue

    print(f"\nFetching {symbol}...")
    df_raw = fetch_bybit_archive(symbol, "2023-03")
    if df_raw.empty:
        print(f"  WARNING: no data for {symbol}")
        continue

    # Parse timestamps
    # Bybit schema: id, timestamp (ms), price, volume, side
    df_raw.columns = [c.lower() for c in df_raw.columns]
    if "timestamp" in df_raw.columns:
        # timestamp in milliseconds
        df_raw["dt"] = pd.to_datetime(df_raw["timestamp"], unit="ms", utc=True)
    elif "time" in df_raw.columns:
        df_raw["dt"] = pd.to_datetime(df_raw["time"], unit="ms", utc=True)
    else:
        print(f"  Unknown timestamp column: {df_raw.columns.tolist()}")
        continue

    # Filter to March 1-21
    df_raw = df_raw[(df_raw["dt"] >= pd.Timestamp(START_DT, tz="UTC")) &
                    (df_raw["dt"] <  pd.Timestamp(END_DT,   tz="UTC"))]
    print(f"  Filtered to Mar 1-21: {len(df_raw):,} rows")

    # Resample to 1-min OHLCV
    df_raw = df_raw.set_index("dt").sort_index()
    ohlcv = df_raw["price"].resample("1min").ohlc()
    ohlcv["volume"]   = df_raw["volume"].resample("1min").sum()
    ohlcv["n_trades"] = df_raw["price"].resample("1min").count()

    # Signed volume
    buy_vol  = df_raw.loc[df_raw["side"].str.lower() == "buy",  "volume"].resample("1min").sum()
    sell_vol = df_raw.loc[df_raw["side"].str.lower() == "sell", "volume"].resample("1min").sum()
    ohlcv["buy_vol"]  = buy_vol.reindex(ohlcv.index, fill_value=0)
    ohlcv["sell_vol"] = sell_vol.reindex(ohlcv.index, fill_value=0)
    ohlcv["obi"] = (ohlcv["buy_vol"] - ohlcv["sell_vol"]) / (ohlcv["buy_vol"] + ohlcv["sell_vol"] + 1e-12)

    # Spread proxy: (High - Low) / Close
    ohlcv["spread_proxy"] = (ohlcv["high"] - ohlcv["low"]) / ohlcv["close"]
    ohlcv["log_return"]   = np.log(ohlcv["close"] / ohlcv["close"].shift(1))

    # Kyle Lambda: |Δprice| / |signed_volume| per minute
    ohlcv["price_chg"]     = ohlcv["close"].diff().abs()
    ohlcv["signed_vol_abs"] = (ohlcv["buy_vol"] - ohlcv["sell_vol"]).abs()
    ohlcv["kyle_lambda"]   = ohlcv["price_chg"] / (ohlcv["signed_vol_abs"] + 1e-8)

    ohlcv = ohlcv.dropna(subset=["open"])
    ohlcv.to_parquet(out_path)
    print(f"  Saved {len(ohlcv)} rows to {out_path.name}")

print("\nAll done.")
