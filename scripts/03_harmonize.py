"""
IAQF 2026 — Data Harmonization Script
Merges all raw 1-minute candle files into a single unified UTC-aligned panel.
Produces: iaqf_harmonized_1m.parquet
"""

import pandas as pd
import numpy as np
import os

RAW_DIR = "/home/ubuntu/iaqf_data/raw"
OUT_DIR = "/home/ubuntu/iaqf_data"

# ── Study window ──────────────────────────────────────────────────────────────
START = pd.Timestamp("2023-03-01 00:00:00", tz="UTC")
END   = pd.Timestamp("2023-03-21 23:59:00", tz="UTC")

# Create a complete 1-minute UTC index for the study window
full_index = pd.date_range(start=START, end=END, freq="1min", tz="UTC")
print(f"Full 1-minute index: {len(full_index)} rows ({START} to {END})")

# ── Helper: load a raw parquet and extract OHLCV ─────────────────────────────
def load_raw(fname: str, cols_map: dict) -> pd.DataFrame:
    """Load a raw parquet, rename columns, and set timestamp as index."""
    fpath = os.path.join(RAW_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  WARNING: {fname} not found, skipping.")
        return pd.DataFrame(index=full_index)
    df = pd.read_parquet(fpath)
    df = df.rename(columns=cols_map)
    df = df.set_index("timestamp_utc")
    # Keep only columns we need
    keep = [c for c in cols_map.values() if c in df.columns and c != "timestamp_utc"]
    df = df[keep]
    # Reindex to full_index, forward-fill gaps (carry-forward for prices)
    df = df.reindex(full_index)
    return df

# ── Load each series ──────────────────────────────────────────────────────────
print("\nLoading raw files...")

# Binance.US BTC/USDT
bnus_btcusdt = load_raw("binanceus_BTCUSDT_1m.parquet", {
    "timestamp_utc": "timestamp_utc",
    "open": "bnus_btcusdt_open", "high": "bnus_btcusdt_high",
    "low": "bnus_btcusdt_low", "close": "bnus_btcusdt_close",
    "volume": "bnus_btcusdt_vol", "num_trades": "bnus_btcusdt_trades",
    "quote_asset_volume": "bnus_btcusdt_quote_vol"
})
print(f"  Binance.US BTC/USDT: {bnus_btcusdt['bnus_btcusdt_close'].notna().sum()} non-null rows")

# Binance.US BTC/USDC
bnus_btcusdc = load_raw("binanceus_BTCUSDC_1m.parquet", {
    "timestamp_utc": "timestamp_utc",
    "open": "bnus_btcusdc_open", "high": "bnus_btcusdc_high",
    "low": "bnus_btcusdc_low", "close": "bnus_btcusdc_close",
    "volume": "bnus_btcusdc_vol", "num_trades": "bnus_btcusdc_trades",
    "quote_asset_volume": "bnus_btcusdc_quote_vol"
})
print(f"  Binance.US BTC/USDC: {bnus_btcusdc['bnus_btcusdc_close'].notna().sum()} non-null rows")

# Binance.US BTC/USD
bnus_btcusd = load_raw("binanceus_BTCUSD_1m.parquet", {
    "timestamp_utc": "timestamp_utc",
    "open": "bnus_btcusd_open", "high": "bnus_btcusd_high",
    "low": "bnus_btcusd_low", "close": "bnus_btcusd_close",
    "volume": "bnus_btcusd_vol", "num_trades": "bnus_btcusd_trades",
    "quote_asset_volume": "bnus_btcusd_quote_vol"
})
print(f"  Binance.US BTC/USD:  {bnus_btcusd['bnus_btcusd_close'].notna().sum()} non-null rows")

# Binance.US BTC/BUSD (USD proxy)
bnus_btcbusd = load_raw("binanceus_BTCBUSD_1m.parquet", {
    "timestamp_utc": "timestamp_utc",
    "open": "bnus_btcbusd_open", "high": "bnus_btcbusd_high",
    "low": "bnus_btcbusd_low", "close": "bnus_btcbusd_close",
    "volume": "bnus_btcbusd_vol", "num_trades": "bnus_btcbusd_trades",
    "quote_asset_volume": "bnus_btcbusd_quote_vol"
})
print(f"  Binance.US BTC/BUSD: {bnus_btcbusd['bnus_btcbusd_close'].notna().sum()} non-null rows")

# Coinbase BTC-USD
cb_btcusd = load_raw("coinbase_BTCUSD_1m.parquet", {
    "timestamp_utc": "timestamp_utc",
    "open": "cb_btcusd_open", "high": "cb_btcusd_high",
    "low": "cb_btcusd_low", "close": "cb_btcusd_close",
    "volume": "cb_btcusd_vol"
})
print(f"  Coinbase BTC-USD:    {cb_btcusd['cb_btcusd_close'].notna().sum()} non-null rows")

# Coinbase BTC-USDC
cb_btcusdc = load_raw("coinbase_BTCUSDC_1m.parquet", {
    "timestamp_utc": "timestamp_utc",
    "open": "cb_btcusdc_open", "high": "cb_btcusdc_high",
    "low": "cb_btcusdc_low", "close": "cb_btcusdc_close",
    "volume": "cb_btcusdc_vol"
})
print(f"  Coinbase BTC-USDC:   {cb_btcusdc['cb_btcusdc_close'].notna().sum()} non-null rows")

# Coinbase BTC-USDT
cb_btcusdt = load_raw("coinbase_BTCUSDT_1m.parquet", {
    "timestamp_utc": "timestamp_utc",
    "open": "cb_btcusdt_open", "high": "cb_btcusdt_high",
    "low": "cb_btcusdt_low", "close": "cb_btcusdt_close",
    "volume": "cb_btcusdt_vol"
})
print(f"  Coinbase BTC-USDT:   {cb_btcusdt['cb_btcusdt_close'].notna().sum()} non-null rows")

# Stablecoin FX
bnus_usdtusd = load_raw("binanceus_USDTUSD_1m.parquet", {
    "timestamp_utc": "timestamp_utc",
    "close": "usdt_usd_close", "high": "usdt_usd_high", "low": "usdt_usd_low",
    "volume": "usdt_usd_vol"
})
print(f"  USDT/USD:            {bnus_usdtusd['usdt_usd_close'].notna().sum()} non-null rows")

bnus_usdcusd = load_raw("binanceus_USDCUSD_1m.parquet", {
    "timestamp_utc": "timestamp_utc",
    "close": "usdc_usd_close", "high": "usdc_usd_high", "low": "usdc_usd_low",
    "volume": "usdc_usd_vol"
})
print(f"  USDC/USD:            {bnus_usdcusd['usdc_usd_close'].notna().sum()} non-null rows")

bnus_usdcusdt = load_raw("binanceus_USDCUSDT_1m.parquet", {
    "timestamp_utc": "timestamp_utc",
    "close": "usdc_usdt_close", "volume": "usdc_usdt_vol"
})
print(f"  USDC/USDT:           {bnus_usdcusdt['usdc_usdt_close'].notna().sum()} non-null rows")

bnus_busdusdt = load_raw("binanceus_BUSDUSDT_1m.parquet", {
    "timestamp_utc": "timestamp_utc",
    "close": "busd_usdt_close", "volume": "busd_usdt_vol"
})
print(f"  BUSD/USDT:           {bnus_busdusdt['busd_usdt_close'].notna().sum()} non-null rows")

# ── Merge all into one panel ──────────────────────────────────────────────────
print("\nMerging all series...")
panel = pd.concat([
    bnus_btcusdt, bnus_btcusdc, bnus_btcusd, bnus_btcbusd,
    cb_btcusd, cb_btcusdc, cb_btcusdt,
    bnus_usdtusd, bnus_usdcusd, bnus_usdcusdt, bnus_busdusdt
], axis=1)

panel.index.name = "timestamp_utc"
panel = panel.reset_index()

# ── Forward-fill price columns (carry last known price for gaps) ──────────────
price_cols = [c for c in panel.columns if c.endswith("_close") or c.endswith("_open")
              or c.endswith("_high") or c.endswith("_low")]
vol_cols   = [c for c in panel.columns if c.endswith("_vol") or c.endswith("_trades")]

# Forward-fill prices (max 5 minutes)
panel[price_cols] = panel[price_cols].ffill(limit=5)
# Fill volume/trades NaN with 0 (no trades in that minute)
panel[vol_cols] = panel[vol_cols].fillna(0)

# ── Data quality report ───────────────────────────────────────────────────────
print(f"\nPanel shape: {panel.shape}")
print(f"Date range: {panel['timestamp_utc'].min()} to {panel['timestamp_utc'].max()}")
print("\nNull counts per price column (after ffill):")
for col in sorted(price_cols):
    null_pct = panel[col].isna().mean() * 100
    print(f"  {col:40s}: {null_pct:.2f}% null")

# ── Save harmonized panel ─────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "iaqf_harmonized_1m.parquet")
panel.to_parquet(out_path, index=False)
print(f"\n✓ Saved harmonized panel: {out_path}")
print(f"  Rows: {len(panel):,}  |  Columns: {len(panel.columns)}")
print(f"  Columns: {list(panel.columns)}")
