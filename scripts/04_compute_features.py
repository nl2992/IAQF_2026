from pathlib import Path
"""
IAQF 2026 — Feature Engineering Script
Computes all required analytical variables from the harmonized panel:
  - Mid-prices (using high-low average as proxy for OHLC data)
  - Intrabar spread proxies (high - low)
  - Log LOP deviations (cross-currency basis)
  - Stablecoin FX deviations
  - Realized volatility (1-hour rolling window)
  - Volume and trade activity metrics
  - Event/regime labels (pre-crisis, crisis, recovery, post)
  - Daily aggregates for summary statistics

Produces: iaqf_features_1m.parquet  (1-minute panel with all features)
          iaqf_features_1h.parquet  (1-hour resampled panel)
          iaqf_daily_summary.parquet (daily aggregates)
"""

import pandas as pd
import numpy as np
import os

IN_PATH  = str(Path(__file__).parent.parent / "data" / "parquet" / "harmonized_raw_1min.parquet")
OUT_DIR  = str(Path(__file__).parent.parent / "data" / "parquet")

# ── Load harmonized panel ─────────────────────────────────────────────────────
print("Loading harmonized panel...")
df = pd.read_parquet(IN_PATH)
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
df = df.sort_values("timestamp_utc").reset_index(drop=True)
print(f"  Shape: {df.shape}  |  {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")

# ── Mid-price proxies ─────────────────────────────────────────────────────────
# For OHLC candle data without L2 order book, mid-price = (high + low) / 2
# This is a standard approximation used in microstructure literature
print("\nComputing mid-prices...")

df["mid_bnus_btcusdt"] = (df["bnus_btcusdt_high"] + df["bnus_btcusdt_low"]) / 2
df["mid_bnus_btcusdc"] = (df["bnus_btcusdc_high"] + df["bnus_btcusdc_low"]) / 2
df["mid_bnus_btcusd"]  = (df["bnus_btcusd_high"]  + df["bnus_btcusd_low"])  / 2
df["mid_bnus_btcbusd"] = (df["bnus_btcbusd_high"]  + df["bnus_btcbusd_low"]) / 2
df["mid_cb_btcusd"]    = (df["cb_btcusd_high"]    + df["cb_btcusd_low"])    / 2
df["mid_cb_btcusdc"]   = (df["cb_btcusdc_high"]   + df["cb_btcusdc_low"])   / 2
df["mid_cb_btcusdt"]   = (df["cb_btcusdt_high"]   + df["cb_btcusdt_low"])   / 2

# Stablecoin mid-prices
df["mid_usdt_usd"]  = (df["usdt_usd_high"]  + df["usdt_usd_low"])  / 2
df["mid_usdc_usd"]  = (df["usdc_usd_high"]  + df["usdc_usd_low"])  / 2

# ── Intrabar spread proxy (high - low) ────────────────────────────────────────
# Parkinson (1980) estimator: proxy for bid-ask spread from OHLC data
print("Computing spread proxies...")

df["spread_bnus_btcusdt"] = df["bnus_btcusdt_high"] - df["bnus_btcusdt_low"]
df["spread_bnus_btcusdc"] = df["bnus_btcusdc_high"] - df["bnus_btcusdc_low"]
df["spread_bnus_btcusd"]  = df["bnus_btcusd_high"]  - df["bnus_btcusd_low"]
df["spread_bnus_btcbusd"] = df["bnus_btcbusd_high"]  - df["bnus_btcbusd_low"]
df["spread_cb_btcusd"]    = df["cb_btcusd_high"]    - df["cb_btcusd_low"]
df["spread_cb_btcusdc"]   = df["cb_btcusdc_high"]   - df["cb_btcusdc_low"]
df["spread_cb_btcusdt"]   = df["cb_btcusdt_high"]   - df["cb_btcusdt_low"]

# Relative spread (spread / mid-price)
df["rel_spread_bnus_btcusdt"] = df["spread_bnus_btcusdt"] / df["mid_bnus_btcusdt"]
df["rel_spread_bnus_btcusdc"] = df["spread_bnus_btcusdc"] / df["mid_bnus_btcusdc"]
df["rel_spread_bnus_btcusd"]  = df["spread_bnus_btcusd"]  / df["mid_bnus_btcusd"]
df["rel_spread_bnus_btcbusd"] = df["spread_bnus_btcbusd"]  / df["mid_bnus_btcbusd"]
df["rel_spread_cb_btcusd"]    = df["spread_cb_btcusd"]    / df["mid_cb_btcusd"]
df["rel_spread_cb_btcusdc"]   = df["spread_cb_btcusdc"]   / df["mid_cb_btcusdc"]
df["rel_spread_cb_btcusdt"]   = df["spread_cb_btcusdt"]   / df["mid_cb_btcusdt"]

# ── Log LOP Deviations ────────────────────────────────────────────────────────
# Δ_USDT = log(BTC/USDT) - log(BTC/USD)
# Δ_USDC = log(BTC/USDC) - log(BTC/USD)
# These measure the cross-currency basis (Law of One Price deviation)
print("Computing log LOP deviations...")

# Binance.US: BTC/USDT vs BTC/USD (direct USD pair)
df["lop_bnus_usdt_vs_usd"]  = np.log(df["mid_bnus_btcusdt"]) - np.log(df["mid_bnus_btcusd"])
df["lop_bnus_usdc_vs_usd"]  = np.log(df["mid_bnus_btcusdc"]) - np.log(df["mid_bnus_btcusd"])
df["lop_bnus_usdt_vs_busd"] = np.log(df["mid_bnus_btcusdt"]) - np.log(df["mid_bnus_btcbusd"])
df["lop_bnus_usdc_vs_busd"] = np.log(df["mid_bnus_btcusdc"]) - np.log(df["mid_bnus_btcbusd"])

# Coinbase: BTC/USDC vs BTC/USD
df["lop_cb_usdc_vs_usd"]    = np.log(df["mid_cb_btcusdc"])   - np.log(df["mid_cb_btcusd"])
df["lop_cb_usdt_vs_usd"]    = np.log(df["mid_cb_btcusdt"])   - np.log(df["mid_cb_btcusd"])

# Cross-exchange: Binance.US vs Coinbase (BTC/USD)
df["lop_bnus_vs_cb_usd"]    = np.log(df["mid_bnus_btcusd"])  - np.log(df["mid_cb_btcusd"])
df["lop_bnus_vs_cb_usdc"]   = np.log(df["mid_bnus_btcusdc"]) - np.log(df["mid_cb_btcusdc"])
df["lop_bnus_vs_cb_usdt"]   = np.log(df["mid_bnus_btcusdt"]) - np.log(df["mid_cb_btcusdt"])

# Absolute LOP deviations
for col in [c for c in df.columns if c.startswith("lop_")]:
    df[f"abs_{col}"] = df[col].abs()

# ── Stablecoin FX Deviations ──────────────────────────────────────────────────
# log(USDT/USD) and log(USDC/USD) — deviation from parity
print("Computing stablecoin FX deviations...")

df["log_usdt_usd_dev"] = np.log(df["usdt_usd_close"])   # log(USDT/USD), ~0 at parity
df["log_usdc_usd_dev"] = np.log(df["usdc_usd_close"])   # log(USDC/USD), ~0 at parity
df["log_usdc_usdt"]    = np.log(df["usdc_usdt_close"])  # log(USDC/USDT)
df["log_busd_usdt"]    = np.log(df["busd_usdt_close"])  # log(BUSD/USDT)

# Stablecoin FX component of LOP deviation
# The BTC/USDT vs BTC/USD basis can be decomposed as:
# log(BTC/USDT) - log(BTC/USD) ≈ log(USD/USDT) = -log(USDT/USD)
# So the stablecoin FX leg is: -log(USDT/USD)
df["stablecoin_fx_usdt"] = -np.log(df["usdt_usd_close"])  # FX component of USDT basis
df["stablecoin_fx_usdc"] = -np.log(df["usdc_usd_close"])  # FX component of USDC basis

# Residual (non-FX) component of LOP deviation
df["lop_residual_bnus_usdt"] = df["lop_bnus_usdt_vs_usd"] - df["stablecoin_fx_usdt"]
df["lop_residual_bnus_usdc"] = df["lop_bnus_usdc_vs_usd"] - df["stablecoin_fx_usdc"]

# ── Realized Volatility ───────────────────────────────────────────────────────
# 1-hour rolling realized volatility (sum of squared log-returns)
# Using 60-minute rolling window
print("Computing realized volatility...")

for series, col in [
    ("bnus_btcusd",  "bnus_btcusd_close"),
    ("bnus_btcusdt", "bnus_btcusdt_close"),
    ("bnus_btcusdc", "bnus_btcusdc_close"),
    ("cb_btcusd",    "cb_btcusd_close"),
    ("cb_btcusdc",   "cb_btcusdc_close"),
]:
    log_ret = np.log(df[col] / df[col].shift(1))
    df[f"rv60_{series}"] = log_ret.pow(2).rolling(60, min_periods=10).sum()
    df[f"logret_{series}"] = log_ret

# ── Volume Metrics ────────────────────────────────────────────────────────────
print("Computing volume metrics...")

# Total BTC volume across all pairs (Binance.US)
df["total_vol_bnus_btc"] = (
    df["bnus_btcusdt_vol"] + df["bnus_btcusdc_vol"] +
    df["bnus_btcusd_vol"]  + df["bnus_btcbusd_vol"]
)
# Quote volume (USD-equivalent)
df["total_quote_vol_bnus"] = (
    df["bnus_btcusdt_quote_vol"] + df["bnus_btcusdc_quote_vol"] +
    df["bnus_btcusd_quote_vol"]  + df["bnus_btcbusd_quote_vol"]
)
# Total trades
df["total_trades_bnus"] = (
    df["bnus_btcusdt_trades"] + df["bnus_btcusdc_trades"] +
    df["bnus_btcusd_trades"]  + df["bnus_btcbusd_trades"]
)
# Volume share by quote currency (Binance.US)
df["vol_share_usdt_bnus"] = df["bnus_btcusdt_vol"] / df["total_vol_bnus_btc"].replace(0, np.nan)
df["vol_share_usdc_bnus"] = df["bnus_btcusdc_vol"] / df["total_vol_bnus_btc"].replace(0, np.nan)
df["vol_share_usd_bnus"]  = df["bnus_btcusd_vol"]  / df["total_vol_bnus_btc"].replace(0, np.nan)
df["vol_share_busd_bnus"] = df["bnus_btcbusd_vol"]  / df["total_vol_bnus_btc"].replace(0, np.nan)

# ── Event/Regime Labels ───────────────────────────────────────────────────────
# Based on the USDC de-peg event timeline:
# Pre-crisis:  Mar 1–9, 2023
# Crisis:      Mar 10–12, 2023 (SVB run → USDC reserve freeze → de-peg)
# Recovery:    Mar 13–15, 2023 (Fed/FDIC backstop → Circle resumes redemptions)
# Post:        Mar 16–21, 2023
print("Assigning event/regime labels...")

def assign_regime(ts):
    d = ts.date()
    import datetime
    if d < datetime.date(2023, 3, 10):
        return "pre_crisis"
    elif d <= datetime.date(2023, 3, 12):
        return "crisis"
    elif d <= datetime.date(2023, 3, 15):
        return "recovery"
    else:
        return "post"

df["regime"] = df["timestamp_utc"].apply(assign_regime)
df["regime_num"] = df["regime"].map({"pre_crisis": 0, "crisis": 1, "recovery": 2, "post": 3})

# Day-of-study counter (1 = Mar 1, 21 = Mar 21)
df["day_of_study"] = (df["timestamp_utc"].dt.date - pd.Timestamp("2023-03-01").date()).apply(lambda x: x.days + 1)

# Hour of day (UTC)
df["hour_utc"] = df["timestamp_utc"].dt.hour

# ── Save 1-minute feature panel ───────────────────────────────────────────────
out_1m = os.path.join(OUT_DIR, "iaqf_features_1m.parquet")
df.to_parquet(out_1m, index=False)
print(f"\n✓ Saved 1-minute feature panel: {out_1m}")
print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

# ── 1-hour resample ───────────────────────────────────────────────────────────
print("\nCreating 1-hour resampled panel...")
df_ts = df.set_index("timestamp_utc")

# For prices: use last value in each hour
price_cols = [c for c in df.columns if any(c.startswith(p) for p in
              ["mid_", "bnus_btc", "cb_btc", "usdt_usd", "usdc_usd", "usdc_usdt", "busd_usdt",
               "lop_", "abs_lop_", "log_usdt", "log_usdc", "stablecoin_fx", "lop_residual"])]
# For volumes: sum
vol_cols_1h = [c for c in df.columns if c.endswith("_vol") or c.endswith("_trades")
               or c.startswith("total_vol") or c.startswith("total_quote") or c.startswith("total_trades")]
# For spreads: mean
spread_cols = [c for c in df.columns if "spread" in c or c.startswith("rv60_")]

agg_dict = {}
for c in df_ts.columns:
    if c in vol_cols_1h:
        agg_dict[c] = "sum"
    elif c in spread_cols:
        agg_dict[c] = "mean"
    elif c == "regime":
        agg_dict[c] = "first"
    elif c in ["regime_num", "day_of_study", "hour_utc"]:
        agg_dict[c] = "first"
    else:
        agg_dict[c] = "last"

df_1h = df_ts.resample("1h").agg(agg_dict).reset_index()
out_1h = os.path.join(OUT_DIR, "iaqf_features_1h.parquet")
df_1h.to_parquet(out_1h, index=False)
print(f"✓ Saved 1-hour panel: {out_1h} ({len(df_1h)} rows)")

# ── Daily summary ─────────────────────────────────────────────────────────────
print("\nCreating daily summary...")
df_day = df_ts.resample("1D").agg({
    # BTC prices (close)
    "bnus_btcusd_close":  ["first", "last", "max", "min"],
    "bnus_btcusdt_close": ["first", "last", "max", "min"],
    "bnus_btcusdc_close": ["first", "last", "max", "min"],
    "cb_btcusd_close":    ["first", "last", "max", "min"],
    "cb_btcusdc_close":   ["first", "last", "max", "min"],
    # Stablecoin FX
    "usdt_usd_close":     ["mean", "min", "max"],
    "usdc_usd_close":     ["mean", "min", "max"],
    # LOP deviations
    "lop_bnus_usdt_vs_usd":  ["mean", "std", "min", "max"],
    "lop_bnus_usdc_vs_usd":  ["mean", "std", "min", "max"],
    "lop_cb_usdc_vs_usd":    ["mean", "std", "min", "max"],
    "abs_lop_bnus_usdt_vs_usd": ["mean", "max"],
    "abs_lop_bnus_usdc_vs_usd": ["mean", "max"],
    "abs_lop_cb_usdc_vs_usd":   ["mean", "max"],
    # Spreads
    "spread_bnus_btcusdt": ["mean"],
    "spread_bnus_btcusdc": ["mean"],
    "spread_bnus_btcusd":  ["mean"],
    "spread_cb_btcusd":    ["mean"],
    "spread_cb_btcusdc":   ["mean"],
    # Relative spreads
    "rel_spread_bnus_btcusdt": ["mean"],
    "rel_spread_bnus_btcusdc": ["mean"],
    "rel_spread_bnus_btcusd":  ["mean"],
    # Volumes
    "bnus_btcusdt_vol": ["sum"],
    "bnus_btcusdc_vol": ["sum"],
    "bnus_btcusd_vol":  ["sum"],
    "cb_btcusd_vol":    ["sum"],
    "cb_btcusdc_vol":   ["sum"],
    # Realized vol
    "rv60_bnus_btcusd":  ["mean"],
    "rv60_bnus_btcusdt": ["mean"],
    "rv60_bnus_btcusdc": ["mean"],
    # Regime
    "regime": ["first"],
    "day_of_study": ["first"],
}).reset_index()

# Flatten multi-level columns
df_day.columns = ["_".join(c).strip("_") if c[1] else c[0] for c in df_day.columns]
df_day = df_day.rename(columns={"timestamp_utc_": "date"})

out_day = os.path.join(OUT_DIR, "iaqf_daily_summary.parquet")
df_day.to_parquet(out_day, index=False)
print(f"✓ Saved daily summary: {out_day} ({len(df_day)} rows)")

# ── Print key statistics ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY STATISTICS BY REGIME")
print("=" * 70)

for regime in ["pre_crisis", "crisis", "recovery", "post"]:
    sub = df[df["regime"] == regime]
    print(f"\n[{regime.upper()}] n={len(sub):,} minutes")
    print(f"  BTC/USD (Binance.US) close: mean={sub['bnus_btcusd_close'].mean():.0f}, "
          f"min={sub['bnus_btcusd_close'].min():.0f}, max={sub['bnus_btcusd_close'].max():.0f}")
    print(f"  USDC/USD close: mean={sub['usdc_usd_close'].mean():.5f}, "
          f"min={sub['usdc_usd_close'].min():.5f}, max={sub['usdc_usd_close'].max():.5f}")
    print(f"  |LOP| BTC/USDT vs USD (Binance.US): mean={sub['abs_lop_bnus_usdt_vs_usd'].mean()*10000:.2f} bps, "
          f"max={sub['abs_lop_bnus_usdt_vs_usd'].max()*10000:.2f} bps")
    print(f"  |LOP| BTC/USDC vs USD (Binance.US): mean={sub['abs_lop_bnus_usdc_vs_usd'].mean()*10000:.2f} bps, "
          f"max={sub['abs_lop_bnus_usdc_vs_usd'].max()*10000:.2f} bps")
    print(f"  |LOP| BTC/USDC vs USD (Coinbase):   mean={sub['abs_lop_cb_usdc_vs_usd'].mean()*10000:.2f} bps, "
          f"max={sub['abs_lop_cb_usdc_vs_usd'].max()*10000:.2f} bps")
    print(f"  Rel spread BTC/USD (Binance.US): mean={sub['rel_spread_bnus_btcusd'].mean()*10000:.2f} bps")

print("\n✓ Feature engineering complete.")
