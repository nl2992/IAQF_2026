"""
compute_table_vii.py
Computes Table VII: Stablecoin Premia/Discounts by Regime and Venue
for IAQF 2026 Q2 analysis.

Outputs:
  - Prints the full table to stdout
  - Prints cross-venue correlations by regime
  - Prints USDC/USDT relative confidence stats
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
PARQ = BASE / "data" / "parquet"
CROSS = BASE / "data" / "cross_exchange"

REGIME_ORDER = ["pre_crisis", "crisis", "recovery", "post"]
REGIME_LABELS = {
    "pre_crisis": "Pre-Crisis (Mar 1–9)",
    "crisis":     "Crisis (Mar 10–12)",
    "recovery":   "Recovery (Mar 13–15)",
    "post":       "Post (Mar 16–21)",
}

# ── 1. Load Binance.US panel ──────────────────────────────────────────────────
print("Loading Binance.US panel...")
bnus = pd.read_parquet(PARQ / "panel_1min.parquet",
                       columns=["log_usdc_usd_dev", "log_usdt_usd_dev", "regime", "timestamp_utc"])
# Use timestamp_utc as the index
bnus = bnus.set_index("timestamp_utc")
bnus.index = bnus.index.tz_convert("UTC")
print(f"  Binance.US panel: {len(bnus):,} rows")
print(f"  Regimes: {bnus['regime'].value_counts().to_dict()}")

# ── 2. Load Kraken stablecoin FX data ────────────────────────────────────────
print("\nLoading Kraken stablecoin data...")

def load_kraken(fname, col_name):
    path = CROSS / fname
    if not path.exists():
        print(f"  WARNING: {fname} not found")
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    # Compute log deviation from par in bps
    # close price is the USDC/USD or USDT/USD rate
    close_col = [c for c in df.columns if "close" in c.lower()]
    if not close_col:
        print(f"  WARNING: no close column in {fname}, columns: {df.columns.tolist()}")
        return None
    price = df[close_col[0]]
    dev_bps = np.log(price) * 10000  # log(S) in bps, where S≈1 so log(S)≈S-1
    return dev_bps.rename(col_name)

kraken_usdc = load_kraken("kraken_usdcusd_1min.parquet", "kraken_usdc_usd_dev")
kraken_usdt = load_kraken("kraken_usdtusd_1min.parquet", "kraken_usdt_usd_dev")
kraken_usdcusdt = load_kraken("kraken_usdcusdt_1min.parquet", "kraken_usdcusdt_dev")

if kraken_usdc is not None:
    print(f"  Kraken USDC/USD: {len(kraken_usdc):,} rows, range {kraken_usdc.index.min()} to {kraken_usdc.index.max()}")
if kraken_usdt is not None:
    print(f"  Kraken USDT/USD: {len(kraken_usdt):,} rows, range {kraken_usdt.index.min()} to {kraken_usdt.index.max()}")
if kraken_usdcusdt is not None:
    print(f"  Kraken USDC/USDT: {len(kraken_usdcusdt):,} rows")

# ── 3. Assign regimes to Kraken data ─────────────────────────────────────────
# Use the same regime timestamps as Binance.US
# Regime boundaries (UTC):
#   pre_crisis:  Mar 1 00:00 – Mar 9 23:59
#   crisis:      Mar 10 00:00 – Mar 12 23:59
#   recovery:    Mar 13 00:00 – Mar 15 23:59
#   post:        Mar 16 00:00 – Mar 21 23:59

def assign_regime(idx):
    regimes = pd.Series(index=idx, dtype=str)
    regimes[(idx >= "2023-03-01") & (idx < "2023-03-10")] = "pre_crisis"
    regimes[(idx >= "2023-03-10") & (idx < "2023-03-13")] = "crisis"
    regimes[(idx >= "2023-03-13") & (idx < "2023-03-16")] = "recovery"
    regimes[(idx >= "2023-03-16") & (idx <= "2023-03-21 23:59")] = "post"
    return regimes

# ── 4. Compute Table VII ──────────────────────────────────────────────────────
def regime_stats(series, regime_col, label):
    """Compute P5, median, P95, min, max by regime."""
    rows = []
    for reg in REGIME_ORDER:
        mask = regime_col == reg
        s = series[mask].dropna()
        if len(s) == 0:
            rows.append({"venue_pair": label, "regime": reg, "N": 0,
                         "P5": np.nan, "Median": np.nan, "P95": np.nan, "Min": np.nan})
        else:
            rows.append({
                "venue_pair": label,
                "regime": reg,
                "N": len(s),
                "P5":    round(np.percentile(s, 5), 1),
                "Median": round(np.median(s), 1),
                "P95":   round(np.percentile(s, 95), 1),
                "Min":   round(s.min(), 1),
            })
    return rows

rows = []

# Binance.US USDC/USD
rows += regime_stats(bnus["log_usdc_usd_dev"], bnus["regime"], "Binance.US  USDC/USD")

# Binance.US USDT/USD
rows += regime_stats(bnus["log_usdt_usd_dev"], bnus["regime"], "Binance.US  USDT/USD")

# Kraken USDC/USD
if kraken_usdc is not None:
    kraken_regimes = assign_regime(kraken_usdc.index)
    rows += regime_stats(kraken_usdc, kraken_regimes, "Kraken      USDC/USD")

# Kraken USDT/USD
if kraken_usdt is not None:
    kraken_usdt_regimes = assign_regime(kraken_usdt.index)
    rows += regime_stats(kraken_usdt, kraken_usdt_regimes, "Kraken      USDT/USD")

# Kraken USDC/USDT (relative confidence)
if kraken_usdcusdt is not None:
    kraken_usdcusdt_regimes = assign_regime(kraken_usdcusdt.index)
    rows += regime_stats(kraken_usdcusdt, kraken_usdcusdt_regimes, "Kraken      USDC/USDT")

df_table = pd.DataFrame(rows)

# ── 5. Print Table VII ────────────────────────────────────────────────────────
print()
print("=" * 100)
print("TABLE VII: Stablecoin Premia/Discounts by Regime and Venue (basis points, log-deviation from par)")
print("=" * 100)
print(f"{'Venue / Pair':<28} {'Regime':<28} {'N':>6} {'P5':>8} {'Median':>8} {'P95':>8} {'Min':>8}")
print("-" * 100)

prev_pair = None
for _, row in df_table.iterrows():
    if row["venue_pair"] != prev_pair and prev_pair is not None:
        print()
    prev_pair = row["venue_pair"]
    print(f"{row['venue_pair']:<28} {REGIME_LABELS[row['regime']]:<28} {row['N']:>6,} "
          f"{row['P5']:>8.1f} {row['Median']:>8.1f} {row['P95']:>8.1f} {row['Min']:>8.1f}")

print("=" * 100)
print("Notes: All values in basis points. Log-deviation = ln(S_t) × 10,000 where S_t is the stablecoin/USD rate.")
print("       Negative values = stablecoin trading at a discount to USD par.")
print("       USDC/USDT = ln(S^USDC/USDT) × 10,000: negative = USDC at discount to USDT.")

# ── 6. Cross-venue correlations ───────────────────────────────────────────────
print()
print("=" * 80)
print("Cross-Venue Correlations: Binance.US vs Kraken USDC/USD Deviation by Regime")
print("=" * 80)

if kraken_usdc is not None:
    # Both indices are already tz-aware UTC
    bnus_usdc = bnus["log_usdc_usd_dev"].copy()
    kraken_usdc_utc = kraken_usdc.copy()
    kraken_usdc_utc.index = kraken_usdc_utc.index.tz_convert("UTC")

    merged = pd.merge(
        bnus_usdc.rename("bnus"),
        kraken_usdc_utc.rename("kraken"),
        left_index=True, right_index=True,
        how="inner"
    )
    # Assign regime from timestamp
    merged["regime"] = assign_regime(merged.index)

    print(f"{'Regime':<28} {'N':>6} {'Corr(BinUS, Kraken)':>22} {'BinUS Var (bps²)':>18} {'Kraken Var (bps²)':>18}")
    print("-" * 80)
    for reg in REGIME_ORDER:
        sub = merged[merged["regime"] == reg].dropna()
        if len(sub) < 5:
            print(f"  {REGIME_LABELS[reg]:<26} {'N/A':>6}")
            continue
        corr = sub["bnus"].corr(sub["kraken"])
        var_bnus = sub["bnus"].var()
        var_kraken = sub["kraken"].var()
        print(f"  {REGIME_LABELS[reg]:<26} {len(sub):>6,} {corr:>22.4f} {var_bnus:>18.1f} {var_kraken:>18.1f}")
    print("=" * 80)

# ── 7. USDC vs USDT relative confidence ──────────────────────────────────────
print()
print("=" * 80)
print("USDC vs USDT Relative Confidence (Binance.US): USDC dev minus USDT dev by regime")
print("=" * 80)
bnus["relative_conf"] = bnus["log_usdc_usd_dev"] - bnus["log_usdt_usd_dev"]
print(f"{'Regime':<28} {'N':>6} {'Median (bps)':>14} {'P5 (bps)':>10} {'P95 (bps)':>10} {'Min (bps)':>10}")
print("-" * 80)
for reg in REGIME_ORDER:
    s = bnus.loc[bnus["regime"] == reg, "relative_conf"].dropna()
    if len(s) == 0:
        continue
    print(f"  {REGIME_LABELS[reg]:<26} {len(s):>6,} {np.median(s):>14.1f} "
          f"{np.percentile(s,5):>10.1f} {np.percentile(s,95):>10.1f} {s.min():>10.1f}")
print("=" * 80)
print("Interpretation: Negative = USDC at larger discount than USDT (USDC more stressed).")
print("                Near-zero = both stablecoins equally stressed.")

print("\nDone.")
