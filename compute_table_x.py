"""
compute_table_x.py
Computes Table X: Spread, Depth-Proxy, and Volatility by Regime and Quote Currency
for IAQF 2026 Q3 analysis.

Data source: data/parquet/panel_1min.parquet (already on disk)

Spread columns available:
  spread_bnus_btcusdt  — absolute spread (close - open proxy) in price units
  spread_bnus_btcusdc  — same for BTC/USDC
  spread_bnus_btcusd   — same for BTC/USD

Relative spread columns:
  rel_spread_bnus_btcusdt  — spread / mid (dimensionless)
  rel_spread_bnus_btcusdc
  rel_spread_bnus_btcusd

Volatility:
  rv60_bnus_btcusd  — 60-min rolling realized variance (Parkinson) on BTC/USD

Volume / depth proxy:
  bnus_btcusdt_volume, bnus_btcusdc_volume — quote volume per minute
  vol_share_usdt_bnus, vol_share_usdc_bnus — share of total BTC volume

Outputs:
  - Table X  (spread / depth / vol by pair × regime)
  - Table X-B (stress ratios: crisis / pre-crisis)
  - Figure X  (time series of spread and vol with regime shading)
  - table_x_spread_depth_vol.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE  = Path(__file__).parent
PARQ  = BASE / "data" / "parquet"
FIGS  = BASE / "figures" / "lop"
FIGS.mkdir(parents=True, exist_ok=True)

# ── Columbia colour palette ───────────────────────────────────────────────────
COLUMBIA_BLUE  = '#003087'
COLUMBIA_GOLD  = '#C4A44B'
COLUMBIA_GREY  = '#6B6B6B'
CRISIS_RED     = '#C8102E'
RECOVERY_AMBER = '#E87722'
POST_GREEN     = '#2E7D32'

REGIME_COLORS = {
    'pre_crisis': COLUMBIA_BLUE,
    'crisis':     CRISIS_RED,
    'recovery':   RECOVERY_AMBER,
    'post':       POST_GREEN,
}
REGIME_LABELS = {
    'pre_crisis': 'Pre-Crisis (Mar 1–9)',
    'crisis':     'Crisis (Mar 10–12)',
    'recovery':   'Recovery (Mar 13–15)',
    'post':       'Post (Mar 16–21)',
}
REGIME_ORDER = ['pre_crisis', 'crisis', 'recovery', 'post']

REGIME_SPANS = [
    ('pre_crisis', '2023-03-01', '2023-03-09 23:59'),
    ('crisis',     '2023-03-10', '2023-03-12 23:59'),
    ('recovery',   '2023-03-13', '2023-03-15 23:59'),
    ('post',       '2023-03-16', '2023-03-21 23:59'),
]

# ── Load panel ────────────────────────────────────────────────────────────────
print("Loading panel_1min.parquet...")
panel = pd.read_parquet(PARQ / "panel_1min.parquet")
panel = panel.set_index("timestamp_utc")
panel.index = pd.to_datetime(panel.index, utc=True)
print(f"  Shape: {panel.shape}")
print(f"  Date range: {panel.index[0]} → {panel.index[-1]}")

# ── Convert relative spread to bps ────────────────────────────────────────────
# rel_spread = (high - low) / mid  (Parkinson-like), already dimensionless
# Multiply by 10,000 to get bps
for pair in ['btcusdt', 'btcusdc', 'btcusd']:
    col = f'rel_spread_bnus_{pair}'
    if col in panel.columns:
        panel[f'spread_bps_{pair}'] = panel[col] * 10_000
    else:
        print(f"  WARNING: {col} not found")

# ── Depth proxy: use volume per minute as depth proxy ─────────────────────────
# Higher volume = deeper book. We report median and P5 (thin-book indicator).
# Also compute volume share as a fraction indicator.
for pair in ['btcusdt', 'btcusdc', 'btcusd']:
    vol_col = f'bnus_{pair}_volume'
    if vol_col in panel.columns:
        panel[f'vol_{pair}'] = panel[vol_col]
    else:
        print(f"  WARNING: {vol_col} not found")

# ── Volatility: rv60_bnus_btcusd (60-min Parkinson RV) ───────────────────────
# Already in panel. Convert to bps² or keep as-is (it's in log² units).
# For display, take sqrt to get vol in log-return units, then ×10000 for bps.
if 'rv60_bnus_btcusd' in panel.columns:
    panel['vol_bps'] = np.sqrt(panel['rv60_bnus_btcusd'].clip(lower=0)) * 10_000
else:
    print("  WARNING: rv60_bnus_btcusd not found")
    panel['vol_bps'] = np.nan

# ── BTC/USDC only available from Mar 12 ──────────────────────────────────────
# Mask pre-listing rows
usdc_start = pd.Timestamp('2023-03-12', tz='UTC')
panel.loc[panel.index < usdc_start, 'spread_bps_btcusdc'] = np.nan
panel.loc[panel.index < usdc_start, 'vol_btcusdc'] = np.nan

# ── TABLE X: Spread, Depth, Volatility by Regime and Pair ────────────────────
print()
print("=" * 110)
print("TABLE X: Spread, Depth-Proxy, and Volatility by Regime and Quote Currency (Binance.US)")
print("=" * 110)
print(f"{'Pair':<12} {'Regime':<26} {'N':>6}  "
      f"{'Spread P50 (bps)':>18} {'Spread P95 (bps)':>18}  "
      f"{'Vol P50 (BTC/min)':>20} {'Vol P5 (BTC/min)':>18}  "
      f"{'RV Vol P50 (bps)':>18} {'RV Vol P95 (bps)':>18}")
print("-" * 110)

rows_out = []
pairs = [
    ('BTC/USDT', 'btcusdt'),
    ('BTC/USDC', 'btcusdc'),
    ('BTC/USD',  'btcusd'),
]

prev_pair = None
for pair_label, pair_key in pairs:
    sp_col  = f'spread_bps_{pair_key}'
    vol_col = f'vol_{pair_key}'

    for reg in REGIME_ORDER:
        sub = panel[panel['regime'] == reg]
        sp  = sub[sp_col].dropna()   if sp_col  in panel.columns else pd.Series(dtype=float)
        vol = sub[vol_col].dropna()  if vol_col in panel.columns else pd.Series(dtype=float)
        rv  = sub['vol_bps'].dropna()

        n = len(sp) if len(sp) > 0 else len(vol)

        sp_p50  = np.median(sp)  if len(sp)  > 0 else np.nan
        sp_p95  = np.percentile(sp, 95) if len(sp) > 0 else np.nan
        vol_p50 = np.median(vol) if len(vol) > 0 else np.nan
        vol_p5  = np.percentile(vol, 5) if len(vol) > 0 else np.nan
        rv_p50  = np.median(rv)  if len(rv)  > 0 else np.nan
        rv_p95  = np.percentile(rv, 95) if len(rv) > 0 else np.nan

        if pair_label != prev_pair and prev_pair is not None:
            print()
        prev_pair = pair_label

        def fmt(v, decimals=2):
            return f"{v:.{decimals}f}" if not np.isnan(v) else "—"

        print(f"{pair_label:<12} {REGIME_LABELS[reg]:<26} {n:>6,}  "
              f"{fmt(sp_p50):>18} {fmt(sp_p95):>18}  "
              f"{fmt(vol_p50, 1):>20} {fmt(vol_p5, 1):>18}  "
              f"{fmt(rv_p50):>18} {fmt(rv_p95):>18}")

        rows_out.append({
            'pair': pair_label, 'regime': reg,
            'N': n,
            'spread_p50_bps': round(sp_p50, 2) if not np.isnan(sp_p50) else None,
            'spread_p95_bps': round(sp_p95, 2) if not np.isnan(sp_p95) else None,
            'vol_p50_btc': round(vol_p50, 1) if not np.isnan(vol_p50) else None,
            'vol_p5_btc':  round(vol_p5, 1)  if not np.isnan(vol_p5)  else None,
            'rv_vol_p50_bps': round(rv_p50, 2) if not np.isnan(rv_p50) else None,
            'rv_vol_p95_bps': round(rv_p95, 2) if not np.isnan(rv_p95) else None,
        })

print("=" * 110)
print("Notes:")
print("  Spread = Parkinson high-low estimator × 10,000 (bps). Depth proxy = BTC volume per 1-min bar.")
print("  P5 of volume = thin-book indicator (lower = thinner). RV Vol = sqrt(60-min Parkinson RV) × 10,000.")
print("  BTC/USDC listed on Binance.US on Mar 12; pre-listing rows excluded (shown as —).")

# ── Save CSV ──────────────────────────────────────────────────────────────────
df_out = pd.DataFrame(rows_out)
df_out.to_csv(BASE / "table_x_spread_depth_vol.csv", index=False)
print(f"\nSaved: table_x_spread_depth_vol.csv")

# ── TABLE X-B: Stress Ratios ──────────────────────────────────────────────────
print()
print("=" * 80)
print("TABLE X-B: Stress Ratios (Crisis / Pre-Crisis)")
print("=" * 80)
print(f"{'Pair':<12} {'Metric':<30} {'Pre-Crisis':>14} {'Crisis':>14} {'Ratio':>10}")
print("-" * 80)

for pair_label, pair_key in pairs:
    sp_col  = f'spread_bps_{pair_key}'
    vol_col = f'vol_{pair_key}'

    pre = panel[panel['regime'] == 'pre_crisis']
    cri = panel[panel['regime'] == 'crisis']

    metrics = [
        ('Spread P50 (bps)',    sp_col,    50),
        ('Spread P95 (bps)',    sp_col,    95),
        ('Volume P50 (BTC/min)', vol_col,  50),
        ('Volume P5 (BTC/min)', vol_col,    5),
    ]

    printed_pair = False
    for metric_label, col, pct in metrics:
        pre_s = pre[col].dropna() if col in panel.columns else pd.Series(dtype=float)
        cri_s = cri[col].dropna() if col in panel.columns else pd.Series(dtype=float)
        if len(pre_s) == 0 or len(cri_s) == 0:
            continue
        pre_v = np.percentile(pre_s, pct)
        cri_v = np.percentile(cri_s, pct)
        ratio = cri_v / pre_v if abs(pre_v) > 1e-8 else np.nan
        pair_str = pair_label if not printed_pair else ''
        printed_pair = True
        print(f"  {pair_str:<10} {metric_label:<30} {pre_v:>14.2f} {cri_v:>14.2f} {ratio:>10.2f}×")
    print()

print("=" * 80)
print("Interpretation: Ratio > 1 for spread = wider (worse). Ratio < 1 for volume = thinner (worse).")

# ── FIGURE X: Spread and Volatility Time Series ───────────────────────────────
print("\nGenerating Figure X...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                         gridspec_kw={'height_ratios': [2, 2, 1.5]})
fig.suptitle('Figure X: Binance.US Microstructure by Quote Currency — Spread, Volume, and Volatility',
             fontsize=13, fontweight='bold', y=0.98)

# Regime shading helper
def shade_regimes(ax):
    for regime, start, end in REGIME_SPANS:
        ax.axvspan(pd.Timestamp(start, tz='UTC'), pd.Timestamp(end, tz='UTC'),
                   alpha=0.08, color=REGIME_COLORS[regime], zorder=0)

# Panel 1: Spread (bps)
ax = axes[0]
shade_regimes(ax)
if 'spread_bps_btcusdt' in panel.columns:
    ax.plot(panel.index, panel['spread_bps_btcusdt'].rolling(60).median(),
            color=COLUMBIA_BLUE, lw=1.2, label='BTC/USDT spread (60-min median)')
if 'spread_bps_btcusd' in panel.columns:
    ax.plot(panel.index, panel['spread_bps_btcusd'].rolling(60).median(),
            color=COLUMBIA_GOLD, lw=1.2, label='BTC/USD spread (60-min median)', alpha=0.85)
if 'spread_bps_btcusdc' in panel.columns:
    ax.plot(panel.index, panel['spread_bps_btcusdc'].rolling(60).median(),
            color=CRISIS_RED, lw=1.2, label='BTC/USDC spread (60-min median, from Mar 12)', alpha=0.85)
ax.set_ylabel('Spread (bps)', fontsize=10)
ax.legend(loc='upper left', fontsize=8)
ax.set_ylim(bottom=0)
ax.set_title('Parkinson High-Low Spread Estimator', fontsize=10)

# Panel 2: Volume (BTC/min)
ax = axes[1]
shade_regimes(ax)
if 'vol_btcusdt' in panel.columns:
    ax.plot(panel.index, panel['vol_btcusdt'].rolling(60).median(),
            color=COLUMBIA_BLUE, lw=1.2, label='BTC/USDT volume (60-min median)')
if 'vol_btcusd' in panel.columns:
    ax.plot(panel.index, panel['vol_btcusd'].rolling(60).median(),
            color=COLUMBIA_GOLD, lw=1.2, label='BTC/USD volume (60-min median)', alpha=0.85)
if 'vol_btcusdc' in panel.columns:
    ax.plot(panel.index, panel['vol_btcusdc'].rolling(60).median(),
            color=CRISIS_RED, lw=1.2, label='BTC/USDC volume (60-min median, from Mar 12)', alpha=0.85)
ax.set_ylabel('Volume (BTC/min)', fontsize=10)
ax.legend(loc='upper left', fontsize=8)
ax.set_title('Volume per Minute (Depth Proxy)', fontsize=10)

# Panel 3: Realized Volatility
ax = axes[2]
shade_regimes(ax)
if 'vol_bps' in panel.columns:
    ax.plot(panel.index, panel['vol_bps'].rolling(60).median(),
            color=COLUMBIA_GREY, lw=1.2, label='RV Vol BTC/USD (60-min median)')
ax.set_ylabel('RV Vol (bps)', fontsize=10)
ax.set_xlabel('Date (UTC)', fontsize=10)
ax.legend(loc='upper left', fontsize=8)
ax.set_title('60-min Parkinson Realized Volatility (BTC/USD)', fontsize=10)

# Regime legend
patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.3, label=REGIME_LABELS[r])
           for r in REGIME_ORDER]
axes[0].legend(handles=axes[0].get_lines() + patches, loc='upper left', fontsize=7, ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out_path = FIGS / "fig_x_spread_depth_vol.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")
print("\nDone.")
