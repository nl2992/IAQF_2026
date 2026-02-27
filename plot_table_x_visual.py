"""
plot_table_x_visual.py
Generates a clean publication-quality visual for Table X and Table X-B.
Uses the already-computed numbers — no parquet re-read needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

OUT = Path('/home/ubuntu/IAQF_2026/figures/lop')
OUT.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = '#003087'   # Columbia Blue (pre-crisis)
C_RED    = '#C8102E'   # Crisis
C_AMBER  = '#E87722'   # Recovery
C_GREEN  = '#2E7D32'   # Post
C_GOLD   = '#C4A44B'
C_GREY   = '#6B6B6B'
C_LIGHT  = '#B9D9EB'

REGIME_COLORS = [C_BLUE, C_RED, C_AMBER, C_GREEN]
REGIME_LABELS = ['Pre-Crisis\n(Mar 1–9)', 'Crisis\n(Mar 10–12)',
                 'Recovery\n(Mar 13–15)', 'Post\n(Mar 16–21)']

# ── Table X data (from compute_table_x.py output) ────────────────────────────
# Pairs: BTC/USDT, BTC/USD  (USDC excluded — too illiquid, mostly zeros)
pairs       = ['BTC/USDT', 'BTC/USD']
pair_colors = [C_BLUE, C_GOLD]

# Spread P50 (bps)
spread_p50 = {
    'BTC/USDT': [3.67,  7.51,  12.41,  9.23],
    'BTC/USD':  [4.62,  9.31,  13.72, 11.31],
}
# Spread P95 (bps)
spread_p95 = {
    'BTC/USDT': [14.80, 26.15, 38.26, 26.31],
    'BTC/USD':  [16.12, 28.30, 40.43, 27.78],
}
# Volume P50 (BTC/min)
vol_p50 = {
    'BTC/USDT': [0.56, 1.33, 3.20, 0.70],
    'BTC/USD':  [1.77, 3.88, 6.50, 5.40],
}
# RV Volatility P50 (bps)
rv_p50 = [538.49, 732.03, 882.63, 796.93]   # same for both (BTC/USD base)
rv_p95 = [887.75, 1119.71, 1457.14, 1052.12]

# ── Table X-B stress ratios ───────────────────────────────────────────────────
ratio_labels  = ['Spread P50', 'Spread P95', 'Vol P50', 'Vol P5']
ratios_usdt   = [2.05, 1.77, 2.39, 18.83]
ratios_usd    = [2.01, 1.76, 2.19,  4.52]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor('white')

gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.52, wspace=0.38,
                       left=0.07, right=0.97, top=0.90, bottom=0.09)

ax_sp50  = fig.add_subplot(gs[0, 0])   # Spread P50
ax_sp95  = fig.add_subplot(gs[0, 1])   # Spread P95
ax_vol   = fig.add_subplot(gs[0, 2])   # Volume P50
ax_rv    = fig.add_subplot(gs[1, 0])   # RV Volatility
ax_ratio = fig.add_subplot(gs[1, 1:])  # Stress ratios (wide)

x = np.arange(4)
w = 0.35

def style_ax(ax, title, ylabel, ylim=None):
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIME_LABELS, fontsize=7.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)
    if ylim:
        ax.set_ylim(ylim)
    # Shade crisis bar background
    ax.axvspan(0.5, 1.5, color=C_RED, alpha=0.05, zorder=0)

# ── Panel 1: Spread P50 ───────────────────────────────────────────────────────
for i, (pair, col) in enumerate(zip(pairs, pair_colors)):
    bars = ax_sp50.bar(x + (i - 0.5) * w, spread_p50[pair], w,
                       color=col, alpha=0.85, label=pair, zorder=3)
    for bar, val in zip(bars, spread_p50[pair]):
        ax_sp50.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=7, color=col)
style_ax(ax_sp50, 'Spread — Median (bps)', 'bps')
ax_sp50.legend(fontsize=7.5, loc='upper left', framealpha=0.7)

# ── Panel 2: Spread P95 ───────────────────────────────────────────────────────
for i, (pair, col) in enumerate(zip(pairs, pair_colors)):
    bars = ax_sp95.bar(x + (i - 0.5) * w, spread_p95[pair], w,
                       color=col, alpha=0.85, label=pair, zorder=3)
    for bar, val in zip(bars, spread_p95[pair]):
        ax_sp95.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=7, color=col)
style_ax(ax_sp95, 'Spread — 95th Percentile (bps)', 'bps')

# ── Panel 3: Volume P50 ───────────────────────────────────────────────────────
for i, (pair, col) in enumerate(zip(pairs, pair_colors)):
    bars = ax_vol.bar(x + (i - 0.5) * w, vol_p50[pair], w,
                      color=col, alpha=0.85, label=pair, zorder=3)
    for bar, val in zip(bars, vol_p50[pair]):
        ax_vol.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=7, color=col)
style_ax(ax_vol, 'Volume — Median (BTC/min)', 'BTC/min')

# ── Panel 4: RV Volatility ────────────────────────────────────────────────────
bars_rv = ax_rv.bar(x, rv_p50, 0.5, color=C_GREY, alpha=0.75, label='P50', zorder=3)
# Error bars to P95
ax_rv.errorbar(x, rv_p50,
               yerr=[np.zeros(4), np.array(rv_p95) - np.array(rv_p50)],
               fmt='none', color=C_GREY, capsize=5, lw=1.5, zorder=4)
for bar, val in zip(bars_rv, rv_p50):
    ax_rv.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
               f'{val:.0f}', ha='center', va='bottom', fontsize=7.5, color=C_GREY)
style_ax(ax_rv, 'Realized Volatility — BTC/USD (bps)', 'bps')
ax_rv.text(0.98, 0.97, 'Error bars → P95', transform=ax_rv.transAxes,
           fontsize=7, ha='right', va='top', color=C_GREY)

# ── Panel 5: Stress Ratios (Table X-B) ───────────────────────────────────────
xr = np.arange(len(ratio_labels))
wr = 0.32

b1 = ax_ratio.bar(xr - wr/2, ratios_usdt, wr, color=C_BLUE, alpha=0.85,
                  label='BTC/USDT', zorder=3)
b2 = ax_ratio.bar(xr + wr/2, ratios_usd,  wr, color=C_GOLD, alpha=0.85,
                  label='BTC/USD',  zorder=3)

for bar, val in zip(b1, ratios_usdt):
    ax_ratio.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                  f'{val:.2f}×', ha='center', va='bottom', fontsize=8,
                  fontweight='bold', color=C_BLUE)
for bar, val in zip(b2, ratios_usd):
    ax_ratio.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                  f'{val:.2f}×', ha='center', va='bottom', fontsize=8,
                  fontweight='bold', color=C_GOLD)

ax_ratio.axhline(1.0, color='black', lw=1.0, linestyle='--', alpha=0.5, zorder=2)
ax_ratio.set_xticks(xr)
ax_ratio.set_xticklabels(ratio_labels, fontsize=9)
ax_ratio.set_ylabel('Crisis / Pre-Crisis Ratio', fontsize=9)
ax_ratio.set_title('Table X-B: Stress Ratios — Crisis vs. Pre-Crisis', fontsize=10, fontweight='bold', pad=6)
ax_ratio.spines['top'].set_visible(False)
ax_ratio.spines['right'].set_visible(False)
ax_ratio.tick_params(axis='y', labelsize=8)
ax_ratio.legend(fontsize=8.5, loc='upper left', framealpha=0.7)
ax_ratio.text(0.98, 0.97,
              'Ratio > 1 for spread = wider (worse)\nRatio > 1 for volume = deeper (better)',
              transform=ax_ratio.transAxes, fontsize=7.5, ha='right', va='top',
              color=C_GREY, style='italic')

# ── Main title ────────────────────────────────────────────────────────────────
fig.suptitle(
    'Table X: Binance.US Microstructure by Regime and Quote Currency\n'
    'Spread (Parkinson bps), Volume Depth Proxy (BTC/min), and Realized Volatility',
    fontsize=12, fontweight='bold', y=0.97
)

out = OUT / 'table_x_visual.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')
