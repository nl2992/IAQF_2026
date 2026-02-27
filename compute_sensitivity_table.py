"""
Sensitivity Table: GENIUS Act Redemption Credibility — Columbia theme
Professional redesign: clean horizontal bar charts with value annotations,
consistent with the IEEE / academic figure style used in the project.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from pathlib import Path

# ── Columbia palette ──────────────────────────────────────────────────────────
CU_BLUE     = '#003087'
CU_MID      = '#1A5276'
CU_CROWN    = '#2E86C1'
CU_LIGHT    = '#85C1E9'
CU_GOLD     = '#C4A44A'
CU_CRIMSON  = '#C4122F'
CU_GREY_BG  = '#F5F7FA'
CU_GRID     = '#D5D8DC'
CU_TEXT     = '#1A1A2E'

# Scenario colours: dark→light blue for hypothetical, gold for near-actual, red for actual
COLORS = [CU_BLUE, CU_MID, CU_CROWN, CU_LIGHT, CU_GOLD, CU_CRIMSON]

# ── Parameters ────────────────────────────────────────────────────────────────
SIGMA_INNOV     = 15.0
COST_HURDLE_BPS = 20.0
DELTA_T         = 1.0

HL_MIN   = [60, 120, 240, 360, 480, 602.7]
LABELS   = ['1 hr\n(GENIUS full)', '2 hr', '4 hr', '6 hr', '8 hr', '10 hr\n(actual crisis)']
LABELS_T = ['1 hr (GENIUS full)', '2 hr', '4 hr', '6 hr', '8 hr', '10 hr (actual crisis)']

# ── Analytics ─────────────────────────────────────────────────────────────────
def kappa(hl):       return np.log(2) / hl
def sigma_ss(k):
    phi = np.exp(-k * DELTA_T)
    return SIGMA_INNOV / np.sqrt(1 - phi**2)
def eht(hl):         return 1.44 * hl
def hit_rate(k):
    return 2 * (1 - norm.cdf(COST_HURDLE_BPS / sigma_ss(k))) * 100
def mae99(k, hl):
    ss = sigma_ss(k); T = max(1, eht(hl))
    u  = np.sqrt(2 * np.log(T)); b = 1.0 / u
    return ss * (u + b * np.log(-np.log(0.01)))

rows = []
for hl, lab in zip(HL_MIN, LABELS_T):
    k = kappa(hl)
    rows.append({'Scenario': lab, 'HL_hrs': hl/60, 'kappa': k,
                 'sigma_ss': sigma_ss(k), 'eht_hrs': eht(hl)/60,
                 'hit_rate': hit_rate(k), 'mae': mae99(k, hl)})
df = pd.DataFrame(rows)
df.to_csv('/home/ubuntu/IAQF_2026/table_sensitivity_genius.csv', index=False)

eht_v  = df['eht_hrs'].values
hr_v   = df['hit_rate'].values
mae_v  = df['mae'].values

# ── Figure layout ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Georgia', 'Times New Roman', 'DejaVu Serif'],
    'axes.titlesize':    11,
    'axes.labelsize':    9,
    'xtick.labelsize':   8.5,
    'ytick.labelsize':   8.5,
    'figure.facecolor':  CU_GREY_BG,
    'axes.facecolor':    'white',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.spines.left':  True,
    'axes.spines.bottom':True,
    'axes.grid':         True,
    'grid.color':        CU_GRID,
    'grid.linewidth':    0.6,
    'grid.alpha':        0.7,
})

fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor(CU_GREY_BG)

# ── Columbia header ───────────────────────────────────────────────────────────
hdr = fig.add_axes([0.0, 0.945, 1.0, 0.055])
hdr.set_facecolor(CU_BLUE); hdr.axis('off')
hdr.set_xlim(0,1); hdr.set_ylim(0,1)
hdr.text(0.5, 0.72, 'IAQF 2026  ·  Columbia MAFN',
         ha='center', va='center', fontsize=8.5, color='#B9D9EB',
         style='italic', fontfamily='serif')
hdr.text(0.5, 0.25,
         'Fig. R-1  —  GENIUS Act "What-If": Effect of Redemption Credibility on Arbitrage Dynamics',
         ha='center', va='center', fontsize=12.5, color='white',
         fontweight='bold', fontfamily='serif')

# ── Three horizontal-bar panels (top row) ─────────────────────────────────────
gs = GridSpec(1, 3, figure=fig,
              top=0.91, bottom=0.44,
              hspace=0.0, wspace=0.42,
              left=0.06, right=0.97)

y     = np.arange(len(LABELS))
h_bar = 0.55   # bar height

def style_hbar_ax(ax, title, xlabel, x_max, vline=None, vline_label=None,
                  vline_color=CU_GOLD, x_fmt='{:.0f}'):
    ax.set_title(title, fontweight='bold', color=CU_BLUE, pad=8,
                 fontsize=11, loc='left', fontfamily='serif')
    ax.set_xlabel(xlabel, color=CU_TEXT, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(LABELS, fontsize=8.5)
    ax.set_xlim(0, x_max)
    ax.tick_params(axis='y', length=0)
    ax.spines['left'].set_color(CU_GRID)
    ax.spines['bottom'].set_color(CU_GRID)
    ax.invert_yaxis()   # top = 1 hr (best), bottom = 10 hr (worst)
    if vline is not None:
        ax.axvline(vline, color=vline_color, linestyle='--', linewidth=1.4,
                   label=vline_label, zorder=5)
        ax.legend(fontsize=8, framealpha=0.9, edgecolor=CU_GRID,
                  loc='lower right', handlelength=1.5)

# Panel A — Expected Holding Time
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.barh(y, eht_v, height=h_bar, color=COLORS,
                 edgecolor='white', linewidth=0.7, zorder=3)
style_hbar_ax(ax1, 'A.  Expected Holding Time', 'Hours',
              x_max=max(eht_v)*1.22,
              vline=eht_v[-1], vline_label=f'Actual: {eht_v[-1]:.1f} hrs',
              vline_color=CU_CRIMSON)
for bar, val in zip(bars1, eht_v):
    ax1.text(bar.get_width() + max(eht_v)*0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.1f} hrs', va='center', ha='left', fontsize=8,
             color=CU_TEXT, fontweight='bold')

# Panel B — Hit-Rate
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.barh(y, hr_v, height=h_bar, color=COLORS,
                 edgecolor='white', linewidth=0.7, zorder=3)
style_hbar_ax(ax2, 'B.  Hit-Rate: |Deviation| > 20 bps', '% of 1-min bars',
              x_max=105,
              vline=hr_v[-1], vline_label=f'Actual: {hr_v[-1]:.1f}%',
              vline_color=CU_CRIMSON)
for bar, val in zip(bars2, hr_v):
    ax2.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', ha='left', fontsize=8,
             color=CU_TEXT, fontweight='bold')

# Panel C — MAE P99
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.barh(y, mae_v, height=h_bar, color=COLORS,
                 edgecolor='white', linewidth=0.7, zorder=3)
style_hbar_ax(ax3, 'C.  Max Adverse Excursion P99', 'bps',
              x_max=max(mae_v)*1.22,
              vline=COST_HURDLE_BPS, vline_label=f'Cost hurdle: {COST_HURDLE_BPS:.0f} bps',
              vline_color=CU_GOLD)
for bar, val in zip(bars3, mae_v):
    ax3.text(bar.get_width() + max(mae_v)*0.02, bar.get_y() + bar.get_height()/2,
             f'{val:.0f}', va='center', ha='left', fontsize=8,
             color=CU_TEXT, fontweight='bold')

# ── Divider line ──────────────────────────────────────────────────────────────
fig.add_artist(plt.Line2D([0.04, 0.97], [0.43, 0.43],
                          transform=fig.transFigure,
                          color=CU_GRID, linewidth=1.0))

# ── Table (bottom half) ───────────────────────────────────────────────────────
tbl_ax = fig.add_axes([0.04, 0.04, 0.92, 0.37])
tbl_ax.axis('off')

col_labels = ['Scenario', 'Half-Life\n(hrs)', 'κ (min⁻¹)', 'σ_ss\n(bps)',
              'E[Hold]\n(hrs)', 'Hit-Rate\n> 20 bps', 'MAE P99\n(bps)']
table_data = [
    [r['Scenario'],
     f"{r['HL_hrs']:.1f}",
     f"{r['kappa']:.5f}",
     f"{r['sigma_ss']:.1f}",
     f"{r['eht_hrs']:.2f}",
     f"{r['hit_rate']:.1f}%",
     f"{r['mae']:.0f}"]
    for _, r in df.iterrows()
]

tbl = tbl_ax.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    bbox=[0.0, 0.0, 1.0, 1.0]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)

for j in range(len(col_labels)):
    cell = tbl[(0, j)]
    cell.set_facecolor(CU_BLUE)
    cell.set_text_props(color='white', fontweight='bold', fontfamily='serif')
    cell.set_edgecolor('#FFFFFF')

for i in range(len(table_data)):
    is_crisis = (i == len(table_data) - 1)
    for j in range(len(col_labels)):
        cell = tbl[(i + 1, j)]
        cell.set_edgecolor('#E8ECF0')
        if is_crisis:
            cell.set_facecolor('#FDECEA')
            cell.set_text_props(fontweight='bold', color=CU_CRIMSON, fontfamily='serif')
        elif i % 2 == 0:
            cell.set_facecolor('#EBF3FB')
            cell.set_text_props(fontfamily='serif')
        else:
            cell.set_facecolor('white')
            cell.set_text_props(fontfamily='serif')

tbl_ax.set_title('Table R-1.  Sensitivity of Arbitrage Metrics to Hypothetical Half-Life Reduction',
                 fontsize=10, fontweight='bold', color=CU_BLUE, pad=8,
                 loc='left', fontfamily='serif')

# ── Footnote ──────────────────────────────────────────────────────────────────
fig.text(0.5, 0.005,
         'Methodology: All quantities derived analytically from the verified crisis AR(1) innovation std '
         '(σ_ε = 15 bps, Δt = 1 min). No new data collected. Entry threshold = 20 bps (cost hurdle from '
         'arbitrage simulation). E[Hold] = 1.44 × HL (Avellaneda & Lee 2010). '
         'MAE P99 uses Pickands extreme-value approximation for the supremum of a stationary Gaussian process.',
         ha='center', va='bottom', fontsize=7.5, color='#666666',
         style='italic', fontfamily='serif')

out = Path('/home/ubuntu/IAQF_2026/figures/lop/fig_r1_genius_sensitivity.png')
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=CU_GREY_BG)
plt.close()
print(f'Saved: {out}')
