# IAQF 2026 Student Competition Submission

**Project:** Analysis of Stablecoin Market Fragmentation and Microstructure during the March 2023 USDC De-Peg Crisis

**Author:** Columbia MAFN

**Date:** February 2026

---

## 1. Project Overview

This project provides a comprehensive empirical analysis of the cryptocurrency market's behavior during the March 2023 USDC de-peg event, submitted for the 2026 IAQF Student Competition. The study addresses four research questions:

1.  **Cross-Currency Basis (LOP):** How does the price of BTC/USDT compare to BTC/USD over time, and what drives persistent deviations once transaction costs are considered?
2.  **Stablecoin Dynamics:** How do premium/discount patterns in stablecoin-quoted markets vary across exchanges and regimes?
3.  **Liquidity & Fragmentation:** Does liquidity differ systematically across quote currencies? How do order book depth, spread, and volatility vary?
4.  **Regulatory Overlay:** What are the implications of the GENIUS Act and stablecoin settlement adoption for market structure and efficiency?

---

## 2. Addendum: Paper Artifact Map

This section maps the figures and tables used in the final paper to their corresponding artifacts in this repository. **Paper references exported artifacts directly; rerunning is not required for evaluation.** Notebooks and scripts are included for reproducibility.

### Main Narrative Figures (Fig. 1–7)

| Paper Label | Repo Path | Producer Notebook | Data Source(s) |
| :--- | :--- | :--- | :--- |
| **Fig. 1** | `figures/master/master_fig1_price_lop_overview.png` | `IAQF_Master_Analysis.ipynb` | `data/parquet/panel_1hour.parquet` |
| **Fig. 2** | `figures/master/master_fig3_stablecoin_depeg.png` | `IAQF_Master_Analysis.ipynb` | `data/parquet/panel_1min.parquet` |
| **Fig. 3** | `figures/master/master_fig9_crisis_deep_dive.png` | `IAQF_Master_Analysis.ipynb` | `data/parquet/panel_1min.parquet` |
| **Fig. 4** | `figures/master/master_fig6_kyle_lambda.png` | `IAQF_Master_Analysis.ipynb` | `data/parquet/l2_all_pairs_1min.parquet` |
| **Fig. 5** | `figures/master/master_fig11_regression_coefs.png` | `IAQF_Master_Analysis.ipynb` | `data/parquet/panel_1min.parquet` |
| **Fig. 6** | `figures/lop/fig_x_spread_depth_vol.png` | `compute_table_x.py` | `data/parquet/panel_1min.parquet` |
| **Fig. 7** | `figures/lop/fig_r1_genius_sensitivity.png` | `compute_sensitivity_table.py` | `table_sensitivity_genius.csv` |

### Appendix Figures

| Paper Label | Repo Path | Producer Notebook | Data Source(s) |
| :--- | :--- | :--- | :--- |
| **Fig. K-1** | `figures/kraken/kraken_fig1_wedge_timeseries.png` | `IAQF_CrossExchange_Kraken.ipynb` | `data/cross_exchange/kraken_btcusd_1min.parquet` |
| **Fig. K-2** | `figures/kraken/kraken_fig2_stablecoin_fx.png` | `IAQF_CrossExchange_Kraken.ipynb` | `data/cross_exchange/kraken_usdcusd_1min.parquet` |
| **Fig. K-3** | `figures/kraken/kraken_fig3_regime_bars.png` | `IAQF_CrossExchange_Kraken.ipynb` | `data/cross_exchange/kraken_btcusd_1min.parquet` |
| **Fig. B-1** | `figures/bybit/bybit_fig1_crisis_window.png` | `IAQF_CrossExchange_Bybit_executed.ipynb` | `data/cross_exchange/bybit_btcusdt_1min.parquet` |
| **Fig. OU-1**| `figures/ou/ou_fig1_residual_timeseries.png` | `IAQF_OU_Analysis_executed.ipynb` | *Self-fetches from Binance.US API* |
| **Fig. OU-2**| `figures/ou/ou_fig2_impulse_response.png` | `IAQF_OU_Analysis_executed.ipynb` | *Self-fetches from Binance.US API* |

### Tables

| Paper Label | Description | Producer Script | Data Source(s) |
| :--- | :--- | :--- | :--- |
| **Table I** | Data Sources | `generate_data.py`, `fetch_cross_exchange.py` | *N/A* |
| **Table II**| Data-to-Question Mapping | *N/A* | *N/A* |
| **Table VII**| Stablecoin Premia/Discounts | `compute_table_vii.py` | `panel_1min.parquet`, `kraken_*.parquet` |
| **Table VIII**| Cross-Venue Correlation | `compute_table_vii.py` | `panel_1min.parquet`, `kraken_usdcusd_1min.parquet` |
| **Table IX**| USDC vs USDT Confidence | `compute_table_vii.py` | `panel_1min.parquet` |
| **Table X** | Spread, Depth, Volatility | `compute_table_x.py` | `panel_1min.parquet` |
| **Table XI**| Stress Ratios | `compute_table_x.py` | `panel_1min.parquet` |
| **Table B-1**| Bybit Microstructure | `IAQF_CrossExchange_Bybit_executed.ipynb` | `bybit_btcusdt_1min.parquet` |
| **Table OU-1**| OU Parameters & Half-Life | `IAQF_OU_Analysis_executed.ipynb` | *Self-fetches from Binance.US API* |

---

## 3. Key Results Used in Paper

-   **LOP Deviation Peak:** The Law-of-One-Price deviation between BTC/USDC and BTC/USD peaked at over **1,200 bps** during the crisis, a 100-fold increase from the pre-crisis median of -0.17 bps (Fig. 1, Fig. 3).
-   **USDC Discount & Synchronization:** The USDC/USD exchange rate traded at a median discount of **-325 bps** on Binance.US and **-301 bps** on Kraken during the crisis, with a minimum discount of ~1400 bps on both venues (Table VII). The cross-venue correlation of this deviation jumped from near-zero to **0.91** during the crisis, confirming a systemic, market-wide event (Table VIII).
-   **Liquidity Fragmentation:** During the crisis, Kyle's Lambda (price impact) for BTC/USDC was **64× higher** than for BTC/USDT, and the Amihud illiquidity ratio was **4.4× higher**. Spreads for both BTC/USD and BTC/USDT more than **doubled** (Tables X-XI, Fig. 4).
-   **Persistence of Mispricing:** The Ornstein-Uhlenbeck half-life of the LOP deviation increased from **3.2 minutes** pre-crisis to **602.7 minutes (10.0 hours)** during the crisis, indicating a severe breakdown in arbitrage capacity (Table OU-1, Fig. OU-2).
-   **Offshore Robustness:** The Bybit (offshore, non-USD) BTC/USDT market also experienced significant stress, with spreads widening **2.5×**. The timing of this deterioration was highly correlated with the USDC de-peg stress on Binance.US, confirming market-wide contagion (Table B-1, Fig. B-1).

---

## 4. Additional Work (Not Included in Main Paper)

We explored a Covered Interest Parity (CIP) style framing using crypto perpetuals/futures to construct a synthetic USD funding rate. This work, while contextually relevant, was judged by our supervisor as somewhat off-tangent and too extensive for the main submission. It remains as a separate line of inquiry and may be provided as a separate document on request.

---

## 5. How to Run (Optional Reproducibility)

### Prerequisites

Install all required Python packages:

```bash
pip install pandas numpy matplotlib seaborn statsmodels scipy \
            hmmlearn arch scikit-learn tslearn openpyxl pyarrow \
            requests jupyter nbformat nbconvert
```

### Step 1 — Generate the LOP Panel Data (~5 min)

Run from the **project root** (`IAQF_2026/`):

```bash
python generate_data.py --skip-l2
```

This fetches all 1-minute OHLCV data from Binance.US and Coinbase, builds the harmonized panel, and saves `panel_1min.parquet`, `panel_1hour.parquet`, and `panel_daily.parquet` to `data/parquet/`. The `--skip-l2` flag skips the ~20 GB tick data download.

### Step 2 — Generate L2 Microstructure Data (~60–90 min, optional)

To also run the L2 microstructure sections of the master notebook:

```bash
python generate_data.py
```

This additionally downloads ~20 GB of `aggTrades` tick data from the Binance public archive (`data.binance.vision`) and processes ~150 million trades into 1-minute L2 metrics.

### Step 3 — Generate Cross-Exchange Data (~2 hours, optional)

To run the Kraken and Bybit cross-exchange notebooks:

```bash
python fetch_cross_exchange.py
```

This fetches Kraken tick trades (BTC/USD, USDC/USD, USDT/USD, USDC/USDT) and Bybit BTC/USDT tick data via their public REST APIs.

### Step 4 — Run Notebooks

Open any notebook from the `notebooks/` directory. All data is loaded using relative paths (`../data/parquet/` or `../data/cross_exchange/`):

```bash
cd notebooks/
jupyter notebook IAQF_Master_Analysis.ipynb
```

**Notebook dependency summary:**

| Notebook | Data Required | Self-Fetches? |
|---|---|---|
| `IAQF_Master_Analysis.ipynb` | `data/parquet/panel_1min.parquet` + L2 files | No — run `generate_data.py` first |
| `IAQF_Advanced_Models.ipynb` | `data/parquet/panel_1min.parquet` | No — run `generate_data.py --skip-l2` first |
| `IAQF_Arbitrage_Simulation_executed.ipynb` | `data/parquet/panel_1min.parquet` | No |
| `IAQF_BasisRisk_Analysis_executed.ipynb` | `data/parquet/panel_1min.parquet` | No |
| `IAQF_OU_Analysis_executed.ipynb` | None | **Yes** — fetches from Binance.US API |
| `IAQF_CrossExchange_Bybit_executed.ipynb` | `data/cross_exchange/bybit_btcusdt_1min.parquet` | Partially (USDC/USD from Binance.US) |
| `IAQF_CrossExchange_Kraken.ipynb` | `data/cross_exchange/kraken_*.parquet` | Partially (BTC/USD from Binance.US + Coinbase) |

---

## 6. Directory Structure

```
IAQF_2026/
├── README.md                                     ← This file
│
├── notebooks/                                    ← All Jupyter notebooks
│   ├── IAQF_Master_Analysis.ipynb                ← Phase 1 (LOP) + Phase 2 (L2), 39 cells
│   ├── IAQF_Advanced_Models.ipynb                ← HMM, GARCH, VAR, RF, DTW, MS, Hawkes, PCA
│   ├── IAQF_Arbitrage_Simulation_executed.ipynb  ← Triangular arbitrage simulation
│   ├── IAQF_BasisRisk_Analysis_executed.ipynb    ← Basis decomposition, stress sensitivity, tails
│   ├── IAQF_OU_Analysis_executed.ipynb           ← OU mean-reversion analysis by regime
│   ├── IAQF_CrossExchange_Bybit_executed.ipynb   ← Bybit L2 order book depth/spread analysis
│   └── IAQF_CrossExchange_Kraken.ipynb           ← Kraken 3-way BTC/USD venue wedge analysis
│
├── data/
│   ├── parquet/                                  ← Generated by generate_data.py
│   │   ├── panel_1min.parquet                    ← 30,240 rows × 74 cols — main LOP panel
│   │   ├── panel_1hour.parquet                   ← 504 rows — hourly resampled
│   │   ├── panel_daily.parquet                   ← 21 rows — daily aggregates
│   │   ├── harmonized_raw_1min.parquet           ← raw OHLCV before feature engineering
│   │   ├── l2_BTCUSDT_1min.parquet               ← 30,240 rows — L2 metrics BTCUSDT
│   │   ├── l2_BTCUSDC_1min.parquet               ← 14,010 rows — L2 metrics BTCUSDC
│   │   └── l2_all_pairs_1min.parquet             ← 44,250 rows — combined L2 panel
│   └── cross_exchange/                           ← Generated by fetch_cross_exchange.py
│       ├── kraken_btcusd_1min.parquet            ← 29,881 rows
│       ├── kraken_usdcusd_1min.parquet           ← 20,794 rows
│       ├── kraken_usdtusd_1min.parquet
│       ├── kraken_usdcusdt_1min.parquet
│       └── bybit_btcusdt_1min.parquet            ← 30,240 rows
│
├── generate_data.py                              ← Generates all data/parquet/ files
├── fetch_cross_exchange.py                       ← Generates data/cross_exchange/ files
│
├── figures/                                      ← All output figures (30+)
│   ├── master/                                   ← Figs 1–5 from master notebook
│   ├── advanced/                                 ← Figs 6–11 from advanced models
│   ├── arb/                                      ← Figs 12–14 from arbitrage simulation
│   ├── br/                                       ← Figs 15–19 from basis risk analysis
│   ├── ou/                                       ← Figs OU-1 to OU-3 from OU analysis
│   ├── bybit/                                    ← Figs B-1 to B-3 from Bybit analysis
│   └── kraken/                                   ← Figs K-1 to K-3 from Kraken analysis
│
└── docs/
    └── IAQFStudentCompetition2026.pdf            ← Original competition problem statement
```
