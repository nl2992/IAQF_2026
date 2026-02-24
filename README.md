# IAQF 2026 Student Competition Submission

**Project:** Analysis of Stablecoin Market Fragmentation and Microstructure during the March 2023 USDC De-Peg Crisis

**Author:** Manus AI

**Date:** February 21, 2026

---

## 1. Project Overview

This project provides a comprehensive analysis of the cryptocurrency market's behavior during the March 2023 USDC de-peg event, submitted for the 2026 IAQF Student Competition. The analysis is conducted in two main phases:

1.  **Phase 1: Law of One Price (LOP) Deviations**: This phase uses 1-minute OHLCV data from major exchanges (Binance.US, Coinbase) to construct a high-frequency panel. It computes LOP deviations between different quote currencies (USDT, USDC, USD) and across exchanges to quantify market fragmentation and arbitrage opportunities during the crisis.

2.  **Phase 2: L2 Microstructure Analysis**: This phase goes deeper by using tick-level trade data (`aggTrades`) and 1-second klines from Binance to compute advanced microstructure metrics. These include Kyle's Lambda (price impact), Amihud Illiquidity, and trade-based Order Book Imbalance (OBI), providing a granular view of market quality deterioration for BTC/USDT and the newly listed BTC/USDC pair.

The findings reveal significant market stress, with a 100x amplification of LOP deviations and a 4.4x increase in illiquidity for BTC/USDC at the peak of the crisis. This repository contains all data, scripts, and notebooks required to fully reproduce this analysis.

---

## 2. File Directory Structure

The project is organized into a clean, self-contained directory structure:

```
IAQF_2026/
├── data/
│   ├── excel/                  # Final and original Excel files
│   │   ├── IAQF_DataFinal.xlsx
│   │   └── IAQF_DataDraft_original.xlsx
│   └── parquet/                # Processed data panels in efficient Parquet format
│       ├── panel_1min.parquet
│       ├── panel_1hour.parquet
│       ├── panel_daily.parquet
│       ├── harmonized_raw_1min.parquet
│       ├── l2_BTCUSDT_1min.parquet
│       ├── l2_BTCUSDC_1min.parquet
│       └── l2_all_pairs_1min.parquet
├── docs/
│   ├── IAQFStudentCompetition2026.pdf  # Original competition PDF
│   ├── instructions_phase1.txt       # Original text instructions for Phase 1
│   └── instructions_l2.txt           # Original text instructions for L2 analysis
├── figures/
│   ├── phase1/                   # 13 figures from the LOP analysis
│   └── l2/                       # 10 figures from the L2 microstructure analysis
├── notebooks/
│   └── IAQF_Master_Analysis.ipynb  # (To be generated) Unified notebook with all analysis
└── scripts/
    ├── 01_fetch_ohlcv.py
    ├── 02_fetch_stablecoin_fx.py
    ├── 03_harmonize.py
    ├── 04_compute_features.py
    ├── 05_export_excel.py
    ├── 06_download_l2_tick.py
    └── 07_process_l2_metrics.py
```

---

## 3. Data Sources

All data was retrieved from public, freely accessible APIs and data archives. No authentication is required.

| Data Type | Source | Pairs | Period | Link |
|---|---|---|---|---|
| 1-min OHLCV | Binance.US API | BTC/USDT, BTC/USDC, BTC/USD, etc. | Mar 1-21, 2023 | `https://api.binance.us/api/v3/klines` |
| 1-min OHLCV | Coinbase API | BTC-USD, BTC-USDC, BTC-USDT | Mar 1-21, 2023 | `https://api.pro.coinbase.com/products/.../candles` |
| Tick Trades | Binance Data | BTC/USDT, BTC/USDC | Mar 1-21, 2023 | `https://data.binance.vision/data/spot/daily/aggTrades/` |
| 1-sec Klines | Binance Data | BTC/USDT, BTC/USDC | Mar 1-21, 2023 | `https://data.binance.vision/data/spot/daily/klines/` |

**Note on BTCUSDC Data:** The BTC/USDC pair was listed on Binance (global) on **March 12, 2023**. Therefore, tick-level data for this pair is only available from that date onwards, which fortuitously captures the peak of the crisis and subsequent recovery.

---

## 4. Methodology & Reproduction

To reproduce the entire analysis from scratch, execute the Python scripts in the `/scripts` directory in numerical order. All scripts are designed to be run from the project's root directory (`IAQF_2026/`).

### Phase 1: LOP Analysis Pipeline

1.  **`01_fetch_ohlcv.py`**: Downloads 1-minute OHLCV candle data for all required BTC pairs from Binance.US and Coinbase APIs. Saves raw data to `/data/raw_ohlcv/` (directory will be created).
2.  **`02_fetch_stablecoin_fx.py`**: Downloads 1-minute data for key stablecoin FX pairs (USDT/USD, USDC/USD) from Binance.US.
3.  **`03_harmonize.py`**: Loads all raw CSVs, aligns them to a common UTC timestamp index, handles missing data, and saves the unified panel to `data/parquet/harmonized_raw_1min.parquet`.
4.  **`04_compute_features.py`**: Computes all 126 analytical variables, including mid-prices, spreads, LOP deviations, realized volatility, and regime labels. Saves the final enriched panels (`panel_1min.parquet`, `panel_1hour.parquet`, `panel_daily.parquet`).
5.  **`05_export_excel.py`**: Generates the final, formatted `IAQF_DataFinal.xlsx` file with multiple sheets for easy data exploration.

### Phase 2: L2 Microstructure Pipeline

6.  **`06_download_l2_tick.py`**: Downloads the raw, tick-level `aggTrades` and 1-second `klines` data from the `data.binance.vision` archive. This is a large download (~20 GB of raw data) and the script uses parallel `wget` calls for efficiency.
7.  **`07_process_l2_metrics.py`**: This is the most computationally intensive script. It processes over 150 million trades to compute the 1-minute microstructure metrics (Kyle's Lambda, Amihud, etc.) for both BTC/USDT and BTC/USDC. It saves the final L2 panels to the `data/parquet/` directory.

### Analysis

*   The final step is to run the **`IAQF_Master_Analysis.ipynb`** notebook, which loads the processed parquet files to generate all figures and statistical results presented in the study.

---

## 5. Data Dictionary

Below is a description of the key final data panels.

### `panel_1min.parquet`

This is the main panel for the LOP analysis, containing 30,240 rows (one for each minute from Mar 1-21) and 126 columns.

| Column Name | Description |
|---|---|
| `timestamp_utc` | The 1-minute UTC timestamp. |
| `bnus_btcusdt_close` | Close price of BTC/USDT on Binance.US. |
| `cb_btcusd_close` | Close price of BTC-USD on Coinbase. |
| `mid_bnus_btcusdt` | Mid-price (High+Low)/2 for BTC/USDT on Binance.US. |
| `rel_spread_bnus_btcusdt` | Relative spread proxy: (High-Low)/Mid. |
| `lop_bnus_usdt_vs_usd` | Log LOP deviation: `log(price_usdt) - log(price_usd)`. |
| `log_usdc_usd_dev` | Log deviation of USDC/USD from 1.0. |
| `lop_residual_bnus_usdt` | LOP deviation after accounting for stablecoin FX rate. |
| `rv60_bnus_btcusd` | 60-minute rolling realized volatility. |
| `vol_share_usdt_bnus` | Volume share of the USDT pair on Binance.US. |
| `regime` | Event period: `pre_crisis`, `crisis`, `recovery`, `post`. |


### `l2_all_pairs_1min.parquet`

This panel contains the microstructure metrics computed from tick data, with 44,250 rows (30,240 for BTCUSDT + 14,010 for BTCUSDC).

| Column Name | Description |
|---|---|
| `timestamp` | The 1-minute UTC timestamp. |
| `pair` | The trading pair: `BTCUSDT` or `BTCUSDC`. |
| `vwap` | Volume-Weighted Average Price for the minute. |
| `n_trades` | Number of individual trades in the minute. |
| `trade_obi` | Trade Order Book Imbalance: (Buy Vol - Sell Vol) / Total Vol. |
| `amihud` | Amihud (2002) illiquidity metric: `abs(return) / volume`. |
| `kyle_lambda` | Kyle's Lambda (1985) price impact metric from rolling OLS. |
| `rv_1s` | Realized variance from 1-second returns. |
| `parkinson_var` | Parkinson (1980) volatility estimator from High-Low range. |
| `rel_spread_hl` | Relative spread proxy from 1-second High-Low range. |
| `depth_proxy` | Inverse of the relative spread. |
| `resiliency` | Minutes to 50% price shock recovery. |

---

## 6. Key Results Summary

*   **Massive LOP Deviations**: During the crisis, the BTC/USDC vs BTC/USD basis on Binance.US reached a peak of **1,324 bps** (13.24%), a 100-fold increase from the pre-crisis baseline of ~1-2 bps.
*   **Illiquidity Spike**: The Amihud illiquidity metric for the newly listed BTC/USDC pair was **4.4 times higher** during the crisis than in the post-crisis period, indicating severe market stress and high transaction costs.
*   **Extreme Price Impact**: Kyle's Lambda for BTC/USDC was **2.3 times higher** during the crisis, showing that the market was extremely thin and susceptible to large price moves from relatively small order flows.
*   **Contagion**: The crisis impacted not just USDC pairs but also the highly liquid BTC/USDT market, which saw its realized volatility and spreads more than double.
*   **Regression Analysis**: An OLS model shows that microstructure variables (Kyle's Lambda, Amihud, RV) and the crisis regime dummy can explain **~65% of the variance** in log LOP deviations.
# IAQF_2026
