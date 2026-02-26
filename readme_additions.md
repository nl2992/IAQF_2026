
---

## 11. Results: Cross-Exchange Analysis — Kraken & Bybit

To directly address Research Question 2 ("premium/discount patterns vary across exchanges") and Question 3 ("liquidity differs systematically"), we conducted a dedicated cross-exchange analysis using two additional venues: **Kraken** and **Bybit**. This provides an external benchmark to the primary analysis, which was focused on Binance.US and Coinbase.

### 11.1 Kraken: 3-Way BTC/USD Venue Wedge Analysis

We introduce Kraken as a third major US-based exchange, constructing a 3-way BTC/USD price wedge between Binance.US, Coinbase, and Kraken. This allows us to observe how arbitrage opportunities and venue-specific pricing evolved across the three largest US dollar liquidity pools.

#### Figure K-1: 3-Way BTC/USD Venue Wedge Time Series

The time series of pairwise wedges shows a dramatic expansion during the crisis. The Binance.US vs. Coinbase wedge (Panel B) reached nearly -80 bps, indicating a significant Coinbase premium. The introduction of Kraken reveals even more complex dynamics, with the Coinbase vs. Kraken wedge (Panel D) also showing significant dislocation.

![Figure K-1: 3-Way BTC/USD Venue Wedge Time Series](figures/kraken/kraken_fig1_wedge_timeseries.png)

#### Table K-1: Venue Wedge Distribution by Regime

Quantifying the visual evidence, the table below shows that the median absolute wedge and the 95th percentile of the wedge distribution increased dramatically during the crisis. The probability of a large deviation (`|wedge| > 20 bps`) for the Binance.US-Coinbase pair jumped from 0.0% in the pre-crisis period to 8.8% during the crisis and 14.1% in the recovery.

```
Table K-1: Venue Wedge Distribution by Regime
====================================================================================================
    Regime        Wedge     N  Median (bps)  Mean (bps)  Std (bps)  95th pct (bps)  5th pct (bps)  P(|wedge|>5bps)  P(|wedge|>20bps)
Pre Crisis     BinUS-CB 12388         -0.17       -0.17       1.27            1.75          -2.12              0.4               0.0
Pre Crisis BinUS-Kraken 12388         -0.44       -0.49       2.57            3.65          -4.72              6.5               0.0
Pre Crisis    CB-Kraken 12388         -0.21       -0.32       2.41            3.59          -4.47              5.8               0.0
    Crisis     BinUS-CB  4320         -3.65       -6.84       8.94            1.43         -24.73             43.6               8.8
    Crisis BinUS-Kraken  4320         -2.68       -4.47       8.12            6.11         -20.22             43.0               5.1
    Crisis    CB-Kraken  4320          1.28        2.37       6.89           13.41          -6.20             33.0               2.0
  Recovery     BinUS-CB  4320         -6.04       -8.96      10.18            2.18         -26.98             55.4              14.1
  Recovery BinUS-Kraken  4320         -3.39       -5.84       9.73            6.51         -23.30             52.5               9.2
  Recovery    CB-Kraken  4320          2.82        3.12       6.61           14.20          -7.27             43.0               1.6
      Post     BinUS-CB  8640         -2.03       -3.50       5.45            2.95         -14.21             30.5               1.6
      Post BinUS-Kraken  8633          0.97        1.83       9.43           19.32         -13.62             48.3               5.5
      Post    CB-Kraken  8633          2.76        5.32       7.73           21.45          -3.13             35.4               7.0
```

#### Figure K-2: Stablecoin FX Deviation Across Venues

Comparing the USDC/USD price on Binance.US and Kraken reveals that the de-pegging was a market-wide phenomenon, not an issue specific to one exchange. Both venues saw USDC trade at a steep discount, reaching below -1200 bps (-12%) at the peak of the crisis. The recovery paths are also remarkably similar, suggesting strong arbitrage links between the venues.

![Figure K-2: Stablecoin FX Deviation Across Venues](figures/kraken/kraken_fig2_stablecoin_fx.png)

### 11.2 Bybit: Offshore BTC/USDT Microstructure

We use tick-level trade data for the BTC/USDT perpetual swap contract from Bybit, a large offshore derivatives exchange, as a proxy for global, non-US market sentiment and liquidity. This helps isolate the "US effect" (e.g., regulatory concerns, banking issues) from the global market's reaction.

#### Figure B-1 & B-2: Bybit Microstructure Time Series

The time series of spread and depth proxies on Bybit shows a clear degradation of market quality during the crisis period, even for a USDT-quoted pair on an offshore venue. The spread proxy (a measure of transaction costs) spiked, while the depth proxy (a measure of available liquidity) plummeted. This indicates that the USDC crisis had a spillover effect on the broader crypto market structure.

![Figure B-1: Bybit Spread and Depth](figures/bybit/bybit_fig1_spread_depth_ts.png)
![Figure B-2: Bybit Microstructure](figures/bybit/bybit_fig2_microstructure_ts.png)

#### Table B-1: Bybit Microstructure by Regime

The table below summarizes the key microstructure metrics for Bybit's BTC/USDT market across the four regimes. During the crisis, the median spread proxy more than doubled from 5.3 bps to 12.1 bps, while Kyle's Lambda (a measure of price impact) increased by over 4x, indicating that trades had a much larger effect on price. This confirms a significant reduction in market quality and liquidity.

```
Table B-1: Bybit BTC/USDT Microstructure by Regime
======================================================================================
    Regime      Metric         Mean          Std       Median     95th pct  P(Stress)
Pre-Crisis      Spread        7.307       6.489        5.301       21.018        3.8
Pre-Crisis       Depth   158892.234  273278.344    83031.250   505433.188        3.8
Pre-Crisis      Lambda       12.169      20.945        5.529       41.369        3.8
    Crisis      Spread       15.932      16.349       12.131       46.196       61.2
    Crisis       Depth    36338.281   45038.359    23809.523   110031.250       61.2
    Crisis      Lambda       55.231     123.352       22.906      212.453       61.2
  Recovery      Spread       10.450       9.587        7.969       30.557       26.4
  Recovery       Depth    73153.438   88243.500    48046.875   234375.000       26.4
  Recovery      Lambda       27.841      46.523       13.988       94.234       26.4
      Post      Spread        6.783       5.839        5.148       18.984        0.0
      Post       Depth   195652.562  316288.562   111328.125   656250.000        0.0
      Post      Lambda       11.593      21.034        5.195       38.082        0.0
```

This cross-exchange analysis confirms that the March 2023 crisis was a systemic event. The USDC de-peg was consistent across major venues, and its effects on liquidity and price stability spilled over into the broader market, impacting even non-USD pairs on offshore exchanges.
