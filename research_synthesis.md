# Research Synthesis & Analysis Plan

**Author**: Manus AI
**Date**: February 23, 2026

## 1. Introduction

This document synthesizes findings from academic literature, regulatory analysis, and market commentary to address the four core questions of the IAQF 2026 Student Competition. It maps these findings to the analytical outputs already generated from our dataset and, most importantly, identifies specific, high-impact analytical gaps. For each gap, a new, informative model is proposed to provide a more complete answer to the competition's prompts. The goal is not just to run more models, but to select models that directly address the nuanced economic and regulatory questions at hand.

---

## 2. Question 1: Cross-Currency Basis

> *"How does the price of BTC/USDT compare to BTC/USD over time? Do we observe persistent differences once transaction costs are considered, and what drives those differences?"*

### Research Synthesis

The academic consensus is that the Law of One Price (LOP) is systematically violated in cryptocurrency markets [1] [2]. The BTC/USDT vs. BTC/USD basis is not zero and is driven by two primary components: (1) the deviation of the stablecoin's price from its $1 peg (the "FX component"), and (2) a residual basis reflecting market frictions and order flow imbalances [3]. Arbitrage is limited by transaction costs, capital controls, and exchange-specific frictions (e.g., withdrawal limits, counterparty risk), creating a no-arbitrage band around the theoretical price [1]. During periods of market stress, this band widens significantly as arbitrageurs face heightened risk and capital constraints, leading to persistent deviations.

### Existing Code Coverage

Our analysis has thoroughly documented the existence and magnitude of the LOP deviations. The `IAQF_Master_Analysis.ipynb` notebook provides:

- **Time-series plots** of the raw BTC/USDT and BTC/USDC prices versus BTC/USD, visually demonstrating the divergence during the crisis.
- **Log LOP deviation calculations** (`lop_bnus_usdt_vs_usd`, `lop_bnus_usdc_vs_usd`), which quantify the basis in percentage terms.
- **Decomposition** of the LOP deviation into its `stablecoin_fx_component` and a `lop_residual`, confirming the findings of Alexander & Imeraj [3].
- **Statistical tests** (Kruskal-Wallis, Mann-Whitney) showing that the LOP deviations are statistically different across pre-crisis, crisis, and post-crisis regimes.
- **Cointegration tests** confirming a long-run equilibrium relationship between the prices.
- **Vector Autoregression (VAR)** models that trace the dynamic impact of a shock in one variable (e.g., USDC/USD) on the LOP deviation.

### Identified Gaps & Proposed Models

While we have documented the *what*, we can do more to explain the *why* and *how*.

| Gap | Proposed Model/Analysis | Justification |
|---|---|---|
| **1. Arbitrage Profitability** | **Triangular Arbitrage Simulation**: Model a strategy (BTC/USD → BTC/USDT → USDT/USD → USD) including realistic transaction costs (0.1% taker fees), withdrawal fees, and estimated blockchain latency. Calculate the net profit/loss per minute. | This directly answers the "once transaction costs are considered" part of the question. It moves beyond showing a deviation exists to quantifying whether it was an exploitable arbitrage opportunity, providing a measure of market inefficiency. |
| **2. Speed of Convergence** | **Ornstein-Uhlenbeck (OU) Process / Half-Life Calculation**: Fit an OU model to the `lop_residual` series for each regime (pre-crisis, crisis, post-crisis). Calculate the half-life of the deviation in each regime. | This quantifies how quickly the market corrects itself. We can test the hypothesis that the speed of mean-reversion collapsed during the crisis, providing a direct measure of how arbitrage efficiency broke down. |
| **3. Dynamic Relationships** | **Rolling Cointegration Analysis**: Perform rolling Engle-Granger cointegration tests on a 24-hour window. Plot the test statistic over time. | A single cointegration test assumes a stable long-run relationship. A rolling test can visually demonstrate the *breakdown* of this relationship during the crisis, showing a structural shift in market dynamics. |

---

## 3. Question 2: Stablecoin Dynamics

> *"How do premium/discount patterns in stablecoin quoted markets (e.g., USDT vs USDC) vary across exchanges and regimes? How might forthcoming U.S. regulation affect confidence in these instruments?"*

### Research Synthesis

The March 2023 USDC de-peg was a classic coordination failure triggered by a credit event (SVB failure) that exposed a portion of Circle's reserves [4]. The key mechanism was the temporary failure of the primary arbitrage channel: with Coinbase and others suspending USDC↔USD conversions over the weekend, institutional arbitrageurs could not redeem USDC for $1, breaking the peg's primary defense [5]. This highlights the distinction between the primary (issuance/redemption) and secondary (exchange trading) markets. During this period, USDT, despite its historically more opaque reserves, experienced a "flight to quality" effect, trading at a premium as traders sought an alternative stablecoin not exposed to US banking risk [5]. This reveals that in a crisis, the specific nature of the risk (US banking vs. offshore regulatory) determines the direction of capital flight.

### Existing Code Coverage

- **Time-series plots** of USDC/USD and USDT/USD, clearly showing the USDC de-peg and the slight USDT premium.
- **Regime-based statistics** quantifying the average premium/discount in each period.
- **Volume analysis** showing the shift in trading activity from USDC- to USDT-quoted pairs during the crisis.
- **Granger causality tests** linking the USDC de-peg to the BTC/USDC LOP deviation.

### Identified Gaps & Proposed Models

| Gap | Proposed Model/Analysis | Justification |
|---|---|---|
| **1. Contagion Effects** | **Dynamic Conditional Correlation (DCC) GARCH**: Model the time-varying correlation between the returns of USDC/USD, USDT/USD, and DAI/USD (another major stablecoin). | This directly tests for contagion. Did the shock to USDC spill over and increase correlation with other stablecoins? A DCC-GARCH model can capture this dynamic relationship far better than a static correlation matrix. |
| **2. Re-Peg Dynamics** | **Survival Analysis (e.g., Kaplan-Meier Estimator)**: Treat the de-peg as a "failure" state. Model the probability of "surviving" (i.e., remaining de-pegged) over time. Calculate the median time to re-peg. | This provides a more rigorous answer to "how long did it take to recover?" than simple observation. It is a standard way to model the duration of an event and can be used to compare the 2023 event to other historical de-pegs. |

---

## 4. Question 3: Liquidity & Fragmentation

> *"Does liquidity differ systematically across quote currencies? How do order book depth, spread, and volatility vary between BTC quoted in USD versus stablecoins?"*

### Research Synthesis

Liquidity is highly fragmented in crypto markets, both across exchanges and across quote currencies on the same exchange [6]. USDT pairs are generally the most liquid globally, especially on offshore exchanges, while USD pairs lead during US trading hours. This fragmentation means that liquidity is not a single number but a complex, dynamic property. During stress events, liquidity can evaporate from one market (e.g., BTC/USDC) and reappear in another (e.g., BTC/USDT), amplifying price dislocations. Key microstructure metrics like bid-ask spreads, order book depth, and price impact (e.g., Kyle's Lambda) are the standard tools for quantifying these differences [7].

### Existing Code Coverage

Our L2 analysis provides an exceptionally strong foundation here:

- **Comprehensive microstructure metrics** computed from tick data: Kyle's Lambda, Amihud Illiquidity, Realized Volatility, Spread Proxies, and Trade OBI for both BTC/USDT and BTC/USDC.
- **Direct comparison** of these metrics across the two pairs and across regimes, showing that illiquidity (Lambda, Amihud) spiked dramatically in the BTC/USDC pair during the crisis.
- **Random Forest analysis** identifying Kyle's Lambda and Amihud as the most important predictors of the crisis regime.
- **PCA / Factor Analysis** showing that a single "market stress" factor, heavily loaded on spread and price impact metrics, explains a significant portion of the variance.

### Identified Gaps & Proposed Models

Our coverage is very strong, but we can add more nuance.

| Gap | Proposed Model/Analysis | Justification |
|---|---|---|
| **1. Intraday Liquidity Patterns** | **Intraday Seasonality Decomposition**: Use a statistical method (e.g., STL decomposition or trigonometric regression) to decompose the spread and depth series into trend, seasonal (intraday pattern), and residual components for each quote currency. | This would formally model the U-shaped intraday liquidity pattern. We can test if the crisis *structurally altered* the intraday pattern (e.g., did liquidity disappear during Asian hours for USDC markets?), providing a more granular view of fragmentation. |

---

## 5. Question 4: Regulatory Overlay

> *"Tie your empirical findings to the broader policy context... Why might regulated stablecoins alter cross-currency trading patterns? What implications does the GENIUS Act... have for the structure and efficiency of these markets?"*

### Research Synthesis

The **GENIUS Act of 2025** imposes a federal framework for payment stablecoins, mandating 1:1 reserves of high-quality liquid assets (cash, T-bills), monthly public reserve reports, and orderly redemption plans [8]. It creates a dual charter system for bank and non-bank issuers. Crucially, it prioritizes stablecoin holders in the event of issuer insolvency. The goal is to prevent runs like the one on USDC by increasing transparency and confidence in the reserve backing. The adoption of regulated stablecoins for settlement by major payment networks like Visa and Mastercard would further integrate them into the traditional financial system, increasing their utility and demand [9].

**Implications**:
- **Reduced De-peg Risk**: By mandating high-quality reserves and clear redemption rights, the Act should theoretically reduce the probability and severity of de-pegs caused by credit or liquidity risk in the issuer's reserves. This would reduce the `stablecoin_fx_component` of the LOP deviation.
- **Increased Confidence & Homogenization**: A "GENIUS-compliant" stablecoin would become a benchmark for safety. This could lead to a concentration of liquidity in a few regulated, US-domiciled stablecoins. The difference between USDC and USDT would become even more stark: one fully regulated, the other offshore.
- **Shift in Trading Patterns**: If regulated stablecoins become the preferred settlement asset, we would expect the `lop_residual` to shrink for pairs quoted against them. Arbitrage would become more efficient as counterparty risk associated with the stablecoin itself diminishes. The basis would be driven more by pure market frictions (latency, fees) than by credit risk premia.

### Existing Code Coverage

Our empirical findings provide a perfect 
case study of what happens when confidence is lost. The massive spike in the LOP deviation, the flight to USDT, and the explosion in illiquidity metrics like Kyle's Lambda are exactly the phenomena the GENIUS Act aims to prevent.

- The **Hawkes Process** result showing a super-critical, self-exciting process of LOP spikes during the crisis is a direct measure of the instability the Act seeks to curb.
- The **Random Forest** result showing price impact metrics (Lambda, Amihud) were the best predictors of the crisis highlights the importance of monitoring market liquidity, something regulators would do under the new framework.

### Identified Gaps & Proposed Models

This question is more about interpretation than new models, but one model can directly simulate the *potential future impact* of the regulation.

| Gap | Proposed Model/Analysis | Justification |
|---|---|---|
| **1. Simulating Post-Regulation Impact** | **Agent-Based Model (ABM) or Counterfactual Simulation**: Build a simple ABM with two types of arbitrageurs: one trading between BTC/USD and BTC/USDT (high perceived risk), and one trading between BTC/USD and a hypothetical "GENIUS-compliant" BTC/USDC (low perceived risk). Simulate a market shock and compare the magnitude and persistence of the LOP deviation in both systems. | This is a powerful way to answer the "what if" question posed by the regulatory change. It moves from interpreting past data to creating a forward-looking simulation based on the principles of the GENIUS Act. It would provide a quantitative estimate of how much the regulation could improve market efficiency and reduce the basis. |

---

## 6. Summary & Final Recommendation

Our existing analysis is comprehensive, but the research synthesis reveals clear opportunities to add significant depth and directly address the nuances of the competition questions. The highest-impact additions would be:

1.  **Triangular Arbitrage Simulation**: To directly address the role of transaction costs.
2.  **DCC-GARCH**: To formally model contagion between stablecoins.
3.  **Agent-Based Model (ABM)**: To simulate the future impact of the GENIUS Act.

These three models are not redundant; they are targeted, informative, and demonstrate a sophisticated understanding of the underlying market mechanics and regulatory context. I will proceed with implementing these three analyses in a new, final notebook.

---

## References

[1] Makarov, I., & Schoar, A. (2020). Trading and arbitrage in cryptocurrency markets. *Journal of Financial Economics, 135*(2), 293-319.

[2] Kroeger, L., & Sarkar, A. (2017). The Law of One Bitcoin Price? *Federal Reserve Bank of Philadelphia Working Paper No. 17-23*.

[3] Alexander, C., & Imeraj, A. (2022). Fragmentation, price formation and cross-impact in Bitcoin markets. *Applied Mathematical Finance, 29*(3), 191-220.

[4] Watsky, C., Allen, J., et al. (2024, February 23). Primary and Secondary Markets for Stablecoins. *FEDS Notes, Board of Governors of the Federal Reserve System*.

[5] Du, C. (2025, December 17). In the Shadow of Bank Runs: Lessons from the Silicon Valley Bank Failure and Its Impact on Stablecoins. *FEDS Notes, Board of Governors of the Federal Reserve System*.

[6] Kaiko Research. (2024, August 12). How is crypto liquidity fragmentation impacting markets? *Kaiko Insights*.

[7] Angerer, M., et al. (2025). Order Book Liquidity on Crypto Exchanges. *Journal of Risk and Financial Management, 18*(3), 124.

[8] The White House. (2025, July 18). Fact Sheet: President Donald J. Trump Signs GENIUS Act into Law. *Fact Sheets*.

[9] McKinsey & Company. (2026, February 18). Stablecoins in payments: What the raw transaction numbers miss. *McKinsey Insights*.
