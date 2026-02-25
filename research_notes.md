# IAQF Research Notes — Literature & Regulatory Review

## Q1: Cross-Currency Basis / LOP

### Key Papers
- **Makarov & Schoar (2020, JFE)** — "Trading and Arbitrage in Cryptocurrency Markets": Documents persistent LOP violations across exchanges. Key finding: deviations are larger across geographic regions (Korea premium) than within regions. Arbitrage is limited by capital flow frictions, not just transaction costs. Deviations can persist for hours to days.
- **Kroeger & Sarkar (2017, Philadelphia Fed)** — "The Law of One Bitcoin Price?": Studies 6 exchanges, 15 pairs. Finds significant price differences driven by exchange-specific frictions (withdrawal limits, KYC, banking access).
- **Alexander & Imeraj (2022, Applied Mathematical Finance)** — "Fragmentation, Price Formation and Cross-Impact in Bitcoin Markets": Directly addresses BTC/USDT vs BTC/USD. Key finding: the BTC/USDT–BTC/USD price difference has TWO components: (1) the USDT/USD FX rate deviation, and (2) a residual "basis" from order flow fragmentation. The residual is mean-reverting but can be large during stress.
- **John, Li, Liu (2024, SSRN)** — "Pricing and Arbitrage across 80 Cryptocurrency Exchanges": Studies BTC/USDT, BTC/USDC, BTC/USD across 80 exchanges. Finds stablecoin arbitrage spreads are time-varying and regime-dependent.
- **Ranaldo, Viswanath-Natraj et al. (2024, Swiss Finance Institute)** — "Blockchain Currency Markets": Decomposes DEX order flow; finds only the residual component (not the FX component) exhibits lasting price impact.

### Key Mechanisms Driving Cross-Currency Basis
1. **Stablecoin FX deviation** (USDT/USD ≠ 1 or USDC/USD ≠ 1): Directly translates to BTC/USDT vs BTC/USD difference. This is the mechanical component.
2. **Order flow fragmentation**: Different trader populations on USDT vs USD markets create independent price pressure. Arbitrageurs bridge the gap but face latency and capital constraints.
3. **Limits to arbitrage**: Transfer delays between exchanges (blockchain confirmation times), capital requirements, counterparty risk, and banking access constraints prevent instantaneous arbitrage.
4. **Liquidity asymmetry**: USDT markets are far more liquid globally (especially on Asian exchanges). During stress, liquidity withdrawal from USDC markets amplifies deviations.
5. **Transaction costs**: Taker fees (~0.1%), withdrawal fees, and blockchain gas fees create a no-arbitrage band. Deviations within this band are not exploitable.

### What Our Code Covers (Q1)
- ✓ Log LOP deviations computed (lop_bnus_usdt_vs_usd, lop_bnus_usdc_vs_usd)
- ✓ FX vs residual decomposition (stablecoin_fx_component, lop_residual)
- ✓ Regime comparison (Kruskal-Wallis, Mann-Whitney)
- ✓ Cointegration tests (Engle-Granger)
- ✓ VAR + IRF (propagation of USDC shock to LOP)
- ✓ Markov-Switching (regime identification in LOP series)
- ✓ Hawkes Process (self-exciting LOP spikes)
- ✗ MISSING: Transaction cost-adjusted no-arbitrage band analysis
- ✗ MISSING: Speed of mean-reversion (half-life) by regime
- ✗ MISSING: Triangular arbitrage profitability analysis (BTC/USDT → USDT/USD → USD)
- ✗ MISSING: Rolling cointegration to show structural break at crisis

---

## Q2: Stablecoin Dynamics

### Key Events — March 2023 USDC De-peg
- March 8: Silvergate Bank announces liquidation
- March 10: SVB enters FDIC receivership; Circle announces $3.3B of USDC reserves (~8% of total $40B) are stuck at SVB
- March 10 evening: USDC begins trading below $1 on secondary markets
- March 11 (Saturday): USDC falls to $0.863 low; Coinbase suspends USDC↔USD conversions (weekend banking closure)
- March 12 (Sunday): US Treasury, FDIC, Fed announce SVB depositors will be made whole
- March 13 (Monday): USDC begins recovering; Binance lists BTCUSDC (new pair, born into crisis)
- March 21: USDC fully re-pegged near $1.000

### Key Papers
- **Federal Reserve FEDS Note (Watsky et al., Feb 2024)** — "Primary and Secondary Markets for Stablecoins": 
  - Primary market (institutional redemption at $1) vs secondary market (exchange trading) distinction is critical
  - During the crisis, Coinbase suspended USDC↔USD conversions, breaking the primary market arbitrage channel
  - This is why the de-peg was so severe: the arbitrage mechanism that normally keeps USDC at $1 was disabled
  - USDT actually APPRECIATED during the crisis (flight to USDT as perceived safer alternative)
  - DAI (crypto-collateralized) also de-pegged due to USDC collateral exposure
  
- **Federal Reserve FEDS Note (Du, Dec 2025)** — "In the Shadow of Bank Runs": Provides minute-level account of the crisis. Key finding: USDC de-peg was driven by a coordination failure — rational holders sold USDC even if they believed SVB deposits would be recovered, because they feared others would sell first.

- **Ma, Zeng et al. (2025, BFI/Wharton)** — "Stablecoin Runs and the Centralization of Arbitrage": Shows that stablecoin arbitrage is highly concentrated among a small number of institutional arbitrageurs. When these arbitrageurs face capital constraints or uncertainty, the arbitrage mechanism fails and de-pegs can be severe and prolonged.

- **Joshi (2025)** — "Arbitrage Effectiveness and Stablecoin Run": Arbitrage fails when frictions overwhelm participation and fundamentals, making price convergence impossible. Documents that the USDC de-peg was a case of arbitrage failure, not just slow arbitrage.

### USDT vs USDC Structural Differences
- **USDT (Tether)**: Backed by commercial paper, T-bills, and other assets. Issuer (Tether Ltd.) is offshore (BVI). Less transparent reserves. Dominant in Asian markets. Market cap ~$80B in March 2023.
- **USDC (Circle)**: Backed by cash and short-term US Treasuries. Issuer (Circle) is US-based, regulated. More transparent reserves. Dominant in US/DeFi markets. Market cap ~$40B in March 2023.
- **Key asymmetry**: USDT's offshore structure means it is less exposed to US bank failures but more exposed to regulatory risk. USDC's US-regulated structure means it is more exposed to US bank failures but less exposed to regulatory crackdown.
- **During crisis**: USDT actually appreciated slightly (premium to $1) as traders fled USDC. This is the "flight to USDT" phenomenon — counterintuitive given USDT's historically more opaque reserves.

### What Our Code Covers (Q2)
- ✓ USDC/USD and USDT/USD time series
- ✓ De-peg magnitude and timing
- ✓ Volume shifts between USDT and USDC markets
- ✓ Regime-based comparison
- ✓ Granger causality (USDC/USD ↔ LOP USDC)
- ✗ MISSING: USDT premium/discount analysis (USDT actually went above $1 during crisis)
- ✗ MISSING: Speed of re-peg (half-life of USDC recovery)
- ✗ MISSING: Contagion analysis (did USDC de-peg affect USDT, DAI, other stablecoins?)
- ✗ MISSING: Primary vs secondary market arbitrage channel analysis (Coinbase suspension effect)

---

## Q3: Liquidity & Fragmentation (to be researched)

---

## Q4: Regulatory Overlay / GENIUS Act (to be researched)

