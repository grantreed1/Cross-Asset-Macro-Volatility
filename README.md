# Cross-Asset-Macro-Volatility

## Harvesting the Volatility Risk Premium
We implement a **cross-asset relative value volatility strategy** designed to harvest the Volatility Risk Premium (VRP) or the structural spread between implied and realized volatility. The strategy operates across a diversified macroeconomic universe, including Equities, Gold, Oil, FX, and High-Yield Credit

## Core Methodology

### 1. Signal Generation
The core signal utilizes a rolling Z-score to identify cross-asset relative-value pairs. By systematically shorting historically overpriced implied volatility and funding underpriced protection, the strategy isolates the pure VRP spread.

### 2. Options P&L Modeling
To isolate volatility returns from underlying price movements without requiring tick-level options data, daily P&L is simulated using a second-order Taylor Series expansion of the Black-Scholes model:

$$dP \approx \Delta dS + \frac{1}{2} \Gamma (dS)^2 + \mathcal{V} d\sigma + \Theta dt$$

This attribution model effectively neutralizes delta exposure while capturing the non-linear gamma effects and primary vega exposure.

### 3. Risk Management & Sizing
An **inverse-volatility weighting framework** is applied to equalize risk contributions across the disparate asset classes (e.g., balancing high-volatility Oil against lower-volatility FX). Compared to an equal-weight baseline, this dynamic scaling drastically improves risk-adjusted returns and minimizes drawdown depth.

## Key Findings & Performance (2015–2025)
The cross-asset VRP approach exhibited significant outperformance over traditional equity and passive volatility benchmarks.

* **Superior Risk-Adjusted Returns:** The strategy achieved a **1.97 Sharpe Ratio**, significantly outperforming the S&P 500 (0.72) and passive short-volatility instruments.
* **Low Correlation & Tail Resilience:** Acting as a powerful portfolio diversifier, the strategy maintained a Beta of **0.077** and a correlation of just **0.113** to the S&P 500.
* **Drawdown Mitigation:** Unlike passive short-volatility ETFs (e.g., SVXY) which suffer catastrophic tail risk, the integrated long-leg protection constrained the maximum drawdown to **-11.79%** during major market shocks.

## Capacity & Execution Constraints
Designed with institutional viability in mind, the strategy operates within strict liquidity parameters:
* **Target AUM:** $10M – $100M
* **Execution:** Limits daily turnover to strictly **< 1% of Average Daily Volume (ADV)** across the underlying highly liquid, large-cap ETFs and their respective ATM options markets.
