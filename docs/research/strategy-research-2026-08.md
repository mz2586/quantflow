# QuantFlow — Strategy Research Report

**Generated:** 2026-08-10 · **Commit:** `e9d3222`
**Verdict: no strategy passed. Buy-and-hold beat all thirteen on every symbol.**

---

## 0. Scope, stated plainly

Two independent runs, both on real Binance data. Their scopes differ and the differences
matter when reading the numbers.

| | Run A — Research sweep | Run B — Laboratory |
|---|---|---|
| Purpose | Rank against acceptance thresholds | Diagnose *why*, and per regime |
| Symbols | BTC, ETH, SOL, BNB (4) | BTC, SOL (2) |
| Period | 2021-01-01 → 2026-08-04 (5.6 y) | Final 20,000 bars (~2.3 y) |
| Bars | 195,876 | 40,000 |
| Cost models | Realistic | Realistic **and** zero-cost |
| Starting equity | 10,000 USDT per strategy per symbol | Same |

**Only one timeframe was tested: 1h.** The platform supports twelve
(1m/3m/5m/15m/30m/1h/2h/4h/6h/12h/1d/1w) and none of the other eleven has been run. Any
statement below about timeframe is a *hypothesis*, not a result. This is the single
largest gap in the evidence, and section 12 explains why it is also the most promising
place to look next.

Run B is smaller because the full-scale laboratory run needs two full sweeps and did not
fit in the memory available on the machine it was run on: the worker pool could not be
given the whole dataset. That is a hardware constraint, recorded
rather than worked around.

---

## 1. Strategies tested — 14

| Family | Strategies |
|---|---|
| Benchmark | `buy_and_hold` |
| Trend | `ema_cross`, `macd_trend`, `triple_ma`, `keltner_trend` |
| Breakout | `donchian_breakout`, `bollinger_squeeze`, `dual_thrust`, `opening_range_breakout` |
| Mean reversion | `rsi_reversion`, `bollinger_reversion`, `zscore_reversion` |
| Momentum | `momentum_roc` |
| Volume | `volume_breakout` |

All at default parameters. No optimisation was performed, deliberately — see section 13.

## 2. Timeframes tested — 1

`1h`. Nothing else. See section 0.

## 3. Symbols tested — 4

BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT — 48,960–48,978 hourly bars each. Fourteen bars are
missing across the whole set (0.03%), identical on all four symbols, which identifies them
as real Binance outages rather than fetch failures.

---

## 4–9. Results (Run A: 4 symbols, 5.6 years, realistic costs)

Costs: 0.10% taker per fill (0.20% per round trip), volume-scaled slippage, market orders
filled at the next bar's open. No BNB discount, no VIP tier.

| # | Strategy | Net return | Profit factor | Sharpe | Max DD | Win rate | Trades | Total fees |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | dual_thrust | +5.37% | 1.11 | 0.35 | 4.62% | 40.1% | 1,442 | 3,045 |
| 2 | momentum_roc | +16.21% | 1.10 | 0.42 | 10.74% | 36.2% | 1,997 | 4,683 |
| **3** | **buy_and_hold** *(benchmark)* | **+1621.68%** | no losses | **0.83** | 81.97% | 100% | 4 | **729** |
| 4 | triple_ma | +8.99% | 1.04 | 0.18 | 10.29% | 32.8% | 2,687 | 5,939 |
| 5 | keltner_trend | +0.88% | 1.00 | 0.03 | 9.68% | 35.3% | 2,689 | 5,647 |
| 6 | volume_breakout | +0.12% | 1.00 | 0.01 | 7.62% | 35.8% | 1,796 | 3,711 |
| 7 | ema_cross | +2.26% | 1.01 | 0.08 | 8.47% | 34.7% | 1,200 | 2,573 |
| 8 | rsi_reversion | −1.57% | 0.83 | −0.36 | 2.75% | 55.8% | 208 | 403 |
| 9 | opening_range_breakout | −6.88% | 0.89 | −0.50 | 15.09% | 40.0% | 2,933 | 5,993 |
| 10 | donchian_breakout | −1.04% | 0.98 | −0.05 | 13.82% | 29.1% | 2,299 | 4,839 |
| 11 | bollinger_reversion | −8.31% | 0.79 | −0.88 | 9.43% | 54.9% | 1,725 | 3,288 |
| 12 | macd_trend | −7.63% | 0.90 | −0.43 | 15.12% | 32.7% | 2,933 | 5,890 |
| 13 | bollinger_squeeze | −1.90% | 0.73 | −0.52 | 3.73% | 25.4% | 364 | 721 |
| 14 | zscore_reversion | −14.49% | 0.74 | −1.07 | 15.21% | 42.7% | 1,465 | 2,691 |

Fees are the sum across four symbols on 40,000 USDT of total starting capital.
`buy_and_hold`'s profit factor is undefined — it had no losing trades — and is reported as
such rather than as a large number.

---

## 10. Buy-and-hold comparison

Per symbol, paying the **same** fees and slippage on both legs:

| Symbol | Buy-and-hold return | Max drawdown |
|---|---:|---:|
| BTC/USDT | +119.66% | 77.20% |
| ETH/USDT | +152.49% | 81.34% |
| SOL/USDT | **+4,743.88%** | **96.80%** |
| BNB/USDT | +1,470.68% | 72.54% |
| **Mean** | **+1,621.68%** | **81.97%** |

**Every strategy underperformed holding by 1,605–1,636 percentage points.** The best
active strategy returned 16.21% against 1,621.68%. Buy-and-hold also had the **highest
Sharpe of anything tested** (0.83 against a best-active 0.42), on **four trades** and 729
USDT of fees against 2,573–5,993.

The benchmark is computed from the price series rather than traded through the engine.
This was forced by a measured failure: run through the engine, the risk engine correctly
refuses naked entries and attaches the default 2% stop, which on BTC produced **27 trades,
a 0% win rate and a negative return over a period in which the asset rose**. Against that
broken benchmark several strategies showed "+31% vs hold" and would have read as
candidates. Fixing it inverted the conclusion of this entire report.

**The benchmark also fails the acceptance gate**, on drawdown (77–97% against a 35%
ceiling) and trade count. "Just hold" wins on return while being unholdable by the same
risk standard applied to every candidate. That is a genuine finding, not a scoring
artefact, and it is the reason this report does not end with "so buy and hold".

---

## 11. Why each failed

Run B ran every strategy twice — once priced, once with **zero** fees and slippage. That
comparison is the only way to separate "the signal is worthless" from "the signal works
and the venue took it", and the two demand opposite responses.

Distribution of primary causes: **6 no_signal, 6 over_trading, 2 insufficient_sample.**

| Strategy | Net (priced) | Net (frictionless) | Trades | Cause |
|---|---:|---:|---:|---|
| macd_trend | −13.38% | **+2.28%** | 1,387 | over_trading |
| opening_range_breakout | −14.88% | −2.41% | 1,293 | over_trading |
| zscore_reversion | −13.00% | −4.86% | 781 | no_signal |
| donchian_breakout | −6.07% | **+1.20%** | 640 | over_trading |
| triple_ma | −4.04% | +2.19% | 524 | no_signal |
| keltner_trend | −7.01% | −0.80% | 535 | no_signal |
| momentum_roc | −7.43% | −3.02% | 437 | no_signal |
| volume_breakout | −4.04% | −0.10% | 353 | no_signal |
| bollinger_reversion | −3.89% | −0.32% | 316 | no_signal |
| dual_thrust | −1.82% | **+1.53%** | 301 | over_trading |
| ema_cross | −2.20% | +0.22% | 224 | over_trading |
| bollinger_squeeze | −0.42% | +0.41% | 71 | over_trading |
| rsi_reversion | −0.30% | +0.27% | 49 | insufficient_sample |
| buy_and_hold | −27.24% | −27.06% | 2 | insufficient_sample |

Read against Run A's thresholds, the specific breaches were:

- **Lost to the benchmark** — all 13. Universal and decisive.
- **Fee-dominated** — `triple_ma` on BTC paid **226%** of gross profit in fees;
  `dual_thrust` 133%; `momentum_roc` 106%. Gross-profitable, net-negative.
- **Profit factor below 1.30** — ten of thirteen sit between 0.90 and 1.13. That cluster
  around 1.0 is the signature of no edge after costs, not of a mis-tuned edge.
- **Sharpe below 0.50** — all thirteen. Best was 0.42.
- **Insufficient sample** — `rsi_reversion` (49 trades in Run B), `bollinger_squeeze` (71).

Note the disagreement between runs: `momentum_roc` and `triple_ma` are positive in Run A
and negative in Run B. Same strategies, different period and symbol set. That instability
is itself evidence — a result that flips sign when the window moves is not an edge.

---

## 12. What deserves further research

**1. Slower timeframes — the strongest lead by a wide margin.**
Four strategies made money with costs removed and lost with them applied:
`macd_trend` (+2.28% → −13.38%), `dual_thrust` (+1.53% → −1.82%), `donchian_breakout`
(+1.20% → −6.07%), `ema_cross` (+0.22% → −2.20%). These are not bad ideas; they are ideas
whose per-trade edge is smaller than the cost of taking it. At 1,000–3,000 trades, 0.20%
per round trip decides the outcome. The direct test is 4h and 1d, which is cheap and has
not been run.

**2. Regime gating.** Six of fourteen are provably regime-dependent — profitable in some
conditions, loss-making in others, on reliable samples:

| Strategy | Works in | Loses in |
|---|---|---|
| macd_trend | sideways/ranging/high | ranging/low, ranging/normal |
| opening_range_breakout | ranging/high, bull/trending/normal | ranging/low, ranging/normal |
| donchian_breakout | sideways/ranging/low | ranging/high, ranging/normal |
| dual_thrust | sideways/ranging/low | ranging/normal |
| ema_cross | sideways/ranging/high | ranging/low, ranging/normal |
| rsi_reversion | sideways/ranging/low | ranging/normal |

Eight distinct regimes were observed. A blended average describes none of them. `macd_trend` made
+90.21 across 73 trades in `sideways/ranging/high` while losing overall — that is a
gating problem, not a signal problem.

**3. Maker-only execution.** Tests the same cost hypothesis from the other side. Untested.

**4. Anything with a low trade count.** `rsi_reversion` (208 trades, 55.8% win rate, the
best win rate of any strategy, only 403 USDT of fees) and `bollinger_squeeze` (364 trades)
are the only two that did not trade themselves to death. Both were rejected for
insufficient sample rather than for being wrong.

---

## 13. Hypotheses to discard

**High-frequency signal trading on 1h crypto, at retail fees.** Ten profit factors
between 0.90 and 1.13 across six unrelated families is not thirteen coincidences; it is
the absence of an exploitable edge at this frequency and cost. Stop looking here.

**That any of these six is worth optimising on this data.** `zscore_reversion`,
`keltner_trend`, `momentum_roc`, `volume_breakout`, `bollinger_reversion`, `triple_ma` all
lost money at **zero cost**. No execution improvement rescues a signal that loses money
for free. Parameters fitted to the period that rejected them would be curve-fitting with
extra steps.

**That volume carries usable information here.** `volume_breakout` was included
specifically to test that. It returned −0.10% frictionless — indistinguishable from zero
before a single fee was charged.

**That mean reversion works on 1h crypto.** All three implementations failed, two of them
with no signal at all. `bollinger_reversion` had a 54.9% win rate and still lost 8.31%:
it won often and small, lost rarely and large.

**That "diversifying" across these four symbols reduces risk.** Measured correlations are
BTC–ETH +0.87, BTC–SOL +0.84, ETH–SOL +0.88, BNB against the others +0.71 to +0.75. Four
positions here are close to one position at four times the size. (This is also the defect
that made the correlation risk rule silently inert until it was fixed — position-aligned
returns reported BTC–SOL as +0.02.)

**That the platform is the constraint.** It is not. The engineering is sound and the
measurement is trustworthy; the strategies are not profitable. That distinction is the
report's main conclusion.

---

## 14. Standing constraints

- **Live trading is disabled and has never been armed.** No authenticated order has ever
  been sent. The interlock requires five simultaneous conditions and is tested to refuse
  when any single one is absent.
- **Nothing here justifies capital.** These are in-sample results at default parameters on
  one timeframe over one period containing exactly one crypto cycle. Passing the gate — which
  nothing did — would make a strategy a candidate for walk-forward validation, not for money.
- **Reproduce Run A:**
  `quantflow research run --symbols "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT" --timeframe 1h --start 2021-01-01 --end 2026-08-04T07:00:00+00:00 --costs realistic`
  Pass `--end` explicitly; live ingestion keeps extending the dataset, so a run without it
  is not reproducible.
