# Research findings — August 2026

First full sweep of the strategy research framework. **Every strategy was rejected.**

Reproduce with:

```bash
quantflow research run --symbols "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT" \
  --timeframe 1h --start 2021-01-01 --end 2026-08-04T07:00:00+00:00 --costs realistic
```

Report artefacts land in `reports/` (gitignored — regenerable). This file records the
conclusions, which are not.

---

## Setup

| | |
|---|---|
| Period | 2021-01-01 → 2026-08-04 (5.6 years) |
| Symbols | BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT |
| Bars | 195,876 real hourly bars (~48,970 per symbol) |
| Costs | Binance base tier: 0.10% taker per fill, 0.20% per round trip, volume-scaled slippage, fills at the next bar's open |
| Strategies | 13 candidates + buy-and-hold benchmark |
| Starting equity | 10,000 USDT per strategy per symbol |

---

## Leaderboard

| # | Strategy | Net return | PF | Sharpe | Max DD | Win rate | Trades |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | dual_thrust | 5.38% | 1.11 | 0.36 | 4.6% | 40.1% | 1,442 |
| 2 | momentum_roc | 16.22% | 1.10 | 0.42 | 10.7% | 36.2% | 1,997 |
| **3** | **buy_and_hold** *(benchmark)* | **1621.68%** | no losses | **0.83** | 82.0% | 100% | 4 |
| 4 | triple_ma | 8.99% | 1.04 | 0.18 | 10.3% | 32.8% | 2,687 |
| 5 | keltner_trend | 0.90% | 1.01 | 0.04 | 9.7% | 35.3% | 2,688 |
| 6 | volume_breakout | 0.12% | 1.00 | 0.01 | 7.6% | 35.8% | 1,796 |
| 7 | ema_cross | 2.27% | 1.01 | 0.08 | 8.5% | 34.7% | 1,200 |
| 8 | rsi_reversion | −1.57% | 0.83 | −0.36 | 2.7% | 55.8% | 208 |
| 9 | opening_range_breakout | −6.88% | 0.89 | −0.50 | 15.1% | 40.0% | 2,933 |
| 10 | donchian_breakout | −1.03% | 0.98 | −0.05 | 13.8% | 29.1% | 2,299 |
| 11 | bollinger_reversion | −8.31% | 0.79 | −0.88 | 9.4% | 54.9% | 1,725 |
| 12 | macd_trend | −7.63% | 0.90 | −0.43 | 15.1% | 32.7% | 2,933 |
| 13 | bollinger_squeeze | −1.90% | 0.73 | −0.52 | 3.7% | 25.4% | 364 |
| 14 | zscore_reversion | −14.49% | 0.74 | −1.07 | 15.2% | 42.7% | 1,465 |

Buy-and-hold per symbol: BTC +119.66%, ETH +152.49%, SOL +4743.88%, BNB +1470.68%.

---

## Findings

**1. Nothing beat holding, on any symbol.** The best candidate returned 16.22% against a
benchmark of 1621.68%. Buy-and-hold also had the highest Sharpe (0.83) of anything tested;
the best strategy Sharpe was 0.42, below the 0.50 floor. This is the whole result.

**2. Fees consumed the edge.** Several strategies were gross-profitable and net-negative —
the venue was the only winner:

| Strategy / symbol | Fees as share of gross profit |
|---|---:|
| triple_ma on BTC | 226% |
| dual_thrust on BTC | 133% |
| momentum_roc on BTC | 106% |
| dual_thrust on BNB | 57% |
| momentum_roc on BNB | 53% |

At 1,000–3,000 trades over the period, 0.20% per round trip is decisive. This is the most
actionable finding: it points at **longer holding periods or maker-only execution**, not at
parameter tuning.

**3. Profit factors cluster at 1.0.** Ten of thirteen sit between 0.90 and 1.13 — the
signature of no edge once costs are paid, rather than of a mis-tuned edge.

**4. The benchmark also fails the gate, and that matters.** Holding meant drawdowns of 77%
(BTC), 81% (ETH), **96.8%** (SOL) and 73% (BNB) — far past the 35% ceiling applied to every
candidate. "Just hold" wins on return while being unholdable by the same risk standard. It
is reported for context, not as a candidate.

---

## Two defects this sweep exposed

**The benchmark was silently broken and would have inverted the conclusion.** Run through
the trading engine, buy-and-hold is not buy-and-hold: the risk engine correctly refuses
naked entries and attaches the default 2% stop, which on BTC produced **27 trades, a 0% win
rate and a negative return over a period in which the asset rose**. Measured against that
fake benchmark, several strategies showed "+31% vs hold" and would have read as candidates.
Buy-and-hold is now computed from the price series (`research/benchmark.py`) while still
paying fees and slippage on both legs.

**Profit factor rendered as 162,203.** With no losing trades the ratio is undefined;
`compute_metrics` reports gross profit so the value serialises and sorts, but printing it
under a column headed "Profit factor" reads as a broken calculation. Now renders
"no losses" — and the regression test written for it caught a second occurrence in the
per-symbol table.

---

## What this does not establish

These are **in-sample** results, at **default parameters**, on **one timeframe**, over
**one historical period** that contains exactly one full crypto cycle.

It does not prove no edge exists. It proves these thirteen do not, and that any future
candidate has to clear a bar that costs two fills to achieve.

## Next

1. Re-run at 4h and 1d. If fee drag is the binding constraint, longer holding periods
   should improve the ranking materially; if they do not, the problem is the signals.
2. Maker-only execution — limit entries rather than market orders — to test the same
   hypothesis from the cost side.
3. Walk-forward validation (`backtest/walkforward.py`) for anything that survives.
4. Do not tune these thirteen on this data. Parameters fitted to the period that rejected
   them would be curve-fitting with extra steps.
