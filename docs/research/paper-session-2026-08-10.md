# Paper-session log — 2026-08-10 (single strategy, mostly replayed bars)

> ## This is not a performance claim
>
> **This document is a log of one simulated session on one strategy. It is not evidence
> that QuantFlow is profitable, and it must not be read, quoted or excerpted as such.**
>
> - **No real money was traded.** Execution was simulated throughout; live trading was
>   disabled.
> - **Most of it is not even live paper trading.** The majority of these trades come from
>   *historical daily bars replayed through the live engine*. The realtime portion added
>   roughly one trade per fortnight per symbol.
> - **It is a single configuration** — one strategy, one timeframe — selected after the
>   fact from a library of 44. Reporting the result of one arm of a many-armed search is
>   how a random walk is made to look like a skill.
> - **It was never repeated, and it has no out-of-sample holdout.** No walk-forward, no
>   confidence interval, n=1.
> - **Simulated fills flatter the result.** Real execution adds queue position, partial
>   fills, slippage, funding and outages. Every one of them works against the trader.
> - **The broader research contradicts the optimistic reading.** In the systematic sweep,
>   **no strategy beat simply holding the asset.** See
>   [`strategy-research-2026-08.md`](strategy-research-2026-08.md) — that is the primary
>   research document, and this one is a footnote to it.
>
> **Simulated and past results do not indicate, predict or guarantee future performance.**
> See the risk disclaimer in the [README](../../README.md).

---

## What was run

| | |
|---|---|
| Date | 2026-08-10 14:09 UTC |
| Session | `paper-live` |
| Strategy | `volume_breakout` (1 of 44) |
| Timeframe | 1d |
| Market data | live Bybit V5 |
| Execution | **simulated** |
| Live trading | **disabled** |
| Trade source | **majority replayed historical daily bars**, remainder realtime |

## What the simulation produced

Recorded for completeness. **Read the caveats above before reading this table**, and note
that the row that actually matters is the last one.

| Metric | Simulated value |
|---|---:|
| Starting equity | 10,000.00 USDT |
| Ending equity | 14,311.68 USDT |
| Change over the session | +43.12% |
| Closed trades | 123 |
| Win rate | 47.2% (58 wins / 65 losses) |
| Profit factor | 1.56 |
| Max drawdown | 11.40% |
| Average trade duration | 219.9 h (9.2 days) |
| Largest winning trade | +292.62 |
| Largest losing trade | −158.67 |
| Total fees | 263.03 |
| **Beat buy-and-hold?** | **No.** |

## Reading the numbers honestly

**It did not beat holding the asset.** Over the same window, buy-and-hold returned more.
This configuration's only advantage was drawdown — 11.4% against 85.9% for buy-and-hold —
and a drawdown advantage on a single unrepeated sample is not an edge, it is one
observation.

Most trades lost. The win rate is 47.2%; the simulated account grew because the average
win (207.69) exceeded the average loss (118.99), not because losses were rare. That
distribution is fragile: it depends on a small number of large winners, and a strategy
that depends on tail wins is exactly the kind whose backtest most overstates its live
behaviour.

Fees consumed 2.2% of gross profit — under simulated fills. Real fills would consume more.

## Why this file is kept

Because deleting an inconvenient result and keeping a convenient one is how a research
record becomes marketing. It is kept as a record of what a single simulated session
produced, with its limitations stated at the top rather than buried at the bottom.

**For the actual research finding, read
[`strategy-research-2026-08.md`](strategy-research-2026-08.md): no strategy passed, and
buy-and-hold beat all thirteen tested on every symbol.**
