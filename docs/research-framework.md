# Strategy Research Framework

A framework for deciding, on evidence and before any money moves, whether a strategy is
worth pursuing. Its primary job is to **reject**. Finding a strategy that works is rare;
the common and far more valuable outcome is establishing that one does not.

```
quantflow research thresholds     # the standard a strategy must clear
quantflow research run            # backtest everything, rank the survivors
```

---

## 1. The strategy interface

Already the platform's contract — the framework did not need a new one.

A strategy is a **pure decision function**: bars and portfolio state in, a signal out. It
cannot place orders, cannot size positions and cannot reach the exchange. That is not
style; it is what makes it structurally impossible for a strategy to route around the risk
engine, and it is why the same object runs unmodified in backtest, paper and live.

```python
class Strategy(ABC):
    strategy_id: ClassVar[str]
    description: ClassVar[str]
    params_model: ClassVar[type[StrategyParams]]   # Pydantic, frozen, extra="forbid"

    @property
    @abstractmethod
    def warmup_bars(self) -> int: ...

    @abstractmethod
    def generate(self, context: StrategyContext) -> Signal: ...
```

Enforced by the base class, not by convention:

| Guarantee | How |
|---|---|
| No look-ahead | `StrategyContext.history` holds only closed bars up to the decision bar. There is no field containing a future price. |
| No half-formed indicators | The engine withholds the strategy entirely until `warmup_bars` closed bars exist. |
| Determinism | `generate` must be pure. `dual_thrust` derives its holding period from `position.opened_at` rather than an instance counter for exactly this reason. |
| Containment | A strategy that raises produces a HOLD, not an engine crash — one bad strategy must not abandon positions held by others. |
| Honest attribution | A signal whose `strategy_id` does not match the emitter is rejected. |
| Typo-proof parameters | `extra="forbid"` — a misspelled parameter raises instead of silently using the default. |

---

## 2. The strategy library

Fourteen strategies across six families. Breadth is the point: a leaderboard populated
with variations on one idea ranks parameter choices while appearing to rank ideas.

| Family | Strategies |
|---|---|
| Benchmark | `buy_and_hold` |
| Trend | `ema_cross`, `macd_trend`, `triple_ma`, `keltner_trend` |
| Breakout | `donchian_breakout`, `bollinger_squeeze`, `dual_thrust`, `opening_range_breakout` |
| Mean reversion | `rsi_reversion`, `bollinger_reversion`, `zscore_reversion` |
| Momentum | `momentum_roc` |
| Volume | `volume_breakout` |

Every one attaches ATR-scaled protective levels through a shared helper
(`strategy/library/_protection.py`). If each placed its stop slightly differently, that
difference would show up in the leaderboard as though it were a difference in the *idea* —
precisely the confound the framework exists to remove.

---

## 3. Costs

Two ways a backtest lies: omitting fees, and assuming a fill at the price that triggered
the decision. Both are corrected, and made explicit rather than buried in an engine
default.

| Preset | Assumptions |
|---|---|
| `realistic` (default) | Binance spot base tier: 0.10% taker per fill, 0.20% per round trip. No BNB discount, no VIP tier — both are privileges that can be withdrawn. Volume-scaled slippage. |
| `pessimistic` | Double fees plus a fixed 10 bp slippage floor. A robustness check, not a forecast. |
| `zero_cost` | Diagnostic only, to quantify how much of an edge costs consume. Never a basis for a decision. |

Slippage scales with the share of a bar's volume an order consumes, so a strategy is
penalised for wanting size the market could not have supplied. A flat basis-point
assumption would let a strategy trade unlimited quantity at fixed cost — the most common
way a backtest flatters an illiquid idea.

Market orders fill at the **next bar's open**, never at the close that triggered them.

---

## 4. The benchmark

`buy_and_hold` is computed directly from the price series (`research/benchmark.py`), not
traded through the engine.

This was not a shortcut — it was forced by a measured failure. Run through the engine, the
risk engine correctly refuses to let any entry go out unprotected and attaches the default
2% stop. On BTC that stop is hit within days, the position re-enters, and the result was
**27 trades, a 0% win rate and a negative return over a period in which the asset rose**. A
benchmark that broken makes every strategy look good.

Holding an asset is a property of the market, not of a trading system: no stop, no
re-entry, no risk engine, because a person holding an asset has none of those. It still
pays full entry and exit cost — fees and slippage on both legs — so the comparison stays
honest. Marked to market every bar, so its Sharpe and drawdown describe the real
experience of holding, including the part where it falls 70%.

---

## 5. Acceptance thresholds

Fixed in advance, applied mechanically. Once results are on screen it is trivially easy to
justify a strategy that missed a bar — "the drawdown was one bad month", "the sample is
short but the Sharpe is good" — and that is how a losing strategy reaches production.

| Criterion | Requirement | Why |
|---|---|---|
| Net return | ≥ 10% | After costs, over the whole period. |
| Profit factor | ≥ 1.30 | Below this there is no margin for the live-vs-backtest gap, and that gap is always worse than expected. |
| Sharpe | ≥ 0.50 | Below this the curve is indistinguishable from luck at any obtainable sample size. |
| Max drawdown | ≤ 35% | A drawdown nobody can sit through is not a strategy. |
| Win rate | ≥ 25% | Set low deliberately — trend systems win rarely and win big. |
| Trades | 30–5,000 | Too few is luck; too many is a fee-generation machine. |
| Fees / gross profit | ≤ 50% | Past this the venue takes most of the edge. |
| Beats buy-and-hold | Required | Taking on execution, parameter and operational risk to underperform holding is worse than doing nothing. |

Two rules that matter more than the numbers:

- **Every** failure is reported, not the first. Missing one metric is a tuning problem;
  missing five is a dead end, and stopping early hides the difference.
- A strategy is accepted only if it passed on **every** symbol. Passing on one market and
  failing on three is a strategy that found one favourable regime.

Rejected strategies stay in the report with their reasons attached. A record of what was
tried and why it failed is the most reusable output a research process produces.

---

## 6. Ranking

Ranking on one metric is how research goes wrong. Sort by return and the top fills with
strategies that took enormous risk; sort by Sharpe and it fills with strategies that traded
four times.

So the leaderboard reports two things separately:

- **Per-metric ranks** — where each strategy placed on each of the six criteria, so a
  reader can see *why* something ranks where it does.
- **A composite** — the mean of those ranks, deliberately *ordinal*. Averaging raw numbers
  would let one unbounded metric (return) drown five bounded ones (win rate ≤ 1). A rank
  average cannot be dominated that way.

Accepted strategies always sort above rejected ones regardless of score: the gate is the
decision, the score only orders within it.

---

## 7. Performance

The sweep is 14 strategies × 4 symbols × ~49,000 bars. Two changes made it tractable:

**Bounded history.** Indicators recompute over the whole visible window every bar, so cost
is O(bars × window). The engine default hands every strategy 5,000 bars even when it
declared it needs 52. The research window is `max(warmup × 3, 300)` instead.

Recursive indicators decay geometrically, so this is not free but is negligible —
*measured*, not assumed:

| Strategy | Full-window return | Bounded return | Δ | Trades |
|---|---|---|---|---|
| `ema_cross` | 3.1356384030…% | 3.1356384031…% | 1.1e-12 | identical |
| `keltner_trend` | −0.33678373886…% | −0.33678380128…% | 6.2e-10 | identical |

**Process pool.** Backtests are pure, CPU-bound and independent. Market data is shipped to
each worker once via a pool initialiser rather than with every task.

> On macOS and Windows the pool uses the *spawn* start method, which re-imports the calling
> module in each worker. A script calling `ResearchRunner.run()` must guard its entry point
> with `if __name__ == "__main__":`. The CLI is already guarded.

---

## 8. Output

Three self-contained artefacts per run, written to the report directory:

- `research-<timeframe>-<costs>.md` — for a terminal or a pull request
- `research-<timeframe>-<costs>.html` — no CDN, no external fonts; a report that needs a
  network stops rendering the moment it is opened without one
- `research-<timeframe>-<costs>.json` — for diffing one research run against another

Each states its own period, symbols, costs and thresholds. A leaderboard without them is a
set of numbers with no meaning attached.

---

## 9. What this does not establish

Results are **in-sample**, over one historical period, at one parameter set. Passing the
gate makes a strategy a candidate for walk-forward validation — not a candidate for
capital. `backtest/walkforward.py` is the next step, and no strategy should be funded on
the strength of a leaderboard position alone.
