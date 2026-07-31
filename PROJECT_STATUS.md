# QuantFlow — Project Status

**Updated:** 2026-07-31
**Build:** 737 tests passing · Ruff clean · `mypy --strict` clean · 74 source files

---

## Current task

Phase 3 groundwork — FastAPI application and the API surface (M7), then the backtesting
engine (M12).

---

## Completed

### M0 — Repository foundation ✅
`src/` layout, `pyproject.toml` with Ruff + Black + `mypy --strict` + pytest, Makefile,
GitHub Actions CI (quality / integration / dashboard / docker jobs), `.env.example`
documenting every setting.

### M1 — Core cross-cutting layer ✅
- Pydantic v2 settings: nested env vars, production guardrails, secrets as `SecretStr`
- **Live trading is disarmed by default** and requires an explicit confirmation token
- `structlog` logging with secret redaction and contextvar correlation ids
- `Clock` protocol (`SystemClock` / `FrozenClock`) — no code calls `datetime.now` directly
- Decimal precision helpers with explicit, side-aware rounding
- Async DI container with singleton caching and reverse-order teardown
- Typed exception hierarchy rooted at `QuantFlowError`

### M2 — Domain model ✅
Pure value objects, no IO: `Symbol`, `Instrument`, `Candle`/`CandleSeries` (with gap
detection), `OrderRequest`/`Order`/`Fill` (enforced OMS transition table, idempotent
fills), `Position` (FIFO lots, position flips, `ClosedTrade` round-trips),
`PortfolioSnapshot`, `Signal`. Hypothesis property tests cover the accounting invariants.

### M3 — Persistence ✅
15 ORM models, `NUMERIC(28,12)` money columns, repositories as the sole record↔domain
translation point, async engine + `UnitOfWork`, Alembic baseline plus a TimescaleDB
hypertable migration that no-ops gracefully on plain Postgres.

### M4 — Cache & messaging ✅
Namespaced Redis facade with a `Decimal`-safe orjson codec, distributed lock with TTL and
compare-and-delete release, pub/sub event bus, Redis-stream work queue.

### M5 — Exchange integration ✅
`ExchangeGateway` protocol; Binance REST over CCXT (rate limiting, retry with full jitter,
error translation, clock-drift check); websocket client with auto-reconnect, stale-socket
detection and gap detection; shared `SimulatedBroker` with a deliberately pessimistic fill
model used by **both** backtest and paper trading.

### M6 — Market-data pipeline ✅
Resumable, idempotent paginated backfill that reports gaps rather than hiding them;
Hive-partitioned Parquet store via Polars; timeframe resampling that drops partial buckets.

### M8 — Docker & infrastructure ✅ *(brought forward)*
Multi-stage Dockerfile (non-root, tini, cached dependency layer); compose stack on
non-conflicting host ports (55432 / 56379 / 8100) so it cannot disturb other local
Postgres/Redis instances.

### M9 — Strategy engine ✅
`Strategy` ABC as a **pure decision function** — it cannot place orders or size positions,
which is what makes bypassing the risk engine structurally impossible. Aligned-output
indicator library (SMA, EMA, Wilder, RSI, ATR, Bollinger, MACD, Donchian, crossings), a
registry with JSON-Schema introspection, and three reference strategies: EMA cross, RSI
mean reversion (trend-filtered), Donchian breakout.

### M10 — Risk engine ✅
- **Every order passes through `RiskEngine.approve`. There is no other path to a venue.**
- Sizers: fixed-fractional (stop-anchored), volatility-target, fixed-notional
- 13 rules covering: mandatory stop loss, max position %, max total exposure, max
  concurrent positions, max daily loss (halts the day), max drawdown (latches the kill
  switch), max leverage, order rate, venue rules, cash sufficiency
- Kill switch is **latched and persistent** — a restart does not clear it, and it *fails
  closed* if its state cannot be read
- Exits are exempt from exposure limits, so a limit can never trap an open position

### M11 — Execution & portfolio ✅
Portfolio manager as the single source of truth (cash and positions move atomically,
fills idempotent by id, crash recovery, venue reconciliation). Execution engine with a
fixed order of operations and a redundant pre-submission protection assert, stale-signal
rejection, and a guard that refuses to submit from a non-live engine to a production
gateway.

---

## Remaining

| Milestone | Scope | Status |
| --- | --- | --- |
| **M7** | FastAPI app, routers, middleware, metrics, health | next |
| **M12** | Event-driven backtester, metrics, walk-forward, Plotly reports | pending |
| **M13** | Paper-trading engine on live data | pending |
| **M14** | Telegram notifications and dispatcher | pending |
| **M15** | Analytics, trade journal, attribution | pending |
| **M16** | React + Vite + TS + Tailwind dashboard | pending |
| **M17** | AI: Optuna optimiser, regime detection, sentiment, LLM journal | pending |
| **M18** | Auth, coverage gate, runbook, ADRs, threat model | pending |

---

## Blockers

**None blocking progress.** Everything below is external and does not gate the remaining
build; each has a working default.

| Item | Impact | Needed |
| --- | --- | --- |
| Binance API credentials | Live/testnet trading and private endpoints are untestable end-to-end. Public market data, backtesting and paper trading all work without them. | `QF_EXCHANGE__API_KEY` / `QF_EXCHANGE__API_SECRET` (testnet keys from testnet.binance.vision are enough) |
| Telegram bot token | M14 delivery cannot be verified against the real API; transport is tested with a stubbed HTTP layer. | `QF_NOTIFICATIONS__TELEGRAM_BOT_TOKEN` + chat id |
| Anthropic API key | M17 LLM journal analysis runs against a stub without it. | `QF_AI__ANTHROPIC_API_KEY` |
| News provider key | Sentiment interface has no live feed. | `QF_AI__NEWS_API_KEY` |

**Decisions I made rather than blocking on** (all reversible, all documented in-code):
1. VectorBT and Backtrader are **optional extras**, not core dependencies — both carry
   heavy conflicting transitive pins. Adapters will be provided; the primary backtester is
   our own so that live and backtest share one execution path.
2. TimescaleDB over plain Postgres, with graceful degradation.
3. FIFO lot accounting rather than average cost, for auditable trade-level attribution.
4. Local host ports moved off 5432/6379 to avoid colliding with your existing containers.

---

## Bugs found and fixed by the test suite

These were real defects, not test artefacts:

1. **`TokenBucket` livelocked forever.** Float drift made a "full" bucket hold
   `0.9999999999999999`, so the `>= 1` check never passed and the acquire loop spun on a
   ~1e-18 delay. Fixed with an epsilon plus a minimum wait; regression test added.
2. **Bulk candle upsert exceeded asyncpg's 32767 bind-parameter cap** (a tighter limit
   than Postgres's own 65535). Chunk size 5000 → 3000.
3. **`OrderRepository.save` touched an unloaded relationship**, emitting IO outside
   SQLAlchemy's async greenlet context.
4. **`Base.metadata` was only populated as a side effect of importing repositories**, so
   `create_all` could silently produce an empty schema.
5. **`SimulatedBroker.submit` loaded the instrument but never validated against it** —
   letting a backtest fill orders Binance would have rejected.
6. **Order-book level ordering validation was inverted.**

---

## Next actions

1. FastAPI application factory, middleware, health/metrics, market-data and risk routers.
2. Event-driven backtester reusing the live risk and execution path, with a look-ahead
   regression test and an analytic fixture whose PnL is known exactly.
3. Paper-trading engine on the same fill model.
4. Then Phase 3 (dashboard, notifications) and Phase 4 (AI).
