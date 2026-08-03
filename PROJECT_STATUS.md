# QuantFlow — Project Status

**Updated:** 2026-08-03 · **Commit:** `f8d1734` · **Version:** 0.1.0

**Live trading is disabled and has never been enabled.** No authenticated order has ever
been sent. See [Live trading](#4-live-trading-safety-interlock).

---

## 1. At a glance

| | |
|---|---|
| Test suite | **1026 passed, 0 failed** (2m 13s) |
| Dashboard tests | **26 passed** |
| `mypy --strict` | clean — 118 source files |
| `ruff` / `black` | clean |
| `tsc --noEmit` / `eslint` | clean |
| Coverage | 82% (statement + branch) |
| Docker stack | api, db, redis, worker healthy; migrate exits 0 |
| Market data | 10,285 contiguous real 1h bars each for BTC/USDT and ETH/USDT — **0 gaps** |
| Trading mode | `paper` |

---

## 2. Completed modules

Every module below is implemented, typed, tested, and exercised against the running
stack — not scaffolded.

| Module | LOC | What it does |
|---|---:|---|
| `core` | 1,426 | Settings (Pydantic v2, nested env), structlog with secret redaction, injected `Clock`, Decimal precision helpers, async DI container, typed error hierarchy |
| `domain` | 2,074 | Pure model, no IO: `Symbol`/`Instrument`, `Candle`/`CandleSeries` with gap detection, order state machine with idempotent fills, FIFO position lots, portfolio, signals |
| `persistence` | 1,971 | SQLAlchemy 2 async, Alembic (2 migrations), repositories returning detached-safe frozen dataclasses, chunked UPSERTs sized for asyncpg's 32,767-param cap |
| `cache` | 498 | Redis with explicit retry/backoff, typed namespacing |
| `exchange` | 2,158 | CCXT Binance gateway, token-bucket rate limiter, instrument sync, `SimulatedBroker` fill model shared by backtest/paper/live |
| `marketdata` | 825 | Resumable downloader, resampling, store with gap verification |
| `strategy` | 1,208 | Registry + 3 strategies (`donchian_breakout`, `ema_cross`, `rsi_reversion`), indicator library. Strategies are **pure decision functions** — they cannot place orders or size positions |
| `risk` | 1,529 | 13-rule engine, position sizers, latched persistent kill switch |
| `portfolio` | 370 | Cash + positions applied atomically, idempotent on fill id, FIFO trade attribution |
| `execution` | 468 | Order lifecycle, retry, reconciliation |
| `backtest` | 1,910 | Event-driven, no look-ahead, self-contained HTML reports |
| `paper` | 641 | Live-data paper engine, persisted sessions |
| `live` | 453 | Live runner behind a five-condition arming interlock (disabled) |
| `ai` | 1,051 | Regime detection + `AIAdvice` that can only veto or shrink conviction |
| `analytics` | 494 | Attribution by strategy/symbol/side/hour, streaks, concentration, plain-language warnings |
| `notifications` | 785 | Dispatcher with severity filter, dedup, rate limiting; Telegram + null transports |
| `api` | 2,102 | FastAPI: health, readiness, market data, portfolio, risk, strategies, backtest, analytics, websocket, Prometheus metrics |
| `cli` | 559 | `config`, `serve`, `data`, `backtest`, `trade`, `risk` |
| `workers` | 165 | Background ingestion + instrument refresh |
| `dashboard` | 1,346 | React 18 + Vite + TypeScript + Tailwind + Recharts |

### Risk rules (all 13 active)

`KillSwitch` · `TradingHalted` · `StopLossRequired` · `MaxPositionSize` ·
`MaxTotalExposure` · `MaxConcurrentPositions` · `MaxLeverage` · `OrderNotional` ·
`MaxDailyLoss` · `MaxDrawdown` · `OrderRate` · `Instrument` · `SufficientCash`

Every mandated rule from the brief is enforced: stop loss, position sizing, max daily
loss, max drawdown, max concurrent positions, kill switch.

---

## 3. Architecture

```
                        ┌──────────────────────────┐
                        │   Dashboard (React/TS)   │
                        │  Vite proxy → API :8100  │
                        └────────────┬─────────────┘
                                     │ REST + WebSocket
                        ┌────────────▼─────────────┐
                        │      FastAPI (api)       │
                        │ health · market · risk   │
                        │ portfolio · analytics    │
                        └────────────┬─────────────┘
                                     │
   ┌─────────────────────────────────┼─────────────────────────────────┐
   │                                 │                                 │
┌──▼───────────┐            ┌────────▼────────┐              ┌─────────▼────────┐
│  Backtest    │            │  Paper engine   │              │  Live runner     │
│  engine      │            │                 │              │  ✖ DISABLED      │
└──┬───────────┘            └────────┬────────┘              └─────────┬────────┘
   │                                 │                                 │
   └─────────────────┬───────────────┴─────────────────┬───────────────┘
                     │                                 │
              ┌──────▼──────┐                   ┌──────▼───────┐
              │  Strategy   │  Signal           │  Portfolio   │
              │  (pure fn)  │─────────┐         │  manager     │
              └─────────────┘         │         └──────▲───────┘
                                      │                │ fills
                              ┌───────▼────────┐       │
                              │   AI engine    │       │
                              │ veto / shrink  │       │
                              │ only — never   │       │
                              │ enlarges       │       │
                              └───────┬────────┘       │
                                      │                │
                        ╔═════════════▼════════════╗   │
                        ║      RISK ENGINE         ║   │
                        ║  13 rules · sizing       ║   │
                        ║  kill switch (latched)   ║   │
                        ║  ── MANDATORY GATE ──    ║   │
                        ╚═════════════┬════════════╝   │
                                      │ approved order │
                              ┌───────▼────────┐       │
                              │   Execution    │───────┘
                              │    engine      │
                              └───────┬────────┘
                                      │
                   ┌──────────────────┼──────────────────┐
                   │                                     │
          ┌────────▼─────────┐                 ┌─────────▼─────────┐
          │ SimulatedBroker  │                 │  Binance (CCXT)   │
          │ backtest · paper │                 │  ✖ requires arming│
          └──────────────────┘                 └───────────────────┘

  Infrastructure:  PostgreSQL/TimescaleDB :55432 · Redis :56379 · worker (ingest)
```

**Invariants enforced by the design, not by convention:**

- No path from signal to venue bypasses the risk engine.
- AI can only veto or shrink: the conviction multiplier is constrained to `[0,1]` at
  construction, so no AI field can create, enlarge, or flip a position.
- Money is `Decimal` throughout; `float` appears only inside vectorised analytics.
  Money crosses the wire as **strings** — a JSON number would be corrupted by JS floats.
- All datetimes are UTC-aware, enforced by Ruff `DTZ` plus an injected `Clock`.
- Backtest exposes only closed bars up to bar *i*; orders match against bar *i+1*.
- One fill model shared by backtest, paper, and live.

---

## 4. Live trading safety interlock

Live trading requires **all five** conditions simultaneously. Any one missing leaves it
disarmed:

1. `ENABLE_LIVE_TRADING=true` in the environment
2. Trading mode set to `live`
3. An explicit confirmation token
4. Credentials present
5. Not pointed at testnet

Current state: **not armed, and never has been.** No authenticated order has been sent.

---

## 5. Remaining modules and work

| # | Item | Notes |
|---|---|---|
| 1 | **M18 hardening** | Load testing, chaos/failure injection, backpressure under sustained websocket load |
| 2 | **AI engine breadth** | `RuleBasedRegimeDetector` is the default and is tested; the Gaussian-mixture detector is opt-in and degrades gracefully, but neither has been validated against out-of-sample performance |
| 3 | **Dashboard breadth** | Trades are fetched and drive the PnL chart, but there is no per-trade table; no backtest-launch UI; no multi-session comparison |
| 4 | **Reconciliation** | Execution-engine reconciliation is implemented but has never run against a real venue, because live trading has never been armed |
| 5 | **Migration drift check** | Migrations are hand-verified; no CI job asserts models and migrations agree |
| 6 | **Coverage gaps** | `workers/runner.py` 0%, `live/runner.py` 61%, `execution/engine.py` 75% — the three least-exercised paths, and two of them are the ones that would touch real money |

---

## 6. Production blockers

These are genuine external dependencies. None can be resolved by writing code.

| Blocker | Needed to |
|---|---|
| **Binance API credentials with trade permission** | Send any real order. Deliberately absent |
| **Explicit human decision to arm live trading** | Flip the five-condition interlock. Standing instruction is: do not enable |
| **Telegram bot token + chat id** | Deliver notifications. The dispatcher works; only the `null` transport is enabled |
| **Strategy validation on out-of-sample data** | The one strategy run end-to-end (`donchian_breakout`) **lost money**: −10.66% over 14 months, 26.1% win rate, fees consumed 140% of gross profit, longest losing streak 14 trades. It is not fit to trade capital as configured |
| **Offsite backup + restore drill** | `pg_dump` runs; a restore has never been rehearsed |
| **Secret management** | Secrets come from `.env`. Production needs a real secret store |
| **TLS / auth on the API** | The API is unauthenticated and HTTP-only. Safe on localhost, not deployable as-is |

The fourth row is the substantive one. The platform is sound; **the strategy is not
profitable**, and no amount of engineering fixes that.

---

## 7. Code statistics

| | Files | Lines |
|---|---:|---:|
| Python source | 87 | 20,702 |
| Python tests | 31 | 11,230 |
| Dashboard TS/TSX | 6 | 1,346 |
| Migrations | 2 | — |
| **Total** | **126** | **33,278** |

- Test-to-source ratio: **0.54:1**
- 1,026 Python tests + 26 dashboard tests
- Coverage 82% statement+branch
- 9 commits

---

## 8. Verified against the running system

Not asserted from code — measured on 2026-08-03 against the live stack.

**Paper session `dashboard-demo`** — `donchian_breakout`, BTC/USDT 1h, 10,000 USDT start:

```
bars 10,085 → signals 262 → orders 262 → fills 330 → rejections 0
closed trades 165 (43 wins, 26.1%)
final equity 8,933.80   return −10.66%   max drawdown 11.25%
```

**All 17 dashboard endpoints verified through the Vite proxy**, every one HTTP 200:
readiness, health, portfolio, equity curve, trades, orders, sessions, risk status,
risk events, market series, candles (BTC + ETH), attribution, warnings, strategies,
notifications, Prometheus metrics (271 samples).

**Rendering confirmed in-browser**: the attribution table reads
`donchian_breakout 165 −1,066.20 26.1%`, which reconciles exactly with equity 8,933.80
from 10,000.

**Kill switch**: engages, latches, persists across restart, and writes an audit trail.

---

## 9. Next actions

1. M18 hardening — load and failure-injection testing.
2. Raise coverage on `workers/runner.py`, `live/runner.py`, `execution/engine.py`.
3. Walk-forward validation across strategies and parameters — the current strategy loses
   money and should not be a candidate for capital.
4. Add a per-trade table and backtest-launch UI to the dashboard.
5. Leave live trading disabled pending an explicit human decision.
