# Architecture

QuantFlow is an event-driven trading system. The organising principle is that **a strategy
runs through the same risk, execution and accounting code in backtest, paper and live
mode.** What a backtest measures is the machinery that would actually trade.

## Layers

```
src/quantflow/
  core/           settings, structured logging, injected Clock, Decimal precision, DI, errors
  domain/         pure value objects and invariants — no IO
  persistence/    async SQLAlchemy 2 models, repositories, unit of work, Alembic
  cache/          Redis client, distributed locks, pub/sub event bus
  exchange/       ExchangeGateway protocol; Bybit V5 REST + websocket; in-process simulator
  marketdata/     historical downloader, Parquet store, resampling, live ingest
  strategy/       Strategy ABC, indicator library, registry, 44-strategy library
  orchestrator/   scoring, selection and pyramiding across the registry
  risk/           position sizing, hard limits, exposure, conviction, kill switch
  portfolio/      FIFO lot accounting, PnL, funding, equity curve
  position/       intrabar position management
  execution/      OMS state machine, fee and slippage models, maker-first order router
  live/           session runner, venue reconciliation, equity resolution, heartbeat
  paper/          live-data paper engine with simulated fills
  backtest/       event-driven engine, metrics, walk-forward, HTML reports
  analytics/      performance analytics, trade journal, attribution
  ai/, aitrader/  optimiser, regime detection, optional LLM journal analysis
  intelligence/   market context
  universe/       tradeable-asset discovery and eligibility
  neutral/        market-neutral construction
  notifications/  notifier protocol, Telegram transport, dispatcher
  api/            FastAPI app, routers, schemas, middleware
  workers/        scheduled and background jobs
  cli/            `quantflow` entry point
  forex/          EXPERIMENTAL FX adapters — never placed an order
  lab/, research/ research harnesses
dashboard/        React 18 + Vite 5 + TypeScript + Tailwind
```

## The dependency rule

`domain/` depends on nothing but the standard library and `core/`. Engines depend on
**protocols** (`ExchangeGateway`, `OrderRouter`, `Clock`, `Notifier`), never on drivers.
That is what lets the same engine run against a live venue, a simulator, or a fake in a
test, without a branch anywhere in the trading logic.

## One bar, end to end

1. **Market data** arrives — a closed candle from the websocket feed, or the next bar from
   the historical store in a backtest.
2. **The orchestrator** scores the registry against the current context and selects a
   candidate, rather than running one fixed strategy.
3. **The strategy** sees a `StrategyContext` carrying only *closed* bars, the portfolio
   snapshot and any open position. It returns a `Signal` with a direction, an optional
   conviction, and optional stop and target prices.
4. **The risk engine** sizes the position and applies every hard limit. There is no path
   around it. It can refuse, and refusal is a normal outcome.
5. **The router** converts the sized intent into an `OrderRequest`, snapping price to the
   instrument's tick size and quantity to its lot size, and rejecting sub-minimum notional
   locally rather than making the venue do it. Entries are maker-first where configured.
6. **The exchange gateway** submits, attaching `stopLoss` / `takeProfit` to the entry
   order with `tpslMode=Partial`.
7. **The reconciler** treats the venue as the source of truth: whether the position exists
   and whether it is protected is read back from the exchange, never inferred.
8. **The portfolio** books the fill against FIFO lots and updates realised and unrealised
   PnL, funding and the equity curve.
9. **Persistence and the API** record the result; the dashboard reads it.

## Why "the venue is the source of truth"

A position that local state believes is protected, but that the exchange has no stop
attached to, is an unbounded loss waiting for a gap. So "protected" means the exchange
says so. `live/reconcile.py` re-reads positions and stops from the venue, and
`position/intrabar.py` manages protection between bar closes rather than waiting for one.

## Modes

| Mode | Data | Fills | Same risk/execution path? |
|---|---|---|---|
| Backtest | Historical store | Simulated | Yes |
| Paper | Live feed | Simulated | Yes |
| Demo | Live feed | **Real venue**, virtual funds | Yes |
| Live | Live feed | **Real venue**, real funds | Yes |

## Storage

PostgreSQL 16 for orders, fills, positions, closed trades, sessions, risk events and
equity points; Alembic for migrations. Redis 7 for cache, distributed locks and the pub/sub
event bus. Historical candles are stored as Parquet under `QF_STORAGE__DATA_DIR`.

## What is deliberately absent

- No message broker. The event bus is Redis pub/sub and the process is single-node.
- No HA or failover. An outage means an unmanaged position; that is a documented
  limitation, not an oversight.
- No strategy sandbox. Strategies are trusted Python running in-process.
