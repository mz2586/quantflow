# QuantFlow

AI-powered algorithmic trading platform for Binance.

Event-driven, fully typed Python 3.12. Strategies run through the **same** risk, execution
and accounting code paths in backtest, paper and live mode — so a backtested edge is
measured against the machinery that will actually trade it.

> **Live trading is disarmed by default.** It requires `QF_TRADING__MODE=live` *and* an
> explicit confirmation token. Read [`docs/risk.md`](docs/risk.md) before arming it.

---

## Contents

- [Architecture](docs/architecture.md)
- [Implementation plan & milestones](docs/implementation-plan.md)
- [Risk model](docs/risk.md)
- [Operations runbook](docs/runbook.md)
- [Current status](PROJECT_STATUS.md)

## Quick start

```bash
cp .env.example .env          # then fill in what you need

# Infrastructure only (Postgres + Redis)
make infra-up

# Local toolchain
make install                  # uv venv + editable install with dev extras
make migrate                  # alembic upgrade head
make test                     # pytest
make check                    # lint + format check + types + tests

# Full stack in Docker (api + worker + db + redis)
make up
open http://localhost:8000/docs
```

## Layout

```
src/quantflow/
  core/          settings, structured logging, clock, decimal precision, DI, errors
  domain/        pure value objects and invariants — no IO
  persistence/   async SQLAlchemy models, repositories, unit of work
  cache/         Redis client, distributed locks, pub/sub event bus
  exchange/      ExchangeGateway protocol; Binance REST + websocket; paper broker
  marketdata/    historical downloader, Parquet store, resampling, live ingest
  strategy/      Strategy ABC, indicator library, strategy registry
  risk/          position sizing, hard limits, kill switch
  portfolio/     lot-level accounting, PnL, equity curve
  execution/     OMS state machine, fee and slippage models, order router
  backtest/      event-driven engine, metrics, walk-forward, HTML reports
  paper/         live-data paper trading engine
  analytics/     performance analytics, trade journal, attribution
  ai/            Optuna optimiser, regime detection, sentiment, LLM journal analysis
  notifications/ notifier protocol, Telegram transport, dispatcher
  api/           FastAPI app, routers, schemas, middleware
  workers/       scheduled and background jobs
  cli/           `quantflow` command-line entry point
dashboard/       React + Vite + TypeScript + Tailwind
tests/           unit + integration
```

## Non-negotiable engineering rules

1. **`Decimal` for money.** `float` appears only inside vectorised analytics.
2. **UTC-aware datetimes only.** Enforced by Ruff `DTZ`; time comes from a `Clock`.
3. **No IO in the domain layer.** Engines depend on protocols, never on drivers.
4. **Every order passes the risk engine.** There is no bypass path.
5. **Strict typing.** `mypy --strict` over `src/` and `tests/`.
6. **No look-ahead.** The backtester only ever exposes closed bars to a strategy.

## Make targets

| Target | Purpose |
| --- | --- |
| `make install` | Create `.venv` and install the package with dev extras |
| `make lint` / `make fmt` | Ruff check / Black + Ruff autofix |
| `make type` | `mypy --strict` |
| `make test` / `make test-int` | Unit tests / integration tests (needs infra) |
| `make check` | Everything CI runs |
| `make infra-up` / `make infra-down` | Postgres + Redis only |
| `make up` / `make down` / `make logs` | Full Docker stack |
| `make migrate` / `make migration m="..."` | Apply / generate Alembic migrations |
| `make download` | Backfill historical candles |
| `make backtest` | Run a backtest and emit an HTML report |
| `make dashboard` | Vite dev server |

## Licence

Proprietary. All rights reserved.
