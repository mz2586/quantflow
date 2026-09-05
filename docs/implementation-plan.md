# QuantFlow — Implementation Plan

Authoritative milestone plan. Each milestone is independently testable and ends in a
local commit. Released state is tracked in [`CHANGELOG.md`](../CHANGELOG.md).

## Guiding constraints

| Constraint | Decision |
| --- | --- |
| Money arithmetic | `decimal.Decimal` everywhere in the domain. `float` only inside vectorised analytics. |
| Time | Timezone-aware UTC only. Enforced by Ruff `DTZ` + a `Clock` abstraction for testability. |
| IO | All exchange/DB/cache access sits behind a `Protocol`; engines depend on the protocol, never the driver. |
| Live trading | Disabled by default. Requires `QF_TRADING__MODE=live` **and** an explicit non-default `allow_live` confirmation token. |
| Risk | Orders are unconditionally routed through the risk engine. There is no code path from a strategy signal to an exchange that bypasses it. |
| Errors | Typed exception hierarchy rooted at `QuantFlowError`; no bare `except`. |

## Milestones

### M0 — Repository foundation
- Package layout (`src/` layout), `pyproject.toml`, Ruff/Black/Mypy strict, pytest config.
- `.env.example`, `.gitignore`, `Makefile`, CI workflow.
- **DoD:** `make lint type test` runs green on an empty test suite.

### M1 — Core cross-cutting layer *(Phase 1)*
- Settings (Pydantic v2, nested env, secret types), structured logging (structlog, JSON in prod,
  console in dev, request/correlation IDs), typed error hierarchy, `Clock`, precision helpers,
  DI container.
- **DoD:** unit tests for settings precedence, log redaction of secrets, decimal quantisation.

### M2 — Domain model *(Phase 1)*
- Frozen dataclasses / Pydantic models: `Symbol`, `Instrument`, `Candle`, `Trade`, `OrderBook`,
  `OrderRequest`, `Order`, `Fill`, `Position`, `PortfolioSnapshot`, `Signal`.
- Enums for side / order type / TIF / status / timeframe with exchange-agnostic semantics.
- **DoD:** property-based tests (Hypothesis) on position accounting invariants.

### M3 — Persistence *(Phase 1)*
- Async SQLAlchemy 2.0 engine + session factory, declarative base with naming convention,
  ORM models, repositories, unit-of-work. Alembic baseline migration; TimescaleDB hypertable for
  candles when the extension is present (graceful fallback to a plain partitioned table).
- **DoD:** integration tests against a real Postgres container; `alembic upgrade head` + `downgrade base` round-trips.

### M4 — Cache & messaging *(Phase 1)*
- Redis client factory, JSON/orjson codec, distributed lock, pub/sub event bus, stream-backed
  work queue, TTL-aware caches for tickers/instruments.
- **DoD:** tests against `fakeredis` for logic + a real Redis for the lock semantics.

### M5 — Exchange integration *(Phase 1)*
- `ExchangeGateway` protocol. Binance implementation over CCXT (spot + USDⓈ-M futures) with
  token-bucket rate limiting, exponential backoff, error translation, symbol/precision mapping,
  and a websocket market-data client with auto-reconnect and gap detection.
- **DoD:** unit tests with mocked CCXT + `respx`; opt-in `network` test against Binance testnet.

### M6 — Market data pipeline *(Phase 1)*
- Historical OHLCV downloader (paginated, resumable, idempotent upserts, integrity/gap report),
  Parquet/Polars store, resampler, live ingest service.
- **DoD:** downloader tests over a fake gateway covering pagination, gaps, and dedupe.

### M7 — FastAPI application *(Phase 1)*
- App factory, lifespan wiring, middleware (request ID, timing, error envelope), Prometheus
  metrics, `/healthz` `/readyz`, market-data and system routers, OpenAPI metadata.
- **DoD:** `httpx.ASGITransport` tests for health, error envelope, and market-data endpoints.

### M8 — Docker & infrastructure *(Phase 1)*
- Multi-stage Dockerfile (non-root, wheel-cached), compose stack (api, worker, db, redis),
  healthchecks, `.dockerignore`, Make targets.
- **DoD:** `docker compose up` reaches healthy; `/readyz` returns 200 from the container.

### M9 — Strategy engine *(Phase 2)*
- `Strategy` ABC + `StrategyContext`, indicator library (pure, vectorised, incremental-safe),
  parameter schema via Pydantic, registry with entry-point discovery, three reference strategies
  (EMA cross, RSI mean-reversion, Donchian breakout).
- **DoD:** golden-value indicator tests, strategy determinism tests.

### M10 — Risk engine *(Phase 2)*
- Position sizing (fixed fractional, ATR-based, volatility target), mandatory stop-loss guard,
  max daily loss, max drawdown, max concurrent positions, exposure/notional caps, kill switch
  with persistent latched state.
- **DoD:** one test per rule, both allow and deny paths; a test asserting an order without a
  stop-loss is always rejected.

### M11 — Execution & portfolio *(Phase 2)*
- OMS state machine, fee/slippage models, order router, portfolio manager with FIFO lot
  accounting, realised/unrealised PnL, equity curve.
- **DoD:** state-machine transition table tests; PnL reconciliation tests.

### M12 — Backtesting engine *(Phase 2)*
- Event-driven, bar-close-honest backtester (no look-ahead), the same risk + execution code paths
  as live, metrics suite (CAGR, Sharpe, Sortino, Calmar, max DD, exposure, win rate, profit
  factor, turnover), walk-forward splitter, Plotly HTML report. Adapters for VectorBT
  (fast sweeps) and Backtrader (cross-validation).
- **DoD:** analytic fixture where expected PnL is known exactly; look-ahead regression test.

### M13 — Paper trading engine *(Phase 2)*
- Live market data + simulated broker sharing the backtest fill model, persisted state, crash
  recovery, reconciliation against the exchange clock.
- **DoD:** end-to-end test driving a scripted data feed through strategy → risk → paper broker.

### M14 — Notifications *(Phase 3)*
- Notifier protocol, Telegram transport (httpx, retry, rate-limit aware), templated events
  (fill, risk breach, kill switch, daily digest), dispatcher with severity routing.
- **DoD:** transport tests with `respx`; template snapshot tests.

### M15 — Analytics & reporting *(Phase 2/3)*
- Performance analytics service, trade journal, attribution by strategy/symbol/regime,
  API endpoints feeding the dashboard.
- **DoD:** metric tests against known series.

### M16 — React dashboard *(Phase 3)*
- Vite + React + TypeScript + Tailwind. Typed API client, live candlestick chart, equity curve,
  positions/orders tables, risk panel with kill switch, backtest runner + report viewer,
  websocket live updates.
- **DoD:** `npm run build` + `tsc --noEmit` clean; vitest unit tests on data transforms.

### M17 — AI engine *(Phase 4)*
- Parameter optimisation (Optuna, walk-forward objective, overfitting guards), market-regime
  detection (feature pipeline + Gaussian-mixture/HMM classifier), news-sentiment provider
  interface, LLM-backed trade-journal analysis behind a provider abstraction.
- **DoD:** deterministic-seed optimiser tests; regime-labelling tests on synthetic regimes;
  LLM client tested against a stub transport.

### M18 — Hardening
- Secret handling audit, API auth, coverage gate, load smoke test, runbook, ADRs, threat model.
- **DoD:** CI enforces lint + types + tests + coverage threshold.
