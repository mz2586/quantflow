# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the major version is `0`, the public API — settings names, CLI commands, REST
routes and strategy ids — may change in a minor release.

## [Unreleased]

## [0.1.0] — 2026-09-05

First public release. Everything below already existed; this entry records the state the
project is published in rather than a set of changes from a previous public version.

### Added

- **Bybit V5 integration** over CCXT — REST and websocket — across three environments:
  `demo` (`api-demo.bybit.com`), `testnet` and `mainnet`. Spot and futures.
- **Event-driven backtest engine** that exposes only closed bars to a strategy, with
  metrics, walk-forward analysis and HTML reports.
- **Paper-trading engine** against live market data with simulated fills.
- **Live/demo session runner** with venue reconciliation, intrabar position management and
  a maker-first order router.
- **44 strategies** across trend, breakout, mean-reversion, volatility, volume and
  calendar families, plus `buy_and_hold` as an explicit benchmark.
- **Adaptive orchestrator** that scores and selects from the registry each bar rather than
  running one fixed strategy.
- **Risk engine** on the path of every order: per-position and portfolio exposure caps,
  concurrent-position limit, daily-loss halt, latching drawdown kill switch, mandatory
  stop loss, leverage and single-order notional ceilings.
- **Venue constraint handling** — tick-size and lot-size snapping, minimum-notional
  rejection, `PostOnly` mapped as a Bybit V5 time-in-force, and TP/SL attached to the
  entry order with `tpslMode=Partial`.
- **REST API** (FastAPI) with ~43 endpoints, plus `/healthz`, `/readyz` and `/metrics`.
- **React dashboard** (Vite, TypeScript, Tailwind): session status, equity curve with peak
  and drawdown, positions and orders as the venue reports them, closed-trade ledger, PnL,
  strategy/symbol/side/exit-reason attribution, decision log and a data-freshness panel.
- **CLI** (`quantflow`): `data`, `backtest`, `trade`, `risk`, `research`, `serve`.
- **Persistence** on PostgreSQL 16 via async SQLAlchemy 2 and Alembic; Redis 7 for cache,
  locks and pub/sub.
- **Optional Telegram notifications** and an optional, provider-agnostic LLM journal
  analyser that defaults to a no-network, no-credential `null` client.
- **Experimental FX package** (`src/quantflow/forex/`) with OANDA v20 and Bybit MT5
  adapters. **It has never placed an order** against any account, live or demo.

### Security

- Live trading is **disabled by default** and requires three independent, deliberate
  settings changes: `QF_TRADING__MODE=live`, a matching
  `QF_TRADING__LIVE_CONFIRMATION` token, and `QF_EXCHANGE__ENV=mainnet`.
- Demo and mainnet credentials are held in **separate settings fields** so switching
  environments cannot carry a key from one to the other.
- The demo launcher refuses to start against mainnet **before any client is constructed**.
- Credentials are typed as `pydantic.SecretStr`, coerced from blank to absent, and
  redacted from every log line.
- `QF_API_HOST` now defaults to `127.0.0.1`. Several API routers return account data
  without authentication; see [SECURITY.md](SECURITY.md).

### Documentation

- README rewritten for public release with a prominent trading-risk disclaimer and no
  claim of profitability or edge.
- Added `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, this changelog,
  `docs/architecture.md`, `docs/risk.md` and `docs/runbook.md`.
- Research reports moved to `docs/research/`. The strategy research report records that
  **no strategy tested beat buy-and-hold** over the period examined.

### Licensing

- Released under [Apache-2.0](LICENSE). The project was previously marked proprietary.

[Unreleased]: https://github.com/mz2586/quantflow/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mz2586/quantflow/releases/tag/v0.1.0
