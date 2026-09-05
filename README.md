# QuantFlow

An event-driven, fully typed algorithmic trading platform for **Bybit**, written in
Python 3.12 with a React dashboard.

Strategies run through the **same** risk, execution and accounting code in backtest, paper
and live mode — so what a backtest measures is the machinery that would actually trade.

> ## ⚠️ Trading risk
>
> **Trading cryptocurrency carries a substantial risk of loss. You can lose more than you
> expect, and you can lose everything you put in.**
>
> This software is provided for research and education. It is **not** financial advice, it
> makes **no** claim to be profitable, and it does **not** have a proven edge. Backtest,
> demo and paper-trading results are simulations; **past and simulated results do not
> indicate, predict, or guarantee future performance.** Real execution introduces queue
> position, partial fills, slippage, funding, outages and latency that no simulator
> reproduces, and all of them work against you.
>
> The project's own research report (`docs/research/`) records that **no strategy in this
> library beat simply holding the asset** over the period tested.
>
> You alone are responsible for any order this software places and for complying with the
> laws, tax rules and exchange terms that apply to you. **Live trading is disabled by
> default and you must deliberately enable it. Do not enable it with money you cannot
> afford to lose.**

---

## Contents

- [Documentation](#documentation)
- [What it does](#what-it-does)
- [Support matrix](#support-matrix)
- [Architecture](#architecture)
- [Paper, demo and live](#paper-demo-and-live)
- [Installation](#installation)
- [Configuration](#configuration)
- [API keys](#api-keys)
- [Running](#running)
- [Dashboard](#dashboard)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Security](#security)
- [Licence](#licence)

---

## Documentation

| | |
|---|---|
| [Architecture](docs/architecture.md) | Layers, the dependency rule, one bar end to end |
| [Risk model](docs/risk.md) | Every hard limit, the interlocks, and what they cannot protect you from |
| [Operations runbook](docs/runbook.md) | Start, stop, monitor, incidents, log hygiene |
| [Strategy research](docs/research/strategy-research-2026-08.md) | The systematic sweep. **No strategy beat buy-and-hold.** |
| [Bybit integration notes](docs/exchange/bybit-integration.md) | Venue behaviour and quirks |
| [Security policy](SECURITY.md) · [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) | |

## What it does

- **Backtests** 44 strategies over historical candles with an event-driven engine that
  never shows a strategy a bar that has not closed.
- **Paper-trades** against live market data with simulated fills.
- **Demo-trades** against Bybit's demo venue — real matching engine, real fills, virtual
  funds.
- **Live-trades**, if you explicitly arm it.
- **Selects** strategies adaptively per bar through an orchestrator that scores candidates
  rather than running one fixed strategy.
- **Enforces risk** on every order — there is no bypass path.
- **Reports** through a REST API, a React dashboard, Prometheus metrics and optional
  Telegram notifications.

## Support matrix

| | |
|---|---|
| **Exchange** | Bybit (V5, via CCXT). Environments: `demo`, `testnet`, `mainnet`. |
| **Markets** | Spot and futures (linear perpetuals). |
| **Timeframes** | 1m · 3m · 5m · 15m · 30m · 1h · 2h · 4h · 6h · 8h · 12h · 1d · 3d · 1w |
| **Python** | 3.12 (only — `>=3.12,<3.13`) |
| **Node** | 22+ (dashboard only) |
| **Database** | PostgreSQL 16 |
| **Cache** | Redis 7 |
| **FX** | `src/quantflow/forex/` — **EXPERIMENTAL**, never run against any account. |

Other exchanges are not supported. Binance support was removed.

## Architecture

### Strategies

44 registered strategies across distinct families — trend, breakout, mean reversion,
volatility, volume, calendar — plus `buy_and_hold` as an explicit benchmark. The spread is
deliberate: a library of variations on one idea would rank parameter choices while
appearing to rank ideas.

```bash
quantflow backtest strategies      # list every registered strategy with its parameters
```

**Active by default** — the demo launcher (`scripts/run_demo_bot.py`) does not pin a
strategy. The orchestrator scores the whole registry each bar and selects. The launcher
fixes only the universe (`BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT`), a `15m` bar
interval, and a cap of 4 additional meme markets. Override with `QF_BOT_STRATEGIES`,
`QF_BOT_SYMBOLS`, `QF_BOT_TIMEFRAME`.

**Experimental** — the `forex/` package (OANDA v20 and Bybit MT5 adapters) is written to
each venue's published API and tested against fakes. **It has never placed an order.**

### Risk

Every order passes the risk engine before it reaches a venue. Limits are mandatory; the
engine refuses to start without them.

| Control | Variable |
|---|---|
| Max notional per position, as a fraction of equity | `QF_RISK__MAX_POSITION_PCT` |
| Max aggregate gross exposure | `QF_RISK__MAX_TOTAL_EXPOSURE_PCT` |
| Max concurrent positions | `QF_RISK__MAX_CONCURRENT_POSITIONS` |
| Daily loss halt (blocks new entries for the UTC day) | `QF_RISK__MAX_DAILY_LOSS_PCT` |
| Drawdown kill switch (latches) | `QF_RISK__MAX_DRAWDOWN_PCT` |
| Mandatory stop loss | `QF_RISK__REQUIRE_STOP_LOSS` |
| Max leverage | `QF_RISK__MAX_LEVERAGE` |
| Max single-order notional | `QF_RISK__MAX_ORDER_NOTIONAL` |

### Execution

An order-management state machine with idempotent fills, a maker-first router, fee and
slippage models, and a reconciler that treats the **venue** as the source of truth for
whether a position is protected. Stops and targets attach to the entry order
(`stopLoss` / `takeProfit`, `tpslMode=Partial`); `set_trading_stop` handles post-entry
attachment. Prices snap to the instrument's tick size and quantities to its lot size
before submission.

### Venue constraints you will hit

Bybit publishes a `tick_size`, `lot_size` and `min_notional` per instrument, and QuantFlow
enforces all three locally so the venue does not have to reject you.

**This means a small account cannot trade some instruments at all.** If your risk limits
size a position below an instrument's minimum notional, that instrument is skipped — it is
not an error, and it is not something the software can work around. `PostOnly` is a
time-in-force on Bybit V5, not an order flag; a post-only order that would cross is
cancelled, not repriced.

## Paper, demo and live

| Mode | Data | Fills | Funds | How to select |
|---|---|---|---|---|
| **Backtest** | Historical | Simulated | None | `QF_TRADING__MODE=backtest` |
| **Paper** *(default)* | Live | Simulated | None | `QF_TRADING__MODE=paper` |
| **Demo** | Live | **Real venue** | **Virtual** | `QF_EXCHANGE__ENV=demo` + demo keys |
| **Live** | Live | **Real venue** | **REAL MONEY** | Three deliberate steps, below |

Demo is the recommended way to evaluate this software. Orders reach a real matching
engine, fill against a real book and hold real positions — but the funds are virtual, so a
bug costs you nothing.

### Enabling live trading

**Read this section twice before acting on it.** Live trading is disabled by default and
requires **three separate, deliberate changes**:

1. `QF_TRADING__MODE=live`
2. `QF_TRADING__LIVE_CONFIRMATION=I_UNDERSTAND_THE_RISK`
3. `QF_EXCHANGE__ENV=mainnet` and a mainnet API key/secret pair

If any one of those is missing, the process refuses to start rather than trading anyway.
The confirmation token is **not a secret** — it is an intent gate, and its only job is to
make sure nobody reaches live order flow by editing one variable.

**Before you take that third step:**

- Run in demo for long enough to see a losing streak, not just a winning one.
- Set every `QF_RISK__*` limit deliberately. The defaults are examples, not advice.
- Create the mainnet API key **without withdrawal permission**, and IP-restrict it.
- Know how to stop the bot and flatten positions (see [Running](#running)).
- Accept that you may lose the entire balance.

## Installation

Requires **Python 3.12**, [`uv`](https://github.com/astral-sh/uv), Docker (recommended),
and **Node 22+** for the dashboard.

```bash
git clone https://github.com/mz2586/quantflow.git
cd quantflow

make install          # uv venv + editable install with dev + ai extras
make env              # creates .env from .env.example if absent
make infra-up         # Postgres + Redis on loopback only
make migrate          # alembic upgrade head
make test             # 3,100+ unit tests
```

Optional extras: `make install-research` adds VectorBT and Backtrader (heavy, frequently
conflicting transitive pins — install only if you need them).

Full stack in Docker instead:

```bash
make up               # api + worker + db + redis
open http://localhost:8100/docs
```

## Configuration

All configuration is environment variables, read from `.env`. Nested settings use a
double underscore: `QF_<SECTION>__<FIELD>`.

`.env.example` is the reference and is split into a **DEMO** block and a **LIVE** block.
The live block ships commented out. `.env` is gitignored — **never commit it**.

### API keys

Credentials for demo and mainnet live in **separate variables** so that switching
environments cannot carry a mainnet key onto a demo host or the reverse:

```bash
QF_EXCHANGE__ENV=demo
QF_EXCHANGE__DEMO_API_KEY=
QF_EXCHANGE__DEMO_API_SECRET=
```

**Creating a Bybit demo key:** log in to Bybit → switch to **Demo Trading** → API →
create an API key **scoped to demo**. A demo key does not work on mainnet and a mainnet
key does not work on demo. Grant **trade** permission only — never **withdrawal**.

Leave every credential blank to run backtests and read public market data.

## Running

```bash
# Backtest
make backtest STRATEGY=ema_cross SYMBOL=BTC/USDT TIMEFRAME=1h START=2024-01-01 END=2025-01-01

# Paper trading
make paper STRATEGY=ema_cross SYMBOL=BTC/USDT

# API + dashboard
make api                              # http://localhost:8000
make dashboard                        # http://localhost:5173

# Unattended demo bot (restarts on crash, exponential backoff)
nohup caffeinate -s scripts/bot_supervisor.sh > scratchpad/bot-supervisor.log 2>&1 &
```

### Stopping

```bash
touch scratchpad/BOT_STOP             # clean stop; waits for the current run to exit
quantflow risk halt                   # block new entries immediately, keep positions
python scripts/flatten_demo.py        # close open demo positions
make down                             # stop the Docker stack
```

`BOT_STOP` is the correct way to stop the supervisor. Killing the process leaves the lock
directory behind; the supervisor clears a stale stop file on next start.

### Monitoring

```bash
tail -f scratchpad/bot.log
quantflow risk status
curl localhost:8000/healthz            # liveness
curl localhost:8000/readyz             # readiness
curl localhost:8000/metrics            # Prometheus
```

## Dashboard

React 18 + Vite 5 + TypeScript + Tailwind, served by ~43 FastAPI endpoints.

```bash
make dashboard-install
make api                               # backend first
make dashboard                         # http://localhost:5173
```

Vite proxies `/api`, `/healthz`, `/readyz` and `/metrics` to `http://localhost:8100` in
development, so the browser sees one origin and CORS never enters the picture. Override
with `QF_API_URL`.

It shows: session status and health, live equity curve with running peak and drawdown,
open positions and working orders **as the venue reports them**, closed-trade ledger,
period and cumulative PnL, per-strategy / per-symbol / per-side / per-exit-reason
attribution, recent decision-engine activity, and a freshness panel stating how current
each source on the page is.

## Testing

```bash
make test         # unit tests
make test-int     # integration tests (needs make infra-up)
make cov          # coverage, fails under 80%
make lint         # ruff + black --check
make type         # mypy --strict
make check        # everything CI runs
cd dashboard && npm run test && npm run build
```

Tests marked `network` hit a live venue and are excluded everywhere by default.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `REFUSING TO START: QF_EXCHANGE__ENV resolves to '…', not 'demo'` | The demo launcher is demo-only by design. Set `QF_EXCHANGE__ENV=demo`. Nothing was connected or ordered. |
| `live mode requires QF_TRADING__LIVE_CONFIRMATION=…` | Live arming needs the mode **and** the token. Intentional. |
| `price … is not a multiple of tick …` | The instrument's tick size changed, or a price bypassed rounding. |
| Order rejected, notional below minimum | The account is too small for that instrument at your risk limits. Not fixable in software. |
| `another supervisor holds … — refusing to start a second` | A supervisor is already running. Two would trade the same account against each other. |
| Orders rejected on 1m/5m bars | The fill model rejects orders exceeding 10% of a bar's volume. 15m is the shortest practical interval. |
| Dashboard shows no session | No session is running, or the API points at a different database. |
| `make check` fails on a fresh install | Dependency floors are unpinned; a newer toolchain may flag new issues. Report it. |

## Known limitations

- **One exchange.** Bybit only.
- **No proven edge.** The research report records that no strategy tested beat
  buy-and-hold over the period examined. Only the `1h` timeframe was swept; the other
  thirteen are untested.
- **Forex is experimental** and has never placed an order.
- **The API has no authentication outside production-like environments**, and several
  routers have none at all. See [Security](#security).
- **Small accounts are structurally limited** by exchange minimums.
- **Python dependencies are not lock-pinned.**
- **Not audited.** No third party has reviewed this code.
- **Single-node.** No HA, no failover. An outage means an unmanaged position.

## Security

**Do not expose the API to the internet.** Several routers — including `dashboard`,
`account`, `portfolio` and `analytics`, which return your balance, positions, orders and
PnL — carry no authentication dependency, and `X-API-Key` enforcement is active only in
production-like environments. `docker-compose.yml` binds everything to `127.0.0.1`; the
`make api` target does not. Reach it from another device with an SSH tunnel or an
authenticating reverse proxy, never an open port.

- **Never commit `.env`.** It is gitignored. Verify with `git check-ignore .env`.
- Grant API keys **trade** permission only — never withdrawal — and IP-restrict them.
- Rotate any key that has ever been pasted into a file, a log, a terminal recording or a
  chat.
- Secrets are typed as `SecretStr` and stripped from every log line, but treat log files
  as sensitive anyway.
- Redis ships with no password; the loopback binding is the boundary.

To report a vulnerability, see [SECURITY.md](SECURITY.md). **Please do not open a public
issue for a security problem.**

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). `make check` must pass. Non-negotiables:
`Decimal` for money, UTC-aware datetimes only, no IO in the domain layer, every order
through the risk engine, `mypy --strict`, no look-ahead.

## Licence

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE).

Apache-2.0 permits commercial and private use, modification and redistribution, and
grants a patent licence from contributors. It requires that you keep the licence and
copyright notices and state significant changes. The software is provided **"AS IS",
without warranties or conditions of any kind** — see sections 7 and 8 of the licence, and
the [trading-risk disclaimer](#️-trading-risk) above.
