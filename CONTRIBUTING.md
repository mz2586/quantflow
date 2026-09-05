# Contributing to QuantFlow

Thanks for considering it. This is software that places real orders on an exchange, so the
bar is deliberately higher than for a typical library — a bug here can cost someone money.

## Before you start

- **Security issues do not go in the issue tracker.** See [SECURITY.md](SECURITY.md).
- For anything larger than a bug fix, open an issue first. It is better to disagree about
  an approach in a paragraph than in a 900-line diff.
- By contributing you agree your work is licensed under [Apache-2.0](LICENSE).

## Setting up

```bash
git clone https://github.com/mz2586/quantflow.git
cd quantflow
make install          # uv venv (Python 3.12) + editable install with dev + ai extras
make env              # .env from .env.example
make infra-up         # Postgres + Redis, loopback only
make migrate
make check            # lint + types + tests. This must pass.
```

Dashboard:

```bash
make dashboard-install
cd dashboard && npm run lint && npm run typecheck && npm run test && npm run build
```

## The quality gate

`make check` runs Ruff, Black `--check`, `mypy --strict` and the unit suite. **CI runs the
same thing.** A pull request that does not pass it will not be reviewed until it does.

```bash
make fmt      # black + ruff --fix, before you commit
make lint
make type
make test
make cov      # coverage; CI fails under 80%
```

## Non-negotiable engineering rules

These are not style preferences. Each one exists because breaking it has caused a real
defect in this codebase.

1. **`Decimal` for money.** `float` appears only inside vectorised analytics, never in
   sizing, accounting, order construction or PnL.
2. **UTC-aware datetimes only.** Enforced by Ruff's `DTZ` rules. Time comes from an
   injected `Clock`, never from `datetime.now()`.
3. **No IO in the domain layer.** `domain/` is pure. Engines depend on protocols, never on
   drivers.
4. **Every order passes the risk engine.** There is no bypass path, and adding one will be
   rejected.
5. **Strict typing.** `mypy --strict` over `src/` and `tests/`. `# type: ignore` needs a
   comment explaining why.
6. **No look-ahead.** The backtester only ever exposes closed bars to a strategy. A change
   that lets a strategy see the bar it is trading is a correctness bug, not an
   optimisation.
7. **The venue is the source of truth.** For whether a position exists, whether it is
   protected, and at what price — ask the exchange, do not infer it from local state.

## Tests

- Unit tests must not touch the network, a database, or the clock.
- Mark tests that need infrastructure `@pytest.mark.integration`, and tests that hit a
  live venue `@pytest.mark.network`. Network tests are excluded everywhere by default and
  are never run in CI.
- A bug fix should come with a test that fails before it and passes after.
- Property-based tests (Hypothesis) are welcome for anything arithmetic.

## Adding a strategy

1. Subclass `Strategy` in `src/quantflow/strategy/library/`.
2. Give it a unique `strategy_id` and a one-line `description` that says what makes it
   *different from the others* — not what indicator it uses.
3. Declare a `StrategyParams` model with real bounds.
4. Register it in `library/__init__.py`.
5. Add tests covering warmup, the entry condition, the exit condition and at least one
   degenerate series (flat prices, a gap, insufficient history).

Please do not submit a fourteenth variation on an EMA crossover. The library spans
distinct families on purpose: a leaderboard populated with variations on one idea ranks
parameter choices while appearing to rank ideas.

**Do not submit backtest results as evidence a strategy is good.** They will not be
treated as such. See `docs/research/` for why.

## Commits and pull requests

- Conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- Explain **why** in the body. The diff already shows what.
- One logical change per PR.
- Never commit `.env`, credentials, logs, databases, or anything under `scratchpad/`.
- Never commit an API key, an account identifier, or a local filesystem path — including
  in comments. Check your diff before you push.

## Things that will be declined

- A bypass around the risk engine, however convenient.
- Defaults that make live trading easier to reach by accident.
- Performance claims in the README or in strategy docstrings.
- New runtime dependencies without a clear justification.
- `float` in the money path.
