## What and why

<!-- What changes, and why. The diff already shows what; explain the why. -->

## How it was verified

<!-- Commands run and their result. Not "should work". -->

- [ ] `make check` passes (Ruff, Black, `mypy --strict`, unit tests)
- [ ] `cd dashboard && npm run lint && npm run typecheck && npm run test && npm run build` (if the dashboard changed)
- [ ] New or changed behaviour is covered by a test that fails without this change

## Trading-safety checklist

- [ ] No bypass around the risk engine
- [ ] Live trading is no easier to reach by accident than before
- [ ] `Decimal` in the money path; no `float`
- [ ] UTC-aware datetimes only; time comes from the injected `Clock`
- [ ] No look-ahead — the backtester still exposes only closed bars
- [ ] Venue remains the source of truth for position existence and protection

## Secrets

- [ ] No API key, secret, token, account identifier or local filesystem path in the diff — including in comments
- [ ] No `.env`, log, database or `scratchpad/` file is added

## Notes for the reviewer

<!-- Anything that would be hard to see from the diff. -->
