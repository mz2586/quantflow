# Operations runbook

Day-to-day operation of a QuantFlow session. Assumes `make install`, `make infra-up` and
`make migrate` have been done — see the [README](../README.md).

## Start

### Backtest

```bash
make backtest STRATEGY=ema_cross SYMBOL=BTC/USDT TIMEFRAME=1h \
              START=2024-01-01 END=2025-01-01
```

### Paper

```bash
make paper STRATEGY=ema_cross SYMBOL=BTC/USDT
```

### Demo, unattended

```bash
# .env must carry QF_EXCHANGE__ENV=demo and a demo key pair
nohup caffeinate -s scripts/bot_supervisor.sh > scratchpad/bot-supervisor.log 2>&1 &
```

The supervisor restarts the bot if it dies, with exponential backoff (10 s → 300 s) so a
bot failing instantly — bad credentials, venue down — cannot spin thousands of times a
minute. It takes a lock directory and **refuses to start a second instance**: two
supervisors would mean two bots trading the same account against each other.

`caffeinate` is macOS-only; on Linux drop it.

### API and dashboard

```bash
make api                     # http://localhost:8000  (loopback by default)
make dashboard               # http://localhost:5173
```

## Stop

```bash
touch scratchpad/BOT_STOP    # clean stop: waits for the current run to exit
```

This is the correct way. `BOT_STOP` is checked between runs; the supervisor clears a stale
stop file on its next start, so a leftover file does not silently prevent a restart.

Other controls, in increasing severity:

```bash
quantflow risk halt                 # stop new entries, keep existing positions and stops
python scripts/flatten_demo.py      # close open demo positions
make down                           # stop the Docker stack
```

Killing the supervisor process instead leaves the lock directory behind. Remove
`scratchpad/.bot.lock` if a legitimate restart is refused.

## Monitor

```bash
tail -f scratchpad/bot.log
quantflow risk status
curl localhost:8000/healthz      # liveness
curl localhost:8000/readyz       # readiness — checks DB and Redis
curl localhost:8000/metrics      # Prometheus
```

The dashboard's **freshness panel** states how current each source on the page is. Use it
before trusting anything else on the screen: a stale panel next to a live one is the
failure mode this exists to make visible.

## Routine checks

| Interval | Check |
|---|---|
| Each session start | `quantflow risk status` — kill switch not latched, no stale halt |
| Daily | Positions on the dashboard match the venue's own UI |
| Daily | Every open position shows a venue-side stop |
| Weekly | Log size — `scratchpad/bot.log` grows without bound and is not rotated |
| Weekly | `npm audit`, and a dependency refresh |

## Incidents

### The session died and the supervisor is backing off

Read `scratchpad/bot.log` for the last error before the exit. Common causes: credentials
rejected, venue unreachable, a price that failed tick-size validation. The supervisor will
keep retrying; if the cause is configuration it will keep failing, so fix it rather than
waiting.

### A position exists at the venue with no stop

Treat as urgent — it is an unbounded loss. Attach one from the exchange UI, or flatten.
Then look for a reconciliation error in the log; this is exactly what the reconciler exists
to catch, so its silence is itself a finding.

### The kill switch latched

It does not reset on its own, by design. Establish *why* before lifting it. Flatten
manually if positions remain. Only then restart.

### The dashboard shows no session

Either nothing is running, or the API is pointed at a different database than the bot.
Check `QF_DATABASE__*` matches in both environments.

### Orders rejected: notional below minimum

The account is too small for that instrument at your current risk limits. Not fixable in
software — see [risk.md](risk.md).

### Orders rejected on 1m or 5m bars

The fill model rejects an order exceeding 10% of a bar's volume, and a 5m bar carries
roughly a third of a 15m bar's volume. 15m is the shortest practical interval.

## Disk and log hygiene

**`scratchpad/bot.log` is not rotated and will grow to gigabytes.** Nothing in the repo
truncates it. Rotate or truncate it yourself:

```bash
: > scratchpad/bot.log        # truncate while the bot holds the handle
```

`scratchpad/`, `logs/`, `reports/` and `data/` are all gitignored. Nothing in them should
ever be committed, and nothing in them is safe to publish without reading it first.

## Upgrading

```bash
git pull
make install                 # dependencies may have moved
make migrate                 # apply any new migrations
make check                   # must pass before you restart a session
```

Stop the bot before migrating. A schema change under a running session is not tested and
not supported.
