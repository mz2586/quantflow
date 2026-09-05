# Risk model

> **Read this before enabling live trading.** The controls below are real and they are
> enforced, but they are bounds on *this software's* behaviour. They cannot bound the
> market. A gap through a stop, an exchange outage, or a venue that rejects your closing
> order will all lose you money that no limit in this file prevents.

## The one invariant

**Every order passes the risk engine.** There is no bypass path, and a pull request adding
one will be declined. The engine can refuse, and refusal is a normal, expected outcome —
a session that places no orders for a day is working, not broken.

## Hard limits

All of these are mandatory. The engine refuses to start if they are absent.

| Control | Variable | What it does |
|---|---|---|
| Per-position cap | `QF_RISK__MAX_POSITION_PCT` | Max notional of one position as a fraction of equity |
| Portfolio cap | `QF_RISK__MAX_TOTAL_EXPOSURE_PCT` | Max aggregate gross notional as a fraction of equity |
| Position count | `QF_RISK__MAX_CONCURRENT_POSITIONS` | Hard ceiling on simultaneous open positions |
| Daily loss halt | `QF_RISK__MAX_DAILY_LOSS_PCT` | Blocks **new entries** for the remainder of the UTC day |
| Drawdown kill switch | `QF_RISK__MAX_DRAWDOWN_PCT` | **Latches.** Flattens and stops opening anything |
| Mandatory stop | `QF_RISK__REQUIRE_STOP_LOSS` | Rejects any entry without a stop |
| Default stop distance | `QF_RISK__DEFAULT_STOP_LOSS_PCT` | Used when a strategy supplies no stop |
| Leverage | `QF_RISK__MAX_LEVERAGE` | Ceiling; `1` means unlevered |
| Order notional | `QF_RISK__MAX_ORDER_NOTIONAL` | Absolute cap on a single order |

**The shipped values are examples, not advice.** Set them yourself, deliberately.

### Halt vs kill switch

They are different and the difference matters:

- The **daily loss halt** stops new entries and resets at the UTC day boundary. Existing
  positions keep their stops. Lift it early with `quantflow risk resume`.
- The **drawdown kill switch latches**. It does not reset on its own. It is the control of
  last resort and it is meant to require a human.

```bash
quantflow risk status     # current state, and why
quantflow risk halt       # engage manually — stops new entries, keeps positions
quantflow risk resume     # lift a daily halt
```

## Position sizing

Sizing runs before every entry and takes the **minimum** of: the per-position cap, the
remaining portfolio exposure headroom, the single-order notional cap, and what the stop
distance permits. A strategy's `conviction` (0–1) may scale down within those bounds; it
can never scale beyond them.

The result is then reconciled against the instrument's own constraints — see below — and
if the outcome is below the venue's minimum, **the trade is skipped**. That is correct
behaviour, not a bug to work around.

## Venue constraints

Bybit publishes a `tick_size`, a `lot_size` and a `min_notional` per instrument.
QuantFlow enforces all three locally:

- Prices snap to tick size before submission. An unsnapped price is rejected by the venue
  with `price … is not a multiple of tick …` and fails the session.
- Quantities snap to lot size.
- Orders below `min_notional` are refused locally with a named rule rather than sent.

**Small accounts are structurally limited by this.** If your risk limits size a position
below an instrument's minimum, that instrument is untradeable for you at that equity. No
amount of configuration changes that — it is the exchange's rule, not QuantFlow's.

## Protection

Stops and targets attach to the **entry order** (`stopLoss` / `takeProfit`, with
`tpslMode=Partial`), so a filled entry is protected at the venue from the moment it fills
rather than after a follow-up round trip. `set_trading_stop` handles post-entry attachment
and adjustment.

"Protected" means **the exchange says so**. The reconciler re-reads positions and their
stops from the venue; it never infers protection from local state. A position the venue
reports without a stop is treated as unprotected regardless of what the local record says.

Between bar closes, `position/intrabar.py` manages protection on ticks — staged stop
advancement, partial closes and reversals — because a 15-minute bar is a long time to wait
when price is moving against an open position.

## Live-trading interlocks

Three independent, deliberate settings are required. Missing any one is a startup refusal,
not a silent fallback:

1. `QF_TRADING__MODE=live`
2. `QF_TRADING__LIVE_CONFIRMATION=I_UNDERSTAND_THE_RISK`
3. `QF_EXCHANGE__ENV=mainnet` with a mainnet key pair

Additionally: the settings validator refuses `live` combined with the exchange testnet;
demo and mainnet credentials live in **separate fields** so switching environments cannot
carry a key across; and `scripts/run_demo_bot.py` calls `refuse_mainnet()` *before any
client is constructed*, exiting with status 2.

The confirmation token is a fixed public constant. It is an **intent gate against
accident**, not a secret and not a security control.

## What the risk engine cannot protect you from

- **Gaps.** A stop is an instruction to trade at market once a level trades. It is not a
  guaranteed price.
- **Exchange outages.** If the venue is unreachable, positions are unmanaged.
- **Liquidation** on a leveraged account, which the exchange performs on its own schedule.
- **A strategy that is simply wrong.** Limits bound the size of the loss per trade, not the
  number of losing trades.
- **Model risk in the backtest.** Simulated fills are optimistic by construction.
- **You.** Widening a limit after a loss is how limits stop working.
