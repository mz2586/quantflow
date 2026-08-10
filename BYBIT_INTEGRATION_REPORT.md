# Bybit V5 Integration Report

**Generated:** 2026-08-10 · **Commit:** `fee993d` · **Branch:** `ai-trader`
**Network:** Bybit V5 testnet (`api-testnet.bybit.com`) · **Category:** spot

---

## Verdict

**Items 1–7 cannot pass. No API credentials are configured on this machine.**

`.env` contains empty placeholders for `QF_EXCHANGE__API_KEY` and
`QF_EXCHANGE__API_SECRET`, and no relevant environment variables are set. An API key was
provided in conversation but **no secret** — Bybit V5 signs every private request with an
HMAC-SHA256 of key *and* secret, so a key alone cannot authenticate anything.

This is not a code defect. Everything that can be verified without possessing a valid key
has been verified and passes, including the authentication path itself.

| # | Item | Status | Basis |
|---|---|---|---|
| 1 | API authentication | **Partially verified** | Signing, transport and error mapping proven end-to-end. Cannot confirm a *valid* key is accepted. |
| 2 | Account type detection | **Blocked** | Requires an authenticated call. |
| 3 | Balance verification | **Blocked** | Requires an authenticated call. |
| 4 | Position verification | **Blocked** | Requires an authenticated call. |
| 5 | Order placement | **Blocked** | Construction verified against live venue rules; submission requires credentials *and* explicit confirmation. |
| 6 | Order cancellation | **Blocked** | Nothing to cancel until something is placed. |
| 7 | Fill verification | **Blocked** | Requires an authenticated call. |
| 8 | This report | **Complete** | — |

---

## 1. Authentication — partially verified

The full auth path was exercised using a **deliberately invalid** credential pair. This
proves every link in the chain except possession of a real key:

```
supports_trading with credentials present : True
request signed and transmitted to testnet : yes
venue response  : {"retCode":10003,"retMsg":"API key is invalid."}
mapped to       : ExchangeAuthenticationError
```

Bybit rejected the key at the *application* layer, not the transport layer. That means the
request was correctly signed, correctly routed to a V5 private endpoint, and the venue's
error was correctly translated into the platform's own exception type. What remains
unproven is only that a **valid** key is accepted.

The credential-free gateway also fails closed, refusing private calls locally before any
network round trip:

```
supports_trading without credentials : False
fetch_balances                       : refused locally -> ExchangeError
fetch_open_orders                    : refused locally -> ExchangeError
raw_account_info                     : refused locally -> ExchangeError
```

That ordering matters: an unauthenticated build cannot leak a malformed private request to
the venue at all.

## 2–4. Account type, balances, positions — blocked

All three require a signed call. The code paths exist and are typed:

- `raw_account_info()` calls `privateGetV5AccountInfo` directly. There is no CCXT-unified
  equivalent, and it is needed *before* any balance is read: V5 returns **different balance
  structures for UNIFIED and CLASSIC accounts**, so a balance figure is not trustworthy
  until the account type is known.
- `fetch_balances()` maps V5 balances onto the platform's `Balance` type.
- `fetch_positions()` returns an empty list on spot rather than raising — spot has no
  positions endpoint on V5, and "no positions" is the correct answer for a spot account.

None has executed against a real account. **Treat all three as unproven.**

## 5. Order placement — construction verified, submission blocked

Order construction was validated against **live venue metadata** pulled from testnet:

```
instrument   : tick=0.1  step=0.000001  minNotional=5.0 USDT
market bid   : 65,255.70
constructed  : BUY LIMIT 0.000168 @ 32,627.80  = 5.48 USDT
  price on tick grid  : True
  quantity on step grid: True
  clears min notional : True
  Instrument.validate_order(): PASS
```

Routing and enum translation, confirmed against the V5 contract:

```
symbol  : spot = BTC/USDT      linear = BTC/USDT:USDT
category: spot = spot          futures = linear
types   : LIMIT -> limit       MARKET -> market
TIF     : GTC -> GTC           GTD -> GTC  (V5 has no GTD)
```

The order is deliberately priced 50% below the bid at minimum size. That exercises the
whole placement and cancellation path with no realistic chance of filling, so the
verification cannot accidentally take a position.

**Nothing has been submitted.** `scripts/verify_bybit.py --place-order` is gated behind
both the flag and an interactive confirmation that prints the exact order and whether the
target is testnet or mainnet.

## 6–7. Cancellation and fills — blocked

Both depend on step 5. `cancel_order` and `fetch_my_trades` are implemented and typed but
have never run against Bybit. A failed cancel is reported with an explicit instruction to
check the venue, because it is the one outcome that can leave a live order behind.

---

## What *is* proven

Read paths were verified against live Bybit V5 testnet (no credentials required):

| Check | Result |
|---|---|
| `load_instruments` | 557 instruments |
| `server_time` | clock sync verified |
| `fetch_candles` | real OHLCV, correct timestamps |
| `fetch_ticker` | bid 65,343.5 / ask 65,343.6 |
| `fetch_order_book` | levels parsed, spread correct |
| `fetch_recent_trades` | side and price parsed |
| Auth error mapping | `retCode 10003` → `ExchangeAuthenticationError` |
| Credential-free refusal | all private calls fail closed |

**1297 unit tests pass**, including 65 Bybit connector tests. `mypy --strict`, `ruff` and
`black` clean.

---

## To unblock

1. Generate a Bybit API key **pair** (key *and* secret). Grant **Read + Trade only** — no
   withdrawal permission. A key that can withdraw turns any bug or leak into a total loss
   rather than a bad trade.
2. Add an IP allowlist for this machine.
3. Write both values to `.env` (already gitignored). Do not paste them into a chat or a
   terminal that is being logged.
4. Run `python scripts/verify_bybit.py` — steps 1–5 execute automatically, read-only.
5. Run `python scripts/verify_bybit.py --place-order` for steps 5–7. It will show the exact
   order and require typing `yes`.

Start on testnet (`QF_EXCHANGE__TESTNET=true`, already set) and prove the write paths there
before a mainnet key exists.

---

## Standing constraints

- **Live trading is disabled and has never been armed.** No authenticated order has ever
  been sent to any venue, on any network.
- The five-condition interlock is unchanged and still requires an explicit human decision.
- An API key was pasted into conversation earlier in this session. If it is real it should
  be treated as compromised and rotated, regardless of anything else here.
