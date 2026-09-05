# Bybit V5 Integration Report

**Generated:** 2026-08-10 · **Branch:** `ai-trader`
**Network:** Bybit V5 **MAINNET** (`api.bybit.com`) · **Category:** spot

---

## Verdict

**Items 1–4 and 7 pass against a real account. Items 5–6 are blocked — the account holds
no funds.**

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | API authentication | **PASS** | Signed request accepted by mainnet |
| 2 | Account type detection | **PASS** | `unifiedMarginStatus=5`, `marginMode=REGULAR_MARGIN` |
| 3 | Balance verification | **PASS** | Call succeeds; **0 currencies funded** |
| 4 | Position verification | **PASS** | 0 open |
| 5 | Order placement | **BLOCKED** | Zero balance — an order cannot be funded |
| 6 | Order cancellation | **BLOCKED** | Nothing to cancel |
| 7 | Fill verification | **PASS (empty)** | `fetch_my_trades` returns cleanly, 0 history |
| 8 | This report | **Complete** | — |

Items 5 and 6 are blocked by an **empty account, not by code**. Every other private
endpoint responds correctly, which means the signing, routing, category selection and
response parsing all work; there is simply nothing to trade with.

---

## 1. Authentication — PASS

Diagnosed by error code rather than guesswork. Bybit distinguishes:

| Code | Meaning |
|---|---|
| `10003` | key string not recognised |
| `10004` | signature error — **the key is valid**, the secret is wrong |
| `10010` | IP not allowlisted |

The first attempt returned `10003`. A bounded set of key variants then produced `10004`,
which proved the key was correct and isolated the fault to the secret. Both values had the
same defect: a character rendering as capital `I` in the source screenshot is actually a
lowercase `l`. Applying that one substitution to the secret authenticated immediately.

Worth recording as a process lesson: **credentials transcribed from a screenshot are
unreliable**. The key and secret were exactly the right length (18 and 36) and cleanly
alphanumeric, so every structural check passed while the values were still wrong.

## 2. Account type — PASS

```
unifiedMarginStatus : 5
marginMode          : REGULAR_MARGIN
dcpStatus           : OFF
spotHedgingStatus   : OFF
isMasterTrader      : false
```

This is a **Unified Trading Account**, not a Classic account. It matters because V5 returns
different balance structures per account type, which is why the type is established before
any balance is read rather than after.

`REGULAR_MARGIN` means cross/portfolio margin is not enabled — relevant if derivatives are
ever traded, irrelevant for spot.

## 3. Balances — PASS, account is empty

```
0 currencies with a non-zero balance
```

The endpoint responds correctly and parses cleanly. **The account is unfunded.** This is
the binding constraint on items 5–6.

## 4. Positions — PASS

`0 open`. Expected: this is a spot configuration, and `fetch_positions` returns an empty
list on spot by design rather than raising.

## 5–6. Order placement and cancellation — BLOCKED

Not attempted. With a zero balance the venue would reject on insufficient funds, which
would test nothing about the adapter.

What *is* verified is order **construction**, against live venue metadata:

```
instrument   : tick=0.1  step=0.000001  minNotional=5.0 USDT
constructed  : BUY LIMIT 0.000168 @ 32,627.80  = 5.48 USDT
  price on tick grid   : True
  quantity on step grid: True
  clears min notional  : True
  Instrument.validate_order(): PASS
```

To complete these two items the account needs roughly **10 USDT** — enough to clear the
5 USDT minimum notional with headroom. The verifier places a limit order 50% below the
bid at minimum size, so it exercises the full path with no realistic chance of filling.

## 7. Fills — PASS (empty)

`fetch_my_trades` returns cleanly with 0 history, consistent with an account that has never
traded. The call path, authentication and parsing are verified; fill *parsing* is not,
because there are no fills to parse.

---

## Security findings

These are not incidental — they are the most consequential items in this report.

**1. The key is over-scoped.** The grant includes `Wallet - Account Transfer`,
`Subaccount Transfer`, `Convert`, `Earn` and `Contracts`. This platform uses **none** of
them; it needs *SPOT - Trade* and read access only. Transfer permission means a leaked key
can move funds rather than merely trade them — the difference between a bad trade and a
drained account.

**2. Both values are exposed.** They were shared as a screenshot, so they exist in the
conversation history and as an image file on disk. They should be rotated.

**3. No IP allowlist** was configured. Bybit supports one and it is the cheapest available
mitigation.

**4. Verification ran against mainnet.** `QF_EXCHANGE__TESTNET=false`. Read-only, so
nothing was at risk, but the write paths should be proven on testnet before a funded
mainnet key exists.

---

## Standing constraints

- **Live trading is disabled and has never been armed.** No order has been sent to any
  venue on any network.
- The five-condition interlock is unchanged and still requires an explicit human decision.
- Nothing in this verification placed, modified or cancelled an order.

## To finish items 5–6

1. Rotate the key. Scope it to **SPOT - Trade + read only** — no wallet transfer.
2. Add an IP allowlist.
3. Fund the account with ~10 USDT, or switch to testnet and use faucet funds.
4. Run `python scripts/verify_bybit.py --place-order`. It prints the exact order and the
   network, and requires typing `yes` before anything is sent.
