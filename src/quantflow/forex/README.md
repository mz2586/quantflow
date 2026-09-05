# QuantFlow Forex worker — EXPERIMENTAL

> **⚠️ EXPERIMENTAL — this package has never placed an order.**
>
> Nothing here has been run against a live or a demo FX account. No FX credentials exist
> in this project. It is excluded from the supported surface of the v0.1.0 release and is
> not covered by the demo-venue validation the crypto path has. Treat it as a design under
> review, not as working software.


FX execution for QuantFlow. A transport-agnostic domain layer plus **two** interchangeable
venue adapters, so that Forex does not depend on any one broker — or on any one operating
system.

> **Nothing here has been run against a live or demo account.** No FX credentials exist in
> this project. Every adapter is written to its venue's published API and tested against
> fakes. The first real order will be the first real order.

---

## Which route should I take?

| | **OANDA v20** (`oanda_worker.py`) | **Bybit MT5** (`mt5_worker.py`) |
|---|---|---|
| Runs on headless Linux | **Yes** — plain HTTPS | **No** — Windows only |
| Runs on macOS | Yes | No |
| Needs a GUI terminal / gateway process | No | Yes, the MT5 terminal |
| Transport | REST + HTTP streaming | Local IPC via the `MetaTrader5` package |
| Free demo | Yes, self-service practice account | Yes, MT5 demo server |
| Position sizing unit | units of base currency (converted to lots here) | lots |

**If you are deploying to a Linux VPS or a container, use OANDA.** The MT5 adapter is kept
because Bybit's FX product only exists there, and because an operator who already has that
account should not have to abandon it — but it cannot run without Windows.

---

## Route A — OANDA v20 (recommended; Linux-native)

### 1. Open a free practice account

<https://www.oanda.com/> → open an **fxTrade Practice** account. Signup asks for country of
residence, name, email, phone and a password. No KYC documents and no deposit. OANDA's help
pages state a practice account does not expire.

### 2. Check your division first — this is the one real blocker

OANDA's developer docs state the v20 API is available to **all divisions except OANDA Global
Markets and OANDA TMS BROKERS S.A.** Your country of residence at signup decides your
division, and OANDA routes some regions (including the UAE) to OANDA Global Markets, whose
platform list does not include the REST API.

So, before writing any config: open the account, go to the account **HUB → Manage API
Access**, and try to generate a personal access token. **If that page is not there, OANDA is
not available to you** and the fallbacks worth trying, in order, are Capital.com (REST + WS,
free self-service demo), cTrader Open API via an IC Markets or Pepperstone demo (note that
Spotware must approve your app registration, with no published turnaround), or MetaApi.cloud
as a paid REST bridge onto any MT5 broker demo.

### 3. Collect two values

* the **personal access token** from Manage API Access
* the **account id**, of the form `001-001-1234567-001`

### 4. Set the environment

```bash
export QF_OANDA_TOKEN='<your personal access token>'
export QF_OANDA_ACCOUNT_ID='001-001-1234567-001'
export QF_OANDA_ENVIRONMENT='practice'   # default; 'live' is refused unless you opt in
```

`QF_OANDA_ALLOW_LIVE=1` is the only way to point this at real money, and you should not set
it. The worker refuses to construct against the live host without it.

### 5. Verify

```python
from quantflow.forex.oanda_worker import OandaCredentials, OandaWorker, capabilities

print(capabilities().describe())  # says exactly what is missing, if anything
worker = OandaWorker(OandaCredentials.from_env())
print(worker.get_account())
print([i.symbol for i in worker.get_symbols()][:7])
```

Practice host is `https://api-fxpractice.oanda.com`, streaming is
`https://stream-fxpractice.oanda.com`. Using the live host with a practice token returns
*"Insufficient authorization to perform request."*

### Things that will bite you

* **OANDA trades units of base currency, not lots.** One unit of `EUR_USD` is one euro; a
  standard lot is 100,000 units. `units_from_lots` / `lots_from_units` are the only places
  that conversion happens. Direction is the **sign** of `units` — there is no side field.
* **Tick value is only correct out of the box when the quote currency is your account
  currency** (any `*_USD` pair on a USD account). For everything else, pass
  `home_conversion_factor` from `/v3/accounts/{id}/pricing?includeHomeConversions=true` into
  `instrument_from_oanda`, or sizing will be wrong by the FX rate.
* **A `201` on an order does not mean a fill.** `orderFillTransaction` is only present when
  the order filled immediately; `submit_order` reports `PLACED` rather than assuming.
* **Rate limits:** 120 REST requests/second per IP, at most 2 new connections/second, at most
  20 concurrent streams. The price stream emits a heartbeat every 5 seconds — use its absence
  as the liveness signal.
* **Reconciliation:** persist `lastTransactionID` and poll `get_fills_since_id`. The
  time-ranged `get_fills` works but costs an extra request per page.

---

## Route B — Bybit MT5 (Windows only)

Bybit's FX (`EURUSD+`, `GBPUSD+`, …) is a **CFD product on MetaTrader 5**. It is not on
Bybit's V5 REST API: the `fx`, `forex`, `tradfi` and `cfd` categories all return
`retCode=10001 "Illegal category"`, `/v5/tradfi/*` returns 404, and CCXT carries no FX
endpoints. It needs a **separate MT5 account** with its own login id, server and password.

The `MetaTrader5` PyPI package ships **win_amd64 wheels only** and drives a locally-running
MT5 terminal over IPC. It is therefore deliberately **not** in `pyproject.toml` — adding it
would break `pip install` on Linux and macOS — and is imported lazily at connect time.

### Operator steps

1. **Create the Bybit MT5 CFD account** in the Bybit app or web UI. It is separate from your
   spot/derivatives account and issues its own credentials.
2. **Choose a DEMO server.** Note the login id (numeric), the password and the exact server
   name. Do not start on a live server.
3. **Provision a Windows host** — a VM or VPS, Windows 10/11 or Server, x64.
4. **Install the MetaTrader 5 terminal**, log in with those credentials, and confirm the
   symbols appear in Market Watch.
5. **Enable algo trading**: Tools → Options → Expert Advisors → *Allow algorithmic trading*.
   Orders are silently refused without it.
6. **Install Python 3.12 (64-bit)** and the bridge: `py -3.12 -m pip install MetaTrader5`.
7. **Set the environment** on that host:

   ```bat
   set QF_MT5_LOGIN=12345678
   set QF_MT5_PASSWORD=your-password
   set QF_MT5_SERVER=Bybit-Demo
   rem optional:
   set QF_MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe
   set QF_MT5_TIMEOUT_MS=60000
   ```

8. **Check, then serve:**

   ```bat
   py -3.12 -m quantflow.forex.mt5_worker --check
   py -3.12 -m quantflow.forex.mt5_worker --host 127.0.0.1 --port 8787
   ```

   `--check` prints every blocker and exits non-zero when the host cannot run — safe to gate
   a deployment script on.

### Safety rails

* The worker **refuses to construct** against a server whose name does not contain `demo`,
  and **refuses to trade** an account whose MT5 `trade_mode` is `REAL`, unless
  `QF_MT5_ALLOW_LIVE=1` is set deliberately.
* The terminal is single-threaded and holds state per process, so the HTTP service funnels
  every call through one worker thread. Do not call `MT5Worker` from multiple threads.

---

## Layout

```
forex/
  instruments.py  ForexInstrument, TradeMode, pip/point maths, MAJORS ranking
  sizing.py       lots_for_risk — risk / (stop points x value per point)
  costs.py        spread, commission, slippage, swap incl. the triple-swap day
  sessions.py     the FX week, Asian/London/NY classification, weekend + Friday close
  exits.py        intrabar stop/target resolution, ambiguity flagged not hidden
  plan.py         plan_trade — the LONG / SHORT / NO-TRADE gate
  protocol.py     ForexBroker (the interface), the DTOs, reconciliation, staleness
  oanda_worker.py OANDA v20 transport   (Linux OK)
  mt5_worker.py   MetaTrader 5 transport (Windows only)
```

Everything above `protocol.py` is the domain layer: it imports no venue SDK, holds no
connection, and does not know which transport is wired in.

## The interface

Every transport implements exactly these ten methods:

```python
get_account()      get_symbols()     subscribe_ticks()  get_bars()      submit_order()
modify_stop()      close_position()  get_orders()       get_positions() get_fills()
```

Session management (`connect` / `disconnect`) sits on a separate optional `ForexConnection`
protocol, so a stateless REST transport is not forced to fake a session it does not have.

## Why FX sizing is not crypto sizing

The crypto formula `quantity = risk / (price * stop_pct)` returns a number of base units.
Handing that to an FX venue that measures volume in 100,000-unit lots opens a position five
orders of magnitude too large. FX sizing always goes:

```
lots = account_risk / (stop_distance_points * value_per_point_per_lot)
```

`value_per_point_per_lot` comes from the venue's tick value, which already folds in contract
size and the account-currency conversion — so JPY pairs (3 digits, 0.001 point) and 5-digit
pairs fall out of the same expression with no special-casing. A size that cannot be expressed
at or above `min_lot` is **rejected with a reason**, never silently rounded to zero.
