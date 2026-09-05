#!/usr/bin/env bash
# Keep the demo bot running unattended: restart it if it dies, stop when told to.
#
# Launch detached:
#   nohup caffeinate -s scripts/bot_supervisor.sh > scratchpad/bot-supervisor.log 2>&1 &
#
# Stop it:  touch scratchpad/BOT_STOP     (clean, waits for the current run to exit)
#
# Restart backoff exists so a bot that fails instantly - bad credentials, venue down -
# does not spin thousands of times a minute filling the disk and hammering the API.

set -uo pipefail

# Repo root, derived from this script's own location so the supervisor works from any
# checkout. Override with QF_REPO if the scripts live outside the repository.
REPO="${QF_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PY="${REPO}/.venv/bin/python"
LOG="${REPO}/scratchpad/bot.log"
STOP="${REPO}/scratchpad/BOT_STOP"
LOCK="${REPO}/scratchpad/.bot.lock"

MIN_BACKOFF=10
MAX_BACKOFF=300

cd "${REPO}" || exit 1
mkdir -p "${REPO}/scratchpad"

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

# Two supervisors would mean two bots trading the same account against each other.
if ! mkdir "${LOCK}" 2>/dev/null; then
    echo "[$(stamp)] another supervisor holds ${LOCK} — refusing to start a second"
    exit 1
fi
trap 'rmdir "${LOCK}" 2>/dev/null' EXIT

# A stop file left over from last time would exit immediately and look like a crash.
if [ -f "${STOP}" ]; then
    echo "[$(stamp)] clearing stale stop file ${STOP}"
    rm -f "${STOP}"
fi

# DEMO ONLY. The launcher checks this too; checking here as well means a drifted config
# cannot start even one process.
if ! grep -qE '^QF_EXCHANGE__ENV=demo' "${REPO}/.env"; then
    echo "[$(stamp)] REFUSING: .env does not set QF_EXCHANGE__ENV=demo"
    exit 2
fi

export ENABLE_LIVE_TRADING=true
export QF_TRADING__MODE=live
export QF_TRADING__LIVE_CONFIRMATION=I_UNDERSTAND_THE_RISK

# --- Capital deployment -----------------------------------------------------
# Equity now comes from the venue balance (~100k) instead of a hardcoded 10k, so every
# percentage below applies to the real account.
#
# 5% per position, the top of the sanctioned 2-5% band. With five symbols that is at most
# 25% of the book deployed - well inside the 60% aggregate exposure cap, which therefore
# is not the binding limit here.
# 20% of the wallet per position, on explicit authorisation. Against ~49,800 that
# is ~9,960 base, and conviction lifts a top-decile candidate to ~12,950 (1.30x).
# Four times the old ~2,487.
#
# The ceiling this creates, stated plainly: three correlated slots at maximum
# conviction is ~78% of the wallet in what is effectively ONE directional crypto
# bet, because BTC/ETH/SOL/BNB/XRP move together. The correlation guard bounds
# the number of slots, not the correlation between them. A 5% adverse move in
# crypto with the book full is roughly -1,900 on the wallet.
#
# This is a deliberate risk decision, not a default.
# PER-POSITION CAP REMOVED for the 100 USDT experiment, 2026-08-21 — and this time it is
# safe, because the allocation bug is fixed. 1.0 now means "the whole 100 USDT allocation",
# not "the whole 49,612 wallet": the sizer resolves equity as min(venue, allocation) = 100,
# proven against PositionSizer output rather than startup text.
#
# ABSOLUTE CEILING: 100.00 USDT of notional. One minimum BTC lot costs 76.25 (76% of the
# allocation) and one ETH lot 23.70 (24%). Both now fit; neither can exceed 100.
#
# The gross cap moves with it because the validator requires
# max_position_pct <= max_total_exposure_pct, and a 76% position cannot exist under a 60%
# ceiling. Both are percentages OF 100 USDT, so the real exposure limit is 100 USDT.
#
# Everything protective is untouched: stops, targets, hard max loss, invalidation, loss
# acceleration, 4h stale-loser, 0.30% profit protection, correlation, 2-leg pyramiding,
# venue minimums, quantity and price precision.
export QF_RISK__MAX_POSITION_PCT="${QF_RISK__MAX_POSITION_PCT:-0.20}"
export QF_RISK__MAX_TOTAL_EXPOSURE_PCT="${QF_RISK__MAX_TOTAL_EXPOSURE_PCT:-1.0}"
#
# MAX_CONCURRENT_POSITIONS is deliberately NOT raised. It is already 10 against five
# traded symbols and one position per symbol, so it cannot bind: raising it would deploy
# no additional capital and would only look like it had. Position size and the symbol
# count are what actually govern deployment.
#
# The per-order notional cap must move with position size or it silently clips the orders
# the new sizing produces: 5% of ~100k is ~5,000, exactly the old ceiling.
# Must clear the position cap or it silently clips it: 20% of ~49,800 at 1.30x
# conviction is ~12,950, which the old 12,000 ceiling would have truncated,
# making the conviction multiplier decorative on exactly the trades it is for.
export QF_RISK__MAX_ORDER_NOTIONAL="${QF_RISK__MAX_ORDER_NOTIONAL:-20000}"

# ---------------------------------------------------------------------------
# CAPITAL BASE — the full venue USDT wallet
#
# Unset on purpose. With no allocation the launcher takes the sizing base from
# the authoritative Bybit DEMO USDT balance, which is the intent: the wallet is
# the account.
#
# The 10,000 ceiling that used to live here was never actually binding, and that
# is worth recording. `resolve_starting_equity` applied it at startup, and the
# reconciler's `anchor_cash` then re-anchored portfolio cash to the venue wallet
# on the first pass - so every percentage limit was already computed against
# ~49,774 rather than 10,000. Positions were 5% of the wallet (2,488), not 5% of
# the allocation (500). The dashboard said 10,000 while sizing used the wallet.
#
# Only the USDT balance counts. BTC and ETH sitting in the account are inventory
# on a USDT-quoted book, not buying power.
# 100 USDT EXPERIMENT, 2026-08-21. The venue wallet is ~49,400 but the bot is scoped to a
# fixed allocation, so every percentage limit is measured against 100 USDT rather than the
# wallet: starting_equity = min(venue_balance, allocation).
#
# Consequence, measured before deploying and NOT worked around: at a 20% position cap the
# largest permitted position is 20.00 USDT, while one minimum venue lot costs 76.39 USDT on
# BTC (0.001) and 23.74 USDT on ETH (0.01). Both exceed the cap, so both are refused by the
# sizer's below_venue_min_quantity path. The cap was not raised, the minimum was not
# bypassed, and quantities were not rounded up to force a fill.
export QF_BOT_EQUITY_FROM_VENUE="${QF_BOT_EQUITY_FROM_VENUE:-true}"
export QF_BOT_ALLOCATION="${QF_BOT_ALLOCATION:-1000}"

# ---------------------------------------------------------------------------
# CORRELATED POSITION CAP
#
# Raised from 2 to 3 on explicit authorisation, to lift capital utilisation.
#
# Understand what this buys and what it costs. The traded universe is BTC, ETH,
# SOL, BNB and XRP - five instruments that move as one, which is why the guard
# exists at all: five alt positions are usually one BTC position in costume. A
# third slot therefore does not add a third independent bet. It adds ~50% more
# size to the same directional bet, and drawdown scales with it.
#
# Everything else still binds: aggregate exposure, max order notional, per
# position size, venue-side stops. The third slot is a ceiling, not a quota -
# nothing fills it unless the candidate qualifies on its own merits.
export QF_RISK__MAX_CORRELATED_POSITIONS="${QF_RISK__MAX_CORRELATED_POSITIONS:-3}"

# ---------------------------------------------------------------------------
# MAKER-FIRST ENTRIES
#
# Enter passively with post-only limits instead of crossing the spread. Bybit
# charges 0.06% taker against 0.01% maker, so a round trip falls from ~0.12% to
# ~0.02%.
#
# Enabled on evidence, not preference. Session demo-10k-fresh over 11 trades:
#   gross wins   +19.60
#   gross losses -15.50
#   gross edge    +4.10   <- the strategy is profitable before costs
#   fees         -25.99   <- 6.3x the entire edge
#   net          -21.89
# Average gross edge per trade +0.37 against an average fee of -2.36. The signals
# were never the problem; the cost of acting on them was.
#
# The tradeoff is real and deliberate: a passive entry sometimes does not fill, so
# this trades LESS often, not more. Missing a setup costs nothing. Paying taker on
# every setup costs 2.36 each.
#
# Entries only - exits, stop-entries and any price a strategy chose itself are
# left untouched, because a reduce-only order waiting for a passive fill is not
# protection. Unfilled entries are abandoned after QF_RISK__ENTRY_LIMIT_MAX_BARS
# bars rather than resting on an expired setup.
export QF_RISK__MAKER_FIRST_ENTRIES="${QF_RISK__MAKER_FIRST_ENTRIES:-true}"
export QF_RISK__ENTRY_LIMIT_MAX_BARS="${QF_RISK__ENTRY_LIMIT_MAX_BARS:-3}"

# A session anchored at the allocation. Deliberately NOT the earlier
# demo-15m-20260813, whose equity curve is anchored near 49,900 — re-anchoring it
# to 10,000 would put a step change into a curve that never experienced one, and
# every return measured across that step would be meaningless. The old session's
# rows are untouched and remain queryable. Stable across restarts, so a crash
# resumes this session rather than scattering state across new ones.
# A clean session, created 2026-08-15 after archiving everything before it.
#
# Every earlier session was marked completed rather than deleted, so their trades,
# equity curves and orders remain queryable by session id — but none of it belongs
# to this run. The reconciler floors its execution lookback at session creation, so
# this session cannot adopt a fill made before it existed, which is what quietly
# imported another run's PnL into the two 10k sessions that preceded it.
# demo-100usdt-fresh was contaminated before it began: two trades of 9,842 and 8,727
# notional were opened under the allocation bug, sized off the 49K wallet. Its history is
# archived, not deleted; the clean measurement runs under a distinct id so the two cannot
# be mixed in any report.
export QF_BOT_SESSION_ID="${QF_BOT_SESSION_ID:-demo-1000usdt-clean}"

# ---------------------------------------------------------------------------
# STRATEGY ROSTER — the exact 13 from commit 493527f
#
# An A/B rollback. The expanded 43-strategy pool produced 25 trades at 32% wins,
# -74.11 gross and -148.82 net, and the question is whether the expansion caused
# it. These are the 13 that existed at 493527f ("strategy research framework with
# 11 new strategies"), read out of git rather than reconstructed - there was never
# a 17-strategy roster, and the real progression was 4 -> 13 -> 22 -> 43.
#
# The other 30 stay in the repository and the registry. They are simply not
# eligible to generate entries while this experiment runs.
#
# Only the roster changes. Timeframe, cost model, targets, maker-first, conviction
# sizing, correlation cap and position cap are all held constant, or the result
# would not attribute to anything.
export QF_BOT_STRATEGIES="${QF_BOT_STRATEGIES:-bollinger_reversion,bollinger_squeeze,donchian_breakout,dual_thrust,ema_cross,keltner_trend,macd_trend,momentum_roc,opening_range_breakout,rsi_reversion,triple_ma,volume_breakout,zscore_reversion}"

# Meme markets join the traded set, filtered at startup by the eligibility rules
# (liquidity, spread, staleness, volatility band, flash breaker, venue minimums).
# They compete through the same orchestrator and risk engine as everything else —
# no reserved capital, no relaxed thresholds.
# OFF while the entry universe is BTC/ETH: discovering meme markets that may not be
# entered costs a startup scan and adds streams for nothing.
export QF_BOT_MEME="${QF_BOT_MEME:-false}"

# Non-crypto asset classes. Bybit lists gold, silver, WTI, Brent, 193 single-name equities
# and index ETFs as ordinary USDT linear perpetuals in the same category as BTC, tagged by
# a `symbolType` field the gateway now reads. They therefore reach the existing sizing,
# reconciliation, intrabar exit and risk stack unchanged - no venue adapter is involved.
#
# Filtered at startup by the same eligibility rules as the memes (liquidity, spread,
# staleness, volatility band, flash breaker, venue minimums, order-vs-bar-volume ceiling),
# but against per-class thresholds: the meme volatility floor of 0.4% per bar would reject
# gold at 0.18% and the S&P at 0.04% - and BTC at 0.20% - so applying it unchanged would
# not filter these markets, it would delete them.
#
# Strategy families are gated per class: volume strategies are refused outside crypto
# because a synthetic equity perpetual's volume measures participation in the derivative
# rather than in the share, and swing-structure is refused because these underlyings gap
# over their own closed sessions.
# ---------------------------------------------------------------------------
# ON. These classes are discovered, subscribed, evaluated and sized exactly like
# crypto. What they cannot yet do is place an order, and that is an account state
# on Bybit's side rather than anything in this repo: the venue refuses the ORDER,
# never the data. Probed live on 2026-08-14, one unfillable post-only limit per
# class, and the answers are three SEPARATE agreements - signing one leaves the
# other two refusing:
#
#   XAU/USDT, XAG/USDT  retCode 110123  "You must agree to the Trading Terms..."
#   CL/USDT             retCode 110125  "You must agree to the Crude Oil Trading Terms..."
#   AAPL/USDT, SPY/USDT retCode 110126  "You must sign the required agreement..."
#   BTC/USDT            accepted        (crypto needs no agreement)
#
# Leaving them off was previously the only safe option, because an order rejection
# failed the whole session and the supervisor would restart-loop whenever a metal
# happened to signal. That is no longer true: a 110123/110125/110126 is now
# translated to ProductAgreementRequiredError, which quarantines just that asset
# class and lets the crypto book carry on. So enabling these costs the crypto book
# nothing, and buys real evaluation evidence on the new markets today.
#
# TO ACTUALLY TRADE THEM: sign the three product agreements in the Bybit DEMO UI
# (Account > Agreements / the one-off prompt shown when opening the market's
# trading page). No deploy is needed afterwards - the quarantine is discovered at
# runtime and clears on the next restart. Re-probe with:
# Verify against the venue rather than assuming: query the connected key's own
# permissions, do not infer them from the account type.
#
# OFF by default: crypto only.
#
# Until those agreements are signed, every non-crypto selection reaches the venue and
# comes back refused - and the engine has still spent that selection on a market it
# could not trade. Leaving the classes on costs nothing in code, but it does cost the
# crypto book the per-class stream budget and the orchestrator's attention. Turning
# them back on is these four lines and a restart.
export QF_BOT_METALS="${QF_BOT_METALS:-false}"
export QF_BOT_ENERGY="${QF_BOT_ENERGY:-false}"
export QF_BOT_EQUITIES="${QF_BOT_EQUITIES:-false}"
export QF_BOT_INDICES="${QF_BOT_INDICES:-false}"

# Crypto universe, widened from the five majors on 2026-08-16.
#
# Not a loosened filter - every gate, threshold, stop rule and cost model is unchanged.
# It is a wider net for the SAME gates: with five symbols on 15m bars the engine gets
# ~20 evaluations an hour, and the 0.40% net-edge floor is strict enough that a pool
# that small can go hours without a qualifier. Twelve roughly doubles the candidate
# flow without touching what qualifies.
#
# Ranked by measured 24h turnover on this venue (2026-08-16), floor ~10M USDT so every
# addition clears the liquidity rules on its own merits. DOGE is deliberately absent:
# it belongs to the meme universe, which is discovered separately, and listing it here
# would enter it twice.
# SUBSCRIBED set. Wider than the entry universe ON PURPOSE.
#
# WLD, FARTCOIN and LINK carry open positions taken before the universe narrowed. A symbol
# that is not subscribed gets no marks, no intrabar stop management and no reconciliation,
# so dropping them here would strand three live positions with nothing but their venue-side
# stops between them and the market. They stay subscribed so they are managed to the exit;
# QF_BOT_ENTRY_SYMBOLS below is what stops them being re-entered.
export QF_BOT_SYMBOLS="${QF_BOT_SYMBOLS:-BTC/USDT,ETH/USDT}"

# NEW-ENTRY universe, as of 2026-08-17: BTC and ETH only.
#
# Gates OPENING only. A CLOSE is never gated — a position must always be able to exit,
# whatever the universe says — and reconciliation, stop management and target management
# are untouched. Every other symbol remains in the registry and can be re-enabled by
# editing this one line.
# NEW ENTRIES PAUSED on 2026-08-19, on evidence, not on a timer.
#
# 26 trades this session: 9W/17L, GROSS -40.38, FEES 152.20, NET -192.57. Gross being
# negative is the decisive number — the trades lose money before a single fee is charged,
# so this is not a cost problem that better execution can fix.
#
# The target-ceiling deploy at 16:07 on 08-18 was given a fair sample: 10 trades, 3W/7L,
# gross -75.19, net -123.45, average hold 161.7 minutes. Holds tripled, targets became
# realistic, and gross stayed negative. That rules out target realism as the cause and
# points at the regime: BTC 15m ATR near 0.14% against a 0.075-0.11% round trip leaves no
# room for an edge to exist.
#
# This pauses OPENING only. Every open position keeps its venue stop, target, trailing,
# reconciliation and emergency handling, and both symbols stay subscribed and evaluated so
# the regime can be measured while flat. Set back to "BTC/USDT,ETH/USDT" to resume.
#
# DO NOT resume on a timer. Resume needs evidence: a materially higher volatility regime,
# or positive measured GROSS expectancy over a fresh qualifying sample.
# RESUMED 2026-08-19 by explicit instruction, overriding the evidence-based pause above.
# The operator's decision is to let live results settle it rather than wait on a volatility
# condition. The pause rationale is left in place above as the record of why it was set.
export QF_BOT_ENTRY_SYMBOLS="${QF_BOT_ENTRY_SYMBOLS:-BTC/USDT,ETH/USDT}"

# Controlled pyramiding, authorised 2026-08-18. Two entry legs per symbol, no more.
#
# A second leg is admitted ONLY when the case has genuinely changed — a different strategy
# family (the same taxonomy confluence uses), a different regime, or a score materially
# better than the thesis already open. Same family + same direction + same regime + a
# near-identical score is refused as the same opinion twice, which is what the evidence
# says a naive pyramid would buy here: 132 selections in one session, all long, four
# correlated trend families, scores inside a 0.008 band.
#
# A leg is never added to a position that is underwater. Bybit nets in one-way mode, so a
# second leg enlarges the first and moves its average entry — adding to a loser would be
# averaging down under another name.
#
# The 20% per-symbol cap is NOT raised. The legs SHARE it: the sizer subtracts what the
# symbol already holds, so leg two is sized into the room that is left.
export QF_BOT_MAX_LEGS_PER_SYMBOL="${QF_BOT_MAX_LEGS_PER_SYMBOL:-2}"
export QF_PYRAMID_MIN_SCORE_IMPROVEMENT="${QF_PYRAMID_MIN_SCORE_IMPROVEMENT:-0.02}"

# Solo-family edge bar. LOWERED 0.70% -> 0.60% on 2026-08-18, by instruction, to raise
# trade frequency.
#
# This is the bar a LONE strategy family must clear to trade without corroboration. It was
# my number, not one of the configured quality gates: I set it at twice the 0.35% edge
# floor, which was defensible but arbitrary. Measured over 24h it blocked 12 candidates
# whose edge ranged 0.4654%-0.6938% — all of which had already cleared the edge floor,
# reward:risk, liquidity, stop and target. At 0.60% nine of the twelve are admitted.
#
# Nothing else moves: the 0.35% edge floor, R:R >= 1.5, liquidity, stops, exposure,
# correlation, cooldown and the two-family confluence requirement are all unchanged. This
# only affects the case where ONE family signals alone.
#
# Recorded honestly: at the time of this change fees were 97.48 against 72.84 of gross and
# the session was net -24.64, with BTC 15m ATR at 0.14% versus a 0.16% round-trip cost.
# More trades in that regime is expected to deepen the loss, not reverse it. The lever was
# pulled on instruction after that was stated.
# SET TO 0.60% on 2026-08-19 by explicit instruction. Previously 0.70% (twice the edge
# floor), which was my number rather than a measured one.
#
# This is the bar a LONE strategy family must clear to trade without corroboration. It does
# not touch the 0.35% edge floor, R:R, liquidity, stops or the two-family confluence
# requirement — it only affects the case where ONE family signals alone. Measured over 24h
# when this was last examined, the 0.70% bar was blocking candidates whose edge ranged
# 0.4654%-0.6938%, all of which had already cleared every other gate; 0.60% admits about
# three quarters of them.
export QF_SOLO_FAMILY_MIN_NET_EDGE="${QF_SOLO_FAMILY_MIN_NET_EDGE:-0.006}"

# Two symbols per class. Four classes at two each adds eight streams to the six already
# traded, for fourteen - which is at the top of what this process keeps alive comfortably.
# Ranked by 24h turnover, so the survivors are the liquid ones.
export QF_BOT_MAX_PER_CLASS="${QF_BOT_MAX_PER_CLASS:-2}"

# Ticker-driven profit protection. Runs beside the candle loop: a favourable move
# mid-bar ratchets the venue-side stop instead of waiting for the bar to close.
# Stages: +0.25% breakeven+fees, +0.50% lock +0.20%, +0.75% partial 33% then trail.
export QF_INTRABAR="${QF_INTRABAR:-true}"

# Net-profit exit threshold, as NET profit after estimated round-trip costs (~0.16%).
#
# RAISED 0.02% -> 0.30% on 2026-08-17. At 0.02% this was not profit protection, it was a
# guillotine: the manager closed a position the instant it cleared fees by a hair, so a
# trade aiming at an ATR-based target ~1.3% away was taken at a fiftieth of it. Measured
# live, three closes fired at 0.0753%, 0.0225% and 0.0213% net — 1.4% and 3.4% of their
# target distance — none of them because the thesis had been invalidated.
#
# The evidence is sharper than that: the session's one target-hit winner (+68.88 over
# 5h06m) only ran because intrabar adoption was broken and nothing was managing it. Fixing
# adoption switched this rule on, and the next two winners were cut at +13.40 and +11.13.
#
# At 0.30% net the exit becomes what it was meant to be — a floor that protects a genuinely
# profitable position — while leaving room for the target to actually be reached. The stop,
# the ATR target, the trailing ladder and every risk control are untouched; only the point
# at which profit protection may activate has moved.
export QF_MIN_NET_PROFIT_PCT="${QF_MIN_NET_PROFIT_PCT:-0.003}"

# Time-based stale-loser exit. RAISED 1h -> 4h on 2026-08-18.
#
# At one hour this rule was destroying value rather than protecting it. Audited over the
# last 22 attributed trades it fired four times, on positions whose GROSS was -2.63, -6.22,
# -6.85 and -11.78 — essentially flat, one of them three hundredths of a percent on 8,600
# of notional — and paid a full ~6.50 exit fee each time. Combined gross -27.48 became
# combined net -53.44: the rule roughly doubled the loss and never once saved one. Three of
# the four fired at exactly 60.0 minutes, i.e. on the clock rather than on the thesis.
#
# Against that, every trade that actually made money was held 150-330 minutes, and the
# session's best trade (+68.88, exited on its real target) ran 306 minutes untouched. A 1%
# BTC move on 15m bars typically needs hours; one hour was not long enough for the expected
# edge to be tested at all.
#
# Four hours = sixteen 15m bars. This does NOT mean losers are held for four hours: the
# stop, hard max loss, loss acceleration and thesis invalidation all still fire immediately
# and are untouched. Only the TIME cutoff moved.
export QF_STALE_LOSER_AFTER_S="${QF_STALE_LOSER_AFTER_S:-14400}"

# Live rotation: the full registry. Every registered strategy competes except the
# orchestrator itself and the buy_and_hold benchmark. Set QF_BOT_STRATEGIES to a
# comma-separated list to restrict it; unset means no restriction.
#
# Deliberately unrestricted: the edge validation has not produced a report, so there is
# no evidence on which to retire anything. Narrowing the pool on smoke-test figures
# would be picking winners from noise.

backoff=${MIN_BACKOFF}
runs=0

echo "[$(stamp)] supervisor start (pid $$) — demo bot, log ${LOG}"

while true; do
    if [ -f "${STOP}" ]; then
        echo "[$(stamp)] stop file present — supervisor exiting"
        break
    fi

    runs=$((runs + 1))
    echo "[$(stamp)] starting bot (run #${runs})"
    started=$(date +%s)

    "${PY}" "${REPO}/scripts/run_demo_bot.py" >> "${LOG}" 2>&1
    rc=$?

    ran_for=$(( $(date +%s) - started ))
    echo "[$(stamp)] bot exited rc=${rc} after ${ran_for}s"

    if [ -f "${STOP}" ]; then
        echo "[$(stamp)] stop requested — not restarting"
        break
    fi

    # A refusal to arm (rc=2) is a configuration problem. Restarting cannot fix it and
    # would just loop forever against a venue that keeps saying no.
    if [ "${rc}" -eq 2 ]; then
        echo "[$(stamp)] bot refused to arm (rc=2) — configuration issue, not restarting"
        break
    fi

    # A run that survived a while was healthy; reset the backoff so one bad night does
    # not leave the next restart waiting five minutes.
    if [ "${ran_for}" -ge 120 ]; then
        backoff=${MIN_BACKOFF}
    fi

    echo "[$(stamp)] restarting in ${backoff}s"
    sleep "${backoff}"
    backoff=$(( backoff * 2 ))
    [ "${backoff}" -gt "${MAX_BACKOFF}" ] && backoff=${MAX_BACKOFF}
done

echo "[$(stamp)] supervisor done after ${runs} run(s)"
