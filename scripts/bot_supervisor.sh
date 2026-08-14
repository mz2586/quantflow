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

REPO="/Users/muhammadzohaib/quantflow"
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
export QF_RISK__MAX_POSITION_PCT="${QF_RISK__MAX_POSITION_PCT:-0.05}"
#
# MAX_CONCURRENT_POSITIONS is deliberately NOT raised. It is already 10 against five
# traded symbols and one position per symbol, so it cannot bind: raising it would deploy
# no additional capital and would only look like it had. Position size and the symbol
# count are what actually govern deployment.
#
# The per-order notional cap must move with position size or it silently clips the orders
# the new sizing produces: 5% of ~100k is ~5,000, exactly the old ceiling.
export QF_RISK__MAX_ORDER_NOTIONAL="${QF_RISK__MAX_ORDER_NOTIONAL:-12000}"

# Meme markets join the traded set, filtered at startup by the eligibility rules
# (liquidity, spread, staleness, volatility band, flash breaker, venue minimums).
# They compete through the same orchestrator and risk engine as everything else —
# no reserved capital, no relaxed thresholds.
export QF_BOT_MEME="${QF_BOT_MEME:-true}"

# Ticker-driven profit protection. Runs beside the candle loop: a favourable move
# mid-bar ratchets the venue-side stop instead of waiting for the bar to close.
# Stages: +0.25% breakeven+fees, +0.50% lock +0.20%, +0.75% partial 33% then trail.
export QF_INTRABAR="${QF_INTRABAR:-true}"

# Net-profit exit threshold, as NET profit after estimated round-trip costs (~0.16%).
# 0.0002 = 0.02% net, so the gross move required is ~0.18%. Lowered from 0.05% to
# realise winners earlier; deliberately NOT zero — a position must still be genuinely
# profitable after the costs of getting out of it.
export QF_MIN_NET_PROFIT_PCT="${QF_MIN_NET_PROFIT_PCT:-0.0002}"

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
