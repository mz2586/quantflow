#!/usr/bin/env bash
# Run the edge validation, then render the markdown report from its JSON.
#
# The two steps are chained here rather than in the caller so the report still gets
# written when the terminal that launched this is long gone. Intended to be started
# detached:
#
#   nohup caffeinate -i scripts/chain_report.sh > scratchpad/validate.log 2>&1 &

set -uo pipefail

REPO="/Users/muhammadzohaib/quantflow"
PY="${REPO}/.venv/bin/python"
JSON="${REPO}/reports/edge-validation.json"
LOCK="${REPO}/scratchpad/.validate.lock"

cd "${REPO}" || exit 1
mkdir -p "${REPO}/scratchpad" "${REPO}/reports"

stamp() { date -u "+%Y-%m-%dT%H:%M:%SZ"; }

# Atomic: mkdir fails if another run already holds it. A 24-month sweep is hours long
# and two of them against the same output file would interleave writes.
if ! mkdir "${LOCK}" 2>/dev/null; then
    echo "[$(stamp)] another validation run holds ${LOCK} — refusing to start a second"
    exit 1
fi
trap 'rmdir "${LOCK}" 2>/dev/null' EXIT

echo "[$(stamp)] chain start (pid $$)"
echo "[$(stamp)] step 1/2: scripts/validate_edge_parallel.py"

# The parallel driver imports its methodology from validate_edge.py and only decides which
# process each candidate runs in. Smoke mode must never be inherited from the environment.
unset QF_VALIDATE_SMOKE_BARS
"${PY}" "${REPO}/scripts/validate_edge_parallel.py"
rc=$?

echo "[$(stamp)] validate_edge_parallel.py exited ${rc}"

if [ "${rc}" -ne 0 ]; then
    echo "[$(stamp)] validation did not succeed — not writing a report from a partial run"
    exit "${rc}"
fi

if [ ! -s "${JSON}" ]; then
    echo "[$(stamp)] ${JSON} missing or empty despite a clean exit — nothing to render"
    exit 1
fi

echo "[$(stamp)] step 2/2: scripts/write_validation_report.py"
"${PY}" "${REPO}/scripts/write_validation_report.py"
wrc=$?
echo "[$(stamp)] write_validation_report.py exited ${wrc}"

# Glanceable top line, appended so an earlier run's verdict is never overwritten.
if [ "${wrc}" -eq 0 ]; then
    "${PY}" "${REPO}/scripts/write_verdict_line.py" >> "${REPO}/scratchpad/VERDICT.txt" 2>&1
    echo "[$(stamp)] verdict appended to scratchpad/VERDICT.txt"
fi

echo "[$(stamp)] chain done"
exit "${wrc}"
