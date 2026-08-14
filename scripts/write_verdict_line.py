#!/usr/bin/env python
"""One glanceable line: is the product net-positive out-of-sample, and did anything survive.

Read from reports/edge-validation.json using the same survivor rule as the full report
(SURVIVE_NET, and the same minimum trade count), so the summary line and the tables can
never disagree.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from write_validation_report import SOURCE, SURVIVE_NET

MIN_TRADES = 30


def main() -> int:
    if not SOURCE.exists():
        print(f"{datetime.now(UTC).isoformat()}  NO RESULTS — {SOURCE} missing")
        return 1

    payload = json.loads(SOURCE.read_text())
    results = payload.get("results", {})
    failures = payload.get("failures") or {}

    def oos(name: str) -> dict:
        return results.get(name, {}).get("out_of_sample", {}) or {}

    standalone = [n for n in results if n != "orchestrator"]
    survivors = [
        n
        for n in standalone
        if isinstance(oos(n).get("net_pnl"), int | float)
        and float(oos(n)["net_pnl"]) > SURVIVE_NET
        and int(oos(n).get("trades", 0)) >= MIN_TRADES
    ]

    orch = oos("orchestrator")
    if "net_pnl" in orch:
        net = float(orch["net_pnl"])
        sign = "+" if net > 0 else "-"
        orch_text = (
            f"ORCHESTRATOR OOS net {sign}{abs(net):.2f} over {orch.get('trades', 0)} trades "
            f"({'NET-POSITIVE' if net > 0 else 'NET-NEGATIVE'} after all costs)"
        )
    else:
        orch_text = "ORCHESTRATOR did not produce a result"

    survivor_text = f"survivors: {', '.join(survivors)}" if survivors else "survivors: NONE"
    failure_text = f" | failed: {len(failures)}" if failures else ""

    print(
        f"{datetime.now(UTC).isoformat()}  {orch_text} | {survivor_text} "
        f"({len(standalone)} standalone tested){failure_text}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
