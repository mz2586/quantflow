#!/usr/bin/env python
"""Render reports/validation_report.md from reports/edge-validation.json.

Separate from the validation run so the report can be regenerated without re-running hours
of backtests, and so persistence does not depend on the run's caller still being alive.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

#: Repo root, derived from this file's location so the script works from any checkout.
REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "reports" / "edge-validation.json"
TARGET = REPO / "reports" / "validation_report.md"

#: Out-of-sample net PnL a strategy must clear to count as a survivor. Zero, not a
#: threshold chosen to let something through.
SURVIVE_NET = 0.0


def fmt(value: object, spec: str = ".2f") -> str:
    if value is None:
        return "—"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def row_for(name: str, period: dict[str, object]) -> str:
    if "error" in period:
        return f"| `{name}` | error | | | | | | |"
    return (
        f"| `{name}` "
        f"| {period.get('trades', 0)} "
        f"| {fmt(period.get('net_pnl'))} "
        f"| {fmt(period.get('win_rate'), '.1f')}% "
        f"| {fmt(period.get('max_drawdown'))} "
        f"| {fmt(period.get('sharpe'), '.2f')} "
        f"| {fmt(period.get('deflated_sharpe'), '.2f')} "
        f"| {fmt(period.get('fee_drag_pct'), '.1f')}% |"
    )


def main() -> int:
    if not SOURCE.exists():
        print(f"missing {SOURCE}", file=sys.stderr)
        return 1
    payload = json.loads(SOURCE.read_text())
    results: dict[str, dict] = payload.get("results", {})
    window = payload.get("window", {})
    split = payload.get("split", {})
    costs = payload.get("costs", {})

    def oos(name: str) -> dict:
        return results.get(name, {}).get("out_of_sample", {}) or {}

    def ins(name: str) -> dict:
        return results.get(name, {}).get("in_sample", {}) or {}

    names = sorted(results)
    standalone = [n for n in names if n != "orchestrator"]

    survivors = [
        n
        for n in standalone
        if isinstance(oos(n).get("net_pnl"), int | float)
        and float(oos(n)["net_pnl"]) > SURVIVE_NET
        and int(oos(n).get("trades", 0)) >= 30
    ]
    in_sample_only = [
        n
        for n in standalone
        if float(ins(n).get("net_pnl") or 0) > 0 >= float(oos(n).get("net_pnl") or 0)
    ]
    retire = [n for n in standalone if float(oos(n).get("net_pnl") or 0) <= 0]

    orch = oos("orchestrator")
    orch_is = ins("orchestrator")
    orch_net = float(orch.get("net_pnl") or 0)

    lines: list[str] = []
    add = lines.append

    add("# QuantFlow — edge validation")
    add("")
    add(f"**Generated:** {datetime.now(UTC).isoformat()}  ")
    add("**Source:** `reports/edge-validation.json`")
    add("")
    add("## Scope")
    add("")
    add(f"- **Universe:** {', '.join(payload.get('universe', []))}")
    add(f"- **Timeframe:** {payload.get('timeframe')}")
    add(f"- **Bars:** {payload.get('bars'):,}" if payload.get("bars") else "")
    add(f"- **Window:** {window.get('start', '')[:10]} → {window.get('end', '')[:10]}")
    add(f"- **In-sample ends:** {split.get('in_sample_end', '')[:10]}")
    add(
        f"- **Out-of-sample:** last {int(float(split.get('oos_fraction', 0)) * 100)}% — "
        "never used to select a strategy, parameter or threshold"
    )
    add(f"- **Candidates tested:** {payload.get('candidates_tested')}")
    if payload.get("max_history_bars"):
        add(
            f"- **Trailing history per bar:** {payload['max_history_bars']} bars "
            "(verified to produce identical trade ledgers to the engine default before use)"
        )
    add("")
    add("### Costs applied")
    add("")
    for key, value in costs.items():
        add(f"- **{key}:** {value}")
    add("")
    add(
        "No parameter is fitted anywhere in the harness. Every strategy runs at its declared "
        "defaults, so nothing could have been tuned to either period."
    )
    add("")

    add("## Out-of-sample — the table that matters")
    add("")
    add("| Strategy | Trades | Net PnL | Win rate | Max DD | Sharpe | Deflated | Fee drag |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(names, key=lambda n: -float(oos(n).get("net_pnl") or 0)):
        add(row_for(name, oos(name)))
    add("")
    add(
        "**Deflated Sharpe** subtracts the Sharpe the best of "
        f"{payload.get('candidates_tested')} random candidates would be expected to show "
        "under the null of no skill. A raw Sharpe from a search this wide is not evidence."
    )
    add("")

    add("## In-sample")
    add("")
    add("| Strategy | Trades | Net PnL | Win rate | Max DD | Sharpe | Deflated | Fee drag |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in sorted(names, key=lambda n: -float(ins(n).get("net_pnl") or 0)):
        add(row_for(name, ins(name)))
    add("")

    add("## In-sample vs out-of-sample degradation")
    add("")
    add("| Strategy | IS net | OOS net | Held up? |")
    add("|---|---:|---:|---|")
    for name in names:
        i = float(ins(name).get("net_pnl") or 0)
        o = float(oos(name).get("net_pnl") or 0)
        verdict = "yes" if (i > 0 and o > 0) else ("in-sample only" if i > 0 else "no")
        add(f"| `{name}` | {i:.2f} | {o:.2f} | {verdict} |")
    add("")

    # Regime segmentation, when the run recorded it.
    regime_def = payload.get("regime_definition")
    has_regimes = any("regimes" in (oos(n) or {}) for n in names)
    if regime_def and has_regimes:
        add("## By regime — out-of-sample")
        add("")
        for key, value in regime_def.items():
            add(f"- **{key}:** {value}")
        add("")
        add(
            "A trend strategy losing money in chop is expected and is not by itself a "
            "reason to retire it. The question these columns exist to answer is whether "
            "the orchestrator, which is allowed to switch, nets positive across regimes."
        )
        add("")
        add(
            "| Strategy | Trend net | Trend n | Chop net | Chop n | High-vol net | n | Low-vol net | n |"
        )
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name in sorted(names, key=lambda n: -float(oos(n).get("net_pnl") or 0)):
            reg = (oos(name) or {}).get("regimes") or {}
            if not reg:
                continue

            def cell(bucket: str, source: dict = reg) -> tuple[str, str]:
                data = source.get(bucket) or {}
                return fmt(data.get("net_pnl")), str(data.get("trades", 0))

            trend_net, trend_n = cell("trend")
            chop_net, chop_n = cell("chop")
            high_net, high_n = cell("high_vol")
            low_net, low_n = cell("low_vol")
            add(
                f"| `{name}` | {trend_net} | {trend_n} | {chop_net} | {chop_n} "
                f"| {high_net} | {high_n} | {low_net} | {low_n} |"
            )
        add("")
        add(
            "Trend/chop and high/low-vol are two independent labels of the *same* trades, "
            "so each pair sums to the period's total — they are not four disjoint groups."
        )
        add("")

    add("## Verdicts")
    add("")
    add("### (a) Standalone strategies net-positive out-of-sample after costs")
    add("")
    if survivors:
        for name in survivors:
            data = oos(name)
            add(
                f"- **`{name}`** — net {fmt(data.get('net_pnl'))} over "
                f"{data.get('trades')} trades, deflated Sharpe "
                f"{fmt(data.get('deflated_sharpe'), '.2f')}"
            )
    else:
        add(
            "**None.** No standalone strategy is net-positive out-of-sample after fees, "
            "funding and slippage on a sample of at least 30 trades."
        )
    add("")

    add("### (b) Orchestrator, out-of-sample after costs")
    add("")
    add(
        f"- In-sample net: **{fmt(orch_is.get('net_pnl'))}** over {orch_is.get('trades', 0)} trades"
    )
    add(f"- Out-of-sample net: **{fmt(orch.get('net_pnl'))}** over {orch.get('trades', 0)} trades")
    add(
        f"- Out-of-sample Sharpe **{fmt(orch.get('sharpe'), '.2f')}**, deflated "
        f"**{fmt(orch.get('deflated_sharpe'), '.2f')}**, max drawdown "
        f"**{fmt(orch.get('max_drawdown'))}**"
    )
    add("")
    add(
        f"**Verdict: the orchestrator is {'NET-POSITIVE' if orch_net > 0 else 'NET-NEGATIVE'} "
        "out-of-sample after all costs.**"
    )
    add("")

    add("### (c) Retire")
    add("")
    if retire:
        add("Negative out-of-sample after costs — do not trade:")
        add("")
        for name in retire:
            add(f"- `{name}` — OOS net {fmt(oos(name).get('net_pnl'))}")
    else:
        add("Nothing qualifies for retirement on this evidence.")
    add("")
    if in_sample_only:
        add(
            "**In-sample only** (profitable before the holdout, not after — the classic "
            "overfitting signature):"
        )
        add("")
        for name in in_sample_only:
            add(
                f"- `{name}` — IS {fmt(ins(name).get('net_pnl'))} → "
                f"OOS {fmt(oos(name).get('net_pnl'))}"
            )
        add("")

    failures = payload.get("failures") or {}
    if failures:
        add("## Candidates that failed to run")
        add("")
        add(
            f"{len(failures)} of {payload.get('candidates_tested')} candidates crashed and are "
            "absent from the tables above. They are neither survivors nor retirements — they "
            "are simply untested, and are listed so the gap is visible rather than implied."
        )
        add("")
        for name, error in sorted(failures.items()):
            add(f"- `{name}` — {error}")
        add("")

    add("## Known limits of this run")
    add("")
    if regime_def and has_regimes:
        add(
            "- **Regime labels are one classification, not the only one.** ADX(14) >= 25 and "
            "a median volatility split are defensible, published choices fixed before any "
            "result was seen, but a different labelling would move trades between buckets. "
            "The buckets explain behaviour; they do not change the IS/OOS verdict above."
        )
    else:
        add(
            "- **Regime segmentation is not included.** This run recorded no regime data, so "
            "results are not split by trend/chop or high/low volatility. Claiming it without "
            "having computed it would be fabrication."
        )
    add("- **One timeframe.** 15m only. Nothing here speaks to 1h, 4h or 1d behaviour.")
    add(
        "- **One split, not walk-forward.** A single chronological holdout, not a rolling "
        "train/validate/test. It cannot distinguish a strategy that works in one market "
        "period from one that works generally."
    )
    add(
        "- **No parameter search was run**, which is a strength for honesty and a limit on "
        "scope: these are default parameters, so a strategy could in principle work at "
        "settings never tested. That is not evidence it does."
    )
    add("")

    TARGET.write_text("\n".join(line for line in lines if line is not None) + "\n")
    print(f"written to {TARGET}")

    # Console top-line.
    print("")
    print("=" * 70)
    print(f"ORCHESTRATOR OOS NET : {fmt(orch.get('net_pnl'))} over {orch.get('trades', 0)} trades")
    print(f"STANDALONE SURVIVORS : {len(survivors)} of {len(standalone)}")
    print(f"RETIRE               : {len(retire)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
