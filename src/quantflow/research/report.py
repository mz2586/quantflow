"""Research reports: a leaderboard and a per-strategy breakdown.

Two outputs, both self-contained. Markdown for reading in a terminal or a pull request,
HTML for reading properly. Neither loads anything from a network: a report that needs a
CDN stops rendering the moment it is opened somewhere without one, and a research record
that decays is not a record.

The report states its own assumptions at the top — period, symbols, costs, thresholds —
because a leaderboard without them is a set of numbers with no meaning attached.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from decimal import Decimal

from quantflow.research.leaderboard import METRICS, RankedEntry, leaderboard
from quantflow.research.runner import BENCHMARK_STRATEGY_ID, ResearchOutcome


def _fmt(value: Decimal, spec: str) -> str:
    """Format a Decimal with a format spec, tolerating non-finite values."""
    try:
        return spec.format(value)
    except (ValueError, ArithmeticError):
        return "—"


def _verdict(entry: RankedEntry) -> str:
    """Accepted / rejected / benchmark label."""
    if entry.entry.is_benchmark:
        return "benchmark"
    return "accepted" if entry.entry.accepted else "rejected"


def build_markdown(outcome: ResearchOutcome) -> str:
    """A complete Markdown research report."""
    board = leaderboard(outcome)
    return "\n".join(
        [
            *_md_header(outcome),
            *_md_thresholds(outcome),
            *_md_verdict(board),
            *_md_leaderboard(board),
            *_md_ranks(board),
            *_md_rejections(board),
            *_md_detail(board),
            *_md_failures(outcome),
            *_md_footnotes(),
        ]
    )


def _md_header(outcome: ResearchOutcome) -> list[str]:
    """Title and run provenance."""
    config = outcome.config
    lines: list[str] = []
    lines.append("# Strategy Research Report")
    lines.append("")
    lines.append(
        f"**Period:** {outcome.period_start[:10]} → {outcome.period_end[:10]} · "
        f"**Timeframe:** {config.timeframe.value} · "
        f"**Symbols:** {', '.join(str(s) for s in config.symbols)}"
    )
    lines.append("")
    lines.append(f"**Costs:** {config.costs.summary}")
    lines.append("")
    bars = " · ".join(f"{symbol} {count:,}" for symbol, count in outcome.bars_per_symbol.items())
    lines.append(f"**Bars:** {bars}")
    lines.append("")
    lines.append(
        f"**Runs:** {len(outcome.runs)} completed, {len(outcome.failures)} failed, "
        f"in {outcome.duration_seconds:.1f}s"
    )
    lines.append("")
    return lines


def _md_thresholds(outcome: ResearchOutcome) -> list[str]:
    """The gate that was applied."""
    config = outcome.config
    lines: list[str] = []
    lines.append("## Acceptance thresholds")
    lines.append("")
    lines.append("| Criterion | Requirement |")
    lines.append("|---|---|")
    for name, requirement in config.thresholds.describe().items():
        lines.append(f"| {name} | {requirement} |")
    lines.append("")
    return lines


def _md_verdict(board: Sequence[RankedEntry]) -> list[str]:
    """The headline: did anything pass."""
    lines: list[str] = []
    accepted = [entry for entry in board if entry.entry.accepted and not entry.entry.is_benchmark]
    lines.append("## Verdict")
    lines.append("")
    if accepted:
        names = ", ".join(f"`{entry.entry.strategy_id}`" for entry in accepted)
        lines.append(
            f"**{len(accepted)} of {len(board) - 1} strategies passed** every threshold on "
            f"every symbol: {names}."
        )
    else:
        lines.append(
            f"**No strategy passed.** All {len(board) - 1} non-benchmark strategies failed "
            "at least one threshold on at least one symbol. Per-strategy reasons are below."
        )
    lines.append("")
    return lines


def _md_leaderboard(board: Sequence[RankedEntry]) -> list[str]:
    """The ranked table."""
    lines: list[str] = []
    lines.append("## Leaderboard")
    lines.append("")
    lines.append(
        "Ranked by the mean of the six per-metric ranks. A rejected strategy always sorts "
        "below an accepted one, whatever its score."
    )
    lines.append("")
    header = "| # | Strategy | Verdict | " + " | ".join(m.label for m in METRICS)
    lines.append(header + " | vs hold | Score |")
    lines.append("|---:|---|---|" + "---:|" * (len(METRICS) + 2))
    for entry in board:
        cells = [
            str(entry.position),
            f"`{entry.entry.strategy_id}`",
            _verdict(entry),
            *(_fmt(metric.extract(entry.entry), metric.fmt) for metric in METRICS),
            "—" if entry.entry.excess_return is None else f"{entry.entry.excess_return:+.2%}",
            f"{entry.composite:.2f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _md_ranks(board: Sequence[RankedEntry]) -> list[str]:
    """Where each strategy placed on each individual metric."""
    lines: list[str] = []
    lines.append("## Per-metric ranks")
    lines.append("")
    lines.append("| Strategy | " + " | ".join(m.label for m in METRICS) + " |")
    lines.append("|---|" + "---:|" * len(METRICS))
    for entry in board:
        cells = [f"`{entry.entry.strategy_id}`"] + [
            str(entry.ranks[metric.key]) for metric in METRICS
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _md_rejections(board: Sequence[RankedEntry]) -> list[str]:
    """Every failure, with the number that caused it."""
    lines: list[str] = []
    lines.append("## Rejections")
    lines.append("")
    rejected = [entry for entry in board if not entry.entry.accepted]
    if not rejected:
        lines.append("None.")
    else:
        lines.append(
            "Every failure, with the number that caused it. Kept in the report on purpose: "
            "a record of what was tried and why it failed is what stops the same idea being "
            "re-tested next quarter."
        )
        lines.append("")
        for entry in rejected:
            lines.append(
                f"### `{entry.entry.strategy_id}` — "
                f"passed {entry.entry.symbols_accepted}/{entry.entry.symbols_tested} symbols"
            )
            lines.append("")
            for run in entry.entry.runs:
                if run.accepted:
                    lines.append(f"- **{run.symbol}** — accepted")
                    continue
                for rejection in run.screen.rejections:
                    lines.append(f"- **{run.symbol}** — {rejection.code.value}: {rejection.detail}")
            lines.append("")
    return lines


def _md_detail(board: Sequence[RankedEntry]) -> list[str]:
    """Full per-symbol results."""
    lines: list[str] = []
    lines.append("## Per-symbol detail")
    lines.append("")
    lines.append(
        "| Strategy | Symbol | Net return | Profit factor | Sharpe | Max DD | Win rate | "
        "Trades | Fees | Verdict |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for entry in board:
        for run in entry.entry.runs:
            metrics = run.metrics
            lines.append(
                f"| `{run.strategy_id}` | {run.symbol} | "
                f"{metrics.total_return_pct:.2%} | {metrics.profit_factor:.2f} | "
                f"{metrics.sharpe_ratio:.2f} | {metrics.max_drawdown_pct:.2%} | "
                f"{metrics.win_rate:.2%} | {metrics.trade_count} | "
                f"{metrics.total_fees:.2f} | "
                f"{'accepted' if run.accepted else 'rejected'} |"
            )
    lines.append("")
    return lines


def _md_failures(outcome: ResearchOutcome) -> list[str]:
    """Runs that crashed, recorded rather than silently dropped."""
    if not outcome.failures:
        return []
    lines: list[str] = ["## Failed runs", ""]
    for failure in outcome.failures:
        lines.append(f"- `{failure.strategy_id}` on {failure.symbol}: {failure.error}")
    lines.append("")
    return lines


def _md_footnotes() -> list[str]:
    """How to read the table without over-reading it."""
    lines: list[str] = []
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        f"- `{BENCHMARK_STRATEGY_ID}` pays the same fees and slippage as every other row. "
        "Where it lands in the table is the most informative line in the report."
    )
    lines.append(
        "- A strategy is accepted only if it passed on **every** symbol. Passing on one "
        "market and failing on the rest is a strategy that found one favourable regime."
    )
    lines.append(
        "- These are **in-sample** results over a single historical period. Passing the gate "
        "makes a strategy a candidate for walk-forward validation, not a candidate for capital."
    )
    lines.append("")
    return lines


def build_html(outcome: ResearchOutcome) -> str:
    """A self-contained HTML research report."""
    board = leaderboard(outcome)
    config = outcome.config

    rows: list[str] = []
    for entry in board:
        verdict = _verdict(entry)
        css = {"accepted": "ok", "rejected": "bad", "benchmark": "bench"}[verdict]
        cells = "".join(
            f"<td class='num'>{html.escape(_fmt(metric.extract(entry.entry), metric.fmt))}</td>"
            for metric in METRICS
        )
        excess = "—" if entry.entry.excess_return is None else f"{entry.entry.excess_return:+.2%}"
        rows.append(
            f"<tr class='{css}'>"
            f"<td class='num'>{entry.position}</td>"
            f"<td class='id'>{html.escape(entry.entry.strategy_id)}</td>"
            f"<td><span class='pill {css}'>{verdict}</span></td>"
            f"{cells}"
            f"<td class='num'>{html.escape(excess)}</td>"
            f"<td class='num'>{entry.composite:.2f}</td>"
            "</tr>"
        )

    rejection_blocks: list[str] = []
    for entry in board:
        if entry.entry.accepted:
            continue
        items = []
        for run in entry.entry.runs:
            for rejection in run.screen.rejections:
                items.append(
                    f"<li><b>{html.escape(str(run.symbol))}</b> — "
                    f"<code>{html.escape(rejection.code.value)}</code>: "
                    f"{html.escape(rejection.detail)}</li>"
                )
        if items:
            rejection_blocks.append(
                f"<h3>{html.escape(entry.entry.strategy_id)}</h3><ul>{''.join(items)}</ul>"
            )

    threshold_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td class='num'>{html.escape(requirement)}</td></tr>"
        for name, requirement in config.thresholds.describe().items()
    )
    metric_headers = "".join(f"<th class='num'>{html.escape(m.label)}</th>" for m in METRICS)
    accepted_count = sum(
        1 for entry in board if entry.entry.accepted and not entry.entry.is_benchmark
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strategy Research Report</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ background:#09090b; color:#e4e4e7; margin:0; padding:2rem;
    font:14px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
  .wrap {{ max-width:1200px; margin:0 auto; }}
  h1 {{ font-size:1.5rem; margin:0 0 .35rem; }}
  h2 {{ font-size:1.05rem; margin:2.25rem 0 .6rem; color:#a1a1aa;
    text-transform:uppercase; letter-spacing:.06em; }}
  h3 {{ font-size:.95rem; margin:1.1rem 0 .35rem;
    font-family:ui-monospace,monospace; color:#93c5fd; }}
  .meta {{ color:#a1a1aa; font-size:.85rem; }}
  .verdict {{ margin:1rem 0; padding:.85rem 1rem; border-radius:6px;
    border:1px solid #27272a; background:#111114; }}
  .verdict.none {{ border-color:#7f1d1d; background:#1a0f10; }}
  table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
  th,td {{ padding:.4rem .55rem; border-bottom:1px solid #27272a; text-align:left; }}
  th {{ color:#71717a; font-weight:600; font-size:.72rem;
    text-transform:uppercase; letter-spacing:.05em; }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums;
    font-family:ui-monospace,monospace; }}
  .id {{ font-family:ui-monospace,monospace; color:#93c5fd; }}
  .pill {{ font-size:.68rem; padding:.1rem .45rem; border-radius:99px;
    text-transform:uppercase; letter-spacing:.04em; }}
  .pill.ok {{ background:#064e3b; color:#6ee7b7; }}
  .pill.bad {{ background:#450a0a; color:#fca5a5; }}
  .pill.bench {{ background:#1e3a5f; color:#93c5fd; }}
  tr.bench td {{ background:#0d1420; }}
  code {{ font-family:ui-monospace,monospace; color:#fbbf24; font-size:.85em; }}
  ul {{ margin:.3rem 0 .3rem 1.1rem; padding:0; }}
  li {{ margin:.15rem 0; color:#d4d4d8; }}
  .scroll {{ overflow-x:auto; }}
  footer {{ margin-top:2.5rem; color:#52525b; font-size:.78rem;
    border-top:1px solid #27272a; padding-top:1rem; }}
</style></head><body><div class="wrap">

<h1>Strategy Research Report</h1>
<p class="meta">
  {html.escape(outcome.period_start[:10])} → {html.escape(outcome.period_end[:10])} ·
  {html.escape(config.timeframe.value)} ·
  {html.escape(", ".join(str(s) for s in config.symbols))} ·
  {len(outcome.runs)} runs in {outcome.duration_seconds:.1f}s
</p>
<p class="meta"><b>Costs:</b> {html.escape(config.costs.summary)}</p>

<div class="verdict {"" if accepted_count else "none"}">
  {(
    f"<b>{accepted_count} strategy(ies) passed</b> every threshold on every symbol."
    if accepted_count else
    "<b>No strategy passed.</b> Every non-benchmark strategy failed at least one "
    "threshold on at least one symbol."
  )}
</div>

<h2>Acceptance thresholds</h2>
<div class="scroll"><table>
<thead><tr><th>Criterion</th><th class="num">Requirement</th></tr></thead>
<tbody>{threshold_rows}</tbody></table></div>

<h2>Leaderboard</h2>
<div class="scroll"><table>
<thead><tr><th class="num">#</th><th>Strategy</th><th>Verdict</th>{metric_headers}
<th class="num">vs hold</th><th class="num">Score</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>

<h2>Rejections</h2>
{"".join(rejection_blocks) or "<p class='meta'>None.</p>"}

<footer>
  Ranked by the mean of six per-metric ranks; rejected strategies always sort below
  accepted ones.
  <code>{html.escape(BENCHMARK_STRATEGY_ID)}</code> pays the same fees and slippage as
  every other row.
  A strategy is accepted only if it passed on <b>every</b> symbol.
  These are in-sample results over one historical period: passing this gate makes a
  strategy a candidate for walk-forward validation, not a candidate for capital.
</footer>
</div></body></html>"""


def build_json(outcome: ResearchOutcome) -> str:
    """Machine-readable results, for diffing one research run against another."""
    board = leaderboard(outcome)
    payload = {
        "period": {"start": outcome.period_start, "end": outcome.period_end},
        "timeframe": outcome.config.timeframe.value,
        "symbols": [str(symbol) for symbol in outcome.config.symbols],
        "costs": {
            "name": outcome.config.costs.name,
            "summary": outcome.config.costs.summary,
        },
        "thresholds": outcome.config.thresholds.describe(),
        "bars_per_symbol": outcome.bars_per_symbol,
        "leaderboard": [
            {
                "position": entry.position,
                "strategy_id": entry.entry.strategy_id,
                "accepted": entry.entry.accepted,
                "is_benchmark": entry.entry.is_benchmark,
                "composite": round(entry.composite, 4),
                "ranks": entry.ranks,
                "net_return": str(entry.entry.net_return),
                "profit_factor": str(entry.entry.profit_factor),
                "sharpe_ratio": str(entry.entry.sharpe_ratio),
                "max_drawdown": str(entry.entry.max_drawdown),
                "win_rate": str(entry.entry.win_rate),
                "trade_count": entry.entry.trade_count,
                "excess_return": (
                    None if entry.entry.excess_return is None else str(entry.entry.excess_return)
                ),
                "worst_symbol_return": str(entry.entry.worst_symbol_return),
                "rejections": [
                    {
                        "symbol": str(run.symbol),
                        "code": rejection.code.value,
                        "detail": rejection.detail,
                    }
                    for run in entry.entry.runs
                    for rejection in run.screen.rejections
                ],
            }
            for entry in board
        ],
        "failures": [
            {"strategy_id": f.strategy_id, "symbol": str(f.symbol), "error": f.error}
            for f in outcome.failures
        ],
    }
    return json.dumps(payload, indent=2)


__all__ = ["build_html", "build_json", "build_markdown"]
