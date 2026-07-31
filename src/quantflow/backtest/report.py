"""Self-contained HTML backtest reports.

Plotly figures are embedded into a single file with no external assets, so a report can be
opened offline, emailed, or checked into a research log and still render years later.

The report deliberately leads with **drawdown and trade count**, not with the return.
A large return from four trades is noise, and a report that shows the return first invites
exactly that mistake.
"""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantflow.backtest.engine import BacktestResult, rejection_reasons, signal_summary
from quantflow.backtest.metrics import PerformanceMetrics, is_statistically_thin
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.portfolio import EquityPoint
from quantflow.domain.positions import ClosedTrade

logger = get_logger(__name__)

#: Muted palette that reads on both light and dark backgrounds.
COLOUR_EQUITY = "#2563eb"
COLOUR_DRAWDOWN = "#dc2626"
COLOUR_WIN = "#16a34a"
COLOUR_LOSS = "#dc2626"
COLOUR_GRID = "#e5e7eb"
COLOUR_TEXT = "#111827"

#: Above this share of refused signals, the risk configuration is materially shaping
#: the result and the report says so.
HIGH_REJECTION_SHARE = 0.5

#: Above this, the strategy is effectively always invested and should be compared
#: against buy-and-hold rather than against zero.
NEAR_FULL_EXPOSURE = Decimal("0.95")

#: A decline deeper than this is one few operators sit through without intervening.
SEVERE_DRAWDOWN = Decimal("0.4")


def equity_figure(curve: Sequence[EquityPoint]) -> go.Figure:
    """Equity curve with drawdown beneath it, sharing an x axis."""
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32],
        subplot_titles=("Equity", "Drawdown"),
    )
    times = [point.timestamp for point in curve]

    figure.add_trace(
        go.Scatter(
            x=times,
            y=[float(point.equity) for point in curve],
            name="Equity",
            line={"color": COLOUR_EQUITY, "width": 2},
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Equity %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            y=[float(point.cash) for point in curve],
            name="Cash",
            line={"color": COLOUR_GRID, "width": 1, "dash": "dot"},
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Cash %{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=times,
            # Plotted negative and filled so the depth reads immediately as a loss.
            y=[-float(point.drawdown_pct) * 100 for point in curve],
            name="Drawdown",
            fill="tozeroy",
            line={"color": COLOUR_DRAWDOWN, "width": 1},
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Drawdown %{y:.2f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    figure.update_yaxes(title_text="Equity", row=1, col=1, gridcolor=COLOUR_GRID)
    figure.update_yaxes(title_text="Drawdown %", row=2, col=1, gridcolor=COLOUR_GRID)
    figure.update_xaxes(gridcolor=COLOUR_GRID)
    figure.update_layout(
        height=560,
        margin={"l": 60, "r": 30, "t": 50, "b": 40},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"color": COLOUR_TEXT, "family": "system-ui, -apple-system, sans-serif"},
    )
    return figure


def trade_distribution_figure(trades: Sequence[ClosedTrade]) -> go.Figure:
    """Histogram of per-trade net PnL, split by outcome."""
    wins = [float(trade.net_pnl) for trade in trades if trade.is_win]
    losses = [float(trade.net_pnl) for trade in trades if not trade.is_win]

    figure = go.Figure()
    if wins:
        figure.add_trace(go.Histogram(x=wins, name="Wins", marker_color=COLOUR_WIN, opacity=0.8))
    if losses:
        figure.add_trace(
            go.Histogram(x=losses, name="Losses", marker_color=COLOUR_LOSS, opacity=0.8)
        )
    figure.update_layout(
        height=340,
        barmode="overlay",
        title="Trade PnL distribution",
        xaxis_title="Net PnL",
        yaxis_title="Trades",
        margin={"l": 60, "r": 30, "t": 50, "b": 40},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"color": COLOUR_TEXT, "family": "system-ui, -apple-system, sans-serif"},
    )
    figure.update_xaxes(gridcolor=COLOUR_GRID)
    figure.update_yaxes(gridcolor=COLOUR_GRID)
    return figure


def cumulative_trades_figure(trades: Sequence[ClosedTrade]) -> go.Figure:
    """Cumulative realised PnL, trade by trade.

    Reveals whether a result came from a steady edge or from one or two outliers — the
    equity curve alone can hide that.
    """
    figure = go.Figure()
    if trades:
        ordered = sorted(trades, key=lambda trade: trade.exit_time)
        running = ZERO
        cumulative: list[float] = []
        for trade in ordered:
            running += trade.net_pnl
            cumulative.append(float(running))
        figure.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative) + 1)),
                y=cumulative,
                mode="lines+markers",
                name="Cumulative PnL",
                line={"color": COLOUR_EQUITY, "width": 2},
                marker={"size": 4},
            )
        )
    figure.update_layout(
        height=340,
        title="Cumulative realised PnL by trade",
        xaxis_title="Trade number",
        yaxis_title="Cumulative net PnL",
        margin={"l": 60, "r": 30, "t": 50, "b": 40},
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"color": COLOUR_TEXT, "family": "system-ui, -apple-system, sans-serif"},
    )
    figure.update_xaxes(gridcolor=COLOUR_GRID)
    figure.update_yaxes(gridcolor=COLOUR_GRID)
    return figure


def _metric_card(label: str, value: str, *, tone: str = "neutral") -> str:
    return (
        f'<div class="card {tone}">'
        f'<div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(value)}</div>'
        f"</div>"
    )


def _tone_for(value: Decimal, *, higher_is_better: bool = True) -> str:
    if value == ZERO:
        return "neutral"
    positive = value > ZERO
    return "good" if positive == higher_is_better else "bad"


def _summary_cards(metrics: PerformanceMetrics) -> str:
    # Drawdown and trade count lead deliberately: a large return from four trades is
    # noise, and leading with the return invites exactly that misreading.
    cards = [
        _metric_card(
            "Max drawdown",
            f"{metrics.max_drawdown_pct:.2%}",
            tone=_tone_for(-metrics.max_drawdown_pct),
        ),
        _metric_card("Trades", str(metrics.trade_count)),
        _metric_card(
            "Total return",
            f"{metrics.total_return_pct:.2%}",
            tone=_tone_for(metrics.total_return_pct),
        ),
        _metric_card("CAGR", f"{metrics.cagr:.2%}", tone=_tone_for(metrics.cagr)),
        _metric_card("Sharpe", f"{metrics.sharpe_ratio:.2f}", tone=_tone_for(metrics.sharpe_ratio)),
        _metric_card(
            "Sortino", f"{metrics.sortino_ratio:.2f}", tone=_tone_for(metrics.sortino_ratio)
        ),
        _metric_card("Calmar", f"{metrics.calmar_ratio:.2f}"),
        _metric_card("Win rate", f"{metrics.win_rate:.2%}"),
        _metric_card("Profit factor", f"{metrics.profit_factor:.2f}"),
        _metric_card("Expectancy", f"{metrics.expectancy:,.2f}"),
        _metric_card("Fees paid", f"{metrics.total_fees:,.2f}"),
        _metric_card("Exposure", f"{metrics.exposure_pct:.1%}"),
    ]
    return "\n".join(cards)


def _trades_table(trades: Sequence[ClosedTrade], *, limit: int = 100) -> str:
    if not trades:
        return "<p class='muted'>No closed trades.</p>"

    rows = []
    for index, trade in enumerate(sorted(trades, key=lambda item: item.exit_time)[:limit], 1):
        tone = "good" if trade.is_win else "bad"
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(str(trade.symbol))}</td>"
            f"<td>{html.escape(trade.side.value)}</td>"
            f"<td>{trade.quantity:,.6f}</td>"
            f"<td>{trade.entry_price:,.2f}</td>"
            f"<td>{trade.exit_price:,.2f}</td>"
            f"<td class='{tone}'>{trade.net_pnl:,.2f}</td>"
            f"<td class='{tone}'>{trade.return_pct:.2%}</td>"
            f"<td>{trade.entry_time:%Y-%m-%d %H:%M}</td>"
            f"<td>{trade.exit_time:%Y-%m-%d %H:%M}</td>"
            "</tr>"
        )
    truncated = (
        f"<p class='muted'>Showing the first {limit} of {len(trades)} trades.</p>"
        if len(trades) > limit
        else ""
    )
    return f"""
    <table>
      <thead><tr>
        <th>#</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Exit</th>
        <th>Net PnL</th><th>Return</th><th>Opened</th><th>Closed</th>
      </tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    {truncated}
    """


def _warnings(result: BacktestResult, metrics: PerformanceMetrics) -> str:
    """Surface the caveats that decide whether the numbers mean anything."""
    notes: list[str] = []

    if is_statistically_thin(metrics):
        notes.append(
            f"Only {metrics.trade_count} closed trades. Below ~30 the metrics are "
            "dominated by chance; treat this as a smoke test, not evidence of an edge."
        )
    if result.rejected_signals and not result.orders:
        notes.append(
            f"All {len(result.rejected_signals)} signals were refused by the risk engine "
            "and no orders were placed. Check the rejection reasons below."
        )
    elif result.rejected_signals:
        share = len(result.rejected_signals) / max(1, len(result.signals))
        if share > HIGH_REJECTION_SHARE:
            notes.append(
                f"{share:.0%} of signals were refused by the risk engine. The strategy is "
                "being materially constrained by the risk configuration."
            )
    if metrics.exposure_pct > NEAR_FULL_EXPOSURE:
        notes.append(
            "The strategy was in the market almost continuously; its return is close to "
            "buy-and-hold and should be compared against it."
        )
    if metrics.max_drawdown_pct > SEVERE_DRAWDOWN:
        notes.append(
            f"Max drawdown reached {metrics.max_drawdown_pct:.1%}. Few operators sit "
            "through a decline that deep without intervening."
        )
    if not notes:
        return ""
    items = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
    return f'<section class="warnings"><h2>Caveats</h2><ul>{items}</ul></section>'


def _rejections(result: BacktestResult) -> str:
    reasons = rejection_reasons(result)
    if not reasons:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(reason)}</td><td>{count}</td></tr>"
        for reason, count in list(reasons.items())[:15]
    )
    return f"""
    <section>
      <h2>Risk rejections</h2>
      <table><thead><tr><th>Reason</th><th>Count</th></tr></thead><tbody>{rows}</tbody></table>
    </section>
    """


STYLES = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: #111827; background: #f9fafb; line-height: 1.5;
}
h1 { margin: 0 0 4px; font-size: 24px; }
h2 { margin: 32px 0 12px; font-size: 17px; font-weight: 600; }
.subtitle { color: #6b7280; margin: 0 0 24px; font-size: 14px; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; }
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 14px; }
.card-label { font-size: 11px; text-transform: uppercase; letter-spacing: .04em; color: #6b7280; }
.card-value { font-size: 20px; font-weight: 600; margin-top: 4px;
              font-variant-numeric: tabular-nums; }
.card.good .card-value { color: #16a34a; }
.card.bad .card-value { color: #dc2626; }
.chart { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
         padding: 8px; margin-top: 12px; }
table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb;
        border-radius: 8px; overflow: hidden; font-size: 13px; }
th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid #f3f4f6; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
thead th { background: #f9fafb; font-weight: 600; font-size: 11px;
           text-transform: uppercase; letter-spacing: .04em; color: #6b7280; }
tbody tr:last-child td { border-bottom: none; }
td.good { color: #16a34a; } td.bad { color: #dc2626; }
.muted { color: #6b7280; font-size: 13px; }
.warnings { background: #fffbeb; border: 1px solid #fcd34d; border-radius: 8px;
            padding: 4px 20px 12px; margin-top: 24px; }
.warnings h2 { margin-top: 16px; }
.warnings li { margin: 6px 0; font-size: 14px; }
.meta { font-size: 13px; color: #6b7280; margin-top: 32px;
        border-top: 1px solid #e5e7eb; padding-top: 12px; }
.overflow { overflow-x: auto; }
@media (max-width: 640px) { body { padding: 16px; } }
"""


def render_html(result: BacktestResult, *, title: str | None = None) -> str:
    """Render a complete, self-contained HTML report."""
    metrics = result.metrics()
    heading = title or f"{result.strategy_id} — backtest"
    symbols = ", ".join(str(symbol) for symbol in result.config.symbols)
    period = (
        f"{metrics.start:%Y-%m-%d} to {metrics.end:%Y-%m-%d}"
        if metrics.start and metrics.end
        else "unknown period"
    )

    equity_html = equity_figure(result.equity_curve).to_html(
        full_html=False, include_plotlyjs="cdn", config={"displaylogo": False}
    )
    distribution_html = trade_distribution_figure(result.closed_trades).to_html(
        full_html=False, include_plotlyjs=False, config={"displaylogo": False}
    )
    cumulative_html = cumulative_trades_figure(result.closed_trades).to_html(
        full_html=False, include_plotlyjs=False, config={"displaylogo": False}
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(heading)}</title>
<style>{STYLES}</style>
</head>
<body>
<h1>{html.escape(heading)}</h1>
<p class="subtitle">
  {html.escape(symbols)} · {html.escape(result.config.timeframe.value)} · {html.escape(period)}
  · {result.bars_processed:,} bars
</p>

<div class="cards">{_summary_cards(metrics)}</div>

{_warnings(result, metrics)}

<h2>Equity and drawdown</h2>
<div class="chart">{equity_html}</div>

<h2>Trade analysis</h2>
<div class="chart">{cumulative_html}</div>
<div class="chart">{distribution_html}</div>

<h2>Closed trades</h2>
<div class="overflow">{_trades_table(result.closed_trades)}</div>

{_rejections(result)}

<h2>Configuration</h2>
<div class="overflow"><table><tbody>
  <tr><td>Strategy</td><td>{html.escape(result.strategy_id)}</td></tr>
  <tr><td>Parameters</td><td>{html.escape(json.dumps(result.strategy_params))}</td></tr>
  <tr><td>Starting equity</td>
      <td>{result.config.starting_equity:,.2f}</td></tr>
  <tr><td>Signals / orders / rejected</td>
      <td>{html.escape(json.dumps(signal_summary(result)))}</td></tr>
</tbody></table></div>

<p class="meta">
  Run {html.escape(result.run_id)}
  · generated {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}
  · completed in {result.duration_seconds:.2f}s · QuantFlow
</p>
</body>
</html>"""


def write_report(result: BacktestResult, directory: Path, *, filename: str | None = None) -> Path:
    """Write the report to disk and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    name = filename or f"backtest-{result.strategy_id}-{result.run_id[:8]}.html"
    path = directory / name
    path.write_text(render_html(result), encoding="utf-8")
    logger.info("report.written", path=str(path), run_id=result.run_id)
    return path


def render_summary_json(result: BacktestResult) -> dict[str, Any]:
    """Machine-readable summary for the API and for run comparison."""
    metrics = result.metrics()
    return {
        "run_id": result.run_id,
        "strategy_id": result.strategy_id,
        "params": result.strategy_params,
        "symbols": [str(symbol) for symbol in result.config.symbols],
        "timeframe": result.config.timeframe.value,
        "status": result.status.value,
        "bars": result.bars_processed,
        "duration_seconds": result.duration_seconds,
        "metrics": metrics.to_dict(),
        "signals": signal_summary(result),
        "rejections": rejection_reasons(result),
        "statistically_thin": is_statistically_thin(metrics),
    }
