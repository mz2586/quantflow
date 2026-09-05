#!/usr/bin/env python
"""Write a paper-session log from the live paper session.

Output goes to `reports/` (gitignored), never to the repository root. This is a record
of one simulated session, not a performance claim, and it is deliberately not written
anywhere that would publish it as one.

Reads the database rather than any in-memory engine state, so the report reflects what
was actually persisted — the same source the dashboard serves. A report built from a
running process's memory would disagree with the dashboard the moment either restarts.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import text

from quantflow.core.config import get_settings
from quantflow.core.precision import ZERO
from quantflow.persistence.database import Database

SESSION = sys.argv[1] if len(sys.argv) > 1 else "paper-live"
OPENING_EQUITY = Decimal("10000")


async def main() -> int:
    """Render the report. Returns the closed-trade count."""
    db = Database.from_settings(get_settings())
    async with db.read_session() as s:
        row = (
            await s.execute(
                text("""
                SELECT count(*) AS n,
                       coalesce(sum(CASE WHEN net_pnl>0 THEN 1 ELSE 0 END),0) AS wins,
                       coalesce(sum(net_pnl),0) AS net,
                       coalesce(sum(fees),0) AS fees,
                       coalesce(max(net_pnl),0) AS best,
                       coalesce(min(net_pnl),0) AS worst,
                       coalesce(sum(CASE WHEN net_pnl>0 THEN net_pnl ELSE 0 END),0) AS gross_win,
                       coalesce(abs(sum(CASE WHEN net_pnl<=0 THEN net_pnl ELSE 0 END)),0)
                           AS gross_loss,
                       coalesce(avg(EXTRACT(EPOCH FROM (exit_time-entry_time))/3600),0) AS avg_hours
                FROM closed_trades WHERE session_id=:sid
                """),
                {"sid": SESSION},
            )
        ).one()
        eq = (
            await s.execute(
                text("""
                SELECT coalesce(max(drawdown_pct),0) AS maxdd, count(*) AS pts,
                       coalesce(max(equity),0) AS peak
                FROM equity_snapshots WHERE session_id=:sid
                """),
                {"sid": SESSION},
            )
        ).one()
        meta = (
            await s.execute(
                text("SELECT status, strategy_id, timeframe FROM trading_sessions WHERE id=:sid"),
                {"sid": SESSION},
            )
        ).one_or_none()
    await db.aclose()

    n = int(row.n)
    if n == 0:
        print("no closed trades yet")
        return 0

    wins = int(row.wins)
    net = Decimal(str(row.net))
    current = OPENING_EQUITY + net
    win_rate = Decimal(wins) / Decimal(n)
    gross_loss = Decimal(str(row.gross_loss))
    pf = Decimal(str(row.gross_win)) / gross_loss if gross_loss > ZERO else None
    maxdd = Decimal(str(eq.maxdd)) * 100
    hours = Decimal(str(row.avg_hours))
    ended_up = net > ZERO

    gross_win = Decimal(str(row.gross_win))
    fee_share = (Decimal(str(row.fees)) / gross_win * 100) if gross_win > ZERO else Decimal(0)
    avg_win = gross_win / max(wins, 1)
    avg_loss = gross_loss / max(n - wins, 1)
    win_note = "most trades still lose" if win_rate < Decimal("0.5") else "a majority of trades win"

    md = f"""# Paper-session log — {SESSION}

> **This is not a performance claim.** It is a log of one simulated session on one
> strategy, with no repetition and no out-of-sample holdout. Execution was simulated
> throughout and no real money was traded. Simulated and past results do not indicate,
> predict or guarantee future performance. The systematic research found that **no
> strategy tested beat simply holding the asset** — see
> `docs/research/strategy-research-2026-08.md`, which is the primary research document.

**Generated:** {datetime.now(UTC):%Y-%m-%d %H:%M} UTC · **Session:** `{SESSION}`
**Strategy:** {meta.strategy_id if meta else "?"} · **Timeframe:** {meta.timeframe if meta else "?"}
**Status:** {meta.status if meta else "?"}
**Data:** live Bybit V5 · **Execution:** simulated · **Live trading:** disabled

| Metric | Value |
|---|---:|
| Starting equity | {OPENING_EQUITY:,.2f} USDT |
| Current equity | **{current:,.2f} USDT** |
| Change over the session | {(net / OPENING_EQUITY) * 100:+.2f}% (simulated) |
| Closed trades | {n} |
| Win rate | **{win_rate:.1%}** ({wins} wins / {n - wins} losses) |
| Profit factor | **{pf:.2f}** |
| Max drawdown | {maxdd:.2f}% |
| Average trade duration | {hours:.1f} h ({hours / 24:.1f} days) |
| Largest winning trade | +{Decimal(str(row.best)):,.2f} |
| Largest losing trade | {Decimal(str(row.worst)):,.2f} |
| Total fees | {Decimal(str(row.fees)):,.2f} |
| Session ended up or down | {"up" if ended_up else "down"} |

## Reading this

The simulated account ended **{"up" if ended_up else "down"}** over {n} closed trades. That
is an observation about one sample, not a property of the strategy: a single unrepeated
run cannot distinguish an edge from variance, and this configuration was chosen after the
fact from a library of 44.

Profit factor {pf:.2f} means {pf:.2f} USDT earned for every 1.00 lost. Win rate is
{win_rate:.1%}, so {win_note} —
the account grows because the average win ({avg_win:,.2f}) exceeds the average loss
({avg_loss:,.2f}), not because losses are rare.

Fees consumed {fee_share:.1f}% of gross profit.

## Caveats

- Fills are simulated. Real execution adds queue position and partial fills that a
  simulator cannot reproduce, and both work against the trader.
- Most trades here come from historical daily bars replayed through the live engine. The
  realtime session adds to them at roughly one trade per fortnight per symbol.
- No strategy tested has beaten simply holding these assets. This configuration wins on
  drawdown ({maxdd:.1f}% against 85.9% for buy-and-hold), not on return.
"""
    # Written after the awaits complete: blocking file IO inside a coroutine stalls the
    # loop, and this is the last thing the coroutine does anyway.
    _write(md)
    print(f"wrote {_OUTPUT} at {n} closed trades (session ended {'up' if ended_up else 'down'})")
    return n


#: Under `reports/`, which is gitignored. A session log is operational output, not a
#: repository artefact, and it must not land somewhere that publishes it as a claim.
_OUTPUT = Path(__file__).resolve().parent.parent / "reports" / "paper-session.md"


def _write(markdown: str) -> None:
    """Write the session log to disk."""
    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT.write_text(markdown)


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
