"""Prompt construction.

Two things this file is careful about.

**The model is told what it is not being given.** Each symbol block ends with an explicit
list of unavailable data. A model shown no funding rate will reason about positioning
anyway; told the funding rate is unavailable, it can decline. Silence about a gap is how a
confident answer gets built on nothing.

**The model is told what it cannot do.** It does not size positions, does not set stops,
and cannot bypass the risk engine — those are decided downstream regardless of what it
returns. Saying so keeps it from producing advice shaped around powers it does not have,
and keeps its stated reasoning honest about the decision it is actually making.
"""

from __future__ import annotations

from quantflow.aitrader.context import DecisionContext

SYSTEM_PROMPT = """\
You are a disciplined crypto trading analyst. You decide whether to open, close or avoid \
a position, and nothing else.

Constraints on your role:
- You do NOT choose position size. Sizing is decided by a risk engine you cannot see.
- You do NOT set stop losses. Every entry receives one automatically.
- Your decision passes through a risk engine that may reject it. That is expected.
- You cannot trade any symbol other than those listed.

How to decide:
- Prefer HOLD. Doing nothing is free; a bad trade is not. Trading costs roughly 0.20% \
per round trip, so a move you expect to be smaller than that is not worth taking.
- Only return BUY or SELL when the evidence is specific and the reasoning would survive \
review after a loss.
- If required data is listed as unavailable, treat that as a reason for caution, not \
something to reason around.
- Confidence must reflect genuine conviction. Inflated confidence is worse than a low \
number, because a floor is applied to it downstream.

Respond with ONE JSON object and nothing else:

{"action": "BUY|SELL|HOLD", "symbol": "BTCUSDT", "confidence": 0.0, "reason": "..."}

- action: exactly one of BUY, SELL, HOLD
- symbol: one of the permitted symbols, no separator (e.g. BTCUSDT)
- confidence: a number from 0.0 to 1.0
- reason: one or two sentences citing the specific evidence you used

No prose outside the JSON. No markdown fences.\
"""


def build_user_prompt(  # noqa: PLR0912 - a flat sequence of prompt sections
    context: DecisionContext, *, candles_shown: int = 30
) -> str:
    """Render the decision context as the user message."""
    lines: list[str] = []

    lines.append(f"Time: {context.observed_at.isoformat()}")
    lines.append(f"Mode: {context.mode.upper()}")
    lines.append("")
    lines.append("ACCOUNT")
    lines.append(f"  equity: {context.equity:.2f} {context.base_currency}")
    lines.append(f"  cash:   {context.cash:.2f} {context.base_currency}")

    lines.append("")
    lines.append("OPEN POSITIONS")
    if not context.positions:
        lines.append("  none")
    else:
        for position in context.positions:
            rendered = ", ".join(f"{key}={value}" for key, value in position.items())
            lines.append(f"  {rendered}")

    permitted = ", ".join(f"{item.symbol.base}{item.symbol.quote}" for item in context.symbols)
    lines.append("")
    lines.append(f"PERMITTED SYMBOLS: {permitted}")

    for item in context.symbols:
        lines.append("")
        lines.append(f"=== {item.symbol.base}{item.symbol.quote} ===")
        lines.append(f"  last price: {item.last_price}")
        if item.regime:
            lines.append(f"  regime: {item.regime}")

        if item.indicators:
            lines.append("  indicators:")
            for key, value in item.indicators.items():
                lines.append(f"    {key}: {value}")

        if item.order_book:
            lines.append("  order book:")
            for key, value in item.order_book.items():
                lines.append(f"    {key}: {value}")

        if item.funding is not None:
            lines.append(
                f"  funding rate: {item.funding.rate} per 8h "
                f"({item.funding.annualised:.2%} annualised) "
                f"[from {item.funding.perpetual}, futures not spot]"
            )
        if item.open_interest is not None:
            lines.append(
                f"  open interest: {item.open_interest.contracts} {item.symbol.base} "
                f"[from {item.open_interest.perpetual}, futures not spot]"
            )

        recent = item.candles[-candles_shown:]
        if recent:
            lines.append(f"  recent OHLCV (last {len(recent)} bars, oldest first):")
            lines.append("    time,open,high,low,close,volume")
            for candle in recent:
                lines.append(
                    f"    {candle.open_time.isoformat()},{candle.open},{candle.high},"
                    f"{candle.low},{candle.close},{candle.volume}"
                )

        if item.unavailable:
            lines.append("  UNAVAILABLE (do not reason about these):")
            for missing in item.unavailable:
                lines.append(f"    - {missing}")

    lines.append("")
    lines.append("Return one JSON object with your decision.")
    return "\n".join(lines)


__all__ = ["SYSTEM_PROMPT", "build_user_prompt"]
