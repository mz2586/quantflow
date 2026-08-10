"""Assemble what the model is allowed to see, and say what is missing.

Every field is optional and every absence is stated in the prompt. A model handed a
missing order book with no comment will happily reason about liquidity it was never
given; told plainly that the book is unavailable, it can decline instead. Silence about
missing data is how a confident answer gets built on nothing.

Numbers are rendered as strings at fixed precision. Floats in a prompt invite a model to
echo back `0.30000000000000004`, and a decision derived from a corrupted price is worse
than no decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from quantflow.core.precision import ZERO
from quantflow.domain.instruments import Symbol
from quantflow.domain.market import Candle, OrderBook
from quantflow.domain.portfolio import PortfolioSnapshot
from quantflow.intelligence.derivatives import FundingSnapshot, OpenInterestSnapshot
from quantflow.intelligence.measures import (
    measure_liquidity,
    measure_trend,
    measure_volatility,
    measure_volume,
)
from quantflow.intelligence.regime import classify
from quantflow.strategy.indicators import atr, bollinger_bands, ema, macd, rsi


@dataclass(frozen=True, slots=True)
class SymbolContext:
    """Everything known about one symbol at decision time."""

    symbol: Symbol
    candles: tuple[Candle, ...]
    indicators: dict[str, str] = field(default_factory=dict)
    regime: str | None = None
    order_book: dict[str, str] | None = None
    funding: FundingSnapshot | None = None
    open_interest: OpenInterestSnapshot | None = None
    unavailable: tuple[str, ...] = ()

    @property
    def last_price(self) -> Decimal:
        """Most recent close."""
        return self.candles[-1].close if self.candles else ZERO


@dataclass(frozen=True, slots=True)
class DecisionContext:
    """The full picture handed to the model on one cycle."""

    observed_at: datetime
    base_currency: str
    equity: Decimal
    cash: Decimal
    positions: tuple[dict[str, str], ...]
    symbols: tuple[SymbolContext, ...]
    mode: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the journal."""
        return {
            "observed_at": self.observed_at.isoformat(),
            "mode": self.mode,
            "equity": str(self.equity),
            "cash": str(self.cash),
            "positions": [dict(item) for item in self.positions],
            "symbols": [
                {
                    "symbol": str(item.symbol),
                    "last_price": str(item.last_price),
                    "regime": item.regime,
                    "indicators": item.indicators,
                    "order_book": item.order_book,
                    "funding": item.funding.to_dict() if item.funding else None,
                    "open_interest": (item.open_interest.to_dict() if item.open_interest else None),
                    "unavailable": list(item.unavailable),
                }
                for item in self.symbols
            ],
        }


def build_indicators(candles: Sequence[Candle]) -> tuple[dict[str, str], list[str]]:
    """Compute the indicator set, reporting anything that could not be produced."""
    if not candles:
        return {}, ["indicators (no candles)"]

    closes = [candle.close for candle in candles]
    index = len(closes) - 1
    out: dict[str, str] = {}
    missing: list[str] = []

    def record(name: str, value: Decimal | None, spec: str = "{:.6f}") -> None:
        if value is None:
            missing.append(name)
        else:
            out[name] = spec.format(value)

    record("ema_20", ema(closes, 20)[index], "{:.2f}")
    record("ema_50", ema(closes, 50)[index], "{:.2f}")
    record("ema_200", ema(closes, 200)[index], "{:.2f}")
    record("rsi_14", rsi(closes, 14)[index], "{:.2f}")
    record("atr_14", atr(candles, 14)[index], "{:.2f}")

    upper, middle, lower = bollinger_bands(closes, 20)
    record("bb_upper", upper[index], "{:.2f}")
    record("bb_middle", middle[index], "{:.2f}")
    record("bb_lower", lower[index], "{:.2f}")

    line, signal_line, histogram = macd(closes)
    record("macd", line[index], "{:.4f}")
    record("macd_signal", signal_line[index], "{:.4f}")
    record("macd_histogram", histogram[index], "{:.4f}")

    trend = measure_trend(candles)
    if trend is not None:
        out["trend_strength"] = f"{trend.strength:.3f}"
        out["trend_direction"] = f"{trend.direction:+.5f}"
    else:
        missing.append("trend measure")

    volatility = measure_volatility(candles)
    if volatility is not None:
        out["volatility_vs_baseline"] = f"{volatility.relative_level:.2f}x"
        out["normalized_atr"] = f"{volatility.normalized_atr:.4%}"
    else:
        missing.append("volatility measure")

    volume = measure_volume(candles)
    if volume is not None:
        out["volume_expansion"] = f"{volume.expansion:.2f}x"
    else:
        missing.append("volume measure")

    liquidity = measure_liquidity(candles)
    if liquidity is not None:
        out["typical_bar_quote_volume"] = f"{liquidity.typical_quote_volume:.0f}"

    return out, missing


def summarise_order_book(book: OrderBook | None, *, depth: int = 10) -> dict[str, str] | None:
    """Reduce an order book to the few numbers worth putting in a prompt.

    Raw levels are useless to a model and expensive in tokens. Spread and imbalance are
    the parts that carry information about whether an order will be filled where expected.
    """
    if book is None or not book.bids or not book.asks:
        return None

    best_bid = book.bids[0].price
    best_ask = book.asks[0].price
    mid = (best_bid + best_ask) / Decimal("2")
    spread = best_ask - best_bid

    bid_volume = sum((level.quantity for level in book.bids[:depth]), ZERO)
    ask_volume = sum((level.quantity for level in book.asks[:depth]), ZERO)
    total = bid_volume + ask_volume

    return {
        "best_bid": f"{best_bid:.2f}",
        "best_ask": f"{best_ask:.2f}",
        "spread": f"{spread:.2f}",
        "spread_pct": f"{(spread / mid):.4%}" if mid > ZERO else "n/a",
        "bid_depth": f"{bid_volume:.4f}",
        "ask_depth": f"{ask_volume:.4f}",
        # Above 0.5 means more resting size on the bid than the ask.
        "bid_share": f"{(bid_volume / total):.3f}" if total > ZERO else "n/a",
    }


def describe_positions(snapshot: PortfolioSnapshot) -> tuple[dict[str, str], ...]:
    """Open positions, rendered for the prompt."""
    out: list[dict[str, str]] = []
    for position in snapshot.open_positions:
        mark = snapshot.mark_prices.get(position.symbol)
        entry: dict[str, str] = {
            "symbol": str(position.symbol),
            "side": position.side.value,
            "quantity": f"{position.quantity}",
            "entry_price": f"{position.average_entry_price:.2f}",
            "realized_pnl": f"{position.realized_pnl:.2f}",
        }
        if mark is not None:
            entry["mark_price"] = f"{mark:.2f}"
            entry["unrealized_pnl"] = f"{position.unrealized_pnl(mark):.2f}"
            entry["unrealized_pnl_pct"] = f"{position.unrealized_pnl_pct(mark):.2%}"
        if position.stop_loss_price is not None:
            entry["stop_loss"] = f"{position.stop_loss_price:.2f}"
        out.append(entry)
    return tuple(out)


def build_symbol_context(
    symbol: Symbol,
    candles: Sequence[Candle],
    *,
    order_book: OrderBook | None = None,
    funding: FundingSnapshot | None = None,
    open_interest: OpenInterestSnapshot | None = None,
) -> SymbolContext:
    """Assemble one symbol's context, naming whatever was unavailable."""
    indicators, missing = build_indicators(candles)

    profile = classify(candles) if candles else None
    if profile is None:
        missing.append("market regime (insufficient history)")

    book = summarise_order_book(order_book)
    if book is None:
        missing.append("order book (not fetched or empty)")
    if funding is None:
        missing.append("funding rate (perpetual futures only; unavailable)")
    if open_interest is None:
        missing.append("open interest (perpetual futures only; unavailable)")

    return SymbolContext(
        symbol=symbol,
        candles=tuple(candles),
        indicators=indicators,
        regime=profile.label if profile else None,
        order_book=book,
        funding=funding,
        open_interest=open_interest,
        unavailable=tuple(missing),
    )


__all__ = [
    "DecisionContext",
    "SymbolContext",
    "build_indicators",
    "build_symbol_context",
    "describe_positions",
    "summarise_order_book",
]
