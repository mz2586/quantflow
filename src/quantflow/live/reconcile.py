"""Venue reconciliation.

Local state and the exchange drift: a missed fill, a manual close, a restart, a partial.
Whenever they disagree the venue is right — it holds the money. Anything the local book
believes is a claim, and a claim that contradicts the exchange is simply wrong.

Used at startup, before any signal is acted on. A position the system does not know it holds
cannot be sized against, cannot be stopped out by logic that never sees it, and will not
appear in any equity figure — so trading *around* it is trading blind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from quantflow.core.logging import get_logger
from quantflow.core.precision import ONE, ZERO
from quantflow.domain.instruments import Symbol

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VenuePosition:
    """A position as the exchange reports it."""

    symbol: Symbol
    side: str
    quantity: Decimal
    entry_price: Decimal
    stop_loss_price: Decimal | None
    #: Leverage the VENUE reports for this position. Never assumed: if Bybit holds the
    #: symbol at 10x it reserves a tenth of the margin the bot thinks it has, and every
    #: equity-derived limit would then be measured against a reservation that is not real.
    leverage: Decimal = ONE
    #: Initial margin the venue reports it has reserved, where available.
    venue_margin: Decimal | None = None

    @property
    def is_protected(self) -> bool:
        """Whether the venue is holding a stop for this position."""
        return self.stop_loss_price is not None and self.stop_loss_price > ZERO

    @property
    def margin_required(self) -> Decimal:
        """Margin reserved against this position.

        Prefers the venue's own figure. Falls back to notional / venue-reported leverage -
        still the venue's number, never the bot's assumption.
        """
        if self.venue_margin is not None and self.venue_margin > ZERO:
            return self.venue_margin
        leverage = self.leverage if self.leverage > ZERO else ONE
        return (self.quantity * self.entry_price) / leverage


@dataclass(slots=True)
class ReconciliationReport:
    """What the venue holds versus what was known locally."""

    venue_positions: list[VenuePosition] = field(default_factory=list)
    unknown_locally: list[VenuePosition] = field(default_factory=list)
    unprotected: list[VenuePosition] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Whether local state matches the venue and everything is protected."""
        return not self.unknown_locally and not self.unprotected

    @property
    def is_safe_to_trade(self) -> bool:
        """Whether new signals may be acted on.

        An unprotected live position is the disqualifying case: it is real, it is losing or
        winning right now, and nothing is guarding it.
        """
        return not self.unprotected

    def summary(self) -> str:
        """One line for logs and the harness table."""
        return (
            f"venue={len(self.venue_positions)} "
            f"unknown_locally={len(self.unknown_locally)} "
            f"unprotected={len(self.unprotected)}"
        )


def parse_venue_positions(
    raw_positions: list[dict[str, Any]], *, expected_leverage: Decimal = ONE
) -> list[VenuePosition]:
    """Extract live positions from a ccxt ``fetch_positions`` payload.

    Entries with zero size are skipped: Bybit reports flat symbols alongside real ones, and
    treating those as positions would manufacture drift that does not exist.
    """
    out: list[VenuePosition] = []
    for entry in raw_positions:
        info = entry.get("info", {}) if isinstance(entry, dict) else {}
        size_raw = info.get("size") if isinstance(info, dict) else None
        if size_raw in (None, "", "0", 0):
            continue
        try:
            quantity = Decimal(str(size_raw))
        except (ArithmeticError, ValueError):
            continue
        if quantity <= ZERO:
            continue

        raw_symbol = str(entry.get("symbol", "")).split(":")[0]
        try:
            symbol = Symbol.parse(raw_symbol)
        except Exception:  # pragma: no cover - a symbol we cannot parse is still a warning
            logger.warning("reconcile.unparseable_symbol", symbol=raw_symbol)
            continue

        stop_raw = info.get("stopLoss") if isinstance(info, dict) else None
        stop = None
        if stop_raw not in (None, "", "0", 0):
            try:
                stop = Decimal(str(stop_raw))
            except (ArithmeticError, ValueError):
                stop = None

        entry_raw = info.get("avgPrice") or entry.get("entryPrice") or 0
        try:
            entry_price = Decimal(str(entry_raw))
        except (ArithmeticError, ValueError):
            entry_price = ZERO

        leverage = _decimal_or_none(info.get("leverage") or entry.get("leverage")) or ONE
        venue_margin = _decimal_or_none(
            info.get("positionIM") or info.get("initialMargin") or entry.get("initialMargin")
        )
        if leverage != expected_leverage:
            # Reconcile to the venue, never to the assumption - but say so loudly, because
            # it means the bot's margin view was about to be wrong.
            logger.warning(
                "reconcile.unexpected_leverage",
                symbol=str(symbol),
                venue_leverage=str(leverage),
                expected=str(expected_leverage),
                detail="using the venue value; the bot's margin assumption does not hold",
            )

        out.append(
            VenuePosition(
                symbol=symbol,
                side=str(info.get("side", "")).lower(),
                quantity=quantity,
                entry_price=entry_price,
                stop_loss_price=stop,
                leverage=leverage,
                venue_margin=venue_margin,
            )
        )
    return out


def reconcile(
    venue_positions: list[VenuePosition], known_symbols: set[Symbol]
) -> ReconciliationReport:
    """Compare the venue against what the local book knows."""
    report = ReconciliationReport(venue_positions=list(venue_positions))
    for position in venue_positions:
        if position.symbol not in known_symbols:
            report.unknown_locally.append(position)
        if not position.is_protected:
            report.unprotected.append(position)

    if report.unknown_locally:
        logger.critical(
            "reconcile.unknown_venue_positions",
            symbols=[str(p.symbol) for p in report.unknown_locally],
            detail="the venue holds positions the local book does not know about",
        )
    if report.unprotected:
        logger.critical(
            "reconcile.unprotected_venue_positions",
            symbols=[str(p.symbol) for p in report.unprotected],
            detail="live positions with no server-side stop",
        )
    return report


@dataclass(frozen=True, slots=True)
class VenueAccount:
    """Equity and margin as the exchange reports them.

    On demo or live this is the authoritative account state. A simulated book is a model of
    the account; this *is* the account, so every equity-derived limit should read it rather
    than a reconstruction from fills that may have drifted.
    """

    equity: Decimal
    available: Decimal
    margin_posted: Decimal
    unrealized_pnl: Decimal
    positions: tuple[VenuePosition, ...] = ()

    def matches(self, local_equity: Decimal, *, tolerance: Decimal) -> bool:
        """Whether a locally computed equity agrees with the venue within ``tolerance``."""
        return abs(self.equity - local_equity) <= tolerance


def parse_venue_account(
    balances: dict[str, Any], positions: list[VenuePosition], *, quote: str = "USDT"
) -> VenueAccount:
    """Build the account view from a ccxt balance payload plus parsed positions.

    Prefers the venue's own unified-account totals where present, because those already
    include unrealised PnL and cross-margin effects that a per-asset sum would miss.
    """
    info = balances.get("info", {}) if isinstance(balances, dict) else {}
    equity = _first_decimal(info, ("totalEquity", "totalWalletBalance", "equity"), default=None)
    available = _first_decimal(info, ("totalAvailableBalance", "availableBalance"), default=ZERO)
    margin = _first_decimal(info, ("totalInitialMargin", "totalPositionIM"), default=ZERO)

    unrealised = sum((ZERO for _ in positions), ZERO)
    derived_margin = sum((position.margin_required for position in positions), ZERO)
    if equity is None:
        entry = balances.get(quote) if isinstance(balances, dict) else None
        if isinstance(entry, dict):
            equity = _decimal_or_none(entry.get("total")) or ZERO
        else:
            equity = getattr(entry, "total", ZERO) if entry is not None else ZERO

    return VenueAccount(
        equity=equity or ZERO,
        available=available or ZERO,
        # The venue's own total wins; the per-position sum (itself venue-derived) stands in
        # when the payload omits it.
        margin_posted=margin if margin and margin > ZERO else derived_margin,
        unrealized_pnl=unrealised,
        positions=tuple(positions),
    )


def _decimal_or_none(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    try:
        return Decimal(str(value))
    except (ArithmeticError, ValueError):
        return None


def _first_decimal(
    source: Any, keys: tuple[str, ...], *, default: Decimal | None
) -> Decimal | None:
    """The first parseable value among ``keys``, else ``default``."""
    if not isinstance(source, dict):
        return default
    for key in keys:
        parsed = _decimal_or_none(source.get(key))
        if parsed is not None:
            return parsed
    # Bybit nests the unified account under result.list[0].
    result = source.get("result")
    if isinstance(result, dict):
        entries = result.get("list")
        if isinstance(entries, list) and entries:
            return _first_decimal(entries[0], keys, default=default)
    return default


__all__ = [
    "ReconciliationReport",
    "VenueAccount",
    "VenuePosition",
    "parse_venue_account",
    "parse_venue_positions",
    "reconcile",
]
