"""FX position sizing.

The one rule this module exists to enforce: **size is derived from the money at risk and
the value of the stop distance, never from notional / price.** The crypto formula
(``quantity = risk / (price * stop_pct)``) returns a number of base units; feeding that to
an FX venue that measures volume in 100,000-unit lots opens a position five orders of
magnitude too large. Here the chain is always::

    lots = account_risk / (stop_distance_points * value_per_point_per_lot)

and ``value_per_point_per_lot`` comes from the venue's tick value, so JPY quotes (3 digits,
0.001 point) and 5-digit quotes fall out of the same expression with no special-casing.

Every rejection is explicit. A size that genuinely cannot be expressed above ``min_lot``
returns a rejected :class:`LotSizingResult` carrying a :class:`SizingRejection` — it never
silently becomes zero, and a size that *is* fundable never rounds down to nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ZERO
from quantflow.domain.enums import OrderSide
from quantflow.forex.instruments import ForexInstrument


class SizingRejection(StrEnum):
    """Why a sizing request produced no tradable size."""

    NON_POSITIVE_RISK = "non_positive_risk"
    NON_POSITIVE_STOP = "non_positive_stop_distance"
    ZERO_POINT_VALUE = "zero_value_per_point"
    INSTRUMENT_NOT_TRADABLE = "instrument_not_tradable"
    SIDE_NOT_ALLOWED = "side_not_allowed"
    STOP_WRONG_SIDE = "stop_on_wrong_side"
    BELOW_MIN_LOT = "below_min_lot"


@dataclass(frozen=True, slots=True)
class LotSizingResult:
    """The outcome of a sizing request — accepted with a size, or rejected with a reason."""

    lots: Decimal
    accepted: bool
    raw_lots: Decimal
    value_per_point: Decimal
    risk_per_lot: Decimal
    projected_risk: Decimal
    clamped_to_max: bool = False
    reason: SizingRejection | None = None
    message: str | None = None

    def __bool__(self) -> bool:
        """Truthy only when a tradable size was produced."""
        return self.accepted


def _rejected(
    reason: SizingRejection,
    message: str,
    *,
    raw_lots: Decimal = ZERO,
    value_per_point: Decimal = ZERO,
    risk_per_lot: Decimal = ZERO,
) -> LotSizingResult:
    """Build a rejected result with a zero size and an explicit reason."""
    return LotSizingResult(
        lots=ZERO,
        accepted=False,
        raw_lots=raw_lots,
        value_per_point=value_per_point,
        risk_per_lot=risk_per_lot,
        projected_risk=ZERO,
        reason=reason,
        message=message,
    )


def value_per_point(
    instrument: ForexInstrument,
    tick_value: Decimal | None = None,
    contract_size: Decimal | None = None,
) -> Decimal:
    """Account-currency value of a one-point move on one lot.

    Explicit ``tick_value``/``contract_size`` override the instrument's own metadata, which
    is how a caller injects a freshly-fetched tick value without rebuilding the instrument.
    A tick value of zero falls back to ``contract_size * point``.
    """
    resolved_tick_value = instrument.tick_value if tick_value is None else tick_value
    resolved_contract_size = instrument.contract_size if contract_size is None else contract_size
    if resolved_tick_value > ZERO:
        return resolved_tick_value * instrument.point / instrument.tick_size
    return resolved_contract_size * instrument.point


def stop_distance_points(
    entry_price: Decimal, stop_price: Decimal, instrument: ForexInstrument
) -> Decimal:
    """Absolute distance from entry to stop, in points.

    Absolute by construction, so a long and a short with mirror-image stops produce the
    same distance and therefore the same size.
    """
    return instrument.price_to_points(entry_price - stop_price)


def lots_for_risk(
    account_risk: Decimal,
    stop_distance_points: Decimal,
    tick_value: Decimal,
    contract_size: Decimal,
    instrument: ForexInstrument,
) -> LotSizingResult:
    """Convert a money-at-risk budget into a venue-legal lot size.

    Args:
        account_risk: Money the trade may lose if the stop fills, in account currency.
        stop_distance_points: Distance from entry to stop, in points. Always positive.
        tick_value: Value of one tick on one lot; ``0`` falls back to ``contract_size``.
        contract_size: Units of base currency per lot, used only for that fallback.
        instrument: The symbol being sized.

    Returns:
        An accepted result whose ``lots`` sits on the venue's volume grid at or above
        ``min_lot``, or a rejected result whose ``reason`` says why no size was possible.

    """
    if not instrument.tradable:
        return _rejected(
            SizingRejection.INSTRUMENT_NOT_TRADABLE,
            f"{instrument.symbol} is not tradable at the venue right now",
        )
    if account_risk <= ZERO:
        return _rejected(
            SizingRejection.NON_POSITIVE_RISK,
            f"account_risk must be positive, got {account_risk}",
        )
    if stop_distance_points <= ZERO:
        return _rejected(
            SizingRejection.NON_POSITIVE_STOP,
            f"stop_distance_points must be positive, got {stop_distance_points}",
        )

    point_value = value_per_point(instrument, tick_value, contract_size)
    if point_value <= ZERO:
        return _rejected(
            SizingRejection.ZERO_POINT_VALUE,
            f"{instrument.symbol} has no usable tick value or contract size; "
            "the venue did not report enough metadata to size this trade",
        )

    risk_per_lot = stop_distance_points * point_value
    raw_lots = account_risk / risk_per_lot

    if raw_lots < instrument.min_lot:
        return _rejected(
            SizingRejection.BELOW_MIN_LOT,
            f"risk budget {account_risk} over {stop_distance_points} points buys "
            f"{raw_lots} lots, below the {instrument.min_lot} minimum for "
            f"{instrument.symbol}; widen the budget or tighten the stop",
            raw_lots=raw_lots,
            value_per_point=point_value,
            risk_per_lot=risk_per_lot,
        )

    lots = instrument.quantise_lots(raw_lots)
    # quantise_lots anchors the grid on min_lot, so an at-or-above-minimum raw size can
    # never round to zero. Assert it rather than trusting it.
    if lots < instrument.min_lot:  # pragma: no cover — guarded by quantise_lots
        raise ValidationError(
            "lot quantisation dropped a fundable size below the minimum",
            symbol=instrument.symbol,
            raw_lots=str(raw_lots),
        )

    return LotSizingResult(
        lots=lots,
        accepted=True,
        raw_lots=raw_lots,
        value_per_point=point_value,
        risk_per_lot=risk_per_lot,
        projected_risk=lots * risk_per_lot,
        clamped_to_max=raw_lots > instrument.max_lot,
    )


def lots_for_risk_from_prices(
    account_risk: Decimal,
    entry_price: Decimal,
    stop_price: Decimal,
    instrument: ForexInstrument,
    side: OrderSide,
) -> LotSizingResult:
    """Size from an entry/stop price pair, validating that the stop is on the right side.

    A long's stop must sit below entry and a short's above it. Getting this backwards would
    otherwise size a "stop" that is really a target, so it is rejected rather than sized.
    """
    if not instrument.can_trade(side):
        return _rejected(
            SizingRejection.SIDE_NOT_ALLOWED,
            f"{instrument.symbol} does not permit opening {side.value} "
            f"(trade mode {instrument.trade_mode.value}, tradable={instrument.tradable})",
        )
    stop_is_valid = stop_price < entry_price if side is OrderSide.BUY else stop_price > entry_price
    if not stop_is_valid:
        expected = "below" if side is OrderSide.BUY else "above"
        return _rejected(
            SizingRejection.STOP_WRONG_SIDE,
            f"a {side.value} stop must be {expected} entry: entry={entry_price} stop={stop_price}",
        )
    return lots_for_risk(
        account_risk,
        stop_distance_points(entry_price, stop_price, instrument),
        instrument.tick_value,
        instrument.contract_size,
        instrument,
    )


def risk_for_lots(
    lots: Decimal, stop_distance_points: Decimal, instrument: ForexInstrument
) -> Decimal:
    """Money at risk if ``lots`` is stopped out at ``stop_distance_points``."""
    if lots < ZERO:
        raise ValidationError(f"lots must not be negative, got {lots}", symbol=instrument.symbol)
    if stop_distance_points < ZERO:
        raise ValidationError("stop_distance_points must not be negative")
    return lots * stop_distance_points * instrument.value_per_point_per_lot


def pip_value(instrument: ForexInstrument, lots: Decimal) -> Decimal:
    """Account-currency value of a one-pip move on ``lots``."""
    if lots < ZERO:
        raise ValidationError(f"lots must not be negative, got {lots}", symbol=instrument.symbol)
    return instrument.pip_value_per_lot * lots


def margin_required(instrument: ForexInstrument, lots: Decimal, price: Decimal) -> Decimal | None:
    """Margin needed to hold ``lots`` at ``price``, or ``None`` if the venue never said.

    Returning ``None`` rather than a default keeps an unknown margin rate visible instead
    of letting an invented leverage flow into a risk check.
    """
    if instrument.margin_rate <= ZERO:
        return None
    return instrument.notional(lots, price) * instrument.margin_rate
