"""Decimal arithmetic helpers.

Trading maths is done with :class:`decimal.Decimal`. Binary floats silently accumulate
error and turn "1.10" into "1.1000000000000001", which corrupts fee and PnL accounting and
gets orders rejected for precision violations.

All rounding is explicit: quantities round **down** (never buy more than intended, never
sell more than held), prices round toward the side that is conservative for the trader.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_UP, Decimal, InvalidOperation
from typing import Final

from quantflow.core.errors import ValidationError

ZERO: Final = Decimal("0")
ONE: Final = Decimal("1")
HUNDRED: Final = Decimal("100")

#: Working precision for intermediate results before final quantisation.
INTERNAL_QUANTUM: Final = Decimal("0.00000001")  # 8dp — matches Binance's finest step


def to_decimal(value: Decimal | int | str | float) -> Decimal:
    """Coerce a scalar to :class:`Decimal` without float round-trip surprises.

    Floats are routed through ``repr`` so ``0.1`` becomes ``Decimal("0.1")`` rather than
    the full binary expansion.

    Raises:
        ValidationError: if the value is not a finite number.

    """
    try:
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, float):
            result = Decimal(repr(value))
        else:
            result = Decimal(value)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValidationError(f"cannot convert to Decimal: {value!r}") from exc
    if not result.is_finite():
        raise ValidationError(f"non-finite Decimal value: {value!r}")
    return result


def quantize_down(value: Decimal, step: Decimal) -> Decimal:
    """Round ``value`` **down** to the nearest multiple of ``step``.

    Used for order quantities: an order must never exceed the intended size.
    """
    _require_positive_step(step)
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def quantize_up(value: Decimal, step: Decimal) -> Decimal:
    """Round ``value`` **up** to the nearest multiple of ``step``."""
    _require_positive_step(step)
    return (value / step).to_integral_value(rounding=ROUND_UP) * step


def quantize_nearest(value: Decimal, step: Decimal) -> Decimal:
    """Round ``value`` to the nearest multiple of ``step``, ties to even."""
    _require_positive_step(step)
    return (value / step).to_integral_value(rounding=ROUND_HALF_EVEN) * step


def round_price(value: Decimal, tick_size: Decimal, *, side_is_buy: bool) -> Decimal:
    """Snap a limit price to the exchange tick grid, conservatively.

    A buy limit rounds **down** (we will not pay more than intended); a sell limit rounds
    **up** (we will not accept less than intended).
    """
    return quantize_down(value, tick_size) if side_is_buy else quantize_up(value, tick_size)


def round_stop_price(value: Decimal, tick_size: Decimal, *, position_is_long: bool) -> Decimal:
    """Snap a protective stop to the tick grid, always *away* from the position.

    Not the same rule as :func:`round_price`, and the difference matters. A stop is placed
    by the side opposite the position — a long is closed by a sell — so applying the limit
    convention to the closing side rounds a long's stop **up**, toward the entry it is
    protecting. That silently tightens the risk the engine sized for, and on a wide tick it
    can put the stop on the wrong side of its own trigger price.

    So the direction is taken from the *position*, not from the order that closes it: a
    long's stop rounds down, a short's rounds up. Both move it further from entry, which
    is the safe direction to be wrong in by one tick.
    """
    return quantize_down(value, tick_size) if position_is_long else quantize_up(value, tick_size)


def round_quantity(value: Decimal, step_size: Decimal) -> Decimal:
    """Snap an order quantity to the exchange lot grid, always downward."""
    return quantize_down(value, step_size)


def decimal_places(step: Decimal) -> int:
    """Number of decimal places implied by a tick/step size.

    ``Decimal("0.001") -> 3``, ``Decimal("1") -> 0``, ``Decimal("10") -> 0``.
    """
    _require_positive_step(step)
    exponent = step.normalize().as_tuple().exponent
    if not isinstance(exponent, int):  # pragma: no cover — non-finite guarded above
        raise ValidationError(f"invalid step: {step!r}")
    return max(0, -exponent)


def step_from_precision(precision: int) -> Decimal:
    """Convert an integer decimal precision into a step size."""
    if precision < 0:
        raise ValidationError(f"precision must be non-negative, got {precision}")
    return Decimal(1).scaleb(-precision)


def safe_divide(numerator: Decimal, denominator: Decimal, *, default: Decimal = ZERO) -> Decimal:
    """Divide, returning ``default`` when the denominator is zero."""
    if denominator == ZERO:
        return default
    return numerator / denominator


def pct_change(start: Decimal, end: Decimal) -> Decimal:
    """Fractional change from ``start`` to ``end``. Returns 0 when ``start`` is zero."""
    return safe_divide(end - start, abs(start))


def apply_pct(value: Decimal, pct: Decimal) -> Decimal:
    """Return ``value`` scaled by ``1 + pct`` where ``pct`` is a fraction (0.02 = +2%)."""
    return value * (ONE + pct)


def clamp(value: Decimal, minimum: Decimal, maximum: Decimal) -> Decimal:
    """Constrain ``value`` to ``[minimum, maximum]``."""
    if minimum > maximum:
        raise ValidationError(f"empty clamp range: [{minimum}, {maximum}]")
    return max(minimum, min(maximum, value))


def normalize(value: Decimal) -> Decimal:
    """Strip trailing zeros without switching to scientific notation.

    ``Decimal("1.500") -> Decimal("1.5")``, ``Decimal("1E+2") -> Decimal("100")``.
    """
    if value == value.to_integral_value():
        return value.quantize(ONE)
    return value.normalize()


def _require_positive_step(step: Decimal) -> None:
    if step <= ZERO:
        raise ValidationError(f"step must be positive, got {step}")
