"""Technical indicators.

Pure functions over ``Sequence[Decimal]``, returning a tuple **the same length as the
input** with ``None`` during the warm-up period. Aligned output is the point: an indicator
that silently returns a shorter list forces every caller to do offset arithmetic, and one
off-by-one there is a look-ahead bug that no test will catch by accident.

Arithmetic stays in ``Decimal``. These values feed stop-loss and position-size
calculations, so a float rounding artefact becomes a real order at a real price.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from itertools import pairwise

from quantflow.core.errors import ValidationError
from quantflow.core.precision import ONE, ZERO, safe_divide
from quantflow.domain.market import Candle

Series = tuple[Decimal | None, ...]

#: Directional movement compares each bar with the one before, so it needs a pair.
MIN_BARS_FOR_DIRECTIONAL_MOVEMENT = 2


def _require_period(period: int, minimum: int = 1) -> None:
    if period < minimum:
        raise ValidationError(f"period must be at least {minimum}, got {period}")


def sma(values: Sequence[Decimal], period: int) -> Series:
    """Simple moving average."""
    _require_period(period)
    if not values:
        return ()
    out: list[Decimal | None] = [None] * len(values)
    running = ZERO
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        if index >= period - 1:
            out[index] = running / period
    return tuple(out)


def ema(values: Sequence[Decimal], period: int) -> Series:
    """Exponential moving average.

    Seeded with the SMA of the first ``period`` values, which is the convention every
    charting package uses. Seeding with the first value instead produces a curve that
    disagrees with what a trader sees on screen for hundreds of bars.
    """
    _require_period(period)
    if not values:
        return ()
    out: list[Decimal | None] = [None] * len(values)
    if len(values) < period:
        return tuple(out)

    multiplier = Decimal(2) / Decimal(period + 1)
    seed = sum(values[:period], ZERO) / period
    out[period - 1] = seed
    previous = seed
    for index in range(period, len(values)):
        previous = (values[index] - previous) * multiplier + previous
        out[index] = previous
    return tuple(out)


def wilder_smoothing(values: Sequence[Decimal], period: int) -> Series:
    """Wilder's smoothing (used by RSI, ATR and ADX).

    Distinct from an EMA: Wilder's uses ``1/period`` where an EMA uses ``2/(period+1)``.
    Substituting one for the other silently changes every RSI and ATR reading.
    """
    _require_period(period)
    if len(values) < period:
        return tuple([None] * len(values))

    out: list[Decimal | None] = [None] * len(values)
    seed = sum(values[:period], ZERO) / period
    out[period - 1] = seed
    previous = seed
    for index in range(period, len(values)):
        previous = (previous * (period - 1) + values[index]) / period
        out[index] = previous
    return tuple(out)


def rsi(values: Sequence[Decimal], period: int = 14) -> Series:
    """Relative Strength Index, Wilder's formulation. Range ``[0, 100]``."""
    _require_period(period, minimum=2)
    if len(values) <= period:
        return tuple([None] * len(values))

    gains: list[Decimal] = [ZERO]
    losses: list[Decimal] = [ZERO]
    for previous, current in pairwise(values):
        change = current - previous
        gains.append(max(change, ZERO))
        losses.append(max(-change, ZERO))

    out: list[Decimal | None] = [None] * len(values)
    average_gain = sum(gains[1 : period + 1], ZERO) / period
    average_loss = sum(losses[1 : period + 1], ZERO) / period
    out[period] = _rsi_from_averages(average_gain, average_loss)

    for index in range(period + 1, len(values)):
        average_gain = (average_gain * (period - 1) + gains[index]) / period
        average_loss = (average_loss * (period - 1) + losses[index]) / period
        out[index] = _rsi_from_averages(average_gain, average_loss)
    return tuple(out)


def _rsi_from_averages(average_gain: Decimal, average_loss: Decimal) -> Decimal:
    if average_loss == ZERO:
        # An unbroken run of gains: RSI is 100 by definition, not a division by zero.
        return Decimal(100) if average_gain > ZERO else Decimal(50)
    relative_strength = average_gain / average_loss
    return Decimal(100) - Decimal(100) / (ONE + relative_strength)


def true_range(candles: Sequence[Candle]) -> Series:
    """True range: the greatest of the bar's range and its gap from the prior close."""
    if not candles:
        return ()
    out: list[Decimal | None] = [candles[0].high - candles[0].low]
    for previous, current in pairwise(candles):
        out.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return tuple(out)


def atr(candles: Sequence[Candle], period: int = 14) -> Series:
    """Average True Range.

    The natural unit for volatility-scaled stops and position sizes: a fixed percentage
    stop is far too tight in a calm market and far too wide in a violent one.
    """
    ranges = true_range(candles)
    concrete = [value for value in ranges if value is not None]
    if len(concrete) < period:
        return tuple([None] * len(candles))
    return wilder_smoothing(concrete, period)


def rolling_max(values: Sequence[Decimal], period: int) -> Series:
    """Highest value over a trailing window."""
    _require_period(period)
    out: list[Decimal | None] = [None] * len(values)
    for index in range(period - 1, len(values)):
        out[index] = max(values[index - period + 1 : index + 1])
    return tuple(out)


def rolling_min(values: Sequence[Decimal], period: int) -> Series:
    """Lowest value over a trailing window."""
    _require_period(period)
    out: list[Decimal | None] = [None] * len(values)
    for index in range(period - 1, len(values)):
        out[index] = min(values[index - period + 1 : index + 1])
    return tuple(out)


def stdev(values: Sequence[Decimal], period: int) -> Series:
    """Rolling population standard deviation."""
    _require_period(period, minimum=2)
    out: list[Decimal | None] = [None] * len(values)
    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        mean = sum(window, ZERO) / period
        variance = sum(((value - mean) ** 2 for value in window), ZERO) / period
        out[index] = variance.sqrt()
    return tuple(out)


def bollinger_bands(
    values: Sequence[Decimal], period: int = 20, deviations: Decimal = Decimal("2")
) -> tuple[Series, Series, Series]:
    """Bollinger Bands as ``(upper, middle, lower)``."""
    middle = sma(values, period)
    spread = stdev(values, period)
    upper: list[Decimal | None] = []
    lower: list[Decimal | None] = []
    for centre, deviation in zip(middle, spread, strict=True):
        if centre is None or deviation is None:
            upper.append(None)
            lower.append(None)
        else:
            upper.append(centre + deviation * deviations)
            lower.append(centre - deviation * deviations)
    return tuple(upper), middle, tuple(lower)


def macd(
    values: Sequence[Decimal], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[Series, Series, Series]:
    """MACD as ``(macd_line, signal_line, histogram)``."""
    if fast >= slow:
        raise ValidationError(f"fast period {fast} must be below slow period {slow}")
    fast_ema = ema(values, fast)
    slow_ema = ema(values, slow)

    macd_line: list[Decimal | None] = [
        None if quick is None or slow_value is None else quick - slow_value
        for quick, slow_value in zip(fast_ema, slow_ema, strict=True)
    ]
    concrete = [value for value in macd_line if value is not None]
    signal_concrete = ema(concrete, signal)

    signal_line: list[Decimal | None] = [None] * len(values)
    offset = len(values) - len(concrete)
    for index, value in enumerate(signal_concrete):
        signal_line[offset + index] = value

    histogram: list[Decimal | None] = [
        None if line is None or trigger is None else line - trigger
        for line, trigger in zip(macd_line, signal_line, strict=True)
    ]
    return tuple(macd_line), tuple(signal_line), tuple(histogram)


def crossed_above(fast: Series, slow: Series, index: int) -> bool:
    """Whether ``fast`` crossed above ``slow`` on this bar.

    A *crossing*, not a comparison: it requires the relationship to have been the other way
    on the previous bar. Testing ``fast > slow`` alone re-fires an entry on every bar of a
    trend, which turns one position into hundreds of round trips and a fee bill that
    destroys the strategy.
    """
    if index < 1 or index >= len(fast):
        return False
    previous_fast, previous_slow = fast[index - 1], slow[index - 1]
    current_fast, current_slow = fast[index], slow[index]
    if None in (previous_fast, previous_slow, current_fast, current_slow):
        return False
    assert previous_fast is not None
    assert previous_slow is not None
    assert current_fast is not None
    assert current_slow is not None
    return previous_fast <= previous_slow and current_fast > current_slow


def crossed_below(fast: Series, slow: Series, index: int) -> bool:
    """Whether ``fast`` crossed below ``slow`` on this bar."""
    return crossed_above(slow, fast, index)


def slope(values: Series, index: int, lookback: int = 1) -> Decimal | None:
    """Change in a series over ``lookback`` bars, or ``None`` if unavailable."""
    if index < lookback or index >= len(values):
        return None
    current, previous = values[index], values[index - lookback]
    if current is None or previous is None:
        return None
    return current - previous


def normalized_atr(candles: Sequence[Candle], period: int = 14) -> Series:
    """ATR as a fraction of price, so volatility is comparable across symbols."""
    ranges = atr(candles, period)
    return tuple(
        None if value is None else safe_divide(value, candle.close)
        for value, candle in zip(ranges, candles, strict=True)
    )


def last_value(series: Series) -> Decimal | None:
    """The most recent non-``None`` value in a series."""
    for value in reversed(series):
        if value is not None:
            return value
    return None


def rolling_vwap(candles: Sequence[Candle], period: int) -> Series:
    """Volume-weighted average price over a trailing window.

    A rolling window rather than a session-anchored VWAP: an anchored one needs a session
    boundary, and crypto trades continuously. Windows whose volume is zero yield ``None``
    rather than a divide-by-zero or a silent fallback to the unweighted mean — no volume
    means no volume-weighted price, and pretending otherwise invents a level.
    """
    _require_period(period)
    out: list[Decimal | None] = [None] * len(candles)
    for index in range(period - 1, len(candles)):
        window = candles[index - period + 1 : index + 1]
        volume = sum((candle.volume for candle in window), ZERO)
        if volume <= ZERO:
            continue
        typical = sum(
            (((candle.high + candle.low + candle.close) / 3) * candle.volume for candle in window),
            ZERO,
        )
        out[index] = typical / volume
    return tuple(out)


def stochastic(
    candles: Sequence[Candle], period: int = 14, smooth: int = 3
) -> tuple[Series, Series]:
    """Stochastic oscillator as ``(%K, %D)``, both in 0–100.

    %K is where the close sits inside the period's high–low range; %D is its moving
    average. A flat range (high == low) gives ``None`` rather than 50: a range of zero
    carries no information about position within it.
    """
    _require_period(period)
    _require_period(smooth)
    raw: list[Decimal | None] = [None] * len(candles)
    for index in range(period - 1, len(candles)):
        window = candles[index - period + 1 : index + 1]
        highest = max(candle.high for candle in window)
        lowest = min(candle.low for candle in window)
        span = highest - lowest
        if span <= ZERO:
            continue
        raw[index] = ((candles[index].close - lowest) / span) * Decimal("100")

    # %D smooths %K, so it can only be computed where a full run of %K values exists.
    smoothed: list[Decimal | None] = [None] * len(candles)
    for index in range(smooth - 1, len(candles)):
        recent = raw[index - smooth + 1 : index + 1]
        values = [value for value in recent if value is not None]
        if len(values) == smooth:
            smoothed[index] = sum(values, ZERO) / smooth
    return tuple(raw), tuple(smoothed)


def directional_movement(
    candles: Sequence[Candle], period: int = 14
) -> tuple[Series, Series, Series]:
    """Wilder's ``(+DI, -DI, ADX)``, all on a 0–100 scale.

    ADX measures how *strongly* a market is trending without saying which way, which is
    what makes it a regime input rather than a direction signal. Built on Wilder smoothing
    to match the original definition — an EMA-based approximation drifts from published
    values and would make any threshold chosen here mean something slightly different.
    """
    _require_period(period)
    size = len(candles)
    if size < MIN_BARS_FOR_DIRECTIONAL_MOVEMENT:
        return (None,) * size, (None,) * size, (None,) * size

    plus_moves: list[Decimal] = [ZERO]
    minus_moves: list[Decimal] = [ZERO]
    for index in range(1, size):
        up = candles[index].high - candles[index - 1].high
        down = candles[index - 1].low - candles[index].low
        plus_moves.append(up if up > down and up > ZERO else ZERO)
        minus_moves.append(down if down > up and down > ZERO else ZERO)

    ranges = [value if value is not None else ZERO for value in true_range(candles)]
    smoothed_range = wilder_smoothing(ranges, period)
    smoothed_plus = wilder_smoothing(plus_moves, period)
    smoothed_minus = wilder_smoothing(minus_moves, period)

    plus_di: list[Decimal | None] = [None] * size
    minus_di: list[Decimal | None] = [None] * size
    dx: list[Decimal | None] = [None] * size
    for index in range(size):
        span = smoothed_range[index]
        up_value = smoothed_plus[index]
        down_value = smoothed_minus[index]
        if span is None or up_value is None or down_value is None or span <= ZERO:
            continue
        positive = (up_value / span) * Decimal("100")
        negative = (down_value / span) * Decimal("100")
        plus_di[index] = positive
        minus_di[index] = negative
        total = positive + negative
        if total > ZERO:
            dx[index] = (abs(positive - negative) / total) * Decimal("100")

    defined = [value for value in dx if value is not None]
    adx: list[Decimal | None] = [None] * size
    if len(defined) >= period:
        smoothed_dx = wilder_smoothing(defined, period)
        # Map the smoothed run back onto the original indices it came from.
        positions = [index for index, value in enumerate(dx) if value is not None]
        for offset, index in enumerate(positions):
            adx[index] = smoothed_dx[offset]
    return tuple(plus_di), tuple(minus_di), tuple(adx)


def obv(candles: Sequence[Candle]) -> Series:
    """On-balance volume: a running total that adds volume on up closes and subtracts on down.

    The point is divergence — OBV can keep climbing while price stalls, which says buying
    is being absorbed. It is cumulative from the first bar, so only *changes* in it are
    meaningful; the absolute level depends on where the series happens to start.
    """
    if not candles:
        return ()
    out: list[Decimal | None] = [ZERO]
    total = ZERO
    for index in range(1, len(candles)):
        previous = candles[index - 1].close
        current = candles[index].close
        if current > previous:
            total += candles[index].volume
        elif current < previous:
            total -= candles[index].volume
        out.append(total)
    return tuple(out)


def require_value(series: Series, index: int, name: str) -> Decimal:
    """Read an indicator value, raising a clear error if it is still warming up.

    Raises:
        InsufficientDataError: if the value is unavailable.

    """
    from quantflow.core.errors import InsufficientDataError

    if index >= len(series) or series[index] is None:
        raise InsufficientDataError(
            f"indicator {name!r} has no value at index {index}; "
            "the strategy needs a longer warm-up window",
            indicator=name,
            index=index,
        )
    value = series[index]
    assert value is not None
    return value
