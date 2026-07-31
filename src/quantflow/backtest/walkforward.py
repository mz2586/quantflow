"""Walk-forward analysis.

A single backtest over one period tells you almost nothing about whether an edge is real:
with enough parameters, any historical series can be fitted. Walk-forward splits the data
into consecutive train/test windows, so the reported performance is always measured on
data the parameters were **not** chosen on.

Windows are **anchored in time and never overlap between train and test**. The train
window always ends strictly before its test window begins — an overlap of even one bar
leaks the answer into the exam.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from quantflow.core.errors import InsufficientDataError, ValidationError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.enums import Timeframe
from quantflow.domain.market import Candle

logger = get_logger(__name__)

#: Fewer windows than this cannot distinguish a real edge from luck.
MIN_WINDOWS_FOR_ROBUSTNESS = 3

#: How many windows the CLI summary prints before truncating.
WINDOW_PREVIEW_COUNT = 5


@dataclass(frozen=True, slots=True)
class Window:
    """One train/test split."""

    index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    def __post_init__(self) -> None:
        """Assert the windows are ordered and disjoint."""
        if self.train_end > self.test_start:
            raise ValidationError(
                f"window {self.index}: train ends at {self.train_end.isoformat()} but test "
                f"starts at {self.test_start.isoformat()}; overlapping windows leak the "
                "answer into the test"
            )
        if self.train_start >= self.train_end:
            raise ValidationError(f"window {self.index}: empty train range")
        if self.test_start >= self.test_end:
            raise ValidationError(f"window {self.index}: empty test range")

    @property
    def train_days(self) -> Decimal:
        """Length of the training window in days."""
        return Decimal(str((self.train_end - self.train_start).total_seconds() / 86400))

    @property
    def test_days(self) -> Decimal:
        """Length of the test window in days."""
        return Decimal(str((self.test_end - self.test_start).total_seconds() / 86400))

    def split(self, candles: Sequence[Candle]) -> tuple[list[Candle], list[Candle]]:
        """Partition candles into ``(train, test)`` by this window's bounds."""
        train = [c for c in candles if self.train_start <= c.open_time < self.train_end]
        test = [c for c in candles if self.test_start <= c.open_time < self.test_end]
        return train, test

    def to_dict(self) -> dict[str, Any]:
        """Serialise for reports."""
        return {
            "index": self.index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "train_days": float(self.train_days),
            "test_days": float(self.test_days),
        }


def rolling_windows(
    start: datetime,
    end: datetime,
    *,
    train: timedelta,
    test: timedelta,
    step: timedelta | None = None,
) -> list[Window]:
    """Build rolling (sliding) train/test windows.

    Each window trains on a fixed-length trailing period and tests on the period
    immediately after. The train window slides forward, so old data eventually drops out —
    appropriate when the market regime is expected to change.
    """
    if train <= timedelta(0) or test <= timedelta(0):
        raise ValidationError("train and test windows must be positive")
    if end <= start:
        raise ValidationError("end must be after start")

    advance = step or test
    windows: list[Window] = []
    train_start = start
    index = 0

    while True:
        train_end = train_start + train
        test_end = train_end + test
        if test_end > end:
            break
        windows.append(
            Window(
                index=index,
                train_start=train_start,
                train_end=train_end,
                test_start=train_end,
                test_end=test_end,
            )
        )
        index += 1
        train_start += advance

    if not windows:
        raise InsufficientDataError(
            f"range {start.date()}..{end.date()} is too short for a "
            f"{train.days}d train + {test.days}d test window"
        )
    return windows


def anchored_windows(
    start: datetime,
    end: datetime,
    *,
    initial_train: timedelta,
    test: timedelta,
) -> list[Window]:
    """Build anchored (expanding) train/test windows.

    The train window always starts at ``start`` and grows. Appropriate when more history is
    believed to help, and it uses every available observation — but it also means later
    windows are trained on far more data than earlier ones, so their results are not
    directly comparable.
    """
    if initial_train <= timedelta(0) or test <= timedelta(0):
        raise ValidationError("train and test windows must be positive")

    windows: list[Window] = []
    train_end = start + initial_train
    index = 0

    while train_end + test <= end:
        windows.append(
            Window(
                index=index,
                train_start=start,
                train_end=train_end,
                test_start=train_end,
                test_end=train_end + test,
            )
        )
        index += 1
        train_end += test

    if not windows:
        raise InsufficientDataError(
            f"range {start.date()}..{end.date()} is too short for a "
            f"{initial_train.days}d initial train + {test.days}d test window"
        )
    return windows


def windows_from_bars(
    candles: Sequence[Candle],
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
) -> list[Window]:
    """Build windows by bar count rather than wall-clock duration.

    More robust than duration-based splits when the data has gaps: a "30-day" window over a
    sparsely-traded pair can contain far fewer bars than expected.
    """
    if train_bars < 1 or test_bars < 1:
        raise ValidationError("train_bars and test_bars must be at least 1")
    if len(candles) < train_bars + test_bars:
        raise InsufficientDataError(
            f"{len(candles)} bars available, need at least {train_bars + test_bars}"
        )

    ordered = sorted(candles, key=lambda candle: candle.open_time)
    advance = step_bars or test_bars
    step_delta = ordered[0].timeframe.delta
    windows: list[Window] = []
    index = 0
    cursor = 0

    while cursor + train_bars + test_bars <= len(ordered):
        train_slice = ordered[cursor : cursor + train_bars]
        test_slice = ordered[cursor + train_bars : cursor + train_bars + test_bars]
        windows.append(
            Window(
                index=index,
                train_start=train_slice[0].open_time,
                train_end=test_slice[0].open_time,
                test_start=test_slice[0].open_time,
                test_end=test_slice[-1].open_time + step_delta,
            )
        )
        index += 1
        cursor += advance

    return windows


@dataclass(frozen=True, slots=True)
class WindowResult:
    """In-sample and out-of-sample metrics for one window."""

    window: Window
    params: dict[str, Any]
    in_sample: Any
    """PerformanceMetrics for the train period."""
    out_of_sample: Any
    """PerformanceMetrics for the test period."""

    @property
    def degradation(self) -> Decimal:
        """Out-of-sample Sharpe as a fraction of in-sample Sharpe."""
        from quantflow.backtest.metrics import degradation_ratio

        return degradation_ratio(self.in_sample, self.out_of_sample)

    @property
    def generalised(self) -> bool:
        """Whether the edge survived out of sample at all."""
        return bool(self.out_of_sample.sharpe_ratio > ZERO)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for reports."""
        return {
            "window": self.window.to_dict(),
            "params": self.params,
            "in_sample": self.in_sample.to_dict(),
            "out_of_sample": self.out_of_sample.to_dict(),
            "degradation": float(self.degradation),
            "generalised": self.generalised,
        }


@dataclass(slots=True)
class WalkForwardReport:
    """Aggregate results across every window."""

    results: list[WindowResult] = field(default_factory=list)

    @property
    def window_count(self) -> int:
        """Number of windows evaluated."""
        return len(self.results)

    @property
    def generalisation_rate(self) -> Decimal:
        """Fraction of windows that stayed profitable out of sample.

        The headline number. A strategy that generalises in 3 of 10 windows is
        curve-fitted, whatever its aggregate return says.
        """
        if not self.results:
            return ZERO
        survived = sum(1 for result in self.results if result.generalised)
        return Decimal(survived) / Decimal(len(self.results))

    @property
    def mean_degradation(self) -> Decimal:
        """Average out-of-sample-to-in-sample Sharpe ratio."""
        if not self.results:
            return ZERO
        total = sum((result.degradation for result in self.results), ZERO)
        return total / Decimal(len(self.results))

    @property
    def mean_out_of_sample_sharpe(self) -> Decimal:
        """Average out-of-sample Sharpe — the only Sharpe worth quoting."""
        if not self.results:
            return ZERO
        total = sum((result.out_of_sample.sharpe_ratio for result in self.results), ZERO)
        return total / Decimal(len(self.results))

    @property
    def is_robust(self) -> bool:
        """A deliberately demanding definition of "this might be real".

        Requires a majority of windows to generalise, positive average out-of-sample
        performance, and degradation that is not catastrophic. Anything less should not be
        traded with real money.
        """
        return (
            self.window_count >= MIN_WINDOWS_FOR_ROBUSTNESS
            and self.generalisation_rate >= Decimal("0.6")
            and self.mean_out_of_sample_sharpe > ZERO
            and self.mean_degradation > Decimal("0.3")
        )

    def summary(self) -> dict[str, Any]:
        """Headline figures."""
        return {
            "windows": self.window_count,
            "generalisation_rate": float(self.generalisation_rate),
            "mean_degradation": float(self.mean_degradation),
            "mean_out_of_sample_sharpe": float(self.mean_out_of_sample_sharpe),
            "is_robust": self.is_robust,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full report."""
        return {
            "summary": self.summary(),
            "windows": [result.to_dict() for result in self.results],
        }


def describe_windows(windows: Sequence[Window], timeframe: Timeframe) -> str:
    """Human-readable window plan for the CLI."""
    if not windows:
        return "no windows"
    lines = [f"{len(windows)} walk-forward windows ({timeframe.value}):"]
    for window in windows[:WINDOW_PREVIEW_COUNT]:
        lines.append(
            f"  #{window.index}: train {window.train_start.date()}..{window.train_end.date()} "
            f"→ test {window.test_start.date()}..{window.test_end.date()}"
        )
    if len(windows) > WINDOW_PREVIEW_COUNT:
        lines.append(f"  ... and {len(windows) - WINDOW_PREVIEW_COUNT} more")
    return "\n".join(lines)
