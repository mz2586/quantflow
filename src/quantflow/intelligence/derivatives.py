"""Funding rates and open interest, from the perpetual-futures market.

**These are futures metrics, and the platform trades spot.** Binance spot has no funding
rate and no open interest — they do not exist for a spot pair. What is fetched here comes
from the USD-M perpetual (`BTC/USDT:USDT`), and it is a *positioning* signal about the
same underlying, not a property of the instrument being traded.

That distinction is carried through the type rather than left in a comment, because the
alternative is somebody reading "open interest: 106,392" on a spot dashboard and
concluding the spot market has open interest. Every value here is labelled with the
perpetual symbol it came from.

Funding is worth watching precisely because it is the crowd's cost of carry: persistently
positive funding means longs are paying to stay long, which is what crowded looks like
before it unwinds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from quantflow.core.errors import ExchangeError
from quantflow.core.logging import get_logger
from quantflow.core.precision import ZERO
from quantflow.domain.instruments import Symbol

logger = get_logger(__name__)

#: Funding is paid every eight hours on Binance perpetuals, so an annualised view needs
#: three payments a day.
FUNDING_PERIODS_PER_YEAR = Decimal("1095")


def perpetual_symbol(symbol: Symbol) -> str:
    """The USD-M perpetual ticker for a spot symbol.

    ``BTC/USDT`` becomes ``BTC/USDT:USDT``. Only USDT-quoted pairs have a direct
    perpetual equivalent; anything else has no counterpart and callers must handle that.
    """
    return f"{symbol.base}/{symbol.quote}:{symbol.quote}"


def has_perpetual(symbol: Symbol) -> bool:
    """Whether this spot symbol plausibly has a USD-M perpetual counterpart."""
    return symbol.quote.upper() == "USDT"


@dataclass(frozen=True, slots=True)
class FundingSnapshot:
    """A perpetual funding rate observation."""

    spot_symbol: Symbol
    perpetual: str
    rate: Decimal
    """Per-period rate. Positive means longs pay shorts."""
    observed_at: datetime

    @property
    def annualised(self) -> Decimal:
        """The rate extrapolated to a year, for comparison against a return."""
        return self.rate * FUNDING_PERIODS_PER_YEAR

    @property
    def is_crowded_long(self) -> bool:
        """Whether longs are paying enough to suggest a crowded book.

        0.03% per period is roughly 33% annualised — a level at which the long side is
        paying real money to stay in, which historically precedes unwinds.
        """
        return self.rate >= Decimal("0.0003")

    @property
    def is_crowded_short(self) -> bool:
        """Whether shorts are paying enough to suggest a crowded book."""
        return self.rate <= Decimal("-0.0003")

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the API and reports."""
        return {
            "spot_symbol": str(self.spot_symbol),
            "perpetual": self.perpetual,
            "rate": str(self.rate),
            "annualised": str(self.annualised),
            "crowded_long": self.is_crowded_long,
            "crowded_short": self.is_crowded_short,
            "observed_at": self.observed_at.isoformat(),
            "source": "perpetual futures, not spot",
        }


@dataclass(frozen=True, slots=True)
class OpenInterestSnapshot:
    """A perpetual open-interest observation."""

    spot_symbol: Symbol
    perpetual: str
    contracts: Decimal
    """Open interest in base-currency units."""
    notional: Decimal | None
    observed_at: datetime

    def change_from(self, earlier: OpenInterestSnapshot) -> Decimal | None:
        """Fractional change against an earlier reading.

        Rising open interest with rising price is new money entering; rising open
        interest with falling price is new shorts. The level alone says neither, which
        is why the comparison is the useful operation rather than the reading.
        """
        if earlier.contracts <= ZERO:
            return None
        return (self.contracts - earlier.contracts) / earlier.contracts

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the API and reports."""
        return {
            "spot_symbol": str(self.spot_symbol),
            "perpetual": self.perpetual,
            "contracts": str(self.contracts),
            "notional": str(self.notional) if self.notional is not None else None,
            "observed_at": self.observed_at.isoformat(),
            "source": "perpetual futures, not spot",
        }


class DerivativesSource(Protocol):
    """The subset of a futures venue this module needs.

    A protocol rather than a concrete client so the laboratory and the tests can run
    without a network, and so nothing here depends on CCXT's shape.
    """

    async def fetch_funding_rate(self, symbol: str) -> FundingSnapshot | None:
        """Current funding rate, or ``None`` when the venue has no perpetual for it."""
        ...

    async def fetch_open_interest(self, symbol: str) -> OpenInterestSnapshot | None:
        """Current open interest, or ``None`` when unavailable."""
        ...


class CcxtDerivativesSource:
    """Funding and open interest from Binance USD-M perpetuals via CCXT.

    Read-only and credential-free: these are public endpoints, and this module must never
    be a path through which an authenticated call could be made.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    async def fetch_funding_rate(self, symbol: str) -> FundingSnapshot | None:
        """Fetch the current funding rate for a spot symbol's perpetual."""
        parsed = Symbol.parse(symbol) if isinstance(symbol, str) else symbol
        if not has_perpetual(parsed):
            return None
        ticker = perpetual_symbol(parsed)
        try:
            payload = await self._call("fetch_funding_rate", ticker)
        except ExchangeError:
            logger.warning("derivatives.funding_unavailable", perpetual=ticker)
            return None

        rate = payload.get("fundingRate")
        if rate is None:
            return None
        return FundingSnapshot(
            spot_symbol=parsed,
            perpetual=ticker,
            rate=Decimal(str(rate)),
            observed_at=_timestamp(payload.get("timestamp")),
        )

    async def fetch_open_interest(self, symbol: str) -> OpenInterestSnapshot | None:
        """Fetch current open interest for a spot symbol's perpetual."""
        parsed = Symbol.parse(symbol) if isinstance(symbol, str) else symbol
        if not has_perpetual(parsed):
            return None
        ticker = perpetual_symbol(parsed)
        try:
            payload = await self._call("fetch_open_interest", ticker)
        except ExchangeError:
            logger.warning("derivatives.open_interest_unavailable", perpetual=ticker)
            return None

        amount = payload.get("openInterestAmount")
        if amount is None:
            return None
        value = payload.get("openInterestValue")
        return OpenInterestSnapshot(
            spot_symbol=parsed,
            perpetual=ticker,
            contracts=Decimal(str(amount)),
            notional=Decimal(str(value)) if value is not None else None,
            observed_at=_timestamp(payload.get("timestamp")),
        )

    async def _call(self, method: str, ticker: str) -> dict[str, Any]:
        """Invoke a CCXT method, normalising failure to ExchangeError.

        Raises:
            ExchangeError: on any venue or transport failure.

        """
        try:
            result = getattr(self._client, method)(ticker)
            payload = await result if hasattr(result, "__await__") else result
        except Exception as exc:  # CCXT raises a wide and unstable exception surface
            raise ExchangeError(f"{method} failed for {ticker}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ExchangeError(f"{method} returned {type(payload).__name__}, expected a mapping")
        return payload


def funding_trend(history: Sequence[FundingSnapshot]) -> Decimal | None:
    """Mean funding across a history, or ``None`` when empty.

    The average matters more than the latest print: a single elevated reading is noise,
    while a sustained positive average is the long side paying continuously to stay in.
    """
    if not history:
        return None
    return sum((item.rate for item in history), ZERO) / Decimal(len(history))


def _timestamp(value: Any) -> datetime:
    """Milliseconds since epoch to an aware datetime, defaulting to now."""
    if value is None:
        return datetime.now(UTC)
    return datetime.fromtimestamp(float(value) / 1000, tz=UTC)


__all__ = [
    "FUNDING_PERIODS_PER_YEAR",
    "CcxtDerivativesSource",
    "DerivativesSource",
    "FundingSnapshot",
    "OpenInterestSnapshot",
    "funding_trend",
    "has_perpetual",
    "perpetual_symbol",
]
