"""Typed exception hierarchy.

Every error raised by QuantFlow derives from :class:`QuantFlowError`. Errors carry a
stable ``code`` used by the API error envelope and by log-based alerting, plus an
optional ``details`` mapping that must never contain secrets.
"""

from __future__ import annotations

from typing import Any


class QuantFlowError(Exception):
    """Base class for all QuantFlow errors."""

    code: str = "quantflow_error"
    http_status: int = 500

    def __init__(self, message: str, /, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def to_dict(self) -> dict[str, Any]:
        """Render the error as a JSON-serialisable envelope body."""
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:
        if not self.details:
            return self.message
        rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(self.details.items()))
        return f"{self.message} ({rendered})"


# --------------------------------------------------------------------------- #
# Configuration & wiring
# --------------------------------------------------------------------------- #
class ConfigurationError(QuantFlowError):
    """Invalid, missing or contradictory configuration."""

    code = "configuration_error"


class DependencyNotRegisteredError(QuantFlowError):
    """A dependency was requested from the container but never registered."""

    code = "dependency_not_registered"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class ValidationError(QuantFlowError):
    """Caller-supplied data violates a domain invariant."""

    code = "validation_error"
    http_status = 422


class NotFoundError(QuantFlowError):
    """A requested entity does not exist."""

    code = "not_found"
    http_status = 404


class ConflictError(QuantFlowError):
    """The requested state transition conflicts with the current state."""

    code = "conflict"
    http_status = 409


class AuthenticationError(QuantFlowError):
    """Missing or invalid credentials on an inbound request."""

    code = "authentication_error"
    http_status = 401


# --------------------------------------------------------------------------- #
# Infrastructure
# --------------------------------------------------------------------------- #
class InfrastructureError(QuantFlowError):
    """Base class for failures in an external system."""

    code = "infrastructure_error"
    http_status = 503


class DatabaseError(InfrastructureError):
    """Database access failed."""

    code = "database_error"


class CacheError(InfrastructureError):
    """Redis access failed."""

    code = "cache_error"


class LockAcquisitionError(CacheError):
    """A distributed lock could not be acquired within the timeout."""

    code = "lock_acquisition_error"


# --------------------------------------------------------------------------- #
# Exchange
# --------------------------------------------------------------------------- #
class ExchangeError(InfrastructureError):
    """Base class for exchange-side failures."""

    code = "exchange_error"


class ExchangeConnectionError(ExchangeError):
    """Network-level failure talking to the exchange."""

    code = "exchange_connection_error"


class ExchangeTimeoutError(ExchangeError):
    """The exchange did not respond within the configured timeout."""

    code = "exchange_timeout"


class RateLimitError(ExchangeError):
    """The exchange rejected the request for exceeding its rate limits."""

    code = "rate_limited"
    http_status = 429


class ExchangeAuthenticationError(ExchangeError):
    """The exchange rejected our API credentials."""

    code = "exchange_authentication_error"
    http_status = 502


class InsufficientFundsError(ExchangeError):
    """The account balance cannot support the requested order."""

    code = "insufficient_funds"
    http_status = 400


class OrderRejectedError(ExchangeError):
    """The exchange rejected the order outright."""

    code = "order_rejected"
    http_status = 400


class ProductAgreementRequiredError(OrderRejectedError):
    """The venue will not trade this product until an agreement is signed.

    Bybit gates its non-crypto perpetuals behind per-product terms that have to be
    accepted in the account UI. Until they are, market data, instrument metadata and
    streams all work normally and only order placement is refused — which makes this look
    like an order problem when it is an account problem.

    Separated from a plain :class:`OrderRejectedError` because the correct response is
    different in kind: there is no order that would succeed, so retrying, resizing or
    failing the session are all wrong. The asset class is set aside and the operator is
    told what to sign.
    """

    code = "product_agreement_required"
    http_status = 403

    @property
    def venue_error(self) -> str:
        """The venue's own code for the missing agreement, or ``""``.

        Surfaced as an attribute rather than left in ``details`` because callers branch on
        it — one code per agreement, and the operator has to be told which one to sign.
        """
        return str(self.details.get("venue_error", ""))


class InvalidSymbolError(ExchangeError):
    """The symbol is unknown to the exchange or not tradable."""

    code = "invalid_symbol"
    http_status = 400


class MarketDataError(ExchangeError):
    """Market data was unavailable, incomplete or failed integrity checks."""

    code = "market_data_error"


# --------------------------------------------------------------------------- #
# Trading domain
# --------------------------------------------------------------------------- #
class TradingError(QuantFlowError):
    """Base class for trading-logic failures."""

    code = "trading_error"
    http_status = 400


class StrategyError(TradingError):
    """A strategy raised, or is misconfigured."""

    code = "strategy_error"


class RiskViolationError(TradingError):
    """An order was blocked by the risk engine."""

    code = "risk_violation"
    http_status = 403

    def __init__(self, message: str, /, *, rule: str, **details: Any) -> None:
        super().__init__(message, rule=rule, **details)
        self.rule = rule


class KillSwitchEngagedError(TradingError):
    """Trading is halted because the kill switch is latched."""

    code = "kill_switch_engaged"
    http_status = 423


class ExecutionError(TradingError):
    """The execution engine could not complete the requested action."""

    code = "execution_error"


class InvalidOrderTransitionError(ExecutionError):
    """An order state transition is not permitted by the OMS state machine."""

    code = "invalid_order_transition"
    http_status = 409


class LiveTradingNotArmedError(TradingError):
    """Live trading was requested without the explicit confirmation token."""

    code = "live_trading_not_armed"
    http_status = 403


class BacktestError(QuantFlowError):
    """The backtest could not run or produced an invalid result."""

    code = "backtest_error"
    http_status = 400


class InsufficientDataError(BacktestError):
    """Not enough historical data to run the requested computation."""

    code = "insufficient_data"


# --------------------------------------------------------------------------- #
# Outbound integrations
# --------------------------------------------------------------------------- #
class NotificationError(InfrastructureError):
    """A notification transport failed to deliver."""

    code = "notification_error"


class AIProviderError(InfrastructureError):
    """An AI/LLM provider call failed."""

    code = "ai_provider_error"
