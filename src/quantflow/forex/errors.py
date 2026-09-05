"""Typed errors for the Forex worker.

These sit under :class:`~quantflow.core.errors.QuantFlowError` so the API error envelope
and log-based alerting treat them like every other QuantFlow failure. Nothing here knows
about a specific venue — MT5, OANDA and any future transport raise the same types.
"""

from __future__ import annotations

from quantflow.core.errors import QuantFlowError


class ForexError(QuantFlowError):
    """Base class for every Forex-side failure."""

    code = "forex_error"


class ForexCapabilityError(ForexError):
    """The worker cannot run here, and the message says exactly why.

    Raised instead of an obscure ``ImportError``/``AttributeError`` when a transport is
    missing its platform, its package or its credentials. The message is meant to be
    pasted into a runbook.
    """

    code = "forex_capability_unavailable"
    http_status = 503


class ForexConnectionError(ForexError):
    """The venue could not be reached, or the session was lost."""

    code = "forex_connection_error"
    http_status = 502


class ForexAuthenticationError(ForexError):
    """The venue rejected our credentials."""

    code = "forex_authentication_error"
    http_status = 401


class ForexSymbolError(ForexError):
    """The symbol is unknown to the venue, or not selectable for trading."""

    code = "forex_symbol_error"
    http_status = 422


class ForexOrderRejectedError(ForexError):
    """The venue rejected the order outright."""

    code = "forex_order_rejected"
    http_status = 422


class StaleMarketDataError(ForexError):
    """A quote is older (or newer) than the freshness budget allows.

    FX is not 24/7. A quote that stopped updating is indistinguishable from a quiet
    market unless age is checked explicitly, so every price used for sizing or execution
    passes through :func:`quantflow.forex.protocol.ensure_fresh`.
    """

    code = "forex_stale_market_data"
    http_status = 503


class MarketClosedError(ForexError):
    """The instrument is outside its trading session (weekend, or a venue session gap)."""

    code = "forex_market_closed"
    http_status = 409
