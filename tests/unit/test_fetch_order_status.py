"""Reading an order's state back from Bybit — the call itself, and what it is read as.

Two defects, both found by running the demo bot against the demo venue, both of which made
the OMS blind to any order outcome that produces no execution.

1. ``BybitGateway.fetch_order`` passed FOUR positionals to CCXT's ``fetch_order``, whose
   signature is ``(id, symbol, params)`` — three — not the ``(id, symbol, since, params)``
   shape ``fetch_open_orders`` and ``fetch_my_trades`` use. Every call raised ``TypeError``
   before it reached the network, so no order was ever read back from the venue that had
   just accepted it. A rejection or a cancel leaves no fill behind to reconcile from, so
   those orders sat at ``NEW`` for the life of the session.

2. ``CCXT_TO_ORDER_STATUS`` carried Bybit V5's raw vocabulary in CamelCase while
   :func:`parse_order_status` lowercases before looking up, so those entries could never be
   hit. They fell through to the ``NEW`` default: ``PartiallyFilled`` read as untouched,
   and ``Deactivated`` / ``PartiallyFilledCanceled`` — both terminal — read as still
   working. Three of the four statuses the OMS must tell apart were wrong on the raw path.

3. With the call fixed, ``fetch_order`` still could not see a finished order. CCXT's
   ``fetchOrder`` serves Bybit's *realtime* endpoint — open orders plus a short tail of
   recently-closed ones — and raises ``OrderNotFound`` beyond it. So the only orders whose
   status could be read were the ones still working, which is the opposite of what the OMS
   needs: it already knows about those. Twenty-six orders in the demo session sat at
   ``new``/``partially_filled`` for a day while order history reported every one of them
   ``Filled``.

The signature test binds against the *installed* CCXT rather than a hand-copied stub. A
fake that accepts whatever it is given proves only that the fake is permissive; binding to
``ccxt.bybit.fetch_order`` fails the moment the real contract and the call disagree.
"""

from __future__ import annotations

import inspect
from decimal import Decimal
from typing import Any

import ccxt.async_support as ccxt
import pytest
from pydantic import SecretStr

from quantflow.core.config import ExchangeSettings, MarketType
from quantflow.core.errors import ExchangeAuthenticationError, NotFoundError
from quantflow.domain.enums import OrderStatus
from quantflow.domain.instruments import Instrument, Symbol
from quantflow.exchange.bybit.mapping import parse_order_status
from quantflow.exchange.bybit.rest import FETCH_ORDER_PARAMS, BybitGateway

ETH = Symbol.parse("ETH/USDT")


def eth_instrument() -> Instrument:
    return Instrument(
        symbol=ETH,
        price_tick=Decimal("0.1"),
        quantity_step=Decimal("0.1"),
        min_quantity=Decimal("0.1"),
        min_notional=Decimal("1E-8"),
        market_type=MarketType.FUTURE,
    )


class SignatureBoundClient:
    """A ``fetch_order`` that accepts exactly what the installed CCXT accepts.

    The call is bound against ``ccxt.bybit.fetch_order``'s real signature before anything
    is returned, so an argument list the live client would reject raises here too.
    """

    def __init__(
        self,
        status: str = "closed",
        *,
        filled: float = 0.1,
        realtime_knows: bool = True,
        history_knows: bool = True,
        history_only_conditional: bool = False,
    ) -> None:
        #: What the venue will claim the order's status is.
        self.status = status
        self.filled = filled
        #: Whether each endpoint has heard of the order, set per scenario.
        self.realtime_knows = realtime_knows
        self.history_knows = history_knows
        self.history_only_conditional = history_only_conditional
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.history_calls: list[dict[str, Any]] = []

    def _order(self, id: str, symbol: str | None) -> dict[str, Any]:
        return {
            "id": id,
            "symbol": symbol,
            "side": "buy",
            "type": "market",
            "amount": 0.1,
            "filled": self.filled,
            "average": 1893.9,
            "status": self.status,
            "info": {"orderStatus": self.status},
        }

    async def fetch_order(
        self, id: str, symbol: str | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        inspect.signature(ccxt.bybit.fetch_order).bind(
            self, id, symbol, params if params is not None else {}
        )
        self.calls.append(((id, symbol), dict(params or {})))
        if not self.realtime_knows:
            raise ccxt.OrderNotFound(f"Order {id} was not found.")
        return self._order(id, symbol)

    async def fetch_closed_order(
        self, id: str, symbol: str | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        inspect.signature(ccxt.bybit.fetch_closed_order).bind(
            self, id, symbol, params if params is not None else {}
        )
        self.history_calls.append(dict(params or {}))
        if not self.history_knows:
            raise ccxt.OrderNotFound(f"Order {id} was not found.")
        if self.history_only_conditional and not (params or {}).get("trigger"):
            raise ccxt.OrderNotFound(f"Order {id} was not found.")
        return self._order(id, symbol)


def realtime_miss(status: str, *, filled: float = 0.1, **kwargs: Any) -> SignatureBoundClient:
    """The venue the bug ran into: realtime has forgotten the order, history has not."""
    return SignatureBoundClient(status, filled=filled, realtime_knows=False, **kwargs)


class RefusingClient(SignatureBoundClient):
    """A venue that fails for a reason that is *not* "no such order"."""

    async def fetch_order(
        self, id: str, symbol: str | None = None, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        raise ccxt.AuthenticationError("api key expired")


class StubGateway(BybitGateway):
    """Gateway with only its CCXT client replaced."""

    def __init__(self, client: SignatureBoundClient) -> None:
        super().__init__(
            ExchangeSettings(
                name="bybit",
                api_key=SecretStr("k" * 18),
                api_secret=SecretStr("s" * 36),
                testnet=True,
                market_type=MarketType.FUTURE,
            )
        )
        self._client = client
        self._instruments.put(eth_instrument())


class TestFetchOrderCallShape:
    """Defect 1: the call must match CCXT's real ``fetch_order`` contract."""

    def test_ccxt_fetch_order_takes_three_arguments(self) -> None:
        """The premise of the fix, asserted rather than assumed.

        If a future CCXT gains a ``since`` parameter here, this fails first and explains
        why — rather than the gateway quietly passing the wrong thing again.
        """
        parameters = list(inspect.signature(ccxt.bybit.fetch_order).parameters)

        assert parameters == ["self", "id", "symbol", "params"]

    async def test_fetch_order_binds_against_the_real_signature(self) -> None:
        """The regression: a fourth positional raised TypeError before any network call."""
        client = SignatureBoundClient()
        gateway = StubGateway(client)
        gateway.register_venue_id("local-1", "venue-1")

        order = await gateway.fetch_order("local-1", ETH)

        assert order.venue_order_id == "venue-1"

    async def test_the_venue_order_id_and_symbol_are_what_is_sent(self) -> None:
        client = SignatureBoundClient()
        gateway = StubGateway(client)
        gateway.register_venue_id("local-1", "venue-1")

        await gateway.fetch_order("local-1", ETH)

        assert client.calls[0][0] == ("venue-1", "ETH/USDT:USDT")

    async def test_the_acknowledgement_params_are_still_sent(self) -> None:
        """CCXT raises ArgumentsRequired without it, so params must survive the fix."""
        client = SignatureBoundClient()
        gateway = StubGateway(client)
        gateway.register_venue_id("local-1", "venue-1")

        await gateway.fetch_order("local-1", ETH)

        assert client.calls[0][1] == FETCH_ORDER_PARAMS


class TestStatusReadBackFromTheVenue:
    """A fetched order must carry the venue's own verdict, not a default."""

    @pytest.mark.parametrize(
        ("venue_status", "filled", "expected"),
        [
            ("closed", 0.1, OrderStatus.FILLED),
            ("canceled", 0.0, OrderStatus.CANCELLED),
            ("rejected", 0.0, OrderStatus.REJECTED),
            ("open", 0.05, OrderStatus.NEW),
            ("expired", 0.0, OrderStatus.EXPIRED),
        ],
    )
    async def test_unified_status_survives_the_round_trip(
        self, venue_status: str, filled: float, expected: OrderStatus
    ) -> None:
        client = SignatureBoundClient(venue_status, filled=filled)
        gateway = StubGateway(client)
        gateway.register_venue_id("local-1", "venue-1")

        order = await gateway.fetch_order("local-1", ETH)

        assert order.status is expected

    @pytest.mark.parametrize(
        ("venue_status", "filled", "expected"),
        [
            ("Filled", 0.1, OrderStatus.FILLED),
            ("PartiallyFilled", 0.05, OrderStatus.PARTIALLY_FILLED),
            ("Cancelled", 0.0, OrderStatus.CANCELLED),
            ("Rejected", 0.0, OrderStatus.REJECTED),
        ],
    )
    async def test_raw_v5_status_survives_the_round_trip(
        self, venue_status: str, filled: float, expected: OrderStatus
    ) -> None:
        """A websocket order frame is not lowercased by CCXT before we see it."""
        client = SignatureBoundClient(venue_status, filled=filled)
        gateway = StubGateway(client)
        gateway.register_venue_id("local-1", "venue-1")

        order = await gateway.fetch_order("local-1", ETH)

        assert order.status is expected


class TestFinishedOrdersFallBackToHistory:
    """Defect 3: an order that has left the realtime window is not an unknown order."""

    @pytest.mark.parametrize(
        ("venue_status", "filled", "expected"),
        [
            ("Filled", 0.1, OrderStatus.FILLED),
            ("Cancelled", 0.0, OrderStatus.CANCELLED),
            ("Rejected", 0.0, OrderStatus.REJECTED),
            ("PartiallyFilledCanceled", 0.05, OrderStatus.CANCELLED),
        ],
    )
    async def test_a_finished_order_is_read_from_history(
        self, venue_status: str, filled: float, expected: OrderStatus
    ) -> None:
        """The 26 stuck rows: realtime says "not found", history says "Filled"."""
        client = realtime_miss(venue_status, filled=filled)
        gateway = StubGateway(client)
        gateway.register_venue_id("local-1", "venue-1")

        order = await gateway.fetch_order("local-1", ETH)

        assert order.status is expected

    async def test_history_is_only_consulted_after_realtime_misses(self) -> None:
        """A working order must still be answered by one call, not two."""
        client = SignatureBoundClient("open", filled=0.0)
        gateway = StubGateway(client)
        gateway.register_venue_id("local-1", "venue-1")

        await gateway.fetch_order("local-1", ETH)

        assert client.history_calls == []

    async def test_a_conditional_order_is_found_under_its_own_filter(self) -> None:
        """A stop lives under a different history filter and would otherwise be invisible."""
        client = realtime_miss("Deactivated", filled=0.0, history_only_conditional=True)
        gateway = StubGateway(client)
        gateway.register_venue_id("local-stop", "venue-stop")

        order = await gateway.fetch_order("local-stop", ETH)

        assert order.status is OrderStatus.CANCELLED

    async def test_the_plain_filter_is_tried_before_the_conditional_one(self) -> None:
        client = realtime_miss("Deactivated", filled=0.0, history_only_conditional=True)
        gateway = StubGateway(client)
        gateway.register_venue_id("local-stop", "venue-stop")

        await gateway.fetch_order("local-stop", ETH)

        assert client.history_calls == [{}, {"trigger": True}]

    async def test_an_id_neither_source_knows_is_still_not_found(self) -> None:
        """The fallback must not turn a genuine miss into a fabricated status."""
        client = SignatureBoundClient(realtime_knows=False, history_knows=False)
        gateway = StubGateway(client)
        gateway.register_venue_id("local-ghost", "venue-ghost")

        with pytest.raises(NotFoundError):
            await gateway.fetch_order("local-ghost", ETH)

    async def test_a_non_not_found_failure_is_not_swallowed(self) -> None:
        """An expired key must surface as itself, not as "the order does not exist".

        The fallback keys off "no such order" specifically. If it caught every failure, a
        credential or connectivity fault would be reported as a missing order — and the
        caller would conclude the venue has nothing, which is the most dangerous possible
        wrong answer about a live account.
        """
        client = RefusingClient()
        gateway = StubGateway(client)
        gateway.register_venue_id("local-1", "venue-1")

        with pytest.raises(ExchangeAuthenticationError):
            await gateway.fetch_order("local-1", ETH)

    async def test_a_non_not_found_failure_does_not_reach_history(self) -> None:
        """Nor should it spend a second venue read on a question already answered."""
        client = RefusingClient()
        gateway = StubGateway(client)
        gateway.register_venue_id("local-1", "venue-1")

        with pytest.raises(ExchangeAuthenticationError):
            await gateway.fetch_order("local-1", ETH)

        assert client.history_calls == []


class TestRawStatusVocabulary:
    """Defect 2: the CamelCase V5 entries were unreachable behind a lowercased lookup."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("New", OrderStatus.NEW),
            ("PartiallyFilled", OrderStatus.PARTIALLY_FILLED),
            ("Filled", OrderStatus.FILLED),
            ("Cancelled", OrderStatus.CANCELLED),
            ("Rejected", OrderStatus.REJECTED),
            ("Untriggered", OrderStatus.NEW),
            ("Triggered", OrderStatus.NEW),
            ("Deactivated", OrderStatus.CANCELLED),
            ("PartiallyFilledCanceled", OrderStatus.CANCELLED),
        ],
    )
    def test_bybit_v5_spelling(self, raw: str, expected: OrderStatus) -> None:
        assert parse_order_status(raw) is expected

    @pytest.mark.parametrize(
        "raw",
        ["Deactivated", "PartiallyFilledCanceled", "Cancelled", "Rejected", "Filled"],
    )
    def test_terminal_statuses_are_not_read_as_working(self, raw: str) -> None:
        """The consequence of the bug, stated directly.

        Every one of these fell through to ``NEW``, which is not merely unrecognised — it
        is the answer "this order is still live on the venue". An order the venue had
        finished with stayed open in the local book forever.
        """
        assert parse_order_status(raw).is_terminal

    def test_partially_filled_is_not_read_as_untouched(self) -> None:
        assert parse_order_status("PartiallyFilled") is not OrderStatus.NEW

    def test_an_unknown_status_still_defaults_to_working(self) -> None:
        """The default is deliberate and must survive: never orphan real exposure."""
        assert parse_order_status("SomeStatusBybitAddsLater") is OrderStatus.NEW
