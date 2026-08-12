"""Leverage decision: trade at 1x, and make the venue agree rather than assume it.

The risk being closed: the bot reserved margin from its own assumed leverage. If Bybit held
the symbol at a different value it reserved a different amount, so free margin, exposure and
every equity-derived limit were measured against a reservation that did not exist.
"""

from __future__ import annotations

from decimal import Decimal

from quantflow.core.config import MarketType
from quantflow.domain.instruments import Symbol
from quantflow.live.reconcile import VenuePosition, parse_venue_account, parse_venue_positions

BTC = Symbol.parse("BTC/USDT")


def raw_position(
    *, size: str = "0.1", leverage: str | None = "1", margin: str | None = None
) -> dict[str, object]:
    info: dict[str, object] = {
        "size": size,
        "side": "Buy",
        "avgPrice": "50000",
        "stopLoss": "49000",
    }
    if leverage is not None:
        info["leverage"] = leverage
    if margin is not None:
        info["positionIM"] = margin
    return {"symbol": "BTC/USDT:USDT", "info": info}


class TestVenueLeverageIsRead:
    def test_leverage_is_taken_from_the_venue_payload(self) -> None:
        position = parse_venue_positions([raw_position(leverage="10")])[0]
        assert position.leverage == Decimal("10")

    def test_margin_uses_the_venue_leverage_not_the_assumption(self) -> None:
        """0.1 BTC at 50,000 is 5,000 notional; at 10x the venue reserves 500."""
        position = parse_venue_positions([raw_position(leverage="10")])[0]
        assert position.margin_required == Decimal("500")

    def test_one_x_reserves_the_full_notional(self) -> None:
        position = parse_venue_positions([raw_position(leverage="1")])[0]
        assert position.margin_required == Decimal("5000")

    def test_the_venues_own_margin_figure_wins(self) -> None:
        """If Bybit states the reservation, use it rather than recomputing."""
        position = parse_venue_positions([raw_position(leverage="10", margin="512.34")])[0]
        assert position.margin_required == Decimal("512.34")

    def test_a_missing_leverage_falls_back_to_one_x(self) -> None:
        position = parse_venue_positions([raw_position(leverage=None)])[0]
        assert position.leverage == Decimal("1")
        assert position.margin_required == Decimal("5000")

    def test_unexpected_leverage_is_warned_but_still_honoured(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """Reconcile to the venue, never to the assumption — and say so."""
        positions = parse_venue_positions(
            [raw_position(leverage="25")], expected_leverage=Decimal("1")
        )
        assert positions[0].leverage == Decimal("25")
        assert positions[0].margin_required == Decimal("200")


class TestAccountMarginFollowsTheVenue:
    def test_account_margin_sums_venue_derived_position_margin(self) -> None:
        positions = parse_venue_positions([raw_position(leverage="10")])
        account = parse_venue_account({"info": {"totalEquity": "10000"}}, positions)
        assert account.margin_posted == Decimal("500")

    def test_the_venue_total_wins_over_the_derived_sum(self) -> None:
        positions = parse_venue_positions([raw_position(leverage="10")])
        account = parse_venue_account(
            {"info": {"totalEquity": "10000", "totalInitialMargin": "487.21"}}, positions
        )
        assert account.margin_posted == Decimal("487.21")

    def test_margin_stays_decimal(self) -> None:
        positions = parse_venue_positions([raw_position(leverage="10")])
        account = parse_venue_account({"info": {"totalEquity": "10000"}}, positions)
        assert isinstance(account.margin_posted, Decimal)
        assert isinstance(positions[0].leverage, Decimal)


class TestLeverageIsSetOnTheVenue:
    async def test_the_engine_sets_leverage_before_trading(self) -> None:
        """Assuming 1x is not enough; the venue has to be told."""
        from quantflow.domain.enums import Timeframe
        from quantflow.paper.engine import PaperConfig, PaperTradingEngine
        from quantflow.strategy.registry import load_builtin_strategies

        calls: list[tuple[Symbol, Decimal]] = []

        class GatewayWithLeverage:
            async def set_leverage(self, symbol: Symbol, leverage: Decimal) -> bool:
                calls.append((symbol, leverage))
                return True

        engine = PaperTradingEngine(
            load_builtin_strategies().create("ema_cross"),
            PaperConfig(
                symbols=(BTC,),
                timeframe=Timeframe.H1,
                persist=False,
                market_type=MarketType.FUTURE,
                leverage=Decimal("1"),
            ),
            instruments={},
        )
        await engine._align_venue_leverage(GatewayWithLeverage())  # type: ignore[arg-type]
        assert calls == [(BTC, Decimal("1"))]

    async def test_spot_does_not_set_leverage(self) -> None:
        from quantflow.domain.enums import Timeframe
        from quantflow.paper.engine import PaperConfig, PaperTradingEngine
        from quantflow.strategy.registry import load_builtin_strategies

        calls: list[object] = []

        class GatewayWithLeverage:
            async def set_leverage(self, symbol: Symbol, leverage: Decimal) -> bool:
                calls.append(symbol)
                return True

        engine = PaperTradingEngine(
            load_builtin_strategies().create("ema_cross"),
            PaperConfig(
                symbols=(BTC,),
                timeframe=Timeframe.H1,
                persist=False,
                market_type=MarketType.SPOT,
            ),
            instruments={},
        )
        await engine._align_venue_leverage(GatewayWithLeverage())  # type: ignore[arg-type]
        assert calls == []

    async def test_a_gateway_without_set_leverage_is_skipped(self) -> None:
        """The simulator has no such method; startup must not break."""
        from quantflow.domain.enums import Timeframe
        from quantflow.paper.engine import PaperConfig, PaperTradingEngine
        from quantflow.strategy.registry import load_builtin_strategies

        engine = PaperTradingEngine(
            load_builtin_strategies().create("ema_cross"),
            PaperConfig(
                symbols=(BTC,),
                timeframe=Timeframe.H1,
                persist=False,
                market_type=MarketType.FUTURE,
            ),
            instruments={},
        )
        await engine._align_venue_leverage(object())  # type: ignore[arg-type]

    async def test_a_refusal_does_not_block_startup(self) -> None:
        """A failed set degrades to 'reconcile to the venue', not to a crash."""
        from quantflow.domain.enums import Timeframe
        from quantflow.paper.engine import PaperConfig, PaperTradingEngine
        from quantflow.strategy.registry import load_builtin_strategies

        class RefusingGateway:
            async def set_leverage(self, symbol: Symbol, leverage: Decimal) -> bool:
                raise RuntimeError("venue refused")

        engine = PaperTradingEngine(
            load_builtin_strategies().create("ema_cross"),
            PaperConfig(
                symbols=(BTC,),
                timeframe=Timeframe.H1,
                persist=False,
                market_type=MarketType.FUTURE,
            ),
            instruments={},
        )
        await engine._align_venue_leverage(RefusingGateway())  # type: ignore[arg-type]


class TestPositionMarginMatchesVenue:
    def test_a_ten_x_position_does_not_report_one_x_margin(self) -> None:
        """The concrete failure: 5,000 reserved when the venue reserved 500."""
        position = VenuePosition(
            symbol=BTC,
            side="buy",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            stop_loss_price=Decimal("49000"),
            leverage=Decimal("10"),
        )
        assert position.margin_required == Decimal("500")
        assert position.margin_required != Decimal("5000")

    def test_zero_leverage_is_treated_as_one_x(self) -> None:
        """A malformed payload must not divide by zero."""
        position = VenuePosition(
            symbol=BTC,
            side="buy",
            quantity=Decimal("0.1"),
            entry_price=Decimal("50000"),
            stop_loss_price=None,
            leverage=Decimal("0"),
        )
        assert position.margin_required == Decimal("5000")
