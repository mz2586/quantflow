"""Strategy orchestrator: candidate construction, ranking, selection and ownership."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import pytest

from quantflow.core.errors import ValidationError
from quantflow.domain.enums import MarketRegime, PositionSide, SignalDirection
from quantflow.domain.instruments import Symbol
from quantflow.domain.positions import ClosedTrade
from quantflow.domain.signals import Signal
from quantflow.orchestrator import (
    MIN_TRADES_FOR_EVIDENCE,
    WEIGHTS,
    Candidate,
    StrategyOrchestrator,
    StrategyRecord,
    rank,
    score_candidate,
)
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.strategy.registry import load_builtin_strategies
from tests.conftest import REFERENCE_TIME
from tests.unit.test_strategies import make_context, open_long

SYMBOL = Symbol.parse("BTC/USDT")
OTHER = Symbol.parse("ETH/USDT")


class StubStrategy(Strategy):
    """A member that always emits the signal it was handed."""

    params_model = StrategyParams

    def __init__(self, strategy_id: str, signal: Signal | None = None) -> None:
        self.strategy_id = strategy_id  # type: ignore[misc]
        super().__init__(None)
        self._signal = signal

    @property
    def warmup_bars(self) -> int:
        return 1

    def generate(self, context: StrategyContext) -> Signal:
        if self._signal is None:
            return context.hold("stub holds", self.strategy_id)
        return self._signal


def entry(
    strategy_id: str,
    *,
    symbol: Symbol = SYMBOL,
    conviction: str = "1.0",
    stop: str | None = "90",
    target: str | None = "130",
    price: str = "100",
) -> Signal:
    return Signal(
        symbol=symbol,
        direction=SignalDirection.LONG,
        timestamp=REFERENCE_TIME,
        strategy_id=strategy_id,
        conviction=Decimal(conviction),
        reference_price=Decimal(price),
        stop_loss_price=Decimal(stop) if stop is not None else None,
        take_profit_price=Decimal(target) if target is not None else None,
        reason=f"{strategy_id} entry",
    )


def candidate_from(signal: Signal) -> Candidate:
    return Candidate(
        symbol=signal.symbol,
        strategy_id=signal.strategy_id,
        direction=signal.direction,
        confidence=signal.conviction,
        entry=signal.reference_price or Decimal("100"),
        stop_loss=signal.stop_loss_price,
        take_profit=signal.take_profit_price,
        timestamp=signal.timestamp,
        signal=signal,
    )


def scored(signal: Signal, **kwargs: object) -> Candidate:
    return score_candidate(
        candidate_from(signal),
        regime=kwargs.get("regime", MarketRegime.UNKNOWN),  # type: ignore[arg-type]
        records=kwargs.get("records", {}),  # type: ignore[arg-type]
        open_symbols=kwargs.get("open_symbols", frozenset()),  # type: ignore[arg-type]
        cost_rate=kwargs.get("cost_rate", Decimal("0.002")),  # type: ignore[arg-type]
    )


class TestScoring:
    def test_weights_sum_to_one(self) -> None:
        assert sum(WEIGHTS.values()) == Decimal("1")

    def test_score_stays_within_zero_and_one(self) -> None:
        result = scored(entry("s"))
        assert Decimal("0") <= result.score <= Decimal("1")

    def test_higher_conviction_outranks_lower_all_else_equal(self) -> None:
        strong = scored(entry("strong", conviction="1.0"))
        weak = scored(entry("weak", conviction="0.2"))
        assert strong.score > weak.score

    def test_better_risk_reward_outranks_worse(self) -> None:
        good = scored(entry("good", stop="95", target="130"))  # 1:6
        poor = scored(entry("poor", stop="50", target="105"))  # 1:0.1
        assert good.score > poor.score

    def test_missing_stop_scores_zero_risk_reward_but_stays_eligible(self) -> None:
        result = scored(entry("nostop", stop=None))
        assert result.components["risk_reward"] == Decimal("0")
        assert result.score > Decimal("0")

    def test_thin_record_cannot_move_the_score(self) -> None:
        """A strategy with a handful of wins must not outrank on evidence alone."""
        lucky = StrategyRecord("lucky", trades=3, wins=3, net_pnl=Decimal("100"))
        with_record = scored(entry("lucky"), records={"lucky": lucky})
        without = scored(entry("lucky"))
        assert with_record.components["evidence"] == without.components["evidence"]

    def test_meaningful_record_does_move_the_score(self) -> None:
        proven = StrategyRecord(
            "proven",
            trades=MIN_TRADES_FOR_EVIDENCE,
            wins=MIN_TRADES_FOR_EVIDENCE,
            net_pnl=Decimal("1"),
        )
        assert scored(entry("proven"), records={"proven": proven}).components["evidence"] > Decimal(
            "0.5"
        )

    def test_regime_favours_reversion_in_a_range(self) -> None:
        reversion = scored(entry("rsi_reversion"), regime=MarketRegime.RANGE)
        trend = scored(entry("macd_trend"), regime=MarketRegime.RANGE)
        assert reversion.components["regime"] > trend.components["regime"]

    def test_regime_favours_trend_when_trending(self) -> None:
        trend = scored(entry("macd_trend"), regime=MarketRegime.BULL_TREND)
        reversion = scored(entry("rsi_reversion"), regime=MarketRegime.BULL_TREND)
        assert trend.components["regime"] > reversion.components["regime"]

    def test_unknown_regime_is_neutral_not_a_penalty(self) -> None:
        assert scored(entry("x"), regime=MarketRegime.UNKNOWN).components["regime"] == Decimal(
            "0.5"
        )

    def test_costs_penalise_a_target_barely_beyond_fees(self) -> None:
        thin = scored(entry("thin", target="100.1"), cost_rate=Decimal("0.002"))
        fat = scored(entry("fat", target="130"), cost_rate=Decimal("0.002"))
        assert fat.components["cost"] > thin.components["cost"]

    def test_symbol_already_held_is_penalised(self) -> None:
        held = scored(entry("s"), open_symbols=frozenset({SYMBOL}))
        free = scored(entry("s"))
        assert held.components["correlation"] < free.components["correlation"]

    def test_rank_is_best_first_and_deterministic(self) -> None:
        a = scored(entry("aaa", conviction="0.4"))
        b = scored(entry("bbb", conviction="0.9"))
        assert [item.strategy_id for item in rank([a, b])] == ["bbb", "aaa"]
        assert rank([a, b]) == rank([b, a])


class TestOrchestrator:
    def test_defaults_to_every_registered_strategy_but_the_benchmark(self) -> None:
        """Every registered strategy is a member, minus the benchmark and itself."""
        orchestrator = StrategyOrchestrator()
        ids = {member.strategy_id for member in orchestrator.members}
        registered = set(load_builtin_strategies().names())
        assert ids == registered - {"buy_and_hold", "orchestrator"}
        assert "orchestrator" not in ids  # no orchestrator inside the orchestrator

    def test_requires_at_least_one_member(self) -> None:
        with pytest.raises(ValidationError):
            StrategyOrchestrator(members=[])

    def test_every_member_is_evaluated_on_each_bar(self) -> None:
        """The orchestrator must poll all members, not stop at the first actionable one."""
        seen: list[str] = []

        class Recording(StubStrategy):
            def generate(self, context: StrategyContext) -> Signal:
                seen.append(self.strategy_id)
                return super().generate(context)

        members: Sequence[Strategy] = [
            Recording("one", entry("one")),
            Recording("two"),
            Recording("three", entry("three")),
        ]
        orchestrator = StrategyOrchestrator(members=members)
        orchestrator.evaluate(make_context(SYMBOL, [100] * 30))
        assert seen == ["one", "two", "three"]

    def test_selects_the_highest_scoring_candidate(self) -> None:
        weak = StubStrategy("weak", entry("weak", conviction="0.1", target="101"))
        strong = StubStrategy("strong", entry("strong", conviction="1.0", target="130"))
        orchestrator = StrategyOrchestrator(members=[weak, strong])
        signal = orchestrator.evaluate(make_context(SYMBOL, [100] * 30))
        assert signal.strategy_id == "strong"

    def test_returns_the_members_own_signal_so_attribution_survives(self) -> None:
        original = entry("keltner_trend")
        orchestrator = StrategyOrchestrator(members=[StubStrategy("keltner_trend", original)])
        signal = orchestrator.evaluate(make_context(SYMBOL, [100] * 30))
        assert signal.signal_id == original.signal_id
        assert signal.strategy_id == "keltner_trend"

    def test_holds_when_no_member_is_actionable(self) -> None:
        orchestrator = StrategyOrchestrator(members=[StubStrategy("a"), StubStrategy("b")])
        signal = orchestrator.evaluate(make_context(SYMBOL, [100] * 30))
        assert signal.direction is SignalDirection.HOLD
        assert not signal.is_actionable

    def test_holds_when_every_candidate_fails_the_economic_gates(self) -> None:
        """No trade is a valid outcome; frequency is never forced."""
        poor = StubStrategy("poor", entry("poor", conviction="0.01", stop=None, target="100.01"))
        orchestrator = StrategyOrchestrator(members=[poor])
        signal = orchestrator.evaluate(make_context(SYMBOL, [100] * 30))
        assert not signal.is_actionable
        assert "economic gates" in signal.reason

    def test_holds_when_nothing_clears_the_score_floor(self) -> None:
        """A candidate can be economically viable and still not be worth trading."""
        # Passes every gate — 1:6 reward:risk, reward far above costs — but scores poorly.
        viable = StubStrategy("viable", entry("viable", conviction="0.01", stop="95", target="130"))
        orchestrator = StrategyOrchestrator(members=[viable], params={"min_score": Decimal("0.99")})
        signal = orchestrator.evaluate(make_context(SYMBOL, [100] * 30))
        assert not signal.is_actionable
        assert "below floor" in signal.reason

    def test_multiple_simultaneous_opportunities_are_all_scored(self) -> None:
        members = [StubStrategy(f"s{i}", entry(f"s{i}", conviction="0.5")) for i in range(5)]
        orchestrator = StrategyOrchestrator(members=members)
        orchestrator.evaluate(make_context(SYMBOL, [100] * 30))
        decision = orchestrator.last_decision
        assert decision is not None
        assert decision.evaluated == 5
        assert len(decision.candidates) == 5

    def test_close_signal_without_a_position_is_not_executed(self) -> None:
        close = Signal(
            symbol=SYMBOL,
            direction=SignalDirection.CLOSE,
            timestamp=REFERENCE_TIME,
            strategy_id="closer",
        )
        orchestrator = StrategyOrchestrator(members=[StubStrategy("closer", close)])
        assert not orchestrator.evaluate(make_context(SYMBOL, [100] * 30)).is_actionable


class TestOpenPositionOwnership:
    def test_only_the_owning_strategy_manages_an_open_position(self) -> None:
        exit_signal = Signal(
            symbol=SYMBOL,
            direction=SignalDirection.CLOSE,
            timestamp=REFERENCE_TIME,
            strategy_id="owner",
            reason="owner exits",
        )
        owner = StubStrategy("owner", exit_signal)
        intruder = StubStrategy("intruder", entry("intruder", conviction="1.0"))
        orchestrator = StrategyOrchestrator(members=[owner, intruder])
        orchestrator.adopt(SYMBOL, "owner")

        signal = orchestrator.evaluate(make_context(SYMBOL, [100] * 30, position=open_long(SYMBOL)))
        assert signal.strategy_id == "owner"
        assert signal.direction is SignalDirection.CLOSE

    def test_a_position_with_no_recorded_owner_is_left_alone(self) -> None:
        intruder = StubStrategy("intruder", entry("intruder"))
        orchestrator = StrategyOrchestrator(members=[intruder])
        signal = orchestrator.evaluate(make_context(SYMBOL, [100] * 30, position=open_long(SYMBOL)))
        assert not signal.is_actionable
        assert "no recorded owner" in signal.reason

    def test_ownership_is_recorded_on_selection_and_released_on_exit(self) -> None:
        orchestrator = StrategyOrchestrator(members=[StubStrategy("owner", entry("owner"))])
        orchestrator.evaluate(make_context(SYMBOL, [100] * 30))
        assert orchestrator.owners[SYMBOL] == "owner"

        closer = Signal(
            symbol=SYMBOL,
            direction=SignalDirection.CLOSE,
            timestamp=REFERENCE_TIME,
            strategy_id="owner",
        )
        orchestrator = StrategyOrchestrator(members=[StubStrategy("owner", closer)])
        orchestrator.adopt(SYMBOL, "owner")
        orchestrator.evaluate(make_context(SYMBOL, [100] * 30, position=open_long(SYMBOL)))
        assert SYMBOL not in orchestrator.owners


class TestOwnershipFollowsTheVenue:
    """The venue decides what is open. Ownership must follow it, not the other way round.

    Ownership was released in exactly one place: the owning member returning ``CLOSE`` on a
    bar where the engine still saw a position. Every other way a position ends — a venue
    stop, a take-profit, an intrabar exit, a manual close, a liquidation — left the symbol
    owned forever.

    An owned symbol counts as an open position in the duplicate guard, so the effect was
    total: on 2026-08-14 the live engine declined 52 of 52 candidates citing correlation
    with open positions while the venue held **zero**. Nothing could ever be entered again.
    """

    def test_ownership_is_released_when_the_venue_no_longer_holds_the_position(self) -> None:
        orchestrator = StrategyOrchestrator(members=[StubStrategy("owner", entry("owner"))])
        orchestrator.adopt(SYMBOL, "owner")
        assert orchestrator.owners[SYMBOL] == "owner"

        released = orchestrator.sync_owners(())

        assert released == (SYMBOL,)
        assert orchestrator.owners == {}

    def test_a_symbol_the_venue_still_holds_keeps_its_owner(self) -> None:
        orchestrator = StrategyOrchestrator(members=[StubStrategy("owner", entry("owner"))])
        orchestrator.adopt(SYMBOL, "owner")

        released = orchestrator.sync_owners([SYMBOL])

        assert released == ()
        assert orchestrator.owners[SYMBOL] == "owner"

    def test_syncing_does_not_invent_ownership_for_an_unowned_venue_position(self) -> None:
        # An adopted position needs a strategy to manage it, and this function has no way
        # to know which. Claiming one would hand the position to an arbitrary member.
        orchestrator = StrategyOrchestrator(members=[StubStrategy("owner", entry("owner"))])

        orchestrator.sync_owners([SYMBOL])

        assert SYMBOL not in orchestrator.owners

    def test_a_released_symbol_can_be_entered_again(self) -> None:
        # The point of the fix: after release the duplicate guard must stop firing.
        orchestrator = StrategyOrchestrator(members=[StubStrategy("owner", entry("owner"))])
        orchestrator.adopt(SYMBOL, "owner")
        orchestrator.sync_owners(())

        signal = orchestrator.evaluate(make_context(SYMBOL, [100] * 30))

        assert signal.is_actionable
        assert orchestrator.owners[SYMBOL] == "owner"

    def test_a_closed_round_trip_releases_its_symbol(self) -> None:
        # The local signal, ahead of the next venue read: a completed round-trip is proof
        # the position is gone, so ownership should not survive until the next sync.
        orchestrator = StrategyOrchestrator(members=[StubStrategy("owner", entry("owner"))])
        orchestrator.adopt(SYMBOL, "owner")

        orchestrator.on_trade_closed(
            ClosedTrade(
                symbol=SYMBOL,
                side=PositionSide.LONG,
                quantity=Decimal("1"),
                entry_price=Decimal("100"),
                exit_price=Decimal("101"),
                entry_time=REFERENCE_TIME,
                exit_time=REFERENCE_TIME,
                gross_pnl=Decimal("1"),
                fees=Decimal("0"),
                strategy_id="owner",
            )
        )

        assert SYMBOL not in orchestrator.owners

    def test_record_trade_accumulates_per_strategy(self) -> None:
        orchestrator = StrategyOrchestrator(members=[StubStrategy("a")])
        for pnl in ("5", "-2", "3"):
            orchestrator.record_trade(
                ClosedTrade(
                    symbol=SYMBOL,
                    side=PositionSide.LONG,
                    quantity=Decimal("1"),
                    entry_price=Decimal("100"),
                    exit_price=Decimal("101"),
                    entry_time=REFERENCE_TIME,
                    exit_time=REFERENCE_TIME,
                    gross_pnl=Decimal(pnl),
                    fees=Decimal("0"),
                    strategy_id="a",
                )
            )
        record = orchestrator.records["a"]
        assert record.trades == 3
        assert record.wins == 2
