"""The orchestrator must refuse a strategy whose inputs are meaningless on the instrument.

``tests/unit/test_universe_assets.py`` pins the gating *rule*. This pins the **wiring**:
that the rule is consulted on the live decision path, that a refused member is refused
before it is evaluated rather than after, and that the refusal is visible in the decision
rather than silently dropping a strategy from the roster.

The ordering matters more than it looks. Evaluating a volume strategy on an equity
perpetual and then discarding its signal would still let it count toward the confluence
requirement in the selection layer — the orchestrator would refuse to *act* on it while
still treating it as corroboration, which is precisely the failure the family taxonomy
exists to prevent.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantflow.domain.enums import SignalDirection
from quantflow.domain.instruments import Symbol
from quantflow.domain.signals import Signal
from quantflow.orchestrator import StrategyOrchestrator
from quantflow.strategy.base import Strategy, StrategyContext, StrategyParams
from quantflow.universe.assets import ASSET_CLASS_METADATA_KEY, AssetClass
from tests.conftest import REFERENCE_TIME
from tests.unit.test_strategies import make_context

GOLD = Symbol.parse("XAU/USDT")
BITCOIN = Symbol.parse("BTC/USDT")


class Recording(Strategy):
    """A member that records whether it was polled, and emits a long entry when it was."""

    params_model = StrategyParams

    def __init__(self, strategy_id: str, seen: list[str]) -> None:
        self.strategy_id = strategy_id  # type: ignore[misc]
        super().__init__(None)
        self._seen = seen

    @property
    def warmup_bars(self) -> int:
        return 1

    def generate(self, context: StrategyContext) -> Signal:
        self._seen.append(self.strategy_id)
        return Signal(
            symbol=context.symbol,
            direction=SignalDirection.LONG,
            timestamp=REFERENCE_TIME,
            strategy_id=self.strategy_id,
            conviction=Decimal("1.0"),
            reference_price=Decimal("100"),
            stop_loss_price=Decimal("98"),
            take_profit_price=Decimal("106"),
            reason="recorded",
        )


def context_for(symbol: Symbol, asset_class: AssetClass | None) -> StrategyContext:
    """A decision context carrying the asset class the engine would have resolved."""
    base = make_context(symbol, [100] * 40)
    metadata = {} if asset_class is None else {ASSET_CLASS_METADATA_KEY: asset_class}
    return StrategyContext(
        symbol=base.symbol,
        timeframe=base.timeframe,
        history=base.history,
        now=base.now,
        portfolio=base.portfolio,
        position=base.position,
        regime=base.regime,
        metadata=metadata,
    )


def run(
    members: list[str], symbol: Symbol, asset_class: AssetClass | None
) -> tuple[list[str], object]:
    """Evaluate an orchestrator over the given members, returning who was polled."""
    seen: list[str] = []
    orchestrator = StrategyOrchestrator(members=[Recording(name, seen) for name in members])
    orchestrator.evaluate(context_for(symbol, asset_class))
    return seen, orchestrator.last_decision


@pytest.mark.parametrize(
    "asset_class",
    [AssetClass.METAL, AssetClass.ENERGY, AssetClass.EQUITY, AssetClass.INDEX],
)
def test_a_volume_strategy_is_never_polled_on_a_non_crypto_market(
    asset_class: AssetClass,
) -> None:
    """Refused *before* evaluation, so it cannot become confluence for something else."""
    seen, _ = run(["obv_trend", "ema_cross"], GOLD, asset_class)
    assert "obv_trend" not in seen, "a refused member must not be evaluated at all"
    assert "ema_cross" in seen, "an admitted member must still run"


def test_the_same_volume_strategy_is_polled_on_crypto() -> None:
    """The gate is about the instrument, not about the strategy being unwelcome."""
    seen, _ = run(["obv_trend", "ema_cross"], BITCOIN, AssetClass.CRYPTO)
    assert seen == ["obv_trend", "ema_cross"]


def test_the_refusal_is_recorded_on_the_decision() -> None:
    """A member that vanished without explanation is indistinguishable from a broken one."""
    _, decision = run(["obv_trend", "ema_cross"], GOLD, AssetClass.METAL)
    gated = dict(getattr(decision, "gated", []))
    assert "obv_trend" in gated
    assert "volume" in gated["obv_trend"]
    assert "metal" in gated["obv_trend"]


def test_swing_structure_is_refused_on_a_commodity() -> None:
    seen, _ = run(["swing_structure", "macd_trend"], GOLD, AssetClass.ENERGY)
    assert seen == ["macd_trend"]


@pytest.mark.parametrize(
    "strategy_id",
    [
        "ema_cross",
        "macd_trend",
        "momentum_roc",
        "donchian_breakout",
        "rsi_reversion",
        "volatility_regime",
    ],
)
def test_the_admitted_families_all_run_on_a_metal(strategy_id: str) -> None:
    """Trend, momentum, breakout, reversion and volatility are all admitted on metals."""
    seen, _ = run([strategy_id], GOLD, AssetClass.METAL)
    assert seen == [strategy_id]


def test_a_context_without_an_asset_class_gates_nothing() -> None:
    """Backwards compatibility: an engine that never sets the metadata loses no strategy.

    Every existing test and every engine that predates the classification builds a context
    with empty metadata. Those must keep evaluating the full roster, or this change would
    silently narrow what the backtester runs.
    """
    seen, _ = run(["obv_trend", "swing_structure", "ema_cross"], GOLD, None)
    assert seen == ["obv_trend", "swing_structure", "ema_cross"]


def test_a_strategys_own_declaration_is_honoured_by_the_orchestrator() -> None:
    """``supported_asset_classes`` on the class is read at the decision point."""
    seen: list[str] = []

    class CryptoOnly(Recording):
        supported_asset_classes = frozenset({"crypto", "meme"})

    orchestrator = StrategyOrchestrator(
        members=[CryptoOnly("ema_cross", seen), Recording("macd_trend", seen)]
    )
    orchestrator.evaluate(context_for(GOLD, AssetClass.METAL))
    assert seen == ["macd_trend"], "the declared restriction must refuse the metal"

    seen.clear()
    orchestrator.evaluate(context_for(BITCOIN, AssetClass.CRYPTO))
    assert seen == ["ema_cross", "macd_trend"], "and must admit the class it declared"


def test_the_default_declaration_is_permissive() -> None:
    """A strategy that says nothing is not restricted by its own declaration."""
    assert Strategy.supported_asset_classes == frozenset()
    from quantflow.strategy.registry import load_builtin_strategies

    registry = load_builtin_strategies()
    for name in registry.names():
        strategy_class = registry.get(name)
        assert strategy_class.supported_asset_classes == frozenset(), (
            f"{name} declares a restriction; the library is expected to stay silent and be "
            "gated by family instead"
        )
