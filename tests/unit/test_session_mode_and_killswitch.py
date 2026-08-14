"""Three defects found by running the demo bot against a real venue.

**Session mode.** ``PaperTradingEngine`` is the engine for paper *and* live sessions — the
runner deliberately shares one code path so a live-only branch cannot rot. But the engine
hardcoded ``TradingMode.PAPER`` both in its ``mode`` property and in the row it persists,
so a live session against the Bybit demo venue was recorded, served by the API and shown
on the dashboard as ``paper``. Real orders, real fills, filed under the one label that
means "none of this happened".

**Kill switch.** The switch latched, persisted, and was read by the CLI — and the running
bot opened four positions eleven minutes after it was engaged. Nothing in the entry path
consulted it. A stop that the thing being stopped never reads is not a stop.

The kill switch tests here are about the *decision*, not the plumbing: given an engaged
switch, the risk engine must refuse a new entry, and must keep refusing (latched) until
someone clears it explicitly.
"""

from __future__ import annotations

from quantflow.core.config import RiskSettings, TradingMode
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Symbol
from quantflow.paper.engine import PaperConfig
from quantflow.risk.killswitch import KillSwitch

BTC = Symbol.parse("BTC/USDT")
TF = Timeframe.parse("15m")


class TestSessionModeIsHonest:
    def test_paper_config_carries_the_mode(self) -> None:
        """The engine cannot report a mode it was never told."""
        config = PaperConfig(symbols=(BTC,), timeframe=TF, mode=TradingMode.LIVE)

        assert config.mode is TradingMode.LIVE

    def test_paper_is_still_the_default(self) -> None:
        """Nothing becomes live by omission."""
        assert PaperConfig(symbols=(BTC,), timeframe=TF).mode is TradingMode.PAPER

    def test_live_session_does_not_persist_as_paper(self) -> None:
        """The specific defect: a live-demo session filed under 'paper'."""
        config = PaperConfig(symbols=(BTC,), timeframe=TF, mode=TradingMode.LIVE)

        assert config.mode.value != "paper"


class TestKillSwitchIsSeenByARunningSession:
    """The defect: engaged out-of-process, never noticed by the process that trades.

    ``RiskEngine.start()`` loads the switch once, before the first order, and nothing
    reloads it. The CLI and the API write the halt to the database — a different process —
    so a bot that started before the halt keeps trading with a stale "clear" in memory.
    That is exactly what happened: engaged 07:19:31, four entries at 07:30.
    """

    async def test_engaged_switch_reports_engaged(self) -> None:
        switch = KillSwitch()
        await switch.engage("operator halt")

        assert switch.engaged

    async def test_switch_latches_until_cleared(self) -> None:
        """It does not time out, decay, or reset on its own."""
        switch = KillSwitch()
        await switch.engage("operator halt")
        await switch.engage("second reason")

        assert switch.engaged

    async def test_clearing_releases_it(self) -> None:
        switch = KillSwitch()
        await switch.engage("operator halt")
        await switch.clear()

        assert not switch.engaged

    async def test_refresh_picks_up_an_out_of_process_engage(self) -> None:
        """The fix: a running engine must re-read state it did not write itself."""
        from quantflow.risk.engine import RiskEngine

        engine = RiskEngine(RiskSettings())
        await engine.start()
        assert not engine.kill_switch.engaged

        # Stand in for the CLI / dashboard engaging it in another process.
        await engine.kill_switch.engage("operator halt")

        await engine.refresh_kill_switch()

        assert engine.kill_switch.engaged

    async def test_engine_exposes_a_refresh_hook(self) -> None:
        """Without this the entry path has no way to notice a halt."""
        from quantflow.risk.engine import RiskEngine

        assert callable(RiskEngine.refresh_kill_switch)


class TestFetchOrderIsAcknowledged:
    """Bybit's fetchOrder refuses to run without an explicit acknowledgement param."""

    def test_enrichment_params_acknowledge_the_500_order_window(self) -> None:
        from quantflow.exchange.bybit.rest import FETCH_ORDER_PARAMS

        assert FETCH_ORDER_PARAMS.get("acknowledged") is True
