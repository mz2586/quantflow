"""The ``quantflow`` command-line interface.

Grouped by task: ``data`` for market data, ``backtest`` for research, ``trade`` for
running an engine, ``risk`` for the kill switch, ``db`` for migrations.

Anything that can move money or destroy data asks for confirmation unless ``--yes`` is
passed, and every command reports what it is about to do before doing it.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from quantflow import __version__
from quantflow.core.config import Settings, get_settings
from quantflow.core.errors import QuantFlowError
from quantflow.core.logging import configure_logging
from quantflow.domain.enums import Timeframe
from quantflow.domain.instruments import Instrument, Symbol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantflow.domain.market import Candle
    from quantflow.research import RankedEntry, ResearchOutcome

console = Console()

app = typer.Typer(
    name="quantflow",
    help="AI-powered algorithmic trading platform for Binance.",
    add_completion=False,
)
data_app = typer.Typer(help="Download and inspect market data.", no_args_is_help=True)
backtest_app = typer.Typer(help="Run and inspect backtests.", no_args_is_help=True)
trade_app = typer.Typer(help="Run a trading engine.", no_args_is_help=True)
risk_app = typer.Typer(help="Inspect and control risk state.", no_args_is_help=True)
research_app = typer.Typer(
    help="Rank strategies against fixed acceptance thresholds.", no_args_is_help=True
)

app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(trade_app, name="trade")
app.add_typer(risk_app, name="risk")
app.add_typer(research_app, name="research")


def _settings() -> Settings:
    settings = get_settings()
    configure_logging(settings, service="quantflow-cli")
    return settings


def _parse_symbol(raw: str) -> Symbol:
    parsed = Symbol.parse(raw)
    assert isinstance(parsed, Symbol)
    return parsed


def _parse_date(raw: str) -> datetime:
    """Parse a date or datetime, always producing a UTC-aware value."""
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise typer.BadParameter(f"{raw!r} is not an ISO date (expected YYYY-MM-DD)") from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _fail(exc: Exception) -> None:
    console.print(f"[bold red]error[/] {exc}")
    raise typer.Exit(code=1)


@app.callback(invoke_without_command=True)
def main(
    context: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Print the version and exit.")] = False,
) -> None:
    """QuantFlow CLI."""
    if version:
        console.print(f"quantflow {__version__}")
        raise typer.Exit
    if context.invoked_subcommand is None:
        console.print(context.get_help())
        raise typer.Exit


@app.command()
def config() -> None:
    """Show the effective configuration, with secrets redacted."""
    settings = _settings()
    table = Table(title="QuantFlow configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    rows: list[tuple[str, str]] = [
        ("environment", settings.env.value),
        ("trading mode", settings.trading.mode.value),
        ("live armed", "[bold red]YES[/]" if settings.is_live else "no"),
        ("database", settings.database.safe_dsn),
        ("redis", settings.redis.safe_url),
        (
            "exchange",
            f"{settings.exchange.name} "
            f"({'testnet' if settings.exchange.testnet else 'PRODUCTION'})",
        ),
        ("credentials", "configured" if settings.exchange.has_credentials else "absent"),
        ("base currency", settings.trading.base_currency),
        ("max position %", f"{settings.risk.max_position_pct:.1%}"),
        ("max daily loss %", f"{settings.risk.max_daily_loss_pct:.1%}"),
        ("max drawdown %", f"{settings.risk.max_drawdown_pct:.1%}"),
        ("stop loss required", "yes" if settings.risk.require_stop_loss else "[bold red]NO[/]"),
    ]
    for name, value in rows:
        table.add_row(name, value)
    console.print(table)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@data_app.command("download")
def download_data(
    symbol: Annotated[str, typer.Option(help="Pair, e.g. BTC/USDT")] = "BTC/USDT",
    timeframe: Annotated[str, typer.Option(help="Bar interval")] = "1h",
    start: Annotated[str, typer.Option(help="ISO start date")] = "2024-01-01",
    end: Annotated[str | None, typer.Option(help="ISO end date")] = None,
    resume: Annotated[bool, typer.Option(help="Continue from stored data")] = True,
) -> None:
    """Backfill historical candles into the database."""
    settings = _settings()
    parsed_symbol = _parse_symbol(symbol)
    parsed_timeframe = Timeframe.parse(timeframe)
    start_at = _parse_date(start)
    end_at = _parse_date(end) if end else None

    async def run() -> None:
        from quantflow.exchange.binance.rest import BinanceGateway
        from quantflow.marketdata.downloader import HistoricalDownloader, estimate_requests
        from quantflow.persistence.database import Database

        expected = estimate_requests(start_at, end_at or datetime.now(UTC), parsed_timeframe)
        console.print(
            f"Downloading [cyan]{parsed_symbol}[/] {parsed_timeframe.value} "
            f"from {start_at.date()} (~{expected} requests)"
        )

        gateway = BinanceGateway(settings.exchange)
        database = Database.from_settings(settings)
        try:
            await gateway.connect()
            downloader = HistoricalDownloader(gateway=gateway, database=database)
            result = await downloader.download(
                parsed_symbol,
                parsed_timeframe,
                start=start_at,
                end=end_at,
                resume=resume,
            )
            console.print(result.summary())
            if not result.is_complete and result.integrity:
                console.print(
                    f"[yellow]warning[/] {result.integrity.missing_bar_count} bars are "
                    "missing; run 'data verify' for detail"
                )
        finally:
            await gateway.aclose()
            await database.aclose()

    try:
        asyncio.run(run())
    except QuantFlowError as exc:
        _fail(exc)


@data_app.command("verify")
def verify_data(
    symbol: Annotated[str, typer.Option()] = "BTC/USDT",
    timeframe: Annotated[str, typer.Option()] = "1h",
) -> None:
    """Check a stored series for gaps and anomalies.

    Worth running before any backtest: a run across an undetected gap produces a
    plausible-looking equity curve that is simply wrong.
    """
    settings = _settings()
    parsed_symbol = _parse_symbol(symbol)
    parsed_timeframe = Timeframe.parse(timeframe)

    async def run() -> None:
        from quantflow.persistence.database import Database
        from quantflow.persistence.repositories import CandleRepository

        database = Database.from_settings(settings)
        try:
            async with database.read_session() as session:
                report = await CandleRepository(session).integrity_report(
                    parsed_symbol, parsed_timeframe
                )
        finally:
            await database.aclose()

        console.print(f"[cyan]{parsed_symbol}[/] {parsed_timeframe.value}")
        console.print(f"  bars      {report.candle_count:,}")
        if report.start and report.end:
            console.print(f"  range     {report.start.date()} .. {report.end.date()}")
        if report.is_clean:
            console.print("  [green]contiguous, no anomalies[/]")
            return
        console.print(
            f"  [yellow]gaps      {len(report.gaps)} ({report.missing_bar_count} bars)[/]"
        )
        for gap_start, gap_end in report.gaps[:10]:
            console.print(f"    {gap_start.isoformat()} .. {gap_end.isoformat()}")
        if report.duplicate_open_times:
            console.print(f"  [red]duplicates {len(report.duplicate_open_times)}[/]")

    try:
        asyncio.run(run())
    except QuantFlowError as exc:
        _fail(exc)


@data_app.command("list")
def list_data() -> None:
    """List every stored series."""
    settings = _settings()

    async def run() -> None:
        from quantflow.persistence.database import Database
        from quantflow.persistence.repositories import CandleRepository

        database = Database.from_settings(settings)
        try:
            async with database.read_session() as session:
                series = await CandleRepository(session).available_series()
        finally:
            await database.aclose()

        if not series:
            console.print("[yellow]no stored market data[/]")
            return
        table = Table(show_header=True)
        table.add_column("Symbol", style="cyan")
        table.add_column("Timeframe")
        table.add_column("Bars", justify="right")
        for symbol, timeframe, bars in series:
            table.add_row(symbol.slashed, timeframe.value, f"{bars:,}")
        console.print(table)

    try:
        asyncio.run(run())
    except QuantFlowError as exc:
        _fail(exc)


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #
@backtest_app.command("run")
def run_backtest(
    strategy: Annotated[str, typer.Option(help="Strategy id")] = "ema_cross",
    symbol: Annotated[str, typer.Option()] = "BTC/USDT",
    timeframe: Annotated[str, typer.Option()] = "1h",
    start: Annotated[str, typer.Option()] = "2024-01-01",
    end: Annotated[str | None, typer.Option()] = None,
    equity: Annotated[float, typer.Option(help="Starting equity")] = 10000.0,
    report: Annotated[bool, typer.Option(help="Write an HTML report")] = False,
) -> None:
    """Run a backtest over stored candles."""
    settings = _settings()
    parsed_symbol = _parse_symbol(symbol)
    parsed_timeframe = Timeframe.parse(timeframe)
    start_at = _parse_date(start)
    end_at = _parse_date(end) if end else datetime.now(UTC)

    async def run() -> None:
        from quantflow.backtest.engine import BacktestConfig, BacktestEngine, rejection_reasons
        from quantflow.backtest.metrics import is_statistically_thin
        from quantflow.backtest.report import write_report
        from quantflow.domain.instruments import Instrument
        from quantflow.persistence.database import Database
        from quantflow.persistence.repositories import CandleRepository, InstrumentRepository
        from quantflow.strategy.registry import load_builtin_strategies

        registry = load_builtin_strategies()
        engine_strategy = registry.create(strategy)

        database = Database.from_settings(settings)
        try:
            async with database.read_session() as session:
                candles = await CandleRepository(session).fetch(
                    parsed_symbol, parsed_timeframe, start=start_at, end=end_at
                )
                instrument = await InstrumentRepository(session).get(parsed_symbol)
        finally:
            await database.aclose()

        if not candles:
            console.print(
                f"[red]no stored candles[/] for {parsed_symbol} {parsed_timeframe.value}; "
                "run 'quantflow data download' first"
            )
            raise typer.Exit(code=1)

        console.print(
            f"Backtesting [cyan]{strategy}[/] on {parsed_symbol} "
            f"{parsed_timeframe.value}, {len(candles):,} bars"
        )
        config = BacktestConfig(
            symbols=(parsed_symbol,),
            timeframe=parsed_timeframe,
            starting_equity=Decimal(str(equity)),
            risk=settings.risk,
        )
        result = await BacktestEngine(
            engine_strategy,
            config,
            {parsed_symbol: instrument or Instrument(symbol=parsed_symbol)},
        ).run({parsed_symbol: candles})

        if not result.succeeded:
            console.print(f"[red]backtest failed[/] {result.error}")
            raise typer.Exit(code=1)

        metrics = result.metrics()
        console.print()
        for line in metrics.summary_lines():
            console.print(f"  {line}")
        console.print()

        if is_statistically_thin(metrics):
            console.print(
                f"[yellow]caution[/] only {metrics.trade_count} trades — below ~30 these "
                "metrics are dominated by chance"
            )
        if result.rejected_signals:
            console.print(f"[yellow]{len(result.rejected_signals)} signals refused by risk:[/]")
            for reason, count in list(rejection_reasons(result).items())[:5]:
                console.print(f"    {count:>4} x {reason}")

        if report:
            path = write_report(result, settings.storage.report_dir)
            console.print(f"\nReport: [cyan]{path}[/]")

    try:
        asyncio.run(run())
    except QuantFlowError as exc:
        _fail(exc)


@backtest_app.command("strategies")
def list_strategies() -> None:
    """List every registered strategy."""
    from quantflow.strategy.registry import load_builtin_strategies

    registry = load_builtin_strategies()
    table = Table(show_header=True)
    table.add_column("Strategy", style="cyan")
    table.add_column("Warm-up", justify="right")
    table.add_column("Description")
    for entry in registry.describe_all():
        table.add_row(entry["strategy_id"], str(entry["warmup_bars"]), entry["description"])
    console.print(table)


# --------------------------------------------------------------------------- #
# trade
# --------------------------------------------------------------------------- #
@trade_app.command("paper")
def trade_paper(
    strategy: Annotated[str, typer.Option()] = "ema_cross",
    symbol: Annotated[str, typer.Option()] = "BTC/USDT",
    timeframe: Annotated[str, typer.Option()] = "1h",
    equity: Annotated[float, typer.Option()] = 10000.0,
) -> None:
    """Start a paper-trading session against live market data."""
    settings = _settings()
    parsed_symbol = _parse_symbol(symbol)
    parsed_timeframe = Timeframe.parse(timeframe)

    async def run() -> None:
        from quantflow.exchange.binance.rest import BinanceGateway
        from quantflow.exchange.binance.ws import BinanceStream
        from quantflow.paper.engine import PaperConfig, PaperTradingEngine, candle_feed
        from quantflow.persistence.database import Database
        from quantflow.strategy.registry import load_builtin_strategies

        registry = load_builtin_strategies()
        engine_strategy = registry.create(strategy)

        gateway = BinanceGateway(settings.exchange)
        database = Database.from_settings(settings)
        try:
            await gateway.connect()
            instrument = await gateway.get_instrument(parsed_symbol)

            engine = PaperTradingEngine(
                engine_strategy,
                PaperConfig(
                    symbols=(parsed_symbol,),
                    timeframe=parsed_timeframe,
                    starting_equity=Decimal(str(equity)),
                    risk=settings.risk,
                ),
                instruments={parsed_symbol: instrument},
                database=database,
            )
            await engine.prepare(gateway)
            console.print(
                f"[green]Paper trading[/] {strategy} on {parsed_symbol} "
                f"{parsed_timeframe.value}. Press Ctrl-C to stop."
            )

            stream = BinanceStream(settings.exchange)
            feed = candle_feed(stream, [parsed_symbol], parsed_timeframe)
            await engine.run(feed)
        except KeyboardInterrupt:
            console.print("\n[yellow]stopping[/]")
        finally:
            await gateway.aclose()
            await database.aclose()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        console.print("\nstopped")
    except QuantFlowError as exc:
        _fail(exc)


# --------------------------------------------------------------------------- #
# risk
# --------------------------------------------------------------------------- #
@risk_app.command("status")
def risk_status() -> None:
    """Show the kill switch and configured limits."""
    settings = _settings()

    async def run() -> None:
        from quantflow.persistence.database import Database
        from quantflow.risk.killswitch import KillSwitch

        database = Database.from_settings(settings)
        try:
            switch = KillSwitch(database)
            state = await switch.load()
        finally:
            await database.aclose()

        if state.engaged:
            console.print("[bold red]KILL SWITCH ENGAGED[/]")
            console.print(f"  reason  {state.reason}")
            console.print(f"  since   {state.engaged_at.isoformat() if state.engaged_at else '?'}")
            console.print(f"  by      {state.engaged_by}")
        else:
            console.print("[green]kill switch clear[/]")

    try:
        asyncio.run(run())
    except QuantFlowError as exc:
        _fail(exc)


@risk_app.command("halt")
def risk_halt(
    reason: Annotated[str, typer.Argument(help="Why trading is being halted")],
    actor: Annotated[str, typer.Option()] = "operator",
) -> None:
    """Engage the kill switch, halting all new entries."""
    settings = _settings()

    async def run() -> None:
        from quantflow.persistence.database import Database
        from quantflow.risk.killswitch import KillSwitch

        database = Database.from_settings(settings)
        try:
            state = await KillSwitch(database).engage(reason, actor=actor)
        finally:
            await database.aclose()
        console.print(f"[bold red]kill switch engaged[/] — {state.reason}")

    try:
        asyncio.run(run())
    except QuantFlowError as exc:
        _fail(exc)


@risk_app.command("resume")
def risk_resume(
    actor: Annotated[str, typer.Option()] = "operator",
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation")] = False,
) -> None:
    """Clear the kill switch and allow trading to resume."""
    settings = _settings()

    if not yes:
        confirmed = typer.confirm(
            "Clearing the kill switch allows new positions to be opened. Continue?"
        )
        if not confirmed:
            console.print("aborted")
            raise typer.Exit(code=1)

    async def run() -> None:
        from quantflow.persistence.database import Database
        from quantflow.risk.killswitch import KillSwitch

        database = Database.from_settings(settings)
        try:
            await KillSwitch(database).clear(actor=actor)
        finally:
            await database.aclose()
        console.print("[green]kill switch cleared[/]")

    try:
        asyncio.run(run())
    except QuantFlowError as exc:
        _fail(exc)


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    reload: Annotated[bool, typer.Option(help="Reload on source changes")] = False,
) -> None:
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "quantflow.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
        access_log=False,
    )


def report_path_for(directory: Path, run_id: str) -> Path:
    """Conventional report path for a run id."""
    return directory / f"backtest-{run_id[:8]}.html"


def summarise(values: dict[str, Any]) -> str:
    """Render a mapping as a compact single line, for CLI output."""
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))


# --------------------------------------------------------------------------- #
# research
# --------------------------------------------------------------------------- #
async def _load_research_data(
    settings: Settings,
    symbols: Sequence[Symbol],
    timeframe: Timeframe,
    start_at: datetime,
    end_at: datetime,
) -> tuple[dict[Symbol, Sequence[Candle]], dict[Symbol, Instrument]]:
    """Fetch candles and instruments for every symbol in the sweep."""
    from quantflow.persistence.database import Database
    from quantflow.persistence.repositories import CandleRepository, InstrumentRepository

    data: dict[Symbol, Sequence[Candle]] = {}
    instruments: dict[Symbol, Instrument] = {}
    database = Database.from_settings(settings)
    try:
        async with database.read_session() as session:
            candle_repo = CandleRepository(session)
            instrument_repo = InstrumentRepository(session)
            for symbol in symbols:
                data[symbol] = await candle_repo.fetch(
                    symbol, timeframe, start=start_at, end=end_at
                )
                instruments[symbol] = await instrument_repo.get(symbol) or Instrument(symbol=symbol)
    finally:
        await database.aclose()
    return data, instruments


def _render_leaderboard(board: Sequence[RankedEntry]) -> Table:
    """The leaderboard as a terminal table."""
    table = Table(show_header=True, title="Leaderboard")
    table.add_column("#", justify="right")
    table.add_column("Strategy", style="cyan")
    table.add_column("Verdict")
    for column in ("Net return", "PF", "Sharpe", "Max DD", "Win rate", "Trades", "vs hold"):
        table.add_column(column, justify="right")

    for entry in board:
        item = entry.entry
        verdict = (
            "benchmark"
            if item.is_benchmark
            else ("[green]accepted[/]" if item.accepted else "[red]rejected[/]")
        )
        table.add_row(
            str(entry.position),
            item.strategy_id,
            verdict,
            f"{item.net_return:.2%}",
            f"{item.profit_factor:.2f}",
            f"{item.sharpe_ratio:.2f}",
            f"{item.max_drawdown:.2%}",
            f"{item.win_rate:.2%}",
            str(item.trade_count),
            "-" if item.excess_return is None else f"{item.excess_return:+.2%}",
        )
    return table


def _write_research_reports(outcome: ResearchOutcome, directory: Path, stem: str) -> list[Path]:
    """Write the Markdown, HTML and JSON reports.

    Synchronous, and called outside the event loop on purpose: blocking file IO inside a
    coroutine stalls the loop for no benefit at all.
    """
    from quantflow.research import build_html, build_json, build_markdown

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for suffix, builder in (("md", build_markdown), ("html", build_html), ("json", build_json)):
        path = directory / f"{stem}.{suffix}"
        path.write_text(builder(outcome), encoding="utf-8")
        written.append(path)
    return written


@research_app.command("run")
def research_run(
    symbols: Annotated[
        str, typer.Option(help="Comma-separated pairs to test every strategy on")
    ] = "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT",
    timeframe: Annotated[str, typer.Option()] = "1h",
    start: Annotated[str, typer.Option(help="ISO start date")] = "2021-01-01",
    end: Annotated[str | None, typer.Option(help="ISO end date")] = None,
    equity: Annotated[float, typer.Option(help="Starting equity per run")] = 10000.0,
    costs: Annotated[
        str, typer.Option(help="Cost model: realistic, pessimistic or zero_cost")
    ] = "realistic",
    strategies: Annotated[
        str | None, typer.Option(help="Comma-separated strategy ids; default is all")
    ] = None,
    report: Annotated[bool, typer.Option(help="Write Markdown, HTML and JSON reports")] = True,
) -> None:
    """Backtest every strategy over every symbol and rank the survivors."""
    from decimal import Decimal

    from quantflow.research import ResearchConfig, ResearchRunner, build_cost_model, leaderboard

    settings = _settings()
    parsed_symbols = tuple(
        _parse_symbol(item.strip()) for item in symbols.split(",") if item.strip()
    )
    parsed_timeframe = Timeframe.parse(timeframe)
    start_at = _parse_date(start)
    end_at = _parse_date(end) if end else datetime.now(UTC)
    requested = (
        tuple(item.strip() for item in strategies.split(",") if item.strip()) if strategies else ()
    )
    cost_model = build_cost_model(costs)

    async def load() -> tuple[dict[Symbol, Sequence[Candle]], dict[Symbol, Instrument]]:
        return await _load_research_data(
            settings, parsed_symbols, parsed_timeframe, start_at, end_at
        )

    try:
        data, instruments = asyncio.run(load())
    except QuantFlowError as exc:
        _fail(exc)

    empty = [str(symbol) for symbol, candles in data.items() if not candles]
    if empty:
        console.print(
            f"[red]No stored candles for {', '.join(empty)}.[/] "
            "Run 'quantflow data download' first."
        )
        raise typer.Exit(code=1)

    config = ResearchConfig(
        symbols=parsed_symbols,
        timeframe=parsed_timeframe,
        starting_equity=Decimal(str(equity)),
        costs=cost_model,
        strategy_ids=requested,
    )
    console.print(
        f"Testing over {len(parsed_symbols)} symbols, "
        f"{sum(len(candles) for candles in data.values()):,} bars, "
        f"costs=[cyan]{cost_model.name}[/]"
    )

    try:
        outcome = asyncio.run(ResearchRunner(config).run(data, instruments))
    except QuantFlowError as exc:
        _fail(exc)

    board = leaderboard(outcome)
    console.print(_render_leaderboard(board))

    passed = [entry for entry in board if entry.entry.accepted and not entry.entry.is_benchmark]
    if passed:
        names = ", ".join(entry.entry.strategy_id for entry in passed)
        console.print(f"\n[green]{len(passed)} strategy(ies) passed every threshold:[/] {names}")
    else:
        console.print(
            "\n[yellow]No strategy passed every threshold on every symbol.[/] "
            "See the report for per-strategy reasons."
        )
    if outcome.failures:
        console.print(f"[red]{len(outcome.failures)} run(s) failed.[/]")

    if report:
        stem = f"research-{parsed_timeframe.value}-{cost_model.name}"
        for path in _write_research_reports(outcome, Path(settings.storage.report_dir), stem):
            console.print(f"Report: [cyan]{path}[/]")


@research_app.command("thresholds")
def research_thresholds() -> None:
    """Show the acceptance thresholds a strategy must clear."""
    from quantflow.research import DEFAULT_THRESHOLDS

    table = Table(show_header=True, title="Acceptance thresholds")
    table.add_column("Criterion", style="cyan")
    table.add_column("Requirement", justify="right")
    for name, requirement in DEFAULT_THRESHOLDS.describe().items():
        table.add_row(name, requirement)
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
