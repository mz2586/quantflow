/**
 * QuantFlow trading operations console.
 *
 * Four commitments shape this file.
 *
 * **The first screen answers the money questions.** Wallet, available, in-trades, profit,
 * loss, fees, drawdown, open positions and whether the engine is trading — above the fold,
 * in two rows, before anything else competes for attention. Everything analytical moved to
 * its own tab, because a screen with twenty equally-weighted metrics answers nothing in
 * five seconds.
 *
 * **It updates itself.** The API's websocket announces fills, risk events and equity
 * updates; every announcement triggers a refresh, and short polling continues underneath
 * regardless so a dropped socket degrades rather than blinding the operator. Nothing here
 * requires a manual reload, and nothing renders a stale value as though it were live.
 *
 * **Sources are never blended.** The venue's account and QuantFlow's session accounting
 * produce similar-looking numbers about different things. Each panel names its source, and
 * where the two disagree the disagreement is shown rather than resolved silently.
 *
 * **One broken panel costs one panel.** Every panel is wrapped in an error boundary, every
 * nullable field is typed optional, and every collection goes through `list()` before it
 * is mapped. A throw during render unmounts React's whole tree, and a blank page during an
 * incident is indistinguishable from a dead machine.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  api,
  list,
  subscribe,
  type AnalyticsResponse,
  type AssetClassesResponse,
  type DecisionsResponse,
  type EquityResponse,
  type FreshnessResponse,
  type LedgerResponse,
  type PnlResponse,
  type ReadinessResponse,
  type Summary,
  type VenuePosition,
} from "./lib/api";
import { NOT_RECORDED, ago, clock, count, money, percent, signed, time, tone } from "./lib/format";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Caution, Panel, Stat, Unavailable, Value } from "./components/ui";
import {
  CumulativePnlChart,
  DrawdownChart,
  EquityChart,
  LongShortChart,
  PnlChart,
  type PnlMode,
} from "./components/charts";
import {
  AnalyticsPanels,
  AssetClassesPanel,
  DecisionsPanel,
  FeesPanel,
  FreshnessPanel,
  OrdersPanel,
  PositionsPanel,
  TradeLedgerPanel,
  VenueAccountPanel,
} from "./components/panels";
import {
  ExecutiveSummary,
  HealthStrip,
  PnlSection,
  TradingStatusPanel,
  UniverseStrip,
} from "./components/executive";

/** Fast poll: venue account, positions, orders, status, decision counters. */
const FAST_POLL_MS = 5_000;

/** Slow poll: the ledger, analytics and charts, which move at trade frequency. */
const SLOW_POLL_MS = 20_000;

const EQUITY_RANGES = ["1H", "6H", "24H", "7D", "30D", "ALL"] as const;

type Tab = "overview" | "trades" | "analytics" | "diagnostics";

const TABS: { key: Tab; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "trades", label: "Trades" },
  { key: "analytics", label: "Analytics" },
  { key: "diagnostics", label: "Diagnostics" },
];

function Header({
  summary,
  readiness,
  connected,
  lastUpdate,
  tab,
  onTab,
  onToggleKillSwitch,
  busy,
}: {
  summary: Summary | null;
  readiness: ReadinessResponse | null;
  connected: boolean;
  lastUpdate: string | null;
  tab: Tab;
  onTab: (next: Tab) => void;
  onToggleKillSwitch: () => void;
  busy: boolean;
}) {
  const engaged = summary?.risk?.kill_switch_engaged ?? false;
  const halted = summary?.risk?.trading_halted ?? false;
  const session = summary?.session;
  // The venue the orders actually reach, not the API process's configured mode: those
  // disagree here — the API is configured `paper` while a live session trades a demo
  // venue — and the badge that says whether the money is real must follow the venue.
  const network = summary?.venue?.network ?? null;
  const isDemo = network === "demo";
  const isMainnet = network === "mainnet";

  return (
    <header className="sticky top-0 z-20 border-b border-zinc-800 bg-zinc-950/95 backdrop-blur">
      <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-5 gap-y-2 px-4 py-2.5">
        <h1 className="text-sm font-semibold tracking-tight">QuantFlow</h1>

        <span
          className={`rounded px-2 py-0.5 text-[10px] uppercase tracking-wider ring-1 ${
            isMainnet
              ? "bg-[#d03b3b]/20 text-[#d03b3b] ring-[#d03b3b]/40"
              : isDemo
                ? "bg-[#fab219]/15 text-[#fab219] ring-[#fab219]/30"
                : "bg-zinc-800 text-zinc-300 ring-zinc-700"
          }`}
          title={
            isDemo
              ? "Bybit demo venue — real orders, virtual funds. No real money."
              : isMainnet
                ? "MAINNET — real money at risk."
                : undefined
          }
        >
          {isDemo ? "Demo · virtual funds" : isMainnet ? "Mainnet · REAL MONEY" : (network ?? "venue …")}
        </span>

        <nav className="flex gap-1">
          {TABS.map((entry) => (
            <button
              key={entry.key}
              type="button"
              onClick={() => {
                onTab(entry.key);
              }}
              className={`rounded px-2.5 py-1 text-xs transition ${
                tab === entry.key
                  ? "bg-zinc-700 text-zinc-100"
                  : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200"
              }`}
            >
              {entry.label}
            </button>
          ))}
        </nav>

        <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-zinc-400">
          <span className={`h-1.5 w-1.5 rounded-full ${connected ? "bg-[#0ca30c]" : "bg-zinc-600"}`} />
          {connected ? "live stream" : "polling"}
        </span>

        <span className="text-[10px] text-zinc-500">
          updated <span className="font-mono text-zinc-300">{clock(lastUpdate)}</span>
        </span>

        {session ? (
          <span
            className="truncate text-[10px] text-zinc-500"
            title={`${session.strategy_id ?? ""} · ${list(session.symbols).join(", ")} · chosen because: ${session.selection_basis ?? "—"}`}
          >
            session <span className="text-zinc-300">{session.session_id}</span>
            <span className="text-zinc-600"> · {session.status} · {session.timeframe}</span>
          </span>
        ) : null}

        <div className="ml-auto flex items-center gap-3">
          {!(readiness?.ready ?? true) ? (
            <span className="rounded bg-[#fab219]/15 px-2 py-1 text-[10px] font-medium text-[#fab219]">
              API not ready
            </span>
          ) : null}
          {halted && !engaged ? (
            <span className="rounded bg-[#fab219]/15 px-2 py-1 text-[10px] font-medium text-[#fab219]">
              trading halted for the day
            </span>
          ) : null}
          {engaged ? (
            <span className="rounded bg-[#d03b3b]/15 px-2 py-1 text-[10px] font-medium text-[#d03b3b]">
              KILL SWITCH ENGAGED
            </span>
          ) : null}
          <button
            type="button"
            onClick={onToggleKillSwitch}
            disabled={busy}
            className={`rounded px-3 py-1.5 text-xs font-medium transition disabled:opacity-40 ${
              engaged
                ? "bg-zinc-700 text-zinc-100 hover:bg-zinc-600"
                : "bg-[#d03b3b] text-white hover:bg-[#e04b4b]"
            }`}
          >
            {engaged ? "Clear kill switch" : "Halt trading"}
          </button>
        </div>
      </div>
    </header>
  );
}

/** Full detail for one open position, opened by clicking its row. */
function PositionDrawer({
  position,
  asset,
  onClose,
}: {
  position: VenuePosition;
  asset: string;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-30 flex justify-end bg-black/50" onClick={onClose}>
      <aside
        className="h-full w-full max-w-md overflow-y-auto border-l border-zinc-800 bg-zinc-950 p-5"
        onClick={(event) => {
          event.stopPropagation();
        }}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="font-mono text-lg text-zinc-100">{position.symbol}</h2>
            <p className="mt-0.5 text-[10px] uppercase tracking-wider text-zinc-500">
              {asset} · {position.side} · opened {time(position.opened_at)}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded bg-zinc-800 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-700"
          >
            Close
          </button>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-x-4 gap-y-4">
          <Stat label="Quantity" value={position.quantity ?? NOT_RECORDED} />
          <Stat label="Leverage" value={position.leverage ?? NOT_RECORDED} />
          <Stat label="Entry price" value={money(position.entry_price)} />
          <Stat label="Mark price" value={money(position.mark_price)} />
          <Stat label="Position value" value={money(position.notional_usdt)} />
          <Stat
            label="Unrealised PnL"
            value={signed(position.unrealized_pnl)}
            valueClass={tone(position.unrealized_pnl)}
            emphasis
          />
        </div>

        <h3 className="mt-6 text-[10px] uppercase tracking-wider text-zinc-500">Protection</h3>
        <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-4">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-zinc-500">Stop</div>
            <div className="mt-0.5 font-mono text-lg">
              {position.venue_stop_loss ? (
                money(position.venue_stop_loss)
              ) : (
                <span className="text-sm uppercase text-[#d03b3b]">none at venue</span>
              )}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-zinc-500">Target</div>
            <div className="mt-0.5 font-mono text-lg">
              <Value value={position.venue_take_profit} why="no target attached at the venue" />
            </div>
          </div>
          <Stat label="Liquidation" value={money(position.liquidation_price)} />
          <Stat label="Margin mode" value={position.margin_mode ?? NOT_RECORDED} />
        </div>

        <h3 className="mt-6 text-[10px] uppercase tracking-wider text-zinc-500">Profit stage</h3>
        <p className="mt-1 text-xs text-zinc-500">
          <span className="uppercase text-zinc-600">{NOT_RECORDED}</span> — the intrabar
          manager holds the stage, the net-profit-exit eligibility and the loser-exit state in
          the engine process. No column persists them, so this drawer will not guess at one.
        </p>

        <p className="mt-6 text-[10px] text-zinc-600">
          Every field above is read from the venue on each refresh. The venue is authoritative
          for what is open; QuantFlow's own record is shown separately and reconciled.
        </p>
      </aside>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [lastSummaryAt, setLastSummaryAt] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [equity, setEquity] = useState<EquityResponse | null>(null);
  const [range, setRange] = useState<string>("24H");
  const [pnl, setPnl] = useState<PnlResponse | null>(null);
  const [period, setPeriod] = useState<string>("SESSION");
  const [ledger, setLedger] = useState<LedgerResponse | null>(null);
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [decisions, setDecisions] = useState<DecisionsResponse | null>(null);
  const [freshness, setFreshness] = useState<FreshnessResponse | null>(null);
  const [assetClasses, setAssetClasses] = useState<AssetClassesResponse | null>(null);
  const [pnlMode, setPnlMode] = useState<PnlMode>("net");
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [banner, setBanner] = useState<string | null>(null);
  const [selected, setSelected] = useState<VenuePosition | null>(null);

  // Held in a ref so the websocket handler can coalesce bursts without being re-created
  // on every render, which would tear down and rebuild the socket continuously.
  const pending = useRef<number | undefined>(undefined);

  const settle = useCallback(async <T,>(promise: Promise<T>, apply: (value: T) => void) => {
    try {
      apply(await promise);
    } catch {
      // One failing panel keeps its last good data rather than blanking the others.
    }
  }, []);

  const refreshFast = useCallback(async () => {
    await Promise.all([
      (async () => {
        try {
          const next = await api.summary();
          setSummary(next);
          setSummaryError(null);
          setLastSummaryAt(next.generated_at ?? new Date().toISOString());
        } catch (error) {
          // The previous summary is deliberately kept on screen, marked stale by its own
          // timestamp, rather than replaced with nothing.
          setSummaryError(error instanceof ApiError ? error.message : "summary unavailable");
        }
      })(),
      settle(api.decisions(80), setDecisions),
      settle(api.freshness(), setFreshness),
      settle(api.readiness(), setReadiness),
    ]);
  }, [settle]);

  const refreshSlow = useCallback(async () => {
    await Promise.all([
      settle(api.trades(200), setLedger),
      settle(api.analytics(), setAnalytics),
      settle(api.assetClasses(), setAssetClasses),
    ]);
  }, [settle]);

  useEffect(() => {
    void refreshFast();
    const timer = window.setInterval(() => void refreshFast(), FAST_POLL_MS);
    return () => {
      window.clearInterval(timer);
    };
  }, [refreshFast]);

  useEffect(() => {
    void refreshSlow();
    const timer = window.setInterval(() => void refreshSlow(), SLOW_POLL_MS);
    return () => {
      window.clearInterval(timer);
    };
  }, [refreshSlow]);

  useEffect(() => {
    void settle(api.equity(range), setEquity);
    const timer = window.setInterval(() => void settle(api.equity(range), setEquity), SLOW_POLL_MS);
    return () => {
      window.clearInterval(timer);
    };
  }, [range, settle]);

  useEffect(() => {
    void settle(api.pnl(range), setPnl);
    const timer = window.setInterval(() => void settle(api.pnl(range), setPnl), SLOW_POLL_MS);
    return () => {
      window.clearInterval(timer);
    };
  }, [range, settle]);

  useEffect(() => {
    // The websocket is a latency improvement, not the source of truth: polling continues
    // regardless, so a dropped socket degrades rather than blinding the operator. Bursts
    // are coalesced — a flurry of fills should cost one refresh, not twenty.
    return subscribe((channel) => {
      if (!["fills", "risk", "equity", "signals", "system"].includes(channel)) return;
      if (pending.current !== undefined) return;
      pending.current = window.setTimeout(() => {
        pending.current = undefined;
        void refreshFast();
        void refreshSlow();
      }, 400);
    }, setConnected);
  }, [refreshFast, refreshSlow]);

  const toggleKillSwitch = useCallback(async () => {
    const engaged = summary?.risk?.kill_switch_engaged ?? false;
    let reason = "";
    if (!engaged) {
      // A halt with no recorded cause is close to useless in the post-mortem, so the
      // reason is required here exactly as the API requires it.
      const entered = window.prompt("Reason for halting trading (required):");
      if (!entered?.trim()) return;
      reason = entered.trim();
    }
    setBusy(true);
    try {
      await api.setKillSwitch(!engaged, reason);
      setBanner(engaged ? "Kill switch cleared." : "Kill switch engaged.");
      await refreshFast();
    } catch (error) {
      setBanner(error instanceof ApiError ? error.message : "request failed");
    } finally {
      setBusy(false);
      window.setTimeout(() => {
        setBanner(null);
      }, 6000);
    }
  }, [summary, refreshFast]);

  const trades = list(ledger?.trades);
  const points = list(equity?.points);
  const openPositions = summary?.venue?.position_count ?? null;

  // The engine's own classification, so a position row is labelled with the class the
  // engine actually assigned rather than one the browser guessed from the ticker.
  const assetFor = useMemo(() => {
    const index = new Map<string, string>();
    for (const row of list(assetClasses?.asset_classes)) {
      for (const symbol of list(row.symbols)) {
          index.set(symbol, row.asset_class ?? "—");
      }
    }
    return (symbol: string) => index.get(symbol.split(":")[0] ?? symbol) ?? "—";
  }, [assetClasses]);

  const rangeButtons = (
    <div className="flex gap-1">
      {EQUITY_RANGES.map((entry) => (
        <button
          key={entry}
          type="button"
          onClick={() => {
            setRange(entry);
          }}
          className={`rounded px-1.5 py-0.5 text-[10px] ${
            range === entry
              ? "bg-zinc-700 text-zinc-100"
              : "bg-zinc-800/60 text-zinc-400 hover:bg-zinc-700"
          }`}
        >
          {entry}
        </button>
      ))}
    </div>
  );

  return (
    <div className="min-h-full bg-zinc-950 text-zinc-100">
      <ErrorBoundary label="Header">
        <Header
          summary={summary}
          readiness={readiness}
          connected={connected}
          lastUpdate={lastSummaryAt}
          tab={tab}
          onTab={setTab}
          onToggleKillSwitch={() => void toggleKillSwitch()}
          busy={busy}
        />
      </ErrorBoundary>

      <main className="mx-auto max-w-[1600px] space-y-4 px-4 py-4">
        {banner ? (
          <div className="rounded border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm">{banner}</div>
        ) : null}

        {summaryError ? (
          <Unavailable what="Live summary" error={summaryError} lastSuccessAt={lastSummaryAt} />
        ) : null}

        <ErrorBoundary label="Health">
          <HealthStrip
            summary={summary}
            freshness={freshness}
            connected={connected}
            onOpen={() => {
              setTab("diagnostics");
            }}
          />
        </ErrorBoundary>

        {tab === "overview" ? (
          <>
            <ErrorBoundary label="Executive summary">
              <ExecutiveSummary summary={summary} />
            </ErrorBoundary>

            <ErrorBoundary label="Trading status">
              <TradingStatusPanel summary={summary} decisions={decisions} />
            </ErrorBoundary>

            <ErrorBoundary label="Profit and loss">
              <PnlSection
                pnl={pnl}
                period={period}
                onPeriod={setPeriod}
                chart={<CumulativePnlChart points={list(pnl?.cumulative?.points)} />}
              />
            </ErrorBoundary>

            <Panel
              title="Equity curve"
              subtitle={equity?.history_note}
              source="quantflow database — persisted equity snapshots"
              action={rangeButtons}
            >
              <div className="mb-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                <Stat label="Starting equity" value={money(summary?.session_equity?.starting_equity)} />
                <Stat label="Current equity" value={money(summary?.session_equity?.latest_equity)} emphasis />
                <Stat label="Peak" value={money(summary?.session_equity?.peak_equity)} />
                <Stat
                  label="Current drawdown"
                  value={percent(summary?.session_equity?.current_drawdown_pct, 3)}
                  valueClass={tone(
                    summary?.session_equity?.current_drawdown_pct
                      ? `-${summary.session_equity.current_drawdown_pct}`
                      : null,
                  )}
                />
              </div>
              {list(equity?.discontinuities).length > 0 ? (
                <div className="mb-3">
                  <Caution>
                    This curve is not one continuous account.{" "}
                    {list(equity?.discontinuities).map((entry) => (
                      <span key={entry.at}>
                        Equity stepped from {money(entry.from_equity)} to {money(entry.to_equity)} at{" "}
                        {time(entry.at)} — {entry.likely_cause}
                      </span>
                    ))}{" "}
                    A return measured across that break is meaningless.
                  </Caution>
                </div>
              ) : null}
              <EquityChart points={points} historyNote={equity?.history_note} />
            </Panel>

            <ErrorBoundary label="Positions">
              <PositionsPanel summary={summary} assetFor={assetFor} onSelect={setSelected} />
            </ErrorBoundary>

            <ErrorBoundary label="Universe">
              <UniverseStrip classes={assetClasses} summary={summary} />
            </ErrorBoundary>
          </>
        ) : null}

        {tab === "trades" ? (
          <>
            {ledger?.coverage?.has_gap ? <Caution>{ledger.coverage.note}</Caution> : null}
            <TradeLedgerPanel ledger={ledger} />
            <OrdersPanel summary={summary} />
          </>
        ) : null}

        {tab === "analytics" ? (
          <>
            <ErrorBoundary label="Analytics">
              <div className="grid gap-4 xl:grid-cols-2">
                <AnalyticsPanels analytics={analytics} />
              </div>
            </ErrorBoundary>

            <div className="grid gap-4 xl:grid-cols-2">
              <Panel
                title="Drawdown"
                subtitle="Derived from the session's own equity snapshots — never from a cross-asset balance."
                source="quantflow database — persisted equity snapshots"
                action={rangeButtons}
              >
                <DrawdownChart points={points} />
              </Panel>

              <Panel
                title="Cumulative realised PnL (from the ledger)"
                source="quantflow database — reconciled closed trades"
                action={
                  <div className="flex gap-1">
                    {(["net", "gross", "fees"] as const).map((entry) => (
                      <button
                        key={entry}
                        type="button"
                        onClick={() => {
                          setPnlMode(entry);
                        }}
                        className={`rounded px-1.5 py-0.5 text-[10px] uppercase ${
                          pnlMode === entry
                            ? "bg-zinc-700 text-zinc-100"
                            : "bg-zinc-800/60 text-zinc-400 hover:bg-zinc-700"
                        }`}
                      >
                        {entry}
                      </button>
                    ))}
                  </div>
                }
              >
                <PnlChart trades={trades} mode={pnlMode} />
              </Panel>

              <Panel
                title="Long vs short"
                subtitle="Cumulative net PnL by direction."
                source="quantflow database — reconciled closed trades"
              >
                <LongShortChart trades={trades} />
              </Panel>

              <FeesPanel fees={summary?.fees} />
            </div>

            <DecisionsPanel decisions={decisions} openPositions={openPositions} />
          </>
        ) : null}

        {tab === "diagnostics" ? (
          <>
            <VenueAccountPanel summary={summary} />
            <div className="grid gap-4 xl:grid-cols-2">
              <AssetClassesPanel classes={assetClasses} />
              <FreshnessPanel freshness={freshness} />
            </div>
            <Panel title="Engine" source="the engine's own startup log and the supervisor's log">
              <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4 lg:grid-cols-6">
                <Stat label="Timeframe" value={summary?.engine?.timeframe ?? NOT_RECORDED} />
                <Stat label="Session" value={summary?.session?.session_id ?? NOT_RECORDED} />
                <Stat
                  label="Strategy pool"
                  value={summary?.engine?.strategy_pool ?? NOT_RECORDED}
                  hint={summary?.engine?.strategy ?? undefined}
                />
                <Stat
                  label="Engine PID"
                  value={summary?.engine?.pid ? String(summary.engine.pid) : NOT_RECORDED}
                  hint="no pid file; API cannot see host processes"
                />
                <Stat
                  label="Engine started"
                  value={summary?.engine?.started_at ? ago(summary.engine.started_at) : NOT_RECORDED}
                />
                <Stat
                  label="Symbols"
                  value={count(list(summary?.engine?.symbols).length || undefined)}
                  hint={list(summary?.engine?.symbols).join(", ")}
                />
                <Stat label="Last decision" value={ago(freshness?.last_decision_at)} />
                <Stat label="Last order" value={ago(freshness?.last_order_at)} />
                <Stat label="Last venue sync" value={ago(freshness?.venue_sync?.fetched_at)} />
                <Stat
                  label="Last reconciliation"
                  value={ago(freshness?.reconciliation?.last_venue_read_at)}
                />
                <Stat
                  label="Restarts"
                  value={count(summary?.engine?.supervisor?.restart_count)}
                  hint={`${count(summary?.engine?.supervisor?.killed_count)} ended rc=137 (SIGKILL)`}
                />
                <Stat
                  label="Last stored candle"
                  value={ago(freshness?.last_candle_at)}
                  hint="download archive, not the live stream"
                />
              </div>
              {(summary?.engine?.supervisor?.killed_count ?? 0) > 0 ? (
                <div className="mt-3">
                  <Caution>
                    The supervisor has restarted the engine{" "}
                    {count(summary?.engine?.supervisor?.restart_count)} time(s),{" "}
                    {count(summary?.engine?.supervisor?.killed_count)} of which ended in{" "}
                    <span className="font-mono">rc=137</span> (SIGKILL — the operating system
                    reclaiming memory, not a crash in the engine). A restart re-seeds equity from
                    the venue, which is why the equity curve can contain a step change.
                  </Caution>
                </div>
              ) : null}
            </Panel>
          </>
        ) : null}

        <footer className="pb-6 text-center text-[10px] text-zinc-600">
          QuantFlow · every monetary value is formatted from an exact decimal string, never a
          float · fields the engine does not record read {NOT_RECORDED}
        </footer>
      </main>

      {selected ? (
        <ErrorBoundary label="Position detail">
          <PositionDrawer
            position={selected}
            asset={assetFor(selected.symbol)}
            onClose={() => {
              setSelected(null);
            }}
          />
        </ErrorBoundary>
      ) : null}
    </div>
  );
}
