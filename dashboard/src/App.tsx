/**
 * QuantFlow dashboard.
 *
 * Two things drive the layout. First, **risk state is always visible** — the kill switch
 * and the halt indicator sit in the header, not behind a tab, because the moment you need
 * them is the moment you should not be navigating. Second, every number that could mislead
 * carries its caveat next to it: a win rate over eight trades is labelled as such.
 */

import { useCallback, useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ApiError,
  api,
  subscribe,
  type CandlesResponse,
  type PerformanceReview,
  type Portfolio,
  type ReadinessResponse,
  type RiskEvent,
  type RiskStatus,
  type LiveAccount,
  type LiveFills,
  type SeriesSummary,
  type Session,
  type StrategyDescription,
  type Trade,
} from "./lib/api";
import { ago, chartValue, money, percent, quantity, signed, time, tone } from "./lib/format";

const POLL_INTERVAL_MS = 5_000;

function Panel({
  title,
  children,
  action,
}: {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-zinc-800 bg-zinc-900/60">
      <header className="flex items-center justify-between border-b border-zinc-800 px-4 py-2.5">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-400">{title}</h2>
        {action}
      </header>
      <div className="p-4">{children}</div>
    </section>
  );
}

function Stat({
  label,
  value,
  valueClass = "",
  hint,
}: {
  label: string;
  value: string;
  valueClass?: string;
  hint?: string;
}) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wider text-zinc-500">{label}</div>
      <div className={`mt-0.5 font-mono text-lg tabular-nums ${valueClass}`}>{value}</div>
      {hint ? <div className="text-[11px] text-zinc-600">{hint}</div> : null}
    </div>
  );
}

function Empty({ message }: { message: string }) {
  return <p className="py-6 text-center text-sm text-zinc-500">{message}</p>;
}

/** The header. Risk state is deliberately the most prominent thing on the page. */
function Header({
  readiness,
  risk,
  session,
  connected,
  onToggleKillSwitch,
  busy,
}: {
  readiness: ReadinessResponse | null;
  risk: RiskStatus | null;
  session: Session | null;
  connected: boolean;
  onToggleKillSwitch: () => void;
  busy: boolean;
}) {
  const engaged = risk?.kill_switch.engaged ?? false;
  const halted = risk?.trading_halted ?? false;

  return (
    <header className="sticky top-0 z-10 border-b border-zinc-800 bg-zinc-950/95 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <h1 className="text-sm font-semibold tracking-tight">QuantFlow</h1>

        <span className="rounded bg-zinc-800 px-2 py-0.5 text-[11px] uppercase tracking-wider text-zinc-300">
          {readiness?.trading_mode ?? "…"}
        </span>

        <span className="flex items-center gap-1.5 text-[11px] text-zinc-400">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              readiness?.ready ? "bg-emerald-400" : "bg-rose-400"
            }`}
          />
          {readiness?.ready ? "systems ready" : "not ready"}
        </span>

        <span className="flex items-center gap-1.5 text-[11px] text-zinc-500">
          <span
            className={`h-1.5 w-1.5 rounded-full ${connected ? "bg-emerald-400" : "bg-zinc-600"}`}
          />
          {connected ? "live" : "polling"}
        </span>

        {/* Named explicitly: trades and attribution below are scoped to this run, and an
            operator must never have to guess which numbers they are reading. */}
        {session ? (
          <span
            className="truncate text-[11px] text-zinc-500"
            title={`${session.strategy_id} · ${session.symbols.join(", ")} · ${session.timeframe}`}
          >
            session <span className="text-zinc-300">{session.session_id}</span>
            <span className="text-zinc-600"> · {session.status}</span>
          </span>
        ) : null}

        <div className="ml-auto flex items-center gap-3">
          {halted && !engaged ? (
            <span className="rounded bg-amber-500/15 px-2 py-1 text-[11px] font-medium text-amber-300">
              trading halted for the day
            </span>
          ) : null}

          {engaged ? (
            <span
              className="rounded bg-rose-500/15 px-2 py-1 text-[11px] font-medium text-rose-300"
              title={risk?.kill_switch.reason ?? undefined}
            >
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
                : "bg-rose-600 text-white hover:bg-rose-500"
            }`}
          >
            {engaged ? "Clear kill switch" : "Halt trading"}
          </button>
        </div>
      </div>
    </header>
  );
}

export default function App() {
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [risk, setRisk] = useState<RiskStatus | null>(null);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [portfolioError, setPortfolioError] = useState<string | null>(null);
  const [events, setEvents] = useState<RiskEvent[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [strategies, setStrategies] = useState<StrategyDescription[]>([]);
  const [series, setSeries] = useState<SeriesSummary[]>([]);
  const [candles, setCandles] = useState<CandlesResponse | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [review, setReview] = useState<PerformanceReview | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [account, setAccount] = useState<LiveAccount | null>(null);
  const [accountError, setAccountError] = useState<string | null>(null);
  const [fills, setFills] = useState<LiveFills | null>(null);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [banner, setBanner] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const settle = async <T,>(promise: Promise<T>, apply: (value: T) => void) => {
      try {
        apply(await promise);
      } catch {
        // One failing panel must not blank the others; each keeps its last good data.
      }
    };

    // Resolved first, not in the fan-out: trades and the performance review are scoped to
    // it, and an unscoped call falls back to a trailing window that hides older trades.
    let current: Session | null = null;
    try {
      current = (await api.sessions())[0] ?? null;
      setSession(current);
    } catch {
      // Fall through unscoped rather than blanking the dashboard.
    }
    const sessionId = current?.session_id ?? null;

    await Promise.all([
      settle(api.readiness(), setReadiness),
      settle(api.riskStatus(), setRisk),
      settle(api.riskEvents(25, sessionId), setEvents),
      settle(api.trades(200, sessionId), setTrades),
      settle(api.strategies(), setStrategies),
      settle(api.series(), setSeries),
      settle(api.review(365, sessionId), setReview),
      settle(api.accountFills("BTC/USDT", 25), setFills),
      (async () => {
        try {
          setAccount(await api.account());
          setAccountError(null);
        } catch (error) {
          // Never fall back to stored paper state: a live balance that is silently a
          // backtest number is worse than an error.
          setAccount(null);
          setAccountError(
            error instanceof ApiError ? error.message : "live account unavailable",
          );
        }
      })(),
      (async () => {
        try {
          setPortfolio(await api.portfolio());
          setPortfolioError(null);
        } catch (error) {
          // Expected when no session is running: say so plainly rather than showing
          // a portfolio of zeros that reads as a flat account.
          setPortfolio(null);
          setPortfolioError(
            error instanceof ApiError ? error.message : "portfolio unavailable",
          );
        }
      })(),
    ]);
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => { window.clearInterval(timer); };
  }, [refresh]);

  useEffect(() => {
    // The websocket is a latency improvement, not the source of truth: polling continues
    // regardless, so a dropped socket degrades rather than blinding the operator.
    return subscribe(
      (channel) => {
        if (channel === "fills" || channel === "risk" || channel === "equity") {
          void refresh();
        }
      },
      setConnected,
    );
  }, [refresh]);

  useEffect(() => {
    if (selected) return;
    // Prefer a symbol the running session actually trades. `series[0]` is whatever sorts
    // first, which in practice is a daily series from an unrelated backfill - it cannot
    // move intraday, so the chart reads as frozen while the session is running fine.
    const sessionSymbol = session?.symbols.find((symbol) =>
      series.some((item) => item.symbol === symbol),
    );
    const first = sessionSymbol ?? series[0]?.symbol;
    if (first) setSelected(first);
  }, [series, selected, session]);

  useEffect(() => {
    if (!selected) return;
    // Match the session's timeframe when that series exists, so the chart and the engine
    // are looking at the same bars. Falls back to whatever is stored for the symbol.
    const entry =
      series.find(
        (item) => item.symbol === selected && item.timeframe === session?.timeframe,
      ) ?? series.find((item) => item.symbol === selected);
    void api
      .candles(selected, entry?.timeframe ?? "1h", 300)
      .then(setCandles)
      .catch(() => { setCandles(null); });
  }, [selected, series, session]);

  const toggleKillSwitch = useCallback(async () => {
    const engaged = risk?.kill_switch.engaged ?? false;
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
      await refresh();
    } catch (error) {
      setBanner(error instanceof ApiError ? error.message : "request failed");
    } finally {
      setBusy(false);
      window.setTimeout(() => { setBanner(null); }, 6000);
    }
  }, [risk, refresh]);

  const equityCurve = trades
    .slice()
    .sort((a, b) => a.exit_time.localeCompare(b.exit_time))
    .reduce<{ time: string; pnl: number }[]>((accumulated, trade) => {
      const previous = accumulated.at(-1)?.pnl ?? 0;
      accumulated.push({
        time: time(trade.exit_time),
        pnl: previous + chartValue(trade.net_pnl),
      });
      return accumulated;
    }, []);

  const priceSeries =
    candles?.candles.map((candle) => ({
      time: time(candle.open_time),
      close: chartValue(candle.close),
    })) ?? [];

  return (
    <div className="min-h-full bg-zinc-950 text-zinc-100">
      <Header
        readiness={readiness}
        risk={risk}
        session={session}
        connected={connected}
        onToggleKillSwitch={() => void toggleKillSwitch()}
        busy={busy}
      />

      <main className="mx-auto max-w-7xl space-y-4 px-4 py-5">
        <Panel
          title={
            account
              ? `Live ${account.venue.toUpperCase()} account (${account.network})`
              : "Live exchange account"
          }
        >
          {accountError ? (
            <Empty message={`Not connected: ${accountError}`} />
          ) : !account ? (
            <Empty message="Loading live account…" />
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
                <Stat label="Total balance" value={money(account.total_balance)} />
                <Stat label="Available" value={money(account.available_balance)} />
                <Stat
                  label="Unrealised PnL"
                  value={signed(account.unrealized_pnl)}
                  valueClass={tone(account.unrealized_pnl)}
                />
                <Stat
                  label="Realised PnL"
                  value={fills ? signed(fills.realized_pnl) : "—"}
                  valueClass={fills ? tone(fills.realized_pnl) : ""}
                  {...(fills ? { hint: `fees ${money(fills.total_fees)}` } : {})}
                />
                <Stat label="Open orders" value={String(account.open_order_count)} />
              </div>

              <div className="mt-4 grid gap-4 lg:grid-cols-2">
                <div>
                  <div className="mb-1 text-[11px] uppercase tracking-wider text-zinc-500">
                    Positions ({account.position_count})
                  </div>
                  {account.positions.length === 0 ? (
                    <p className="text-sm text-zinc-500">No open positions.</p>
                  ) : (
                    <table className="w-full text-sm">
                      <tbody className="font-mono tabular-nums">
                        {account.positions.map((position) => (
                          <tr key={position.symbol} className="border-t border-zinc-800">
                            <td className="py-1.5 font-sans">{position.symbol}</td>
                            <td className="py-1.5">{position.side}</td>
                            <td className="py-1.5 text-right">{quantity(position.quantity)}</td>
                            <td className="py-1.5 text-right">{money(position.entry_price)}</td>
                            <td
                              className={`py-1.5 text-right ${tone(position.unrealized_pnl)}`}
                            >
                              {signed(position.unrealized_pnl)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>

                <div>
                  <div className="mb-1 text-[11px] uppercase tracking-wider text-zinc-500">
                    Recent fills{fills ? ` (${fills.symbol}, ${fills.count})` : ""}
                  </div>
                  {!fills || fills.fills.length === 0 ? (
                    <p className="text-sm text-zinc-500">No fills on the venue.</p>
                  ) : (
                    <table className="w-full text-sm">
                      <tbody className="font-mono tabular-nums">
                        {fills.fills.slice(0, 6).map((fill) => (
                          <tr key={fill.fill_id} className="border-t border-zinc-800">
                            <td className="py-1.5 font-sans">{time(fill.timestamp)}</td>
                            <td className="py-1.5">{fill.side}</td>
                            <td className="py-1.5 text-right">{quantity(fill.quantity)}</td>
                            <td className="py-1.5 text-right">{money(fill.price)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
              </div>

              {account.open_orders.length > 0 ? (
                <div className="mt-4">
                  <div className="mb-1 text-[11px] uppercase tracking-wider text-zinc-500">
                    Working orders
                  </div>
                  <table className="w-full text-sm">
                    <tbody className="font-mono tabular-nums">
                      {account.open_orders.map((order) => (
                        <tr key={order.order_id} className="border-t border-zinc-800">
                          <td className="py-1.5 font-sans">{order.symbol}</td>
                          <td className="py-1.5">
                            {order.side} {order.type}
                          </td>
                          <td className="py-1.5 text-right">{quantity(order.quantity)}</td>
                          <td className="py-1.5 text-right">
                            {order.price ? money(order.price) : "market"}
                          </td>
                          <td className="py-1.5 text-right">{order.status}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </>
          )}
        </Panel>

        {banner ? (
          <div className="rounded border border-zinc-700 bg-zinc-900 px-4 py-2 text-sm">
            {banner}
          </div>
        ) : null}

        {review?.warnings.length ? (
          <div className="rounded border border-amber-500/30 bg-amber-500/10 px-4 py-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-amber-300">
              Read before acting on these numbers
            </h3>
            <ul className="mt-2 space-y-1 text-sm text-amber-100/90">
              {review.warnings.map((warning) => (
                <li key={warning}>• {warning}</li>
              ))}
            </ul>
          </div>
        ) : null}

        <div className="grid gap-4 lg:grid-cols-3">
          <div className="lg:col-span-2">
            <Panel title="Portfolio">
              {portfolio ? (
                <>
                  <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                    <Stat label="Equity" value={money(portfolio.equity, portfolio.base_currency)} />
                    <Stat
                      label="Return"
                      value={percent(portfolio.total_return_pct)}
                      valueClass={tone(portfolio.total_return_pct)}
                    />
                    <Stat
                      label="Daily PnL"
                      value={signed(portfolio.daily_pnl)}
                      valueClass={tone(portfolio.daily_pnl)}
                    />
                    <Stat
                      label="Drawdown"
                      value={percent(portfolio.drawdown_pct)}
                      valueClass={
                        chartValue(portfolio.drawdown_pct) > 0.1 ? "text-rose-400" : ""
                      }
                    />
                    <Stat label="Cash" value={money(portfolio.cash)} />
                    <Stat label="Exposure" value={money(portfolio.gross_exposure)} />
                    <Stat label="Leverage" value={`${chartValue(portfolio.leverage).toFixed(2)}x`} />
                    <Stat label="Fees paid" value={money(portfolio.fees_paid)} />
                  </div>

                  <h3 className="mt-6 mb-2 text-[11px] uppercase tracking-wider text-zinc-500">
                    Open positions ({portfolio.position_count})
                  </h3>
                  {portfolio.positions.length === 0 ? (
                    <Empty message="No open positions." />
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="text-left text-[11px] uppercase tracking-wider text-zinc-500">
                            <th className="pb-2">Symbol</th>
                            <th className="pb-2">Side</th>
                            <th className="pb-2 text-right">Qty</th>
                            <th className="pb-2 text-right">Entry</th>
                            <th className="pb-2 text-right">Mark</th>
                            <th className="pb-2 text-right">Unrealised</th>
                            <th className="pb-2 text-right">Stop</th>
                          </tr>
                        </thead>
                        <tbody className="font-mono tabular-nums">
                          {portfolio.positions.map((position) => (
                            <tr key={position.symbol} className="border-t border-zinc-800">
                              <td className="py-1.5 font-sans">{position.symbol}</td>
                              <td className="py-1.5 font-sans uppercase">{position.side}</td>
                              <td className="py-1.5 text-right">{quantity(position.quantity)}</td>
                              <td className="py-1.5 text-right">
                                {money(position.average_entry_price)}
                              </td>
                              <td className="py-1.5 text-right">{money(position.mark_price)}</td>
                              <td
                                className={`py-1.5 text-right ${tone(position.unrealized_pnl)}`}
                              >
                                {signed(position.unrealized_pnl)}
                              </td>
                              <td className="py-1.5 text-right">
                                {position.stop_loss_price ? (
                                  money(position.stop_loss_price)
                                ) : (
                                  <span className="text-rose-400">none</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </>
              ) : (
                <Empty message={portfolioError ?? "Loading…"} />
              )}
            </Panel>
          </div>

          <Panel title="Risk">
            {risk ? (
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-zinc-400">Kill switch</span>
                  <span
                    className={risk.kill_switch.engaged ? "text-rose-400" : "text-emerald-400"}
                  >
                    {risk.kill_switch.engaged ? "ENGAGED" : "clear"}
                  </span>
                </div>
                {risk.kill_switch.reason ? (
                  <p className="rounded bg-zinc-800/60 px-2 py-1.5 text-xs text-zinc-300">
                    {risk.kill_switch.reason}
                    {risk.kill_switch.engaged_by ? ` — ${risk.kill_switch.engaged_by}` : ""}
                  </p>
                ) : null}
                <div className="flex items-center justify-between">
                  <span className="text-zinc-400">Trading halted</span>
                  <span className={risk.trading_halted ? "text-amber-400" : "text-zinc-300"}>
                    {risk.trading_halted ? "yes" : "no"}
                  </span>
                </div>

                <div className="border-t border-zinc-800 pt-3">
                  <div className="mb-2 text-[11px] uppercase tracking-wider text-zinc-500">
                    Limits
                  </div>
                  <dl className="space-y-1 font-mono text-xs tabular-nums">
                    <div className="flex justify-between">
                      <dt className="font-sans text-zinc-400">Max position</dt>
                      <dd>{percent(risk.limits.max_position_pct, 0)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="font-sans text-zinc-400">Max exposure</dt>
                      <dd>{percent(risk.limits.max_total_exposure_pct, 0)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="font-sans text-zinc-400">Daily loss</dt>
                      <dd>{percent(risk.limits.max_daily_loss_pct, 0)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="font-sans text-zinc-400">Max drawdown</dt>
                      <dd>{percent(risk.limits.max_drawdown_pct, 0)}</dd>
                    </div>
                    <div className="flex justify-between">
                      <dt className="font-sans text-zinc-400">Stop required</dt>
                      <dd className={risk.limits.require_stop_loss ? "" : "text-rose-400"}>
                        {risk.limits.require_stop_loss ? "yes" : "NO"}
                      </dd>
                    </div>
                  </dl>
                </div>

                <div className="border-t border-zinc-800 pt-2 text-[11px] text-zinc-500">
                  {risk.rules.length} rules active · sizer {risk.sizer}
                </div>
              </div>
            ) : (
              <Empty message="Loading…" />
            )}
          </Panel>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <Panel
            title="Price"
            action={
              <select
                value={selected ?? ""}
                onChange={(event) => { setSelected(event.target.value); }}
                className="rounded border border-zinc-700 bg-zinc-800 px-2 py-1 text-xs"
              >
                {series.map((entry) => (
                  <option key={`${entry.symbol}-${entry.timeframe}`} value={entry.symbol}>
                    {entry.symbol} {entry.timeframe}
                  </option>
                ))}
              </select>
            }
          >
            {priceSeries.length === 0 ? (
              <Empty message="No stored market data. Run a download first." />
            ) : (
              <>
                {candles && candles.gaps > 0 ? (
                  <p className="mb-2 text-xs text-amber-400">
                    {candles.gaps} bars missing from this range — the chart has holes.
                  </p>
                ) : null}
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={priceSeries}>
                    <CartesianGrid stroke="#27272a" vertical={false} />
                    <XAxis dataKey="time" hide />
                    <YAxis
                      domain={["auto", "auto"]}
                      tick={{ fill: "#71717a", fontSize: 11 }}
                      width={70}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#18181b",
                        border: "1px solid #3f3f46",
                        borderRadius: 6,
                        fontSize: 12,
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="close"
                      stroke="#60a5fa"
                      dot={false}
                      strokeWidth={1.5}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </>
            )}
          </Panel>

          <Panel title="Cumulative realised PnL">
            {equityCurve.length === 0 ? (
              <Empty message="No closed trades yet." />
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={equityCurve}>
                  <CartesianGrid stroke="#27272a" vertical={false} />
                  <XAxis dataKey="time" hide />
                  <YAxis tick={{ fill: "#71717a", fontSize: 11 }} width={70} />
                  <Tooltip
                    contentStyle={{
                      background: "#18181b",
                      border: "1px solid #3f3f46",
                      borderRadius: 6,
                      fontSize: 12,
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="pnl"
                    stroke="#34d399"
                    fill="#34d39922"
                    strokeWidth={1.5}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </Panel>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <Panel title="Recent risk events">
            {events.length === 0 ? (
              <Empty message="No risk events recorded." />
            ) : (
              <ul className="space-y-2">
                {events.slice(0, 8).map((event, index) => (
                  <li
                    key={`${event.created_at}-${index}`}
                    className="border-l-2 border-zinc-700 pl-3 text-sm"
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span
                        className={
                          event.halted_trading
                            ? "font-medium text-rose-400"
                            : "font-medium text-amber-400"
                        }
                      >
                        {event.rule}
                      </span>
                      <span className="text-[11px] text-zinc-500">{ago(event.created_at)}</span>
                    </div>
                    <p className="text-xs text-zinc-400">{event.message}</p>
                  </li>
                ))}
              </ul>
            )}
          </Panel>

          <Panel title="Strategy attribution">
            {!review || review.by_strategy.length === 0 ? (
              <Empty message="No closed trades to attribute." />
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-[11px] uppercase tracking-wider text-zinc-500">
                    <th className="pb-2">Strategy</th>
                    <th className="pb-2 text-right">Trades</th>
                    <th className="pb-2 text-right">Net PnL</th>
                    <th className="pb-2 text-right">Win rate</th>
                  </tr>
                </thead>
                <tbody className="font-mono tabular-nums">
                  {review.by_strategy.map((entry) => (
                    <tr key={entry.key} className="border-t border-zinc-800">
                      <td className="py-1.5 font-sans">
                        {entry.key}
                        {!entry.reliable ? (
                          <span
                            className="ml-1.5 text-[10px] text-amber-500"
                            title="too few trades for this to be meaningful"
                          >
                            small sample
                          </span>
                        ) : null}
                      </td>
                      <td className="py-1.5 text-right">{entry.trade_count}</td>
                      <td className={`py-1.5 text-right ${tone(entry.net_pnl)}`}>
                        {signed(entry.net_pnl)}
                      </td>
                      <td className="py-1.5 text-right">{percent(entry.win_rate, 1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Panel>
        </div>

        <Panel title={`Registered strategies (${strategies.length})`}>
          {strategies.length === 0 ? (
            <Empty message="Loading…" />
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {strategies.map((strategy) => (
                <div key={strategy.strategy_id} className="rounded border border-zinc-800 p-3">
                  <div className="font-mono text-sm text-blue-300">{strategy.strategy_id}</div>
                  <p className="mt-1 text-xs text-zinc-400">{strategy.description}</p>
                  <p className="mt-2 text-[11px] text-zinc-600">
                    warm-up {strategy.warmup_bars} bars
                  </p>
                </div>
              ))}
            </div>
          )}
        </Panel>

        <footer className="pb-6 text-center text-[11px] text-zinc-600">
          QuantFlow {readiness ? `· ${readiness.trading_mode} mode` : ""} · values shown are
          formatted from exact decimal strings
        </footer>
      </main>
    </div>
  );
}
