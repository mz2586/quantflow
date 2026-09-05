/**
 * The first screen.
 *
 * Everything here exists to answer eight questions in under five seconds: how much money
 * the account has, how much of it is in trades, what it has made, what it has lost, what
 * it has paid in fees, how many positions are open, whether it is trading, and — when it
 * is not — why not.
 *
 * The discipline that makes those answers trustworthy is the same one the rest of the
 * console follows: a figure from the venue and a figure from QuantFlow's own accounting
 * are never added together, and every tile says which of the two it came from. Wallet,
 * available and in-trades are the venue's. Profit, loss, fees and drawdown are
 * QuantFlow's, computed from reconciled closed trades.
 */

import type { ReactNode } from "react";
import type {
  AssetClassesResponse,
  DecisionsResponse,
  FreshnessResponse,
  PnlPeriod,
  PnlResponse,
  Summary,
} from "../lib/api";
import { list } from "../lib/api";
import { NOT_RECORDED, ago, count, money, percent, ratio, signed, tone } from "../lib/format";
import { Caution, Contradiction, InsufficientSample, Panel } from "./ui";

/** One headline figure. Larger and quieter than a `Stat`: this row is read at a glance. */
function Tile({
  label,
  value,
  valueClass = "",
  note,
  source,
  size = "lg",
}: {
  label: string;
  value: string;
  valueClass?: string | undefined;
  note?: string | undefined;
  source?: string | undefined;
  size?: "lg" | "xl";
}) {
  const missing = value === NOT_RECORDED;
  return (
    <div className="rounded-lg bg-zinc-900/60 px-3 py-2.5 ring-1 ring-zinc-800">
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</div>
      <div
        className={`mt-1 font-mono tabular-nums ${size === "xl" ? "text-[26px]" : "text-xl"} leading-tight ${
          missing ? "text-[11px] font-sans uppercase tracking-wider text-zinc-600" : valueClass
        }`}
      >
        {value}
      </div>
      {note ? <div className="mt-0.5 text-[10px] text-zinc-500">{note}</div> : null}
      {source ? <div className="mt-0.5 text-[9px] uppercase tracking-wider text-zinc-600">{source}</div> : null}
    </div>
  );
}

interface StatusStyle {
  dot: string;
  text: string;
  ring: string;
}

const NEUTRAL_TONE: StatusStyle = { dot: "bg-zinc-500", text: "text-zinc-300", ring: "ring-zinc-700" };

const STATUS_TONE: Record<string, StatusStyle> = {
  TRADING: { dot: "bg-[#0ca30c]", text: "text-[#0ca30c]", ring: "ring-[#0ca30c]/40" },
  "WAITING FOR QUALIFIED SIGNAL": { dot: "bg-[#fab219]", text: "text-[#fab219]", ring: "ring-[#fab219]/30" },
  FILTERING: { dot: "bg-[#fab219]", text: "text-[#fab219]", ring: "ring-[#fab219]/30" },
  "RISK BLOCKED": { dot: "bg-[#d03b3b]", text: "text-[#d03b3b]", ring: "ring-[#d03b3b]/40" },
  "EXECUTION BLOCKED": { dot: "bg-[#d03b3b]", text: "text-[#d03b3b]", ring: "ring-[#d03b3b]/40" },
  "ENGINE ERROR": { dot: "bg-[#d03b3b]", text: "text-[#d03b3b]", ring: "ring-[#d03b3b]/40" },
  DISCONNECTED: { dot: "bg-[#d03b3b]", text: "text-[#d03b3b]", ring: "ring-[#d03b3b]/40" },
  STARTING: NEUTRAL_TONE,
};

function toneFor(state: string | undefined): StatusStyle {
  return STATUS_TONE[state ?? ""] ?? NEUTRAL_TONE;
}

/** Reduce the derived engine state to the three words the operator actually needs. */
function shortStatus(state: string | undefined): "TRADING" | "WAITING" | "BLOCKED" | "…" {
  if (!state) return "…";
  if (state === "TRADING") return "TRADING";
  if (state.includes("BLOCKED") || state === "ENGINE ERROR" || state === "DISCONNECTED") {
    return "BLOCKED";
  }
  return "WAITING";
}

/**
 * The two-row executive summary.
 *
 * `IN TRADES` is reported as **notional** — the face value the account is exposed to —
 * with the margin figure carried underneath, because those differ by the leverage
 * multiple and an unlabelled "in trades" is ambiguous by exactly that factor.
 */
export function ExecutiveSummary({ summary }: { summary: Summary | null }) {
  const account = summary?.venue?.account;
  const deployed = summary?.venue?.deployed;
  const performance = summary?.trading_performance;
  const equity = summary?.session_equity;
  const positions = summary?.venue?.position_count;

  return (
    <section className="space-y-3">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <Tile
          label="Capital base"
          value={money(equity?.capital_base ?? equity?.starting_equity)}
          note="full venue USDT equity — the base for every percentage limit"
          source="venue wallet"
          size="xl"
        />
        <Tile
          label="Wallet / trading equity"
          value={money(account?.trading_equity_usdt)}
          note="USDT in the venue wallet — may exceed the allocation"
          source="venue"
          size="xl"
        />
        <Tile
          label="Available"
          value={money(account?.available_usdt)}
          note="USDT free to deploy"
          source="venue"
          size="xl"
        />
        <Tile
          label="In trades (notional)"
          value={money(deployed?.notional_usdt)}
          note={`margin ${money(deployed?.margin_usdt)} USDT`}
          source="venue positions"
          size="xl"
        />
        <Tile
          label="Net profit"
          value={signed(performance?.net_realized_pnl)}
          valueClass={tone(performance?.net_realized_pnl)}
          note="realised, after fees"
          source="quantflow — closed trades"
          size="xl"
        />
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Tile
          label="Realised PnL"
          value={signed(performance?.net_realized_pnl)}
          valueClass={tone(performance?.net_realized_pnl)}
          note={`${count(performance?.closed_trades)} closed`}
        />
        <Tile
          label="Unrealised PnL"
          value={signed(summary?.venue?.unrealized_pnl)}
          valueClass={tone(summary?.venue?.unrealized_pnl)}
          note="open positions, marked by the venue"
        />
        <Tile
          label="Total fees"
          value={`-${money(performance?.total_fees)}`}
          valueClass="text-[#c98500]"
          note="reconciled from fills"
        />
        <Tile
          label="Drawdown"
          value={percent(equity?.current_drawdown_pct, 2)}
          valueClass={tone(equity?.current_drawdown_pct ? `-${equity.current_drawdown_pct}` : null)}
          note={`peak ${money(equity?.peak_equity)}`}
        />
        <Tile
          label="Open positions"
          value={count(positions)}
          note={positions === 0 ? "book is flat" : undefined}
        />
        <EngineAndTradingTile summary={summary} />
      </div>
    </section>
  );
}

/**
 * Engine health and trading status, side by side and never merged.
 *
 * These answer different questions and were previously collapsed into one red box. A
 * single order refused by the position cap — the risk engine working exactly as
 * configured — showed the whole engine as BLOCKED while it went on evaluating bars and
 * selecting candidates. An operator reading that goes hunting for a fault that is not
 * there.
 *
 * Engine health comes from the heartbeat the engine writes to Redis; trading status comes
 * from what it is deciding. One can be RUNNING while the other is WAITING, and that
 * combination is the normal state of a selective system.
 */
function EngineAndTradingTile({ summary }: { summary: Summary | null }) {
  const engine = summary?.engine_health;
  const engineState = engine?.state ?? "UNKNOWN";
  const engineOk = engineState === "RUNNING";
  const trading = shortStatus(summary?.status?.state);
  const style = toneFor(summary?.status?.state);
  const refused = summary?.status?.evidence?.last_order_refused;

  return (
    <div className={`rounded-lg bg-zinc-900/60 px-3 py-2.5 ring-1 ${style.ring}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wider text-zinc-500">Engine</span>
        <span
          className={`flex items-center gap-1 text-[11px] font-medium ${
            engineOk ? "text-[#0ca30c]" : "text-[#fab219]"
          }`}
          title={engine?.detail}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${engineOk ? "bg-[#0ca30c]" : "bg-[#fab219]"}`} />
          {engineState}
        </span>
      </div>
      <div className="mt-1 text-[10px] uppercase tracking-wider text-zinc-500">Trading</div>
      <div className={`flex items-center gap-2 text-lg font-semibold ${style.text}`}>
        <span className={`h-2 w-2 rounded-full ${style.dot}`} />
        {trading}
      </div>
      <div className="mt-0.5 truncate text-[10px] text-zinc-500" title={summary?.status?.detail}>
        {summary?.status?.state ?? "…"}
      </div>
      {refused ? (
        <div className="mt-1 truncate text-[10px] text-[#c98500]">last order refused</div>
      ) : null}
    </div>
  );
}

/**
 * Why the engine is or is not trading, in one short block.
 *
 * The contradiction check is the point of this panel. A rejection reason that cites open
 * positions while the venue reports none is not a quiet edge case — it is the engine
 * refusing every candidate against state that no longer exists, and it is invisible
 * unless the two numbers are put side by side.
 */
export function TradingStatusPanel({
  summary,
  decisions,
}: {
  summary: Summary | null;
  decisions: DecisionsResponse | null;
}) {
  const status = summary?.status;
  const counters = summary?.decisions;
  const categories = counters?.by_rejection_category ?? {};
  const entries = Object.entries(categories).sort((a, b) => b[1] - a[1]);
  const leading = entries[0];
  const openPositions = summary?.venue?.position_count ?? null;
  const style = toneFor(status?.state);
  const recent = list(decisions?.decisions).slice(-1)[0];

  const contradicts =
    leading?.[0] === "correlation" && openPositions === 0 && leading[1] > 0;

  return (
    <Panel
      title="Trading status"
      source="engine decision log — every evaluated bar"
      subtitle={status?.detail}
    >
      <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
        <span className={`flex items-center gap-2 font-mono text-lg font-semibold ${style.text}`}>
          <span className={`h-2 w-2 rounded-full ${style.dot}`} />
          {status?.state ?? "…"}
        </span>
        <span className="text-xs text-zinc-400">
          {count(counters?.evaluated)} evaluated · {count(counters?.selected)} selected ·{" "}
          {count(counters?.declined)} declined
        </span>
        <span className="text-xs text-zinc-500">
          last decision {ago(recent?.timestamp ?? counters?.last_at)}
        </span>
      </div>

      {leading ? (
        <p className="mt-3 text-xs text-zinc-300">
          Main rejection reason:{" "}
          <span className="font-mono text-zinc-100">{leading[0]}</span>{" "}
          <span className="text-zinc-500">({leading[1]} of {count(counters?.declined)})</span>
        </p>
      ) : null}

      {contradicts ? (
        <div className="mt-3">
          <Contradiction>
            Every decline cites correlation with an open position, and the venue reports{" "}
            <span className="font-mono">0</span> open positions. The orchestrator is holding
            ownership state for symbols that have already closed at the venue, so the
            duplicate-position guard is firing against positions that do not exist.
          </Contradiction>
        </div>
      ) : null}
    </Panel>
  );
}

const PERIOD_LABELS: Record<string, string> = {
  TODAY: "Today",
  "7D": "7D",
  "30D": "30D",
  SESSION: "Session",
  ALL: "All time",
};

/** One profit-and-loss section, with the period selector and a single cumulative chart. */
export function PnlSection({
  pnl,
  period,
  onPeriod,
  chart,
}: {
  pnl: PnlResponse | null;
  period: string;
  onPeriod: (next: string) => void;
  chart: ReactNode;
}) {
  const order = list(pnl?.order);
  const periods = pnl?.periods ?? {};
  const current: PnlPeriod | undefined = periods[period];

  return (
    <Panel
      title="Profit & loss"
      source="quantflow database — reconciled closed trades"
      subtitle={current?.scope}
      action={
        <div className="flex gap-1">
          {(order.length ? order : Object.keys(periods)).map((entry) => (
            <button
              key={entry}
              type="button"
              onClick={() => {
                onPeriod(entry);
              }}
              className={`rounded px-2 py-0.5 text-[10px] uppercase tracking-wider ${
                period === entry
                  ? "bg-zinc-700 text-zinc-100"
                  : "bg-zinc-800/60 text-zinc-400 hover:bg-zinc-700"
              }`}
            >
              {PERIOD_LABELS[entry] ?? entry}
            </button>
          ))}
        </div>
      }
    >
      <div className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3 lg:grid-cols-6">
        <Tile
          label="Gross profit"
          value={signed(current?.gross_profit)}
          valueClass="text-[#0ca30c]"
          note="winners, net of their own fees"
        />
        <Tile
          label="Gross loss"
          value={`-${money(current?.gross_loss)}`}
          valueClass="text-[#d03b3b]"
          note="losers, net of their own fees"
        />
        <Tile label="Fees" value={`-${money(current?.fees)}`} valueClass="text-[#c98500]" />
        <Tile
          label="Net PnL"
          value={signed(current?.net_pnl)}
          valueClass={tone(current?.net_pnl)}
        />
        <Tile
          label="Win rate"
          value={percent(current?.win_rate, 1)}
          note={`${count(current?.win_count)}W / ${count(current?.loss_count)}L`}
        />
        <Tile
          label="Profit factor"
          value={ratio(current?.profit_factor)}
          note={current?.profit_factor ? undefined : "undefined — no losing trade"}
        />
      </div>

      {current?.sample_is_thin ? (
        <div className="mt-3">
          <Caution>
            Fewer than 10 closed trades in this period
            <InsufficientSample n={current.closed_trades} min={10} /> — these ratios are not
            yet meaningful.
          </Caution>
        </div>
      ) : null}

      <div className="mt-4">{chart}</div>
    </Panel>
  );
}

/** The compact health strip: four lights and the age of the newest reading. */
export function HealthStrip({
  summary,
  freshness,
  connected,
  onOpen,
}: {
  summary: Summary | null;
  freshness: FreshnessResponse | null;
  connected: boolean;
  onOpen: () => void;
}) {
  const venueOk = Boolean(summary?.venue?.authenticated) && !summary?.venue?.freshness?.stale;
  const dataOk = (freshness?.state ?? "") === "DATA FRESH";
  const reconciliationAt = freshness?.reconciliation?.last_venue_read_at;
  const databaseOk = Boolean(summary?.session?.session_id);

  const lights: { label: string; ok: boolean; detail: string }[] = [
    { label: "Venue", ok: venueOk, detail: summary?.venue?.network ?? "—" },
    { label: "Market data", ok: dataOk, detail: freshness?.state ?? "—" },
    { label: "Reconciliation", ok: Boolean(reconciliationAt), detail: ago(reconciliationAt) },
    { label: "Database", ok: databaseOk, detail: databaseOk ? "healthy" : "unknown" },
    { label: "Stream", ok: connected, detail: connected ? "websocket" : "polling" },
  ];

  return (
    <button
      type="button"
      onClick={onOpen}
      className="flex w-full flex-wrap items-center gap-x-6 gap-y-2 rounded-lg bg-zinc-900/60 px-4 py-2 text-left ring-1 ring-zinc-800 transition hover:ring-zinc-700"
    >
      {lights.map((light) => (
        <span key={light.label} className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider">
          <span className={`h-1.5 w-1.5 rounded-full ${light.ok ? "bg-[#0ca30c]" : "bg-[#d03b3b]"}`} />
          <span className="text-zinc-400">{light.label}</span>
          <span className="text-zinc-600">{light.detail}</span>
        </span>
      ))}
      <span className="ml-auto text-[10px] text-zinc-500">
        last update <span className="font-mono text-zinc-300">{ago(summary?.generated_at)}</span> ·
        diagnostics →
      </span>
    </button>
  );
}

/** The market-universe indicator: which classes are live, and what each is doing. */
export function UniverseStrip({
  classes,
  summary,
}: {
  classes: AssetClassesResponse | null;
  summary: Summary | null;
}) {
  const rows = list(classes?.asset_classes);
  const codes = list(summary?.engine?.agreement_codes);

  return (
    <Panel
      title="Market universe"
      source="the running engine's own startup log, plus live venue positions"
    >
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {rows.map((row) => {
          const active = (row.state ?? "").startsWith("ACTIVE");
          const blocked = (row.state ?? "").includes("BLOCKED");
          return (
            <div
              key={row.asset_class}
              className="rounded-lg bg-zinc-900/60 px-3 py-2 ring-1 ring-zinc-800"
              title={row.reason ?? undefined}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider text-zinc-200">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      blocked ? "bg-[#fab219]" : active ? "bg-[#0ca30c]" : "bg-zinc-600"
                    }`}
                  />
                  {row.asset_class}
                </span>
                <span className="text-[9px] uppercase tracking-wider text-zinc-500">
                  {row.state}
                </span>
              </div>
              <div className="mt-1 truncate font-mono text-[10px] text-zinc-400" title={list(row.symbols).join(", ")}>
                {list(row.symbols).length ? list(row.symbols).join(" · ") : "—"}
              </div>
              <div className="mt-0.5 text-[10px] text-zinc-600">
                {count(row.symbol_count)} symbols · {count(row.open_positions)} open
              </div>
            </div>
          );
        })}
      </div>

      {codes.length ? (
        <div className="mt-3">
          <Caution>
            Some classes are subscribed and evaluated but cannot place orders: the venue
            requires product agreements that this account has not signed (retCode{" "}
            <span className="font-mono">{codes.join(", ")}</span>). Sign them in the Bybit
            demo UI — no redeploy is needed, and the block clears on the next restart.
          </Caution>
        </div>
      ) : null}
    </Panel>
  );
}
