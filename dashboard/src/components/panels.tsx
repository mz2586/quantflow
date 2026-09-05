/**
 * The data panels.
 *
 * Each panel names its own source in its header. That is not decoration: the venue's view
 * of the account and QuantFlow's view of the session are different things that produce
 * similar-looking numbers, and the single most damaging thing this dashboard could do is
 * let one be read as the other.
 */

import { Fragment, useState } from "react";
import type {
  AnalyticsResponse,
  AssetClassesResponse,
  AttributionGroup,
  DecisionsResponse,
  FeeAnalysis,
  FreshnessResponse,
  LedgerResponse,
  LedgerTrade,
  Summary,
  VenueOrder,
  VenuePosition,
} from "../lib/api";
import { list } from "../lib/api";
import {
  NOT_RECORDED,
  absent,
  ago,
  clock,
  count,
  duration,
  money,
  percent,
  quantity,
  ratio,
  signed,
  time,
  tone,
} from "../lib/format";
import {
  Caution,
  Contradiction,
  Empty,
  FreshnessBadge,
  InsufficientSample,
  NotRecorded,
  Panel,
  Stat,
  Table,
  Unavailable,
  Value,
} from "./ui";
import { RejectionChart } from "./charts";

/* ------------------------------------------------------------------ venue account -- */

/**
 * The venue account.
 *
 * This panel replaces one that reported a "Total balance" of 99,904.01 — the arithmetic
 * sum of USDT, USDC, one bitcoin and one ether, added as though they shared a unit. The
 * figure had no unit, was roughly double the capital the engine could actually deploy, and
 * looked entirely plausible. Nothing here sums across assets without conversion.
 */
export function VenueAccountPanel({ summary }: { summary: Summary | null }) {
  const venue = summary?.venue;
  const account = venue?.account;
  const freshness = venue?.freshness;

  if (!venue || !account) {
    return (
      <Panel title="Venue account" source="bybit — read live from the exchange">
        <Unavailable
          what="Venue account"
          error={freshness?.error ?? "the API has not returned a venue reading"}
          lastSuccessAt={freshness?.fetched_at ?? null}
        />
      </Panel>
    );
  }

  const unpriced = list(account.unpriced_assets);
  const others = list(account.other_assets);

  return (
    <Panel
      title={`${(venue.venue ?? "venue").toUpperCase()} ${venue.network ?? ""} account`}
      subtitle="Authoritative for balances. USDT is what the engine sizes against; other assets are listed separately."
      source="bybit venue — live read"
      action={<FreshnessBadge freshness={freshness} label="venue" />}
    >
      {freshness?.stale ? (
        <div className="mb-3">
          <Unavailable
            what="Venue reading"
            error={freshness.error}
            lastSuccessAt={freshness.fetched_at}
          />
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat
          label="Trading equity (USDT)"
          value={money(account.trading_equity_usdt, "USDT")}
          emphasis
          hint="the only balance the engine sizes against"
        />
        <Stat
          label="Available USDT"
          value={money(account.available_usdt, "USDT")}
          emphasis
          hint={`${money(account.locked_usdt)} locked in working orders`}
        />
        <Stat
          label="Unrealised PnL (venue)"
          value={absent(venue.unrealized_pnl) ? NOT_RECORDED : signed(venue.unrealized_pnl)}
          valueClass={tone(venue.unrealized_pnl)}
          hint="marked to market by the venue"
        />
        <Stat
          label="Total portfolio value"
          value={
            account.total_portfolio_value_usdt
              ? money(account.total_portfolio_value_usdt, "USDT")
              : NOT_RECORDED
          }
          hint={
            account.total_portfolio_value_usdt
              ? `valued ${clock(account.valued_at)}`
              : `withheld — ${unpriced.join(", ")} could not be priced`
          }
        />
      </div>

      {account.valuation_method && account.total_portfolio_value_usdt ? (
        <p className="mt-2 text-[10px] text-zinc-600">method · {account.valuation_method}</p>
      ) : null}

      <h3 className="mb-2 mt-5 text-[10px] uppercase tracking-wider text-zinc-500">
        Other assets held ({others.length}) — quantities, valued only where a current price exists
      </h3>
      {others.length === 0 ? (
        <p className="text-xs text-zinc-500">No non-USDT holdings.</p>
      ) : (
        <Table head={["Asset", ">Quantity", ">Price (USDT)", ">Value (USDT)", "Valuation source"]}>
          {others.map((asset) => (
            <tr key={asset.asset} className="border-t border-zinc-800">
              <td className="py-1.5 pr-3 font-sans text-zinc-200">{asset.asset}</td>
              <td className="py-1.5 pr-3 text-right">{quantity(asset.quantity)}</td>
              <td className="py-1.5 pr-3 text-right">
                <Value value={asset.price_usdt} why={asset.unpriced_reason ?? undefined} />
              </td>
              <td className="py-1.5 pr-3 text-right">
                {asset.value_usdt ? (
                  money(asset.value_usdt)
                ) : (
                  <NotRecorded why={asset.unpriced_reason ?? undefined} />
                )}
              </td>
              <td className="py-1.5 pr-3 font-sans text-[10px] text-zinc-500">
                {asset.valuation_source ?? asset.unpriced_reason ?? "—"}
              </td>
            </tr>
          ))}
        </Table>
      )}
      <p className="mt-2 text-[10px] text-zinc-600">
        These holdings are <span className="text-zinc-400">not</span> added to trading equity.
        The engine trades USDT-quoted instruments and sizes from the USDT balance alone.
      </p>
    </Panel>
  );
}

/* --------------------------------------------------------------------- positions -- */

function positionState(position: VenuePosition): { label: string; className: string } {
  const pnl = Number.parseFloat(position.unrealized_pnl ?? "0");
  const protectedByStop = !absent(position.venue_stop_loss);
  const target = !absent(position.venue_take_profit);
  if (!protectedByStop) {
    return { label: "UNPROTECTED", className: "bg-[#d03b3b]/15 text-[#d03b3b]" };
  }
  if (pnl > 0 && target) {
    return { label: "PROFIT LOCKED", className: "bg-[#0ca30c]/15 text-[#0ca30c]" };
  }
  if (pnl > 0) return { label: "PROFITABLE", className: "bg-[#0ca30c]/15 text-[#0ca30c]" };
  if (pnl < 0) return { label: "LOSING", className: "bg-[#d03b3b]/15 text-[#d03b3b]" };
  return { label: "STOP PROTECTED", className: "bg-zinc-700/50 text-zinc-300" };
}

export function PositionsPanel({
  summary,
  assetFor,
  onSelect,
}: {
  summary: Summary | null;
  /** Asset class for a symbol, from the engine's own classification. */
  assetFor?: ((symbol: string) => string) | undefined;
  /** Opens the detail drawer. Omitted, rows are not interactive. */
  onSelect?: ((position: VenuePosition) => void) | undefined;
}) {
  const venue = summary?.venue;
  const positions = list(venue?.positions);
  const reconciliation = summary?.book_reconciliation;

  return (
    <Panel
      title={`Open positions (${count(venue?.position_count)})`}
      subtitle="Read from the venue on every refresh — the venue is authoritative for what is actually open."
      source="bybit venue — live read"
      action={<FreshnessBadge freshness={venue?.freshness} label="venue" />}
    >
      {venue?.position_error ? (
        <div className="mb-3">
          <Caution>{venue.position_error}</Caution>
        </div>
      ) : null}

      {reconciliation?.positions_match === false ? (
        <div className="mb-3">
          <Contradiction>
            The venue reports {count(reconciliation.venue_open_positions)} open position(s) but
            QuantFlow's own store has {count(reconciliation.database_open_positions)}. The venue
            is authoritative; the difference is a reconciliation gap in QuantFlow's record, not a
            second opinion about the account.
          </Contradiction>
        </div>
      ) : null}

      {positions.length === 0 ? (
        <Empty message="No open positions on the venue." />
      ) : (
        <Table
          head={[
            "Symbol",
            "Asset",
            "Side",
            "State",
            ">Qty",
            ">Entry",
            ">Mark",
            ">Notional",
            ">Unrealised",
            ">Stop",
            ">Target",
            ">Liq.",
            "Opened",
          ]}
        >
          {positions.map((position) => {
            const state = positionState(position);
            return (
              <tr
                key={position.symbol}
                onClick={onSelect ? () => { onSelect(position); } : undefined}
                className={`border-t border-zinc-800 ${
                  onSelect ? "cursor-pointer transition hover:bg-zinc-800/40" : ""
                }`}
              >
                <td className="py-1.5 pr-3 font-sans text-zinc-200">{position.symbol}</td>
                <td className="py-1.5 pr-3 font-sans text-[10px] uppercase tracking-wider text-zinc-500">
                  {assetFor ? assetFor(position.symbol) : "—"}
                </td>
                <td className="py-1.5 pr-3 font-sans uppercase text-zinc-400">
                  {position.side ?? "—"}
                </td>
                <td className="py-1.5 pr-3">
                  <span
                    className={`rounded px-1.5 py-px font-sans text-[9px] uppercase tracking-wider ${state.className}`}
                  >
                    {state.label}
                  </span>
                </td>
                <td className="py-1.5 pr-3 text-right">{quantity(position.quantity)}</td>
                <td className="py-1.5 pr-3 text-right">{money(position.entry_price)}</td>
                <td className="py-1.5 pr-3 text-right">{money(position.mark_price)}</td>
                <td className="py-1.5 pr-3 text-right">{money(position.notional_usdt)}</td>
                <td className={`py-1.5 pr-3 text-right ${tone(position.unrealized_pnl)}`}>
                  {signed(position.unrealized_pnl)}
                </td>
                <td className="py-1.5 pr-3 text-right">
                  {absent(position.venue_stop_loss) ? (
                    <span className="font-sans text-[10px] uppercase text-[#d03b3b]">none</span>
                  ) : (
                    money(position.venue_stop_loss)
                  )}
                </td>
                <td className="py-1.5 pr-3 text-right">
                  <Value value={position.venue_take_profit} why="no target attached at the venue" />
                </td>
                <td className="py-1.5 pr-3 text-right">
                  <Value value={position.liquidation_price} />
                </td>
                <td className="py-1.5 pr-3 font-sans text-[10px] text-zinc-500">
                  {time(position.opened_at)}
                </td>
              </tr>
            );
          })}
        </Table>
      )}
      <p className="mt-2 text-[10px] text-zinc-600">
        Profit stage, net-profit-exit eligibility and loser-exit state are{" "}
        <span className="uppercase">{NOT_RECORDED}</span> — the engine manages those in memory
        and persists no column for them. The state column above is derived only from the
        venue's own unrealised PnL and the stop/target it holds.
      </p>
    </Panel>
  );
}

/* ------------------------------------------------------------------------ orders -- */

export function OrdersPanel({ summary }: { summary: Summary | null }) {
  const venue = summary?.venue;
  const orders = list(venue?.open_orders);
  const reconciliation = summary?.book_reconciliation;

  return (
    <Panel
      title={`Working orders (${count(venue?.open_order_count)})`}
      subtitle="Status comes from the venue every refresh, so a filled or cancelled order never lingers as NEW."
      source="bybit venue — live read"
      action={<FreshnessBadge freshness={venue?.freshness} label="venue" />}
    >
      {reconciliation?.orders_match === false ? (
        <div className="mb-3">
          <Contradiction>
            The venue has {count(reconciliation.venue_open_orders)} working order(s); QuantFlow's
            order store has {count(reconciliation.database_open_orders)}. Protective stops and
            targets placed at the venue are not being written back to QuantFlow's own record.
          </Contradiction>
        </div>
      ) : null}

      {orders.length === 0 ? (
        <Empty message="No working orders on the venue." />
      ) : (
        <Table
          head={[
            "Symbol",
            "Purpose",
            "Side",
            "Type",
            "Status",
            ">Qty",
            ">Filled",
            ">Price",
            ">Trigger",
            "Flags",
            "Venue id",
            "Created",
          ]}
        >
          {orders.map((order: VenueOrder) => (
            <tr key={order.order_id} className="border-t border-zinc-800">
              <td className="py-1.5 pr-3 font-sans text-zinc-200">{order.symbol}</td>
              <td className="py-1.5 pr-3 font-sans">
                {order.purpose === "stop_loss" ? (
                  <span className="text-[#d03b3b]">stop</span>
                ) : order.purpose === "take_profit" ? (
                  <span className="text-[#0ca30c]">target</span>
                ) : (
                  <span className="text-zinc-500">entry</span>
                )}
              </td>
              <td className="py-1.5 pr-3 font-sans uppercase text-zinc-400">{order.side ?? "—"}</td>
              <td className="py-1.5 pr-3 font-sans text-zinc-400">{order.type ?? "—"}</td>
              <td className="py-1.5 pr-3 font-sans uppercase text-zinc-300">
                {order.status ?? "—"}
              </td>
              <td className="py-1.5 pr-3 text-right">{quantity(order.quantity)}</td>
              <td className="py-1.5 pr-3 text-right">{quantity(order.filled_quantity)}</td>
              <td className="py-1.5 pr-3 text-right">
                <Value value={order.price ? money(order.price) : null} why="market order" />
              </td>
              <td className="py-1.5 pr-3 text-right">
                <Value
                  value={order.trigger_price ? money(order.trigger_price) : null}
                  why="not a conditional order"
                />
              </td>
              <td className="py-1.5 pr-3 font-sans text-[10px] text-zinc-500">
                {order.reduce_only ? "reduce-only" : ""}
              </td>
              <td className="py-1.5 pr-3 text-[10px] text-zinc-500">
                {order.venue_order_id?.slice(0, 8) ?? "—"}
              </td>
              <td className="py-1.5 pr-3 font-sans text-[10px] text-zinc-500">
                {time(order.created_at)}
              </td>
            </tr>
          ))}
        </Table>
      )}
    </Panel>
  );
}

/* -------------------------------------------------------------------------- fees -- */

export function FeesPanel({ fees }: { fees?: FeeAnalysis | undefined }) {
  if (!fees) {
    return (
      <Panel title="Fees and execution cost" source="quantflow database — closed trades">
        <Empty message="No fee data yet." />
      </Panel>
    );
  }

  const ratioValue = Number.parseFloat(fees.fee_to_gross_ratio ?? "0");
  return (
    <Panel
      title="Fees and execution cost"
      subtitle="What trading cost, beside what it earned."
      source="quantflow database — reconciled closed trades"
    >
      {fees.fees_exceed_gross_profit ? (
        <div className="mb-3">
          <Contradiction>
            Fees of {money(fees.total_fees)} are {ratio(fees.fee_to_gross_ratio, 1)}× the gross
            profit of {money(fees.gross_realized_pnl)}. The strategy is gross-profitable and
            net-losing entirely on cost — a different problem from a strategy that is simply
            wrong about direction.
          </Contradiction>
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Total fees" value={money(fees.total_fees)} valueClass="text-[#c98500]" />
        <Stat
          label="Gross realised PnL"
          value={signed(fees.gross_realized_pnl)}
          valueClass={tone(fees.gross_realized_pnl)}
        />
        <Stat
          label="Net realised PnL"
          value={signed(fees.net_realized_pnl)}
          valueClass={tone(fees.net_realized_pnl)}
        />
        <Stat
          label="Fee / gross PnL"
          value={absent(fees.fee_to_gross_ratio) ? NOT_RECORDED : percent(fees.fee_to_gross_ratio, 0)}
          valueClass={ratioValue > 1 ? "text-[#d03b3b]" : ""}
        />
        <Stat label="Avg fee per trade" value={money(fees.average_fee_per_trade)} />
        <Stat
          label="Observed round-trip cost"
          value={percent(fees.average_fee_pct_of_notional, 4)}
          hint="actual fees ÷ entry notional, per closed round-trip"
        />
        <Stat label="Total entry notional" value={money(fees.total_entry_notional)} />
        <Stat label="Closed trades" value={count(fees.closed_trades)} />
      </div>

      <div className="mt-4 grid gap-2 sm:grid-cols-2">
        <div className="rounded border border-zinc-800 px-3 py-2">
          <div className="text-[10px] uppercase tracking-wider text-zinc-500">Entry fees</div>
          <div className="mt-1">
            <NotRecorded why={fees.not_recorded?.entry_fees} />
          </div>
        </div>
        <div className="rounded border border-zinc-800 px-3 py-2">
          <div className="text-[10px] uppercase tracking-wider text-zinc-500">Exit fees</div>
          <div className="mt-1">
            <NotRecorded why={fees.not_recorded?.exit_fees} />
          </div>
        </div>
      </div>
      <p className="mt-2 text-[10px] text-zinc-600">
        The schema stores one combined fee per round-trip, so the entry/exit split cannot be
        recovered. The observed round-trip cost above is measured from real fills; an
        estimated forward cost is not shown because the engine does not persist its cost
        model's output.
      </p>
    </Panel>
  );
}

/* ---------------------------------------------------------------------- decisions -- */

const CATEGORY_LABELS: Record<string, string> = {
  cost: "cost",
  correlation: "correlation",
  confluence: "confluence",
  risk_reward: "R:R",
  sizing: "sizing",
  liquidity: "liquidity",
  risk: "risk",
  regime: "regime",
  score_floor: "score floor",
  other: "other",
};

/**
 * The decision engine.
 *
 * The most important panel here. "No open positions" is compatible with an engine
 * correctly declining a bad market and with one that has silently stopped evaluating
 * anything, and only this panel can tell them apart.
 */
export function DecisionsPanel({
  decisions,
  openPositions,
}: {
  decisions: DecisionsResponse | null;
  /** The venue's open-position count, shown beside correlation rejections deliberately. */
  openPositions?: number | null | undefined;
}) {
  const rows = list(decisions?.decisions);
  const summary = decisions?.summary;
  const categories = summary?.by_rejection_category ?? {};

  // A rejection that blames correlation "with an open position" while no position is open
  // is self-contradictory. The dashboard's job is to make that visible, not to explain it
  // away — the contradiction is evidence of a bug in the engine's ownership tracking, and
  // an operator staring at a flat book deserves to see it stated.
  const correlationRejections = categories.correlation ?? 0;
  const contradiction =
    correlationRejections > 0 && (openPositions ?? 0) === 0 ? correlationRejections : 0;

  return (
    <Panel
      title="Decision engine — why the bot is or is not trading"
      subtitle={summary?.window}
      source={decisions?.source ?? "orchestrator structured log"}
      action={<FreshnessBadge freshness={decisions?.freshness} label="engine log" />}
    >
      {contradiction > 0 ? (
        <div className="mb-3">
          <Contradiction>
            {contradiction} of the recent rejections say a candidate correlates too highly{" "}
            <span className="italic">with an open position</span>, but the venue reports{" "}
            <span className="font-semibold">0 open positions</span>. Those two statements
            cannot both be true. This is consistent with stale ownership state inside the
            orchestrator rather than a genuine correlation block, and it means the engine may
            be declining trades for a reason that no longer exists. Surfaced here only —
            diagnosing and fixing the orchestrator is outside this dashboard's remit.
          </Contradiction>
        </div>
      ) : null}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Bars evaluated" value={count(summary?.evaluated)} />
        <Stat
          label="Selected"
          value={count(summary?.selected)}
          valueClass={(summary?.selected ?? 0) > 0 ? "text-[#0ca30c]" : ""}
        />
        <Stat label="Declined" value={count(summary?.declined)} />
        <Stat label="Open positions (venue)" value={count(openPositions ?? undefined)} />
      </div>

      <h3 className="mb-1 mt-5 text-[10px] uppercase tracking-wider text-zinc-500">
        Rejections by category
      </h3>
      <RejectionChart counts={categories} />

      <h3 className="mb-2 mt-5 text-[10px] uppercase tracking-wider text-zinc-500">
        Recent evaluations ({rows.length})
      </h3>
      {rows.length === 0 ? (
        <Empty message="No decisions found in the engine log tail." />
      ) : (
        <Table
          head={[
            "Time",
            "Symbol",
            "Decision",
            "Strategy",
            "Dir",
            ">Score",
            ">Conf",
            ">Cand",
            "Regime",
            "Category",
            "Reason",
          ]}
        >
          {rows.map((row, index) => (
            <tr key={`${row.timestamp}-${index}`} className="border-t border-zinc-800 align-top">
              <td className="py-1.5 pr-3 font-sans text-[10px] text-zinc-500">
                {time(row.timestamp)}
              </td>
              <td className="py-1.5 pr-3 font-sans text-zinc-200">{row.symbol ?? "—"}</td>
              <td className="py-1.5 pr-3">
                <span
                  className={`rounded px-1.5 py-px font-sans text-[9px] uppercase tracking-wider ${
                    row.outcome === "SELECTED"
                      ? "bg-[#0ca30c]/15 text-[#0ca30c]"
                      : row.outcome === "RISK_BLOCKED"
                        ? "bg-[#d03b3b]/15 text-[#d03b3b]"
                        : "bg-zinc-700/50 text-zinc-300"
                  }`}
                >
                  {row.outcome ?? "—"}
                </span>
              </td>
              <td className="py-1.5 pr-3 font-sans text-zinc-300">
                <Value value={row.strategy} why="the engine logs no strategy for this event" />
              </td>
              <td className="py-1.5 pr-3 font-sans uppercase text-zinc-400">
                <Value value={row.direction} />
              </td>
              <td className="py-1.5 pr-3 text-right">
                <Value value={row.score} />
              </td>
              <td className="py-1.5 pr-3 text-right">
                <Value value={row.confidence} />
              </td>
              <td className="py-1.5 pr-3 text-right">
                <Value value={row.candidates} />
              </td>
              <td className="py-1.5 pr-3 font-sans text-zinc-400">
                <Value value={row.regime} why="regime is not logged on this event" />
              </td>
              <td className="py-1.5 pr-3 font-sans text-[10px] text-[#c98500]">
                {row.rejection_category
                  ? (CATEGORY_LABELS[row.rejection_category] ?? row.rejection_category)
                  : "—"}
              </td>
              <td className="max-w-[22rem] py-1.5 pr-3 font-sans text-[10px] text-zinc-400">
                <Value value={row.reason} why="no reason logged for a successful selection" />
              </td>
            </tr>
          ))}
        </Table>
      )}

      <div className="mt-3 space-y-1">
        <Caution>
          Expected edge and estimated cost are{" "}
          <span className="uppercase">{NOT_RECORDED}</span> in absolute terms. The orchestrator
          logs unit-free component scores (confidence, risk/reward, regime, evidence, cost,
          correlation) between 0 and 1 — the <span className="font-mono">cost</span> score is a
          weighting, not a cost in USDT, and is not shown as one.
        </Caution>
      </div>
    </Panel>
  );
}

/* ---------------------------------------------------------------------- analytics -- */

function AttributionTable({
  groups,
  keyLabel,
  minSample,
  emptyMessage,
}: {
  groups: readonly AttributionGroup[];
  keyLabel: string;
  minSample?: number | undefined;
  emptyMessage: string;
}) {
  if (groups.length === 0) return <Empty message={emptyMessage} />;
  return (
    <Table
      head={[
        keyLabel,
        ">Trades",
        ">Net PnL",
        ">Gross",
        ">Fees",
        ">Win rate",
        ">Profit factor",
        ">Avg",
        ">Best",
        ">Worst",
      ]}
    >
      {groups.map((group, index) => (
        <tr key={`${group.key ?? "unattributed"}-${index}`} className="border-t border-zinc-800">
          <td className="py-1.5 pr-3 font-sans">
            {group.key ?? (
              <span className="text-zinc-500 italic">unattributed</span>
            )}
            {group.reliable === false ? (
              <InsufficientSample n={group.trades} min={minSample} />
            ) : null}
          </td>
          <td className="py-1.5 pr-3 text-right">{count(group.trades)}</td>
          <td className={`py-1.5 pr-3 text-right ${tone(group.net_pnl)}`}>
            {signed(group.net_pnl)}
          </td>
          <td className={`py-1.5 pr-3 text-right ${tone(group.gross_pnl)}`}>
            {signed(group.gross_pnl)}
          </td>
          <td className="py-1.5 pr-3 text-right text-[#c98500]">{money(group.fees)}</td>
          <td className="py-1.5 pr-3 text-right">{percent(group.win_rate, 1)}</td>
          <td className="py-1.5 pr-3 text-right">
            <Value value={ratio(group.profit_factor)} why="no losing trade in this group" />
          </td>
          <td className={`py-1.5 pr-3 text-right ${tone(group.average_net_pnl)}`}>
            {signed(group.average_net_pnl)}
          </td>
          <td className="py-1.5 pr-3 text-right text-[#0ca30c]">{signed(group.best)}</td>
          <td className="py-1.5 pr-3 text-right text-[#d03b3b]">{signed(group.worst)}</td>
        </tr>
      ))}
    </Table>
  );
}

export function AnalyticsPanels({ analytics }: { analytics: AnalyticsResponse | null }) {
  const [assetClass, setAssetClass] = useState<string>("all");
  const symbols = list(analytics?.by_symbol);
  const classes = ["all", ...new Set(symbols.map((row) => row.asset_class ?? "other"))];
  const filtered =
    assetClass === "all" ? symbols : symbols.filter((row) => row.asset_class === assetClass);

  return (
    <>
      <Panel
        title="Strategy analytics"
        subtitle="Realised performance attributed to the strategy that opened each trade."
        source="quantflow database — reconciled closed trades"
      >
        {analytics?.strategy_attribution_note ? (
          <div className="mb-3">
            <Caution>{analytics.strategy_attribution_note}</Caution>
          </div>
        ) : null}
        <AttributionTable
          groups={list(analytics?.by_strategy)}
          keyLabel="Strategy"
          minSample={analytics?.min_sample}
          emptyMessage="No closed trades to attribute."
        />
      </Panel>

      <Panel
        title="Symbol analytics"
        source="quantflow database — reconciled closed trades"
        action={
          <select
            value={assetClass}
            onChange={(event) => {
              setAssetClass(event.target.value);
            }}
            className="rounded border border-zinc-700 bg-zinc-800 px-2 py-1 text-[11px] text-zinc-200"
          >
            {classes.map((entry) => (
              <option key={entry} value={entry}>
                {entry}
              </option>
            ))}
          </select>
        }
      >
        <AttributionTable
          groups={filtered}
          keyLabel="Symbol"
          minSample={analytics?.min_sample}
          emptyMessage="No closed trades for this asset class."
        />
      </Panel>

      <Panel
        title="Long / short analytics"
        source="quantflow database — reconciled closed trades"
      >
        <AttributionTable
          groups={list(analytics?.by_side)}
          keyLabel="Direction"
          minSample={analytics?.min_sample}
          emptyMessage="No closed trades yet."
        />
      </Panel>

      <Panel title="Exit analytics" source="quantflow database — reconciled closed trades">
        {analytics?.exit_reason_available ? (
          <AttributionTable
            groups={list(analytics.by_exit_reason)}
            keyLabel="Exit reason"
            minSample={analytics.min_sample}
            emptyMessage="No exit reasons recorded."
          />
        ) : (
          <Caution>
            <span className="font-semibold uppercase">{NOT_RECORDED}</span> —{" "}
            {analytics?.exit_reason_note ??
              "the engine records no exit reason on a closed trade, so this analysis cannot be produced."}
          </Caution>
        )}
      </Panel>
    </>
  );
}

/* -------------------------------------------------------------------- trade ledger -- */

function TradeDetail({
  trade,
  notRecorded,
}: {
  trade: LedgerTrade;
  notRecorded?: Record<string, string> | undefined;
}) {
  return (
    <tr className="border-t border-zinc-800 bg-zinc-950/60">
      <td colSpan={16} className="px-4 py-3">
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <div className="mb-1 text-[10px] uppercase tracking-wider text-zinc-500">
              Lifecycle
            </div>
            <dl className="space-y-1 text-[11px]">
              <Detail label="Trade id" value={trade.trade_id} />
              <Detail label="Entered" value={time(trade.entry_time)} />
              <Detail label="Exited" value={time(trade.exit_time)} />
              <Detail label="Held" value={duration(trade.holding_seconds)} />
              <Detail label="Strategy" value={trade.strategy_id} why="written by venue reconciliation, which records no strategy" />
              <Detail label="Regime" value={trade.regime} why="regime is recorded as unknown for every trade in this session" />
            </dl>
          </div>
          <div>
            <div className="mb-1 text-[10px] uppercase tracking-wider text-zinc-500">
              Economics
            </div>
            <dl className="space-y-1 text-[11px]">
              <Detail label="Entry notional" value={money(trade.entry_notional)} />
              <Detail label="Gross PnL" value={signed(trade.gross_pnl)} />
              <Detail label="Total fees" value={money(trade.total_fees)} />
              <Detail label="Net PnL" value={signed(trade.net_pnl)} />
              <Detail label="Return" value={percent(trade.return_pct, 4)} />
              <Detail label="Fee ÷ gross" value={trade.fee_share_of_gross ? percent(trade.fee_share_of_gross, 1) : null} why="gross PnL was exactly zero" />
            </dl>
          </div>
          <div>
            <div className="mb-1 text-[10px] uppercase tracking-wider text-zinc-500">
              Not recorded by the engine
            </div>
            <dl className="space-y-1 text-[11px]">
              <Detail label="Exit reason" value={null} why={notRecorded?.exit_reason} />
              <Detail label="MFE" value={null} why={notRecorded?.mfe} />
              <Detail label="MAE" value={null} why={notRecorded?.mae} />
              <Detail label="Slippage" value={null} why="no slippage column exists on any table" />
              <Detail label="Order ids" value={null} why={notRecorded?.order_ids} />
              <Detail label="Position id" value={null} why={notRecorded?.position_id} />
              <Detail label="Venue fill ids" value={null} why={notRecorded?.venue_fill_ids} />
              <Detail label="Partial fills / stop changes" value={null} why="fills link to orders, and closed trades hold no link to their orders" />
            </dl>
          </div>
        </div>
      </td>
    </tr>
  );
}

function Detail({
  label,
  value,
  why,
}: {
  label: string;
  value?: string | null | undefined;
  why?: string | undefined;
}) {
  return (
    <div className="flex justify-between gap-3">
      <dt className="text-zinc-500">{label}</dt>
      <dd className="text-right font-mono text-zinc-200">
        <Value value={value} why={why} />
      </dd>
    </div>
  );
}

export function TradeLedgerPanel({ ledger }: { ledger: LedgerResponse | null }) {
  const [open, setOpen] = useState<string | null>(null);
  const trades = list(ledger?.trades);

  return (
    <Panel
      title={`Trade ledger (${count(ledger?.total)} closed)`}
      subtitle="Every closed round-trip, newest first. Click a row for its detail."
      source="quantflow database — reconciled closed trades"
    >
      {trades.length === 0 ? (
        <Empty message="No closed trades in this session yet." />
      ) : (
        <Table
          head={[
            "#",
            "Symbol",
            "Class",
            "Side",
            "Entered",
            "Exited",
            ">Qty",
            ">Entry",
            ">Exit",
            ">Held",
            ">Gross",
            ">Fees",
            ">Net",
            ">Net %",
            "Exit reason",
            "Strategy",
          ]}
        >
          {trades.map((trade) => (
            // Keyed on the Fragment, not the row: a bare <> in a list has no identity, so
            // React re-creates every row on each poll instead of reconciling them — which
            // also drops the open detail drawer on every refresh.
            <Fragment key={trade.trade_id}>
              <tr
                onClick={() => {
                  setOpen(open === trade.trade_id ? null : trade.trade_id);
                }}
                className="cursor-pointer border-t border-zinc-800 hover:bg-zinc-800/40"
              >
                <td className="py-1.5 pr-3 text-zinc-500">{count(trade.trade_number)}</td>
                <td className="py-1.5 pr-3 font-sans text-zinc-200">{trade.symbol}</td>
                <td className="py-1.5 pr-3 font-sans text-[10px] text-zinc-500">
                  {trade.asset_class}
                </td>
                <td className="py-1.5 pr-3 font-sans uppercase text-zinc-400">{trade.side}</td>
                <td className="py-1.5 pr-3 font-sans text-[10px] text-zinc-500">
                  {time(trade.entry_time)}
                </td>
                <td className="py-1.5 pr-3 font-sans text-[10px] text-zinc-500">
                  {time(trade.exit_time)}
                </td>
                <td className="py-1.5 pr-3 text-right">{quantity(trade.quantity)}</td>
                <td className="py-1.5 pr-3 text-right">{money(trade.entry_price)}</td>
                <td className="py-1.5 pr-3 text-right">{money(trade.exit_price)}</td>
                <td className="py-1.5 pr-3 text-right text-zinc-400">
                  {duration(trade.holding_seconds)}
                </td>
                <td className={`py-1.5 pr-3 text-right ${tone(trade.gross_pnl)}`}>
                  {signed(trade.gross_pnl)}
                </td>
                <td className="py-1.5 pr-3 text-right text-[#c98500]">{money(trade.total_fees)}</td>
                <td className={`py-1.5 pr-3 text-right font-semibold ${tone(trade.net_pnl)}`}>
                  {signed(trade.net_pnl)}
                </td>
                <td className={`py-1.5 pr-3 text-right ${tone(trade.return_pct)}`}>
                  {percent(trade.return_pct, 3)}
                </td>
                <td className="py-1.5 pr-3">
                  <NotRecorded why={ledger?.not_recorded?.exit_reason} />
                </td>
                <td className="py-1.5 pr-3 font-sans text-[10px] text-zinc-400">
                  <Value
                    value={trade.strategy_id}
                    why="this trade was written by venue reconciliation, which records no strategy"
                  />
                </td>
              </tr>
              {open === trade.trade_id ? (
                <TradeDetail trade={trade} notRecorded={ledger?.not_recorded} />
              ) : null}
            </Fragment>
          ))}
        </Table>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------- asset classes -- */

export function AssetClassesPanel({ classes }: { classes: AssetClassesResponse | null }) {
  const rows = list(classes?.asset_classes);
  return (
    <Panel
      title="Asset classes"
      subtitle="ACTIVE means the running engine is subscribed to this class and evaluating it — nothing else."
      source={classes?.source ?? "running engine startup log + venue positions"}
    >
      {rows.length === 0 ? (
        <Empty message="No asset classes reported." />
      ) : (
        <Table head={["Class", "State", ">Symbols", ">Open positions", "Data", "Detail"]}>
          {rows.map((row) => {
            const state = row.state ?? "UNKNOWN";
            const style =
              state === "ACTIVE"
                ? "bg-[#0ca30c]/15 text-[#0ca30c]"
                : state.includes("BLOCKED")
                  ? "bg-[#d03b3b]/15 text-[#d03b3b]"
                  : state.includes("NOT WIRED")
                    ? "bg-zinc-700/40 text-zinc-500"
                    : "bg-[#fab219]/10 text-[#fab219]";
            return (
              <tr key={row.asset_class} className="border-t border-zinc-800 align-top">
                <td className="py-1.5 pr-3 font-sans uppercase text-zinc-200">
                  {row.asset_class}
                  <div className="text-[10px] normal-case text-zinc-600">{row.description}</div>
                </td>
                <td className="py-1.5 pr-3">
                  <span
                    className={`whitespace-nowrap rounded px-1.5 py-px font-sans text-[9px] uppercase tracking-wider ${style}`}
                  >
                    {state}
                  </span>
                </td>
                <td className="py-1.5 pr-3 text-right">{count(row.symbol_count)}</td>
                <td className="py-1.5 pr-3 text-right">{count(row.open_positions)}</td>
                <td className="py-1.5 pr-3 font-sans text-[10px]">
                  {row.data_live ? (
                    <span className="text-[#0ca30c]">live</span>
                  ) : (
                    <span className="text-zinc-600">none</span>
                  )}
                </td>
                <td className="max-w-[22rem] py-1.5 pr-3 font-sans text-[10px] text-zinc-500">
                  {list(row.symbols).join(", ") || (row.reason ?? "—")}
                </td>
              </tr>
            );
          })}
        </Table>
      )}
    </Panel>
  );
}

/* ---------------------------------------------------------------------- freshness -- */

export function FreshnessPanel({ freshness }: { freshness: FreshnessResponse | null }) {
  const state = freshness?.state ?? "UNKNOWN";
  const colour =
    state === "DATA FRESH"
      ? "text-[#0ca30c]"
      : state === "DATA STALE"
        ? "text-[#fab219]"
        : "text-[#d03b3b]";

  return (
    <Panel title="Data freshness" source="API observations of each upstream source">
      <div className={`mb-3 font-mono text-lg ${colour}`}>{state}</div>
      <dl className="space-y-1.5 text-[11px]">
        <Detail label="Last venue sync" value={ago(freshness?.venue_sync?.fetched_at)} />
        <Detail label="Last reconciliation read" value={ago(freshness?.reconciliation?.last_venue_read_at)} />
        <Detail label="Last engine decision" value={ago(freshness?.last_decision_at)} />
        <Detail label="Last equity snapshot" value={ago(freshness?.last_equity_snapshot_at)} />
        <Detail label="Last order recorded" value={ago(freshness?.last_order_at)} />
        <Detail label="Last stored candle" value={ago(freshness?.last_candle_at)} />
      </dl>
      {freshness?.candle_note ? (
        <p className="mt-3 text-[10px] leading-relaxed text-zinc-500">{freshness.candle_note}</p>
      ) : null}
      {list(freshness?.candles).length > 0 ? (
        <>
          <h3 className="mb-1 mt-4 text-[10px] uppercase tracking-wider text-zinc-500">
            Stored candle archive per symbol ({freshness?.timeframe})
          </h3>
          <Table head={["Symbol", ">Last bar"]}>
            {list(freshness?.candles).map((row) => (
              <tr key={row.symbol} className="border-t border-zinc-800">
                <td className="py-1 pr-3 font-sans text-zinc-300">{row.symbol}</td>
                <td className="py-1 pr-3 text-right text-zinc-400">{ago(row.last_open_time)}</td>
              </tr>
            ))}
          </Table>
        </>
      ) : null}
      {freshness?.reconciliation?.note ? (
        <p className="mt-2 text-[10px] text-zinc-600">{freshness.reconciliation.note}</p>
      ) : null}
    </Panel>
  );
}
