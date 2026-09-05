/**
 * The charts.
 *
 * Two rules run through all of them.
 *
 * **Nothing is drawn that was not measured.** Missing points are never interpolated and
 * gaps are never filled: a fabricated point on an equity curve is a fabricated account
 * balance. Where the stored history begins is stated under the chart rather than implied
 * by where the line starts.
 *
 * **One axis per chart.** Two measures on two scales in one frame is the single most
 * misleading thing a chart can do — the crossing point is an artefact of the scales, not
 * of the data. Money and percentages therefore live in separate charts.
 */

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { CumulativePoint, EquityPoint, LedgerTrade } from "../lib/api";
import { list } from "../lib/api";
import { chartTime, chartValue, money, signed, time } from "../lib/format";
import { COLORS, Empty } from "./ui";

const AXIS = { fill: COLORS.axis, fontSize: 10 } as const;

const TOOLTIP_STYLE = {
  background: "#09090b",
  border: "1px solid #3f3f46",
  borderRadius: 6,
  fontSize: 11,
} as const;

function timeAxisProps(): Record<string, unknown> {
  return {
    dataKey: "t",
    type: "number",
    scale: "time",
    domain: ["dataMin", "dataMax"],
    tick: AXIS,
    tickFormatter: (value: number) => time(new Date(value).toISOString()),
    minTickGap: 48,
  };
}

/** Equity over time, with the running peak drawn behind it. */
export function EquityChart({
  points,
  historyNote,
}: {
  points: readonly EquityPoint[];
  historyNote?: string | undefined;
}) {
  const data = list(points)
    .map((point) => ({
      t: chartTime(point.timestamp),
      equity: chartValue(point.equity),
      peak: chartValue(point.running_peak),
    }))
    .filter((row) => row.t > 0);

  if (data.length === 0) return <Empty message="No equity snapshots in this range." />;

  return (
    <>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={COLORS.grid} vertical={false} />
          <XAxis {...timeAxisProps()} />
          <YAxis domain={["auto", "auto"]} tick={AXIS} width={78} tickFormatter={(v) => money(String(v))} />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={(value) => time(new Date(Number(value)).toISOString())}
            formatter={(value: number, name: string) => [money(String(value)), name]}
          />
          <Legend wrapperStyle={{ fontSize: 11, color: COLORS.axis }} />
          <Line
            type="monotone"
            dataKey="peak"
            name="running peak"
            stroke={COLORS.axis}
            strokeDasharray="3 3"
            dot={false}
            strokeWidth={1}
          />
          <Line
            type="monotone"
            dataKey="equity"
            name="equity"
            stroke={COLORS.series1}
            dot={false}
            strokeWidth={2}
          />
        </LineChart>
      </ResponsiveContainer>
      {historyNote ? <p className="mt-1 text-[10px] text-zinc-600">{historyNote}</p> : null}
    </>
  );
}

/** Drawdown from the running peak, as a percentage. Its own chart, never a second axis. */
export function DrawdownChart({
  points,
  largest,
}: {
  points: readonly EquityPoint[];
  largest?: { at?: string; depth?: string } | undefined;
}) {
  const data = list(points)
    .map((point) => ({
      t: chartTime(point.timestamp),
      // Negative so the chart reads downward, which is what a drawdown does.
      dd: -chartValue(point.drawdown_pct) * 100,
    }))
    .filter((row) => row.t > 0);

  if (data.length === 0) return <Empty message="No equity snapshots in this range." />;

  // `data[0]` is `T | undefined` under noUncheckedIndexedAccess even though the empty case
  // returned above, so the seed is made explicit rather than asserted away.
  const trough = data.reduce(
    (worst, row) => (row.dd < worst.dd ? row : worst),
    { t: 0, dd: 0 },
  );

  return (
    <>
      <ResponsiveContainer width="100%" height={160}>
        <AreaChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={COLORS.grid} vertical={false} />
          <XAxis {...timeAxisProps()} />
          <YAxis
            tick={AXIS}
            width={56}
            tickFormatter={(value: number) => `${value.toFixed(2)}%`}
            domain={["dataMin", 0]}
          />
          <Tooltip
            contentStyle={TOOLTIP_STYLE}
            labelFormatter={(value) => time(new Date(Number(value)).toISOString())}
            formatter={(value: number) => [`${Math.abs(value).toFixed(3)}%`, "drawdown"]}
          />
          <Area
            type="monotone"
            dataKey="dd"
            name="drawdown"
            stroke={COLORS.series8}
            fill={`${COLORS.series8}22`}
            strokeWidth={2}
          />
          {/* The deepest point is called out directly rather than left to be hunted for. */}
          <ReferenceLine
            x={trough.t}
            stroke={COLORS.critical}
            strokeDasharray="2 2"
            label={{
              value: `worst ${Math.abs(trough.dd).toFixed(2)}%`,
              fill: COLORS.critical,
              fontSize: 10,
              position: "insideBottomRight",
            }}
          />
        </AreaChart>
      </ResponsiveContainer>
      {largest?.at ? (
        <p className="mt-1 text-[10px] text-zinc-600">
          largest drawdown in range {largest.depth} at {time(largest.at)}
        </p>
      ) : null}
    </>
  );
}

export type PnlMode = "net" | "gross" | "fees";

/**
 * Cumulative realised PnL from the closed-trade ledger.
 *
 * Net is the default because net is what the account actually keeps. Gross and fees are
 * selectable rather than overlaid on a second axis, and the three are all money on one
 * scale so they can share a frame honestly when all are shown.
 */
export function PnlChart({ trades, mode }: { trades: readonly LedgerTrade[]; mode: PnlMode }) {
  const ordered = list(trades)
    .slice()
    .sort((a, b) => (a.exit_time ?? "").localeCompare(b.exit_time ?? ""));

  let net = 0;
  let gross = 0;
  let fees = 0;
  const data = ordered
    .map((trade) => {
      net += chartValue(trade.net_pnl);
      gross += chartValue(trade.gross_pnl);
      fees += chartValue(trade.total_fees);
      return { t: chartTime(trade.exit_time), net, gross, fees: -fees };
    })
    .filter((row) => row.t > 0);

  if (data.length === 0) return <Empty message="No closed trades yet." />;

  const series =
    mode === "net"
      ? [{ key: "net", name: "cumulative net", colour: COLORS.series1 }]
      : mode === "gross"
        ? [
            { key: "gross", name: "cumulative gross", colour: COLORS.series3 },
            { key: "net", name: "cumulative net", colour: COLORS.series1 },
          ]
        : [
            { key: "gross", name: "cumulative gross", colour: COLORS.series3 },
            { key: "fees", name: "cumulative fees (negative)", colour: COLORS.series4 },
            { key: "net", name: "cumulative net", colour: COLORS.series1 },
          ];

  return (
    <ResponsiveContainer width="100%" height={240}>
      <LineChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        <XAxis {...timeAxisProps()} />
        <YAxis tick={AXIS} width={70} tickFormatter={(v: number) => signed(v.toString(), 0)} />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          labelFormatter={(value) => time(new Date(Number(value)).toISOString())}
          formatter={(value: number, name: string) => [signed(value.toString()), name]}
        />
        <Legend wrapperStyle={{ fontSize: 11, color: COLORS.axis }} />
        <ReferenceLine y={0} stroke={COLORS.axis} strokeWidth={1} />
        {series.map((entry) => (
          <Line
            key={entry.key}
            type="monotone"
            dataKey={entry.key}
            name={entry.name}
            stroke={entry.colour}
            dot={false}
            strokeWidth={2}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/**
 * The single cumulative profit-and-loss line on the first screen.
 *
 * Accumulated by the database rather than in the browser, so the chart on a session with
 * thousands of trades costs one bounded query instead of shipping the whole ledger to the
 * client and summing it there. Gross and fees are drawn alongside net because the gap
 * between the first and the last *is* the fee story, and on this account that gap is
 * larger than the edge.
 */
export function CumulativePnlChart({ points }: { points: readonly CumulativePoint[] }) {
  const data = list(points)
    .map((point) => ({
      t: chartTime(point.at),
      net: chartValue(point.cumulative_net),
      gross: chartValue(point.cumulative_gross),
      fees: -chartValue(point.cumulative_fees),
    }))
    .filter((row) => row.t > 0);

  if (data.length === 0) return <Empty message="No closed trades in this window." />;

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        <XAxis {...timeAxisProps()} />
        <YAxis tick={AXIS} width={70} tickFormatter={(v: number) => signed(v.toString(), 0)} />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          labelFormatter={(value) => time(new Date(Number(value)).toISOString())}
          formatter={(value: number, name: string) => [signed(value.toString()), name]}
        />
        <Legend wrapperStyle={{ fontSize: 11, color: COLORS.axis }} />
        <ReferenceLine y={0} stroke={COLORS.axis} strokeWidth={1} />
        <Line type="monotone" dataKey="gross" name="cumulative gross" stroke={COLORS.series3} dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="fees" name="cumulative fees (negative)" stroke={COLORS.series4} dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="net" name="cumulative net" stroke={COLORS.series1} dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Cumulative net PnL split by direction, so a one-sided edge is visible. */
export function LongShortChart({ trades }: { trades: readonly LedgerTrade[] }) {
  const ordered = list(trades)
    .slice()
    .sort((a, b) => (a.exit_time ?? "").localeCompare(b.exit_time ?? ""));

  let long = 0;
  let short = 0;
  const data = ordered
    .map((trade) => {
      const value = chartValue(trade.net_pnl);
      if (String(trade.side).toLowerCase() === "long") long += value;
      else short += value;
      return { t: chartTime(trade.exit_time), long, short };
    })
    .filter((row) => row.t > 0);

  if (data.length === 0) return <Empty message="No closed trades yet." />;

  return (
    <ResponsiveContainer width="100%" height={200}>
      <LineChart data={data} margin={{ top: 4, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={COLORS.grid} vertical={false} />
        <XAxis {...timeAxisProps()} />
        <YAxis tick={AXIS} width={70} tickFormatter={(v: number) => signed(v.toString(), 0)} />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          labelFormatter={(value) => time(new Date(Number(value)).toISOString())}
          formatter={(value: number, name: string) => [signed(value.toString()), name]}
        />
        <Legend wrapperStyle={{ fontSize: 11, color: COLORS.axis }} />
        <ReferenceLine y={0} stroke={COLORS.axis} strokeWidth={1} />
        <Line type="monotone" dataKey="long" name="long" stroke={COLORS.series1} dot={false} strokeWidth={2} />
        <Line type="monotone" dataKey="short" name="short" stroke={COLORS.series2} dot={false} strokeWidth={2} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Rejection reasons as a ranked bar chart — the shape of why the engine is not trading. */
export function RejectionChart({ counts }: { counts: Record<string, number> }) {
  const data = Object.entries(counts)
    .map(([category, value]) => ({ category, value }))
    .sort((a, b) => b.value - a.value);

  if (data.length === 0) return <Empty message="No rejections recorded in this window." />;

  return (
    <ResponsiveContainer width="100%" height={Math.max(120, data.length * 30)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 40, bottom: 0, left: 0 }}>
        <CartesianGrid stroke={COLORS.grid} horizontal={false} />
        <XAxis type="number" tick={AXIS} allowDecimals={false} />
        <YAxis type="category" dataKey="category" tick={AXIS} width={110} />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          formatter={(value: number) => [String(value), "rejections"]}
          cursor={{ fill: "#ffffff08" }}
        />
        <Bar dataKey="value" fill={COLORS.series4} radius={[0, 4, 4, 0]} barSize={14} />
      </BarChart>
    </ResponsiveContainer>
  );
}
