/**
 * Shared presentation primitives.
 *
 * The rule these encode: **a number never appears without its provenance.** Every panel
 * can say where its data came from and when, and every value that the engine does not
 * record renders as `NOT RECORDED` rather than as a zero.
 */

import type { ReactNode } from "react";
import { ErrorBoundary } from "./ErrorBoundary";
import type { Freshness } from "../lib/api";
import { NOT_RECORDED, absent, ago, clock } from "../lib/format";

/** Validated chart/series colours. See `references/palette.md`; checked with the validator. */
export const COLORS = {
  series1: "#3987e5",
  series2: "#d95926",
  series3: "#199e70",
  series4: "#c98500",
  series8: "#e66767",
  good: "#0ca30c",
  warning: "#fab219",
  serious: "#ec835a",
  critical: "#d03b3b",
  grid: "#27272a",
  axis: "#71717a",
  surface: "#18181b",
} as const;

export function Panel({
  title,
  subtitle,
  children,
  action,
  source,
  className = "",
}: {
  title: string;
  subtitle?: string | undefined;
  children: ReactNode;
  action?: ReactNode | undefined;
  /** Where this panel's data comes from. Shown so two panels can never be confused. */
  source?: string | undefined;
  className?: string;
}) {
  return (
    <section className={`rounded-lg border border-zinc-800 bg-zinc-900/60 ${className}`}>
      <header className="flex flex-wrap items-start justify-between gap-2 border-b border-zinc-800 px-4 py-2.5">
        <div className="min-w-0">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-zinc-300">{title}</h2>
          {subtitle ? <p className="mt-0.5 text-[11px] text-zinc-500">{subtitle}</p> : null}
          {source ? (
            <p className="mt-0.5 text-[10px] uppercase tracking-wider text-zinc-600">
              source · {source}
            </p>
          ) : null}
        </div>
        {action}
      </header>
      {/* Per-panel containment: a malformed payload costs this panel, not the page. */}
      <div className="p-4">
        <ErrorBoundary label={title}>{children}</ErrorBoundary>
      </div>
    </section>
  );
}

export function Stat({
  label,
  value,
  valueClass = "",
  hint,
  emphasis = false,
}: {
  label: string;
  value: string;
  valueClass?: string | undefined;
  hint?: string | undefined;
  emphasis?: boolean | undefined;
}) {
  const missing = value === NOT_RECORDED;
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</div>
      <div
        className={`mt-0.5 font-mono tabular-nums ${emphasis ? "text-2xl" : "text-lg"} ${
          missing ? "text-[11px] font-sans uppercase tracking-wider text-zinc-600" : valueClass
        }`}
      >
        {value}
      </div>
      {hint ? <div className="mt-0.5 text-[10px] text-zinc-600">{hint}</div> : null}
    </div>
  );
}

/** A value the engine does not record. Never a zero, never blank. */
export function NotRecorded({ why }: { why?: string | undefined }) {
  return (
    <span
      className="text-[10px] uppercase tracking-wider text-zinc-600"
      title={why ?? "the engine does not record this field"}
    >
      {NOT_RECORDED}
    </span>
  );
}

/** Render a value, or `NOT RECORDED` when the API supplied nothing. */
export function Value({
  value,
  why,
  className = "",
}: {
  value: string | number | null | undefined;
  why?: string | undefined;
  className?: string | undefined;
}) {
  if (absent(value)) return <NotRecorded why={why} />;
  return <span className={className}>{String(value)}</span>;
}

export function Empty({ message }: { message: string }) {
  return <p className="py-6 text-center text-sm text-zinc-500">{message}</p>;
}

/**
 * A panel whose source has failed.
 *
 * Says when the last good reading was, because "unavailable" and "unavailable since two
 * hours ago" call for very different reactions.
 */
export function Unavailable({
  what,
  error,
  lastSuccessAt,
}: {
  what: string;
  error?: string | null | undefined;
  lastSuccessAt?: string | null | undefined;
}) {
  return (
    <div className="rounded border border-[#fab219]/30 bg-[#fab219]/10 px-3 py-2.5 text-xs text-amber-100">
      <div className="font-medium text-[#fab219]">{what} temporarily unavailable</div>
      {lastSuccessAt ? (
        <div className="mt-1 text-amber-100/80">
          last successful update {clock(lastSuccessAt)} ({ago(lastSuccessAt)})
        </div>
      ) : (
        <div className="mt-1 text-amber-100/80">no successful update yet this session</div>
      )}
      {error ? <div className="mt-1 text-amber-200/70">{error}</div> : null}
    </div>
  );
}

/** A small badge showing whether a block is live, stale or failed. */
export function FreshnessBadge({
  freshness,
  label,
}: {
  freshness?: Freshness | undefined;
  label: string;
}) {
  const available = freshness?.available ?? false;
  const stale = freshness?.stale ?? false;
  const state = !available ? "DISCONNECTED" : stale ? "STALE" : "LIVE";
  const colour = !available
    ? "bg-[#d03b3b]/15 text-[#d03b3b] ring-[#d03b3b]/40"
    : stale
      ? "bg-[#fab219]/15 text-[#fab219] ring-[#fab219]/40"
      : "bg-[#0ca30c]/15 text-[#0ca30c] ring-[#0ca30c]/40";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wider ring-1 ${colour}`}
      title={freshness?.error ?? undefined}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {label} {state}
      {freshness?.fetched_at ? (
        <span className="font-mono text-zinc-400">{clock(freshness.fetched_at)}</span>
      ) : null}
    </span>
  );
}

/** Marks a statistic computed over too few observations to mean anything. */
export function InsufficientSample({
  n,
  min,
}: {
  n?: number | undefined;
  min?: number | undefined;
}) {
  return (
    <span
      className="ml-1.5 rounded bg-[#fab219]/15 px-1 py-px text-[9px] uppercase tracking-wider text-[#fab219]"
      title={`${n ?? 0} trades; at least ${min ?? 10} are needed for this to be meaningful`}
    >
      insufficient sample
    </span>
  );
}

export function Table({ head, children }: { head: string[]; children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-[10px] uppercase tracking-wider text-zinc-500">
            {head.map((column) => (
              <th
                key={column}
                className={`whitespace-nowrap pb-2 pr-3 ${
                  column.startsWith(">") ? "text-right" : ""
                }`}
              >
                {column.replace(/^>/, "")}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="font-mono tabular-nums">{children}</tbody>
      </table>
    </div>
  );
}

/** A labelled warning the operator should read before acting on the numbers beside it. */
export function Caution({ children }: { children: ReactNode }) {
  return (
    <div className="rounded border border-[#fab219]/30 bg-[#fab219]/10 px-3 py-2 text-[11px] text-amber-100/90">
      {children}
    </div>
  );
}

/** A contradiction between two sources that the operator needs to see. */
export function Contradiction({ children }: { children: ReactNode }) {
  return (
    <div className="rounded border border-[#d03b3b]/40 bg-[#d03b3b]/10 px-3 py-2 text-[11px] text-rose-100">
      {children}
    </div>
  );
}
