/**
 * Display formatting.
 *
 * Every function takes a **string** (the wire format) and returns a string for display.
 * Values are parsed to a number only at the last possible moment, purely for formatting —
 * no arithmetic is ever done on them here, because arithmetic is what loses precision.
 */

/** Parse a wire value for display only. Returns 0 for anything unparseable. */
function toNumber(value: string | number | null | undefined): number {
  if (value === null || value === undefined) return 0;
  const parsed = typeof value === "number" ? value : Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

/** Format a currency amount with thousands separators. */
export function money(value: string | null | undefined, currency = ""): string {
  if (value === null || value === undefined) return "—";
  const formatted = toNumber(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return currency ? `${formatted} ${currency}` : formatted;
}

/** Format a fraction as a percentage. */
export function percent(value: string | null | undefined, digits = 2): string {
  if (value === null || value === undefined) return "—";
  return `${(toNumber(value) * 100).toFixed(digits)}%`;
}

/** Format a signed amount, always showing the sign. */
export function signed(value: string | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const parsed = toNumber(value);
  const formatted = Math.abs(parsed).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return `${parsed >= 0 ? "+" : "−"}${formatted}`;
}

/** Format a quantity, trimming trailing zeros without losing significant digits. */
export function quantity(value: string | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const trimmed = value.includes(".") ? value.replace(/0+$/, "").replace(/\.$/, "") : value;
  return trimmed || "0";
}

/** Tailwind text colour for a signed value. */
export function tone(value: string | null | undefined): string {
  if (value === null || value === undefined) return "text-zinc-400";
  const parsed = toNumber(value);
  if (parsed > 0) return "text-emerald-400";
  if (parsed < 0) return "text-rose-400";
  return "text-zinc-400";
}

/** Format an ISO timestamp as a short local time. */
export function time(iso: string | null | undefined): string {
  if (!iso) return "—";
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime())
    ? "—"
    : parsed.toLocaleString(undefined, {
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
}

/** Format an ISO timestamp as a relative age, e.g. "3m ago". */
export function ago(iso: string | null | undefined): string {
  if (!iso) return "—";
  const parsed = new Date(iso).getTime();
  if (Number.isNaN(parsed)) return "—";
  const seconds = Math.max(0, Math.floor((Date.now() - parsed) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

/** Numeric value for a chart axis. Display only — never fed back into a calculation. */
export function chartValue(value: string | null | undefined): number {
  return toNumber(value);
}
