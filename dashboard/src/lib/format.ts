/**
 * Display formatting.
 *
 * Every function takes the **wire value** (a string) and returns a string for display.
 * Values are parsed to a number only at the last possible moment, purely for formatting —
 * no arithmetic is ever done here, because arithmetic is what loses precision.
 *
 * The distinction that matters most: `null`/`undefined` means **the engine does not record
 * this**, and it must never render as `0`. A ledger showing `0.00` for a quantity nobody
 * measured is indistinguishable from one that measured zero, and the reader has no way to
 * tell that they are looking at an absence.
 */

/** What a field the engine does not record looks like on screen. */
export const NOT_RECORDED = "NOT RECORDED";

/** Parse a wire value for display only. Returns 0 for anything unparseable. */
function toNumber(value: string | number | null | undefined): number {
  if (value === null || value === undefined) return 0;
  const parsed = typeof value === "number" ? value : Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

/** True when the API supplied no value at all, as opposed to a value of zero. */
export function absent(value: unknown): boolean {
  return value === null || value === undefined || value === "";
}

/** Format a currency amount with thousands separators. */
export function money(value: string | null | undefined, currency = ""): string {
  if (absent(value)) return NOT_RECORDED;
  const formatted = toNumber(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  return currency ? `${formatted} ${currency}` : formatted;
}

/** Format a fraction as a percentage. */
export function percent(value: string | null | undefined, digits = 2): string {
  if (absent(value)) return NOT_RECORDED;
  return `${(toNumber(value) * 100).toFixed(digits)}%`;
}

/** Format a signed amount, always showing the sign. */
export function signed(value: string | null | undefined, digits = 2): string {
  if (absent(value)) return NOT_RECORDED;
  const parsed = toNumber(value);
  const formatted = Math.abs(parsed).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
  return `${parsed >= 0 ? "+" : "−"}${formatted}`;
}

/** Format a bare ratio, e.g. a profit factor. */
export function ratio(value: string | null | undefined, digits = 2): string {
  if (absent(value)) return NOT_RECORDED;
  return toNumber(value).toFixed(digits);
}

/** Format a count. Zero is a real measurement here, so it renders as 0. */
export function count(value: number | null | undefined): string {
  return value === null || value === undefined ? NOT_RECORDED : String(value);
}

/** Format a quantity, trimming trailing zeros without losing significant digits. */
export function quantity(value: string | null | undefined): string {
  if (absent(value)) return NOT_RECORDED;
  const raw = String(value);
  const trimmed = raw.includes(".") ? raw.replace(/0+$/, "").replace(/\.$/, "") : raw;
  return trimmed || "0";
}

/** Tailwind text colour for a signed value. */
export function tone(value: string | null | undefined): string {
  if (absent(value)) return "text-zinc-500";
  const parsed = toNumber(value);
  if (parsed > 0) return "text-[#0ca30c]";
  if (parsed < 0) return "text-[#d03b3b]";
  return "text-zinc-400";
}

/** Format an ISO timestamp as a short local date and time. */
export function time(iso: string | null | undefined): string {
  if (!iso) return NOT_RECORDED;
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime())
    ? NOT_RECORDED
    : parsed.toLocaleString(undefined, {
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
}

/** Format an ISO timestamp as a wall-clock time, for "last successful update HH:MM:SS". */
export function clock(iso: string | null | undefined): string {
  if (!iso) return "—";
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime())
    ? "—"
    : parsed.toLocaleTimeString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
}

/** Format an ISO timestamp as a relative age, e.g. "3m ago". */
export function ago(iso: string | null | undefined): string {
  if (!iso) return NOT_RECORDED;
  const parsed = new Date(iso).getTime();
  if (Number.isNaN(parsed)) return NOT_RECORDED;
  return duration(Math.max(0, Math.floor((Date.now() - parsed) / 1000))) + " ago";
}

/** Format a number of seconds as a compact duration. */
export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return NOT_RECORDED;
  }
  const whole = Math.floor(seconds);
  if (whole < 60) return `${whole}s`;
  if (whole < 3600) return `${Math.floor(whole / 60)}m`;
  if (whole < 86_400) return `${Math.floor(whole / 3600)}h ${Math.floor((whole % 3600) / 60)}m`;
  return `${Math.floor(whole / 86_400)}d ${Math.floor((whole % 86_400) / 3600)}h`;
}

/** Numeric value for a chart axis. Display only — never fed back into a calculation. */
export function chartValue(value: string | null | undefined): number {
  return toNumber(value);
}

/** Milliseconds since epoch for a chart's time axis, or 0 when unparseable. */
export function chartTime(iso: string | null | undefined): number {
  if (!iso) return 0;
  const parsed = new Date(iso).getTime();
  return Number.isNaN(parsed) ? 0 : parsed;
}
