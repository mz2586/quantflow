import { describe, expect, it } from "vitest";
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
} from "./format";

describe("absence versus zero", () => {
  // The distinction this suite exists to protect: a field the engine does not record must
  // never render as 0. A ledger showing 0.00 for an unmeasured quantity is indistinguishable
  // from one that measured zero, and the reader has no way to tell it is looking at a gap.
  it("renders NOT RECORDED for null and undefined, not zero", () => {
    expect(money(null)).toBe(NOT_RECORDED);
    expect(money(undefined)).toBe(NOT_RECORDED);
    expect(signed(null)).toBe(NOT_RECORDED);
    expect(percent(null)).toBe(NOT_RECORDED);
    expect(ratio(null)).toBe(NOT_RECORDED);
    expect(quantity(null)).toBe(NOT_RECORDED);
    expect(count(null)).toBe(NOT_RECORDED);
    expect(duration(null)).toBe(NOT_RECORDED);
    expect(time(null)).toBe(NOT_RECORDED);
  });

  it("renders a real zero as zero", () => {
    expect(money("0")).toBe("0.00");
    expect(signed("0")).toBe("+0.00");
    expect(percent("0")).toBe("0.00%");
    expect(count(0)).toBe("0");
    expect(duration(0)).toBe("0s");
  });

  it("identifies absence without treating zero as absent", () => {
    expect(absent(null)).toBe(true);
    expect(absent(undefined)).toBe(true);
    expect(absent("")).toBe(true);
    expect(absent(0)).toBe(false);
    expect(absent("0")).toBe(false);
  });
});

describe("money", () => {
  it("formats with thousands separators and two decimals", () => {
    expect(money("49899.34635401")).toBe("49,899.35");
    expect(money("49899.34635401", "USDT")).toBe("49,899.35 USDT");
  });

  it("does not lose the integer part of a large balance", () => {
    expect(money("164459.161")).toContain("164,459");
  });

  it("survives an unparseable value rather than throwing", () => {
    expect(money("not-a-number")).toBe("0.00");
  });
});

describe("signed", () => {
  it("always shows a sign", () => {
    expect(signed("6.28955")).toBe("+6.29");
    expect(signed("-64.805979")).toBe("−64.81");
  });
});

describe("percent", () => {
  it("treats the wire value as a fraction", () => {
    expect(percent("0.4", 1)).toBe("40.0%");
    expect(percent("0.00090140", 3)).toBe("0.090%");
  });
});

describe("quantity", () => {
  it("trims trailing zeros without losing significant digits", () => {
    expect(quantity("4.120000000000")).toBe("4.12");
    expect(quantity("18186")).toBe("18186");
    expect(quantity("0.039000")).toBe("0.039");
  });
});

describe("tone", () => {
  it("colours by sign and stays neutral for an absent value", () => {
    expect(tone("1")).toContain("0ca30c");
    expect(tone("-1")).toContain("d03b3b");
    expect(tone(null)).toContain("zinc");
  });
});

describe("duration", () => {
  it("formats compactly across scales", () => {
    expect(duration(45)).toBe("45s");
    expect(duration(4087)).toBe("1h 8m");
    expect(duration(90_000)).toBe("1d 1h");
  });
});

describe("time helpers", () => {
  it("returns a placeholder for an unparseable timestamp instead of Invalid Date", () => {
    expect(time("nonsense")).toBe(NOT_RECORDED);
    expect(ago("nonsense")).toBe(NOT_RECORDED);
    expect(clock("nonsense")).toBe("—");
  });

  it("formats a wall clock for the last-successful-update line", () => {
    expect(clock("2026-08-14T13:45:00Z")).toMatch(/\d{2}:\d{2}:\d{2}/);
  });
});
