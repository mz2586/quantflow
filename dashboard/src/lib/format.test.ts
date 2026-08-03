/**
 * Formatting tests.
 *
 * The load-bearing property: a high-precision decimal arriving as a string must never be
 * silently corrupted. JavaScript numbers cannot represent these values exactly, so the
 * client keeps them as strings and only parses at the moment of display.
 */

import { describe, expect, it } from "vitest";
import { pathSymbol } from "./api";
import { ago, chartValue, money, percent, quantity, signed, time, tone } from "./format";

describe("money", () => {
  it("formats a decimal string", () => {
    expect(money("1234.5")).toBe("1,234.50");
  });

  it("appends a currency when given", () => {
    expect(money("100", "USDT")).toBe("100.00 USDT");
  });

  it("renders an em dash for missing values", () => {
    expect(money(null)).toBe("—");
    expect(money(undefined)).toBe("—");
  });

  it("does not throw on an unparseable value", () => {
    expect(money("not-a-number")).toBe("0.00");
  });
});

describe("percent", () => {
  it("converts a fraction", () => {
    expect(percent("0.1234")).toBe("12.34%");
  });

  it("honours the digit count", () => {
    expect(percent("0.1", 0)).toBe("10%");
  });
});

describe("signed", () => {
  it("always shows a sign", () => {
    expect(signed("100")).toBe("+100.00");
    expect(signed("-100")).toBe("−100.00");
    expect(signed("0")).toBe("+0.00");
  });
});

describe("quantity", () => {
  it("trims trailing zeros without losing precision", () => {
    expect(quantity("0.10000000")).toBe("0.1");
    expect(quantity("1.23456789")).toBe("1.23456789");
  });

  it("leaves integers alone", () => {
    expect(quantity("5")).toBe("5");
  });
});

describe("tone", () => {
  it("colours by sign", () => {
    expect(tone("1")).toContain("emerald");
    expect(tone("-1")).toContain("rose");
    expect(tone("0")).toContain("zinc");
  });
});

describe("time and ago", () => {
  it("renders an em dash for missing input", () => {
    expect(time(null)).toBe("—");
    expect(ago(null)).toBe("—");
  });

  it("renders an em dash for an invalid date", () => {
    expect(time("nonsense")).toBe("—");
  });

  it("reports a recent time in seconds", () => {
    expect(ago(new Date(Date.now() - 5000).toISOString())).toMatch(/s ago$/);
  });
});

describe("precision", () => {
  it("keeps full precision in the source string", () => {
    // The value itself is never mutated; only its rendering is lossy.
    const exact = "50000.123456789012";
    expect(quantity(exact)).toBe(exact);
  });

  it("chartValue is display-only and never fed back into arithmetic", () => {
    expect(chartValue("1.5")).toBe(1.5);
    expect(chartValue(null)).toBe(0);
  });
});

describe("pathSymbol", () => {
  it("converts a slashed symbol to a path-safe form", () => {
    // BTC%2FUSDT does not work: the server decodes it before routing and then sees an
    // extra path segment, so the route never matches.
    expect(pathSymbol("BTC/USDT")).toBe("BTC-USDT");
  });

  it("leaves an already-safe symbol alone", () => {
    expect(pathSymbol("BTCUSDT")).toBe("BTCUSDT");
  });

  it("still encodes anything else unsafe", () => {
    expect(pathSymbol("A B")).toBe("A%20B");
  });
});
