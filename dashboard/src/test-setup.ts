/**
 * Test environment shims.
 *
 * jsdom implements neither `ResizeObserver` nor element layout, both of which Recharts'
 * `ResponsiveContainer` requires to decide how large a chart should be. Without these the
 * charts throw during render — which the error boundaries dutifully contain, so the page
 * still renders and the *panel* assertions still pass while every chart quietly becomes an
 * error box. That is a worse failure than a crash: the suite stays green while the charts
 * are never actually exercised.
 *
 * These shims are jsdom's missing browser surface, not a stub of anything under test.
 */

class ResizeObserverStub {
  observe(): void {
    // Nothing to observe: jsdom never lays anything out.
  }

  unobserve(): void {
    // See observe().
  }

  disconnect(): void {
    // See observe().
  }
}

globalThis.ResizeObserver = ResizeObserverStub;

// ResponsiveContainer measures its parent and renders nothing at all when the box is zero,
// so the charts need a non-zero size reported for their container.
for (const [property, value] of [
  ["offsetWidth", 1024],
  ["offsetHeight", 640],
  ["clientWidth", 1024],
  ["clientHeight", 640],
] as const) {
  Object.defineProperty(HTMLElement.prototype, property, {
    configurable: true,
    value,
  });
}
