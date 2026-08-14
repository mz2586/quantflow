"""Market-neutral strategies.

Kept apart from ``quantflow.strategy`` on purpose. Those are directional: they take a view
on where price goes and are scored against each other by the orchestrator. These take no
view on direction at all — a delta-hedged pair earns from a mechanical cash flow, funding,
and its edge stands or falls on whether that flow clears the cost of holding the hedge.

Mixing the two would put a strategy with a different return source, different risk profile
and different accounting into a pool that ranks on directional performance, where its
numbers would be neither comparable nor meaningful.
"""

from quantflow.neutral.funding_capture import (
    FundingCaptureParams,
    FundingCaptureResult,
    funding_payment,
    round_trip_cost,
    simulate_funding_capture,
)

__all__ = [
    "FundingCaptureParams",
    "FundingCaptureResult",
    "funding_payment",
    "round_trip_cost",
    "simulate_funding_capture",
]
