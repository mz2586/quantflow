"""Supporting machinery for the trading operations dashboard.

Everything the dashboard needs that is *not* already a domain concern lives here: response
caching, venue valuation, session resolution and orchestrator-log parsing. The router in
:mod:`quantflow.api.routers.dashboard` is deliberately thin and does no computation of its
own, so each piece below can be tested without a running API.
"""

from __future__ import annotations

__all__: list[str] = []
