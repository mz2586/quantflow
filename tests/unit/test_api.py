"""API surface: health, errors, auth, schemas and the risk endpoints.

Exercised through `httpx.ASGITransport` so the real middleware, dependency graph and
serialisation all run — a route tested by calling its function directly proves almost
nothing about what a client actually receives.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from decimal import Decimal

import httpx
import pytest
from fastapi import FastAPI

from quantflow.api.app import create_app
from quantflow.api.deps import AppState
from quantflow.api.schemas import ApiModel, PortfolioResponse, StrategyDescription
from quantflow.core.config import Environment, RiskSettings, Settings
from quantflow.core.errors import NotFoundError, ValidationError
from quantflow.portfolio.manager import PortfolioManager
from quantflow.risk.engine import RiskEngine
from quantflow.strategy.registry import load_builtin_strategies


def build_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, env=Environment.TEST, **overrides)  # type: ignore[arg-type]


@pytest.fixture
def app() -> FastAPI:
    """An app with no external dependencies wired.

    Lifespan is bypassed so the suite never needs a database; each test attaches only the
    state it exercises.
    """
    application = create_app(build_settings())
    application.state.quantflow = AppState(
        settings=application.state.settings,
        registry=load_builtin_strategies(),
        risk=RiskEngine(RiskSettings()),
    )
    return application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    """An HTTP client bound to the app, without running lifespan."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


class TestHealth:
    async def test_healthz_never_touches_a_dependency(self, client: httpx.AsyncClient) -> None:
        # A liveness probe that fails on a database blip gets a healthy process killed.
        response = await client.get("/healthz")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["environment"] == "test"

    async def test_readyz_reports_components(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/readyz")
        assert response.status_code == 200
        body = response.json()
        assert body["ready"] is True  # nothing wired means nothing unhealthy
        assert body["trading_mode"] == "paper"
        assert body["kill_switch_engaged"] is False

    async def test_readyz_returns_503_when_a_component_is_down(
        self, app: FastAPI, client: httpx.AsyncClient
    ) -> None:
        class DeadDatabase:
            async def ping(self) -> bool:
                return False

        app.state.quantflow.database = DeadDatabase()
        response = await client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["ready"] is False

    async def test_metrics_endpoint(self, client: httpx.AsyncClient) -> None:
        await client.get("/healthz")
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "quantflow_http_requests_total" in response.text


class TestRequestContext:
    async def test_request_id_is_returned(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/healthz")
        assert response.headers["X-Request-ID"]
        assert float(response.headers["X-Response-Time-Ms"]) >= 0

    async def test_supplied_request_id_is_honoured(self, client: httpx.AsyncClient) -> None:
        # Lets a caller correlate its own trace id with our logs.
        response = await client.get("/healthz", headers={"X-Request-ID": "trace-abc"})
        assert response.headers["X-Request-ID"] == "trace-abc"


class TestErrorEnvelope:
    async def test_domain_errors_use_the_standard_envelope(self, app: FastAPI) -> None:
        @app.get("/boom-domain")
        async def boom() -> None:
            raise NotFoundError("widget is missing", widget_id="w1")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            response = await http.get("/boom-domain")

        assert response.status_code == 404
        error = response.json()["error"]
        assert error["code"] == "not_found"
        assert error["message"] == "widget is missing"
        assert error["details"]["widget_id"] == "w1"
        assert error["request_id"]

    async def test_validation_errors_map_to_422(self, app: FastAPI) -> None:
        @app.get("/boom-validation")
        async def boom() -> None:
            raise ValidationError("bad input")

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            response = await http.get("/boom-validation")
        assert response.status_code == 422

    async def test_unexpected_errors_do_not_leak_internals(self, app: FastAPI) -> None:
        # An exception string can contain a DSN or a file path; none of it belongs in a
        # response body.
        @app.get("/boom-internal")
        async def boom() -> None:
            raise RuntimeError("postgresql://user:hunter2@db:5432/secret")

        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            response = await http.get("/boom-internal")

        assert response.status_code == 500
        body = response.text
        assert "hunter2" not in body
        assert "postgresql" not in body
        assert response.json()["error"]["code"] == "internal_error"


class TestStrategyEndpoints:
    async def test_lists_every_registered_strategy(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/strategies")
        assert response.status_code == 200
        strategies = response.json()
        identifiers = {entry["strategy_id"] for entry in strategies}
        assert {"ema_cross", "rsi_reversion", "donchian_breakout"} <= identifiers

    async def test_schema_is_exposed_for_dashboard_forms(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/strategies/ema_cross")
        assert response.status_code == 200
        body = response.json()
        # Serialised as `schema` even though the field is `parameter_schema` internally.
        assert "schema" in body
        assert "fast_period" in body["schema"]["properties"]
        assert body["defaults"]["fast_period"] == 12

    async def test_unknown_strategy_returns_404(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/strategies/nope")
        assert response.status_code == 404
        assert "available:" in response.json()["error"]["message"]


class TestRiskEndpoints:
    async def test_status_reports_limits(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/risk/status")
        assert response.status_code == 200
        body = response.json()
        assert body["kill_switch"]["engaged"] is False
        assert body["limits"]["require_stop_loss"] is True
        assert "stop_loss_required" in body["rules"]

    async def test_kill_switch_engage_and_clear(self, client: httpx.AsyncClient) -> None:
        engaged = await client.post(
            "/api/v1/risk/kill-switch",
            json={"engaged": True, "reason": "manual halt", "actor": "tester"},
        )
        assert engaged.status_code == 200
        assert engaged.json()["engaged"] is True
        assert engaged.json()["reason"] == "manual halt"

        status = await client.get("/api/v1/risk/status")
        assert status.json()["kill_switch"]["engaged"] is True

        cleared = await client.post(
            "/api/v1/risk/kill-switch", json={"engaged": False, "actor": "tester"}
        )
        assert cleared.status_code == 200
        assert cleared.json()["engaged"] is False

    async def test_engaging_without_a_reason_is_rejected(self, client: httpx.AsyncClient) -> None:
        # A halt with no recorded cause is close to useless in the post-mortem.
        response = await client.post(
            "/api/v1/risk/kill-switch", json={"engaged": True, "reason": "   "}
        )
        assert response.status_code == 422
        assert "reason is required" in response.json()["error"]["message"]

    async def test_resume_does_not_clear_the_kill_switch(self, client: httpx.AsyncClient) -> None:
        # A daily halt and a latched drawdown breach are different severities; one button
        # that clears both invites clearing the serious one by reflex.
        await client.post(
            "/api/v1/risk/kill-switch",
            json={"engaged": True, "reason": "drawdown breach"},
        )
        response = await client.post("/api/v1/risk/resume")
        assert response.status_code == 200
        assert response.json()["engaged"] is True

    async def test_unknown_fields_are_rejected(self, client: httpx.AsyncClient) -> None:
        response = await client.post(
            "/api/v1/risk/kill-switch",
            json={"engaged": True, "reason": "x", "typo_field": 1},
        )
        assert response.status_code == 422


class TestPortfolioEndpoint:
    async def test_no_session_gives_a_clear_message(self, client: httpx.AsyncClient) -> None:
        # Better than a portfolio of zeros that reads as a flat account.
        response = await client.get("/api/v1/portfolio")
        assert response.status_code == 500
        assert "no active trading session" in response.json()["error"]["message"]

    async def test_reports_live_portfolio_state(
        self, app: FastAPI, client: httpx.AsyncClient, btc
    ) -> None:
        from quantflow.domain.enums import OrderSide
        from quantflow.domain.orders import Fill
        from tests.conftest import REFERENCE_TIME

        manager = PortfolioManager(starting_equity=Decimal("10000"))
        manager.apply_fill(
            Fill(
                fill_id="f1",
                order_id="o1",
                symbol=btc,
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                fee=Decimal("5"),
                fee_currency="USDT",
                timestamp=REFERENCE_TIME,
            )
        )
        manager.update_mark_price(btc, Decimal("55000"))
        app.state.quantflow.portfolio = manager

        response = await client.get("/api/v1/portfolio")
        assert response.status_code == 200
        body = response.json()
        assert body["position_count"] == 1
        assert body["positions"][0]["symbol"] == "BTC/USDT"
        # Money crosses the wire as a string, not a float.
        assert isinstance(body["equity"], str)
        assert Decimal(body["equity"]) == Decimal("10495")


class TestDecimalSerialisation:
    def test_decimals_serialise_as_strings(self) -> None:
        """The property a JavaScript client depends on.

        `0.1 + 0.2 != 0.3` in JS, so a position size delivered as a JSON number can be
        rendered wrong. Strings survive the trip intact.
        """
        response = PortfolioResponse(
            base_currency="USDT",
            equity=Decimal("10495.123456789"),
            cash=Decimal("5000"),
            starting_equity=Decimal("10000"),
            total_return_pct=Decimal("0.0495123456789"),
            realized_pnl=Decimal("0"),
            unrealized_pnl=Decimal("495.123456789"),
            fees_paid=Decimal("5"),
            gross_exposure=Decimal("5500"),
            leverage=Decimal("0.524"),
            drawdown_pct=Decimal("0"),
            daily_pnl=Decimal("495.12"),
            position_count=1,
        )
        payload = json.loads(response.model_dump_json())
        assert payload["equity"] == "10495.123456789"
        assert Decimal(payload["equity"]) == Decimal("10495.123456789")

    def test_schemas_reject_unknown_fields(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            StrategyDescription(
                strategy_id="x",
                description="",
                warmup_bars=1,
                defaults={},
                parameter_schema={},
                unexpected="boom",  # type: ignore[call-arg]
            )

    def test_schemas_are_frozen(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        model = ApiModel()
        with pytest.raises((PydanticValidationError, AttributeError)):
            model.anything = 1  # type: ignore[attr-defined]


class TestOpenApi:
    async def test_openapi_is_served_in_development(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert spec["info"]["title"] == "QuantFlow"
        assert "/api/v1/risk/kill-switch" in spec["paths"]

    async def test_docs_are_hidden_on_an_unauthenticated_production_deployment(
        self,
    ) -> None:
        # The docs are a live description of every endpoint, including the kill switch.
        from pydantic import ValidationError as PydanticValidationError

        from quantflow.core.config import DatabaseSettings

        # Settings validation refuses an unauthenticated production deployment outright,
        # so the app-level docs guard is belt and braces rather than the only defence.
        with pytest.raises(PydanticValidationError, match="api_key is required"):
            Settings(
                _env_file=None,
                env=Environment.PRODUCTION,
                api_key=None,
                secret_key="s" * 32,  # type: ignore[arg-type]
                database=DatabaseSettings(password="a-real-password"),  # type: ignore[arg-type]
            )

    async def test_docs_are_served_when_production_is_authenticated(self) -> None:
        from quantflow.core.config import DatabaseSettings

        settings = Settings(
            _env_file=None,
            env=Environment.PRODUCTION,
            api_key="k" * 32,  # type: ignore[arg-type]
            secret_key="s" * 32,  # type: ignore[arg-type]
            database=DatabaseSettings(password="a-real-password"),  # type: ignore[arg-type]
        )
        application = create_app(settings)
        assert application.docs_url == "/docs"


class TestCors:
    async def test_configured_origin_is_allowed(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/healthz", headers={"Origin": "http://localhost:5173"})
        assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"

    async def test_unknown_origin_is_not_allowed(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/healthz", headers={"Origin": "http://evil.test"})
        assert response.headers.get("access-control-allow-origin") is None
