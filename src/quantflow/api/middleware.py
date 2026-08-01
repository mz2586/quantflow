"""HTTP middleware: request identity, timing, error envelopes and metrics."""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_500_INTERNAL_SERVER_ERROR

from quantflow.core.errors import QuantFlowError
from quantflow.core.logging import bind_log_context, clear_log_context, get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

REQUESTS = Counter(
    "quantflow_http_requests_total",
    "HTTP requests handled",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "quantflow_http_request_seconds",
    "HTTP request duration",
    ["method", "path"],
    # Buckets skewed low: this API backs a live dashboard, so the interesting question
    # is "is it under 100ms", not "is it under 10 seconds".
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
ERRORS = Counter("quantflow_http_errors_total", "HTTP requests that failed", ["code"])


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, bind it to the log context, and time the request.

    The id is echoed in the response header and in the error envelope, so a user-reported
    failure can be traced to exact log lines without guesswork.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Wrap one request."""
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        clear_log_context()
        bind_log_context(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - started
            logger.exception(
                "http.unhandled_error",
                duration_ms=round(duration * 1000, 2),
            )
            raise
        finally:
            clear_log_context()

        duration = time.perf_counter() - started
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Response-Time-Ms"] = f"{duration * 1000:.2f}"

        # Route template, not the raw path: `/orders/{id}` keeps cardinality bounded,
        # where the concrete path would create a new metric series per order.
        route = request.scope.get("route")
        label = getattr(route, "path", request.url.path)
        REQUESTS.labels(request.method, label, response.status_code).inc()
        LATENCY.labels(request.method, label).observe(duration)

        if response.status_code >= HTTP_500_INTERNAL_SERVER_ERROR:
            logger.error(
                "http.server_error",
                status=response.status_code,
                duration_ms=round(duration * 1000, 2),
            )
        return response


async def quantflow_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render a :class:`QuantFlowError` as the standard error envelope."""
    assert isinstance(exc, QuantFlowError)
    request_id = getattr(request.state, "request_id", None)
    ERRORS.labels(exc.code).inc()

    log = logger.warning if exc.http_status < HTTP_500_INTERNAL_SERVER_ERROR else logger.error
    log(
        "http.error",
        code=exc.code,
        status=exc.http_status,
        message=exc.message,
        details=exc.details,
    )
    return JSONResponse(
        status_code=exc.http_status,
        content={"error": {**exc.to_dict(), "request_id": request_id}},
        headers={REQUEST_ID_HEADER: request_id} if request_id else None,
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Render an unexpected exception without leaking internals.

    The message is deliberately generic: an exception string can contain a DSN, a file
    path or part of a query, and none of that belongs in an HTTP response. The request id
    is the bridge to the full traceback in the logs.
    """
    request_id = getattr(request.state, "request_id", None)
    ERRORS.labels("internal_error").inc()
    logger.exception("http.unhandled", error=str(exc))
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "internal_error",
                "message": "an internal error occurred",
                "request_id": request_id,
            }
        },
    )


def metrics_response() -> Response:
    """Render the Prometheus exposition format."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def install(app: FastAPI) -> None:
    """Attach middleware and exception handlers to an app."""
    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(QuantFlowError, quantflow_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
