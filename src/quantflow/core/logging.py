"""Structured logging.

`structlog` renders human-friendly colour output in development and single-line JSON in
production. A `contextvars`-backed binder carries a correlation id (request id, backtest
run id, strategy id) across `await` boundaries without threading it through signatures.

Secrets never reach the log stream: :func:`_redact_secrets` scrubs known-sensitive keys
from every event dict, and :class:`pydantic.SecretStr` values render as ``**********``.
"""

from __future__ import annotations

import logging
import logging.config
import sys
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from typing import Any, Final, cast

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, unbind_contextvars
from structlog.typing import EventDict, Processor, WrappedLogger

from quantflow.core.config import Settings

REDACTED: Final = "***redacted***"

#: Event-dict keys whose values are scrubbed before rendering. Matching is done on a
#: normalised (lowercased) key containing any of these fragments.
SENSITIVE_KEY_FRAGMENTS: Final[frozenset[str]] = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "auth_header",
        "private_key",
        "credential",
        "signature",
        "cookie",
        "session_id",
    }
)

_configured = False


def _redact_secrets(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Replace sensitive values anywhere in the event dict."""
    _redact_mapping(event_dict)
    return event_dict


def _redact_mapping(mapping: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in list(mapping.items()):
        normalised = key.lower()
        if any(fragment in normalised for fragment in SENSITIVE_KEY_FRAGMENTS):
            mapping[key] = REDACTED
        elif isinstance(value, MutableMapping):
            mapping[key] = _redact_mapping(value)
    return mapping


def _add_service_context(service: str, version: str) -> Processor:
    """Build a processor that stamps every event with static service metadata."""

    def processor(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
        event_dict.setdefault("service", service)
        event_dict.setdefault("version", version)
        return event_dict

    return processor


def configure_logging(settings: Settings, *, service: str = "quantflow") -> None:
    """Configure structlog and the stdlib logging bridge.

    Idempotent: repeated calls reconfigure rather than stacking handlers, which keeps
    test runs and uvicorn's own reload behaviour predictable.
    """
    global _configured  # noqa: PLW0603 — module-level singleton guard

    from quantflow import __version__

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        _add_service_context(service, __version__),
        _redact_secrets,
    ]

    renderer: Processor
    if settings.log_format == "json":
        shared.append(structlog.processors.format_exc_info)
        shared.append(structlog.processors.dict_tracebacks)
        renderer = structlog.processors.JSONRenderer(sort_keys=True)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # Third-party loggers: route through our handler, damp the noisy ones.
    for name, level in (
        ("uvicorn", logging.INFO),
        ("uvicorn.error", logging.INFO),
        ("uvicorn.access", logging.WARNING),
        ("sqlalchemy.engine", logging.WARNING),
        ("aiosqlite", logging.WARNING),
        ("asyncio", logging.WARNING),
        ("ccxt", logging.WARNING),
        ("urllib3", logging.WARNING),
        ("httpx", logging.WARNING),
        ("httpcore", logging.WARNING),
        ("websockets", logging.WARNING),
        ("apscheduler", logging.WARNING),
    ):
        third_party = logging.getLogger(name)
        third_party.handlers.clear()
        third_party.propagate = True
        third_party.setLevel(level)

    _configured = True


def get_logger(name: str | None = None, **initial: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound logger.

    Args:
        name: Logger name; defaults to the caller's module via structlog inference.
        **initial: Key/values permanently bound to the returned logger.

    """
    logger = cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))
    return logger.bind(**initial) if initial else logger


def is_configured() -> bool:
    """Whether :func:`configure_logging` has run in this process."""
    return _configured


@contextmanager
def log_context(**values: Any) -> Iterator[None]:
    """Bind values to the ambient log context for the duration of the block."""
    tokens = bind_contextvars(**values)
    try:
        yield
    finally:
        # Restore prior values rather than dropping the keys entirely, so nested
        # contexts unwind correctly.
        structlog.contextvars.reset_contextvars(**tokens)


def bind_log_context(**values: Any) -> None:
    """Bind values to the ambient log context until explicitly cleared."""
    bind_contextvars(**values)


def unbind_log_context(*keys: str) -> None:
    """Remove keys from the ambient log context."""
    unbind_contextvars(*keys)


def clear_log_context() -> None:
    """Drop the entire ambient log context."""
    clear_contextvars()
