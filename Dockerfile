# syntax=docker/dockerfile:1.7
# --------------------------------------------------------------------------- #
# Stage 1 — builder: resolve and install dependencies into a self-contained venv
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# build-essential is needed for any sdist-only wheels; it never reaches the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.9 /uv /usr/local/bin/uv

WORKDIR /build

# Dependency layer: copy only the metadata first so a source-only change does not
# invalidate the (slow) dependency install.
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python 3.12 \
    && mkdir -p src/quantflow \
    && printf '__version__ = "0.0.0"\n' > src/quantflow/__init__.py \
    && VIRTUAL_ENV=/opt/venv uv pip install --python /opt/venv/bin/python ".[ai]"

# Source layer.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    VIRTUAL_ENV=/opt/venv uv pip install --python /opt/venv/bin/python --no-deps .

# --------------------------------------------------------------------------- #
# Stage 2 — runtime: no compilers, no uv, non-root
# --------------------------------------------------------------------------- #
FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    QF_LOG_FORMAT=json

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1001 quantflow \
    && useradd --system --uid 1001 --gid quantflow --create-home quantflow

COPY --from=builder --chown=quantflow:quantflow /opt/venv /opt/venv

WORKDIR /app
COPY --chown=quantflow:quantflow alembic.ini ./
COPY --chown=quantflow:quantflow migrations/ ./migrations/
COPY --chown=quantflow:quantflow scripts/ ./scripts/

RUN mkdir -p /app/data /app/reports && chown -R quantflow:quantflow /app

USER quantflow

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# tini reaps zombies and forwards SIGTERM, so an in-flight order loop gets a clean
# shutdown signal instead of being SIGKILLed.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "quantflow.api.app:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
