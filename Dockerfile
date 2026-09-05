# No `# syntax=` directive: pinning an external frontend means every build pulls an
# image from Docker Hub before it can start, which fails on a flaky network. Docker's
# built-in BuildKit frontend already supports the cache mounts used below.
# --------------------------------------------------------------------------- #
# Stage 1 — builder: resolve and install dependencies into a self-contained venv
# --------------------------------------------------------------------------- #
FROM python:3.14-slim-bookworm AS builder

# UV_HTTP_TIMEOUT: Polars and PyArrow ship wheels well over 100 MB, and uv's 30s
# default is not enough for them on a slow or contended connection. The resulting failure
# reads as a network error rather than a timeout, so it is worth being generous here.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=600

# build-essential is needed for any sdist-only wheels; it never reaches the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.9 /uv /usr/local/bin/uv

WORKDIR /build

# Dependency layer: copy only the metadata first so a source-only change does not
# invalidate the (slow) dependency install.
# LICENSE and NOTICE are build inputs, not documentation: pyproject declares them
# via `license-files`, so hatchling fails without them. They also have to ship in
# the image — Apache-2.0 requires the licence to travel with the distribution.
COPY pyproject.toml README.md LICENSE NOTICE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python 3.12 \
    && mkdir -p src/quantflow \
    && printf '__version__ = "0.0.0"\n' > src/quantflow/__init__.py \
    && VIRTUAL_ENV=/opt/venv uv pip install --python /opt/venv/bin/python ".[ai]"

# Source layer.
# --reinstall-package is essential: the dependency layer above already installed
# quantflow 0.1.0 from the stub, and uv would otherwise treat this identical version as
# satisfied and skip it — shipping an image whose package contains only the stub.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    VIRTUAL_ENV=/opt/venv uv pip install --python /opt/venv/bin/python \
    --no-deps --reinstall-package quantflow .

# --------------------------------------------------------------------------- #
# Stage 2 — runtime: no compilers, no uv, non-root
# --------------------------------------------------------------------------- #
FROM python:3.14-slim-bookworm AS runtime

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
COPY --chown=quantflow:quantflow LICENSE NOTICE ./
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
