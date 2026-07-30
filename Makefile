.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON      := .venv/bin/python
PIP         := uv pip install --python $(PYTHON)
COMPOSE     := docker compose
SYMBOL      ?= BTC/USDT
TIMEFRAME   ?= 1h
STRATEGY    ?= ema_cross
START       ?= 2024-01-01
END         ?= 2025-01-01

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
.PHONY: install
install: ## Create .venv and install the package with dev + ai extras
	uv venv --python 3.12 .venv
	$(PIP) -e ".[dev,ai]"

.PHONY: install-research
install-research: ## Add the VectorBT / Backtrader research extras
	$(PIP) -e ".[dev,ai,research]"

.PHONY: env
env: ## Create .env from the example if absent
	@test -f .env || (cp .env.example .env && echo "created .env — fill it in")

# --------------------------------------------------------------------------- #
# Quality gates
# --------------------------------------------------------------------------- #
.PHONY: fmt
fmt: ## Autoformat (black + ruff --fix)
	$(PYTHON) -m black .
	$(PYTHON) -m ruff check --fix .

.PHONY: lint
lint: ## Lint without modifying files
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

.PHONY: type
type: ## Strict type check
	$(PYTHON) -m mypy

.PHONY: test
test: ## Unit tests
	$(PYTHON) -m pytest -m "not integration and not network"

.PHONY: test-int
test-int: ## Integration tests (requires: make infra-up)
	$(PYTHON) -m pytest -m integration

.PHONY: cov
cov: ## Unit tests with coverage report
	$(PYTHON) -m pytest -m "not integration and not network" \
		--cov --cov-report=term-missing --cov-report=html --cov-fail-under=80

.PHONY: check
check: lint type test ## Everything CI runs

# --------------------------------------------------------------------------- #
# Infrastructure
# --------------------------------------------------------------------------- #
.PHONY: infra-up
infra-up: env ## Start Postgres + Redis only
	$(COMPOSE) up -d db redis
	@$(COMPOSE) ps

.PHONY: infra-down
infra-down: ## Stop Postgres + Redis
	$(COMPOSE) stop db redis

.PHONY: up
up: env ## Start the full stack
	$(COMPOSE) up -d --build
	@$(COMPOSE) ps

.PHONY: down
down: ## Stop the stack
	$(COMPOSE) down

.PHONY: nuke
nuke: ## Stop the stack and delete its volumes (DESTRUCTIVE)
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Tail stack logs
	$(COMPOSE) logs -f --tail=100

.PHONY: psql
psql: ## Open a psql shell in the db container
	$(COMPOSE) exec db psql -U quantflow -d quantflow

.PHONY: redis-cli
redis-cli: ## Open redis-cli in the redis container
	$(COMPOSE) exec redis redis-cli

# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
.PHONY: migrate
migrate: ## Apply all migrations
	$(PYTHON) -m alembic upgrade head

.PHONY: downgrade
downgrade: ## Roll back one migration
	$(PYTHON) -m alembic downgrade -1

.PHONY: migration
migration: ## Autogenerate a migration: make migration m="add foo"
	@test -n "$(m)" || (echo 'usage: make migration m="message"'; exit 1)
	$(PYTHON) -m alembic revision --autogenerate -m "$(m)"

# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
.PHONY: api
api: ## Run the API with autoreload
	$(PYTHON) -m uvicorn quantflow.api.app:create_app --factory --reload --port 8000

.PHONY: worker
worker: ## Run the background worker
	$(PYTHON) -m quantflow.workers.runner

.PHONY: download
download: ## Backfill candles: make download SYMBOL=BTC/USDT TIMEFRAME=1h START=2024-01-01
	$(PYTHON) -m quantflow.cli.main data download \
		--symbol "$(SYMBOL)" --timeframe "$(TIMEFRAME)" --start "$(START)" --end "$(END)"

.PHONY: backtest
backtest: ## Run a backtest and write an HTML report
	$(PYTHON) -m quantflow.cli.main backtest run \
		--strategy "$(STRATEGY)" --symbol "$(SYMBOL)" --timeframe "$(TIMEFRAME)" \
		--start "$(START)" --end "$(END)" --report

.PHONY: paper
paper: ## Start the paper-trading engine
	$(PYTHON) -m quantflow.cli.main trade paper --strategy "$(STRATEGY)" --symbol "$(SYMBOL)"

# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #
.PHONY: dashboard-install
dashboard-install: ## Install dashboard dependencies
	cd dashboard && npm install

.PHONY: dashboard
dashboard: ## Run the Vite dev server
	cd dashboard && npm run dev

.PHONY: dashboard-build
dashboard-build: ## Type-check and build the dashboard
	cd dashboard && npm run build

.PHONY: clean
clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist
