"""Allow ``python -m quantflow`` to run the CLI."""

from __future__ import annotations

from quantflow.cli.main import app

if __name__ == "__main__":
    app()
