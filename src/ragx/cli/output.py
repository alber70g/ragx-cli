"""Stdout/exit conventions for the CLI. stdout is reserved for result payloads;
everything diagnostic goes to stderr (configured in app.main())."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from typing import NoReturn

import typer


def emit_json(doc: dict) -> None:
    """Write exactly one JSON document to stdout, nothing else on stdout."""
    sys.stdout.write(json.dumps(doc) + "\n")


def fail(msg: str, code: int = 2) -> NoReturn:
    """Print an error message to stderr and exit with `code`."""
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def migrate_confirm() -> Callable[[str], bool] | None:
    """Confirm callback for Config.load's legacy-config migration: prompt on a TTY,
    None (migrate silently) for piped/agent runs."""
    if not sys.stdin.isatty():
        return None
    return lambda prompt: typer.confirm(prompt, default=True)
