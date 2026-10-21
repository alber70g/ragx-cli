"""`ragx-cli web`: serve the chunk graph + query UI. Registered onto the main app by app.py."""

from __future__ import annotations

import logging
import sys
import webbrowser

import typer

from ragx.cli.output import fail, migrate_confirm
from ragx.core.config import Config, require_root
from ragx.core.errors import RagxError
from ragx.providers import lmstudio_process
from ragx.web.server import build_server

log = logging.getLogger("ragx.web")


def register(app: typer.Typer) -> None:
    app.command()(web)


def web(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
    open_browser: bool = typer.Option(True, "--open/--no-open", help="open the UI in a browser"),
) -> None:
    """Serve the sub-chunk similarity graph and query UI (requires graph.edge_source="subchunk")."""
    try:
        root = require_root()
        cfg = Config.load(root, confirm=migrate_confirm())
        lmstudio_process.ensure_for_section(cfg, "embeddings")
        server = build_server(root, cfg, host=host, port=port)
    except RagxError as exc:
        fail(str(exc))

    url = f"http://{host}:{port}/"
    log.info("serving at %s", url)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("", file=sys.stderr)
    finally:
        server.server_close()
