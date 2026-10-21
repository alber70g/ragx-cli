"""Builds the ThreadingHTTPServer for `ragx-cli web`: loads the graph once at startup and
wires shared state onto the server instance (read by handlers via `self.server.state`)."""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from ragx.core.config import Config, db_path
from ragx.core.errors import RagxError
from ragx.core.graph_export import build_graph
from ragx.core.store import Store
from ragx.web.handlers import RagxRequestHandler
from ragx.web.query_runner import QueryRunner

log = logging.getLogger("ragx.web")


@dataclass
class ServerState:
    root: Path
    static_root: Path
    graph_bytes: bytes
    graph_gzip: bytes
    query_runner: QueryRunner


def _default_static_root() -> Path:
    return Path(str(resources.files("ragx.web") / "static"))


def build_server(
    root: Path,
    cfg: Config,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    static_root: Path | None = None,
) -> ThreadingHTTPServer:
    if not db_path(root).exists():
        raise RagxError("no index found — run `ragx-cli index` first")
    with Store(db_path(root)) as store:
        graph = build_graph(root, store)
    graph_bytes = json.dumps(graph).encode()
    log.info("graph: %d nodes, %d edges", graph["meta"]["nodes"], graph["meta"]["edges"])

    state = ServerState(
        root=root,
        static_root=static_root if static_root is not None else _default_static_root(),
        graph_bytes=graph_bytes,
        graph_gzip=gzip.compress(graph_bytes),
        query_runner=QueryRunner(root, cfg),
    )
    server = ThreadingHTTPServer((host, port), RagxRequestHandler)
    server.daemon_threads = True
    server.state = state  # type: ignore[attr-defined]
    return server
