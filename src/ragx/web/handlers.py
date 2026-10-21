"""HTTP routing for the web server: static files, /api/graph, /api/chunk/{id}, /api/query.

State (cached graph bytes, static root, QueryRunner) lives on `self.server.state`,
set by `web.server.build_server` — threads share that one object, never a Store handle."""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler

from ragx.core.config import db_path
from ragx.core.errors import RagxError
from ragx.core.store import Store
from ragx.web import static_files

log = logging.getLogger("ragx.web")


class RagxRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        log.info("%s - %s", self.address_string(), fmt % args)

    @property
    def state(self):
        return self.server.state  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        try:
            self._route_get()
        except Exception as exc:  # unexpected failure must not kill the server
            log.exception("GET %s failed", self.path)
            self._error(500, str(exc))

    def do_POST(self) -> None:
        try:
            self._route_post()
        except Exception as exc:
            log.exception("POST %s failed", self.path)
            self._error(500, str(exc))

    def _route_get(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._serve_static("index.html")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/") :])
        elif path == "/api/graph":
            self._serve_graph()
        elif path.startswith("/api/chunk/"):
            self._serve_chunk(path[len("/api/chunk/") :])
        else:
            self._error(404, "not found")

    def _route_post(self) -> None:
        if self.path == "/api/query":
            self._serve_query()
        else:
            self._error(404, "not found")

    # -- /api/graph -------------------------------------------------------

    def _serve_graph(self) -> None:
        gzip_ok = "gzip" in self.headers.get("Accept-Encoding", "")
        body = self.state.graph_gzip if gzip_ok else self.state.graph_bytes
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if gzip_ok:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- /api/chunk/{id} ---------------------------------------------------

    def _serve_chunk(self, raw_id: str) -> None:
        try:
            chunk_id = int(raw_id)
        except ValueError:
            self._error(404, "invalid chunk id")
            return
        with Store(db_path(self.state.root)) as store:
            chunks = store.get_chunks([chunk_id])
            if not chunks:
                self._error(404, "unknown chunk id")
                return
            chunk = chunks[0]
            neighbors = sorted(store.neighbors(chunk_id), key=lambda t: -t[1])
            community = store.community_of(chunk_id)
        self._json(
            200,
            {
                "chunk_id": chunk.id,
                "file": chunk.file_path,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "text": chunk.text,
                "community": community,
                "neighbors": [{"chunk_id": cid, "weight": round(w, 4)} for cid, w in neighbors],
            },
        )

    # -- /api/query -----------------------------------------------------------

    def _serve_query(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._error(400, "invalid JSON body")
            return
        try:
            text, opts = self.state.query_runner.parse_options(payload)
        except ValueError as exc:
            self._error(400, str(exc))
            return
        try:
            doc = self.state.query_runner.run(text, opts)
        except RagxError as exc:
            self._error(502, str(exc))
            return
        self._json(200, doc)

    # -- static files ---------------------------------------------------------

    def _serve_static(self, rel_path: str) -> None:
        target = static_files.resolve(self.state.static_root, rel_path)
        if target is None:
            self._error(404, "not found")
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", static_files.content_type(target))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- helpers ---------------------------------------------------------------

    def _json(self, status: int, doc: dict) -> None:
        body = json.dumps(doc).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})
