"""POST /api/query: request parsing, lazy provider construction, and serialized execution.

Providers are built on first use of their stage (not at server start) and cached for the
server's lifetime, same lazy-and-cache shape as the CLI's `query` command."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from ragx.core.config import Config
from ragx.core.errors import RagxError
from ragx.core.query import QueryOptions, run_query, to_query_json
from ragx.providers import lmstudio_process
from ragx.providers.registry import make_embedder, make_generator, make_reranker

log = logging.getLogger("ragx.web")


class QueryRunner:
    def __init__(self, root: Path, cfg: Config):
        self.root = root
        self.cfg = cfg
        self._lock = threading.Lock()
        self._embedder = None
        self._generator = None
        self._reranker = None

    def _get_embedder(self):
        # embeddings autostart already ran once at server startup (cli/web_cmd.py); only
        # rerank/expansion autostart lazily, on first use of their stage.
        if self._embedder is None:
            self._embedder = make_embedder(self.cfg)
        return self._embedder

    def _get_generator(self):
        if self._generator is None:
            try:
                lmstudio_process.ensure_for_section(self.cfg, "expansion")
            except RagxError as exc:
                log.warning("query expansion backend unavailable: %s", exc)
            self._generator = make_generator(self.cfg)
        return self._generator

    def _get_reranker(self):
        if self._reranker is None:
            lmstudio_process.ensure_for_section(self.cfg, "rerank")
            self._reranker = make_reranker(self.cfg)
        return self._reranker

    def parse_options(self, payload: object) -> tuple[str, QueryOptions]:
        """Raises ValueError (handler maps this to 400) on any malformed field."""
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        text = payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("'text' is required and must be a non-empty string")

        def as_bool(key: str, default: bool) -> bool:
            value = payload.get(key, default)
            if not isinstance(value, bool):
                raise ValueError(f"'{key}' must be a boolean")
            return value

        def as_nonneg_int(key: str, default: int) -> int:
            value = payload.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"'{key}' must be an integer")
            if value < 0:
                raise ValueError(f"'{key}' must be >= 0")
            return value

        hops = payload.get("hops")
        if hops is not None and (isinstance(hops, bool) or not isinstance(hops, int) or hops < 0):
            raise ValueError("'hops' must be a non-negative integer or null")

        default_lines = self.cfg.get("query.context_lines")
        opts = QueryOptions(
            top=as_nonneg_int("top", self.cfg.get("query.top")),
            expand=as_bool("expand", False),
            graph=as_bool("graph", True),
            rerank=as_bool("rerank", False),
            hops=hops,
            explain=as_bool("explain", False),
            before=as_nonneg_int("before", default_lines),
            after=as_nonneg_int("after", default_lines),
        )
        return text, opts

    def run(self, text: str, opts: QueryOptions) -> dict:
        start = time.perf_counter()
        with self._lock:
            out = run_query(
                self.root, self.cfg, self._get_embedder(), text, opts,
                generator=self._get_generator() if opts.expand else None,
                reranker=self._get_reranker() if opts.rerank else None,
            )
        doc = to_query_json(out, max_chunk_chars=self.cfg.get("query.max_chunk_chars"))
        doc["elapsed_ms"] = round((time.perf_counter() - start) * 1000)
        return doc
