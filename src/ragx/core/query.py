"""Query pipeline. Phase 1: plain vector search; later stages hook in behind flags."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ragx.core.config import Config, db_path, vectors_path
from ragx.core.errors import RagxError
from ragx.core.models import QueryOutput, ScoredChunk
from ragx.core.store import Store
from ragx.core.vectors import VectorIndex
from ragx.providers.base import Embedder


@dataclass
class QueryOptions:
    top: int = 8
    files_only: bool = False
    expand: bool = True  # Phase 3
    graph: bool = True  # Phase 2
    rerank: bool = True  # Phase 3
    hops: int | None = None
    explain: bool = False


def run_query(root: Path, cfg: Config, embedder: Embedder, text: str, opts: QueryOptions) -> QueryOutput:
    if not text.strip():
        raise RagxError("empty query")
    with Store(db_path(root)) as store:
        if store.chunk_count() == 0:
            return QueryOutput(query=text)
        dim = int(store.get_meta("embedding_dim") or 0)
        if store.get_meta("embedding_model") != embedder.model:
            raise RagxError(
                f"index built with {store.get_meta('embedding_model')!r}, "
                f"config says {embedder.model!r}; run `ragx index` to rebuild"
            )
        index = VectorIndex.load(vectors_path(root), dim)
        vec = embedder.embed_queries([text])[0]
        hits = index.search(vec, opts.top)
        chunks = {c.id: c for c in store.get_chunks([cid for cid, _ in hits])}
        results = [
            ScoredChunk(chunk=chunks[cid], score=sim, vector_score=sim)
            for cid, sim in hits
            if cid in chunks
        ]
        return QueryOutput(query=text, results=results)


def to_query_json(out: QueryOutput, *, max_chunk_chars: int = 1200) -> dict:
    """Serialize as the versioned agent-facing schema."""
    return {
        "schema": "ragx.query.v1",
        "query": out.query,
        "variants": out.variants,
        "results": [
            {
                "chunk_id": r.chunk.id,
                "file": r.chunk.file_path,
                "line_start": r.chunk.line_start,
                "line_end": r.chunk.line_end,
                "byte_start": r.chunk.byte_start,
                "byte_end": r.chunk.byte_end,
                "score": round(r.score, 6),
                "breakdown": {
                    "vector": round(r.vector_score, 6),
                    "heat": round(r.heat, 6),
                    "rerank": round(r.rerank_score, 6),
                },
                "text": r.chunk.text[:max_chunk_chars],
                "truncated": len(r.chunk.text) > max_chunk_chars,
                **({"explain": r.explain} if r.explain else {}),
            }
            for r in out.results
        ],
    }


def to_files_json(out: QueryOutput) -> dict:
    """Aggregate chunk scores per file: sum of each file's top-3 chunk scores."""
    per_file: dict[str, list[ScoredChunk]] = {}
    for r in out.results:
        per_file.setdefault(r.chunk.file_path, []).append(r)
    files = [
        {
            "file": path,
            "score": round(sum(sorted((r.score for r in rs), reverse=True)[:3]), 6),
            "chunk_ids": [r.chunk.id for r in sorted(rs, key=lambda r: r.score, reverse=True)],
        }
        for path, rs in per_file.items()
    ]
    files.sort(key=lambda f: f["score"], reverse=True)
    return {"schema": "ragx.files.v1", "query": out.query, "variants": out.variants, "files": files}
