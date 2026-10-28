"""Query pipeline: expansion -> fan-out retrieval -> RRF -> heat traversal -> rerank -> combine
-> fold per file."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ragx.core.config import Config, db_path, rc_model_mismatch, vectors_path
from ragx.core.context import attach_context
from ragx.core.errors import RagxError
from ragx.core.expansion import expand_query
from ragx.core.fusion import rrf
from ragx.core.grouping import group_by_file
from ragx.core.models import QueryOutput, ScoredChunk
from ragx.core.scoring import combine
from ragx.core.store import Store
from ragx.core.traversal import propagate_heat, trace_path
from ragx.core.vectors import VectorIndex
from ragx.providers.base import Embedder, Generator, Reranker

RERANK_CAP = 100  # cross-encoder cost guard: rerank at most this many candidates


@dataclass
class QueryOptions:
    top: int = 20  # result entries (files), not chunks
    files_only: bool = False
    expand: bool = False
    graph: bool = True
    rerank: bool = False
    hops: int | None = None
    explain: bool = False
    before: int = 5  # grep -B: non-empty lines of context before each result chunk
    after: int = 5  # grep -A: non-empty lines of context after each result chunk


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _explain(trace: dict[int, dict], cid: int) -> dict | None:
    """Trace entry for `cid`, with `path` (seed-first parent chain) attached and
    `seed` set to `path[0]` so the two fields can never disagree."""
    entry = trace.get(cid)
    if entry is None:
        return None
    path = trace_path(trace, cid)
    return {**entry, "seed": path[0], "path": path}


def _check_model(store: Store, cfg: Config, embedder: Embedder) -> None:
    built_with = store.get_meta("embedding_model")
    if built_with != embedder.model:
        from_rc = rc_model_mismatch(cfg, built_with or "")
        if from_rc:
            raise RagxError("; ".join(from_rc))
        raise RagxError(
            f"index built with {built_with!r}, config says {embedder.model!r}; "
            "run `ragx-cli index --full` to rebuild"
        )


def run_query(
    root: Path,
    cfg: Config,
    embedder: Embedder,
    text: str,
    opts: QueryOptions,
    *,
    generator: Generator | None = None,
    reranker: Reranker | None = None,
) -> QueryOutput:
    if not text.strip():
        raise RagxError("empty query")
    with Store(db_path(root)) as store:
        if store.chunk_count() == 0:
            return QueryOutput(query=text)
        _check_model(store, cfg, embedder)
        index = VectorIndex.load(vectors_path(root), int(store.get_meta("embedding_dim") or 0))

        # 1. expansion (optional, one LLM call)
        variants: list[str] = []
        if opts.expand and generator is not None:
            exp = expand_query(
                generator, text,
                variants=cfg.get("expansion.variants"), hyde=cfg.get("expansion.hyde"),
            )
            variants = exp.variants + ([exp.hyde] if exp.hyde else [])

        # 2. fan-out retrieval + RRF fusion
        q_vecs = embedder.embed_queries([text, *variants])
        per_top = cfg.get("fusion.per_query_top")
        rankings = [[cid for cid, _ in index.search(v, per_top)] for v in q_vecs]
        fused = rrf(rankings, k=cfg.get("fusion.rrf_k"))
        if not fused:
            return QueryOutput(query=text, variants=variants)

        # 3. graph expansion via heat propagation
        heat: dict[int, float] = dict(fused)
        trace: dict[int, dict] = {}
        if opts.graph and store.edge_count() > 0:
            qv = _normalize(q_vecs[0])

            def query_sim_fn(ids: Sequence[int]) -> dict[int, float]:
                vecs = index.get_vectors(list(ids))  # stored normalized (cosine space)
                return {i: max(0.0, sum(a * b for a, b in zip(v, qv))) for i, v in zip(ids, vecs)}

            result = propagate_heat(
                dict(fused), store.neighbors, query_sim_fn,
                hops=opts.hops if opts.hops is not None else cfg.get("traversal.hops"),
                decay=cfg.get("traversal.decay"),
                query_floor=cfg.get("traversal.query_floor"),
                max_frontier=cfg.get("traversal.max_frontier"),
            )
            heat, trace = result.heat, result.trace

        candidates = sorted(set(fused) | set(heat))
        alpha, beta, gamma = (
            cfg.get("scoring.alpha_rerank"), cfg.get("scoring.beta_heat"), cfg.get("scoring.gamma_vector"),
        )

        # 4. cross-encoder rerank (capped shortlist by pre-score)
        rerank_scores: dict[int, float] | None = None
        if opts.rerank and reranker is not None:
            pre = combine(candidates, fused, heat, None, alpha=alpha, beta=beta, gamma=gamma)
            shortlist = sorted(candidates, key=lambda c: (-pre.get(c, 0.0), c))[:RERANK_CAP]
            by_id = {c.id: c for c in store.get_chunks(shortlist)}
            ordered = [cid for cid in shortlist if cid in by_id]
            scores = reranker.score(text, [by_id[c].text for c in ordered])
            rerank_scores = {cid: float(s) for cid, s in zip(ordered, scores)}
            candidates = ordered  # unreranked stragglers would score 0 anyway

        # 5. combined scoring + output
        # every candidate is ranked so a file's chunks can be folded into one entry
        final = combine(candidates, fused, heat, rerank_scores, alpha=alpha, beta=beta, gamma=gamma)
        ranked_ids = sorted(candidates, key=lambda c: (-final.get(c, 0.0), c))
        by_id = {c.id: c for c in store.get_chunks(ranked_ids)}
        ranked = (
            ScoredChunk(
                chunk=by_id[cid],
                score=final.get(cid, 0.0),
                vector_score=fused.get(cid, 0.0),
                heat=heat.get(cid, 0.0),
                rerank_score=(rerank_scores or {}).get(cid, 0.0),
                explain=_explain(trace, cid) if opts.explain else None,
            )
            for cid in ranked_ids
            if cid in by_id
        )
        results = group_by_file(ranked, opts.top)
        attach_context(root, results, opts.before, opts.after)
        return QueryOutput(query=text, variants=variants, results=results)


def to_query_json(out: QueryOutput, *, max_chunk_chars: int = 1200) -> dict:
    """Serialize as the versioned agent-facing schema: one entry per file, its best
    chunks nested under `chunks`."""
    return {
        "schema": "ragx.query.v3",
        "query": out.query,
        "variants": out.variants,
        "results": [
            {
                "file": f.file_path,
                "score": round(f.score, 6),
                "chunks": [_chunk_json(r, max_chunk_chars) for r in f.chunks],
            }
            for f in out.results
        ],
    }


def _chunk_json(r: ScoredChunk, max_chunk_chars: int) -> dict:
    return {
        "chunk_id": r.chunk.id,
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
        **({"context": r.context} if r.context is not None else {}),
    }


def to_files_json(out: QueryOutput) -> dict:
    """File list without chunk text; same entries and scores as `to_query_json`."""
    files = [
        {"file": f.file_path, "score": round(f.score, 6), "chunk_ids": [r.chunk.id for r in f.chunks]}
        for f in out.results
    ]
    return {"schema": "ragx.files.v1", "query": out.query, "variants": out.variants, "files": files}
