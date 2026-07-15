"""Query pipeline: expansion -> fan-out retrieval -> RRF -> heat traversal -> rerank -> combine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ragx.core.config import Config, db_path, vectors_path
from ragx.core.errors import RagxError
from ragx.core.expansion import expand_query
from ragx.core.fusion import rrf
from ragx.core.models import QueryOutput, ScoredChunk
from ragx.core.scoring import combine
from ragx.core.store import Store
from ragx.core.traversal import propagate_heat
from ragx.core.vectors import VectorIndex
from ragx.providers.base import Embedder, Generator, Reranker

RERANK_CAP = 100  # cross-encoder cost guard: rerank at most this many candidates


@dataclass
class QueryOptions:
    top: int = 8
    files_only: bool = False
    expand: bool = True
    graph: bool = True
    rerank: bool = True
    hops: int | None = None
    explain: bool = False


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _check_model(store: Store, embedder: Embedder) -> None:
    built_with = store.get_meta("embedding_model")
    if built_with != embedder.model:
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
        _check_model(store, embedder)
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
        final = combine(candidates, fused, heat, rerank_scores, alpha=alpha, beta=beta, gamma=gamma)
        top_ids = sorted(candidates, key=lambda c: (-final.get(c, 0.0), c))[: opts.top]
        by_id = {c.id: c for c in store.get_chunks(top_ids)}
        results = [
            ScoredChunk(
                chunk=by_id[cid],
                score=final.get(cid, 0.0),
                vector_score=fused.get(cid, 0.0),
                heat=heat.get(cid, 0.0),
                rerank_score=(rerank_scores or {}).get(cid, 0.0),
                explain=trace.get(cid) if opts.explain else None,
            )
            for cid in top_ids
            if cid in by_id
        ]
        return QueryOutput(query=text, variants=variants, results=results)


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
