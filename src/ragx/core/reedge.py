"""Rebuild sub-chunks, graph edges and communities from stored chunks, without re-chunking
or re-embedding files. Backs `ragx-cli index --edges-only`, for switching `graph.edge_source`
(or `subchunk_size_tokens`) on an existing index without paying for a full re-embed."""

from __future__ import annotations

import logging
from pathlib import Path

from ragx.core.communities import leiden_communities
from ragx.core.config import Config, db_path
from ragx.core.errors import ManifestMismatchError, RagxError
from ragx.core.graph import knn_edges
from ragx.core.indexer import IndexStats, _embed_subchunks, _open_vectors, _subchunk_edge_fn
from ragx.core.store import Store
from ragx.providers.base import Embedder

log = logging.getLogger("ragx.index")

_BATCH = 300


def rebuild_edges(root: Path, cfg: Config, embedder: Embedder) -> IndexStats:
    """Rebuild sub-chunks (if `graph.edge_source == "subchunk"`), graph edges and communities
    from the chunks already stored at `root`, per the current [graph] config. Chunk embeddings
    and the vector index are untouched: a chunk that stays whole reuses its own stored vector —
    no provider call re-embeds it."""
    edge_source = cfg.get("graph.edge_source")
    if edge_source not in ("chunk", "subchunk"):
        raise RagxError(f"graph.edge_source must be 'chunk' or 'subchunk', got {edge_source!r}")
    sub_size = cfg.get("graph.subchunk_size_tokens")
    size_tokens = cfg.get("chunking.size_tokens")
    overlap = cfg.get("chunking.overlap")

    if not db_path(root).exists():
        raise RagxError("no index found — run `ragx-cli index` first")
    with Store(db_path(root)) as store:
        if store.chunk_count() == 0:
            raise RagxError("no index found — run `ragx-cli index` first")
        _check_manifest(store, embedder, size_tokens, overlap)

        dim = embedder.dimension()
        index = _open_vectors(root, dim)
        store.clear_subchunks()
        store.clear_edges()

        all_ids = store.all_chunk_ids()
        k, min_sim = cfg.get("graph.k"), cfg.get("graph.min_edge_sim")
        embedded = reused = 0
        if edge_source == "subchunk":
            for i in range(0, len(all_ids), _BATCH):
                batch = all_ids[i : i + _BATCH]
                texts = [c.text for c in store.get_chunks(batch)]
                vectors = index.get_vectors(batch)
                n_embed, n_reuse = _embed_subchunks(store, embedder, batch, texts, vectors, sub_size)
                embedded += n_embed
                reused += n_reuse
            edge_fn = _subchunk_edge_fn(store, index, dim, k, min_sim, cfg.get("graph.near_dup_sim"))
        else:
            def edge_fn(ids_: list[int]) -> dict[int, list[tuple[int, float]]]:
                return knn_edges(index, ids_, k, min_sim)

        edges = edge_fn(all_ids)
        for src, nbrs in edges.items():
            store.replace_edges(src, nbrs)

        assignments = leiden_communities(
            store.all_edges(),
            resolution=cfg.get("communities.resolution"),
            seed=cfg.get("communities.seed"),
        )
        store.replace_communities(assignments)

        store.set_meta("edge_source", edge_source)
        store.set_meta("subchunk_size_tokens", str(sub_size))

        stats = IndexStats(
            edges_total=store.edge_count(),
            communities_total=len(set(assignments.values())),
        )
        if edge_source == "subchunk":
            log.info("sub-chunks: %d embedded, %d reused", embedded, reused)
        log.info(
            "edges-only rebuild: %d chunks, %d edges, %d communities",
            len(all_ids), stats.edges_total, stats.communities_total,
        )
        return stats


def _check_manifest(store: Store, embedder: Embedder, size_tokens: int, overlap: float) -> None:
    """embedding_model/chunk_size_tokens/chunk_overlap must match: edges are rebuilt from
    stored chunk texts and reused chunk embeddings, which are only valid for the embedder and
    chunking the index was built with. edge_source/subchunk_size_tokens are NOT checked here —
    changing them is the point of --edges-only."""
    checks = [
        ("embedding_model", embedder.model),
        ("chunk_size_tokens", str(size_tokens)),
        ("chunk_overlap", str(overlap)),
    ]
    for key, current in checks:
        prev = store.get_meta(key)
        if prev and prev != str(current):
            raise ManifestMismatchError(
                f"index was built with {key}={prev!r} but config says {current!r}; "
                "run `ragx-cli index --full` to rebuild"
            )
