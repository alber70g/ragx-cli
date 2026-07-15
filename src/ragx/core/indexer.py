"""Indexing pipeline: discover -> hash -> chunk -> embed -> vector index + store."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ragx.core.chunking import chunk_text, subchunk_texts
from ragx.core.communities import leiden_communities
from ragx.core.config import Config, db_path, vectors_path
from ragx.core.discovery import discover_files, hash_file
from ragx.core.errors import ManifestMismatchError, RagxError
from ragx.core.graph import affected_ids, knn_edges, subchunk_knn_edges
from ragx.core.models import ChunkDraft, FileRecord
from ragx.core.store import Store
from ragx.core.vectors import VectorIndex
from ragx.providers.base import Embedder

log = logging.getLogger("ragx.index")


@dataclass
class IndexStats:
    files_indexed: int = 0
    files_deleted: int = 0
    files_unchanged: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    edges_total: int = 0
    communities_total: int = 0


def run_index(root: Path, cfg: Config, embedder: Embedder, *, full: bool = False) -> IndexStats:
    """Index the corpus at `root`. Hash-diff incremental by default; rebuild when full."""
    edge_source = cfg.get("graph.edge_source")
    if edge_source not in ("chunk", "subchunk"):
        raise RagxError(f"graph.edge_source must be 'chunk' or 'subchunk', got {edge_source!r}")
    sub_size = cfg.get("graph.subchunk_size_tokens")
    size_tokens = cfg.get("chunking.size_tokens")
    overlap = cfg.get("chunking.overlap")
    db_path(root).parent.mkdir(parents=True, exist_ok=True)
    with Store(db_path(root)) as store:
        _check_manifest(
            store, embedder, edge_source, sub_size, size_tokens, overlap, allow_rewrite=full
        )
        dim = embedder.dimension()
        store.set_meta("embedding_model", embedder.model)
        store.set_meta("embedding_dim", str(dim))
        store.set_meta("edge_source", edge_source)
        store.set_meta("subchunk_size_tokens", str(sub_size))
        store.set_meta("chunk_size_tokens", str(size_tokens))
        store.set_meta("chunk_overlap", str(overlap))

        current = _discover_hashes(root, cfg)
        known = store.get_file_hashes()

        if full:
            to_index, to_delete = list(current), []
            for p in known:
                store.delete_file(p)
            vectors_path(root).unlink(missing_ok=True)
            index = _open_vectors(root, dim)
        else:
            to_index = [p for p, h in current.items() if known.get(p) != h]
            to_delete = [p for p in known if p not in current]
            index = _open_vectors(root, dim)

        stats = IndexStats(files_unchanged=len(current) - len(to_index))

        for path in to_delete:
            removed = store.delete_file(path)
            index.mark_deleted(removed)
            stats.files_deleted += 1
            stats.chunks_deleted += len(removed)

        new_ids: list[int] = []
        for path in sorted(to_index):
            if path in known:  # changed file: drop old chunks first
                removed = store.delete_file(path)
                index.mark_deleted(removed)
                stats.chunks_deleted += len(removed)
            text = (root / path).read_text(encoding="utf-8", errors="replace")
            drafts = chunk_text(text, path, size_tokens=size_tokens, overlap=overlap)
            store.upsert_file(FileRecord(path, current[path], (root / path).stat().st_mtime, len(drafts)))
            if not drafts:
                stats.files_indexed += 1
                continue
            ids = store.insert_chunks(path, drafts)
            vectors = embedder.embed_documents([d.text for d in drafts])
            index.add(ids, vectors)
            if edge_source == "subchunk":
                _embed_subchunks(store, embedder, ids, drafts, vectors, sub_size)
            new_ids.extend(ids)
            stats.files_indexed += 1
            stats.chunks_added += len(ids)
            log.info("indexed %s (%d chunks)", path, len(ids))

        if new_ids:
            k, min_sim = cfg.get("graph.k"), cfg.get("graph.min_edge_sim")
            if edge_source == "subchunk":
                edge_fn = _subchunk_edge_fn(store, index, dim, k, min_sim, cfg.get("graph.near_dup_sim"))
            else:
                def edge_fn(ids_: Sequence[int]) -> dict[int, list[tuple[int, float]]]:
                    return knn_edges(index, ids_, k, min_sim)
            edges = edge_fn(new_ids)
            touched = affected_ids(edges)
            if touched:  # incremental: refresh edge lists of pre-existing neighbors
                edges.update(edge_fn(sorted(touched)))
            for src, nbrs in edges.items():
                store.replace_edges(src, nbrs)
            log.info("graph: %d nodes re-edged", len(edges))

        assignments = leiden_communities(
            store.all_edges(),
            resolution=cfg.get("communities.resolution"),
            seed=cfg.get("communities.seed"),
        )
        store.replace_communities(assignments)
        stats.communities_total = len(set(assignments.values()))
        log.info("communities: %d over %d chunks", stats.communities_total, len(assignments))

        stats.edges_total = store.edge_count()
        index.save()
        return stats


def _check_manifest(
    store: Store,
    embedder: Embedder,
    edge_source: str,
    sub_size: int,
    size_tokens: int,
    overlap: float,
    *,
    allow_rewrite: bool,
) -> None:
    checks = [
        ("embedding_model", embedder.model),
        ("edge_source", edge_source),
        ("chunk_size_tokens", str(size_tokens)),
        ("chunk_overlap", str(overlap)),
    ]
    if edge_source == "subchunk":  # size only shapes the graph when subchunk edges are on
        checks.append(("subchunk_size_tokens", str(sub_size)))
    for key, current in checks:
        prev = store.get_meta(key)
        if prev and prev != str(current) and not allow_rewrite:
            raise ManifestMismatchError(
                f"index was built with {key}={prev!r} but config says {current!r}; "
                "run `ragx-cli index --full` to rebuild"
            )


def _discover_hashes(root: Path, cfg: Config) -> dict[str, str]:
    return {
        rel: hash_file(root / rel)
        for rel in discover_files(
            root,
            cfg.get("corpus.include"),
            cfg.get("corpus.exclude"),
            cfg.get("corpus.respect_gitignore"),
        )
    }


def corpus_drift(root: Path, cfg: Config) -> dict[str, int]:
    """Diff the corpus on disk against the stored file hashes. Read-only."""
    current = _discover_hashes(root, cfg)
    known: dict[str, str] = {}
    if db_path(root).exists():
        with Store(db_path(root)) as store:
            known = store.get_file_hashes()
    return {
        "new": sum(1 for p in current if p not in known),
        "changed": sum(1 for p, h in current.items() if p in known and known[p] != h),
        "deleted": sum(1 for p in known if p not in current),
    }


def _unit(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _embed_subchunks(
    store: Store,
    embedder: Embedder,
    ids: Sequence[int],
    drafts: Sequence[ChunkDraft],
    chunk_vectors: Sequence[Sequence[float]],
    sub_size: int,
) -> None:
    """Split each chunk into sub-chunks and persist their unit vectors.
    A chunk that stays whole reuses its own embedding — no extra provider call."""
    to_embed: list[str] = []
    spans: list[tuple[int, int, int]] = []  # (chunk_id, start, end) into to_embed
    for cid, draft, cvec in zip(ids, drafts, chunk_vectors):
        subs = subchunk_texts(draft.text, sub_size)
        if len(subs) == 1:
            store.insert_subchunk_vectors(cid, [_unit(cvec)])
        else:
            spans.append((cid, len(to_embed), len(to_embed) + len(subs)))
            to_embed.extend(subs)
    if to_embed:
        vectors = embedder.embed_documents(to_embed)
        for cid, start, end in spans:
            store.insert_subchunk_vectors(cid, [_unit(v) for v in vectors[start:end]])


def _subchunk_edge_fn(
    store: Store, index: VectorIndex, dim: int, k: int, min_sim: float, near_dup_sim: float
):
    """Load the sub-chunk matrix + whole-chunk vectors once; return an edge function."""
    rows = store.all_subchunk_vectors()
    parents = [cid for cid, _ in rows]
    matrix = (
        np.frombuffer(b"".join(blob for _, blob in rows), dtype=np.float32).reshape(len(rows), dim)
        if rows
        else np.zeros((0, dim), dtype=np.float32)
    )
    chunk_ids = sorted(set(parents))
    if len(chunk_ids) != store.chunk_count():
        raise RagxError(
            "sub-chunk vectors missing for some chunks (index predates graph.edge_source="
            "'subchunk'); run `ragx-cli index --full` to rebuild"
        )
    chunk_vectors = dict(zip(chunk_ids, index.get_vectors(chunk_ids)))

    def edge_fn(ids_: Sequence[int]) -> dict[int, list[tuple[int, float]]]:
        return subchunk_knn_edges(parents, matrix, ids_, chunk_vectors, k, min_sim, near_dup_sim)

    return edge_fn


def _open_vectors(root: Path, dim: int) -> VectorIndex:
    vp = vectors_path(root)
    return VectorIndex.load(vp, dim) if vp.exists() else VectorIndex.create(vp, dim)
