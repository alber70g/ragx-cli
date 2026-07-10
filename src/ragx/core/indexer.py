"""Indexing pipeline: discover -> hash -> chunk -> embed -> vector index + store."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ragx.core.chunking import chunk_text
from ragx.core.config import Config, db_path, vectors_path
from ragx.core.discovery import discover_files, hash_file
from ragx.core.errors import ManifestMismatchError
from ragx.core.models import FileRecord
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


def run_index(root: Path, cfg: Config, embedder: Embedder, *, changed_only: bool = False) -> IndexStats:
    """Index the corpus at `root`. Full rebuild by default; hash-diff when changed_only."""
    with Store(db_path(root)) as store:
        _check_manifest(store, embedder, allow_rewrite=not changed_only)
        dim = embedder.dimension()
        store.set_meta("embedding_model", embedder.model)
        store.set_meta("embedding_dim", str(dim))

        current = {
            rel: hash_file(root / rel)
            for rel in discover_files(
                root,
                cfg.get("corpus.include"),
                cfg.get("corpus.exclude"),
                cfg.get("corpus.respect_gitignore"),
            )
        }
        known = store.get_file_hashes()

        if changed_only:
            to_index = [p for p, h in current.items() if known.get(p) != h]
            to_delete = [p for p in known if p not in current]
            index = _open_vectors(root, dim)
        else:
            to_index, to_delete = list(current), []
            for p in known:
                store.delete_file(p)
            vectors_path(root).unlink(missing_ok=True)
            index = _open_vectors(root, dim)

        stats = IndexStats(files_unchanged=len(current) - len(to_index))

        for path in to_delete:
            removed = store.delete_file(path)
            index.mark_deleted(removed)
            stats.files_deleted += 1
            stats.chunks_deleted += len(removed)

        size_tokens = cfg.get("chunking.size_tokens")
        overlap = cfg.get("chunking.overlap")
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
            stats.files_indexed += 1
            stats.chunks_added += len(ids)
            log.info("indexed %s (%d chunks)", path, len(ids))

        index.save()
        return stats


def _check_manifest(store: Store, embedder: Embedder, *, allow_rewrite: bool) -> None:
    prev = store.get_meta("embedding_model")
    if prev and prev != embedder.model and not allow_rewrite:
        raise ManifestMismatchError(
            f"index was built with {prev!r} but config says {embedder.model!r}; "
            "run a full `ragx index` to rebuild"
        )


def _open_vectors(root: Path, dim: int) -> VectorIndex:
    vp = vectors_path(root)
    return VectorIndex.load(vp, dim) if vp.exists() else VectorIndex.create(vp, dim)
