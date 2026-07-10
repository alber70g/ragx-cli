"""HNSW-backed vector index (hnswlib), cosine space.

hnswlib doesn't expose a way to enumerate/count soft-deleted labels after
save/load, so we track the deleted-id set and capacity ourselves in a small
JSON sidecar next to the index file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import hnswlib

from ragx.core.errors import RagxError

_M = 16
_EF_CONSTRUCTION = 200


def _meta_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


class VectorIndex:
    """Wraps an hnswlib.Index with resize-on-full and soft-delete tracking."""

    def __init__(self, index: hnswlib.Index, path: Path, capacity: int, deleted: set[int]):
        self._index = index
        self._path = path
        self._capacity = capacity
        self._deleted = deleted

    @classmethod
    def create(cls, path: Path, dim: int, capacity: int = 2048) -> VectorIndex:
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=capacity, M=_M, ef_construction=_EF_CONSTRUCTION)
        return cls(index, path, capacity, set())

    @classmethod
    def load(cls, path: Path, dim: int) -> VectorIndex:
        if not path.exists():
            raise RagxError(f"vector index file not found: {path}")
        meta_path = _meta_path(path)
        capacity = 2048
        deleted: set[int] = set()
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            capacity = meta["capacity"]
            deleted = set(meta["deleted"])
        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(str(path), max_elements=capacity)
        return cls(index, path, capacity, deleted)

    def add(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        if not ids:
            return
        needed = self._index.get_current_count() + len(ids)
        while needed > self._capacity:
            self._capacity *= 2
        if self._capacity > self._index.get_max_elements():
            self._index.resize_index(self._capacity)
        self._index.add_items(list(vectors), list(ids))

    def search(self, vector: Sequence[float], k: int) -> list[tuple[int, float]]:
        active = self.count
        if active == 0 or k <= 0:
            return []
        k = min(k, active)
        ef = max(100, 2 * k)
        self._index.set_ef(ef)
        labels, distances = self._index.knn_query([vector], k=k)
        results = [
            (int(label), max(0.0, min(1.0, 1.0 - float(dist))))
            for label, dist in zip(labels[0], distances[0])
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def mark_deleted(self, ids: Sequence[int]) -> None:
        known_ids = set(self._index.get_ids_list())
        for i in ids:
            if i in known_ids and i not in self._deleted:
                self._index.mark_deleted(i)
                self._deleted.add(i)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(self._path))
        meta = {"capacity": self._capacity, "deleted": sorted(self._deleted)}
        _meta_path(self._path).write_text(json.dumps(meta))

    @property
    def count(self) -> int:
        return self._index.get_current_count() - len(self._deleted)
