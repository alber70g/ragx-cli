"""Core data types shared across modules. Frozen where identity matters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk produced by a chunker, before it has an id or embedding."""

    text: str
    byte_start: int
    byte_end: int
    line_start: int  # 1-based, inclusive
    line_end: int  # 1-based, inclusive


@dataclass(frozen=True)
class Chunk:
    """A stored chunk. `id` is the SQLite rowid and doubles as the HNSW label."""

    id: int
    file_path: str  # relative to corpus root, POSIX separators
    text: str
    byte_start: int
    byte_end: int
    line_start: int
    line_end: int


@dataclass(frozen=True)
class FileRecord:
    path: str  # relative to corpus root, POSIX separators
    content_hash: str  # xxhash64 hexdigest
    mtime: float
    chunk_count: int


@dataclass(frozen=True)
class Edge:
    """Undirected similarity edge. Stored normalized with src < dst."""

    src: int
    dst: int
    weight: float  # cosine similarity in [0, 1]


@dataclass
class ScoredChunk:
    """A query result candidate with its score breakdown."""

    chunk: Chunk
    score: float = 0.0
    vector_score: float = 0.0  # fused RRF score (or raw cosine in no-fusion mode)
    heat: float = 0.0
    rerank_score: float = 0.0
    explain: dict | None = None  # traversal trace: seeds, edge paths, weights


@dataclass
class QueryOutput:
    """Top-level result of the query pipeline; serialized as ragx.query.v1."""

    query: str
    variants: list[str] = field(default_factory=list)
    results: list[ScoredChunk] = field(default_factory=list)
