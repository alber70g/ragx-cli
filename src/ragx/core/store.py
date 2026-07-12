"""Single-file SQLite store for files, chunks, edges, and a key-value manifest."""

from __future__ import annotations

import sqlite3
from array import array
from pathlib import Path
from typing import Sequence

from ragx.core.models import Chunk, ChunkDraft, FileRecord

# v2 adds subchunks (sub-chunk embedding vectors for graph edge construction)
SUBCHUNK_SCHEMA = """
CREATE TABLE subchunks(id INTEGER PRIMARY KEY AUTOINCREMENT,
                       chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                       vector BLOB NOT NULL);
CREATE INDEX subchunks_chunk ON subchunks(chunk_id);
"""

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE files(path TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
                   mtime REAL NOT NULL, chunk_count INTEGER NOT NULL);
CREATE TABLE chunks(id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                    text TEXT NOT NULL, byte_start INTEGER NOT NULL, byte_end INTEGER NOT NULL,
                    line_start INTEGER NOT NULL, line_end INTEGER NOT NULL);
CREATE INDEX chunks_file ON chunks(file_path);
CREATE TABLE edges(src INTEGER NOT NULL, dst INTEGER NOT NULL, weight REAL NOT NULL,
                   PRIMARY KEY(src, dst)) WITHOUT ROWID;
CREATE INDEX edges_dst ON edges(dst);
""" + SUBCHUNK_SCHEMA


class Store:
    def __init__(self, path: Path):
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA foreign_keys = ON")
        user_version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if user_version == 0:
            self.conn.executescript(SCHEMA)
            self.conn.execute("PRAGMA user_version = 2")
            self.conn.commit()
        elif user_version == 1:
            self.conn.executescript(SUBCHUNK_SCHEMA)
            self.conn.execute("PRAGMA user_version = 2")
            self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- manifest -----------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    # -- files ----------------------------------------------------------

    def get_file_hashes(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT path, content_hash FROM files").fetchall()
        return dict(rows)

    def upsert_file(self, rec: FileRecord) -> None:
        self.conn.execute(
            "INSERT INTO files(path, content_hash, mtime, chunk_count) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(path) DO UPDATE SET content_hash = excluded.content_hash, "
            "mtime = excluded.mtime, chunk_count = excluded.chunk_count",
            (rec.path, rec.content_hash, rec.mtime, rec.chunk_count),
        )
        self.conn.commit()

    def delete_file(self, path: str) -> list[int]:
        ids = [
            row[0]
            for row in self.conn.execute("SELECT id FROM chunks WHERE file_path = ?", (path,))
        ]
        if ids:
            placeholders = ",".join("?" * len(ids))
            self.conn.execute(f"DELETE FROM edges WHERE src IN ({placeholders})", ids)
            self.conn.execute(f"DELETE FROM edges WHERE dst IN ({placeholders})", ids)
        self.conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self.conn.commit()
        return ids

    def file_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    # -- chunks ------------------------------------------------------

    def insert_chunks(self, file_path: str, drafts: Sequence[ChunkDraft]) -> list[int]:
        ids = []
        for d in drafts:
            cur = self.conn.execute(
                "INSERT INTO chunks(file_path, text, byte_start, byte_end, line_start, line_end) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (file_path, d.text, d.byte_start, d.byte_end, d.line_start, d.line_end),
            )
            ids.append(cur.lastrowid)
        self.conn.commit()
        return ids

    def get_chunks(self, ids: Sequence[int]) -> list[Chunk]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT id, file_path, text, byte_start, byte_end, line_start, line_end "
            f"FROM chunks WHERE id IN ({placeholders})",
            list(ids),
        ).fetchall()
        by_id = {row[0]: Chunk(*row) for row in rows}
        return [by_id[i] for i in ids if i in by_id]

    def chunk_ids_for_file(self, path: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT id FROM chunks WHERE file_path = ? ORDER BY id", (path,)
        ).fetchall()
        return [row[0] for row in rows]

    def all_chunk_ids(self) -> list[int]:
        rows = self.conn.execute("SELECT id FROM chunks ORDER BY id").fetchall()
        return [row[0] for row in rows]

    def chunk_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # -- subchunks (native float32 vector blobs; rows cascade with their chunk) --

    def insert_subchunk_vectors(self, chunk_id: int, vectors: Sequence[Sequence[float]]) -> None:
        self.conn.executemany(
            "INSERT INTO subchunks(chunk_id, vector) VALUES (?, ?)",
            [(chunk_id, array("f", v).tobytes()) for v in vectors],
        )
        self.conn.commit()

    def all_subchunk_vectors(self) -> list[tuple[int, bytes]]:
        """(chunk_id, float32 blob) for every sub-chunk, ordered by chunk then insertion."""
        return self.conn.execute(
            "SELECT chunk_id, vector FROM subchunks ORDER BY chunk_id, id"
        ).fetchall()

    # -- edges (undirected; stored normalized src < dst) -----------------

    def replace_edges(self, src: int, neighbors: Sequence[tuple[int, float]]) -> None:
        self.conn.execute("DELETE FROM edges WHERE src = ? OR dst = ?", (src, src))
        for dst, weight in neighbors:
            lo, hi = (src, dst) if src < dst else (dst, src)
            self.conn.execute(
                "INSERT INTO edges(src, dst, weight) VALUES (?, ?, ?) "
                "ON CONFLICT(src, dst) DO UPDATE SET weight = excluded.weight",
                (lo, hi, weight),
            )
        self.conn.commit()

    def neighbors(self, chunk_id: int) -> list[tuple[int, float]]:
        rows = self.conn.execute(
            "SELECT dst AS other, weight FROM edges WHERE src = ? "
            "UNION ALL "
            "SELECT src AS other, weight FROM edges WHERE dst = ?",
            (chunk_id, chunk_id),
        ).fetchall()
        return sorted(((other, weight) for other, weight in rows), key=lambda t: -t[1])

    def delete_edges_incident(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        placeholders = ",".join("?" * len(ids))
        self.conn.execute(f"DELETE FROM edges WHERE src IN ({placeholders})", list(ids))
        self.conn.execute(f"DELETE FROM edges WHERE dst IN ({placeholders})", list(ids))
        self.conn.commit()

    def edge_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
