# Plan: zvec as the vector storage engine (replacing hnswlib)

**Status:** on ice, not started · **Date:** 2026-10-28 · **Related:**
`docs/plan-hybrid-search-bm25-lexical-leg-plus-dense-embeddings.md`

**On ice (decided 2026-10-28):** only the hybrid search plan is being built. This plan has no
start date and no trigger. The hybrid plan keeps its keyword index in SQLite FTS5, so nothing
from the hybrid work moves into zvec.

## Summary

Replace the hnswlib index file (`.ragx/vectors.hnsw` + `vectors.hnsw.meta.json` sidecar) with a
zvec collection at `.ragx/vectors.zvec/`. SQLite (`index.db`) stays the store for files, chunks,
edges, sub-chunk vectors, communities and the manifest.

Expected retrieval-quality change: **none**. Both engines run HNSW with cosine. This is an engine
swap. The gate is parity, not improvement.

## Reasons for and against

Facts marked *verified* were checked on 2026-10-28 with `zvec==0.8.0` in a scratch venv
(Python 3.12, macOS arm64). Everything else is from zvec's README or docs and has not been
tested here.

Reasons for:

1. hnswlib's last release is 0.8.0 (2023-12-03), and PyPI has only an sdist. Every
   `pip install ragx-cli` compiles C++. zvec 0.8.0 ships wheels for macOS arm64/x86_64, Linux
   glibc/musl, and Windows, cp310–cp314 (verified on PyPI).
2. Real deletes (verified). The sidecar for soft-deleted ids and the capacity-doubling
   `resize_index` code both go away.
3. WAL-based crash recovery (README claim, not tested).
4. Built-in full-text search (BM25), sparse vectors and scalar filters in the same engine.
   Filters would serve `--since/--until` from `docs/feature-temporal-weighting.md`.
5. Active project: repo created 2025-12-05, last push 2026-10-28, ~16k stars, Apache-2.0.

Reasons against:

1. **Exclusive writer lock (verified).** While one process has the collection open read-write,
   a second process cannot open it, not even read-only:
   `RuntimeError: Can't lock read-only collection: …/LOCK`. Read-only handles can be shared,
   but an open read-only handle blocks a writer. hnswlib has no lock today, so `ragx-cli web`
   and `ragx-cli index` can run at the same time.
2. String document ids. ragx chunk ids are ints (SQLite rowids that double as hnswlib labels).
3. COSINE scores come back as **distance**, where 0 means identical (verified). ragx expects a
   similarity clamped to [0, 1].
4. `fetch` silently leaves out unknown ids (verified). `VectorIndex.get_vectors` must raise
   `RagxError` on an unknown id (`tests/test_vectors.py`).
5. 0.x API churn. The zvec skill bundled in this repo already disagrees with 0.8.0:
   `VectorQuery` is deprecated in favour of `Query`, and `WeightedReRanker` now takes a list of
   weights instead of a dict plus `topn`.
6. Disk footprint: 5.2 MB for a 2-document collection, because of the RocksDB base files
   (verified). Wheel size not checked.
7. After inserts, `stats.index_completeness.dense` was `0.0` (verified). New vectors sit in a
   flat buffer until `optimize()` runs.

At the current corpus scale (1–5k chunks), reasons 2–5 "for" don't fix any measured problem.
The concrete gains are the install path (1) and the deletes (2).

## Scope

In: the `VectorIndex` backend, migration of existing indexes, lock handling, tests, README and
CLAUDE.md.

Out:
- Sub-chunk vectors. They stay as float32 BLOBs in SQLite; they're loaded as an in-memory numpy
  matrix and never searched through HNSW.
- Hybrid search. See the related plan.
- Quantization, scalar filters, DiskANN.

## Design

**Interface stays the same.** `VectorIndex` exposes `create / load / add / search / get_vectors /
mark_deleted / save / count`. Its callers are `core/indexer.py`, `core/query.py`, `core/graph.py`,
`core/reedge.py` and the tests. A new `core/vectors_zvec.py` implements the same methods. Config
key `vectors.backend = "hnswlib" | "zvec"` (added to `DEFAULTS` first) picks the backend. The
dual-backend code exists only for the parity period and is removed in phase 4.

**Collection schema.** One vector field `dense`: `VECTOR_FP32`, `dimension = embedding_dim`,
`HnswIndexParam(metric_type=COSINE, M=16, ef_construction=200)`. These match today's constants in
`vectors.py`; zvec's default `ef_construction` is 100. No scalar fields in this plan.

**Method mapping:**

| `VectorIndex` | zvec 0.8.0 |
|---|---|
| ids | `str(chunk_id)` going in, `int(doc.id)` coming out, converted only inside the wrapper |
| `search(v, k)` | `query(Query(field_name="dense", vector=v, param=HnswQueryParam(ef=max(100, 2k))), topk=k)`; `sim = clamp(1 - score, 0, 1)`; sort descending |
| `get_vectors(ids)` | `fetch(ids, include_vector=True)`; raise `RagxError` if any id is missing |
| `mark_deleted(ids)` | `delete(ids)`; unknown ids must not raise (current contract) |
| `save()` | `flush()`, then `optimize()` at the end of `run_index` / `rebuild_edges` |
| `count` | `stats.doc_count` |

**Handle lifetime and locks.**
- `run_index` and `rebuild_edges` open read-write and close in a `finally`.
- `run_query` opens read-only for each call and closes afterwards. `web/query_runner.py` calls
  `run_query` per request, so the web server holds no long-lived handle.
- A lock error (`RuntimeError` containing `Can't lock`) becomes
  `RagxError("vector index is locked by another ragx process (index running?) — retry when it finishes")`,
  exit 2. Today a query during `index` reads the previous `vectors.hnsw`; with zvec it fails loud
  instead.
- `index --full` must close the handle before `shutil.rmtree` on the directory. Deleting the
  directory under an open handle produced RocksDB flush errors in the spike.

**Manifest.** New meta key `vector_backend`. If it doesn't match the configured backend,
`run_query` fails loud and names both values plus the fix, the same way an `embedding_model`
mismatch is handled.

**Migration without re-embedding.** For every chunk id in SQLite: `hnsw.get_vectors(batch)`,
then insert into zvec, set `vector_backend`, and leave `vectors.hnsw` in place as a backup. No
embedding calls are needed, so this takes seconds. How it's triggered is open question 1.

**stdout contract.** zvec logs from C++. The agent contract requires stdout to be exactly one
JSON document. Call `zvec.init(log_level=LogLevel.ERROR)` once at import, and verify in a CLI
test that stdout stays clean. The spike didn't establish whether zvec logs to stdout or stderr.

**Dependency.** Add `zvec>=0.8,<0.9` to `pyproject.toml`, pinned below 0.9 because of the API
churn. hnswlib stays until phase 4.

## Tasks

| # | Step | Verify |
|---|---|---|
| 0 | Finish the spike: zvec log stream (stdout or stderr), `delete` on unknown id, whether `optimize()` is sync or async, read-only open latency, wheel size | Numbers and answers added to this doc |
| 1 | `vectors.backend` in `DEFAULTS`; `core/vectors_zvec.py`; backend selection in `indexer`, `query`, `reedge` | `uv run pytest -q` green, `ruff` clean |
| 2 | Parametrize `tests/test_vectors.py` and `tests/test_graph.py` over both backends. The tests encode the caller contracts: similarity in [0, 1] sorted descending, unknown id raises, deleted ids never returned, persistence across reopen | Both backends pass the same tests |
| 3 | New tests: lock contention gives `RagxError` and exit 2; manifest mismatch fails loud; stdout of `ragx-cli query` parses as one JSON document with the zvec backend | Tests fail when the mapping is removed |
| 4 | Migration hnswlib → zvec, plus test: vectors equal within fp32 epsilon, and the same top-k on a fixture | Test green |
| 5 | Parity eval on a **copy** of the notes eval corpus (never in place; never `cp` into an existing backup dir): `ragx-cli eval`, both backends, same chunks | r@5 / r@10 / MRR within ±0.01; per-query top-20 overlap ≥ 95%; index time and query p50 ≤ 1.2× hnswlib |
| 6 | If step 5 fails: stop and record the verdict in `research/`. If it passes: flip the default to `zvec` and auto-migrate on the first `index` | Release notes, README, CLAUDE.md updated |
| 7 | One release later: remove hnswlib, the sidecar code and `vectors.backend` | `hnswlib` gone from `pyproject.toml`; tests green |

## Open questions

1. Migration trigger: automatic on the first `index` after the backend flips (recommended; same
   pattern as the legacy `.ragx/config.toml` self-heal), or an explicit flag such as
   `index --migrate-vectors`?
2. After phase 4: remove hnswlib completely (recommended), or keep it as an optional backend?
3. Do you run `ragx-cli web` and `ragx-cli index` at the same time? If yes, the lock needs more
   than a fail-loud error, for example `index` writing to a new directory and swapping it in
   atomically.
