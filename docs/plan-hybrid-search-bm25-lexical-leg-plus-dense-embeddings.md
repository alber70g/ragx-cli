# Plan: hybrid search — BM25 keyword search plus dense embeddings

**Status:** ready for implementation, not started · **Date:** 2026-10-28 · **Builds on:**
`research/bge-m3-sparse-lexical-weights-hybrid-retrieval-leg-for-ragx-vs-sqlite-fts5-bm25.md`
(2026-07-12; that doc's phase 1, changed to on-by-default with a weight) · **Related:**
`docs/plan-zvec-vector-storage-engine-replacing-hnswlib.md`

## Summary

`ragx-cli query` always runs hybrid search: a dense vector search and a BM25 keyword search,
fused at the seed step (`core/query.py`, step 2) with a vector weight `w`, default `0.5`.
Traversal, rerank and `combine` stay unchanged. Keyword backend: SQLite FTS5 with the `trigram`
tokenizer inside `index.db`.

| Flag | Vector weight `w` | Keyword weight `1 − w` |
|---|---|---|
| none | `fusion.vector_weight` (default `0.5`) | `0.5` |
| `--weight 0.6` | 0.6 | 0.4 |
| `--weight 0.3` | 0.3 | 0.7 |
| `--vector` | 1.0 (keyword search skipped; today's behaviour) | 0 |
| `--keyword` | 0 (vector search skipped) | 1.0 |

The fusion happens at the seeds because that's the only point where a chunk that dense search
missed can enter the candidate set. A term in `scoring.combine` could only reorder chunks that
are already candidates.

The build doesn't wait for an eval. Validation runs after the build (task 9), and the `0.5`
default is a product decision, not a measured value.

## Backend choice

Checked on 2026-10-28 with the query "aangifte" against the chunk "de belastingaangifte voor de
zorgverzekering":

| Backend | Matches a compound part? | Notes |
|---|---|---|
| SQLite FTS5 `trigram` | yes (verified) | Needs SQLite ≥ 3.34; local is 3.53.1. Terms shorter than 3 chars can't match |
| SQLite FTS5 `unicode61` | no (verified) | Matches whole words only |
| zvec FTS `ngram` (3–3) | yes (verified) | Only if the zvec plan lands; puts chunk text in a second store |
| zvec FTS `standard` | no (verified) | |
| zvec `BM25EmbeddingFunction` (DashText) | not tested | Built-in vocabularies are `zh`/`en` only; `dashtext` 0.0.7 has no cp313+ wheels |
| BGE-M3 `lexical_weights` | not tested | ~2.3 GB PyTorch model at index and query time; research phase 2 |

**Choice: FTS5 trigram**, for four reasons:
1. Keyword rows live in the same SQLite transaction as the chunk rows, so they stay consistent.
2. No new dependency.
3. No interaction with zvec's writer lock.
4. It works whether or not the zvec plan lands.

zvec's single-call hybrid query doesn't fit here. ragx runs its own fusion, graph traversal and
`--explain` trace per ranking.

## Design

### Keyword index

**Schema v4** (`store.py`, `PRAGMA user_version` 3 → 4). An external-content table, so chunk
text isn't stored twice:

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id',
                                           tokenize='trigram');
CREATE TRIGGER chunks_fts_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text); END;
CREATE TRIGGER chunks_fts_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text); END;
```

- **Triggers keep it in sync.** Chunk rows are deleted only through `files … ON DELETE CASCADE`;
  `Store.delete_file` never deletes chunks directly. Verified 2026-10-28: the `AFTER DELETE`
  trigger fires on the cascade, and FTS5's `integrity-check` passes afterwards. Chunks are
  never updated, so no update trigger is needed.
- **Migration from v3** runs on `Store` open: the statements above, then
  `INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')`. Existing indexes get keyword search on
  upgrade without running `index`. The rebuild time on the notes corpus isn't measured yet
  (task 1).
- **No manifest guard.** The tokenizer is fixed at `trigram`. If a tokenizer setting is ever
  added, it gets a guard like `edge_source`.

### Keyword search

New method `Store.fts_search(text, top) -> list[int]`:
1. Split the query text on whitespace.
2. Drop tokens shorter than 3 characters.
3. Quote each remaining token (`"` doubled inside) and join them with ` OR `.
4. `SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?`.
   `bm25()` is negative, and lower means a better match.
5. If no tokens are left, return `[]`.

The keyword search uses the original query text only. Expansion variants and the HyDE passage
feed the dense side only.

### Weighted fusion

New pure function in `core/fusion.py`:

```
fuse(dense_rankings, keyword_ranking, w, k=60) -> dict[int, float]
  dense[c]   = rrf(dense_rankings, k)[c] / len(dense_rankings)   # mean, so 5 variants weigh as 1
  keyword[c] = rrf([keyword_ranking], k)[c]
  fused[c]   = w · dense[c] + (1 − w) · keyword[c]
```

- **The weight is a share of rank-based scores.** A chunk at rank `r` in both lists gets
  `w/(k+r) + (1−w)/(k+r)`. Ranks are used instead of raw scores because BM25 scores are
  unbounded and depend on the query, while cosine is bounded. Rank fusion needs no calibration
  between the two.
- **Dense rankings are averaged.** With `--expand` there are up to 5 dense rankings (query,
  3 variants, HyDE) and 1 keyword ranking. Summing them would give dense about 5/6 of the mass
  at `w = 0.5`. Averaging makes `0.5` an actual 50/50.
- **A side with weight 0 adds no candidates.** At `w = 0`, chunks from the dense lists would
  otherwise enter `fused` with score 0 and become candidates.
- **`w = 1` reproduces today's ranking exactly.** Dividing by `len(dense_rankings)` scales every
  score by the same constant, and both `propagate_heat` (max-aggregation) and `combine` (min-max
  normalization) are unaffected by a constant scale. Without `--expand`, `len = 1`, so even the
  raw values are identical. Task 3 tests this invariant.

`run_query` changes:
- `w == 0`: skip the HNSW search. The query is still embedded when the graph is on, because
  traversal's `query_floor` needs `cos(query, chunk)`. With `--keyword --no-graph` no embedding
  call is made at all, so keyword-only search works without the embedding server.
- `w == 1`: skip `fts_search`.
- `--keyword` with `--expand`: skip expansion (saves the LLM call) and log a warning to stderr.
  Expansion only feeds the dense side.
- `fused` replaces today's `rrf(...)` result everywhere: heat seeds, the γ term in `combine`,
  and the rerank shortlist. `scoring.gamma_vector` keeps its name but now weights the fused seed
  score. README documents this.

### Flags and config

- Config, added to `DEFAULTS` first: `fusion.vector_weight = 0.5`, `fusion.keyword_top = 20`.
  `keyword_top` matches `fusion.per_query_top`.
- `query` flags: `--vector`, `--keyword`, `--weight FLOAT`. They are mutually exclusive; more
  than one is a `RagxError` (exit 2). `--weight` must be within `[0, 1]`. `--weight 1` equals
  `--vector`, and `--weight 0` equals `--keyword`.
- `QueryOptions.vector_weight: float | None`, where `None` means use the config value.
- `web/query_runner.py`: `/api/query` accepts `weight` (0–1). A UI control in `web/static/` is
  out of scope here, because that code is frontend-owned.
- `ragx-cli eval`: `--weight` override applied to every selected config. `--weight 1`
  reproduces the historic benchmark numbers.

### Output

- Per-result scores: `vector` becomes the dense part only, and a new `keyword` field holds the
  keyword part. Both are unweighted. The combined `score` is unchanged in meaning.
- `--explain`: each seed gets `sources: ["vector"]`, `["keyword"]` or both.
- **Schema bump to `ragx.query.v4`.** Default results change, and `vector` changes meaning (it
  was the fused RRF score).

## Tasks

| # | Step | Verify |
|---|---|---|
| 1 | Schema v4, triggers and v3→v4 migration in `store.py` | Tests: insert and cascade delete keep `chunks_fts` in sync; migration backfills an existing v3 db; `integrity-check` passes. Rebuild time on a copy of the notes corpus noted in this doc |
| 2 | `Store.fts_search` with query escaping | Tests: "aangifte" finds "belastingaangifte"; FTS5 syntax characters (`"`, `*`, `-`, `:`, `(`) don't raise; a query with only short tokens returns `[]` |
| 3 | `fusion.fuse` | Tests: `w = 1` gives the same ranking as today's `rrf`; `w = 0` returns keyword ids only; a zero-weight side adds no ids; 5 dense rankings vs 1 keyword ranking at `w = 0.5` give equal mass |
| 4 | `run_query` wiring: `QueryOptions.vector_weight`, skip HNSW at `w = 0`, skip FTS at `w = 1`, embed only when needed, `--keyword --expand` warning | Test: a chunk that dense search doesn't seed but that the keyword search matches is a candidate at `w = 0.5` and absent at `w = 1`. That's the feature's reason to exist; the test fails if fusion moves into `combine`. Test: `--keyword --no-graph` makes no embedder call |
| 5 | Config keys, CLI flags and validation, `/api/query` `weight` | Tests: two mode flags give exit 2; `--weight 1.5` gives exit 2; no flag uses `fusion.vector_weight` |
| 6 | Output: `keyword` score field, `ragx.query.v4`, `--explain` `sources` | Tests: a keyword-only seed shows `["keyword"]`; stdout parses as one JSON document |
| 7 | `ragx-cli eval --weight` | Test: the override reaches every config |
| 8 | README (flags, config keys, schema v4, `gamma_vector` meaning), CLAUDE.md (Flows, Layout, Gotchas), release notes: query results change, `--vector` restores the old behaviour | Done in the same change as tasks 4–6 |
| 9 | Validation, after the build: on the notes eval corpus with the dense index held fixed, `w` ∈ {1, 0.7, 0.5, 0.3, 0}, each with and without `--rerank` | Per-query delta table plus r@5 / r@10 / MRR, written to `research/`. Decide whether the `0.5` default stays. This doesn't block shipping |

## Decisions

All decided 2026-10-28 by the user:

- **This plan goes before the zvec plan.** FTS5 stays the keyword backend even if zvec lands
  later, because the keyword index lives in the same SQLite transaction as the chunk rows.
- **Triggers** keep `chunks_fts` in sync. This replaces the research doc's "no triggers".
- **Hybrid is the default** for every query, with `w = 0.5`. `--vector`, `--keyword` and
  `--weight` override it. This replaces the research doc's opt-in `lexical.enabled = false`.
- **No pre-build failure analysis** (the research doc's pre-check). Build first, validate in
  task 9.
- **No `--hybrid` flag.** Hybrid is the default; a flag for it would do nothing.
- **Scope: this plan only.** The zvec plan is on ice; nothing here waits for it or prepares for
  it.
- **Output schema `ragx.query.v4`.** `breakdown.vector` holds the vector part only, plus a new
  `breakdown.keyword` and `explain.sources` (see Output). This follows the repo convention of
  bumping the schema for additive fields (v3 added `context` and `explain.path`). The rejected
  alternative was no new fields and staying on `v3`, which would leave the output unable to
  show which results came from keyword search.

## Open questions

None. The plan is ready for implementation.
