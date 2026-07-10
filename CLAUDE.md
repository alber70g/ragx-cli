# CLAUDE.md — ragx

## What this is

`ragx` is a local CLI that indexes a corpus of files into a chunk-level **embedding similarity
graph** and answers queries via vector search + **heat-propagation graph traversal** +
**cross-encoder reranking**. Indexing is LLM-free (embeddings only); an LLM is optional at
query time for multi-query/HyDE expansion. Built agent-first: single JSON doc on stdout
(versioned schemas), logs on stderr, exit codes `0` results / `1` empty / `2` error,
byte-exact source locations, `--explain` traces justifying every result.

Design plan: `ragx-cli-plan.md` (phases, parameters, risks). Module build specs:
`CONTRACTS.md` (Phase 0–1) and `CONTRACTS-PHASE23.md` (Phase 2–3) — historical but accurate
API contracts for every core module. Deferred feature specs live in `docs/`.

## Stack

- Python 3.12, managed with **uv** (`uv sync --group dev`, `uv run pytest`). NOT Bun/TS —
  deliberate exception, see plan §2.
- typer (CLI), hnswlib (HNSW vector index, cosine), SQLite (chunks/edges/manifest, WAL),
  httpx (+respx in tests), xxhash (incremental hashing), pathspec (gitignore globs).
- Providers speak **OpenAI-compatible HTTP**. Local default: LM Studio at
  `http://localhost:1234/v1`. `provider="ollama"` auto-switches to `:11434/v1`.
  Env fallbacks: `OPENAI_BASE_URL` (only while base_url is still the default) and
  `OPENAI_API_KEY` (only when `api_key_env` is unset); explicit config always wins.
- Reranker is always local: sentence-transformers CrossEncoder `BAAI/bge-reranker-v2-m3`
  (optional extra `ragx[rerank]`). LM Studio has NO /v1/rerank endpoint — verified; don't try.

## Layout (core/CLI split is the one architectural rule — MCP server later rides on it)

```
src/ragx/
  core/            # all logic lives here, CLI-independent
    config.py      #   .ragx/config.toml load/save, DEFAULTS dict, find_root/require_root
    models.py      #   Chunk, ChunkDraft, FileRecord, Edge, ScoredChunk, QueryOutput
    store.py       #   SQLite: files/chunks/edges/meta; replace_edges normalizes src<dst
    vectors.py     #   hnswlib wrapper; soft-delete via JSON sidecar; get_vectors for graph
    discovery.py   #   file walk, root-level .gitignore, binary/size/junk-dir filters, xxh64
    chunking.py    #   markdown/code/recursive splitters; byte-exact slices; tiny-fragment merge
    graph.py       #   knn_edges + affected_ids (incremental edge maintenance)
    traversal.py   #   propagate_heat: max-aggregation, query floor, frontier cap, trace
    expansion.py   #   one LLM call -> variants + HyDE; NEVER raises (degrades to no-op)
    fusion.py      #   Reciprocal Rank Fusion
    scoring.py     #   min-max normalize + alpha/beta/gamma combine (renormalizes w/o rerank)
    indexer.py     #   run_index: discover->hash->chunk->embed->HNSW+store->kNN edges
    query.py       #   run_query: expansion->fan-out->RRF->traversal->rerank->combine; JSON serializers
    eval.py        #   recall@5/@10 + MRR over file-level ranking, injected query_fn
  providers/       # base.py protocols (Embedder/Generator/Reranker); openai_compat.py;
                   # st_reranker.py; registry.py factories (env-var + api_key_env resolution)
  cli/             # thin shells only: app.py (init/status/config + registration),
                   # pipeline.py (index/query), inspect_cmd.py, eval_cmd.py, output.py
tests/             # 126 tests; mocked HTTP (respx), FakeEmbedder integration tests, no live network
```

## Flows

**Index** (`ragx index [--changed]`): discover files (include/exclude globs, root .gitignore,
skip node_modules/.venv/hidden/binaries/>2MB) → xxhash diff (`--changed` = incremental, default
= full rebuild) → chunk (~800 tok target, chunks are exact byte slices, sub-100-char fragments
merge into neighbors) → embed batched → HNSW add → kNN edges per new chunk (k=8, cos ≥ 0.55)
+ recompute edge lists of pre-existing neighbors that appear in new lists.

**Query** (`ragx query "..."`): optional expansion (1 LLM call, strict-JSON parse, graceful
no-op on failure) → embed all variants, HNSW top-20 each → RRF merge = seeds → heat propagation
(2 hops, decay 0.5, heat = max not sum, neighbor admitted only if cos(query) ≥ 0.35 floor,
frontier ≤ 150) → cross-encoder rerank of top-100 shortlist → final =
0.6·rerank + 0.25·heat + 0.15·vector (each min-max normalized; weights renormalize when a
stage is off). `--no-expand/--no-graph/--no-rerank` skip stages; `--files-only` ranks files
by sum of top-3 chunk scores; `--explain` attaches seed→parent→edge_weight→hop traces.

## State & benchmarks (as of 2026-07-10)

Phases 0–3 of the plan are DONE and validated. Eval corpus: Albert's wiki
(`~/projects/alber70g/notes/wiki`, .md only, 636 files → 1,323 chunks → 5,246 edges); labeled
queries at `<corpus>/.ragx/queries.jsonl` (18, EN+NL). Results (recall@5 / recall@10 / MRR):
baseline .833/.833/.593 · graph-only .778/.833/.522 · graph+rerank .759/**.889**/.568 ·
full .759/**.889**/**.613**. Key finding: graph-only HURTS precision (near-duplicate neighbors
displace direct hits — parameter sweeps plateau below baseline MRR; corpus property, not a
tuning miss), but graph+rerank together deliver the recall win — one Dutch query's target is
unreachable by vector search or rerank-alone, surfaced only via a hop-1 edge then reranked
19→4. Ship graph+rerank together; `--no-graph --no-rerank` is the explicit fast mode.

## Gotchas

- Ornith (`mlx-community/Ornith-1.0-35B-3bit`, Albert's required expansion LLM) is a
  *reasoning* model: answers land in `content` only after thinking; expansion uses
  max_tokens=4096 and takes ~40 s/call. Don't lower it.
- Changing `embeddings.model` invalidates the index; run_query fails loud on manifest
  mismatch; full `ragx index` rebuilds.
- Chunk ids are SQLite AUTOINCREMENT rowids AND hnswlib labels — never reused after delete.
  hnswlib can't enumerate deletions, hence the `vectors.hnsw.meta.json` sidecar.
- `Config.set` rejects keys not in `DEFAULTS` — add new config keys to DEFAULTS first.
- Tests must not hit the network; one provider test self-skips when sentence-transformers
  is installed. Pyright import errors in editors = interpreter not set to `.venv`.
- When running ragx against the wiki corpus: `cd` there, then
  `uv run --project /Users/albert/projects/alber70g/ragx ragx …` — and NEVER `git add`/commit
  from that directory (it's Albert's personal notes repo; `.ragx/` is gitignored there).

## What's next / deferred

- Phase 4 (plan §5): Leiden communities over the edge list, `query --global`.
- Phase 5: MCP server as a second thin shell over `ragx.core` (split already enforced).
- Temporal weighting: full decided spec in `docs/feature-temporal-weighting.md` — opt-in
  `--since/--until/--temporal recent|oldest`, date cascade filename/frontmatter → git → mtime,
  schema v2 migration. Build only after it can be measured against the eval baseline.
- README packaging claims (`uvx ragx`) are aspirational — not yet published to PyPI.

## Dev loop

`uv sync --group dev --extra rerank` · `uv run pytest -q` (126 pass, ~5 s) ·
`uv run ruff check src tests` · file soft cap ~150 lines · expected failures raise `RagxError`
(CLI maps to exit 2). Live smoke: LM Studio must be running with the configured embedding model.
