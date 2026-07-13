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
- typer (CLI), hnswlib (HNSW vector index, cosine), numpy (sub-chunk edge math), SQLite
  (chunks/edges/subchunks/manifest, WAL), httpx (+respx in tests), xxhash (incremental
  hashing), pathspec (gitignore globs).
- Providers speak **OpenAI-compatible HTTP**. Local default: LM Studio at
  `http://localhost:1234/v1`. `provider="ollama"` auto-switches to `:11434/v1`.
  Env fallbacks: `OPENAI_BASE_URL` (only while base_url is still the default) and
  `OPENAI_API_KEY` (only when `api_key_env` is unset); explicit config always wins.
- Machine-level provider settings live in `~/.ragxrc` (TOML; ONLY the embeddings/
  expansion/rerank sections, unknown keys fail loud). Precedence: DEFAULTS < corpus
  config.toml < ~/.ragxrc — the rc OVERRIDES corpus values and logs a stderr warning
  per overridden key. Write via `ragx-cli config set --global <key> <value>`.
  `Config.save` never bakes rc values into the corpus file.
- Reranker is always local: sentence-transformers CrossEncoder `BAAI/bge-reranker-v2-m3`
  (optional extra `ragx-cli[rerank]`). LM Studio has NO /v1/rerank endpoint — verified; don't try.

## Layout (core/CLI split is the one architectural rule — MCP server later rides on it)

```
src/ragx/
  core/            # all logic lives here, CLI-independent
    config.py      #   .ragx/config.toml + ~/.ragxrc (provider sections; rc wins, warns), DEFAULTS, find_root
    models.py      #   Chunk, ChunkDraft, FileRecord, Edge, ScoredChunk, QueryOutput
    store.py       #   SQLite: files/chunks/edges/meta; replace_edges normalizes src<dst
    vectors.py     #   hnswlib wrapper; soft-delete via JSON sidecar; get_vectors for graph
    discovery.py   #   file walk, root-level .gitignore, binary/size/junk-dir filters, xxh64
    chunking.py    #   markdown/code/recursive splitters; byte-exact slices; tiny-fragment merge;
                   #   subchunk_texts (sentence-aligned windows for subchunk edge mode)
    graph.py       #   knn_edges + subchunk_knn_edges (max over sub-chunk pairs, near-dup guard)
                   #   + affected_ids (incremental edge maintenance)
    communities.py #   leiden_communities: pure, seeded, recomputed whole every index run
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
tests/             # 151 tests; mocked HTTP (respx), FakeEmbedder integration tests, no live network
```

## Flows

**Index** (`ragx-cli index [--changed]`): discover files (include/exclude globs, root .gitignore,
skip node_modules/.venv/hidden/binaries/>2MB) → xxhash diff (`--changed` = incremental, default
= full rebuild) → chunk (~800 tok target, chunks are exact byte slices, sub-100-char fragments
merge into neighbors) → embed batched → HNSW add → kNN edges per new chunk (k=8, cos ≥ 0.55)
+ recompute edge lists of pre-existing neighbors that appear in new lists.
Experimental `graph.edge_source="subchunk"` (default `"chunk"`): chunks additionally split into
~128-tok sentence-aligned sub-chunks, embedded separately (float32 blobs in the SQLite
`subchunks` table, cascade-deleted with their chunk; NOT in the query-time HNSW); edge weight =
max cos over sub-chunk pairs; edges with whole-chunk cos ≥ `graph.near_dup_sim` (0.9) dropped.
Query side untouched. In subchunk mode `graph.k` means links per SUB-chunk (no per-chunk cap —
edge budget scales with concept count; graph gets ~2.2x denser). The manifest guards
`edge_source`/`subchunk_size_tokens` like the embedding model — `--changed` after a flip fails
loud. Rationale + literature: `research/fine-grained-sub-chunk-edges-*.md`. MEASURED 2026-07-12
(`.lab` experiments #10–#11): on the scoped notes eval corpus the dense concept graph at hops2
TIES the tuned chunk-graph baseline (r@10 .923 / MRR .718 vs .717) at ~4.5x index embed cost —
no win because the corpus's short single-topic chunks have no dilution headroom (near-dup guard
pruned ~3 edges). hops3 + uncapped budget floods RERANK_CAP (candidates 643). Verdict: default
stays `edge_source="chunk"`; subchunk is opt-in for corpora with long multi-concept chunks, and
if used, drop traversal to hops=2.

**Query** (`ragx-cli query "..."`): optional expansion (1 LLM call, strict-JSON parse, graceful
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

UPDATE 2026-07-13: production embedding model is now `text-embedding-bge-m3` (Q8 GGUF, empty
doc/query prefixes) — notes-repo corpus config + index flipped in place from the worktree-built
index (no re-index; nomic backup at `<corpus>/.ragx/backup-nomic-20260713/`). On
the tuned-params scoped eval corpus (same params as the parameter study above), `rerank` config:
bge-m3 r@5 .885 / r@10 .962 / MRR .755 vs nomic .846/.923/.717 — beats nomic on every metric,
mostly on Dutch/multilingual queries. Details: `research/bge-m3-dense-q8-vs-nomic-q4-benchmark-2026-07-12-worktree-eval-results.md`.

## Gotchas

- Reasoning models as expansion LLM: answers land in `content` only after thinking;
  expansion uses max_tokens=4096 to leave room for that. Don't lower it.
- Changing `embeddings.model` invalidates the index; run_query fails loud on manifest
  mismatch; full `ragx-cli index` rebuilds.
- Chunk ids are SQLite AUTOINCREMENT rowids AND hnswlib labels — never reused after delete.
  hnswlib can't enumerate deletions, hence the `vectors.hnsw.meta.json` sidecar.
- `Config.set` rejects keys not in `DEFAULTS` — add new config keys to DEFAULTS first.
- Tests must not hit the network; one provider test self-skips when sentence-transformers
  is installed. Pyright import errors in editors = interpreter not set to `.venv`.
- When running ragx-cli against the wiki corpus: `cd` there, then
  `uv run --project /Users/albert/projects/alber70g/ragx-cli ragx-cli …` — and NEVER `git add`/commit
  from that directory (it's Albert's personal notes repo; `.ragx/` is gitignored there).

## What's next / deferred

- Phase 4 (plan §5): Leiden community creation is DONE (index-time, `communities.py`,
  read-only via `status`/`inspect communities`/`inspect community`). Remaining: community
  labels, `query --global` for corpus-level questions.
- Phase 5: MCP server as a second thin shell over `ragx.core` (split already enforced).
- Temporal weighting: full decided spec in `docs/feature-temporal-weighting.md` — opt-in
  `--since/--until/--temporal recent|oldest`, date cascade filename/frontmatter → git → mtime,
  schema v2 migration. Build only after it can be measured against the eval baseline.
- README packaging claims (`uvx ragx-cli`; PyPI name ragx-cli, plain `ragx` is name-blocked) are aspirational — not yet published to PyPI.

## Dev loop

`uv sync --group dev --extra rerank` · `uv run pytest -q` (151 pass, ~5 s) ·
`uv run ruff check src tests` · file soft cap ~150 lines · expected failures raise `RagxError`
(CLI maps to exit 2). Live smoke: LM Studio must be running with the configured embedding model.
Changes that impact usage (CLI flags, config keys/precedence, output schemas, install steps)
must get their representation in README.md in the same change.
