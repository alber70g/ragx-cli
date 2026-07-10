# Implementation Plan: `ragx` — Similarity-Graph RAG CLI

A single CLI that indexes a corpus into a chunk-level embedding similarity graph and answers queries through multi-query fan-out, graph expansion, and reranking. No LLM entity extraction at index time. Designed to be driven by humans, coding agents (Claude Code, Cursor, etc.), and later exposed as an MCP server.

---

## 1. Goals and non-goals

**Goals**

1. One self-contained binary/command (`ragx`) with subcommands. No daemon required for MVP.
2. Cheap, LLM-free indexing: chunk → embed → sparse kNN similarity graph.
3. Query pipeline: (a) optional LLM multi-query/HyDE expansion, (b) fan-out vector search, (c) graph traversal with score/heat propagation, (d) cross-encoder rerank, (e) ranked chunks + source files.
4. Agent-first ergonomics: `--json` everywhere, stable schemas, deterministic exit codes, incremental re-indexing.
5. Pluggable providers: embeddings (Ollama, OpenAI-compatible, local sentence-transformers), LLM for expansion (optional), reranker (optional).

**Non-goals (for now)**

- MCP server (Phase 5, architecture prepared but not built).
- Typed/semantic edges, entity extraction, knowledge-graph reasoning.
- Multi-tenant server, auth, remote index hosting.

---

## 2. Technology choices

**Language: Python 3.12, packaged with `uv`, distributed via `uvx ragx` / `pipx`.**
Rationale: the entire retrieval ecosystem (FAISS/HNSW, sentence-transformers cross-encoders, community detection via `igraph`/Leiden) is Python-native. A TypeScript port is possible later but would force reimplementing or shelling out for ANN and reranking. `uv` gives fast, reproducible installs suitable for agent environments.

**Core dependencies**

| Concern | Choice | Notes |
|---|---|---|
| CLI framework | `typer` | Subcommands, auto `--help`, easy `--json` flag plumbing |
| Vector index | `hnswlib` (or FAISS) | File-backed, no server; HNSW doubles as the kNN source for graph edges |
| Metadata + graph store | SQLite (single file) | Tables for chunks, files, edges, index state; portable, inspectable |
| Embeddings | Provider abstraction: `ollama` (default, e.g. `nomic-embed-text`), OpenAI-compatible HTTP, `sentence-transformers` local | Prefix support (`search_document:` / `search_query:`) configurable per model |
| LLM expansion | Provider abstraction: Ollama / OpenAI-compatible / Anthropic API | Fully optional; pipeline must work without any generative LLM |
| Reranker | `sentence-transformers` CrossEncoder (e.g. `BAAI/bge-reranker-v2-m3`) or remote (Cohere/Voyage) | Optional stage, on by default when model available |
| Community detection (Phase 4) | `igraph` + Leiden | On the sparse edge list |

**Index layout on disk** — everything under a `.ragx/` directory next to the corpus (analogous to `.git/`):

```
.ragx/
  config.toml        # providers, chunking params, k, thresholds
  index.db           # SQLite: files, chunks, edges, manifest
  vectors.hnsw       # HNSW index, chunk_id ↔ label mapping in SQLite
```

---

## 3. CLI surface

```
ragx-cli init [path]                  # create .ragx/, write default config.toml
ragx-cli index [path] [--changed]     # chunk, embed, build/update kNN graph
ragx-cli query "..." [--json] [--top N] [--no-expand] [--no-graph] [--no-rerank]
              [--files-only] [--hops H] [--explain]
ragx-cli inspect chunk <id> | file <path> | neighbors <id>   # debug the graph
ragx-cli status                       # index freshness, counts, config summary
ragx-cli config get|set <key> [value]
```

Conventions for agent use: `--json` emits a single JSON document on stdout (schema versioned, `"schema": "ragx.query.v1"`); logs go to stderr; exit code 0 = success with results, 1 = success but empty, 2 = error. `query` also accepts the query on stdin when the argument is `-`.

---

## 4. Pipeline design

### 4.1 Indexing (LLM-free)

1. **Discover files.** Respect `.gitignore` + `ragx` include/exclude globs from config. Store per-file content hash (xxhash) for incremental runs — `--changed` re-processes only new/modified/deleted files.
2. **Chunking.** Structure-aware where cheap: split Markdown on headings, code on function boundaries (tree-sitter optional, regex fallback), plain text by recursive character splitting. Defaults: ~800 tokens, 15% overlap. Every chunk records `file, byte_start, byte_end, line_start, line_end` so agents can jump to exact locations.
3. **Embedding.** Batch through the configured provider, applying document prefixes if the model requires them. Store vectors in HNSW; store chunk metadata in SQLite.
4. **Graph construction.** For every chunk, query HNSW for top-`k` neighbors (default k=8) above a similarity floor (default cosine ≥ 0.55, configurable). Write undirected weighted edges to SQLite (`edges(src, dst, weight)`), deduplicated. This is the entire "graph build" — one ANN pass over data already in memory.
5. **Incremental updates.** New/changed chunks: embed, insert, compute their edges, and recompute edges only for chunks that appear in the new chunks' neighbor lists. Deleted files: remove chunks, vectors (HNSW soft-delete + periodic `ragx-cli index --compact`), and incident edges.

### 4.2 Query

Stages are individually skippable via flags so agents can trade quality for latency.

1. **Expansion (optional, 1 LLM call).** One prompt returns JSON with 2–4 sub-queries/reformulations plus one short HyDE passage. With `--no-expand` (or no LLM configured), only the raw query is used — the pipeline degrades gracefully to steps 2–4.
2. **Fan-out retrieval.** Embed each query variant (with `search_query:` prefix if applicable), search HNSW top-`m` (default 20) per variant, merge via Reciprocal Rank Fusion. Result: seed set with fused scores.
3. **Graph expansion — heat propagation.** Initialize seed chunks with `heat = fused_score`. For `H` hops (default 2): each node propagates `heat × edge_weight × decay` (decay default 0.5) to neighbors; a node's heat is the max (not sum) of incoming contributions to avoid hub inflation. Query-bias filter: a neighbor is only admitted if its own cosine similarity to the original query embedding ≥ floor (default 0.35) — this keeps traversal anchored to the question rather than drifting through the corpus. Cap the frontier (default 150 nodes).
4. **Rerank.** Cross-encoder scores (query, chunk_text) for all candidates. Final score = `α · rerank + β · heat + γ · fused_vector_score` (defaults 0.6/0.25/0.15, config-tunable). Without a reranker: `heat` + vector score only.
5. **Output.** Top-N chunks with text, file path, line range, final score, and score breakdown. `--files-only` aggregates chunk scores per file (sum of top-3 chunk scores) and returns ranked files — the mode coding agents will use most. `--explain` adds the traversal trace: which seed produced each result, via which edges, with which weights (the "glass box" property from the NetApp article).

---

## 5. Phased delivery

**Phase 0 — Skeleton (½ day).** Repo, `uv` project, `typer` app, `init`/`status`/`config`, SQLite schema + migrations, provider interfaces (`Embedder`, `Generator`, `Reranker`) with an Ollama implementation. CI: ruff + pytest.

**Phase 1 — Baseline vector RAG (1–2 days).** `index` end-to-end (discovery, hashing, chunking, embedding, HNSW) and `query` as plain vector search with `--json`. Deliverable: a usable local semantic search CLI. This is the checkpoint to validate chunking quality on a real corpus (e.g. your transcripts set) before adding graph machinery.

**Phase 2 — Similarity graph (1–2 days).** Edge construction during indexing, `inspect` commands, heat-propagation traversal in `query`, `--explain` output, incremental edge maintenance. Evaluate: recall@10 with graph on vs. off on a small labeled query set (~20 queries is enough to steer).

**Phase 3 — Expansion + rerank (1–2 days).** Multi-query/HyDE expansion stage, RRF fusion, cross-encoder reranker, combined scoring, `--no-*` flags. Add a tiny built-in eval harness: `ragx-cli eval queries.jsonl` reporting recall@k / MRR per pipeline configuration, so parameter tuning (k, decay, floors, α/β/γ) is measurable instead of vibes.

**Phase 4 — Scale + communities (optional, 2–3 days).** Leiden community detection over the edge list; per-community extractive labels (top-TF-IDF terms — still no LLM required) or optional LLM summaries; `query --global` mode that routes through communities for corpus-level questions. Performance work: batch/parallel embedding, HNSW `ef` tuning, compaction.

**Phase 5 — MCP (later, deliberately deferred).** Because all query logic lives in a `ragx.core` library layer with the CLI as a thin shell, the MCP server is a second thin shell: `ragx-cli serve --mcp` exposing `query`, `inspect`, `status` as tools over stdio. No refactor needed if the core/CLI split is respected from Phase 0 — that split is the one architectural rule to enforce early.

---

## 6. Key parameters (config.toml defaults)

```toml
[chunking]   size_tokens = 800; overlap = 0.15
[graph]      k = 8; min_edge_sim = 0.55
[traversal]  hops = 2; decay = 0.5; query_floor = 0.35; max_frontier = 150
[fusion]     rrf_k = 60; per_query_top = 20
[scoring]    alpha_rerank = 0.6; beta_heat = 0.25; gamma_vector = 0.15
[embeddings] provider = "ollama"; model = "nomic-embed-text"; doc_prefix = "search_document: "; query_prefix = "search_query: "
[expansion]  enabled = true; provider = "ollama"; model = "..."; variants = 3; hyde = true
[rerank]     enabled = true; model = "BAAI/bge-reranker-v2-m3"
```

---

## 7. Risks and mitigations

1. **Hub chunks** (boilerplate, headers) attract many edges and dominate traversal. Mitigate: max-not-sum heat aggregation, per-node degree cap (drop edges beyond top-k both directions), optional stopword-density filter at chunking time.
2. **Similarity ≠ relevance.** Graph neighbors of a relevant chunk can be topically adjacent but useless. The query-floor filter during traversal and the cross-encoder rerank are the two guards; the eval harness (Phase 3) verifies they're working.
3. **HNSW deletes** degrade the index over time. Soft-delete + `--compact` rebuild; acceptable for corpora that change incrementally.
4. **Embedding model changes** invalidate everything. Store model name + dimension in the manifest; refuse to mix, offer `ragx-cli index --rebuild`.
5. **Agent misuse of long output.** Default `--top 8`, chunk text truncated to a configurable max in JSON output with `byte_start/byte_end` so agents fetch full text from the file themselves.

---

## 8. Definition of done (pre-MCP)

- `uvx ragx-cli init && ragx-cli index && ragx-cli query "..." --json --files-only` works on a fresh repo with only Ollama running.
- Incremental `ragx-cli index --changed` after editing one file touches only affected chunks/edges.
- Eval harness shows graph expansion improving recall@10 over the Phase 1 baseline on the reference query set, with rerank recovering precision.
- `--explain` can justify every returned chunk via seed → edge path → scores.

