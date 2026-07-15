# Changelog

All notable changes to `ragx-cli` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.5.0] — 2026-07-15

### Added

- **`ragx-cli models`**: recommends an embedding + reranker combo (quality tier,
  RAM/macOS detection), downloads models through LM Studio (`lms get --yes`), and writes
  the config to `ragx.toml` (with an `index --full` warning when the embedding model
  changed). Scriptable: `--quality fast|balanced|best|jina-nano`, `--embed-engine`,
  `--rerank-engine`, `--yes`, `--dry-run`, `--json` (schema `ragx.models.v1`).
  Interactive `init` offers the flow as its final step. Embedding catalog:
  EmbeddingGemma-300M / BGE-M3 / Qwen3-Embedding-0.6B / Jina-v5-nano (CC-BY-NC,
  llama-server engine only); reranker: bge-reranker-v2-m3 (multilingual).
- **`llama-server` rerank engine** (`rerank.provider = "llama-server"`): reranking via
  llama.cpp's `/v1/rerank` on a GGUF (`rerank.gguf`), auto-spawning `llama-server`
  (`rerank.server_bin`, `rerank.base_url`) when nothing is listening and terminating it
  at exit. Reranker GGUFs download through LM Studio (`ragx-cli models --rerank-engine
  llama-server`), so reranking needs **no direct huggingface.co access** and no torch/
  sentence-transformers install. Q8_0 GGUFs validated against the safetensors originals
  (scores within ~0.5 logit, identical ordering).
- **`llama-server` embedding engine** (`embeddings.provider = "llama-server"`): embeddings
  served by an auto-spawned `llama-server --embedding` on `embeddings.gguf` (same managed
  lifecycle as the rerank engine; default port 9813). `ragx-cli models --embed-engine
  llama-server` wires it up. With both engines on llama-server, LM Studio is purely a
  model downloader — nothing but ragx-managed processes run at query time. Also unblocks
  EuroBERT models (e.g. `jina-embeddings-v5-text-nano-retrieval-GGUF`) that LM Studio
  downloads but cannot serve.

## [0.4.0] — 2026-07-15

### Changed (BREAKING)

- **`ragx-cli index` is now incremental by default** (previously a full rebuild).
  It hash-diffs the corpus and re-processes only new/modified/deleted files. The old
  behavior moved to `ragx-cli index --full`; `--changed` remains as a deprecated,
  hidden alias for the (now default) incremental mode.
- **Chunking params joined the index manifest**: changing `chunking.size_tokens` or
  `chunking.overlap` now fails loud on an incremental run (previously it silently left
  old files chunked under the old params). Fix is `ragx-cli index --full`.

### Added

- **`ragx-cli status` reports corpus drift**: a `drift` object
  (`{"new": N, "changed": N, "deleted": N}`, same in text output) diffing the corpus
  on disk against the stored file hashes, so agents and humans can see when a
  re-index is due without running one.

## [0.3.3] — 2026-07-15

### Added

- **Early hint when the reranker model download stalls**: the huggingface_hub retry
  loop inside the cross-encoder load can spin for minutes before failing. The first
  retry warning now also prints a one-shot stderr pointer to the README's "Reranker on
  restricted networks" section (manual copy, mirror, or `rerank.enabled false`).

## [0.3.2] — 2026-07-15

_0.3.1 was tagged but never published — its release workflow referenced a nonexistent
action tag; 0.3.2 is the same change plus the CI fix._

### Changed

- **Legacy config self-heals**: finding a pre-0.3.0 `.ragx/config.toml` no longer fails
  loud — on a TTY, `ragx-cli` asks before moving it to `ragx.toml` (declining prints the
  `mv` and exits); non-interactive runs migrate automatically with a stderr notice. Both
  files existing at once is still a hard error.

## [0.3.0] — 2026-07-14

### Changed (BREAKING)

- **Corpus config moved from `.ragx/config.toml` to `ragx.toml` at the corpus root.**
  The config file is now meant to be committed with your corpus, while `.ragx/` (index
  data only) stays gitignored. Commands fail loud with a migration hint until you
  `mv .ragx/config.toml ragx.toml`; the index itself is untouched. `ragx.toml` is never
  indexed as corpus content.

### Added

- **Interactive `init`**: on a TTY, `ragx-cli init` probes local LM Studio
  (`localhost:1234`) and Ollama (`localhost:11434`), lists the models each server
  reports, and walks through embeddings, query-expansion, and corpus include/exclude
  settings — every prompt pre-filled with the defaults. Piped stdin, `--yes`, or
  `--no-interactive` writes the defaults unchanged, exactly as before.
- **Reranker graceful degradation offline**: when huggingface.co is unreachable and the
  cross-encoder model isn't cached, queries degrade to vector+graph scoring with a
  stderr warning instead of failing. README gains an offline/restricted-network install
  guide for pre-seeding the model cache.
- **Leiden communities (Phase 4, read-only)**: community detection over the similarity
  graph at index time; surfaced via `status`, `inspect communities`, and
  `inspect community <id>`.
- **Opt-in sub-chunk edges** (`graph.edge_source = "subchunk"`): edge weights become the
  max cosine over ~128-token sentence-aligned sub-chunk pairs, with a near-duplicate
  guard. Measured on the notes eval corpus it ties the tuned chunk-graph baseline at
  ~2x index cost — default stays `"chunk"`; intended for corpora with long
  multi-concept chunks (drop traversal to `hops=2` if used).

### Docs / research

- BGE-M3 embedding study: `text-embedding-bge-m3` (Q8 GGUF) beats nomic Q4 on every
  eval metric (rerank MRR .717 → .755), mostly on Dutch/multilingual queries — now the
  recommended embedding model.
- Sub-chunk edge ablation write-ups (chunk vs subchunk, across both embedding models).

## [0.2.0] — 2026-07-12

### Added

- **Machine-level provider settings in `~/.ragxrc`** (TOML, embeddings/expansion/rerank
  sections only, unknown keys fail loud). Precedence: defaults < corpus config <
  `~/.ragxrc`, with a stderr warning per overridden key. Written via
  `ragx-cli config set --global <key> <value>`.

### Docs / research

- Parameter study (2026-07): offline sweep harness; tuned `traversal.hops=3` +
  `scoring.alpha=0.9` lifts full-pipeline MRR +13.7% over the shipped defaults;
  documents rerank-weight dominance and the shortlist-selection role of vector+heat.

## [0.1.1] — 2026-07-10

### Fixed

- Documentation and code comments renamed from `ragx` to `ragx-cli`; clarified command
  usage instructions.

## [0.1.0] — 2026-07-10

Initial release: Phases 0–3 of [the plan](ragx-cli-plan.md), built and validated.

### Added

- **CLI & storage**: typer CLI (`init`/`status`/`config`/`index`/`query`/`inspect`/`eval`),
  SQLite store (files/chunks/edges/meta, WAL), provider abstraction
  (Embedder/Generator/Reranker) over OpenAI-compatible HTTP.
- **Baseline vector RAG**: file discovery (gitignore-aware), byte-exact chunking,
  batched embeddings, HNSW search, incremental `index --changed` via xxhash diff.
- **Similarity graph**: kNN edge construction (k=8, cos ≥ 0.55), heat-propagation
  traversal (2 hops, max-aggregation, query floor, frontier cap), `--explain` traces.
- **Quality & measurement**: multi-query/HyDE expansion (one LLM call, degrades to
  no-op), Reciprocal Rank Fusion, local cross-encoder rerank
  (`BAAI/bge-reranker-v2-m3`), `eval` harness (recall@5/@10, MRR).
- **Agent-first conventions**: single JSON doc on stdout with versioned schemas, logs
  on stderr, exit codes `0` results / `1` empty / `2` error.
- MIT license; PyPI packaging as `ragx-cli` with trusted-publishing release workflow.

[0.3.3]: https://github.com/alber70g/ragx-cli/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/alber70g/ragx-cli/compare/v0.3.0...v0.3.2
[0.3.0]: https://github.com/alber70g/ragx-cli/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/alber70g/ragx-cli/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/alber70g/ragx-cli/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/alber70g/ragx-cli/releases/tag/v0.1.0
