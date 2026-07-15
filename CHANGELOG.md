# Changelog

All notable changes to `ragx-cli` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [0.3.1] — 2026-07-15

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

[0.3.1]: https://github.com/alber70g/ragx-cli/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/alber70g/ragx-cli/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/alber70g/ragx-cli/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/alber70g/ragx-cli/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/alber70g/ragx-cli/releases/tag/v0.1.0
