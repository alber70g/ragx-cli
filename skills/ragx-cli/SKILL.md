---
name: ragx-cli
description: |
  Query a ragx-cli semantic index instead of grepping. Use when the user asks a conceptual or
  natural-language question about a corpus ("where is X discussed/decided?", "find notes about Y"),
  when you are about to grep/rg for a concept (not an exact identifier) in a directory that contains
  a `.ragx/` folder or `ragx.toml`, or when the user says `/ragx`, "search my notes", or "semantic
  search". Covers running queries, interpreting the JSON output, and keeping the index fresh.
---

# ragx-cli — semantic search over an indexed corpus

Indexes files into a chunk-level embedding similarity graph; answers queries via vector search +
graph traversal + cross-encoder rerank. Agent-first: one JSON doc on stdout, logs on stderr,
deterministic exit codes, byte-exact source locations.

## When

A corpus is ragx-indexed if its root has `ragx.toml` and `.ragx/`.

- Use ragx-cli for conceptual, natural-language questions ("where do we discuss auth tradeoffs?",
  cross-language queries). Full questions beat keywords.
- Use grep/rg for exact identifiers and literal strings. ragx does not replace it.
- Not on PATH: `uvx ragx-cli …` works (PyPI name is `ragx-cli`). Run from anywhere inside the
  corpus — root is found upward.

## Commands

```bash
ragx-cli status --json          # root, model, counts, and drift (new/changed/deleted vs index)
ragx-cli query "question" --json --files-only   # which files are relevant — most-used mode
ragx-cli query "..." --json --top 8             # ranked chunks with exact locations
ragx-cli query "..." --json --no-expand --no-rerank   # fast: skip LLM call + cross-encoder
echo "..." | ragx-cli query - --json            # long/multiline query from stdin
ragx-cli index                  # incremental reindex (hash-based, cheap) — the default
ragx-cli index --full           # full rebuild: re-chunks + re-embeds everything
```

Other query flags: `--no-graph`, `--hops N`, `--explain`.

## Output

- stdout is exactly one JSON doc (`ragx.query.v1` / `ragx.files.v1` / `ragx.status.v1`); logs go
  to stderr. Never parse stderr.
- Exit codes: `0` results, `1` success-but-empty (report "nothing found", don't retry as failure),
  `2` error.
- Chunk results carry `file`, `line_start/end`, `byte_start/end`, `score`, `breakdown`. JSON
  `text` is truncated — read the file at the line range for full context.
- Typical flow: `--files-only` to pick files → `--top 8` for locations → read files at ranges.

## Freshness

- `status` includes `drift` counts; nonzero means the index is stale → run `ragx-cli index`
  (incremental, safe, converges even after include/exclude glob changes).
- Changing `embeddings.model`, `chunking.*`, or `graph.edge_source` invalidates the index:
  commands fail loud on the manifest mismatch, and the fix is `ragx-cli index --full` —
  re-embeds everything, so confirm before running on a large corpus.
- `ragx-cli models` picks + downloads an embedding/reranker combo. All files download via
  LM Studio; serving is per engine — `--embed-engine llama-server|lm-studio` and
  `--rerank-engine llama-server|sentence-transformers`. With both on `llama-server`
  (the default when llama.cpp is installed), ragx auto-spawns/reaps the servers and
  needs neither LM Studio nor huggingface.co at query time. Other flags: `--quality
  fast|balanced|best|jina-nano`, `--yes`, `--dry-run`, `--json`.

## Why did/didn't a result appear?

- `ragx-cli query "..." --explain` — per-result trace: seed, graph edge, hop.
- `ragx-cli inspect neighbors <chunk_id>` / `inspect chunk <id>` / `inspect file <path>` /
  `inspect communities` / `inspect community <id>`.

## Gotchas

- The embedding provider must be reachable (default: LM Studio at `localhost:1234`; Ollama and
  any OpenAI-compatible endpoint via config). On connection error, report it and suggest starting
  the provider — never silently fall back to grep.
- Query expansion degrades gracefully (LLM down → no-op); embeddings do not.
- `--changed` is a deprecated alias — incremental is the default now.
- The corpus may be someone's personal notes repo: never `git add`/commit there. Querying and
  `ragx-cli index` are safe.

## New corpus (only when asked)

```bash
ragx-cli init
ragx-cli config set embeddings.model <model-id>   # match what the provider serves
ragx-cli index
```

Provider recipes are in the ragx-cli README. Check `ragx-cli status` and
`curl -s http://localhost:1234/v1/models` before guessing model ids.
