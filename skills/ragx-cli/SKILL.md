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
graph traversal, with optional cross-encoder rerank (`--rerank`) and LLM expansion (`--expand`).
Agent-first: one JSON doc on stdout, logs on stderr, deterministic exit codes, byte-exact source
locations.

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
ragx-cli query "..." --json --top 10            # one entry per file, best chunks + exact locations
ragx-cli query "..." --json --rerank            # higher precision: cross-encoder rerank (slower)
echo "..." | ragx-cli query - --json            # long/multiline query from stdin
ragx-cli index                  # incremental reindex (hash-based, cheap) — the default
ragx-cli index --full           # full rebuild: re-chunks + re-embeds everything
ragx-cli index --edges-only     # rebuild sub-chunks/edges/communities only; keeps embeddings
```

Other query flags: `--expand` (one LLM call: multi-query + HyDE), `--no-graph` (pure vector),
`--hops N`, `--explain`, `-B/-A N` (context lines before/after each chunk), `-C N` (both sides;
default 5). `--no-expand`/`--no-rerank` are deprecated no-ops (both stages are off by default).

## Output

- stdout is exactly one JSON doc (`ragx.query.v3` / `ragx.files.v1` / `ragx.status.v1`); logs go
  to stderr. Never parse stderr.
- Exit codes: `0` results, `1` success-but-empty (report "nothing found", don't retry as failure),
  `2` error.
- `ragx.query.v3` results are one entry per file (default 20): `file`, `score` (sum of its
  chunks), `chunks` (up to 3, each with `line_start/end`, `byte_start/end`, `score`,
  `breakdown`, `text`, `context`). JSON `text` is truncated — read the file at the line range
  for full context. `context` carries `-B`/`-A` grep-style lines around the chunk (default 5
  each; `-C N`/`-B N`/`-A N`, or `query.context_lines` in config).
- Typical flow: `--files-only` to pick files → `--json` for line ranges → read files at ranges.

## Freshness

- `status` includes `drift` counts; nonzero means the index is stale → run `ragx-cli index`
  (incremental, safe, converges even after include/exclude glob changes).
- Changing `embeddings.model`, `chunking.*`, or `graph.edge_source`/`subchunk_size_tokens`
  invalidates the index: commands fail loud on the manifest mismatch, naming the changed
  key and old/new value. For an `edge_source`/`subchunk_size_tokens` change, `ragx-cli index
  --edges-only` is the cheap fix — rebuilds sub-chunks/edges/communities, keeps chunk
  embeddings. Otherwise `ragx-cli index --full` — re-embeds everything, so confirm before
  running on a large corpus — or pin the old value back in `ragx.toml`.
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
