# `ragx-cli init`: reuse of downloaded models and running services across corpora

Date: 2026-10-28. Feeds `docs/feature-init-reuse-local-models-and-services.md`.

## Question

When `ragx-cli init` runs in a new repository, can it detect models that are already
downloaded and services that are already running (LM Studio, llama-server), so a second
corpus does not download or run a second copy?

## Current behaviour (read from code, 2026-10-28, v0.11.0)

| Resource | Where | Shared across corpora today? | Gap |
|---|---|---|---|
| Downloaded GGUFs | LM Studio models dir (`core/lmstudio.py:models_root`) | Yes, on disk | `init` never checks it; `init --yes` writes `nomic` DEFAULTS even when `bge-m3` is on disk. `models` does check (`find_installed`) |
| Cross-encoder weights | Hugging Face cache | Yes, on disk | Not probed anywhere |
| LM Studio server + loaded models | One app, port 1234 | Yes | `init` (`cli/detect.py`) lists *served* model ids only; no catalog match, no prefixes, no "loaded" signal |
| llama-server embed/rerank | Child of each ragx process, fixed ports 9813/9814 (`providers/llama_process.py`) | Accidentally | `healthy()` checks `/health` only. Corpus B reuses corpus A's server without checking the GGUF → wrong model possible. A exiting (atexit `terminate`) kills the server under B |
| `~/.ragxrc` | Machine-level provider config | Yes | `init` ignores it; writes all DEFAULTS into `ragx.toml`, which the rc then overrides with one warning per key |

Config precedence in code: `DEFAULTS < ragx.toml < ~/.ragxrc` (`core/config.py`,
`Config.load`). `Config.save` writes `DEFAULTS` merged with the file data, so every key is
always present in `ragx.toml`.

## Verified on this machine

- `lms ps --json` reports loaded models with `type`, `modelKey`, `path`, `ttlMs`
  (currently `text-embedding-bge-m3`, type `embedding`, `ttlMs: null`).
- llama.cpp `llama-server` 0.4.1 (build 10964) supports router mode (`--models-dir`,
  `--models-preset`, `--models-max`, `--no-models-autoload`) and `--sleep-idle-seconds`.
- `GET /props` on a single-model llama-server returns `model_path` = the absolute GGUF path
  (tested: spawned on `bge-m3-Q8_0.gguf`, port 9899; `/v1/models` returns the same path as
  the id). Usable to check which GGUF a server on a port is serving.

## External options for sharing llama-server across processes

1. **llama.cpp router mode** — one `llama-server` without `-m`, models loaded on demand by
   the request's `model` field, per-model args via an INI preset, idle stop per model.
   Caveat: `--reranking` changes pooling for every model in the router; embedding models
   need `reranking = false` in their preset (issue ggml-org/llama.cpp#29760, earlier bug
   #21256 closed not-planned). Sources: llama.cpp `tools/server/README.md`;
   huggingface.co/blog/ggml-org/model-management-in-llamacpp.
2. **llama-swap** (mostlygeek/llama-swap) — Go proxy; one config file, starts
   `llama-server` per model on demand, proxies `/v1/embeddings` and `/v1/rerank`, `ttl`
   unload, `matrix` for concurrent sets. Adds an external binary users must install.
3. **LM Studio** — already a shared long-lived app. `lms load` without `--ttl` keeps the
   model loaded indefinitely; JIT loading + auto-evict only apply to JIT-loaded models
   (lmstudio.ai/docs/developer/core/ttl-and-auto-evict). ragx uses `lms load`, so no
   eviction thrash between corpora.
4. **Keep per-process children, add an identity check** — no new process model; verify
   `/props.model_path` against `<section>.gguf` before reusing a port.

## Decision (Albert, 2026-10-28, turn-based discovery)

Option 4. No shared daemon, no router, no llama-swap: the request is about `init` not
duplicating models/services, and a detached daemon contradicts plan §1 ("no daemon").
Router mode and llama-swap stay documented here as the path if a shared llama-server is
wanted later. Detection uses live probes only (no registry of corpora). Full decision list
in the feature doc.
