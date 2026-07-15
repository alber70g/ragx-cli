# LM Studio model downloads + curated embedding/reranker catalog for `ragx-cli models`

Date: 2026-07-15. Sources: lmstudio.ai official docs, lmstudio-ai/lms + lmstudio-python source,
Hugging Face model cards, live tests on LM Studio 0.4.12 (this machine). Full agent reports
summarized here; doc URLs at the bottom.

## 1. How to trigger LM Studio downloads programmatically

Three mechanisms exist; capabilities verified live on 0.4.12:

| Mechanism | Unattended | Needs server up | Notes |
|---|---|---|---|
| `lms get <ref> --yes` | yes (exit 0/1, ANSI noise on stdout) | **no** | `<ref>` = catalog id (`google/embedding-gemma-300m`), search term, or full HF URL; `@quant` suffix; `--gguf`/`--mlx` filters. `--yes` auto-picks the hardware-preferred variant. |
| REST `POST /api/v1/models/download` + poll `GET /api/v1/models/download/status/:job_id` | yes (clean JSON) | yes, ≥0.4.0 | structured errors (`model_not_found`); optional bearer auth off by default. |
| Python SDK `lmstudio` (`client.repository.search_models → get_download_options → download`) | yes (blocking + progress callback) | yes | extra dependency; nicest API. |

- `lms` ships with the app (linked at `~/.lmstudio/bin/lms` after first app run); detect via
  `shutil.which("lms")` + that fallback path. `lms version` prints no semver (use the app plist
  on macOS if ever needed).
- Verify success + resolve the served model key via `lms ls --json`: entries carry `modelKey`,
  `type` (`"embedding"`/`"llm"`), `format` (`gguf`/`mlx`), `path` (relative to models root),
  `sizeBytes`, `architecture`.
- Downloads land in `<lmstudio-home>/models/{publisher}/{repo}/…`; home found via
  `~/.lmstudio-home-pointer` (internal, not contractual).
- REST has **no** catalog-search endpoint; SDK does.

**Hard limitation (verified live, twice):** LM Studio only downloads GGUF or MLX-converted
artifacts. Plain transformers safetensors repos are rejected at resolution time —
`lms get https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base --yes` → exit 1,
"No download options available". A GGUF conversion of that reranker exists
(`jolleyboy/gte-reranker-modernbert-base-GGUF`) and downloads fine, but sentence-transformers
cannot load GGUF, and LM Studio has no /v1/rerank endpoint to serve it. **Conclusion: the
reranker cannot ride the LM Studio download path in any usable form; it stays on the
HF/sentence-transformers path** (huggingface_hub `snapshot_download` for pre-fetching).

## 2. Requested models — verdicts

### jina-embeddings-v5-text-nano (`jinaai/jina-embeddings-v5-text-nano`)
Real (released 2026-02-18): 239M EuroBERT-based, 768 dims (Matryoshka 32–768), 8192 ctx,
71.0 MTEB-en — best-in-class sub-500M. BUT:
- **EuroBERT is not supported by stock llama.cpp** — Jina's cards say to use *their fork*;
  LM Studio ships stock llama.cpp, so the GGUF repos (`…-nano-retrieval[-GGUF]`) won't load.
- The MLX ports (`…-nano-mlx`) are custom-code (own `model.py`), not LM Studio-loadable.
- License **CC-BY-NC-4.0** (non-commercial) — bad default for a recommendation catalog.
→ Excluded from the catalog until upstream llama.cpp support lands. Revisit: watch
ggml-org/llama.cpp for EuroBERT and ollama/ollama#14641.

### gte-reranker-modernbert-base (`Alibaba-NLP/gte-reranker-modernbert-base`)
149M ModernBERT cross-encoder, 8192 ctx, Apache-2.0, loads in `CrossEncoder(...)` natively
(transformers ≥4.48, no trust_remote_code). BEIR 56.73; third-party Feb-2026 benchmark: tied
best Hit@1 at 8× smaller than a 1.2B competitor, ahead of bge-reranker-v2-m3 — on **English**.
It is English-only; ragx's eval corpus is EN+NL where bge-reranker-v2-m3's multilingual
training is the reason the Dutch recall win exists. → Catalog **English tier** reranker;
bge-reranker-v2-m3 stays the multilingual tier/default.

## 3. Curated catalog (all embedding entries verified LM Studio-loadable)

| Tier | Model | Why | Prefixes |
|---|---|---|---|
| fast | `lmstudio-community/embeddinggemma-300m-qat-GGUF` (curated catalog id `google/embedding-gemma-300m`) | 308M, 100+ langs, in LM Studio's own catalog; 2048 ctx / 768 dims cap quality | query `"task: search result \| query: "`, doc `"title: none \| text: "` |
| balanced | BGE-M3 GGUF (`gaianet/bge-m3-GGUF`) | current ragx production model; ~100 langs, 1024 dims, 8192 ctx, MIT; benchmarked best on the ragx eval corpus | none |
| best | `Qwen/Qwen3-Embedding-0.6B-GGUF` | Apache-2.0, 100+ langs, 32K ctx, MRL 1024 dims; top multilingual MTEB family; runs on upstream llama.cpp (early 0.3.16/0.3.17 LM Studio bugs since fixed) | query instruct prefix, doc none |
| reranker (multilingual, default) | `BAAI/bge-reranker-v2-m3` | current default; only clean multilingual CrossEncoder without trust_remote_code | – |
| reranker (English) | `Alibaba-NLP/gte-reranker-modernbert-base` | 149M, faster, beats bge-v2-m3 on EN benchmarks | – |

Rejected alternatives: `gte-multilingual-reranker-base` (306M, good numbers, but needs
trust_remote_code — security/UX friction), `jina-reranker-v2-base-multilingual` (NC license,
trust_remote_code, 1024-tok window), `Qwen3-Reranker-0.6B` (causal-LM yes/no scoring, doesn't
fit the CrossEncoder wrapper).

All three embedding tiers are multilingual, so the "multilingual?" question only steers the
reranker (EN → gte, otherwise bge-v2-m3). Spec detection: total RAM (<8 GB → force fast tier);
macOS → let `lms get --yes` hardware-preselect MLX where LM Studio's catalog offers it
(pinned GGUF URLs stay GGUF — they run on Metal fine; that's how bge-m3 Q8 runs today).

## 4. Design consequences for `ragx-cli models`

- Prefer `lms` CLI subprocess over REST/SDK: works with the server down, zero new deps.
  Absent `lms` → graceful message pointing at lmstudio.ai.
- After `lms get` exit 0: re-run `lms ls --json`, match the new entry by repo fragment, use its
  `modelKey` as `embeddings.model`, and write the tier's doc/query prefixes to config.
- Reranker pre-fetch: `huggingface_hub.snapshot_download` (ships with the `rerank` extra);
  degrade to a printed hint when the extra isn't installed.
- Changing `embeddings.model` invalidates the index (manifest guard) → command must end with a
  loud `ragx-cli index --full` pointer, never auto-reindex.

## Decision outcomes (2026-07-15, reviewed by Albert — folded from the session's DECISIONS.md)

1. **jina-embeddings-v5-text-nano**: first excluded (stock LM Studio couldn't load EuroBERT),
   then EuroBERT landed in upstream llama.cpp and the retrieval GGUF proved out end-to-end via
   `llama-server --embedding` → **included as the 4th catalog option** (`--quality jina-nano`),
   llama-server engine only, CC-BY-NC warning attached. LM Studio still downloads-but-misclassifies
   it (`type: "llm"` → its /v1/embeddings refuses; candidate LM Studio bug report).
2. **Reranker delivery**: HF-only at first (LM Studio rejects safetensors; ST can't read GGUF);
   superseded the same day by `rerank.provider="llama-server"` — GGUF via LM Studio, served by
   an auto-spawned `llama-server --rerank`. Q8_0 GGUFs validated ≈ safetensors (within ~0.5
   logit, identical ordering — the all-negative bge logits on toy Dutch pairs match safetensors
   exactly; model behavior, not conversion loss). Verified on brew llama.cpp 9960.
3. **gte-reranker-modernbert-base (English-only) dropped entirely** per Albert — one multilingual
   reranker (bge-reranker-v2-m3) keeps the measured Dutch-recall win; the multilingual question
   left the CLI with it.
4. **Downloads**: `lms get --yes` subprocess (works with the server down, zero deps) over REST/SDK.
5. **MLX**: no hard pinning; `lms get --yes` hardware-preselects; explicit `gguf_ref`s exist per
   entry because the llama-server engine can't load MLX.
6. **Embedding tiers**: embeddinggemma-300m / bge-m3 / qwen3-embedding-0.6b confirmed by Albert.
7. **Single-engine mode** (Albert's direction): `embeddings.provider="llama-server"` mirrors the
   rerank engine (shared `providers/llama_process.py`, ports 9813/9814, atexit reaping); both
   engine prompts default to llama-server when the binary exists — LM Studio becomes
   downloader-only at query time. Trade-off: ~1–2 s cold-spawn per engine per process lifetime.
8. **No auto-reindex on model switch** — loud `ragx-cli index --full` pointer instead; rebuild
   spend stays explicit.

## Doc URLs

lmstudio.ai/docs/cli · /docs/cli/get · /docs/developer/rest · /docs/developer/rest/download ·
/docs/developer/rest/download-status · /docs/python · github.com/lmstudio-ai/lms ·
github.com/lmstudio-ai/lmstudio-python · huggingface.co/jinaai/jina-embeddings-v5-text-nano ·
huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base · huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF ·
lmstudio.ai/models/google/embedding-gemma-300m · huggingface.co/BAAI/bge-reranker-v2-m3
