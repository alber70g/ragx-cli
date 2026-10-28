# Feature: `init` reuses downloaded models and running services

**Status:** planned 2026-10-28, not started. Research:
`research/init-reuse-of-downloaded-models-and-running-services-across-corpora-*.md`.

## Goal

`ragx-cli init` in a new repository picks up the embedding/reranker setup that this machine
already has (downloaded, loaded in LM Studio, or served by a running llama-server) instead
of writing defaults that trigger a new download or a second model in memory. A llama-server
on a shared port is never silently reused when it serves a different GGUF.

## Decisions

| # | Decision | Source |
|---|---|---|
| D0 | **OPEN — resolve before step 4.** Albert's model: "the local ragx.toml overwrites the global one". Code today: `~/.ragxrc` overrides `ragx.toml` (`config.py` `Config.load`; CLAUDE.md "rc OVERRIDES corpus values"). See "Open: config precedence" below | 2026-10-28 |
| D1 | Detection by live probes only: `lms ls/ps --json`, LM Studio/Ollama `/models`, llama-server `/props` on 9813/9814, Hugging Face cache, `llama-server` on PATH. No registry of corpora | discovery Q1 |
| D2 | llama-server stays a per-process child. Before reusing a port, compare `/props.model_path` with `<section>.gguf`; on mismatch raise `RagxError` naming port and both GGUFs | Q2, Q7 |
| D3 | A detected setup is preselected: TTY shows it as the default with its reason; `--yes`/non-TTY uses it without asking | Q3 |
| D4 | Provider answers (embeddings, expansion, rerank) go to `~/.ragxrc`, manual picks included | Q4, Q8 |
| D5 | `ragx.toml` omits provider keys that `~/.ragxrc` sets → no override warnings | Q5 |
| D6 | `init` creates `~/.ragxrc` only when it is absent; an existing rc is read, never modified by `init` | Q6 |
| D7 | Expansion LLM is not probed; its prompt stays as today | Q9 |
| D8 | `ragx-cli models` writes to `~/.ragxrc` when it exists (stderr notice), else `ragx.toml` as today | Q10 |

## Open: config precedence (D0)

The two models conflict; pick one before implementing, do not blend.

- **A. Keep rc > ragx.toml (code today).** Risk: seeding a new rc changes the provider of
  corpora initialized earlier whose `ragx.toml` names another embedding model; their next
  `query` fails on the manifest check. Needed mitigation: the manifest-mismatch error names
  `~/.ragxrc` as the override source and the fixes (`index --full`, or remove the key from
  the rc).
- **B. Flip to ragx.toml > rc (Albert's stated model).** Existing corpora keep their
  explicit keys, so seeding an rc cannot change them. D5 becomes mandatory: any key written
  to `ragx.toml` beats the rc, so `Config.save` must stop baking provider DEFAULTS. Breaks
  the current use of `config set --global` to override a committed `ragx.toml`; README,
  CLAUDE.md and `test_config_rc.py` change with it.

Recommendation: B. It matches the stated intent and removes the existing-corpus risk
without an error-message workaround. Release notes must flag the precedence change.

## Detection rules (`core/machine_probe.py`, pure + injected probes)

`probe_machine(...) -> MachineState` gathers facts; `pick_reuse(state) -> ReuseChoice | None`
is pure and ranks. Only catalog models (`core/catalog.py`) are auto-picked, because their
doc/query prefixes are known. Non-catalog models stay visible in the TTY lists as today.

Embeddings, first match wins:
1. Catalog embedding loaded in LM Studio (`lms ps`, `type=embedding`, matches
   `repo_fragment`) → `provider=openai`, `model=modelKey`. Reason: "loaded in LM Studio".
2. llama-server on 9813 whose `model_path` matches a catalog `gguf_fragment` →
   `provider=llama-server`, `gguf=model_path`. Reason: "served by a running llama-server".
3. Catalog embedding downloaded in LM Studio, tier order balanced → best → fast →
   jina-nano. Engine as `models_cmd._resolve_engine` defaults: llama-server when on PATH
   and a `.gguf` matches `gguf_fragment`; else LM Studio unless `requires_llama_server`.

Rerank, first match wins:
1. llama-server on 9814 serving the catalog reranker GGUF → `provider=llama-server`.
2. Reranker GGUF in LM Studio + `llama-server` on PATH → `provider=llama-server`.
3. `BAAI/bge-reranker-v2-m3` in the HF cache + `sentence_transformers` importable →
   `provider=sentence-transformers` (the DEFAULTS value).

## Init flow after the change

- rc exists → no provider probing. Print "provider settings from ~/.ragxrc: …"; skip the
  embeddings prompts; prompt only for keys the rc does not set (written to `ragx.toml`).
- rc absent, TTY → probe; prompts preselect the `ReuseChoice` with its reason; provider
  answers seed `~/.ragxrc` (stderr: "created ~/.ragxrc — applies to every ragx corpus on
  this machine"). Skip the "download models now?" offer when embeddings and rerank were both
  reused.
- rc absent, `--yes`/non-TTY → probe; reuse found → seed rc with it; nothing found → write
  the default `ragx.toml` as today, no rc.

## Steps

1. `core/machine_probe.py`: `MachineState`, `probe_machine` (probes injected), `pick_reuse`
   → verify: unit tests per ranking rule with fake `lms` JSON, respx `/props`, fake HF cache;
   a test that a non-catalog loaded model is never auto-picked (unknown prefixes would
   corrupt the index).
2. `llama_process.py`: `serving_gguf()` via `/props`; `ensure()` and the providers'
   `healthy()` path raise on mismatch → verify: respx test with a foreign `model_path`
   raises and names both paths; matching path reuses without spawning.
3. `config.py`: `write_rc_values(dict)` (one write, all keys); `Config.save` omits provider
   keys the rc sets and `file_data` does not → verify: saved `ragx.toml` lacks rc keys;
   `Config.load` emits no override warning after `init`.
4. Resolve D0; if B: flip merge order in `Config.load`, update warning text → verify:
   `test_config_rc.py` rewritten to encode "corpus file wins"; a test that a pre-existing
   corpus keeps its embedding model after an rc is seeded.
5. `cli/app.py` + `cli/init_prompt.py`: wire the init flow above → verify: CLI tests for
   rc-exists, rc-absent+TTY (CliRunner input), rc-absent+`--yes` with and without a reuse hit.
6. `cli/models_cmd.py`: D8 → verify: with an rc present, `models --yes` writes rc keys and
   leaves `ragx.toml` provider keys untouched.
7. README (init section, `~/.ragxrc` precedence, `models`), CLAUDE.md Stack/Flows → verify:
   `uv run pytest -q` and `uv run ruff check src tests` pass.

## Out of scope

Shared/detached llama-server, llama.cpp router mode, llama-swap (see research doc for when
to revisit); expansion LLM detection; a registry of corpora.
