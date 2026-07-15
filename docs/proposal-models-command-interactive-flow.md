# Proposal: `ragx-cli models` interactive flow redesign

**Status:** implemented 2026-07-15 (`cli/models_ui.py` + `cli/models_cmd.py`); kept as the
design rationale. The open decision below was resolved as proposed: warn-and-honor
interactively, auto-downgrade non-interactively.

## Problem

The current interactive flow (`src/ragx/cli/models_cmd.py:62-104`) asks all three
questions **before** showing any of the information needed to answer them:

1. **Blind prompts.** `quality (fast/balanced/best/jina-nano) [balanced]:` offers four
   opaque words. The catalog knows each tier's model name, parameter count, languages,
   context length, and license — none of it is shown at prompt time. Machine specs print
   *after* the answers.
2. **Engine prompts don't state tradeoffs.** "llama-server vs sentence-transformers"
   means nothing unless you already know one avoids huggingface.co entirely and the
   other needs the `[rerank]` extra. That context lives only in `--help` flag text.
3. **Silent downgrade.** Pick `best` on a <8 GB machine and `recommend()`
   (`catalog.py:133-135`) substitutes `fast`, telling you only afterwards. No way to
   override or reconsider.
4. **The jina-nano trap.** Choosing `jina-nano` with the default `lm-studio` embed
   engine hard-errors at the end ("rerun with `--embed-engine llama-server`") — even
   though the catalog already encodes the constraint (`requires_llama_server`) and
   could drive the default.
5. **Fragile input.** A typo in the free-text tier name raises `RagxError`, exit 2,
   start over.
6. **Consequences surface too late.** Download sizes are never shown; "your index is
   now invalid, run `index --full`" prints *after* `ragx.toml` is written.

## Proposed flow

Principle: **context → choice → consequence → commit.** Detect everything first, show
it, ask with numbered menus, and present a plan before touching anything.

```
$ ragx-cli models

machine: 32 GB RAM · macOS · llama-server: on PATH · LM Studio (lms): found

── Step 1/3 · embedding model ─────────────────────────────────────────
   Determines retrieval quality. Changing it later invalidates the
   index (full re-index required).

   #  tier       model                  size  langs  ctx   notes
   1  fast       EmbeddingGemma 300M    308M  100+   2K    lightest; small corpora, low RAM
 » 2  balanced   BGE-M3                 568M  ~100   8K    ragx's benchmarked production model
   3  best       Qwen3-Embedding-0.6B   0.6B  100+   32K   highest quality; slower indexing
   4  jina-nano  Jina v5 nano           239M  multi  8K    best <500M quality · CC-BY-NC
                                                           (non-commercial) · llama-server only
   select 1-4 [2]:

── Step 2/3 · engines ─────────────────────────────────────────────────
   embedding serving:
 » 1  llama-server           ragx auto-spawns llama.cpp; LM Studio only
                             downloads (detected on PATH)
   2  lm-studio              LM Studio app must be running to serve
   select 1-2 [1]:

   reranker:
 » 1  llama-server           GGUF via LM Studio — no huggingface.co needed
   2  sentence-transformers  downloads from Hugging Face; needs the
                             ragx-cli[rerank] extra
   select 1-2 [1]:

── Step 3/3 · plan ────────────────────────────────────────────────────
   embedding  BGE-M3 Q8 GGUF (~600 MB)          already downloaded ✓
   reranker   bge-reranker-v2-m3 Q8 (~600 MB)   will download via LM Studio
   config     ragx.toml → embeddings.*, rerank.* (7 keys)
   re-index   REQUIRED — embedding model changes; run `ragx-cli index --full` after

   proceed? [Y/n]:
```

### How each problem maps to a fix

- **Specs shown first**, including whether `llama-server` was found on PATH — the
  engine defaults stop being invisible magic.
- **Numbered menus with a re-prompt loop** instead of one-shot free text. `»` marks
  the recommended default; Enter accepts it. Invalid input re-prompts instead of
  exiting.
- **RAM guard becomes visible, not substituting.** On a <8 GB machine the `»` marker
  moves to `fast` and heavy rows get a `⚠ needs ≥8 GB` annotation. If the user still
  picks `best`, honor it after a one-line confirm — a balanced choice means respecting
  an informed override. Non-interactive `--quality best` keeps today's auto-downgrade
  + note (an agent can't be asked); this interactive/non-interactive asymmetry should
  be stated in the README.
- **jina-nano stops being a trap:** picking it prints
  `→ embedding engine locked to llama-server (LM Studio cannot serve this model)` and
  skips that question instead of erroring at the end.
- **The plan step** shows download sizes, marks what is already downloaded (the
  `_ensure_downloaded` check moves before the confirm), lists which config keys
  change, and states the re-index consequence *before* the write.

## Implementation sketch

Small and contained — no behavior change for flags / `--json` / non-interactive:

1. **`core/catalog.py`** — add structured fields to the choices (`params`, `context`,
   `languages`, `download_mb`) instead of parsing prose out of `notes`.
   Verify: existing tests pass; table renders from fields.
2. **`cli/models_cmd.py`** — a `_pick_numbered(title, rows, default)` helper
   (validation loop, `»` marker) replacing `typer.prompt`/`_pick_engine` in
   interactive mode; reorder so specs + `lms`/`llama-server` detection and the
   already-downloaded check happen before any question.
   Verify: interactive smoke run against a scratch corpus.
3. **`recommend()`** — split recommendation from enforcement: return annotations
   (RAM warning, license note, engine lock) rather than silently swapping tiers in
   interactive mode.
   Verify: unit tests for low-RAM interactive vs non-interactive paths.
4. **README** — update the `models` section in the same change (CLAUDE.md rule:
   usage-visible changes ship with their docs).

Estimated diff: ~150 lines, centered on `models_cmd.py`.

### Rendering choice

Plain f-string column alignment is enough. `rich` is already available (transitive
dependency of typer ≥0.12) if fancier tables are ever wanted, but no new dependency
and no speculative widgets now.

### Deliberate non-goal

No full TUI (arrow-key selection à la questionary). Numbered menus keep the code
testable with piped stdin and keep the agent-first stdin/stdout contract intact.

## Open decision

Interactive RAM-override behavior: warn-and-honor (proposed) vs keep the current
auto-downgrade everywhere. Proposal argues warn-and-honor interactively, downgrade
non-interactively.
