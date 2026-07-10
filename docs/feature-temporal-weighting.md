# Feature: temporal weighting (deferred — post Phase 3)

Decided with Albert 2026-07-10 via discovery interview. Build **after** the Phase 3 eval harness
exists, so its effect on recall/MRR is measured against a stable baseline before anyone trusts it.

## Decisions

1. **Opt-in only.** Temporal signals never influence ranking by default. No δ-term in the default
   scoring formula. All behavior is activated by explicit `ragx-cli query` flags.
2. **Capabilities** (all three):
   - **Hard filters:** `--since <date>` / `--until <date>` constrain candidates by document date.
   - **Recency bias, directional:** a decay boost that is *configurable in direction* —
     `--temporal recent` (newer wins) or `--temporal oldest` (earlier wins, e.g. "where did this
     idea first appear"). Suggested shape: exponential decay with configurable half-life
     (`[temporal] half_life_days = 180`), applied as a multiplicative modifier or an opt-in
     scoring term only when the flag is present.
   - **Temporal query matching:** when the query names a period ("what did we discuss in August
     2023"), bias retrieval toward documents dated in that window. Extraction of the period can
     ride the existing Phase 3 expansion LLM call (one prompt returns sub-queries + optional
     date range) so it costs no extra call; degrade to no-op without an LLM.
3. **Date source — cascade**, first hit wins:
   1. Content dates: filename patterns (`2023-08-04-…`, `2025-09-02 …`) and frontmatter
      (`date:` key). These say what the document is *about* — most of Albert's wiki is dated
      meeting/daily notes where mtime and git dates lie.
   2. Git: last-commit date touching the file (corpus repo, e.g. `~/projects/alber70g/notes`).
   3. File mtime as final fallback.
   Store the resolved date and its provenance (`filename|frontmatter|git|mtime`).

## Implementation notes for whoever builds it

- Schema: `files.doc_date` (ISO date, nullable) + `files.date_source` → SQLite `user_version = 2`
  migration. Backfill needs only a metadata pass (re-run date extraction per file) — **no
  re-embedding required**; add e.g. `ragx-cli index --refresh-dates`.
- Chunks inherit their file's date (chunk-level dates from in-text mentions were considered and
  parked — noisy, and most of the corpus is single-date documents).
- Filters apply before traversal (constrain seeds and admitted neighbors); bias applies after
  scoring. Keep the graph/heat machinery date-agnostic.
- Eval: extend `queries.jsonl` with dated queries (`"expect_period": ["2023-08", …]`) and verify
  the flags improve those without regressing undated queries.
