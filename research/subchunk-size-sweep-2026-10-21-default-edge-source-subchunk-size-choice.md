# Sub-chunk size sweep (2026-10-21): choosing `graph.subchunk_size_tokens` for the subchunk default

Context: `graph.edge_source="subchunk"` becomes the default. This sweep picks the default
`graph.subchunk_size_tokens` (currently 128; `chunking.size_tokens` = 800) and judges whether
the setting should be an absolute token size or a ratio of chunk size. Lab entry: `.lab/log.md`
experiment 13. Raw data and scripts: `.lab/grids/subchunk-size-sweep-20261021/`.

## Setup

- ragx 0.8.0 (`9d07024`), run from an isolated worktree; no source changes.
- Corpus: `~/projects/alber70g/notes-wt-bgem3`, BGE-M3 Q8 embeddings (LM Studio), scoped
  include `wiki/clients/**/*.md` + `wiki/workstreams/**/*.md`, exclude `**/transcript.md`.
  383 files, 720 chunks. Chunk length (chars): mean 2,032 (~508 tokens, chars/4
  approximation), median 2,123, p10 357, p90 3,184, max 4,800.
- 26 labeled EN+NL queries. Reranker: sentence-transformers `BAAI/bge-reranker-v2-m3`.
- Each arm: `ragx-cli index --full`, then `ragx-cli eval queries.jsonl --configs
  baseline,graph,rerank --json` (top 10). Candidate counts (|seeds ∪ heat|) logged by a
  wrapper around `propagate_heat`. All arms in one LM Studio session.
- Regime D: ragx DEFAULTS (hops 2, decay .5, floor .35, frontier 150, α/β/γ .6/.25/.15,
  k 8, min_edge_sim .55, near_dup_sim .9). Regime T: corpus-tuned (hops 3, α/β/γ .9/0/.1),
  evaluated on the saved D-arm indexes for chunk, 128 and 256.

## Results

Baseline (vector only) is .9231 / .9615 / .6850 (r@5 / r@10 / MRR) in every arm.

| Arm | Edges | Avg degree | Sub-chunks/chunk | Unsplit chunks | Index wall | Candidates (D, mean) | graph (D) | rerank (D) | rerank (T) |
|---|---|---|---|---|---|---|---|---|---|
| chunk | 2,826 | 7.9 | – | – | 21 s | 222 | .9231/.9615/.6658 | .9231/.9615/.6747 | .9231/.9615/.7875 |
| sub 64 | 11,222 | 31.2 | 9.14 | 77 | 98 s | 530 | .9231/.9615/.6658 | .9231/.9615/.6843 | – |
| sub 96 | 7,897 | 21.9 | 6.08 | 117 | 70 s | 479 | .9231/.9615/.6658 | .9231/.9615/.6779 | – |
| sub 128 | 6,335 | 17.6 | 4.53 | 142 | 59 s | 435 | .9231/.9615/.6658 | .9231/.9615/.6779 | .9231/.9615/.7875 |
| sub 192 | 4,853 | 13.5 | 3.02 | 199 | 46 s | 384 | .9231/.9615/.6658 | .9231/.9615/.6779 | – |
| **sub 256** | **3,998** | **11.1** | **2.27** | **255** | **41 s** | **332** | .9231/.9615/.6658 | .9231/.9615/.6779 | .9231/.9615/.7875 |

"Unsplit" = chunks at or below 1.5 × target chars, which `subchunk_texts` keeps whole (their
sub-chunk vector is the chunk vector). Regime T candidates: chunk 343, sub128 543, sub256 458.

## Findings

1. **The `graph` config (0.8.0 product default) does not react to the edge set.** Hit ranks are
   identical for all 26 queries across all six arms. Diagnostic on the sub-256 index: zero
   graph-only files (all chunks vector score 0) reach the top 10 for any query. Under default
   scoring a graph-only chunk scores at most `0.625 × decay × edge_weight × parent_heat` with no
   vector term, so the RRF seeds always outrank it. `graph` differs from `baseline` on one
   query (q23, rank 1 → 2), a per-file folding effect, not a new candidate.
2. **rerank (D) is a tie.** Spread .6747–.6843 comes from two queries moving one rank each
   (q09, q15). That is far below the ~.04 MRR noise band from prior lab work.
3. **rerank (T) is bit-identical across chunk, 128 and 256** (.7875). This matches the β=0
   mechanism from the 2026-07-13 ablation: the pre-rerank shortlist is ~20 seeds plus the
   lowest-id reachable chunks, and at hops 3 those sets coincide.
4. **Cost scales with chunk length / sub-chunk size.** Relative to chunk mode: 64 → 4.0× edges,
   4.7× index time; 128 → 2.24× edges, 2.8× time; 256 → 1.41× edges, 2.0× time.
5. **No size floods in a way that costs metrics.** Candidates exceed RERANK_CAP=100 in 48/52
   graph calls even in chunk mode; density raises the mean from 222 to 530 with no measured
   damage. Flooding pressure is pre-existing, not introduced by subchunk mode.

## Verdict

**Recommended default: `graph.subchunk_size_tokens = 256`.** All sizes tie on both eval paths;
the protocol tie-break (lowest index cost, fewest edges) selects 256: 3,998 edges vs 6,335 at
128, index time 41 s vs 59 s, and the smallest candidate pool of the subchunk arms.

Tension to weigh: the literature notes (`research/fine-grained-sub-chunk-edges-*.md`) suggest
sentence-aligned windows of ~64–128 tokens to isolate concepts. At 256, 35% of chunks on this
corpus stay whole, so the graph sits closer to chunk mode. This corpus has no measured dilution
headroom (#10, #11, 2026-07-13), so it cannot show whether 128 would beat 256 on a corpus of
long multi-concept chunks. 192 is the middle option (4,853 edges, 46 s) if that risk matters
more than cost.

No evidence in this sweep supports changing other defaults alongside the flip:
- `graph.k` (per-sub-chunk, uncapped): edge budget grows with sub-chunks/chunk; no metric
  effect measured at any density.
- `traversal.hops`: default is already 2, the setting #11 recommended for subchunk mode.
- `graph.near_dup_sim`: not varied here; the indexer does not log prune counts (unverified
  how many pairs were pruned per arm).

## Absolute tokens vs ratio of chunk size

Quality gives no signal: metrics are flat from 64 to 256 tokens (ratio 1/12.5 to 1/3.1 of
`chunking.size_tokens`=800). Observed mechanics:

- What scales with the ratio: sub-chunks per chunk, edge count, index time and candidate pool
  all track `actual chunk length / subchunk size`.
- What is absolute: the embedding-quality floor (~50–100 tokens per the literature notes) is a
  property of the embedding model, and `subchunk_texts` already applies absolute floors
  (`min_part = max(MIN_CHUNK_CHARS, target/4)`, unsplit below 1.5 × target).
- Failure mode of absolute: if a user sets `chunking.size_tokens` ≤ ~1.5 × the sub-chunk size
  (e.g. 300 with 256), no chunk splits and subchunk mode silently equals chunk mode plus the
  near-dup guard.
- Failure mode of a ratio: small chunks (e.g. 200 tokens at 1/4) produce 50-token windows at
  the embedding-noise floor.

Judgement: keep an absolute token setting; it maps to the constraint that matters (embedding
quality per window). The degenerate case is cheaper to address with a doctor/stderr warning
when `subchunk_size_tokens × 1.5 ≥ chunking.size_tokens` than with ratio semantics. Not
implemented; not measured.

## Caveats

- One corpus, 26 queries, short notes (median ~530 tokens vs the 800-token target): the
  multi-concept headroom subchunk edges are meant to exploit is mostly absent.
- Under defaults the `graph` path is structurally blind to edge quality on this corpus, so the
  primary metric could not discriminate between sizes. The size choice is driven by cost.
- Absolute numbers are not comparable with the July experiments (#10–#12): baseline MRR is
  .685 here vs .640 then on the same corpus. Likely cause is 0.8.0's per-file result folding;
  not verified.
- Index times are single runs against a warm LM Studio; treat as relative.
- Corpus restored: `.ragx/config.toml`, `index.db`, `vectors.hnsw`,
  `vectors.hnsw.meta.json` sha256-identical to `.ragx/backup-subchunk-sweep-20261021/`; the
  temporary root `ragx.toml` was removed.
