# BGE-M3 (Q8) vs nomic-embed-text-v1.5 (Q4_K_M) — eval benchmark, 2026-07-12

Empirical companion to
`bge-m3-dense-embeddings-as-ragx-provider-multilingual-quality-and-threshold-calibration.md`
(literature side). Run by a benchmark agent in a detached worktree of the notes
repo (`~/projects/alber70g/notes-wt-bgem3`, HEAD `20840e0`), so the production
corpus/index stayed untouched.

## Protocol

- Corpus: scoped notes eval corpus (383 files / 720 chunks), 26 EN+NL labeled
  queries (`.ragx/queries.jsonl`).
- Both legs used IDENTICAL retrieval params (tuned 2026-07: hops=3, decay=.5,
  query_floor=.35, max_frontier=150, alpha_rerank=.9, beta_heat=0,
  gamma_vector=.1, graph k=8, min_edge_sim=.55). No threshold retuning.
- Leg 1 (reference): the production nomic index copied verbatim from the main
  corpus → `ragx-cli eval .ragx/queries.jsonl --configs baseline,rerank --json`.
- Leg 2: config flipped to `model = "text-embedding-bge-m3"` (LM Studio Q8
  GGUF; Q8 chosen deliberately to match nomic-FP16's ~500-650 MB size class),
  `doc_prefix = ""` / `query_prefix = ""` (BGE-M3 takes no prefixes) → full
  re-index (105.6 s) → same eval command.
- Reranker identical in both legs (local BAAI/bge-reranker-v2-m3 CrossEncoder).

## Results (26 queries, top=10)

| Config   | Metric    | nomic | bge-m3 Q8 | delta        |
|----------|-----------|-------|-----------|--------------|
| baseline | recall@5  | .846  | **.923**  | +.077        |
| baseline | recall@10 | .846  | **.962**  | +.115        |
| baseline | MRR       | .561  | **.640**  | +.079 (+14%) |
| rerank   | recall@5  | .846  | **.885**  | +.038        |
| rerank   | recall@10 | .923  | **.962**  | +.038        |
| rerank   | MRR       | .717  | **.755**  | +.038 (+5.3%)|

All deltas well above the ~1% rebuild-non-determinism noise floor.

Query-level: 3 of nomic-baseline's 4 misses become hits under BGE-M3 —
"Radboud kandidaat" (miss → rank 2), "Borgo naar productie" (miss → rank 9),
"Omring thuiszorg" (miss under both nomic configs → rank 1 in both BGE-M3
configs). Dutch queries benefit disproportionately, consistent with BGE-M3's
multilingual training and MTEB-NL standing. Sole remaining miss in every
config: the "training-app met Veronique" query.

## Graph / index stats

|                              | nomic            | bge-m3 Q8        |
|------------------------------|------------------|------------------|
| files / chunks               | 383 / 720        | 383 / 720        |
| edges                        | 2,849            | 2,826 (−0.8%)    |
| edge weight min/median/max   | .558 / .813 / 1.0| .550 / .704 / 1.0|
| edges ≥ 0.9 (near-dup range) | 99               | 20               |
| eval wall time               | 562 s            | 577 s            |

The predicted threshold-transfer failure did NOT materialize as degeneracy:
edge count stayed flat, and BGE-M3's edge-weight distribution shifted DOWN
(median .704 vs .813, 5x fewer near-dup-range edges) rather than compressing
upward. Even the no-graph baseline config improved most, so the win is
embedding quality, not threshold luck — and the CLS/mean-pooling confound is
not indicated. Sanity probes (Omring intro meeting, managed-hosting SLA) both
returned the correct top document.

## Caveats

- **Quant asymmetry**: nomic ran at Q4_K_M, BGE-M3 at Q8. Part of the delta
  could be quantization quality rather than model architecture. The isolating
  run (`text-embedding-baai-bge-m3-568m` vs the Q8) was not executed.
- Under BGE-M3 the rerank config's recall@5 (.885) trails its own baseline
  (.923) — the CrossEncoder occasionally demotes a top-5 hit (same pattern
  existed under nomic; MRR still improves with rerank).
- BGE-M3's median edge weight is .11 lower, so the fixed thresholds
  (min_edge_sim=.55, query_floor=.35) are effectively STRICTER for it.
  A retune pass on the BGE-M3 distribution may unlock more.

## Verdict / next steps

BGE-M3 Q8 beats nomic Q4_K_M on every metric at identical retrieval
parameters, with the largest gains on Dutch queries. Candidate follow-ups,
in order: (1) threshold retune (min_edge_sim, query_floor) on the BGE-M3
cosine distribution via `.lab/harness.py`; (2) Q8-vs-568m quant isolation run;
(3) decide whether to flip the production corpus config (full re-index of the
main corpus required; manifest guard makes `--changed` fail loud as designed).
The worktree still holds the ready BGE-M3 index + config for further sweeps.
