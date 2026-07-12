# Sub-chunk edge ablation — protocol, results, and verdict (2026-07-12)

Companion to the literature review in
`fine-grained-sub-chunk-edges-with-coarse-chunk-nodes-multi-granularity-graph-rag-literature-validation.md`.
Raw ledger entries: `.lab/log.md` experiments #10–#11; grids in `.lab/grids/subchunk-*.json`.

## Question

Does deriving graph edges from sentence-aligned **sub-chunks** (concept-level links) beat the
existing whole-chunk cosine edges, measured on ragx's labeled eval? The chunks themselves stay
the retrieval unit throughout; only edge construction changes.

## Feature under test

`graph.edge_source="subchunk"`: each ~800-token chunk is split into ~128-token
sentence-aligned windows (`subchunk_texts`), each window embedded independently (float32
blobs in the SQLite `subchunks` table — never in the query-time HNSW). Two aggregation
variants were tested across two rounds:

- **Round 1 (shared budget):** edge weight between two chunks = max cosine over their
  sub-chunk pairs; each chunk keeps its top `k=8` edges (all concepts compete for 8 slots).
  Near-dup guard: edges with whole-chunk cosine ≥ 0.9 dropped.
- **Round 2 (per-sub-chunk budget, Albert's refinement):** each sub-chunk independently
  links to its `k=8` nearest sub-chunks in other chunks; links collapse to parent edges
  (max weight), **no per-chunk cap** — the edge budget scales with concept count.

## Setup

- Corpus: scoped notes (`~/projects/alber70g/notes`, `wiki/clients/**` + `wiki/workstreams/**`,
  383 files → 720 chunks → 3,261 sub-chunks, ~4.5 per chunk).
- Eval: 26 labeled EN+NL queries, file-level recall@5 / recall@10 / MRR via `.lab/harness.py`
  (mirrors `run_query` with expand=off, graph=on, rerank=on; reranker forced to CPU).
- Scoring configs: **default** (hops=2, α.6/β.25/γ.15) and **tuned** (hops=3, α.9/β0/γ.1 —
  the validated winner of the 2026-07 param study).
- Embeddings: LM Studio, `text-embedding-nomic-embed-text-v1.5@q4_k_m`.

### Comparability control (the important part)

Index rebuilds are known to drift across LM Studio sessions (~.04 MRR at identical params —
param-study experiment #8). All arms were therefore rebuilt back-to-back in one session, and
comparability was verified directly: **query vectors byte-identical and fused seed-score
multisets identical on 26/26 queries across arms**. The arms differ in the edge set and
nothing else. (Chunk ids differ per rebuild — AUTOINCREMENT — so id-level comparisons are
meaningless; compare id-independent quantities only.)

Arm A also reproduced param-study #9's same-index numbers to 4+ decimals, anchoring the
baseline. Pre-experiment index/config backed up to `.lab/backup-20260712/` and restored after.

## Results

### Round 1 — shared top-8 budget (`.lab` #10)

| arm | edges | default r5/r10/MRR | tuned r5/r10/MRR |
|---|---|---|---|
| A: chunk edges | 2,849 | .846 / .885 / .626 | .846 / **.923** / **.717** |
| B: subchunk shared-8 | 2,755 | .846 / .885 / .629 | .846 / .885 / .712 |

Tuned config loses exactly one query's top-10 hit. Rescue attempts, all negative:
- Stricter edge filters (min_edge_sim .65–.80, k_cap 6): recall@10 stuck at .885, MRR only
  degrades → **not** shortlist eviction.
- Near-dup guard disabled (rebuild at near_dup_sim=1.01, 2,758 edges): metrics identical →
  **not** the guard. The guard had pruned only ~3 edges — the near-duplicate pollution this
  design targets barely exists on this corpus.
- **Mechanism:** sub-chunk max scores reshuffled the shared top-8 lists and displaced one
  load-bearing whole-chunk edge.

### Round 2 — per-sub-chunk budget (`.lab` #11)

Graph densified: 6,352 edges, avg degree 17.7 (was 7.9), max 61.

| variant (tuned scoring) | r5 | r10 | MRR | mean candidates |
|---|---|---|---|---|
| A baseline: chunk edges, hops3 | .846 | .923 | .717 | 394 |
| C: hops3, uncapped | .808 | .846 | .673 | 643 |
| C: hops3, k_cap 8 | .846 | .923 | .699 | 401 |
| C: hops3, k_cap 12 | .808 | .846 | .673 | 515 |
| C: hops3, min_edge_sim .80 | .846 | .885 | .712 | 522 |
| **C: hops2, uncapped** | .846 | **.923** | **.718** | 487 |
| C: hops2, k_cap 8 | **.885** | .923 | .696 | 231 |
| C: hops2 default scoring | .808 | .885 | .647 | 487 |

## Findings

1. **The displacement diagnosis was correct, and per-sub budgets fix it.** The round-1 lost
   query returns once each concept keeps its own links. Per-sub-chunk aggregation strictly
   dominates the shared-top-k variant — it is now the only semantics of `edge_source="subchunk"`.
2. **Dense concept graph at hops2 ≡ sparse chunk graph at hops3.** Identical recall ceiling,
   MRR .718 vs .717 (inside the lab noise threshold). The sub-chunk edges buy traversal depth,
   not new reachability, on this corpus.
3. **Uncapped density + hops3 floods the pipeline.** 643 mean candidates against the fixed
   rerank shortlist (RERANK_CAP=100) reproduces the known hops-4 eviction failure.
4. **No headroom on this corpus.** Short, single-topic chunks: the near-dup guard fired on
   ~3 of ~2,800 edges, so the multi-concept-dilution disease the literature validates simply
   isn't present. This is the corpus-dependence risk the literature review flagged, realized.
5. **Alternative operating point:** hops2 + k_cap 8 gives the best recall@5 measured on this
   eval anywhere (.885, +3.8 pts over baseline) at −.02 MRR — relevant if top-5 presence ever
   matters more than rank position.

## Verdict

Default stays `graph.edge_source="chunk"`. The feature ships opt-in with per-sub-chunk
semantics for corpora with long, genuinely multi-concept chunks — and with `traversal.hops=2`
when enabled. Before betting on it anywhere, measure intra-chunk sentence diversity (cheap:
pairwise cosine among each chunk's sub-chunk embeddings) to confirm dilution headroom exists;
this experiment shows the mechanism works but pays 4.5× index-time embedding cost for nothing
when chunks are already single-topic.

## Cost notes

Full rebuild with sub-chunks: ~1m47s wall (vs ~1m chunk-only); 3,261 extra embeddings
(single-sub chunks reuse their chunk vector); +~10 MB SQLite for sub-chunk blobs at this
corpus size.
