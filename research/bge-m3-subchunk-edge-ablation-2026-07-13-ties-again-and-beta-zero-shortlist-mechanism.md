# Subchunk-edge ablation under BGE-M3 (2026-07-13): ties again; β=0 shortlist mechanism found

Re-run of the 2026-07-12 nomic subchunk ablation
(`subchunk-edge-ablation-2026-07-12-*.md`) after the embedding switch to
`text-embedding-bge-m3` (Q8), to test whether the "no dilution headroom"
verdict was a nomic artifact. Run in the `notes-wt-bgem3` worktree; the
chunk-mode index was backed up and byte-identically restored afterward
(sha256-verified).

## Results (26 queries, identical tuned params: decay .5, floor .35, α/β/γ = .9/0/.1)

| Configuration | eval config | r@5 | r@10 | MRR |
|---|---|---|---|---|
| chunk, hops=3 (production) | rerank | .885 | .962 | **.755** |
| chunk, hops=2 | rerank | **.923** | .962 | .713 |
| subchunk, hops=3 | rerank | .885 | .962 | .7548 |
| subchunk, hops=2 | rerank | .885 | .962 | .7548 |
| all of the above | baseline | .923 | .962 | .640 |

**Verdict: unchanged.** Subchunk ties chunk at best (Δ within noise), at 2.0x
index time (213.8 s vs 105 s) and 2.24x edges (6,341 vs 2,826; 3,261
subchunks, 4.53/chunk). The near-dup guard pruned exactly 20 pairs (all 20
whole-chunk-cos ≥ .9 pairs; ~3 under nomic). Default stays
`edge_source="chunk"`. Bonus: chunk@hops2 posts the best r@5 anywhere (.923)
by trading −.042 MRR — echoes the nomic hops2 finding.

Rebuild-noise read: the leg-B graph-off baseline matched the reference
baseline to 4 decimals — zero measurable rebuild drift within one LM Studio
session (the ~.04 MRR drift previously observed was cross-session).

## The interesting part: why subchunk hops2 ≡ hops3 ≡ chunk hops3, bit-identical

Candidate flooding DID reproduce (mean 543 candidates at subchunk-hops3 vs
RERANK_CAP=100, all 26 queries over cap) — but caused zero metric damage,
and the diagnostic replication of `run_query` found the mechanism:

With tuned **β=0**, `combine(..., rerank=None)` renormalizes to w_heat=0,
w_vector=1.0 (`scoring.py`), so the pre-rerank shortlist (`query.py:111-112`)
is ordered purely by normalized RRF vector score with an
**ascending-chunk-id tie-break**. Graph-only candidates all carry vector=0,
so the shortlist is always ~20 RRF seeds + ~80 LOWEST-ID reachable chunks.
On a dense graph the hop-2 and hop-3 low-id reachable sets coincide, hence
bit-identical results. On the sparser chunk graph, hop reach still changes
the id-ordered filler, hence chunk hops2 ≠ hops3 (.713 vs .755).

**Design smell (latent, independent of this ablation):** under β=0 the
graph's entire contribution to the rerank path rides on which zero-vector
chunks survive an arbitrary id-ordered tie-break. The measured graph+rerank
recall win is real, but its mechanism is "graph nominates ~80 arbitrary-by-id
reachable chunks for the cross-encoder to sort out" — not heat-guided
selection. Candidate fix directions (unmeasured): order graph-only shortlist
filler by heat even when β=0 (heat as tie-break, not score term), or by edge
weight; would need an eval pass. Also: the indexer has no near-dup prune
logging (count had to be inferred); a stderr line would make future
ablations cheaper.

## Notes

- The stubborn "training-app met Veronique" query misses top-10 in every
  configuration and leg — a labeling/content question, not a graph one.
- Raw eval JSONs + candidate-diagnostic script preserved in the session
  scratchpad (legA/B/C-eval.json, diag.py).
- Worktree restored: 383 files / 720 chunks / 2,826 edges, chunk mode,
  sha256 byte-identical to pre-experiment; backup dir left in place.
