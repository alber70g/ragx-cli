# BGE-M3 multi-vector (ColBERT / MaxSim late interaction) — rerank alternative and token-level graph edges for ragx

**Date:** 2026-07-12
**Question:** BGE-M3 can emit `colbert_vecs` — one contextualized vector per token, scored by
MaxSim late interaction. Should ragx use them (a) to replace or supplement the
`bge-reranker-v2-m3` cross-encoder, or (b) as a graph edge source — the token-granularity
limit of the already-measured `edge_source="subchunk"` mechanism?

**Prior work this builds on (read first):**
`fine-grained-sub-chunk-edges-with-coarse-chunk-nodes-multi-granularity-graph-rag-literature-validation.md`
and `subchunk-edge-ablation-2026-07-12-two-round-experiment-protocol-results-and-verdict.md`.
The measured verdict there: sub-chunk (~128-token) max-cosine edges TIE the tuned chunk-graph
baseline on the notes corpus (r@10 .923/MRR .718 vs .717) at ~4.5× index embed cost, because
short single-topic chunks have no dilution headroom. ColBERT MaxSim is the same idea pushed to
per-token granularity — that framing anchors everything below.

---

## TL;DR verdict

**Do not adopt. Neither slot survives contact with ragx's own measurements or the literature.**

1. **As a reranker:** MaxSim late interaction is a *cheaper* reranker than a cross-encoder,
   not a *better* one. Every controlled comparison (ColBERT's own paper, ColBERTv2-era BEIR
   results, current reranker leaderboards) puts cross-encoders at or above late interaction
   on quality; the late-interaction win is 100×+ lower query-time FLOPs — bought by storing
   per-token document vectors. ragx already pays the cross-encoder cost on a ~100-candidate
   shortlist and the eval shows that stage is load-bearing (the Dutch 19→4 rescue). Swapping
   in MaxSim trades measured quality for speed ragx doesn't need, plus a ~GB-scale storage
   bill it definitely doesn't want.
2. **As an edge source:** the subchunk ablation predicts this is dead on arrival for the
   notes corpus. Token-level MaxSim attacks the same disease (pooled-embedding dilution) that
   the 128-token experiment proved barely exists there (near-dup guard fired on ~3 of ~2,800
   edges). No published work builds chunk–chunk kNN traversal edges from token-level MaxSim;
   the closest literature (SDR, AGRaME) says sentence-level sub-units already capture the
   max-matching gain, and AGRaME specifically warns that finer-granularity scores derived
   from representations not trained for that granularity are unreliable. Token-level is
   strictly more permissive than sub-chunk max — it amplifies the round-1 displacement
   failure mode, doesn't fix it.
3. **Storage/serving:** BGE-M3's token vectors are 1024-dim (8× ColBERT's 128), ~1.2 GB fp16
   for the 720-chunk eval corpus and ~16 GB for a 10k-chunk corpus — versus +10 MB measured
   for the subchunks table. Compression (token pooling, residual quantization) makes it
   ~6–12× smaller but never subchunk-cheap, and no OpenAI-compatible HTTP server can carry
   token vectors at all — the OpenAI embeddings response shape is one vector per input.
   `colbert_vecs` means in-process FlagEmbedding, a new provider type outside ragx's HTTP
   provider architecture.
4. **One honest correction to the brief:** BGE-M3's paper does NOT report colbert standalone
   as weaker than dense. In the (corrected, v3) ablations multi-vector *beats* dense
   standalone on MIRACL (69.0 vs 67.8 nDCG@10) and on long-doc MLDR (57.6 vs 52.5); it is
   *sparse* that is weak on MIRACL and *dense* that is weak on MLDR. The all-mode weighted
   fusion is always best (MIRACL 70.0, MLDR 65.0). What remains true: BAAI themselves
   recommend a cross-encoder (`bge-reranker`) on top for final quality, and ship colbert as
   part of a fusion, not as the finisher.
5. **If ragx ever wants a middle stage** between traversal and the cross-encoder (the one
   place a MaxSim-ish scorer could plausibly help — see Q5), the already-stored sub-chunk
   vectors ARE a pooled multi-vector representation (~4.5 vectors/chunk ≈ token pooling at
   factor ~180). Measure that free variant first; only if it moves the needle does paying
   for real token vectors deserve a thought.

---

## Q1 — MaxSim reranking vs cross-encoder reranking: better, or only cheaper?

**Answer: only cheaper. On quality, cross-encoders sit at or above late interaction in every
comparison that controls for model class; the honest cases where ColBERT-style rerankers
"win" are size-matched or latency-constrained comparisons.**

- **ColBERT's own paper is the cleanest controlled evidence.** On MS MARCO reranking,
  ColBERT scores MRR@10 34.9 vs 36.0 for a BERT-base cross-encoder trained by the same
  authors with the same loss — the cross-encoder has the quality edge, and ColBERT's claim is
  the efficiency: 170× faster, ~14,000× fewer FLOPs per query (Khattab & Zaharia, SIGIR 2020,
  https://people.eecs.berkeley.edu/~matei/papers/2020/sigir_colbert.pdf). The paper's ablation
  (already verified in the prior research file) also shows MaxSim > average-sim > single
  vector *within* late interaction — granularity helps the bi-encoder family, but doesn't
  close the gap to full cross-attention.
- **Current leaderboard placement:** BEIR avg nDCG@10 compilations place
  `bge-reranker-v2-m3` (~71.5) above `Jina-ColBERT-v2` (~70.1), with LLM-based cross-encoders
  (Qwen3-Reranker, bge-reranker-v2-gemma) further ahead (~73.7–77)
  (https://presenc.ai/research/best-open-weight-reranker-models-2026). Practitioner
  comparisons consistently characterize late interaction as "approaches cross-encoder
  quality (within 1–3 nDCG@10 points)" while being 2–10× faster with precomputed doc vectors
  (https://wiki.charleschen.ai/ai/processed/wiki/llm-core/rag/queries/reranking-models-and-tradeoffs,
  https://localaimaster.com/blog/reranking-cross-encoders-guide — both secondary sources;
  the direction matches the primary evidence above).
- **The answerai-colbert result is a size-matched claim, not a class win.**
  `answerai-colbert-small-v1` (33M params) "vastly outperforms cross-encoders its size" and
  beats bge-base-en-v1.5 — i.e., a great 33M model beats mediocre 33M cross-encoders and
  110M single-vector embedders (https://huggingface.co/answerdotai/answerai-colbert-small-v1,
  https://www.answer.ai/posts/2024-08-13-small-but-mighty-colbert.html). Its own training
  used **BGE-M3-reranker cross-encoder scores as the teacher** — the distillation direction
  itself tells you which architecture holds more ranking signal. The same post is candid that
  ColBERT models are uneven outside classic search-shaped tasks.
- **BAAI's own guidance:** the BGE-M3 model card recommends re-ranking with
  `bge-reranker`/`bge-reranker-v2` after retrieval for best quality — colbert mode is
  positioned as a retrieval/fusion signal, not as the final ranking stage
  (https://huggingface.co/BAAI/bge-m3).
- **Where the MaxSim speed actually comes from:** doc token vectors precomputed at index
  time; query time is one query encoding + a (|q|×|d|) dot-product max. That's real — tens
  of ms for a 100-candidate shortlist vs seconds for a CPU cross-encoder pass. But ragx runs
  the reranker locally on an interactive CLI where the whole query already tolerates the
  cross-encoder; the eval architecture decision ("ship graph+rerank together") was made *with*
  that cost counted. Buying back seconds by storing ~GB of token vectors and accepting a
  known quality step-down inverts ragx's priorities.

**Verdict Q1:** strictly a cost/quality trade-down for ragx. If rerank latency ever becomes
the complaint, the first lever is shortlist size or a smaller cross-encoder, not an
architecture swap.

## Q2 — Token vectors as graph edge source (chunk–chunk MaxSim edges)

**Answer: the measured subchunk ablation predicts DOA on this corpus, and token granularity
adds new failure modes on top. Plausible only for long multi-concept chunks — and even there,
subchunk mode is the cheaper first step and it's already built.**

Framing: chunk–chunk MaxSim over token vectors is `subchunk_knn_edges` with window size → 1
token. The mechanism family is identical: derive the edge weight from max-matching over
sub-units instead of one pooled cosine (`src/ragx/core/graph.py`).

- **The headroom argument transfers with full force.** The subchunk experiments established
  that on the notes corpus the disease this whole family treats — multi-concept dilution in
  pooled chunk embeddings — barely exists: the near-dup guard pruned ~3 of ~2,800 edges, and
  the dense per-sub-chunk graph tied the baseline exactly (MRR .718 vs .717). Token vectors
  fight the same dilution; a corpus without dilution has nothing for them to recover. The
  dilution theorem (arXiv:2603.21437, verified in the prior file) predicts gains scale with
  intra-chunk semantic diversity — which is what should be measured before *any* finer-grain
  edge experiment, and it applies to token granularity even more than to 128-token windows.
- **No literature precedent for token-MaxSim chunk–chunk edges.** Searches found ColBERT/
  MaxSim used exclusively for query→document scoring; nothing builds a document–document kNN
  traversal graph from token-level MaxSim. The closest published pairwise-document work is
  SDR (ACL Findings 2021), which gets its +11.6pp MRR gain from **sentence**-level max
  aggregation — i.e., the field's evidence says sub-unit granularity around sentences already
  captures the max-matching benefit for doc-pair similarity. The multi-granularity scoring
  paper AGRaME (arXiv:2405.15028, verified previously) adds the sharpest caution: sub-unit
  scores computed from representations not *trained* for that granularity degrade badly
  (27.9 vs 36.8 P@1). BGE-M3's colbert head is trained for query→passage MaxSim, not
  passage→passage; using it for chunk–chunk edge weights is exactly the untrained-transfer
  AGRaME warns about.
- **Token granularity is more permissive than sub-chunk max, not less.** Round 1 of the
  ablation failed because sub-chunk max scores reshuffled edge lists and displaced a
  load-bearing whole-chunk edge; round 2 fixed it with per-sub-chunk budgets. With ~800
  tokens per chunk, a single incidental token pair (a shared date, a code identifier, a
  Dutch stopword — BGE-M3, unlike Stanford ColBERT, does not punctuation-mask its doc
  tokens) can spike the pairwise max. Doc–doc MaxSim in its sum-of-max form is also
  asymmetric and length-sensitive (sum over A's tokens of max over B's ≠ the reverse), so
  ragx would additionally need to invent a symmetrization for its undirected `src<dst` edge
  store — a design decision with no literature to lean on.
- **Compute reality:** edge construction needs pairwise MaxSim over candidate pairs. Even
  pruned to the current k=8 dense-kNN candidates (~5,800 pairs for 720 chunks), each pair is
  an ~800×800×1024 dot-product block (~0.7 GFLOP) — ~4 TFLOP total, minutes on GPU, painful
  on CPU, versus the sub-chunk variant's near-free 3,261×3,261 matrix.
- **Where it *could* win:** a corpus of long, genuinely multi-concept chunks (the Fraunhofer
  ablation and the dilution theorem both say granularity effects are corpus-dependent). But
  the decided escalation path there is: measure intra-chunk sub-chunk diversity → enable
  `edge_source="subchunk"` at hops2 → only if subchunk *wins but saturates* would token
  granularity be a rational next rung. That ladder has three untriggered gates before ColBERT
  vectors enter the picture.

**Verdict Q2:** dead on arrival for the notes corpus; on long-chunk corpora it is at best the
step *after* a subchunk-mode win that hasn't happened anywhere yet.

## Q3 — Storage and compute reality

BGE-M3's `colbert_vecs` are **1024-dim** per token (the colbert projection is hidden→hidden,
1024→1024, in FlagEmbedding's M3 implementation) — 8× the 128-dim vectors ColBERT/ColBERTv2
were engineered around. All per-token costs below are therefore ~8× worse than the numbers
in the ColBERT literature.

| representation | per chunk (~800 tok) | 720-chunk corpus | 10k-chunk corpus |
|---|---|---|---|
| current: 1 dense vec (1024-d fp32) | 4 KiB | ~3 MB | ~40 MB |
| subchunks table (measured, opt-in) | ~4.5 × 3 KiB* | **+~10 MB (measured)** | ~+140 MB |
| colbert fp16, raw | ~1.6 MB | **~1.2 GB** | **~16.4 GB** |
| + token pooling ×3 (99% quality) | ~550 KiB | ~0.4 GB | ~5.5 GB |
| + token pooling ×4 (97% quality) | ~410 KiB | ~0.3 GB | ~4.1 GB |
| ColBERTv2-style 2-bit residual (260 B/tok at 1024-d) | ~210 KiB | ~150 MB | ~2.1 GB |

\* nomic vectors are 768-d fp32 in the current stack; row shown for scale comparison.

- **Pruning/pooling is real but bounded.** ColBERTv2's residual compression (centroid id +
  1–2-bit residual) achieves 6–10× reduction with ~no quality loss at 2-bit (MS MARCO index
  154 GiB → 25 GiB; MRR@10 36.2 → 36.2) — but it is implemented for 128-d vectors inside the
  PLAID index format, not as a reusable codec for 1024-d blobs
  (https://aclanthology.org/2022.naacl-main.272/). Answer.AI's token pooling (hierarchical
  clustering + mean-pool at index time) is the SQLite-friendly option: pool factor 2 = 50%
  fewer vectors at ~100.6% retrieval performance, factor 3 = 66% reduction at ~99%, factor 4
  = 75% at ~97%; model-agnostic, no training (https://www.answer.ai/posts/colbert-pooling.html,
  arXiv:2409.14683). Keeping top-k "important" tokens is a rougher variant of the same idea.
- **Is there a sane SQLite-blob story?** Mechanically yes — the `subchunks` table pattern
  (float32/float16 blobs, cascade-deleted, never in the HNSW) extends to a `token_vecs`
  table, and pooling factor 4 + fp16 lands at ~0.3 GB for the eval corpus. But calibrate:
  that is **30×** the measured subchunk cost for the same corpus, ~4 GB at 10k chunks, and
  the ablation showed the 10 MB version already bought nothing here. There's also a
  conceptual convergence worth naming: pooling BGE-M3 tokens down to a handful of clustered
  vectors per chunk is *structurally the subchunk representation ragx already stores* —
  sentence-aligned mean-ish vectors, ~4.5 per chunk, from the corpus's own embedding model.
  The expensive new machinery reproduces the cheap existing one as its own limit.
- **Compute:** encoding the corpus through BGE-M3 (568M params, XLM-RoBERTa-large) in-process
  is roughly the cost class of running the CrossEncoder over every chunk once — minutes on
  Apple Silicon MPS for 720 chunks, and it must re-run on every changed chunk. Incremental
  indexing survives (per-chunk blobs, like subchunks), but full rebuilds get meaningfully
  slower than the measured 1m47s subchunk rebuild.

**Verdict Q3:** local-first-friendliness dies at raw fp16 and is merely wounded with
aggressive pooling. The floor of the cost curve (~pool-factor-∞) is the subchunks table ragx
already has for +10 MB.

## Q4 — Serving reality and the paper's own ablations

- **Only FlagEmbedding-style in-process inference yields `colbert_vecs`.**
  `BGEM3FlagModel.encode(..., return_colbert_vecs=True)` (or raw HF transformers + the
  colbert linear head) is the canonical path (https://bge-model.com/bge/bge_m3.html).
  LM Studio, Ollama, and llama.cpp serve the GGUF bge-m3 through embedding endpoints that
  return **one pooled dense vector per input** — the OpenAI-compatible embeddings response
  schema has no shape for per-token matrices, so this isn't an LM Studio limitation ragx can
  wait out; the protocol itself can't carry the payload
  (https://ollama.com/library/bge-m3). Consequence for ragx architecture: colbert support
  means a new **in-process** provider (sentence-transformers/FlagEmbedding class, precedent:
  `providers/st_reranker.py`), a heavyweight optional extra, and a manifest-guarded second
  model — while embeddings keep flowing through the HTTP provider. Two models, two runtimes,
  one index.
- **Correction to the research brief:** the claim "colbert_vecs are reported weaker
  standalone than dense in the paper's own ablations" is **not what the (corrected) paper
  shows**. BGE M3-Embedding (arXiv:2402.03216v3): MIRACL avg nDCG@10 — Dense 67.8, Sparse
  53.9, **Multi-vec 69.0**, Dense+Sparse 68.9, All 70.0. MLDR (long docs) — Dense 52.5,
  Sparse 62.2, **Multi-vec 57.6**, Dense+Sparse 64.8, All 65.0; the paper states multi-vector
  brings "5.1+ points improvement over the dense method" on long documents
  (https://arxiv.org/html/2402.03216v3). So standalone colbert ≥ dense in the paper's own
  tables; it's sparse that's weak short-form and dense that's weak long-form. Note the HF
  card's 2024-07-01 news: the original MIRACL numbers were corrected upward due to an eval
  scripting mistake — conclusions unchanged, but don't cite v1 numbers.
- **What the weighted fusion shows:** `All` (w·dense + w·sparse + w·colbert; the model card's
  `compute_score` example uses weights [0.4, 0.2, 0.4], the paper's rank fusion uses a plain
  sum) is the best row everywhere, beating the best single mode by ~1 point on MIRACL and
  ~7.4 points over multi-vec alone on MLDR. The honest reading: colbert is a *contributing
  signal* in an ensemble whose gain over dense-alone is real but modest short-form, and BAAI
  still points at the cross-encoder for final reranking. Self-knowledge-distillation ablation
  confirms the three heads are co-trained to be fused, not used solo (Table 6, ibid.).
- Also relevant: BGE-M3's own docs flag "the heavy cost of multi-vector method" and recommend
  retrieving candidates with dense/sparse and applying colbert only in the rerank fusion
  (https://bge-model.com/bge/bge_m3.html) — i.e., even its authors position MaxSim exactly
  where Q5 below puts it, and nowhere else.

## Q5 — If ever adopted, where would it plug in?

Three candidate slots, ranked worst → least-bad:

1. **Edge builder — worst.** Everything in Q2: no headroom on this corpus (measured), no
   literature precedent, untrained-granularity transfer risk, symmetrization design burden,
   TFLOP-scale edge construction, and the full storage bill from Q3. Rejected.
2. **Rerank-stage replacement — second-worst.** Replaces ragx's strongest measured stage
   (rerank is what converts graph recall into MRR; the Dutch query's 19→4 rescue is
   rerank-dependent) with a scorer that the literature puts 1–3 nDCG points *below*
   cross-encoders, to solve a latency problem ragx hasn't logged. Rejected.
3. **Second-stage shortlist pruner before the cross-encoder — least-bad.** This is the only
   slot pointed at a *measured* ragx failure: candidate flooding. Dense graphs / hops3
   produce 400–650 mean candidates against RERANK_CAP=100, and the current shortlist cut is
   heat+vector scores — the hops-4 and subchunk-hops3 experiments both died by eviction at
   that cut. A MaxSim pass over all candidates (cheap once vectors exist) could choose the
   100 better than diluted whole-chunk cosine does, upstream of the unchanged cross-encoder.
   It's additive, degradable (skip when vectors absent), and touches no edge semantics.
   **But:** it only matters when the pipeline is run in a candidate-flooding configuration
   the defaults deliberately avoid (tuned default is hops3 on the sparse chunk graph at ~394
   candidates), and the free approximation — MaxSim over the already-stored sub-chunk
   vectors in subchunk mode — should be measured first. If ~4.5 pooled vectors per chunk
   don't improve the shortlist cut, 800 token vectors at 100× the storage are not going to
   be justified by the same eval.

**Overall:** no slot earns the cost today. The trigger conditions that would reopen this file:
(a) ragx targets a long-document corpus where chunks are genuinely multi-concept AND subchunk
mode measurably wins there; (b) shortlist eviction becomes a default-config problem; or
(c) rerank latency becomes a real complaint AND a measured MaxSim-vs-CrossEncoder ablation on
ragx's own eval shows the quality gap is inside noise.

---

## Comparison: token-level MaxSim vs the existing subchunk-edge mechanism

| dimension | `edge_source="subchunk"` (built, measured) | BGE-M3 colbert MaxSim (hypothetical) |
|---|---|---|
| sub-unit | ~128-tok sentence-aligned windows, ~4.5/chunk | 1 token, ~800/chunk |
| vectors from | corpus embedding model (LM Studio, HTTP) | separate BGE-M3, in-process FlagEmbedding only |
| trained for this use | plain embeddings, granularity-agnostic | trained for query→passage MaxSim only (AGRaME risk for chunk–chunk) |
| aggregation | max cos over sub-chunk pairs, per-sub budget | sum-of-max (asymmetric, length-sensitive) or max-of-max (single-token spikes) |
| storage (720 chunks) | +10 MB measured | ~1.2 GB fp16; ~0.3–0.4 GB pooled; ~150 MB w/ 2-bit codec that doesn't exist off-the-shelf at 1024-d |
| index-time compute | ~1.8× wall (measured 1m47s) | BGE-M3 forward over corpus + TFLOP-scale pairwise MaxSim for edges |
| measured result | ties tuned baseline; no headroom on short single-topic chunks | untested; same no-headroom prediction applies a fortiori |
| incremental indexing | per-chunk blobs, cascade delete — works | same pattern possible, heavier |
| relationship | — | subchunk mode ≈ token pooling at factor ~180; colbert is the expensive limit of the same curve |

## Concrete ragx implications

- No change to defaults, config surface, or roadmap. This research closes the "should we go
  finer than sub-chunks?" question with the same answer the ablation gave sub-chunks, for
  stronger reasons.
- The one idea worth keeping on the shelf: **sub-chunk-vector MaxSim as the shortlist cut**
  when `edge_source="subchunk"` is active — zero new storage, one new scoring function in
  `.lab/harness.py`, directly targets the measured RERANK_CAP eviction failure. Only worth
  running if/when a corpus actually uses subchunk mode.
- If a future corpus motivates colbert anyway, the architecture cost is known upfront: an
  in-process provider (st_reranker precedent), a `token_vecs` blob table (subchunks
  precedent), token pooling factor 3–4 at write time (answerai recipe), manifest-guarded
  model identity, and acceptance that `--changed` rebuilds re-encode through a 568M model.

## Risks / open questions

- **Unmeasured central claim:** "cross-encoder ≥ MaxSim rerank quality" is robust across the
  literature but has never been run on ragx's 26-query EN+NL eval. Dutch behavior is the
  wildcard — BGE-M3 is strongly multilingual and its colbert head might handle the Dutch
  queries differently than the cross-encoder. Cheap to test if ever needed (FlagEmbedding in
  `.lab/`, no pipeline changes).
- **The MICE direction** (precomputed-document cross-encoders, reported ~4× faster than full
  CE at equal-or-better quality) could obsolete this whole trade-off from the other side —
  if a MICE-style model ships for multilingual use, it beats both options here. Secondary
  source only (charleschen.ai wiki); primary paper not verified. Watch, don't build.
- **Source hygiene note:** one search hit ("The Reranking Tax", clawrxiv.io) mimics arXiv
  but is not arXiv and was excluded from evidence. The reranker-benchmark blog posts cited
  for leaderboard numbers (presenc.ai, iotdigitaltwinplm.com) self-describe as
  illustrative/compiled, not primary benchmarks — they were used only for relative ordering,
  which matches the primary ColBERT/ColBERTv2 papers.
- **Not investigated:** MUVERA-style fixed-dimensional encodings of multi-vector sets (would
  let token-level similarity ride the existing HNSW as single vectors); PLAID serving for a
  future MCP-server-scale corpus. Both irrelevant until a corpus with measured dilution
  headroom exists.

## Source table

| Source | Venue/Year | Role | Confidence |
|---|---|---|---|
| ColBERT (Khattab & Zaharia, SIGIR 2020) https://people.eecs.berkeley.edu/~matei/papers/2020/sigir_colbert.pdf | SIGIR 2020 | CE 36.0 vs ColBERT 34.9 MRR@10 controlled; 170×/14,000× efficiency; MaxSim>avg ablation | primary, fetched |
| ColBERTv2 (Santhanam et al.) https://aclanthology.org/2022.naacl-main.272/ | NAACL 2022 | residual compression 6–10× (154→16–25 GiB), 2-bit ≈ lossless (36.2→36.2 MRR@10) | primary, fetched |
| BGE M3-Embedding https://arxiv.org/html/2402.03216v3 | arXiv v3 (corrected) 2024 | dense/sparse/multi-vec/All ablations: MIRACL 67.8/53.9/69.0/70.0; MLDR 52.5/62.2/57.6/65.0 | primary, fetched |
| BGE-M3 model card https://huggingface.co/BAAI/bge-m3 | HF 2024 | recommends bge-reranker on top; compute_score weights [.4/.2/.4]; MIRACL correction notice | primary, fetched |
| BGE-M3 docs https://bge-model.com/bge/bge_m3.html | BAAI docs | "heavy cost of multi-vector"; retrieve dense/sparse, fuse colbert at rerank | primary, fetched |
| Token Pooling (Clavié et al.) https://arxiv.org/html/2409.14683 + https://www.answer.ai/posts/colbert-pooling.html | arXiv/blog 2024 | pool ×2 ≈ 100.6%, ×3 ≈ 99%, ×4 ≈ 97% retrieval performance | primary, fetched |
| answerai-colbert-small-v1 https://huggingface.co/answerdotai/answerai-colbert-small-v1 + announcement post | Answer.AI 2024 | size-matched "beats cross-encoders its size"; trained on BGE-M3-reranker teacher scores; uneven off-domain | primary, fetched |
| Open-weight reranker compilation https://presenc.ai/research/best-open-weight-reranker-models-2026 | blog 2026 | BEIR ordering: bge-reranker-v2-m3 71.5 > Jina-ColBERT-v2 70.1 | secondary, ordering only |
| Reranking trade-offs wiki https://wiki.charleschen.ai/ai/processed/wiki/llm-core/rag/queries/reranking-models-and-tradeoffs | wiki | CE-vs-ColBERT latency/storage table; MICE pointer | secondary, unverified |
| Ollama bge-m3 https://ollama.com/library/bge-m3 | Ollama | GGUF serving returns pooled dense only | primary, fetched |
| SDR, AGRaME, dilution theorem, KGP, subchunk ablation | see prior research files | granularity/max-matching evidence base; measured no-headroom verdict | verified previously |
