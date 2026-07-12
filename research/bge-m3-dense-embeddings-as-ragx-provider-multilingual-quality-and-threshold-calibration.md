# BGE-M3 dense embeddings as ragx provider — multilingual quality and threshold calibration

**Date:** 2026-07-12
**Question:** Should ragx switch its embedding provider from `nomic-embed-text-v1.5` (768-dim,
English-trained, prefix-based) to BGE-M3 dense (`dense_vecs`, 1024-dim, multilingual, no
prefixes)? The eval corpus is mixed EN+NL; one Dutch query's target is currently unreachable by
vector search alone (only via graph+rerank). LM Studio already serves
`text-embedding-baai-bge-m3-568m` on the same OpenAI-compatible `/v1/embeddings` endpoint, so
dense mode is a pure config swap.

**Method:** Literature/context research only (Tavily + Exa searches, primary-source fetches of
the BGE-M3 paper, HF model cards, FlagEmbedding docs, MTEB-NL paper). **A parallel agent is
empirically benchmarking BGE-M3 dense on the 26-query eval corpus right now — its numbers land
in a separate document and are the deciding evidence.** This doc supplies expectations, gotchas,
and the recalibration protocol.

---

## TL;DR verdict

**Literature says the swap is promising and cheap to try, with two hard requirements.**
BGE-M3 dense is a genuinely multilingual retriever (trained on 100+ languages, SOTA-class on
MIRACL/MKQA at its release; competitive on the Dutch-specific MTEB-NL benchmark), while
nomic-embed-text-v1.5 is an **English-trained model** that multiple sources flag as unsuitable
for multilingual work — so the Dutch-query failure is plausibly a model gap fixable at the
source. The two hard requirements:

1. **Drop the prefixes.** BGE-M3 explicitly needs no query/doc instructions
   (`embeddings.doc_prefix = ""`, `embeddings.query_prefix = ""`). Keeping nomic's prefixes
   would feed BGE-M3 out-of-distribution input.
2. **Do not reuse the absolute cosine thresholds.** `graph.min_edge_sim = 0.55`,
   `graph.near_dup_sim = 0.9`, and `traversal.query_floor = 0.35` were calibrated on nomic's
   cosine distribution. The literature is unanimous that absolute cosine thresholds do not
   transfer across models; BGE-family models are specifically documented as having a shifted,
   compressed similarity distribution. Rank-based parts of ragx (HNSW top-20, kNN k=8, RRF)
   transfer fine; the three absolute floors must be re-derived from the new distribution.

Secondary caveats: ~4x embed compute (568M vs 137M params), +33% vector storage (1024 vs 768
dims — trivial at this corpus size), prefer a Q8/F16 GGUF over Q4 for embeddings, and verify
LM Studio applies **CLS pooling** (BGE models degrade significantly under mean pooling).

---

## Q1 — Dense retrieval quality vs nomic-embed-text-v1.5 (multilingual focus)

### BGE-M3 published numbers

From the BGE-M3 paper (Chen et al. 2024, arXiv:2402.03216; MIRACL numbers corrected upward
2024-07-01 per the model card):

- **MIRACL (18 languages, nDCG@10):** dense-only **67.8** avg vs mE5-large 65.4, mDPR/
  mContriever far below; all three modes combined 70.0. Dense alone beats all baselines on
  average and in most individual languages.
- **MKQA (cross-lingual, 25 langs → EN, Recall@100):** dense **75.1** (all modes 75.5) vs
  strongest baseline ~70.9.
- **MLDR (multilingual long-doc, 8k tokens, nDCG@10):** dense 52.5; notably the paper's own
  sparse mode beats dense by ~10 points on long docs — dense-only is the *weakest* of M3's
  modes on long documents (not a concern at ragx's ~800-token chunks).
- MMTEB mean(task): 59.56 — mid-pack by 2025+ standards (see Q5), but its retrieval-specific
  multilingual numbers remain strong for its size.

Sources: https://arxiv.org/html/2402.03216v3 · https://huggingface.co/BAAI/bge-m3

### Dutch specifically — MTEB-NL

The MTEB-NL benchmark paper (Banar et al., arXiv:2509.12340, ACL Findings 2026) is the best
Dutch-specific evidence. On the Dutch **retrieval** category:

| Model | NL retrieval |
|---|---|
| multilingual-e5-large-instruct | 61.4 |
| **bge-m3** | **60.0** |
| multilingual-e5-large | 59.1 |
| jina-embeddings-v3 | 59.1 |
| Qwen3-Embedding-0.6B | 57.1 |

BGE-M3 is the best non-instruct multilingual model of its size class on Dutch retrieval —
and beats Qwen3-Embedding-0.6B there despite Qwen3's much higher MMTEB average (a caution
against picking by leaderboard average). **No nomic model appears in MTEB-NL at all** —
consistent with nomic v1.5 not being a multilingual model.
Source: https://arxiv.org/html/2509.12340v1

### nomic-embed-text-v1.5's multilingual status

- v1 and v1.5 were **trained on English text**; Nomic's multilingual offering is the separate
  `nomic-embed-text-v2-moe` (~100 languages incl. Dutch — but max context drops to 512).
  Sources: https://discuss.4d.com/t/getting-to-know-the-models-nomic-embed-text/37026 ·
  https://hub.docker.com/r/ai/nomic-embed-text-v2-moe
- Independent model-selection guides state it plainly: nomic v1.5 is "not suitable for
  multilingual work" (https://www.antanaskovic.com/en/blog/choosing-text-embedding-model-that-actually-fits-your-project)
  and rate its multilingual support "Limited"
  (https://ai-marketinglabs.com/lab-experiments/nv-embed-vs-bge-m3-vs-nomic-picking-the-right-embeddings-for-pinecone-rag).
- MTEB (English v1) ~62.28 for nomic v1.5 — it's a good *English* model; Dutch queries run
  out-of-distribution.

### Would this fix the Dutch query at the source?

Plausibly yes — that query's target is currently invisible to nomic-based vector search
(surfaced only via a hop-1 edge + rerank 19→4). A model actually trained on Dutch should score
Dutch query→Dutch chunk pairs far more reliably, potentially making the target a direct seed.
This is exactly what the parallel empirical run will show; the literature only says the failure
mode (English-trained encoder on Dutch text) is the expected one and the fix (multilingual
encoder) is the standard remedy. Note the eval is only ~30% NL, so an aggregate win also
requires BGE-M3 not to regress on the English queries — its English MIRACL/BEIR numbers are
close to nomic-class, so a wash on EN + a win on NL is the expected shape.

---

## Q2 — Prefix / instruction conventions: BGE-M3 needs none

Confirmed from the primary sources:

- Official model card FAQ: "The only difference is that the BGE-M3 model **no longer requires
  adding instructions to the queries**." (https://huggingface.co/BAAI/bge-m3)
- Maintainer answer in HF discussion #35 ("Do I need to add the prefix 'query: ' and
  'passage: '?"): "**No. bge-m3 doesn't need prefix instruction**", and it supports both
  query→passage and query→query retrieval without prefixes.
  (https://huggingface.co/BAAI/bge-m3/discussions/35)
- FlagEmbedding model table lists the "query instruction for retrieval" column **empty** for
  bge-m3 (vs the "Represent this sentence for searching relevant passages:" instruction of
  bge-\*-v1.5). (https://raw.githubusercontent.com/FlagOpen/FlagEmbedding/master/README.md)
- Dense embeddings are L2-normalized by default (same discussion #35), so cosine = dot product,
  matching ragx's HNSW cosine space.

**ragx config:** set `embeddings.doc_prefix = ""` and `embeddings.query_prefix = ""` alongside
the model swap. Leaving nomic's `search_document: `/`search_query: ` prefixes in place would
prepend literal noise tokens the model was never trained on.

---

## Q3 — Cosine distribution and ragx's absolute thresholds

### BGE-family distributions are documented as shifted

- The bge-\*-v1.5 model cards state it outright: contrastive training with **temperature 0.01**
  compresses "the similarity distribution of the current BGE model … about in the interval
  [0.6, 1]", so "a similarity score greater than 0.5 does not indicate that the two sentences
  are similar", and thresholds should be picked per-corpus (they suggest 0.8–0.9 for v1.5).
  "What matters is the relative order of the scores, not the absolute value."
  (https://huggingface.co/BAAI/bge-large-en-v1.5)
- BGE-M3's own README example suggests M3-dense sits **lower** than v1.5: related pairs score
  ~0.63–0.68 and *unrelated* pairs ~0.35 in the worked example (https://huggingface.co/BAAI/bge-m3).
  If that generalizes, ragx's `query_floor = 0.35` would sit roughly at the *unrelated* level
  (admitting nearly everything) and `min_edge_sim = 0.55` lands somewhere in the ambiguous
  middle — different percentiles than under nomic. Only the corpus-level measurement can pin
  this down.

### The literature/practice consensus: absolute thresholds never transfer

- **Steck et al. 2024, "Is Cosine-Similarity of Embeddings Really About Similarity?"** (Netflix
  Research): under some training/regularization regimes raw cosine values can be essentially
  arbitrary — two models trained on the same data can disagree on raw cosine with no notion of
  which is correct. (https://research.netflix.com/publication/is-cosine-similarity-of-embeddings-really-about-similarity)
- **Anisotropy** (Ethayarajh 2019 lineage): embedding spaces concentrate vectors in a cone, so
  the expected cosine of *random* pairs — the "similarity floor" — is model-specific. An
  inter-model consistency study measured mean cosine ranging 0.739–0.915 and floors −0.07 to
  0.665 across four models (nomic v1.5 measured at mean 0.850, floor 0.379); at a fixed 0.85
  threshold only 66% of pairs got unanimous classifications, and disagreement concentrates
  exactly at threshold boundaries. ⚠️ Source is clawRxiv (agent-generated preprint archive) —
  treat numbers as illustrative, not authoritative; the mechanism it describes is standard.
  (https://clawrxiv.io/abs/2604.01479)
- **Calibration theory:** an arXiv paper on calibrated similarity proves that all *order-based*
  constructions — nearest-neighbor lists, threshold graphs at quantiles, ranking — are invariant
  under monotone recalibration, while absolute values require a learned monotonic map.
  (https://arxiv.org/html/2601.16907)
- **Practice guides** (Mixpeek; two 2026 migration write-ups) all prescribe the same protocol:
  embed a held-out paired sample with both models, build a score-percentile map ("what was 0.82
  in the old space is ~0.71 in the new space"), re-derive every hard-coded threshold, and treat
  thresholds as `value@model` configuration, never universal constants.
  (https://mixpeek.com/guides/calibrating-similarity-scores ·
  https://tianpan.co/blog/2026-05-13-embedding-migration-black-hole-thresholds-business-rules ·
  https://pyckle.co/blog/how-to-migrate-off-openai-embeddings)

### What this means for ragx's specific knobs

| Knob | Type | Transfers? |
|---|---|---|
| HNSW top-20 per variant, `fusion.*` (RRF), `graph.k = 8` | rank-based | **yes** — invariant under monotone score changes |
| `graph.min_edge_sim = 0.55` | absolute cosine floor | **no** — controls graph density; too low → dense graph (frontier/RERANK_CAP flooding, as seen in the subchunk ablation), too high → graph contributes nothing |
| `traversal.query_floor = 0.35` | absolute cosine floor | **no** — if BGE-M3's unrelated-pair level is ~0.35, this floor stops filtering |
| `graph.near_dup_sim = 0.9` | absolute cosine ceiling (subchunk mode only) | **no** — 0.9 sits at a different percentile in a compressed distribution |
| `scoring.*` weights | operate on min-max normalized scores | mostly yes — normalization absorbs scale, but the *shape* of the vector-score distribution still shifts the blend slightly |

Percentile matching against the current index is the cheap first-order recalibration: record the
percentile of 0.55 in nomic's candidate-pair cosine distribution and of 0.35 in nomic's
query→neighbor cosines on the 26 eval queries, then read off BGE-M3's cosines at the same
percentiles. Fine-tune from there with the eval harness (`.lab/harness.py` sweeps).

---

## Q4 — Practical: context, dims, GGUF quantization, serving

- **Context length:** both models are natively 8192 tokens (nomic v1.5's 8192 is often silently
  truncated by runtime defaults — Ollama ships 2048; the same class of default applies in other
  llama.cpp frontends). Irrelevant at ragx's ~800-token chunks either way, but BGE-M3 removes
  any anxiety for future longer-chunk experiments. (https://www.morphllm.com/ollama-embedding-models)
- **Dimensions / storage:** 1024 vs 768 = **+33%** per-vector storage and proportionally more
  HNSW distance compute (cost is linear in dim). At the eval corpus scale (1,323 chunks) that's
  ~5.4 MB vs ~4.1 MB of float32 — a non-issue. Subchunk mode's float32 blobs scale the same way.
- **Embed compute:** 568M vs 137M params → roughly **4x** slower indexing per token on the same
  hardware. Full-rebuild time on the wiki corpus will be noticeably longer; incremental
  `--changed` indexing unaffected in kind.
- **GGUF quantization for embedding models:** general GGUF guidance (Q8 near-lossless, Q4_K_M
  "acceptable") is derived from *generation* perplexity; the one embedding-specific benchmark
  found puts numbers on retrieval: **Q8 weights cost ~0.2–0.5 nDCG points; Q4 costs 1–2 points
  and is "not recommended for production retrieval"** (Apple-Silicon local-embeddings shootout
  covering nomic/bge-m3/qwen3 — moderate credibility, single source).
  (https://contracollective.com/blog/local-embeddings-apple-silicon-nomic-bge-qwen3-m5-max-2026 ·
  https://willitrunai.com/blog/quantization-guide-gguf-explained)
  Note the *current* nomic runs at `@q4_k_m`, so the incumbent is already paying the Q4 tax;
  for the swap, pick F16 or Q8_0 of bge-m3 in LM Studio if available (~1.2 GB at F16 — cheap).
  **Check which quant `text-embedding-baai-bge-m3-568m` actually is before benchmarking.**
- **Pooling gotcha (the sneaky one):** BGE models use the **[CLS] last-hidden-state** as the
  sentence embedding — the official docs warn "If you use mean pooling, there will be a
  significant decrease in performance." llama.cpp serves embeddings with a configurable
  `--pooling`; community bge-m3 GGUF setups explicitly set `pooling = cls`. LM Studio rides
  llama.cpp, and properly converted GGUFs carry pooling metadata, but this should be
  sanity-checked once: embed 2–3 pairs via LM Studio and compare cosines against
  sentence-transformers/FlagEmbedding reference values.
  (https://bge-model.com/tutorial/1_Embedding/1.2.3.html ·
  https://discuss.4d.com/t/introduction-how-to-build-your-own-private-semantic-search-engine/38584)

---

## Q5 — Newer alternatives, briefly

**Qwen3-Embedding** (also in LM Studio) is the strongest open successor line: 0.6B scores 64.33
on MMTEB vs BGE-M3's 59.56, with 4B at 69.45 and 8B at 70.58 (topping the multilingual
leaderboard at release), 32K context, Matryoshka dims. Two catches for ragx: (a) it is
**instruction-aware** — queries want an `Instruct: …\nQuery: ` prefix (Qwen reports a 1–5%
quality drop without), which reintroduces exactly the prefix machinery BGE-M3 removes (ragx's
`query_prefix` key handles it, so it's compatible); (b) on the Dutch-specific MTEB-NL retrieval
category Qwen3-0.6B (57.1) actually *trails* BGE-M3 (60.0) — leaderboard averages ≠ Dutch
retrieval. The 0.6B is the same weight class as BGE-M3, so it's a legitimate follow-up
candidate for the same eval harness run, but not the priority.
(https://github.com/QwenLM/Qwen3-Embedding · https://huggingface.co/Qwen/Qwen3-Embedding-0.6B ·
https://arxiv.org/html/2509.12340v1)
Nomic's own `nomic-embed-text-v2-moe` (multilingual, 768-dim) is hampered by its 512-token max
context — below ragx's ~800-token chunks, so it's disqualified without re-chunking.

---

## Concrete implications for ragx

**Config swap (corpus config.toml or ~/.ragxrc):**

```toml
[embeddings]
model = "text-embedding-baai-bge-m3-568m"   # verify quant; prefer F16/Q8_0
doc_prefix = ""
query_prefix = ""
```

Changing `embeddings.model` trips the manifest guard → full `ragx-cli index` rebuild (expected,
fails loud on `--changed`).

**What to re-measure on the 26-query eval (parallel agent's protocol should cover):**

1. Baseline (`--no-graph --no-rerank`) recall@5/@10 + MRR — is Dutch fixed at the source? Do EN
   queries hold?
2. The cosine distributions themselves: (a) all candidate kNN-pair cosines during edge building
   — where do 0.55 and 0.9 fall as percentiles vs nomic? (b) query→neighbor cosines during
   traversal — where does 0.35 fall? Edge count at `min_edge_sim=0.55` unadjusted vs
   percentile-matched value.
3. Full pipeline with recalibrated thresholds (and the tuned hops=3/alpha=.9 settings) vs the
   nomic baseline `.833/.833/.593 → .759/.889/.613` reference chain.
4. Frontier/candidate counts — a silently denser graph reproduces the subchunk-ablation failure
   mode (candidates flooding RERANK_CAP).

**Risks:**

- Reusing old thresholds silently changes graph density and traversal admission — results would
  be *wrong-shaped*, not obviously broken. Recalibrate before comparing.
- Unknown quant of the LM Studio artifact; Q4 embedding GGUFs measurably lose retrieval quality.
- Wrong pooling (mean instead of CLS) would degrade everything quietly — one-time sanity check
  against reference embeddings.
- ~4x embed cost on index builds; fine for this corpus, worth noting for bigger ones.
- BGE-M3's headline wins used dense+sparse+ColBERT hybrid; ragx uses dense only. Expectations
  should track the *dense-only* rows (still ahead of the multilingual field of its size, but the
  paper's flashiest numbers are the hybrid ones).

## Open questions

1. What quant/pooling does LM Studio's `text-embedding-baai-bge-m3-568m` actually ship? (Check
   the model card in LM Studio; verify cosines against FlagEmbedding reference.)
2. Where do ragx's three thresholds land as percentiles of BGE-M3's corpus distribution — and
   does simple percentile matching recover baseline-equivalent graph density? (Empirical run.)
3. Does fixing Dutch at the seed level *reduce* the marginal value of graph+rerank (the current
   recall win came from exactly that failure), i.e. does the tuned alpha_rerank=.9 blend need
   re-sweeping?
4. Is Qwen3-Embedding-0.6B worth a follow-up run despite its weaker MTEB-NL retrieval score,
   given instruction-prefix support already exists in ragx's config?
5. nomic-embed-text v1.5 at q4_k_m vs BGE-M3 at F16 confounds model quality with quant quality —
   if BGE-M3 wins, a nomic-F16 control would attribute the gain cleanly (probably not worth the
   time unless the margin is small).
