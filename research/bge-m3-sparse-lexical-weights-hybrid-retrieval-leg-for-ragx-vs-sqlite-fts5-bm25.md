# BGE-M3 sparse (lexical_weights) as a hybrid retrieval leg for ragx — vs SQLite FTS5/BM25

**Date:** 2026-07-12
**Question:** Should ragx add a sparse/lexical retrieval leg — specifically BGE-M3's learned
`lexical_weights` (SPLADE-like per-token weights) — fused into the existing pipeline, to catch
the queries dense retrieval misses (Dutch compound words, proper nouns, client names, technical
identifiers)? And is plain SQLite FTS5/BM25 a good-enough zero-infra alternative, given the hard
local-first constraint (no vector DB) and the fact that LM Studio's `/v1/embeddings` can only
serve dense vectors?

**Method:** Parallel web research (Tavily + Exa) across five angles: BGE-M3 sparse mechanics and
paper numbers, hybrid fusion practice (RRF vs weighted sum), SQLite-native sparse/lexical storage
patterns, BM25-vs-learned-sparse published gaps, and Apple Silicon inference cost. Primary
sources fetched where numbers are load-bearing (BGE-M3 paper, FlagEmbedding source, Bruch et al.
fusion analysis). Local facts checked against the ragx codebase and the live eval corpus
(26 labeled EN+NL queries confirmed at `~/projects/alber70g/notes/.ragx/queries.jsonl`).

---

## Verdict (TL;DR)

**Ship the lexical leg in two phases, and only build phase 2 if phase 1 measures short.**

1. **Phase 1 — SQLite FTS5/BM25 as an extra RRF ranking at the seed-fusion point.** Zero new
   inference infrastructure (FTS5 ships inside SQLite), zero new models, ~1 day of work, fully
   measurable against the 26-query eval. Use the `trigram` tokenizer (or `unicode61
   remove_diacritics 2`) — trigram specifically handles Dutch compound words and technical
   identifiers, which word-boundary BM25 does not. The published out-of-domain gap between BM25
   and learned sparse is modest (≈1–8 nDCG points; BM25 actually *wins* on long documents in the
   BGE-M3 paper's own table), so FTS5 plausibly captures most of the lexical win.
2. **Phase 2 — BGE-M3 `lexical_weights` with a SQLite inverted-index table**, only if phase-1
   failure analysis shows misses attributable to BM25's *weighting* (not tokenization). This
   requires running the ~2.3 GB FlagEmbedding PyTorch model locally at **both index and query
   time** — LM Studio cannot serve `lexical_weights` — which is a real, recurring query-latency
   and memory cost, partially mitigated by the fact that ragx already runs a same-size local
   PyTorch model (bge-reranker-v2-m3) via the `[rerank]` extra.

**Fusion point: seeds, not final scoring.** The motivating failure mode is dense *misses* —
the target chunk never enters the candidate set. A sparse score added as a 4th term in
`scoring.combine` can only reorder candidates that dense already found; it cannot recover a miss.
Feeding the sparse top-k as one more ranking into the existing `fusion.rrf` call (alongside the
per-variant HNSW lists) adds candidates that then flow through traversal and reranking — exactly
the mechanism (add recall, let the cross-encoder restore precision) that ragx's own graph+rerank
finding already validated. Optionally *also* carry the sparse score into `combine` later as a
tuning refinement (the fusion literature says a tuned weighted sum slightly beats RRF), but that
is a second-order optimization.

**Do a 1-hour failure analysis before writing any code:** for each of the 26 queries, check
whether the relevant file is already in the seed set, and whether the query's distinctive terms
(client names, Dutch compounds) appear verbatim in the target file. If dense+expansion already
seeds every target, there is no lexical headroom and the whole feature is moot — the sub-chunk
ablation (2026-07-12) just demonstrated on this same corpus that literature-backed mechanisms can
have zero headroom here.

---

## Q1 — How BGE-M3 `lexical_weights` work, and how they compare empirically

### Mechanism

- Per token *i* of the input, the weight is `w = ReLU(W_lex^T · H[i])` — a single learned linear
  head + ReLU over the encoder's contextualized hidden states. If a token appears multiple times
  only its **max** weight is kept; special/unused tokens are dropped
  ([BGE docs](https://bge-model.com/bge/bge_m3.html),
  [FlagEmbedding m3.py source](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/embedder/encoder_only/m3.py)).
- Output data structure: `dict[str_token_id -> float weight]` (token ids are XLM-RoBERTa
  vocabulary ids; sparse dimensionality = 250,002 = vocab size)
  ([VectorChord writeup](https://blog.vectorchord.ai/unleash-the-power-of-sparse-vector)).
- Relevance score = `s_lex = Σ_{t ∈ q ∩ p} (w_qt · w_pt)` — sum of weight *products* over tokens
  present in **both** query and passage. `compute_lexical_matching_score` is literally a dict
  intersection loop ([HF model card](https://huggingface.co/BAAI/bge-m3), FlagEmbedding source).
- **Crucial difference from SPLADE:** the FlagEmbedding implementation keeps weights only for
  tokens *actually present* in the text — there is **no term expansion** (SPLADE's MLM head
  activates related vocabulary terms not in the input;
  [Zilliz comparison](https://medium.com/@zilliz_learn/exploring-bge-m3-and-splade-two-machine-learning-models-for-generating-sparse-embeddings-0772de2c52a7)).
  So M3-sparse does **not** solve vocabulary mismatch; it is a *contextualized re-weighting of
  BM25's job* (importance from transformer context instead of TF×IDF), on a fixed multilingual
  subword vocabulary.

### Empirical comparison (BGE-M3 paper, [arXiv:2402.03216](https://arxiv.org/html/2402.03216v3))

| method | tokenizer | MIRACL | MKQA | MLDR (long docs) |
|---|---|---|---|---|
| BM25 | proper Lucene analyzer | 38.5 | 40.9 | **64.1** |
| BM25 | xlm-roberta | 31.9 | 39.9 | 53.6 |
| M3-Sparse | xlm-roberta | **53.9** | **45.3** | 62.2 |
| M3-Dense | — | 67.8 avg | ~71 | 52.5 |
| M3-All (dense+sparse+colbert) | — | 70.0 | 75.5 | 65.0 |

Reading this honestly:

- M3-sparse crushes BM25 *when BM25 is forced onto the same subword tokenizer* (+22 MIRACL). Vs
  a properly configured per-language BM25 analyzer the gap shrinks to +15.4 (MIRACL) / +4.4
  (MKQA), and **BM25 wins on long documents** (64.1 vs 62.2 MLDR). The paper itself concedes
  "BM25 remains a highly competitive baseline."
- Sparse alone is far below dense alone on short-text benchmarks (53.9 vs 67.8 MIRACL). Its value
  is as a *complementary* leg: Dense+Sparse > Dense on essentially every language row, and the
  self-knowledge-distillation training is what lifts sparse from 36.7 → 53.9 on MIRACL
  (i.e., the learned weights genuinely carry signal BM25's statistics don't).
- **Dutch specifically** (MKQA `nl` row, cross-lingual NL-query → EN-corpus): BM25 42.5,
  M3-Sparse 52.9, M3-Dense 71.3, Dense+Sparse 71.8, All 72.3. Cross-lingual is sparse's *worst*
  case (few shared terms); the ragx corpus is mostly monolingual matching (NL query → NL note),
  which behaves more like MIRACL where the sparse contribution is larger. MIRACL has no Dutch
  track, so no direct monolingual-NL number exists.
- Independent replication caveat: a 2026 review notes official sources don't isolate per-dataset
  sparse-only vs BM25 on English BEIR, and "any claim about the isolated lift from BGE-M3's
  sparse mode versus classical BM25 requires local evaluation on your own corpus"
  ([axiomlogica benchmark review](https://axiomlogica.com/ai-ml/bge-m3-bge-reranker-benchmarks-dense-lexical-multi-vector-retrieval)).
- A relevant negative result: hybrid BM25+BGE via Elasticsearch *underperformed* BGE-dense alone
  on SciFact (0.68 vs 0.73 nDCG@10) — when the dense retriever is already strong, naive fusion
  with a weaker lexical leg can pull the ranking *down*
  ([FlagEmbedding issue #17](https://github.com/FlagOpen/FlagEmbedding/issues/17)). This mirrors
  ragx's own graph-only finding (recall additions displace precision without a reranker) and is
  the strongest argument for fusing at seeds and keeping the rerank stage on.

**Why the subword vocabulary matters for this corpus:** XLM-R's SentencePiece tokenizer splits
Dutch compounds into subword pieces, so a query containing one constituent can match a compound
in the note at the token level — something whitespace/word-boundary BM25 structurally cannot do
(compound-word languages are a known weak spot of word tokenizers; see the FTS5 discussion in
Q3). The same applies to camelCase/kebab-case technical identifiers. This is a genuine advantage
of *both* M3-sparse and FTS5-with-trigram over classic word-level BM25.

---

## Q2 — Fusion: RRF vs weighted sum, and where to plug in

### What the literature and vendors say

- **Bruch, Gai, Ingber — "An Analysis of Fusion Functions for Hybrid Retrieval"**
  ([arXiv:2210.11934](https://arxiv.org/abs/2210.11934), TOIS 2023): convex combination (weighted
  sum over normalized scores) **outperforms RRF in-domain and out-of-domain**; RRF is *sensitive
  to its k parameter*; CC is largely agnostic to the normalization choice and needs only a
  handful of labeled examples to tune its single weight. This is the most rigorous single source
  on the question.
- **T2-RAGBench ablation** ([arXiv:2604.01733](https://arxiv.org/html/2604.01733v1), 2026):
  convex combination α=0.5 beat RRF k=60 (R@5 0.726 vs 0.695); best RRF variant was k=10 (0.716).
  Same paper: hybrid BM25+dense via RRF beat both legs on every metric, and hybrid + cross-encoder
  rerank dominated everything (R@5 0.816) — the "fuse then rerank" architecture ragx already has.
- **BGE-M3 paper practice:** hybrid = plain weighted sum, demo weights
  `[w_dense, w_sparse, w_colbert] = [0.4, 0.2, 0.4]`; dense-heavy `α ≈ 0.7` is the common
  starting point for dense+sparse-only setups.
- **Vendors:** Qdrant offers RRF and distribution-based score fusion and explicitly says
  [neither dominates — use your eval set](https://qdrant.tech/documentation/search/hybrid-queries).
  Weaviate switched its *default* from ranked fusion (RRF) to **relative score fusion**
  (min-max-normalized weighted sum — exactly what `scoring.normalize` does) in v1.24
  ([Weaviate forum](https://forum.weaviate.io/t/hybrid-search-explanation-explanation/2251)).
  Practice has drifted toward tuned score fusion, with RRF as the robust zero-tuning default.

### The right fusion point for ragx

Two candidates exist in `run_query`:

- **(a) Seed-level: sparse top-k as an extra ranking in the existing `rrf(rankings)` call**
  (`src/ragx/core/fusion.py`, called at `query.py` step 2). The sparse leg *adds candidates* the
  dense fan-out never saw; those seeds then get graph traversal, rerank shortlisting, and final
  scoring for free. This is the only fusion point that can fix a dense *miss* — and dense misses
  (client names, Dutch terms surfacing rank >20) are the entire motivation.
- **(b) Score-level: sparse as a 4th term in `scoring.combine`** (delta_sparse next to
  alpha/beta/gamma). Better per the CC-beats-RRF literature *for ordering*, but it can only
  reorder candidates already present — useless against misses. Also requires computing sparse
  scores for all candidates (cheap: dict-intersection per candidate).

**Recommendation: (a) is the load-bearing change; (b) is optional tuning.** Note an asymmetry to
watch: expansion produces up to 5 dense rankings (query + 3 variants + HyDE) vs 1 sparse ranking,
so the sparse list gets ~1/6 of RRF influence. That is probably *correct* to start (dense is the
stronger retriever here), and it degrades gracefully when expansion is off (1 dense + 1 sparse).
Cheap extension: also run the variant strings through the sparse leg (trivial for FTS5; one more
batched forward pass for M3). If seed-RRF underweights sparse in measurement, a weighted-RRF
(per-ranking weight, as Qdrant supports) is a 3-line change to `rrf()` — but don't build it
speculatively.

---

## Q3 — Storing/serving sparse without a vector DB

### Inverted index in SQLite (for M3 lexical_weights) — sane and standard

A learned-sparse index *is* an inverted index with real-valued weights — that is literally how
Qdrant implements its sparse index internally. In SQLite:

```sql
CREATE TABLE sparse_postings(
  token_id INTEGER NOT NULL,
  chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
  weight   REAL NOT NULL,
  PRIMARY KEY(token_id, chunk_id)
) WITHOUT ROWID;
```

Query = for the query's ~10–50 nonzero tokens, one indexed lookup each, `SUM(w_q · w_p)` grouped
by chunk_id — a single SQL statement. Scale check for this corpus: 1,323 chunks × roughly
400–800 distinct subword tokens per ~800-token chunk ≈ 0.5–1M posting rows, ~20–30 MB, and
millisecond-scale queries. At 1–5k chunks this is comfortably inside SQLite's competence; the
`vstash` paper measured 20.9 ms median hybrid search at 50k chunks in a single SQLite file
([arXiv:2604.15484](https://arxiv.org/html/2604.15484)). Verdict: known/sane pattern, no vector
DB needed, cascades cleanly with the existing chunk lifecycle like `subchunks` already does.

### FTS5/BM25 — the zero-new-infra alternative

The SQLite + FTS5 (+ optional sqlite-vec) hybrid-RRF pattern is thoroughly established:
[Simon Willison / Alex Garcia's canonical SQL](https://simonwillison.net/2024/Oct/4/hybrid-full-text-search-and-vector-search-with-sqlite/),
[a 200-line walkthrough](https://dev.to/soytuber/building-a-hybrid-rag-in-200-lines-sqlite-fts5-sqlite-vec-rrf-38h1),
[sqliteai/sqlite-rag](https://github.com/sqliteai/sqlite-rag),
[litesearch](https://github.com/Karthik777/litesearch),
[liamca/sqlite-hybrid-search](https://github.com/liamca/sqlite-hybrid-search), and the
BEIR-validated [vstash](https://arxiv.org/html/2604.15484). ragx wouldn't even need sqlite-vec —
HNSW stays as-is; only the FTS5 leg is new, as an external-content table over the existing
`chunks` table (no text duplication):

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  text, content='chunks', content_rowid='id', tokenize='trigram'
);
```

Populated inside the existing indexer transaction (no triggers needed — ragx owns all writes).
`bm25(chunks_fts)` gives the ranking (note: FTS5 rank is "smaller = better").

**Tokenizer choice is the EN+NL crux:**

- `unicode61` (default): word-boundary, no stemming, no compound handling. Porter stemmer is
  English-only. Dutch stemming needs the
  [fts5-snowball extension](https://github.com/abiliojr/fts5-snowball) — a C extension build,
  which is exactly the infra friction ragx avoids.
- `trigram`: language-agnostic 3-gram matching — handles Dutch compounds (query "factuur" matches
  "factuurverwerking"), diacritics-adjacent forms, code identifiers, and partial matches, at
  ~2–3× index size and no linguistic stemming. Given EN+NL + technical identifiers, **trigram is
  the right default for this corpus**; compound-word languages are a documented failure of
  word-level tokenizers ([HN discussion](https://news.ycombinator.com/item?id=41198422)).
- Query strings must be escaped/quoted before `MATCH` (FTS5 has operator syntax; raw user/LLM
  text can be a syntax error).

**Would FTS5 capture most of the lexical win without the PyTorch model?** The published evidence
(Q4) says: probably yes for this corpus class. The failure modes motivating the feature — exact
proper nouns, client names, identifiers, Dutch compounds appearing verbatim in the target note —
are precisely the queries where BM25 ≈ learned sparse, because both reduce to "match the rare
literal token"; learned weights mainly add value on *graded importance* of common terms.

---

## Q4 — BM25 phase 1 → learned sparse phase 2: what's the measured gap?

Published ablations, out-of-domain (the regime a personal-notes corpus is in — nothing was
trained on it):

- **BEIR zero-shot nDCG@10:** BM25 42.3 · SPLADE 43.2 · SPLADE++ ~46–50 · SparseEmbed 46.5
  ([SIRE, AAAI 2025 table](https://ojs.aaai.org/index.php/AAAI/article/view/38537/42499)).
  Plain SPLADE's zero-shot edge over BM25 is ~1 point; the best distillation-trained variants get
  +4–8. The same table shows the famous inversion: dense TAS-B beats BM25 in-domain and *loses*
  to it out-of-domain.
- **In-domain the gap is huge** (MS MARCO MRR@10: BM25 18.4 vs SPLADE++ 38) — but ragx will never
  be in-domain.
- **BGE-M3 vs analyzer-BM25** (Q1 table): +15.4 MIRACL (multilingual short text — the most
  ragx-like setting), +4.4 MKQA, **−1.9 MLDR** (long docs).
- 2026 survey-style figures ("BM25 42.9 vs SPLADE 51.3 BEIR"; "hybrid +15–30% recall over
  single-method") circulate but are secondary
  ([searchatlas](https://searchatlas.com/blog/dense-vs-sparse-retrieval),
  [supermemory guide](https://supermemory.ai/blog/hybrid-search-guide)); treat as directional.

**Conclusion:** phasing is not just viable, it's the epistemically correct order. Phase 1 (FTS5)
answers the *real* question — "does this corpus/eval have lexical headroom at all?" — for one day
of work. The expected additional lift of learned sparse over a well-tokenized BM25 is single-digit
nDCG points in published out-of-domain evals, on a 26-query eval where one query ≈ 3.8 points of
recall. Phase 2 is only worth building if phase 1 shows (i) lexical seeds rescue some queries AND
(ii) other lexical-shaped failures remain that trace to term *weighting* rather than tokenization.

---

## Q5 — Cost of running BGE-M3 sparse locally (the painful part)

- **Model:** BAAI/bge-m3 = XLM-RoBERTa-large backbone, ~568M params, ~2.27 GB fp16.
  FlagEmbedding (`BGEM3FlagModel`) is PyTorch; `pip install FlagEmbedding` pulls torch +
  transformers. ragx already depends on sentence-transformers for the reranker
  (bge-reranker-v2-m3 — same XLM-R-large size class), so the *dependency* infra exists; the
  *memory* cost is a second ~2.3 GB model resident at query time.
- **LM Studio / Ollama / llama.cpp cannot help:** GGUF embedding endpoints return dense vectors
  only; the sparse head (`W_lex`) is not part of any OpenAI-compatible response schema. Verified
  project constraint; every sparse deployment found in research runs FlagEmbedding (PyTorch) or a
  server wrapping it.
- **Apple Silicon:** FlagEmbedding's `devices` parameter passes strings through to torch, so
  `"mps"` is *accepted* but not explicitly supported/tested upstream (multi-process pooling and
  fp16 paths are CUDA-oriented; community reports of MPS quirks with fp16). Safe assumption:
  CPU or MPS-with-fp32, i.e., meaningfully slower than the llama.cpp numbers. For calibration,
  bge-m3 *dense* via Ollama/llama.cpp on a current Mac mini does ~5.9 req/s at p50 ≈ 159 ms
  ([nullmirror benchmark](https://nullmirror.com/en/blog/2026-02-28-embedding-models-on-affordable-cloud-vms-and-apple-silicon));
  PyTorch-MPS for a short query encode is plausibly 100–400 ms. MLX runs bge-m3 ~50% faster than
  llama.cpp for batch embedding ([contracollective](https://contracollective.com/blog/local-embeddings-apple-silicon-nomic-bge-qwen3-m5-max-2026))
  but MLX ports don't expose the sparse head either.
- **Index-time (one-off, tolerable):** 1,323 chunks × ~800 tokens, batched — minutes on MPS,
  maybe tens of minutes CPU. Comparable to the 4.5× embed-cost lesson from the subchunk
  experiment: acceptable if it buys measured quality.
- **Query-time (recurring, the real cost):** every query needs a sparse encode of the query text
  (and ideally the expansion variants → 5 short forward passes, batchable into one). Adds
  ~0.1–0.5 s latency + 2.3 GB RAM + model load time (or a resident process). This lands on every
  single query forever, which is why phase-gating on measured headroom matters.
- **Not investigated in depth:** community ONNX exports of bge-m3 that include the sparse head
  exist; ONNX-on-CoreML/CPU could cut the PyTorch dependency and memory. Open question below.

---

## Concrete ragx integration sketch (not built — design only)

### Phase 1 (FTS5)

- **Schema (store.py, user_version 3):** `chunks_fts` external-content FTS5 table (above);
  populate/delete inside the same transactions that write `chunks` (indexer owns all writes; no
  triggers). Migration backfills from existing `chunks`.
- **Query path (query.py step 2):** after building dense `rankings`, if enabled:
  `sparse_ranking = store.fts_search(escaped(text), top=cfg lexical.seed_top)` →
  `rankings.append(sparse_ranking)` → existing `rrf()` unchanged. Everything downstream
  (traversal, rerank, combine) untouched.
- **Config (add to DEFAULTS first — `Config.set` rejects unknown keys):**
  ```toml
  [lexical]
  enabled = false          # opt-in, like edge_source
  backend = "fts5"         # "fts5" | "bge-m3" (phase 2)
  tokenizer = "trigram"    # fts5 only: "trigram" | "unicode61"
  seed_top = 20            # matches fusion.per_query_top
  include_variants = false # also run expansion variants through the lexical leg
  ```
- **Manifest guard:** `lexical_backend` + `lexical_tokenizer` keys, guarded like
  `edge_source` — `--changed` after a flip fails loud.
- **Explain trace:** tag seeds with their source list(s) (`dense`, `lexical`) so `--explain` can
  justify lexical-rescued results; also makes the eval failure analysis trivial.

### Phase 2 (BGE-M3 sparse) — only if measured headroom remains

- `sparse_postings` table (schema above) + a `SparseEncoder` protocol in `providers/base.py` with
  a FlagEmbedding-backed implementation behind an optional extra (`ragx-cli[sparse]`), mirroring
  how `st_reranker.py` wraps sentence-transformers. Index-time: encode chunks
  (`return_sparse=True, return_dense=False`), write postings. Query-time: encode query (+
  variants), SQL scoring query, feed ranking into the same RRF slot. Manifest guards
  `sparse_model`.

### Measurement protocol (26-query EN+NL eval, `.lab/harness.py`)

1. **Pre-check (before any code):** per query, is the target file in the seed set today? Do the
   query's rare terms appear verbatim in the target? Classify the current misses:
   lexical-shaped vs semantic-shaped. No lexical-shaped misses → stop.
2. Baseline (current tuned config: hops=3, α=.9) vs **+FTS5-unicode61** vs **+FTS5-trigram**,
   recall@5/@10/MRR, full pipeline AND `--no-rerank` (to confirm the rerank-rescues-recall
   mechanism rather than hoping).
3. Ablate `seed_top` (10/20/50) and `include_variants`.
4. Per-query delta table — with n=26, report *which* queries moved, not just averages; NL queries
   and client-name queries are the hypothesis-bearing subset.
5. Only then, same protocol for the M3-sparse backend; decision threshold: it must beat FTS5 by
   more than rebuild-cost-equivalent noise (the rebuild non-determinism noted in the 2026-07
   param-tuning memory applies here too — hold the dense index fixed across arms).

---

## Risks

- **No headroom (highest likelihood):** eval queries are verbose/semantic; dense+expansion+rerank
  already hits .889 recall@10. The sub-chunk experiment on this exact corpus proved a plausible
  mechanism can tie baseline. The pre-check exists to catch this for free.
- **Hybrid hurting a strong dense baseline:** documented (BM25+BGE < BGE alone,
  [FlagEmbedding #17](https://github.com/FlagOpen/FlagEmbedding/issues/17)). Mitigations: fuse at
  seeds (adds candidates, doesn't override dense ordering), keep rerank on, measure `--no-rerank`
  arm explicitly.
- **RRF dilution asymmetry:** 1 lexical list vs up to 5 dense lists; if the lexical signal is
  real but drowned, weighted RRF is the fix — but measure first.
- **FTS5 MATCH syntax injection:** LLM-generated variants contain quotes/operators; must escape.
  Trigram queries under 3 chars return nothing.
- **FTS5 external-content desync:** external-content tables silently return garbage if the
  content table changes outside the FTS transaction; ragx owns all writes, but tests must cover
  the incremental (`--changed`) path.
- **M3-sparse specifics:** no term expansion (won't fix true vocabulary mismatch — that's what
  dense/HyDE are for); MPS support in FlagEmbedding is untested upstream; 2nd resident 2.3 GB
  model; index invalidation on model change.
- **26 queries is a small eval:** one query = 3.8 recall points; per-query analysis and
  seed-provenance traces matter more than the headline metric.

## Open questions

- Do ONNX exports of bge-m3 with the sparse head (community `bge-m3-onnx` repos) produce
  weights matching FlagEmbedding closely enough to swap in? Would remove the PyTorch/memory
  objection to phase 2.
- Would running expansion *variants* through the lexical leg help or add noise? (Variants are
  paraphrases — lexically they may diverge from the target's actual terms.)
- Is there value in a degenerate "phase 0": exact-substring boost for known entity names (client
  list is small and enumerable) — even cheaper than FTS5, though it starts growing a bespoke
  matcher that FTS5 already is. Library-over-build says FTS5.
- If phase 2 ever lands: BGE-M3 could also replace the dense model (nomic) so dense+sparse come
  from one encoder — the `notes-wt-bgem3` worktree suggests this was already contemplated; that
  is a separate experiment with its own manifest invalidation.

## Sources (key)

| Source | Role | Credibility |
|---|---|---|
| [BGE-M3 paper, arXiv:2402.03216](https://arxiv.org/html/2402.03216v3) | sparse mechanism + BM25 comparison tables | primary, peer-reviewed (ACL Findings 2024) |
| [FlagEmbedding m3.py source](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/embedder/encoder_only/m3.py) | lexical_weights data structure, scoring impl, device handling | primary source code |
| [Bruch et al., arXiv:2210.11934](https://arxiv.org/abs/2210.11934) | CC-vs-RRF fusion analysis | primary, TOIS |
| [T2-RAGBench, arXiv:2604.01733](https://arxiv.org/html/2604.01733v1) | 2026 fusion + hybrid ablations incl. CC vs RRF numbers | primary, recent |
| [SIRE AAAI 2025 table](https://ojs.aaai.org/index.php/AAAI/article/view/38537/42499) | BM25 vs SPLADE/learned-sparse BEIR zero-shot gap | primary, peer-reviewed |
| [FlagEmbedding issue #17](https://github.com/FlagOpen/FlagEmbedding/issues/17) | hybrid-hurts-strong-dense negative result | practitioner report + author replies |
| [Simon Willison: hybrid FTS5+vec SQLite](https://simonwillison.net/2024/Oct/4/hybrid-full-text-search-and-vector-search-with-sqlite/) | canonical SQLite hybrid-RRF pattern | high-credibility practitioner |
| [vstash, arXiv:2604.15484](https://arxiv.org/html/2604.15484) | SQLite-only hybrid validated on BEIR, latency numbers | primary, recent preprint |
| [Qdrant hybrid queries docs](https://qdrant.tech/documentation/search/hybrid-queries) / [Weaviate forum](https://forum.weaviate.io/t/hybrid-search-explanation-explanation/2251) | vendor fusion practice (RRF/DBSF/relative-score) | official docs |
| [nullmirror Apple Silicon embed benchmark](https://nullmirror.com/en/blog/2026-02-28-embedding-models-on-affordable-cloud-vms-and-apple-silicon) | bge-m3 latency calibration on Macs | independent benchmark (llama.cpp path, dense-only) |
| [fts5-snowball](https://github.com/abiliojr/fts5-snowball) / [HN FTS5 thread](https://news.ycombinator.com/item?id=41198422) | Dutch stemming/compound-word limits of FTS5 tokenizers | community, directional |
