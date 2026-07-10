# ragx module contracts — Phase 2–3 build

Same ground rules as CONTRACTS.md (read its "Global rules" and "Report protocol" — they apply
verbatim). Own ONLY your listed files; never modify shared/core files or other modules' files.
Shared APIs you may rely on (already implemented — read them first):
`core/models.py`, `core/errors.py`, `core/config.py`, `core/store.py` (esp. `neighbors`,
`replace_edges`), `core/vectors.py` (esp. `search`, `get_vectors`), `core/query.py`
(`QueryOptions`, `QueryOutput`, `run_query`, `to_files_json`), `providers/base.py`,
`cli/output.py`. NOTE: `cli/app.py` wiring is integration-owned — CLI modules expose
`register(app)` and their tests build a fresh `typer.Typer()` and call `register` themselves.

---

## Module F — kNN edge construction

**Files:** `src/ragx/core/graph.py`, `tests/test_graph.py`

```python
def knn_edges(index: VectorIndex, ids: Sequence[int], k: int, min_sim: float) -> dict[int, list[tuple[int, float]]]:
    # for each id: search its own stored vector (index.get_vectors) for k+1 hits,
    # drop self, keep sim >= min_sim, cap at k. Returns id -> [(neighbor_id, sim), ...] sim-desc.
def affected_ids(new_edges: dict[int, list[tuple[int, float]]]) -> set[int]:
    # every neighbor id referenced in new_edges values, minus new_edges' own keys —
    # the incremental-maintenance set whose edge lists the indexer must recompute.
```

Pure functions over VectorIndex; no Store access. Batch `get_vectors` in one call.
Tests: constructed unit vectors with known cosines — self excluded, min_sim filter, k cap,
sim-desc order, affected_ids correctness, empty ids -> {}.

---

## Module G — heat-propagation traversal

**Files:** `src/ragx/core/traversal.py`, `tests/test_traversal.py`

```python
@dataclass
class TraversalResult:
    heat: dict[int, float]   # node -> final heat
    trace: dict[int, dict]   # node -> {"seed": int, "parent": int | None, "edge_weight": float, "hop": int}

def propagate_heat(
    seeds: dict[int, float],
    neighbors_fn: Callable[[int], list[tuple[int, float]]],
    query_sim_fn: Callable[[Sequence[int]], dict[int, float]],   # BATCH: ids -> cosine vs query
    *, hops: int = 2, decay: float = 0.5, query_floor: float = 0.35, max_frontier: int = 150,
) -> TraversalResult:
```

Semantics (exact):
- Init: every seed gets heat = its seed score, trace {"seed": itself, "parent": None,
  "edge_weight": 0.0, "hop": 0}. Seeds are exempt from the query floor.
- For hop h = 1..hops: frontier = nodes whose heat was set/improved in the previous hop
  (hop 1: the seeds). For each frontier node u and each (v, w) from neighbors_fn(u):
  contribution = heat[u] * w * decay. A non-seed v is admissible only if its query similarity
  >= query_floor — batch query_sim_fn over all first-seen candidates of the hop, cache results.
  Aggregation is MAX, not sum: heat[v] = max(existing, contribution); on improvement, v's trace
  becomes {"seed": trace[u]["seed"], "parent": u, "edge_weight": w, "hop": h}.
- Frontier cap: after each hop keep only the top `max_frontier` improved nodes by heat as the
  next frontier (others keep their heat but don't propagate). Ties break by lower node id.
- A seed's heat never decreases below its seed score. hops=0 -> seeds only.
- Deterministic for identical inputs; no randomness.

Tests (hand-built adjacency dicts, no Store): max-not-sum at a hub node; per-hop decay values
exact (seed 1.0, w 0.8 -> hop1 0.4, hop2 chain 0.16); query-floor exclusion (and that an excluded
node also doesn't relay); frontier cap; trace parent-chain always reaches a seed; hops=0.

---

## Module H — expansion, RRF fusion, score combination

**Files:** `src/ragx/core/expansion.py`, `src/ragx/core/fusion.py`, `src/ragx/core/scoring.py`,
`tests/test_expansion.py`, `tests/test_fusion.py`

```python
# expansion.py
@dataclass
class Expansion:
    variants: list[str]   # reformulations/sub-queries, excludes the original
    hyde: str | None

def expand_query(gen: Generator, query: str, *, variants: int = 3, hyde: bool = True) -> Expansion:
    # ONE gen.generate() call. Prompt demands strict JSON: {"queries": [...], "hyde": "..."}
    # (omit hyde from the prompt when hyde=False). Parse defensively: strip markdown fences,
    # take first '{' .. last '}'. On ANY failure (generator exception, bad JSON, wrong shape):
    # log a warning to stderr and return Expansion([], None) — NEVER raise. Clamp to `variants`
    # items; drop empty strings and case-insensitive duplicates of the original query.

# fusion.py
def rrf(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    # Reciprocal Rank Fusion: score(d) = sum over rankings containing d of 1/(k + rank), rank 1-based.

# scoring.py
def normalize(scores: dict[int, float]) -> dict[int, float]:
    # min-max to [0,1]; empty -> {}; all-equal -> all 1.0
def combine(candidates: Iterable[int], vector: dict[int, float], heat: dict[int, float],
            rerank: dict[int, float] | None, *, alpha: float, beta: float, gamma: float) -> dict[int, float]:
    # missing candidate scores are 0.0 raw; each component min-max normalized over candidates first.
    # with rerank:  alpha*r + beta*h + gamma*v
    # without rerank: weights renormalized -> (beta/(beta+gamma))*h + (gamma/(beta+gamma))*v
```

Tests: expansion with a mock Generator returning valid JSON / fenced JSON / garbage / raising;
rrf hand-computed values and stable ordering; normalize edge cases; combine with and without
rerank (weight renormalization exact).

---

## Module I — eval harness

**Files:** `src/ragx/core/eval.py`, `src/ragx/cli/eval_cmd.py`, `tests/test_eval.py`

`queries.jsonl`: one JSON object per line: `{"query": str, "relevant_files": [str, ...]}`
(paths relative to corpus root). Blank lines skipped; malformed line -> RagxError naming the line number.

```python
# eval.py — decoupled from the pipeline via an injected callable
QueryFn = Callable[[str, QueryOptions], QueryOutput]

def load_queries(path: Path) -> list[dict]
def evaluate(queries: list[dict], configs: list[tuple[str, QueryOptions]], query_fn: QueryFn,
             *, top: int = 10) -> dict:
    # For each config and query: out = query_fn(query, opts); rank FILES via
    # ragx.core.query.to_files_json(out)["files"] order. Metrics per config, averaged over queries:
    #   recall_at_5, recall_at_10  (fraction of relevant_files present in top k files)
    #   mrr                        (reciprocal rank of the FIRST relevant file, 0 if absent)
    # Returns {"schema": "ragx.eval.v1", "top": top, "query_count": n,
    #          "configs": [{"name", "recall_at_5", "recall_at_10", "mrr"}, ...]}
    # plus per-query detail under "queries": [{"query", "per_config": {name: {"hit_ranks": [...]}}}]

# eval_cmd.py
def register(app: typer.Typer) -> None: ...
# ragx eval QUERIES_FILE [--json] [--top N] [--configs csv]
# built-in configs: baseline (expand/graph/rerank all off), graph (only graph on),
#                   full (all on). Default: run all three.
# Constructs QueryOptions accordingly and query_fn from run_query + make_embedder(cfg) —
# keep this thin; integration may extend it. Human output: aligned table of config x metrics.
# Exit codes: 0 ok, 2 on missing/malformed file.
```

Tests: metric math against a fake query_fn with canned QueryOutput fixtures (verify recall@5/10
and MRR by hand-computed values); load_queries error cases; CLI registered on a fresh Typer with
a monkeypatched evaluate/query_fn (no embeddings, no network).

---

## Module J — inspect commands

**Files:** `src/ragx/cli/inspect_cmd.py`, `tests/test_inspect.py`

```python
def register(app: typer.Typer) -> None:   # app.add_typer(inspect_app, name="inspect")
# ragx inspect chunk ID [--json]      -> chunk text, file, line/byte range + its edges
#                                        (neighbor id, weight, neighbor's file path)
# ragx inspect file PATH [--json]     -> file record + its chunks (id, line range, first 80 chars)
# ragx inspect neighbors ID [--json]  -> store.neighbors(ID) enriched with each neighbor's file/lines
```

All read-only over Store (find root via require_root; no embeddings, no vectors file needed).
JSON schemas: `ragx.inspect.chunk.v1`, `ragx.inspect.file.v1`, `ragx.inspect.neighbors.v1`.
Exit codes: unknown chunk id or file path -> stderr message, exit 2; entity exists but has no
edges/chunks -> exit 1 (success-but-empty), matching the CLI conventions.
Tests: CliRunner on a fresh Typer + `register`; build a real Store in tmp_path with a few files/
chunks/edges directly (no embedding provider); cover all three subcommands, --json shape,
exit codes 0/1/2.
