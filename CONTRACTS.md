# ragx-cli module contracts (Phase 0–1 build)

Ground truth for the module-parallel build. Each module owns exactly the files listed for it.
**Do not create or edit files outside your module** — shared types live in `core/models.py`,
provider protocols in `providers/base.py`, config in `core/config.py`, errors in `core/errors.py`
(all already written; read them first, import from them, never modify them).

Global rules:

- Python 3.12, `uv sync --group dev` to set up, `uv run pytest tests/<your tests> -q` must pass,
  `uv run ruff check <your files>` must be clean.
- Soft cap ~150 lines per file; split if you blow past it.
- No live network in tests (mock HTTP with `respx`; models/backends must not be required).
- Raise `RagxError` (or a subclass) for expected failures; let bugs raise naturally.

---

## Module A — SQLite store

**Files:** `src/ragx/core/store.py`, `tests/test_store.py`

Single-file SQLite store for files, chunks, edges, and a key-value manifest.

```python
class Store:
    def __init__(self, path: Path): ...   # open/create, apply schema, WAL mode, foreign_keys ON
    def close(self) -> None: ...           # also __enter__/__exit__
    # manifest (embedding model, dim, schema info)
    def get_meta(self, key: str) -> str | None: ...
    def set_meta(self, key: str, value: str) -> None: ...
    # files
    def get_file_hashes(self) -> dict[str, str]: ...      # path -> content_hash
    def upsert_file(self, rec: FileRecord) -> None: ...
    def delete_file(self, path: str) -> list[int]: ...    # cascades chunks + incident edges; returns removed chunk ids
    def file_count(self) -> int: ...
    # chunks
    def insert_chunks(self, file_path: str, drafts: Sequence[ChunkDraft]) -> list[int]: ...
    def get_chunks(self, ids: Sequence[int]) -> list[Chunk]: ...   # input order preserved; unknown ids skipped
    def chunk_ids_for_file(self, path: str) -> list[int]: ...
    def all_chunk_ids(self) -> list[int]: ...
    def chunk_count(self) -> int: ...
    # edges (undirected; store normalized src < dst)
    def replace_edges(self, src: int, neighbors: Sequence[tuple[int, float]]) -> None: ...
    def neighbors(self, chunk_id: int) -> list[tuple[int, float]]: ...  # both directions, weight desc
    def delete_edges_incident(self, ids: Sequence[int]) -> None: ...
    def edge_count(self) -> int: ...
```

Schema v1 (set `PRAGMA user_version = 1`):

```sql
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE files(path TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
                   mtime REAL NOT NULL, chunk_count INTEGER NOT NULL);
CREATE TABLE chunks(id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
                    text TEXT NOT NULL, byte_start INTEGER NOT NULL, byte_end INTEGER NOT NULL,
                    line_start INTEGER NOT NULL, line_end INTEGER NOT NULL);
CREATE INDEX chunks_file ON chunks(file_path);
CREATE TABLE edges(src INTEGER NOT NULL, dst INTEGER NOT NULL, weight REAL NOT NULL,
                   PRIMARY KEY(src, dst)) WITHOUT ROWID;
CREATE INDEX edges_dst ON edges(dst);
```

`AUTOINCREMENT` is deliberate: chunk ids are HNSW labels and must never be reused after deletion.
`insert_chunks` requires the file row to exist first (upsert_file before insert_chunks).

Tests must cover: roundtrip of every entity; delete_file cascade removes chunks and edges touching
them (both src and dst side); replace_edges normalization + dedup (inserting (5,3,w) then querying
neighbors(3) and neighbors(5) both work); id monotonicity after delete + reinsert; meta roundtrip.

---

## Module B — Providers

**Files:** `src/ragx/providers/openai_compat.py`, `src/ragx/providers/st_reranker.py`,
`src/ragx/providers/registry.py`, `tests/test_providers.py`

```python
# openai_compat.py — used for LM Studio (localhost:1234/v1), Ollama's /v1, and OpenAI itself
class OpenAICompatEmbedder:   # implements providers.base.Embedder
    def __init__(self, base_url: str, model: str, doc_prefix: str = "", query_prefix: str = "",
                 api_key: str | None = None, batch_size: int = 32, timeout: float = 120.0): ...
    # POST {base_url}/embeddings with {"model", "input": [prefixed texts]}
    # batches of batch_size; 3 retries with exponential backoff on 5xx/connect errors
    # dimension(): lazily embeds one probe string, caches the length
    # empty input list -> [] without any HTTP call

class OpenAICompatGenerator:  # implements providers.base.Generator
    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 timeout: float = 300.0): ...
    # POST {base_url}/chat/completions, messages=[system, user], temperature=0.3
    # returns choices[0].message.content

# st_reranker.py — lazy `import sentence_transformers` inside __init__;
# ImportError -> RagxError("... install with: uv tool install ragx-cli --with ragx-cli[rerank]")
class STReranker:             # implements providers.base.Reranker
    def __init__(self, model: str): ...
    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...  # CrossEncoder.predict

# registry.py — factories reading Config (see core/config.py DEFAULTS for keys)
def make_embedder(cfg: Config) -> Embedder: ...
    # embeddings.provider: "openai" -> OpenAICompatEmbedder with [embeddings] settings
    #                      "ollama" -> same class, base_url defaulting to http://localhost:11434/v1
    # unknown provider -> RagxError
def make_generator(cfg: Config) -> Generator | None: ...  # None when expansion.enabled is false
def make_reranker(cfg: Config) -> Reranker | None: ...
    # None when rerank.enabled is false; on RagxError from missing extra: log warning to stderr, return None
```

Tests: respx-mocked embeddings (prefix application, batching — e.g. batch_size=2 with 5 inputs →
3 requests; order preserved across batches), retry on a 500 then success, dimension() caching
(one probe call), generator payload shape, registry dispatch incl. disabled → None. No real network.

---

## Module C — Discovery + chunking

**Files:** `src/ragx/core/discovery.py`, `src/ragx/core/chunking.py`,
`tests/test_discovery.py`, `tests/test_chunking.py`

```python
# discovery.py
def discover_files(root: Path, include: list[str], exclude: list[str],
                   respect_gitignore: bool = True) -> list[str]: ...
    # sorted relative POSIX paths; include/exclude are gitwildmatch globs (use pathspec)
    # always skipped: .ragx/, .git/, hidden dirs, binaries (NUL byte in first 8KB), files > 2MB
    # gitignore: root-level .gitignore only (MVP; document the limitation)
def hash_file(path: Path) -> str: ...   # xxhash.xxh64 hexdigest of file bytes

# chunking.py
def chunk_text(text: str, path: str, size_tokens: int = 800, overlap: float = 0.15) -> list[ChunkDraft]: ...
```

Chunking rules:

- Token estimate: 1 token ≈ 4 chars → target = size_tokens*4 chars, hard max 1.5× target.
- Dispatch on extension: `.md`/`.markdown` → markdown splitter; code
  (`.py .ts .js .tsx .jsx .go .rs .java .rb`) → code splitter; anything else → recursive splitter.
- Markdown: split at heading lines (`^#{1,6} `), heading stays with its section; adjacent small
  sections merge until ≈target; oversized sections fall through to the recursive splitter.
- Code: split at top-level `def`/`class`/`function`/`fn`/`func` boundaries (regex, no tree-sitter
  in MVP); oversized pieces fall through to the recursive splitter.
- Recursive: split by separators ["\n\n", "\n", ". ", " "] to fit max, with `overlap` fraction of
  the previous chunk's tail prepended as range overlap.
- **Invariant (test this):** every draft's `text` == `text_bytes[byte_start:byte_end].decode("utf-8")`
  where text_bytes = the original file's utf-8 bytes — chunks are exact slices of the source
  (overlap = overlapping ranges, never synthesized text). line_start/line_end are 1-based inclusive.
- Empty/whitespace-only files → []. Never emit empty-text drafts.

Tests: the slice invariant on multi-byte (emoji/accented) content; markdown heading split + small-
section merging; oversized-section fallback; overlap ranges actually overlap; discovery honors
include/exclude/.gitignore, skips binaries and .ragx.

---

## Module D — HNSW vector index

**Files:** `src/ragx/core/vectors.py`, `tests/test_vectors.py`

```python
class VectorIndex:
    @classmethod
    def create(cls, path: Path, dim: int, capacity: int = 2048) -> VectorIndex: ...
    @classmethod
    def load(cls, path: Path, dim: int) -> VectorIndex: ...   # missing file -> RagxError
    def add(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None: ...
    def search(self, vector: Sequence[float], k: int) -> list[tuple[int, float]]: ...
    def mark_deleted(self, ids: Sequence[int]) -> None: ...
    def save(self) -> None: ...
    @property
    def count(self) -> int: ...   # active (non-deleted) element count
```

- hnswlib `space="cosine"`, M=16, ef_construction=200, query-time ef = max(100, 2k).
- `search` returns (id, cosine_similarity) sorted desc; similarity = 1 − hnswlib distance,
  clamped to [0,1]; k silently clamped to active count; empty index → [].
- `add` auto-resizes (double capacity) when full. ids are arbitrary non-reused ints (SQLite rowids).
- Deleted elements never appear in results (hnswlib `mark_deleted`); deleting unknown ids is a no-op.
- save/load roundtrip preserves search results and deletions.

Tests: nearest-neighbor sanity on constructed vectors (a query nearest to its own cluster),
resize past initial capacity, delete-then-search exclusion, save/load roundtrip, empty-index search.

---

## Module E — CLI shell

**Files:** `src/ragx/cli/app.py`, `src/ragx/cli/output.py`, `tests/test_cli.py`

typer app exposing `init`, `status`, `config`. (`index`/`query` are wired in a later integration
pass — the app must be importable and testable without them.)

Conventions (these ARE the product for agent callers — be exact):

- `--json` on `status`: emit exactly one JSON document to stdout, nothing else on stdout.
- All logging/diagnostics → stderr (configure `logging` to stderr in `main()`).
- Exit codes: 0 = success with results, 1 = success but empty result, 2 = error.
  Map `RagxError` → stderr message + exit 2 (no traceback); unexpected exceptions may traceback.

```python
# output.py
def emit_json(doc: dict) -> None: ...           # json.dumps to stdout with trailing newline
def fail(msg: str, code: int = 2) -> NoReturn: ...

# app.py
app = typer.Typer(...)
# ragx-cli init [path]      -> write_default_config at path (default cwd); error (exit 2) if .ragx-cli exists;
#                          prints created config path
# ragx-cli status [--json]  -> root path, embedding model/provider, counts (files/chunks/edges) read via
#                          Store if index.db exists else zeros, schema "ragx.status.v1" in JSON mode
# ragx-cli config get KEY / ragx-cli config set KEY VALUE  -> Config.get / Config.set + save
def main() -> None: ...   # entry point declared in pyproject [project.scripts]
```

Module E may import Store (module A) **only** inside `status` behind a
`try/except ImportError` fallback to zeros, so the module builds independently; the import will
resolve after integration. Tests: use `typer.testing.CliRunner` + tmp_path; cover init (fresh +
already-initialized), config get/set roundtrip persisted to disk, status --json shape + stdout
purity (stdout parses as JSON), exit codes incl. `config get bogus.key` → 2.

---

## Report protocol (all modules)

When done: `git add -A && git commit -m "<module>: <summary>"` in your worktree, then report:
branch (`git rev-parse --abbrev-ref HEAD`), commit SHA, pytest pass/fail counts, ruff status,
and any deviations from this contract. On any blocker: STOP — report the exact error output and
what you attempted instead of improvising around the contract.
