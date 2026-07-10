"""Config: .ragx/config.toml is the single source of truth. Written by `init`, read everywhere."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import tomli_w

from ragx.core.errors import NotInitializedError, RagxError

RAGX_DIR = ".ragx"

DEFAULTS: dict[str, dict[str, Any]] = {
    "corpus": {"include": ["**/*"], "exclude": [], "respect_gitignore": True},
    "chunking": {"size_tokens": 800, "overlap": 0.15},
    "graph": {"k": 8, "min_edge_sim": 0.55},
    "traversal": {"hops": 2, "decay": 0.5, "query_floor": 0.35, "max_frontier": 150},
    "fusion": {"rrf_k": 60, "per_query_top": 20},
    "scoring": {"alpha_rerank": 0.6, "beta_heat": 0.25, "gamma_vector": 0.15},
    "embeddings": {
        "provider": "openai",
        "base_url": "http://localhost:1234/v1",
        "model": "text-embedding-nomic-embed-text-v1.5@q4_k_m",
        "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ",
        "batch_size": 32,
        "api_key_env": "",  # NAME of an env var holding the API key; empty = no auth header
    },
    "expansion": {
        "enabled": True,
        "provider": "openai",
        "base_url": "http://localhost:1234/v1",
        "model": "mlx-community/Ornith-1.0-35B-3bit",
        "variants": 3,
        "hyde": True,
        "api_key_env": "",
    },
    "rerank": {"enabled": True, "provider": "sentence-transformers", "model": "BAAI/bge-reranker-v2-m3"},
    "query": {"top": 8, "max_chunk_chars": 1200},
}


def find_root(start: Path | None = None) -> Path | None:
    """Walk up from `start` (default cwd) looking for a .ragx/ directory."""
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / RAGX_DIR).is_dir():
            return p
    return None


def require_root(start: Path | None = None) -> Path:
    root = find_root(start)
    if root is None:
        raise NotInitializedError("no .ragx/ found — run `ragx init` first")
    return root


def config_path(root: Path) -> Path:
    return root / RAGX_DIR / "config.toml"


def db_path(root: Path) -> Path:
    return root / RAGX_DIR / "index.db"


def vectors_path(root: Path) -> Path:
    return root / RAGX_DIR / "vectors.hnsw"


def _deep_merge(base: dict, override: dict) -> dict:
    out = {k: dict(v) for k, v in base.items()}
    for section, values in override.items():
        out.setdefault(section, {}).update(values)
    return out


class Config:
    """Two-level (section.key) config with defaults merged under the file's values."""

    def __init__(self, data: dict[str, dict[str, Any]]):
        self.data = data

    @classmethod
    def load(cls, root: Path) -> Config:
        path = config_path(root)
        file_data: dict = {}
        if path.exists():
            with path.open("rb") as f:
                file_data = tomllib.load(f)
        return cls(_deep_merge(DEFAULTS, file_data))

    def get(self, dotted: str) -> Any:
        section, _, key = dotted.partition(".")
        if not key:
            raise RagxError(f"config keys are 'section.key', got: {dotted!r}")
        try:
            return self.data[section][key]
        except KeyError:
            raise RagxError(f"unknown config key: {dotted!r}") from None

    def set(self, dotted: str, value: Any) -> None:
        section, _, key = dotted.partition(".")
        if not key or section not in DEFAULTS or key not in DEFAULTS[section]:
            raise RagxError(f"unknown config key: {dotted!r}")
        current = DEFAULTS[section][key]
        if isinstance(current, bool):
            value = str(value).lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, list) and isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        self.data.setdefault(section, {})[key] = value

    def save(self, root: Path) -> None:
        path = config_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            tomli_w.dump(self.data, f)


def write_default_config(root: Path) -> Path:
    """Create .ragx/ with a default config.toml. Used by `ragx init`."""
    cfg = Config({k: dict(v) for k, v in DEFAULTS.items()})
    cfg.save(root)
    return config_path(root)
