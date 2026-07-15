"""Config: ragx.toml at the corpus root (committable; .ragx/ holds only index data),
plus ~/.ragxrc for machine-level provider settings (embeddings/expansion/rerank).
The rc overrides corpus values — with a warning."""

from __future__ import annotations

import logging
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tomli_w

from ragx.core.errors import NotInitializedError, RagxError

RAGX_DIR = ".ragx"
CONFIG_FILE = "ragx.toml"
PROVIDER_SECTIONS = ("embeddings", "expansion", "rerank")

log = logging.getLogger("ragx.config")


def rc_path() -> Path:
    return Path.home() / ".ragxrc"

DEFAULTS: dict[str, dict[str, Any]] = {
    "corpus": {"include": ["**/*"], "exclude": [], "respect_gitignore": True},
    "chunking": {"size_tokens": 800, "overlap": 0.15},
    "graph": {
        "k": 8,
        "min_edge_sim": 0.55,
        "edge_source": "chunk",  # "chunk" (whole-chunk cosine, k per chunk) | "subchunk" (k per SUB-chunk)
        "subchunk_size_tokens": 128,
        "near_dup_sim": 0.9,  # subchunk mode: drop edges whose whole-chunk cosine is already >= this
    },
    "traversal": {"hops": 2, "decay": 0.5, "query_floor": 0.35, "max_frontier": 150},
    "communities": {"resolution": 1.0, "seed": 42},
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
        "model": "qwen3.5-9b",
        "variants": 3,
        "hyde": True,
        "api_key_env": "",
    },
    "rerank": {"enabled": True, "provider": "sentence-transformers", "model": "BAAI/bge-reranker-v2-m3"},
    "query": {"top": 8, "max_chunk_chars": 1200},
}


def find_root(start: Path | None = None) -> Path | None:
    """Walk up from `start` (default cwd) looking for a ragx.toml file or .ragx/ directory."""
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / CONFIG_FILE).is_file() or (p / RAGX_DIR).is_dir():
            return p
    return None


def require_root(start: Path | None = None) -> Path:
    root = find_root(start)
    if root is None:
        raise NotInitializedError("no ragx.toml or .ragx/ found — run `ragx-cli init` first")
    return root


def config_path(root: Path) -> Path:
    return root / CONFIG_FILE


def db_path(root: Path) -> Path:
    return root / RAGX_DIR / "index.db"


def vectors_path(root: Path) -> Path:
    return root / RAGX_DIR / "vectors.hnsw"


def _migrate_legacy(root: Path, confirm: Callable[[str], bool] | None) -> None:
    """Self-heal a pre-0.3.0 corpus: move .ragx/config.toml to ragx.toml at the root.

    `confirm` (interactive callers) is asked before touching anything; declining keeps
    the old fail-loud error. Without it (agents, library use) the move just happens,
    with a stderr notice. Both files existing stays a hard error — picking one could
    silently discard edits."""
    legacy = root / RAGX_DIR / "config.toml"
    if not legacy.exists():
        return
    target = root / CONFIG_FILE
    if target.exists():
        raise RagxError(
            f"both {target} and legacy {legacy} exist — config moved to ragx.toml "
            f"in 0.3.0; keep one and delete the other"
        )
    if confirm is not None and not confirm(
        f"config moved in 0.3.0: move legacy {legacy} to {target}?"
    ):
        raise RagxError(f"config not migrated — run `mv {legacy} {target}` to proceed")
    legacy.rename(target)
    log.warning("config moved in 0.3.0: migrated %s -> %s", legacy, target)


def _deep_merge(base: dict, override: dict) -> dict:
    out = {k: dict(v) for k, v in base.items()}
    for section, values in override.items():
        out.setdefault(section, {}).update(values)
    return out


def _coerce(section: str, key: str, value: Any) -> Any:
    """Coerce a (usually string) value to the type of its DEFAULTS entry."""
    current = DEFAULTS[section][key]
    if isinstance(current, bool):
        return str(value).lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, list) and isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return value


def load_rc(rc: Path | None = None) -> dict[str, dict[str, Any]]:
    """Parse ~/.ragxrc. Only provider sections with known keys are allowed — fails loud."""
    path = rc if rc is not None else rc_path()
    if not path.exists():
        return {}
    with path.open("rb") as f:
        try:
            data = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            raise RagxError(f"malformed {path}: {exc}") from exc
    for section, values in data.items():
        if section not in PROVIDER_SECTIONS:
            raise RagxError(
                f"{path}: section [{section}] not allowed — only {', '.join(PROVIDER_SECTIONS)}"
            )
        for key in values:
            if key not in DEFAULTS[section]:
                raise RagxError(f"{path}: unknown key {section}.{key}")
    return data


def write_rc_value(dotted: str, value: Any, rc: Path | None = None) -> Any:
    """Set one provider key in ~/.ragxrc (created if missing). Returns the coerced value."""
    section, _, key = dotted.partition(".")
    if not key or section not in DEFAULTS or key not in DEFAULTS[section]:
        raise RagxError(f"unknown config key: {dotted!r}")
    if section not in PROVIDER_SECTIONS:
        raise RagxError(
            f"{dotted!r} is corpus-level — only {', '.join(PROVIDER_SECTIONS)} keys go in ~/.ragxrc"
        )
    path = rc if rc is not None else rc_path()
    data = load_rc(path)
    coerced = _coerce(section, key, value)
    data.setdefault(section, {})[key] = coerced
    with path.open("wb") as f:
        tomli_w.dump(data, f)
    return coerced


class Config:
    """Two-level (section.key) config: DEFAULTS < corpus ragx.toml < ~/.ragxrc.

    `data` is the effective merged view; `file_data` is the raw corpus file so that
    `save` never bakes rc overrides into the corpus config.
    """

    def __init__(self, data: dict[str, dict[str, Any]], file_data: dict | None = None):
        self.data = data
        self.file_data = file_data if file_data is not None else data

    @classmethod
    def load(
        cls,
        root: Path,
        rc: Path | None = None,
        confirm: Callable[[str], bool] | None = None,
    ) -> Config:
        _migrate_legacy(root, confirm)
        path = config_path(root)
        file_data: dict = {}
        if path.exists():
            with path.open("rb") as f:
                file_data = tomllib.load(f)
        rc_data = load_rc(rc)
        for section, values in rc_data.items():
            for key, rc_value in values.items():
                corpus_value = file_data.get(section, {}).get(key)
                if corpus_value is not None and corpus_value != rc_value:
                    log.warning(
                        "~/.ragxrc overrides %s.%s: %r (corpus) -> %r (rc)",
                        section, key, corpus_value, rc_value,
                    )
        return cls(_deep_merge(DEFAULTS, _deep_merge(file_data, rc_data)), file_data)

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
        coerced = _coerce(section, key, value)
        self.data.setdefault(section, {})[key] = coerced
        if self.file_data is not self.data:
            self.file_data.setdefault(section, {})[key] = coerced

    def save(self, root: Path) -> None:
        path = config_path(root)
        with path.open("wb") as f:
            tomli_w.dump(_deep_merge(DEFAULTS, self.file_data), f)


def write_default_config(root: Path) -> Path:
    """Write a default ragx.toml at the corpus root. Used by `ragx-cli init`."""
    cfg = Config({k: dict(v) for k, v in DEFAULTS.items()})
    cfg.save(root)
    return config_path(root)
