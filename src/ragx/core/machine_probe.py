"""What this machine already has for embeddings/rerank, and which of it `init` reuses.

`probe_machine` gathers facts through injected live probes (`lms ls/ps`, llama-server
`/props` on the fixed ports, `llama-server` on PATH, the Hugging Face cache) — no registry
of corpora. `pick_reuse` ranks them and is pure. Only catalog models are auto-picked:
their doc/query prefixes are known, and wrong prefixes silently corrupt an index."""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragx.core import lmstudio
from ragx.core.catalog import EMBEDDINGS, RERANKER, EmbeddingChoice
from ragx.core.config import DEFAULTS
from ragx.core.errors import RagxError
from ragx.core.lmstudio import InstalledModel
from ragx.providers import llama_process

LMSTUDIO_URL = DEFAULTS["embeddings"]["base_url"]
LLAMA_EMBED_URL = "http://127.0.0.1:9813/v1"  # registry's llama-server embeddings default
LLAMA_RERANK_URL = DEFAULTS["rerank"]["base_url"]
TIER_ORDER = ("balanced", "best", "fast", "jina-nano")

log = logging.getLogger("ragx.probe")


@dataclass(frozen=True)
class MachineState:
    lms_loaded: tuple[InstalledModel, ...] = ()  # `lms ps --json`
    lms_installed: tuple[InstalledModel, ...] = ()  # `lms ls --json`
    models_root: Path = field(default_factory=Path)  # LM Studio models dir (ls paths are relative)
    embed_server_gguf: str | None = None  # /props model_path of a llama-server on 9813
    rerank_server_gguf: str | None = None  # same, on 9814
    llama_on_path: bool = False
    reranker_cached: bool = False  # HF cache holds RERANKER.model + sentence-transformers importable


@dataclass(frozen=True)
class Reuse:
    label: str  # catalog label, e.g. "BGE-M3"
    reason: str  # shown to the user, e.g. "loaded in LM Studio"
    values: dict[str, Any]  # dotted provider keys to write


@dataclass(frozen=True)
class ReuseChoice:
    embeddings: Reuse | None
    rerank: Reuse | None


def hf_cached(repo: str, cache_dir: str | None = None) -> bool:
    """True when sentence-transformers is importable and `repo` is in the HF cache."""
    if importlib.util.find_spec("sentence_transformers") is None:
        return False
    from huggingface_hub import try_to_load_from_cache  # a sentence-transformers dependency

    return isinstance(try_to_load_from_cache(repo, "config.json", cache_dir=cache_dir), str)


def probe_machine(
    *,
    find_lms: Callable[[], str | None] = lmstudio.find_lms,
    serving_gguf: Callable[[str], str | None] = llama_process.serving_gguf,
    which: Callable[[str], str | None] = shutil.which,
    reranker_cached: Callable[[str], bool] = hf_cached,
) -> MachineState:
    """Best effort: a probe that fails counts as "nothing there", never as an error."""
    lms = find_lms()
    loaded: list[InstalledModel] = []
    installed: list[InstalledModel] = []
    if lms is not None:
        try:
            loaded = lmstudio.list_loaded(lms)
            installed = lmstudio.list_models(lms)
        except (RagxError, OSError, subprocess.TimeoutExpired) as exc:
            log.debug("LM Studio probe failed: %s", exc)
    return MachineState(
        lms_loaded=tuple(loaded),
        lms_installed=tuple(installed),
        models_root=lmstudio.models_root(),
        embed_server_gguf=serving_gguf(LLAMA_EMBED_URL),
        rerank_server_gguf=serving_gguf(LLAMA_RERANK_URL),
        llama_on_path=which("llama-server") is not None,
        reranker_cached=reranker_cached(RERANKER.model),
    )


def pick_reuse(state: MachineState) -> ReuseChoice | None:
    emb, rr = _pick_embeddings(state), _pick_rerank(state)
    return ReuseChoice(emb, rr) if emb or rr else None


def _matches(fragment: str, model: InstalledModel) -> bool:
    needle = fragment.lower()
    return needle in model.model_key.lower() or needle in model.path.lower()


def _gguf(models: tuple[InstalledModel, ...], fragment: str) -> InstalledModel | None:
    """A downloaded single-file GGUF llama-server can load (not an MLX/model directory)."""
    return next(
        (m for m in models if _matches(fragment, m) and m.path.lower().endswith(".gguf")), None
    )


def _lmstudio_embedding(models: tuple[InstalledModel, ...], e: EmbeddingChoice) -> InstalledModel | None:
    if e.requires_llama_server:  # LM Studio downloads it but cannot serve it
        return None
    return next((m for m in models if m.type == "embedding" and _matches(e.repo_fragment, m)), None)


def _embed_values(e: EmbeddingChoice, model: str, gguf: str | None) -> dict[str, Any]:
    values: dict[str, Any] = {
        "embeddings.model": model,
        "embeddings.doc_prefix": e.doc_prefix,
        "embeddings.query_prefix": e.query_prefix,
    }
    if gguf is None:
        values |= {"embeddings.provider": "openai", "embeddings.base_url": LMSTUDIO_URL}
    else:
        values |= {"embeddings.provider": "llama-server", "embeddings.base_url": LLAMA_EMBED_URL,
                   "embeddings.gguf": gguf}
    return values


def _pick_embeddings(state: MachineState) -> Reuse | None:
    tiers = [EMBEDDINGS[t] for t in TIER_ORDER]
    for e in tiers:  # 1. already in LM Studio's memory
        m = _lmstudio_embedding(state.lms_loaded, e)
        if m is not None:
            return Reuse(e.label, "loaded in LM Studio", _embed_values(e, m.model_key, None))
    serving = state.embed_server_gguf
    for e in tiers if serving else ():  # 2. a llama-server on 9813 serves it
        if e.gguf_fragment.lower() in serving.lower():
            resolved = Path(serving).resolve()
            same = (m for m in state.lms_installed if (state.models_root / m.path).resolve() == resolved)
            key = next((m.model_key for m in same), Path(serving).stem)
            return Reuse(e.label, "served by a running llama-server", _embed_values(e, key, serving))
    for e in tiers:  # 3. downloaded; engine as `ragx-cli models` defaults it
        g = _gguf(state.lms_installed, e.gguf_fragment) if state.llama_on_path else None
        if g is not None:
            gguf = str(state.models_root / g.path)
            return Reuse(e.label, "downloaded in LM Studio, served by llama-server",
                         _embed_values(e, g.model_key, gguf))
        m = _lmstudio_embedding(state.lms_installed, e)
        if m is not None:
            return Reuse(e.label, "downloaded in LM Studio", _embed_values(e, m.model_key, None))
    return None


def _pick_rerank(state: MachineState) -> Reuse | None:
    llama = {"rerank.provider": "llama-server", "rerank.model": RERANKER.model}
    serving = state.rerank_server_gguf
    if serving and RERANKER.gguf_fragment.lower() in serving.lower():
        return Reuse(RERANKER.label, "served by a running llama-server",
                     llama | {"rerank.gguf": serving})
    g = _gguf(state.lms_installed, RERANKER.gguf_fragment) if state.llama_on_path else None
    if g is not None:
        return Reuse(RERANKER.label, "downloaded in LM Studio, served by llama-server",
                     llama | {"rerank.gguf": str(state.models_root / g.path)})
    if state.reranker_cached:
        return Reuse(RERANKER.label, "in the Hugging Face cache",
                     {"rerank.provider": "sentence-transformers", "rerank.model": RERANKER.model})
    return None
