"""`doctor`: verify the effective config end to end — every configured provider is
reachable (llama-server engines get spawned here exactly as index/query would spawn
them) and the index on disk still matches. Pure logic; the CLI shell formats the
report and maps it to an exit code."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import httpx

from ragx.core.config import PROVIDER_SECTIONS, Config, config_path, db_path, rc_path
from ragx.core.errors import RagxError
from ragx.core.indexer import corpus_drift
from ragx.core.store import Store
from ragx.providers import lmstudio_process
from ragx.providers.base import Embedder, Generator, Reranker

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
PROBE = "ragx doctor probe"

EmbedderFactory = Callable[[Config], Embedder]
GeneratorFactory = Callable[[Config], Generator | None]
RerankerFactory = Callable[[Config], Reranker | None]


@dataclass
class Check:
    name: str
    status: str  # OK | WARN | FAIL | SKIP
    detail: str
    hint: str = ""
    elapsed_ms: int = 0


@dataclass
class DoctorReport:
    root: Path
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != FAIL for c in self.checks)


def run_doctor(
    root: Path,
    cfg: Config,
    *,
    embedder_factory: EmbedderFactory,
    generator_factory: GeneratorFactory,
    reranker_factory: RerankerFactory,
) -> DoctorReport:
    """Run every check in order. Never raises: failures become FAIL checks."""
    return DoctorReport(
        root,
        [
            _timed("config", lambda: _check_config(root, cfg)),
            _timed("embeddings", lambda: _check_embeddings(cfg, embedder_factory)),
            _timed("expansion", lambda: _check_expansion(cfg, generator_factory)),
            _timed("rerank", lambda: _check_rerank(cfg, reranker_factory)),
            _timed("index", lambda: _check_index(root, cfg)),
        ],
    )


def to_doctor_json(report: DoctorReport) -> dict:
    return {
        "schema": "ragx.doctor.v1",
        "root": str(report.root),
        "ok": report.ok,
        "checks": [vars(c) for c in report.checks],
    }


def _timed(name: str, fn: Callable[[], tuple[str, str, str]]) -> Check:
    start = time.monotonic()
    try:
        status, detail, hint = fn()
    except RagxError as exc:  # every expected provider failure lands here
        status, detail, hint = FAIL, str(exc), ""
    return Check(name, status, detail, hint, int((time.monotonic() - start) * 1000))


def _check_config(root: Path, cfg: Config) -> tuple[str, str, str]:
    line = " · ".join(
        f"{s}={cfg.get(f'{s}.provider')}/{cfg.get(f'{s}.model')}" for s in PROVIDER_SECTIONS
    )
    detail = f"{config_path(root)} — {line}"
    if rc_path().exists():
        detail += f" (+ {rc_path()} overrides)"
    return OK, detail, ""


def _check_embeddings(cfg: Config, factory: EmbedderFactory) -> tuple[str, str, str]:
    provider = cfg.get("embeddings.provider")
    done = _prefix(lmstudio_process.ensure_for_section(cfg, "embeddings"))
    try:
        embedder = factory(cfg)
        vectors = embedder.embed_queries([PROBE])
    except RagxError as exc:
        return FAIL, f"{done}{exc}", _embed_hint(cfg, provider)
    if not vectors or not vectors[0]:
        return FAIL, f"{done}{provider}/{cfg.get('embeddings.model')} returned an empty embedding", ""
    return OK, f"{done}{provider}/{embedder.model} answered, dim={len(vectors[0])}", ""


def _prefix(actions: list[str]) -> str:
    return f"{'; '.join(actions)} — " if actions else ""


def _embed_hint(cfg: Config, provider: str) -> str:
    """llama-server errors already say what to do; LM Studio needs a pointer to whichever
    half of the handshake broke — autostart being off, or the start itself."""
    if provider == "llama-server":
        return ""
    base_url = cfg.get("embeddings.base_url")
    if not cfg.get("embeddings.autostart"):
        return (
            f"autostart is off — start the server yourself at {base_url} (LM Studio: "
            "`lms server start`), or `ragx-cli config set embeddings.autostart true`"
        )
    return (
        f"check `lms server status` and that {cfg.get('embeddings.model')!r} is downloaded "
        "(`ragx-cli models`), or hand the process to ragx with "
        "`ragx-cli models --embed-engine llama-server`"
    )


def _check_expansion(cfg: Config, factory: GeneratorFactory) -> tuple[str, str, str]:
    if not cfg.get("expansion.enabled"):
        return SKIP, "disabled (expansion.enabled = false)", ""
    generator = factory(cfg)  # constructing it validates api_key_env; no network yet
    if generator is None:
        return SKIP, "disabled (expansion.enabled = false)", ""
    done = _prefix(lmstudio_process.ensure_for_section(cfg, "expansion"))
    base_url = getattr(generator, "base_url", cfg.get("expansion.base_url"))
    served = _probe_models(base_url)
    off_hint = "turn it off with `ragx-cli config set expansion.enabled false`"
    if served is None:
        return FAIL, f"{done}no server answered at {base_url}/models", off_hint
    if generator.model not in served:
        return (
            WARN,
            f"{done}{base_url} is up but does not list {generator.model!r}",
            "LM Studio may still load it on demand (JIT); otherwise fix expansion.model",
        )
    return OK, f"{done}{base_url} serves {generator.model!r}", ""


def _probe_models(base_url: str) -> list[str] | None:
    """Model ids served at `base_url`, or None when nothing answered."""
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=5.0)
        resp.raise_for_status()
        data = resp.json().get("data", [])
    except Exception:
        return None
    return [m["id"] for m in data if isinstance(m, dict) and "id" in m]


def _check_rerank(cfg: Config, factory: RerankerFactory) -> tuple[str, str, str]:
    if not cfg.get("rerank.enabled"):
        return SKIP, "disabled (rerank.enabled = false)", ""
    reranker = factory(cfg)
    if reranker is None:
        return FAIL, "reranker could not be built", "see the warning above"
    scores = reranker.score(PROBE, ["a completely unrelated sentence", PROBE])
    provider = cfg.get("rerank.provider")
    return (
        OK,
        f"{provider}/{reranker.model} scored a probe pair ({scores[0]:.2f}, {scores[1]:.2f})",
        "",
    )


def _check_index(root: Path, cfg: Config) -> tuple[str, str, str]:
    db = db_path(root)
    if not db.exists():
        return WARN, "no index yet", "run `ragx-cli index`"
    with Store(db) as store:
        chunks, files = store.chunk_count(), store.file_count()
        built_with = store.get_meta("embedding_model")
    if chunks == 0:
        return WARN, "index is empty", "run `ragx-cli index`"
    configured = cfg.get("embeddings.model")
    if built_with and built_with != configured:
        return (
            FAIL,
            f"index was built with {built_with!r} but config says {configured!r}",
            "run `ragx-cli index --full` to rebuild",
        )
    drift = corpus_drift(root, cfg)
    detail = f"{files} files, {chunks} chunks, built with {built_with!r}"
    if any(drift.values()):
        return (
            WARN,
            f"{detail} — {drift['new']} new, {drift['changed']} changed, {drift['deleted']} deleted",
            "run `ragx-cli index`",
        )
    return OK, f"{detail}, no drift", ""
