"""LM Studio integration: locate the `lms` CLI, download models, list what's installed.

LM Studio only downloads GGUF/MLX artifacts (plain safetensors repos are rejected), and
`lms get` works even when the server is down — which is why downloads shell out to the CLI
instead of the REST API. See research/lm-studio-model-download-api-*.md.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ragx.core.errors import RagxError

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@dataclass(frozen=True)
class InstalledModel:
    model_key: str
    type: str  # "embedding" | "llm"
    format: str  # "gguf" | "mlx" | ...
    path: str  # relative to the LM Studio models root
    size_bytes: int


def models_root() -> Path:
    """LM Studio's models directory. `~/.lmstudio-home-pointer` names the app home
    (internal mechanism, best-effort); fall back to the default location."""
    pointer = Path.home() / ".lmstudio-home-pointer"
    if pointer.is_file():
        home = Path(pointer.read_text().strip())
        if home.is_dir():
            return home / "models"
    return Path.home() / ".lmstudio" / "models"


def find_lms() -> str | None:
    """Locate the `lms` CLI: PATH first, then the app's default link in ~/.lmstudio/bin."""
    found = shutil.which("lms")
    if found:
        return found
    fallback = Path.home() / ".lmstudio" / "bin" / "lms"
    return str(fallback) if fallback.is_file() else None


def list_models(lms: str) -> list[InstalledModel]:
    """Parse `lms ls --json` into InstalledModel records."""
    proc = subprocess.run([lms, "ls", "--json"], capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        detail = _ANSI.sub("", (proc.stderr or proc.stdout).strip())
        raise RagxError(f"`lms ls --json` failed: {detail}")
    try:
        entries = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RagxError(f"`lms ls --json` returned invalid JSON: {exc}") from exc
    return [
        InstalledModel(
            model_key=e.get("modelKey", ""),
            type=e.get("type", ""),
            format=e.get("format", ""),
            path=e.get("path", ""),
            size_bytes=int(e.get("sizeBytes", 0)),
        )
        for e in entries
        if isinstance(e, dict)
    ]


def find_installed(lms: str, repo_fragment: str) -> InstalledModel | None:
    """First installed model whose key or path contains `repo_fragment` (case-insensitive)."""
    needle = repo_fragment.lower()
    for model in list_models(lms):
        if needle in model.model_key.lower() or needle in model.path.lower():
            return model
    return None


def download(lms: str, ref: str) -> None:
    """Run `lms get <ref> --yes`, streaming its progress output to stderr.

    `ref` is a catalog id (`google/embedding-gemma-300m`), a search term, or a full
    Hugging Face URL. Raises RagxError with lms's final message on failure.
    """
    proc = subprocess.Popen(
        [lms, "get", ref, "--yes"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    tail = ""
    stream = proc.stdout
    assert stream is not None
    for chunk in iter(lambda: stream.read(256), ""):
        sys.stderr.write(chunk)
        sys.stderr.flush()
        cleaned = _ANSI.sub("", chunk).strip()
        if cleaned:
            tail = cleaned.splitlines()[-1]
    if proc.wait() != 0:
        raise RagxError(f"`lms get {ref}` failed: {tail or 'see output above'}")
