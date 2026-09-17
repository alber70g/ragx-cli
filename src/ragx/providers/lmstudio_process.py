"""Managed LM Studio backend: start the local server and load the configured model when
a provider section points at LM Studio.

Parallel to llama_process.py with one deliberate difference — LM Studio is a long-lived
app we hand off to (`lms server start`), not a child process we own, so ragx starts it
but never stops it. Only local URLs are ever touched; a remote OpenAI-compatible
endpoint is not ours to manage."""

from __future__ import annotations

import logging
import time
from urllib.parse import urlparse

import httpx

from ragx.core import lmstudio
from ragx.core.config import Config, effective_base_url
from ragx.core.errors import RagxError

LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
READY_TIMEOUT_S = 60

log = logging.getLogger("ragx.lmstudio")


def ensure_for_section(cfg: Config, section: str) -> list[str]:
    """Make `<section>.model` answerable at that section's base_url. Returns the actions
    taken, empty when nothing was needed (or when the section isn't LM Studio's to serve)."""
    if cfg.get(f"{section}.provider") != "openai" or not cfg.get(f"{section}.autostart"):
        return []
    return ensure_ready(effective_base_url(cfg, section), cfg.get(f"{section}.model"))


def ensure_ready(base_url: str, model: str) -> list[str]:
    """Start the server if nothing answers at `base_url`, then load `model` if it is not
    already in memory. Raises RagxError when either step cannot be done.

    Actions are logged as they happen — a later failure must not swallow the fact that
    the server was already started."""
    if not _is_local(base_url):
        return []
    actions: list[str] = []

    def did(action: str) -> None:
        log.info("%s", action)
        actions.append(action)

    if not server_up(base_url):
        port = _port(base_url)
        lmstudio.start_server(_require_lms(base_url), port)
        _wait_ready(base_url)
        did(f"started the LM Studio server on port {port}")
    lms = lmstudio.find_lms()
    if lms is None:
        return actions  # something else is serving this port — not ours to load into
    if _is_loaded(lms, model):
        return actions
    key = _resolve_key(lms, model)
    log.info("loading %r into LM Studio (this can take a while)", key)
    lmstudio.load_model(lms, key)
    did(f"loaded {key!r} into LM Studio")
    return actions


def _resolve_key(lms: str, model: str) -> str:
    """The `lms` model key to load. Checking up front turns LM Studio's generic
    "Model not found" into a message that names the models this machine actually has."""
    installed = lmstudio.find_installed(lms, model)
    if installed is not None:
        return installed.model_key
    available = [m.model_key for m in lmstudio.list_models(lms)]
    listed = ", ".join(available[:8]) or "none"
    raise RagxError(
        f"LM Studio has no model matching {model!r} — download it (`ragx-cli models`), "
        f"or point the config at one it has: {listed}"
    )


def server_up(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url.rstrip('/')}/models", timeout=2.0).status_code == 200
    except Exception:
        return False


def _is_local(base_url: str) -> bool:
    return (urlparse(base_url).hostname or "") in LOCAL_HOSTS


def _port(base_url: str) -> int:
    port = urlparse(base_url).port
    if port is None:
        raise RagxError(f"base_url needs an explicit port to start LM Studio: {base_url}")
    return port


def _require_lms(base_url: str) -> str:
    lms = lmstudio.find_lms()
    if lms is None:
        raise RagxError(
            f"nothing is listening at {base_url} and LM Studio's `lms` CLI was not found — "
            "install LM Studio (https://lmstudio.ai) and launch it once, start your server "
            "manually, or disable autostart (`ragx-cli config set embeddings.autostart false`)"
        )
    return lms


def _wait_ready(base_url: str) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if server_up(base_url):
            return
        time.sleep(0.3)
    raise RagxError(f"LM Studio's server did not answer at {base_url} within {READY_TIMEOUT_S}s")


def _is_loaded(lms: str, model: str) -> bool:
    needle = model.lower()
    return any(needle in identity.lower() for identity in lmstudio.loaded_identities(lms))
