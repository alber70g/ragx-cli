"""Managed llama.cpp server process, shared by the llama-server providers.

When nothing answers at the base URL, spawns `llama-server` on a GGUF (rerank or
embedding mode), waits for /health, and terminates the process at interpreter exit.
GGUFs typically come from LM Studio's download dir — no huggingface.co needed."""

from __future__ import annotations

import atexit
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ragx.core.errors import RagxError

HEALTH_TIMEOUT_S = 120
UPGRADE_HINT = "your llama.cpp is too old for this model architecture — upgrade it (e.g. `brew upgrade llama.cpp`)"


class LlamaServerProcess:
    """Lifecycle manager: healthy() probe, ensure() lazy spawn, atexit shutdown.

    `section` is the config section ("rerank"/"embeddings") — used only in messages.
    `mode` is the llama-server flag: "--rerank" or "--embedding".
    """

    def __init__(self, base_url: str, gguf: str, server_bin: str, *, mode: str, section: str) -> None:
        self.root_url = base_url.rstrip("/").removesuffix("/v1")
        self.gguf = Path(gguf).expanduser() if gguf else None
        self._server_bin = server_bin
        self._mode = mode
        self._section = section
        self._proc: subprocess.Popen | None = None

    def healthy(self) -> bool:
        try:
            return httpx.get(f"{self.root_url}/health", timeout=2).status_code == 200
        except httpx.TransportError:
            return False

    def check_spawnable(self) -> str:
        """Return the resolved binary; raise when auto-spawn cannot possibly work."""
        s = self._section
        if self.gguf is None:
            raise RagxError(
                f"no llama-server listening at {self.root_url} and {s}.gguf is not set — "
                f"start one (`llama-server -m <model.gguf> {self._mode}`) or run `ragx-cli models`"
            )
        if not self.gguf.is_file():
            raise RagxError(f"{s}.gguf does not exist: {self.gguf}")
        binary = shutil.which(self._server_bin)
        if binary is None and Path(self._server_bin).expanduser().is_file():
            binary = str(Path(self._server_bin).expanduser())
        if binary is None:
            raise RagxError(
                f"llama-server binary not found ({self._server_bin!r}) — install llama.cpp "
                f"(e.g. `brew install llama.cpp`) or set {s}.server_bin"
            )
        return binary

    def ensure(self) -> None:
        if not self.healthy():
            self._spawn()

    def _spawn(self) -> None:
        binary = self.check_spawnable()
        assert self.gguf is not None  # check_spawnable raised otherwise
        port = urlparse(self.root_url).port
        if port is None:
            raise RagxError(
                f"{self._section}.base_url needs an explicit port for auto-spawn: {self.root_url}"
            )
        log = tempfile.NamedTemporaryFile(
            mode="w+", prefix="ragx-llama-server-", suffix=".log", delete=False
        )
        self._proc = subprocess.Popen(
            [binary, "-m", str(self.gguf), self._mode, "--host", "127.0.0.1", "--port", str(port)],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        atexit.register(self.shutdown)
        deadline = time.monotonic() + HEALTH_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                tail = Path(log.name).read_text(errors="replace")[-2000:]
                hint = f" — {UPGRADE_HINT}" if "unknown model architecture" in tail else ""
                raise RagxError(
                    f"llama-server exited while loading {self.gguf.name}{hint} (log: {log.name})"
                )
            if self.healthy():
                return
            time.sleep(0.3)
        self.shutdown()
        raise RagxError(f"llama-server not healthy after {HEALTH_TIMEOUT_S}s (log: {log.name})")

    def shutdown(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None
