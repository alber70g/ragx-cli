"""LM Studio autostart. Every `lms` call and every HTTP probe is stubbed — the suite
must never start or load into the developer's real LM Studio."""

from __future__ import annotations

import pytest

from ragx.core import lmstudio
from ragx.core.config import DEFAULTS, Config
from ragx.core.errors import RagxError
from ragx.providers import lmstudio_process
from ragx.providers.lmstudio_process import ensure_for_section, ensure_ready

LOCAL = "http://localhost:1234/v1"


class FakeLms:
    """Stands in for the `lms` CLI: records calls, and starting flips the probe to up."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.running = False
        self.loaded: list[str] = []
        self.installed = ["bge-m3"]  # model keys `lms ls` would report

    def start_server(self, lms: str, port: int) -> None:
        self.calls.append(("start", port))
        self.running = True

    def load_model(self, lms: str, key: str) -> None:
        self.calls.append(("load", key))
        self.loaded.append(key)


@pytest.fixture
def lms(monkeypatch) -> FakeLms:
    fake = FakeLms()
    monkeypatch.setattr(lmstudio_process, "server_up", lambda url: fake.running)
    monkeypatch.setattr(lmstudio, "find_lms", lambda: "/fake/lms")
    monkeypatch.setattr(lmstudio, "start_server", fake.start_server)
    monkeypatch.setattr(lmstudio, "load_model", fake.load_model)
    monkeypatch.setattr(lmstudio, "loaded_identities", lambda _: list(fake.loaded))
    monkeypatch.setattr(lmstudio, "list_models", lambda _: [_installed(k) for k in fake.installed])
    monkeypatch.setattr(
        lmstudio,
        "find_installed",
        lambda _, needle: next((_installed(k) for k in fake.installed if needle in k), None),
    )
    return fake


def _installed(key: str) -> lmstudio.InstalledModel:
    return lmstudio.InstalledModel(
        model_key=key, type="embedding", format="gguf", path=f"{key}/{key}.gguf", size_bytes=1
    )


def test_starts_the_server_and_loads_the_model(lms):
    actions = ensure_ready(LOCAL, "bge-m3")
    assert lms.calls == [("start", 1234), ("load", "bge-m3")]
    assert actions == [
        "started the LM Studio server on port 1234",
        "loaded 'bge-m3' into LM Studio",
    ]


def test_server_up_and_model_loaded_does_nothing(lms):
    lms.running = True
    lms.loaded = ["text-embedding-BGE-M3"]  # identities are matched case-insensitively
    assert ensure_ready(LOCAL, "bge-m3") == []
    assert lms.calls == []


def test_server_up_but_model_not_loaded_only_loads(lms):
    lms.running = True
    lms.loaded = ["some-other-model"]
    assert ensure_ready(LOCAL, "bge-m3") == ["loaded 'bge-m3' into LM Studio"]
    assert lms.calls == [("load", "bge-m3")]


def test_remote_endpoint_is_never_touched(monkeypatch, lms):
    """An OpenAI/cloud base_url is not ours to start — no probe, no `lms` call."""
    monkeypatch.setattr(
        lmstudio_process, "server_up", lambda url: pytest.fail("must not probe a remote endpoint")
    )
    assert ensure_ready("https://api.openai.com/v1", "text-embedding-3-small") == []
    assert lms.calls == []


def test_model_not_downloaded_names_what_is_available(lms):
    """LM Studio's own error is a bare "Model not found" — ragx must say which models
    this machine has instead."""
    lms.running = True
    lms.installed = ["some-other-model"]
    with pytest.raises(RagxError) as exc:
        ensure_ready(LOCAL, "bge-m3")
    assert "no model matching 'bge-m3'" in str(exc.value)
    assert "some-other-model" in str(exc.value)
    assert lms.calls == []


def test_missing_lms_cli_fails_loud_with_the_manual_alternatives(monkeypatch, lms):
    monkeypatch.setattr(lmstudio, "find_lms", lambda: None)
    with pytest.raises(RagxError) as exc:
        ensure_ready(LOCAL, "bge-m3")
    assert "autostart" in str(exc.value) and "start your server manually" in str(exc.value)


def test_foreign_server_on_the_port_is_left_alone(monkeypatch, lms):
    """Something already answers but LM Studio isn't installed — don't try to load."""
    lms.running = True
    monkeypatch.setattr(lmstudio, "find_lms", lambda: None)
    assert ensure_ready(LOCAL, "bge-m3") == []


def _cfg(**overrides) -> Config:
    cfg = Config({k: dict(v) for k, v in DEFAULTS.items()})
    for key, value in overrides.items():
        cfg.set(key.replace("__", "."), value)
    return cfg


def test_section_gate_skips_non_lmstudio_providers(monkeypatch):
    monkeypatch.setattr(
        lmstudio_process, "ensure_ready", lambda *a: pytest.fail("llama-server manages itself")
    )
    assert ensure_for_section(_cfg(embeddings__provider="llama-server"), "embeddings") == []


def test_section_gate_respects_autostart_false(monkeypatch):
    monkeypatch.setattr(lmstudio_process, "ensure_ready", lambda *a: pytest.fail("autostart is off"))
    assert ensure_for_section(_cfg(embeddings__autostart=False), "embeddings") == []


def test_section_gate_uses_the_effective_base_url(monkeypatch):
    """OPENAI_BASE_URL retargets the section (registry honors it) — autostart must see
    the same URL, or it boots LM Studio for a corpus that talks to the cloud."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    seen: list[str] = []
    monkeypatch.setattr(lmstudio_process, "ensure_ready", lambda url, model: seen.append(url) or [])
    ensure_for_section(_cfg(), "embeddings")
    assert seen == ["https://api.openai.com/v1"]
