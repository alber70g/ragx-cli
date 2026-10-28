"""`ragx-cli init` reusing this machine's setup: an existing ~/.ragxrc is read and never
modified (D6); without one, a detected catalog setup is preselected (D3) and provider
answers seed the rc (D4), which ragx.toml then leaves out (D5)."""

from __future__ import annotations

import logging
import tomllib

import httpx
import pytest
import respx
from typer.testing import CliRunner

from ragx.cli.app import app
from ragx.core import machine_probe
from ragx.core.catalog import RERANKER
from ragx.core.config import CONFIG_FILE, Config, load_rc, rc_path
from ragx.core.lmstudio import InstalledModel
from ragx.core.machine_probe import MachineState

runner = CliRunner()

BGE_M3 = InstalledModel("text-embedding-bge-m3", "embedding", "gguf", "gpustack/bge-m3-GGUF/bge-m3-Q8_0.gguf", 1)
RERANK_GGUF = "/m/bge-reranker-v2-m3-Q8_0.gguf"
RC_TEXT = '[embeddings]\nprovider = "llama-server"\nmodel = "rc-model"\n'


@pytest.fixture
def machine(monkeypatch):
    """BGE-M3 loaded in LM Studio + a llama-server on 9814 serving the catalog reranker."""
    state = MachineState(lms_loaded=(BGE_M3,), rerank_server_gguf=RERANK_GGUF)
    monkeypatch.setattr(machine_probe, "probe_machine", lambda: state)


@pytest.fixture
def no_probe(monkeypatch):
    monkeypatch.setattr(machine_probe, "probe_machine", lambda: pytest.fail("rc exists: must not probe"))


@pytest.fixture
def no_models_flow(monkeypatch):
    monkeypatch.setattr("ragx.cli.models_cmd.run_flow", lambda root, **kw: pytest.fail("no download offer"))


@pytest.fixture
def servers_down():
    with respx.mock:
        for url in ("http://localhost:1234/v1/models", "http://localhost:11434/v1/models"):
            respx.get(url).mock(side_effect=httpx.ConnectError("down"))
        yield


def _toml(tmp_path) -> dict:
    return tomllib.loads((tmp_path / CONFIG_FILE).read_text())


def _loads_without_override_warning(tmp_path, caplog) -> Config:
    with caplog.at_level(logging.WARNING, logger="ragx.config"):
        cfg = Config.load(tmp_path)
    assert not [r for r in caplog.records if "overrides" in r.getMessage()]
    return cfg


# --- rc exists: read, never probed past, never modified -------------------------


def test_rc_exists_yes_writes_corpus_config_without_rc_keys(tmp_path, no_probe, caplog):
    rc_path().write_text(RC_TEXT)
    result = runner.invoke(app, ["init", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "provider settings from ~/.ragxrc: embeddings.provider=llama-server" in result.output
    assert rc_path().read_text() == RC_TEXT
    data = _toml(tmp_path)
    assert "model" not in data["embeddings"] and "provider" not in data["embeddings"]
    assert data["embeddings"]["batch_size"] == 32
    assert _loads_without_override_warning(tmp_path, caplog).get("embeddings.model") == "rc-model"


def test_rc_exists_tty_asks_only_keys_the_rc_lacks(tmp_path, no_probe, servers_down, caplog):
    rc_path().write_text(RC_TEXT)
    # embeddings: base URL + API key env (rc sets provider + model); expansion off;
    # corpus defaults; models offer declined
    answers = "\n".join(["", "", "n", "", "", "", "n"]) + "\n"
    result = runner.invoke(app, ["init", str(tmp_path), "--interactive"], input=answers)
    assert result.exit_code == 0, result.output
    assert "provider (openai/ollama)" not in result.output  # rc sets the provider
    assert "number or name" not in result.output and "  model" not in result.output
    assert rc_path().read_text() == RC_TEXT
    data = _toml(tmp_path)
    assert data["expansion"]["enabled"] is False  # answered keys the rc lacks go to ragx.toml
    assert "model" not in data["embeddings"]
    _loads_without_override_warning(tmp_path, caplog)


# --- rc absent: probe, preselect, seed the rc -----------------------------------


def test_tty_preselects_reuse_and_seeds_rc(tmp_path, machine, servers_down, no_models_flow, caplog):
    # Enter twice accepts both detected setups; expansion off; corpus defaults. Both reused:
    # no "download models now?" offer (no_models_flow would fail the test).
    answers = "\n".join(["", "", "n", "", "", ""]) + "\n"
    result = runner.invoke(app, ["init", str(tmp_path), "--interactive"], input=answers)
    assert result.exit_code == 0, result.output
    assert "embeddings: use BGE-M3 (loaded in LM Studio)? [Y/n]" in result.output
    assert f"reranker: use {RERANKER.label} (served by a running llama-server)? [Y/n]" in result.output
    assert "recommend + download models" not in result.output
    assert "applies to every ragx corpus on this machine" in result.output
    rc = load_rc()
    assert rc["embeddings"]["model"] == "text-embedding-bge-m3"
    assert rc["embeddings"]["provider"] == "openai"
    assert rc["rerank"] == {"provider": "llama-server", "model": RERANKER.model, "gguf": RERANK_GGUF}
    assert rc["expansion"] == {"enabled": False}
    data = _toml(tmp_path)
    assert "model" not in data["embeddings"] and "gguf" not in data["rerank"]
    cfg = _loads_without_override_warning(tmp_path, caplog)
    assert cfg.get("rerank.gguf") == RERANK_GGUF


def test_tty_declined_reuse_falls_back_to_prompts_and_still_seeds_rc(tmp_path, machine, servers_down):
    answers = "\n".join([
        "n",  # embeddings: decline BGE-M3
        "", "", "my-embedder", "",  # provider, base URL, model, API key env
        "n",  # reranker: decline
        "n", "", "", "",  # expansion off, corpus defaults
        "",  # download offer -> default no (nothing was reused)
    ]) + "\n"
    result = runner.invoke(app, ["init", str(tmp_path), "--interactive"], input=answers)
    assert result.exit_code == 0, result.output
    assert "recommend + download models" in result.output
    rc = load_rc()
    assert rc["embeddings"]["model"] == "my-embedder"  # D4: manual picks go to the rc too
    assert "rerank" not in rc


def test_yes_with_reuse_seeds_rc_without_prompts(tmp_path, machine, caplog):
    result = runner.invoke(app, ["init", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert "embeddings: reusing BGE-M3 (loaded in LM Studio)" in result.output
    assert "rerank: reusing" in result.output
    rc = load_rc()
    assert rc["embeddings"]["model"] == "text-embedding-bge-m3"
    assert rc["rerank"]["gguf"] == RERANK_GGUF
    assert "expansion" not in rc  # not probed (D7), not asked
    assert "model" not in _toml(tmp_path)["embeddings"]
    _loads_without_override_warning(tmp_path, caplog)


def test_yes_without_reuse_writes_defaults_and_no_rc(tmp_path):
    result = runner.invoke(app, ["init", str(tmp_path), "--yes"])
    assert result.exit_code == 0, result.output
    assert not rc_path().exists()
    assert _toml(tmp_path)["embeddings"]["model"] == "text-embedding-nomic-embed-text-v1.5@q4_k_m"
