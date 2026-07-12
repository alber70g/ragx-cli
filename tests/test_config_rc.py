"""~/.ragxrc: machine-level provider settings that override corpus config (with a warning)."""

from __future__ import annotations

import logging
import tomllib

import pytest
from typer.testing import CliRunner

from ragx.cli.app import app
from ragx.core.config import Config, config_path, load_rc, write_default_config, write_rc_value
from ragx.core.errors import RagxError

runner = CliRunner()


@pytest.fixture
def rc(_isolate_home):
    return _isolate_home / ".ragxrc"


def test_no_rc_is_a_noop(tmp_path, rc):
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    assert cfg.get("embeddings.model") == "text-embedding-nomic-embed-text-v1.5@q4_k_m"


def test_rc_overrides_corpus_and_warns(tmp_path, rc, caplog):
    write_default_config(tmp_path)
    rc.write_text('[embeddings]\nmodel = "rc-model"\n')
    with caplog.at_level(logging.WARNING, logger="ragx.config"):
        cfg = Config.load(tmp_path)
    assert cfg.get("embeddings.model") == "rc-model"
    assert any("~/.ragxrc overrides embeddings.model" in r.getMessage() for r in caplog.records)


def test_rc_matching_corpus_value_does_not_warn(tmp_path, rc, caplog):
    write_default_config(tmp_path)
    rc.write_text('[rerank]\nmodel = "BAAI/bge-reranker-v2-m3"\n')  # same as default/corpus
    with caplog.at_level(logging.WARNING, logger="ragx.config"):
        cfg = Config.load(tmp_path)
    assert cfg.get("rerank.model") == "BAAI/bge-reranker-v2-m3"
    assert not caplog.records


def test_rc_applies_without_corpus_file(tmp_path, rc, caplog):
    # .ragx/ exists but no config.toml was written: rc applies silently
    (tmp_path / ".ragx").mkdir()
    rc.write_text('[expansion]\nmodel = "rc-llm"\nenabled = false\n')
    with caplog.at_level(logging.WARNING, logger="ragx.config"):
        cfg = Config.load(tmp_path)
    assert cfg.get("expansion.model") == "rc-llm"
    assert cfg.get("expansion.enabled") is False
    assert not caplog.records


def test_rc_rejects_non_provider_section(tmp_path, rc):
    write_default_config(tmp_path)
    rc.write_text("[graph]\nk = 12\n")
    with pytest.raises(RagxError, match=r"section \[graph\] not allowed"):
        Config.load(tmp_path)


def test_rc_rejects_unknown_key(tmp_path, rc):
    write_default_config(tmp_path)
    rc.write_text('[embeddings]\nbogus = "x"\n')
    with pytest.raises(RagxError, match="unknown key embeddings.bogus"):
        Config.load(tmp_path)


def test_rc_malformed_toml(tmp_path, rc):
    write_default_config(tmp_path)
    rc.write_text("not [valid toml")
    with pytest.raises(RagxError, match="malformed"):
        Config.load(tmp_path)


def test_write_rc_value_creates_coerces_and_rejects(rc):
    coerced = write_rc_value("embeddings.batch_size", "64")
    assert coerced == 64
    assert load_rc(rc) == {"embeddings": {"batch_size": 64}}
    with pytest.raises(RagxError, match="corpus-level"):
        write_rc_value("graph.k", "12")
    with pytest.raises(RagxError, match="unknown config key"):
        write_rc_value("embeddings.bogus", "x")


def test_config_set_does_not_bake_rc_into_corpus(tmp_path, rc):
    write_default_config(tmp_path)
    rc.write_text('[embeddings]\nmodel = "rc-model"\n')
    cfg = Config.load(tmp_path)
    cfg.set("graph.k", "12")
    cfg.save(tmp_path)
    saved = tomllib.loads(config_path(tmp_path).read_text())
    assert saved["graph"]["k"] == 12
    assert saved["embeddings"]["model"] == "text-embedding-nomic-embed-text-v1.5@q4_k_m"


def test_cli_set_global_and_effective_get(tmp_path, rc, monkeypatch):
    runner.invoke(app, ["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["config", "set", "--global", "embeddings.model", "rc-model"])
    assert result.exit_code == 0
    assert "embeddings.model = rc-model" in result.stdout
    assert load_rc(rc) == {"embeddings": {"model": "rc-model"}}

    result = runner.invoke(app, ["config", "get", "embeddings.model"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "rc-model"


def test_cli_set_global_rejects_corpus_key(tmp_path, monkeypatch):
    runner.invoke(app, ["init", str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["config", "set", "--global", "graph.k", "12"])
    assert result.exit_code == 2
