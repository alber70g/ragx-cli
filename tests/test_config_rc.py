"""~/.ragxrc: machine-level provider settings that override corpus config (with a warning)."""

from __future__ import annotations

import logging
import tomllib

import pytest
from typer.testing import CliRunner

from ragx.cli.app import app
from ragx.core.config import (
    Config,
    config_path,
    load_rc,
    write_default_config,
    write_rc_value,
    write_rc_values,
)
from ragx.core.doctor import FAIL, _check_index
from ragx.core.errors import ManifestMismatchError, RagxError
from ragx.core.indexer import run_index
from ragx.core.query import QueryOptions, run_query
from tests.test_integration import FakeEmbedder, make_corpus

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


def _write_legacy(tmp_path):
    (tmp_path / ".ragx").mkdir()
    (tmp_path / ".ragx" / "config.toml").write_text("[graph]\nk = 12\n")


def test_legacy_config_auto_migrates_without_confirm(tmp_path, rc, caplog):
    # pre-0.3.0 corpora self-heal: non-interactive loads just move the file, with a notice
    _write_legacy(tmp_path)
    with caplog.at_level(logging.WARNING, logger="ragx.config"):
        cfg = Config.load(tmp_path)
    assert cfg.get("graph.k") == 12
    assert config_path(tmp_path).exists()
    assert not (tmp_path / ".ragx" / "config.toml").exists()
    assert any("migrated" in r.getMessage() for r in caplog.records)


def test_legacy_config_confirm_accepted_migrates(tmp_path, rc):
    _write_legacy(tmp_path)
    prompts: list[str] = []
    cfg = Config.load(tmp_path, confirm=lambda msg: prompts.append(msg) or True)
    assert cfg.get("graph.k") == 12
    assert config_path(tmp_path).exists()
    assert len(prompts) == 1 and "config moved in 0.3.0" in prompts[0]


def test_legacy_config_confirm_declined_fails_loud(tmp_path, rc):
    # declining the prompt keeps the old fail-loud behavior and touches nothing
    _write_legacy(tmp_path)
    with pytest.raises(RagxError, match="not migrated"):
        Config.load(tmp_path, confirm=lambda msg: False)
    assert (tmp_path / ".ragx" / "config.toml").exists()
    assert not config_path(tmp_path).exists()


def test_legacy_and_new_config_both_present_fails_loud(tmp_path, rc):
    # migrating over an existing ragx.toml could discard edits — never pick silently
    _write_legacy(tmp_path)
    config_path(tmp_path).write_text("[graph]\nk = 5\n")
    with pytest.raises(RagxError, match="keep one"):
        Config.load(tmp_path)
    assert (tmp_path / ".ragx" / "config.toml").exists()


def test_root_found_via_config_file_alone(tmp_path):
    # fresh clone: ragx.toml is committed, .ragx/ is gitignored and absent
    from ragx.core.config import find_root
    write_default_config(tmp_path)
    sub = tmp_path / "docs"
    sub.mkdir()
    assert find_root(sub) == tmp_path


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


# --- rc-sourced keys: D5 (ragx.toml leaves them out) and step 4 (errors name the rc) ---


def test_write_rc_values_validates_all_before_writing(rc):
    with pytest.raises(RagxError, match="corpus-level"):
        write_rc_values({"embeddings.model": "m", "graph.k": "12"})
    assert not rc.exists()  # one bad key: nothing written, not a half-seeded rc
    assert write_rc_values({"embeddings.model": "m", "embeddings.batch_size": "8"}) == {
        "embeddings.model": "m", "embeddings.batch_size": 8,
    }
    assert load_rc(rc) == {"embeddings": {"model": "m", "batch_size": 8}}


def test_default_config_omits_rc_keys_so_load_does_not_warn(tmp_path, rc, caplog):
    # the rc wins anyway; writing the defaults for its keys into ragx.toml would only make
    # every later command print an override warning per key
    rc.write_text('[embeddings]\nmodel = "rc-model"\nprovider = "llama-server"\n')
    write_default_config(tmp_path)
    saved = tomllib.loads(config_path(tmp_path).read_text())
    assert "model" not in saved["embeddings"] and "provider" not in saved["embeddings"]
    assert saved["embeddings"]["batch_size"] == 32  # keys the rc does not set stay
    with caplog.at_level(logging.WARNING, logger="ragx.config"):
        cfg = Config.load(tmp_path)
    assert not caplog.records
    assert cfg.rc_keys == {"embeddings.model", "embeddings.provider"}
    assert cfg.get("embeddings.model") == "rc-model"


class RcModelEmbedder(FakeEmbedder):
    model = "rc-model"


def _index_then_rc_switches_model(tmp_path, rc) -> Config:
    """Index built with fake-embedder, then the rc sets another model — the D0 consequence
    of the rc winning over ragx.toml."""
    make_corpus(tmp_path)
    write_default_config(tmp_path)
    run_index(tmp_path, Config.load(tmp_path), FakeEmbedder())
    rc.write_text('[embeddings]\nmodel = "rc-model"\n')
    return Config.load(tmp_path)


def test_index_mismatch_from_rc_names_rc_not_ragx_toml(tmp_path, rc):
    cfg = _index_then_rc_switches_model(tmp_path, rc)
    with pytest.raises(ManifestMismatchError) as exc_info:
        run_index(tmp_path, cfg, RcModelEmbedder())
    message = str(exc_info.value)
    assert "~/.ragxrc" in message
    assert "config set --global embeddings.model fake-embedder" in message
    assert "--full" in message
    assert "ragx.toml to keep" not in message  # pinning in ragx.toml has no effect here


def test_query_and_doctor_mismatch_from_rc_name_rc(tmp_path, rc):
    cfg = _index_then_rc_switches_model(tmp_path, rc)
    with pytest.raises(RagxError, match=r"~/\.ragxrc sets embeddings.model = 'rc-model'"):
        run_query(tmp_path, cfg, RcModelEmbedder(), "dogs", QueryOptions())
    status, detail, fix = _check_index(tmp_path, cfg)
    assert status == FAIL
    assert "~/.ragxrc" in detail
    assert "config set --global" in fix


def test_corpus_model_mismatch_keeps_ragx_toml_advice(tmp_path, rc):
    make_corpus(tmp_path)
    write_default_config(tmp_path)
    cfg = Config.load(tmp_path)
    run_index(tmp_path, cfg, FakeEmbedder())
    with pytest.raises(ManifestMismatchError) as exc_info:
        run_index(tmp_path, cfg, RcModelEmbedder())
    assert "in ragx.toml" in str(exc_info.value)
    assert "~/.ragxrc" not in str(exc_info.value)
