"""Shared fixtures. HOME is isolated per-test so a developer's real ~/.ragxrc
can never leak into the suite (the rc overrides corpus config by design), and the
LM Studio autostart is disarmed so no test can start or load into the real app."""

from __future__ import annotations

import pytest

from ragx.providers import lmstudio_process


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _no_lmstudio_autostart(monkeypatch):
    """`index`/`query`/`doctor` ask LM Studio to start and load a model; a test run must
    never touch the developer's machine. Tests covering that path call
    `lmstudio_process.ensure_ready` directly with the `lms` calls stubbed out."""
    monkeypatch.setattr(lmstudio_process, "ensure_for_section", lambda cfg, section: [])
