"""Shared fixtures. HOME is isolated per-test so a developer's real ~/.ragxrc
can never leak into the suite (the rc overrides corpus config by design)."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    return home
