"""`doctor`: run the config health checks and render them. Registered by app.py."""

from __future__ import annotations

from pathlib import Path

import typer

from ragx.cli.output import emit_json, fail, migrate_confirm
from ragx.core.config import Config, require_root
from ragx.core.doctor import FAIL, OK, SKIP, WARN, DoctorReport, run_doctor, to_doctor_json
from ragx.core.errors import RagxError
from ragx.providers.registry import make_embedder, make_generator, make_reranker

MARKS = {OK: "✓", WARN: "!", FAIL: "✗", SKIP: "-"}


def register(app: typer.Typer) -> None:
    app.command()(doctor)


def doctor(
    path: Path | None = typer.Argument(None),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Check that the configured providers start and answer, and that the index matches.

    Starts whatever ragx itself manages (the llama-server engines), exactly as `index`
    and `query` would. Exit 0 = healthy, 1 = a check failed, 2 = doctor could not run.
    """
    try:
        root = require_root(path)
        cfg = Config.load(root, confirm=migrate_confirm())
    except RagxError as exc:
        fail(str(exc))
    report = run_doctor(
        root,
        cfg,
        embedder_factory=make_embedder,
        generator_factory=make_generator,
        reranker_factory=lambda c: make_reranker(c, strict=True),
    )
    if json_out:
        emit_json(to_doctor_json(report))
    else:
        _render(report)
    if not report.ok:
        raise typer.Exit(code=1)


def _render(report: DoctorReport) -> None:
    for check in report.checks:
        secs = f"  {check.elapsed_ms / 1000:.1f}s" if check.elapsed_ms >= 100 else ""
        typer.echo(f"{MARKS[check.status]} {check.name:<11} {check.detail}{secs}")
        if check.hint:
            typer.echo(f"  → {check.hint}")
    failed = sum(1 for c in report.checks if c.status == FAIL)
    warned = sum(1 for c in report.checks if c.status == WARN)
    if failed or warned:
        typer.echo(f"\n{failed} failed, {warned} warning(s)")
    else:
        typer.echo("\nall checks passed")
