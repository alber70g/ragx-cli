"""Interactive presentation for `ragx-cli models`: machine header, tier table,
numbered pickers, and the pre-confirm plan. Rendering + prompting only — no
downloads, no config writes. Everything goes to stderr (stdout stays pure for
--json)."""

from __future__ import annotations

import sys

import typer

from ragx.core.catalog import EMBEDDINGS, EmbeddingChoice, RerankerChoice, Specs, ram_limited

EMBED_ENGINES = (
    ("llama-server", "ragx auto-spawns llama.cpp; LM Studio only downloads"),
    ("lm-studio", "LM Studio app must be running to serve embeddings"),
)
RERANK_ENGINES = (
    ("llama-server", "GGUF via LM Studio — no huggingface.co access needed"),
    ("sentence-transformers", "downloads from Hugging Face; needs the ragx-cli[rerank] extra"),
)


def echo(msg: str = "") -> None:
    typer.echo(msg, err=True)


def header(specs: Specs, llama_found: bool, lms_found: bool | None) -> None:
    """One line of detected facts; lms_found=None means it wasn't looked up (dry-run)."""
    ram = f"{specs.ram_gb:.0f} GB RAM" if specs.ram_gb else "RAM unknown"
    parts = [
        ram,
        "macOS" if specs.macos else sys.platform,
        f"llama-server: {'on PATH' if llama_found else 'not found'}",
    ]
    if lms_found is not None:
        parts.append(f"LM Studio (lms): {'found' if lms_found else 'NOT FOUND'}")
    echo("machine: " + " · ".join(parts))


def _rule(title: str) -> None:
    echo()
    echo(f"── {title} " + "─" * max(4, 58 - len(title)))


def pick_numbered(rows: list[str], default: int) -> int:
    """Numbered menu (`»` marks the default); loops until a valid 1-based pick."""
    for i, row in enumerate(rows, 1):
        echo(f" {'»' if i == default else ' '} {i}  {row}")
    while True:
        raw = typer.prompt(f"  select 1-{len(rows)}", default=str(default), err=True).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(rows):
            return int(raw)
        echo(f"  enter a number between 1 and {len(rows)}")


def pick_tier(specs: Specs) -> str:
    limited = ram_limited(specs)
    tiers = list(EMBEDDINGS)
    _rule("embedding model")
    echo("   determines retrieval quality — changing it later requires a full re-index")
    echo()
    echo(f"   #  {'tier':<10} {'model':<24} {'size':<5} {'langs':<6} {'ctx':<4} notes")
    rows = []
    for name in tiers:
        e = EMBEDDINGS[name]
        note = e.notes + ("  ⚠ needs ≥8 GB RAM" if limited and name in ("balanced", "best") else "")
        rows.append(f"{e.tier:<10} {e.label:<24} {e.params:<5} {e.languages:<6} {e.context:<4} {note}")
    default = tiers.index("fast" if limited else "balanced") + 1
    while True:
        tier = tiers[pick_numbered(rows, default) - 1]
        if limited and tier in ("balanced", "best") and not typer.confirm(
            f"  only {specs.ram_gb:.0f} GB RAM detected — {tier} may be slow here; use it anyway?",
            default=False,
            err=True,
        ):
            continue
        return tier


def pick_engine(label: str, options: tuple[tuple[str, str], ...], default: str, llama_found: bool) -> str:
    _rule(label)
    names = [name for name, _ in options]
    rows = [
        f"{name:<22} {desc}" + ("  (detected on PATH)" if name == "llama-server" and llama_found else "")
        for name, desc in options
    ]
    return names[pick_numbered(rows, names.index(default) + 1) - 1]


def show_plan(
    *,
    embedding: EmbeddingChoice,
    reranker: RerankerChoice,
    emb_engine: str,
    rerank_engine: str,
    emb_installed: bool,
    rr_installed: bool,
    reindex_line: str,
) -> None:
    _rule("plan")
    emb_status = (
        "already downloaded ✓" if emb_installed
        else f"will download via LM Studio (~{embedding.download_mb} MB)"
    )
    if rerank_engine == "llama-server":
        rr_status = (
            "already downloaded ✓" if rr_installed
            else f"will download via LM Studio (~{reranker.download_mb} MB)"
        )
    else:
        rr_status = f"downloads from Hugging Face (~{reranker.download_mb} MB)"
    echo(f"   embedding  {embedding.label:<26} via {emb_engine:<22} {emb_status}")
    echo(f"   reranker   {reranker.label:<26} via {rerank_engine:<22} {rr_status}")
    echo("   config     ragx.toml → embeddings.*, rerank.*")
    echo(f"   re-index   {reindex_line}")
    echo()
