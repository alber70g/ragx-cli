"""Curated model catalog + machine-spec detection for `ragx-cli models`.

Every embedding entry is verified loadable by stock LM Studio (GGUF; `lms get --yes`
hardware-preselects MLX where LM Studio's own catalog offers it). Rerankers run through
sentence-transformers, so they list plain HF repo ids and download via huggingface_hub.
Candidates that failed vetting (jina-embeddings-v5-text-nano: needs a llama.cpp fork;
gte-multilingual-reranker: trust_remote_code) are documented in
research/lm-studio-model-download-api-*.md and DECISIONS.md.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from ragx.core.errors import RagxError

QUALITY_TIERS = ("fast", "balanced", "best", "jina-nano")

LOW_RAM_GB = 8.0


@dataclass(frozen=True)
class EmbeddingChoice:
    tier: str
    label: str
    ref: str  # lm-studio engine: what `lms get` receives (catalog id or full HF URL)
    repo_fragment: str  # substring that identifies it in `lms ls --json` output
    doc_prefix: str
    query_prefix: str
    params: str  # parameter count, for display ("308M")
    languages: str  # language coverage, for display ("100+")
    context: str  # context window, for display ("2K")
    download_mb: int  # approximate download size shown in the plan step
    notes: str  # short qualitative note for the interactive picker table
    # llama-server engine: explicit GGUF repo (the catalog-id ref may resolve to MLX,
    # which llama-server can't load) + the fragment locating the downloaded .gguf file
    gguf_ref: str
    gguf_fragment: str
    # EuroBERT-style models: LM Studio downloads but cannot serve them (verified:
    # it misclassifies them as LLMs) — only the llama-server engine works
    requires_llama_server: bool = False


@dataclass(frozen=True)
class RerankerChoice:
    label: str
    model: str  # HF repo id, loadable by sentence-transformers CrossEncoder
    languages: str
    download_mb: int  # approximate download size shown in the plan step
    notes: str
    # llama-server engine: Q8_0 GGUF downloadable via LM Studio (validated against the
    # safetensors originals — scores within ~0.5 logit, identical ordering)
    gguf_ref: str
    gguf_fragment: str


EMBEDDINGS: dict[str, EmbeddingChoice] = {
    "fast": EmbeddingChoice(
        tier="fast",
        label="EmbeddingGemma 300M",
        ref="google/embedding-gemma-300m",  # curated LM Studio catalog id
        repo_fragment="embeddinggemma",
        doc_prefix="title: none | text: ",
        query_prefix="task: search result | query: ",
        params="308M",
        languages="100+",
        context="2K",
        download_mb=300,
        notes="lightest; small corpora, low RAM",
        gguf_ref="https://huggingface.co/lmstudio-community/embeddinggemma-300m-qat-GGUF",
        gguf_fragment="embeddinggemma-300m-qat",
    ),
    "balanced": EmbeddingChoice(
        tier="balanced",
        label="BGE-M3",
        ref="https://huggingface.co/gaianet/bge-m3-GGUF",
        repo_fragment="bge-m3",
        doc_prefix="",
        query_prefix="",
        params="568M",
        languages="~100",
        context="8K",
        download_mb=600,
        notes="ragx's benchmarked production model",
        gguf_ref="https://huggingface.co/gaianet/bge-m3-GGUF",
        gguf_fragment="bge-m3",
    ),
    "best": EmbeddingChoice(
        tier="best",
        label="Qwen3-Embedding-0.6B",
        ref="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF",
        repo_fragment="qwen3-embedding",
        doc_prefix="",
        query_prefix=(
            "Instruct: Given a web search query, retrieve relevant passages that answer "
            "the query\nQuery: "
        ),
        params="0.6B",
        languages="100+",
        context="32K",
        download_mb=640,
        notes="highest quality; slower indexing",
        gguf_ref="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF",
        gguf_fragment="qwen3-embedding-0.6b",
    ),
    "jina-nano": EmbeddingChoice(
        tier="jina-nano",
        label="Jina embeddings v5 nano",
        ref="https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF@Q8_0",
        repo_fragment="v5-nano-retrieval-q8_0",
        doc_prefix="Document: ",
        query_prefix="Query: ",
        params="239M",
        languages="multi",
        context="8K",
        download_mb=250,
        notes="best sub-500M quality; CC-BY-NC (non-commercial); llama-server only",
        gguf_ref="https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF@Q8_0",
        gguf_fragment="v5-nano-retrieval-q8_0",
        requires_llama_server=True,
    ),
}

RERANKER = RerankerChoice(
    label="BGE reranker v2-m3",
    model="BAAI/bge-reranker-v2-m3",
    languages="multilingual",
    download_mb=600,
    notes="568M, multilingual; keeps the measured Dutch-query recall win",
    gguf_ref="https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF@Q8_0",
    gguf_fragment="bge-reranker-v2-m3-q8_0",
)


@dataclass(frozen=True)
class Specs:
    ram_gb: float
    macos: bool


def detect_specs() -> Specs:
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        ram = 0.0  # unknown (e.g. Windows) — never downgrade on missing data
    return Specs(ram_gb=ram, macos=sys.platform == "darwin")


def ram_limited(specs: Specs) -> bool:
    """True when RAM is known and below the balanced/best comfort threshold."""
    return 0 < specs.ram_gb < LOW_RAM_GB


def recommend(
    quality: str, specs: Specs, *, enforce_ram: bool = True
) -> tuple[EmbeddingChoice, RerankerChoice, list[str]]:
    """Pick an embedding + reranker combo; returns (embedding, reranker, notes).

    enforce_ram=False (interactive callers whose user already confirmed the pick)
    keeps the requested tier on low-RAM machines and warns instead of downgrading.
    """
    if quality not in EMBEDDINGS:
        raise RagxError(f"quality must be one of {'/'.join(QUALITY_TIERS)}, got {quality!r}")
    notes: list[str] = []
    tier = quality
    if ram_limited(specs) and tier in ("balanced", "best"):
        if enforce_ram:
            notes.append(
                f"{specs.ram_gb:.0f} GB RAM detected — dropping to the fast embedding tier"
            )
            tier = "fast"
        else:
            notes.append(
                f"{specs.ram_gb:.0f} GB RAM detected — the {tier} tier may be slow on this machine"
            )
    if tier == "jina-nano":
        notes.append("jina-nano is CC-BY-NC licensed — non-commercial use only")
    return EMBEDDINGS[tier], RERANKER, notes
