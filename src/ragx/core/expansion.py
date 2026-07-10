"""Query expansion: LLM-generated reformulations + a HyDE passage, one generate() call."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass

from ragx.providers.base import Generator

_SYSTEM = (
    "You rewrite search queries for a code/document retrieval system. "
    "Respond with strict JSON only, no markdown, no commentary."
)


@dataclass
class Expansion:
    variants: list[str]  # reformulations/sub-queries, excludes the original
    hyde: str | None


def _build_prompt(query: str, variants: int, hyde: bool) -> str:
    lines = [
        f'Given the search query: "{query}"',
        f"Produce up to {variants} alternative phrasings or related sub-queries "
        "that would help retrieve relevant passages.",
    ]
    if hyde:
        lines.append(
            "Also produce a short hypothetical passage that would answer the query "
            "(HyDE-style)."
        )
        schema = '{"queries": ["...", "..."], "hyde": "..."}'
    else:
        schema = '{"queries": ["...", "..."]}'
    lines.append(f"Respond with JSON matching this exact schema: {schema}")
    return "\n".join(lines)


def _extract_json(raw: str) -> dict:
    text = raw.strip()
    # strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def expand_query(gen: Generator, query: str, *, variants: int = 3, hyde: bool = True) -> Expansion:
    empty = Expansion([], None)
    try:
        raw = gen.generate(_SYSTEM, _build_prompt(query, variants, hyde))
        data = _extract_json(raw)
        if not isinstance(data, dict) or "queries" not in data:
            raise ValueError(f"unexpected JSON shape: {data!r}")
        queries = data["queries"]
        if not isinstance(queries, list):
            raise ValueError(f"'queries' is not a list: {queries!r}")
    except Exception as exc:  # noqa: BLE001 - defensive per contract, never raise
        print(f"warning: query expansion failed: {exc}", file=sys.stderr)
        return empty

    seen = {query.strip().lower()}
    out: list[str] = []
    for item in queries:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= variants:
            break

    hyde_text = data.get("hyde") if hyde and isinstance(data, dict) else None
    if not isinstance(hyde_text, str) or not hyde_text.strip():
        hyde_text = None
    return Expansion(out, hyde_text)
