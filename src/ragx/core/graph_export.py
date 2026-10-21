"""Builds the ragx.graph.v1 payload (nodes/edges/communities) for `ragx-cli web`.

Renders whatever edges the index has: sub-chunk similarity when it was built with
graph.edge_source="subchunk", whole-chunk cosine when edge_source="chunk". An index with
zero edges (nodes only) is valid — the viewer just shows an unconnected cloud."""

from __future__ import annotations

from pathlib import Path

from ragx.core.store import Store

LABEL_MAX_CHARS = 80


def build_graph(root: Path, store: Store) -> dict:
    edge_source = store.get_meta("edge_source")
    sub_size = store.get_meta("subchunk_size_tokens")

    chunks = store.get_chunks(store.all_chunk_ids())
    edges = store.all_edges()
    communities = store.all_communities()

    degree: dict[int, int] = {}
    for src, dst, _ in edges:
        degree[src] = degree.get(src, 0) + 1
        degree[dst] = degree.get(dst, 0) + 1

    nodes = [
        {
            "id": c.id,
            "file": c.file_path,
            "line_start": c.line_start,
            "line_end": c.line_end,
            "label": _label(c.text),
            "community": communities.get(c.id),
            "degree": degree.get(c.id, 0),
        }
        for c in chunks
    ]
    edge_list = [[src, dst, round(weight, 4)] for src, dst, weight in edges]

    return {
        "schema": "ragx.graph.v1",
        "meta": {
            "root": root.name,
            "edge_source": edge_source,
            "subchunk_size_tokens": int(sub_size) if edge_source == "subchunk" and sub_size else None,
            "embedding_model": store.get_meta("embedding_model"),
            "nodes": len(nodes),
            "edges": len(edge_list),
            "communities": store.community_count(),
        },
        "nodes": nodes,
        "edges": edge_list,
    }


def _label(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:LABEL_MAX_CHARS]
    return ""
