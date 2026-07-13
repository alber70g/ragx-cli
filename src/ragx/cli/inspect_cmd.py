"""ragx-cli inspect: read-only introspection into the store (chunks, files, neighbors).

No embeddings or vector index needed — everything here reads Store directly.
"""

from __future__ import annotations

import typer

from ragx.cli.output import emit_json, fail
from ragx.core.config import db_path, require_root
from ragx.core.errors import RagxError
from ragx.core.store import Store

inspect_app = typer.Typer(add_completion=False, no_args_is_help=True)


def register(app: typer.Typer) -> None:
    app.add_typer(inspect_app, name="inspect")


def _open_store() -> Store:
    try:
        root = require_root()
    except RagxError as exc:
        fail(str(exc))
    return Store(db_path(root))


def _neighbor_info(store: Store, chunk_id: int) -> list[dict]:
    raw = store.neighbors(chunk_id)
    others = {c.id: c for c in store.get_chunks([other_id for other_id, _ in raw])}
    return [
        {
            "id": other_id,
            "weight": round(weight, 6),
            "file": others[other_id].file_path if other_id in others else None,
            "line_start": others[other_id].line_start if other_id in others else None,
            "line_end": others[other_id].line_end if other_id in others else None,
        }
        for other_id, weight in raw
    ]


@inspect_app.command("chunk")
def inspect_chunk(chunk_id: int, json_out: bool = typer.Option(False, "--json")) -> None:
    """Show a chunk's text, location, and edges."""
    with _open_store() as store:
        chunks = store.get_chunks([chunk_id])
        if not chunks:
            fail(f"unknown chunk id: {chunk_id}")
        chunk = chunks[0]
        edges = _neighbor_info(store, chunk_id)
    doc = {
        "schema": "ragx.inspect.chunk.v1",
        "id": chunk.id,
        "file": chunk.file_path,
        "byte_start": chunk.byte_start,
        "byte_end": chunk.byte_end,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "text": chunk.text,
        "edges": edges,
    }
    if json_out:
        emit_json(doc)
    else:
        typer.echo(f"chunk {chunk.id}  {chunk.file_path}:{chunk.line_start}-{chunk.line_end}")
        typer.echo(chunk.text)
        typer.echo(f"edges ({len(edges)}):")
        for e in edges:
            typer.echo(f"  {e['id']}  {e['weight']:.4f}  {e['file']}")
    if not edges:
        raise typer.Exit(code=1)


@inspect_app.command("file")
def inspect_file(path: str, json_out: bool = typer.Option(False, "--json")) -> None:
    """Show a file's record and its chunks."""
    with _open_store() as store:
        hashes = store.get_file_hashes()
        if path not in hashes:
            fail(f"unknown file: {path}")
        ids = store.chunk_ids_for_file(path)
        chunks = store.get_chunks(ids)
    chunk_docs = [
        {"id": c.id, "line_start": c.line_start, "line_end": c.line_end, "preview": c.text[:80]}
        for c in chunks
    ]
    doc = {
        "schema": "ragx.inspect.file.v1",
        "path": path,
        "content_hash": hashes[path],
        "chunk_count": len(chunk_docs),
        "chunks": chunk_docs,
    }
    if json_out:
        emit_json(doc)
    else:
        typer.echo(f"file {path}  ({doc['chunk_count']} chunks)  hash={doc['content_hash']}")
        for c in chunk_docs:
            typer.echo(f"  {c['id']}  {c['line_start']}-{c['line_end']}  {c['preview']}")
    if not chunk_docs:
        raise typer.Exit(code=1)


@inspect_app.command("neighbors")
def inspect_neighbors(chunk_id: int, json_out: bool = typer.Option(False, "--json")) -> None:
    """Show a chunk's neighbors enriched with file/line info."""
    with _open_store() as store:
        if not store.get_chunks([chunk_id]):
            fail(f"unknown chunk id: {chunk_id}")
        neighbors = _neighbor_info(store, chunk_id)
    doc = {"schema": "ragx.inspect.neighbors.v1", "id": chunk_id, "neighbors": neighbors}
    if json_out:
        emit_json(doc)
    else:
        typer.echo(f"neighbors of {chunk_id}:")
        for n in neighbors:
            typer.echo(f"  {n['id']}  {n['weight']:.4f}  {n['file']}:{n['line_start']}-{n['line_end']}")
    if not neighbors:
        raise typer.Exit(code=1)


@inspect_app.command("communities")
def inspect_communities(json_out: bool = typer.Option(False, "--json")) -> None:
    """List Leiden communities, largest first, with a file preview per community."""
    with _open_store() as store:
        sizes = store.community_sizes()
        communities = []
        for cid, size in sizes:
            chunks = store.get_chunks(store.community_members(cid))
            files = list(dict.fromkeys(c.file_path for c in chunks))[:3]
            communities.append({"community_id": cid, "size": size, "files": files})
    doc = {"schema": "ragx.inspect.communities.v1", "count": len(communities), "communities": communities}
    if json_out:
        emit_json(doc)
    else:
        for c in communities:
            typer.echo(f"community {c['community_id']}  size {c['size']}  {', '.join(c['files'])}")
    if not communities:
        raise typer.Exit(code=1)


@inspect_app.command("community")
def inspect_community(community_id: int, json_out: bool = typer.Option(False, "--json")) -> None:
    """Show a single community's members."""
    with _open_store() as store:
        ids = store.community_members(community_id)
        if not ids:
            fail(f"unknown community id: {community_id}")
        chunks = store.get_chunks(ids)
    members = [
        {"id": c.id, "file": c.file_path, "line_start": c.line_start, "line_end": c.line_end}
        for c in chunks
    ]
    doc = {
        "schema": "ragx.inspect.community.v1",
        "community_id": community_id,
        "size": len(members),
        "members": members,
    }
    if json_out:
        emit_json(doc)
    else:
        typer.echo(f"community {community_id}  ({doc['size']} members)")
        for m in members:
            typer.echo(f"  {m['id']}  {m['file']}:{m['line_start']}-{m['line_end']}")
