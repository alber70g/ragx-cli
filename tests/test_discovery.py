from pathlib import Path

import xxhash

from ragx.core.discovery import discover_files, hash_file


def _write(root: Path, rel: str, content: str = "hello world\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_discover_sorted_and_basic_include(tmp_path: Path):
    _write(tmp_path, "b.txt")
    _write(tmp_path, "a.txt")
    _write(tmp_path, "sub/c.txt")

    files = discover_files(tmp_path, include=["**/*"], exclude=[])

    assert files == ["a.txt", "b.txt", "sub/c.txt"]


def test_discover_exclude_glob(tmp_path: Path):
    _write(tmp_path, "keep.py")
    _write(tmp_path, "drop.log")

    files = discover_files(tmp_path, include=["**/*"], exclude=["*.log"])

    assert files == ["keep.py"]


def test_discover_include_restricts(tmp_path: Path):
    _write(tmp_path, "keep.py")
    _write(tmp_path, "drop.txt")

    files = discover_files(tmp_path, include=["**/*.py"], exclude=[])

    assert files == ["keep.py"]


def test_discover_skips_ragx_and_git_dirs(tmp_path: Path):
    _write(tmp_path, ".ragx/index.db")
    _write(tmp_path, ".git/HEAD")
    _write(tmp_path, "real.txt")

    files = discover_files(tmp_path, include=["**/*"], exclude=[])

    assert files == ["real.txt"]


def test_discover_skips_root_ragx_toml(tmp_path: Path):
    _write(tmp_path, "ragx.toml", "[graph]\nk = 8\n")
    _write(tmp_path, "sub/ragx.toml", "not the corpus config\n")
    _write(tmp_path, "real.txt")

    files = discover_files(tmp_path, include=["**/*"], exclude=[])

    assert files == ["real.txt", "sub/ragx.toml"]


def test_discover_skips_hidden_dirs(tmp_path: Path):
    _write(tmp_path, ".cache/thing.txt")
    _write(tmp_path, "visible.txt")

    files = discover_files(tmp_path, include=["**/*"], exclude=[])

    assert files == ["visible.txt"]


def test_discover_skips_binary_files(tmp_path: Path):
    binary_path = tmp_path / "data.bin"
    binary_path.write_bytes(b"\x00\x01binary\x00stuff")
    _write(tmp_path, "text.txt")

    files = discover_files(tmp_path, include=["**/*"], exclude=[])

    assert files == ["text.txt"]


def test_discover_skips_oversized_files(tmp_path: Path):
    big = tmp_path / "big.txt"
    big.write_bytes(b"a" * (2 * 1024 * 1024 + 1))
    _write(tmp_path, "small.txt")

    files = discover_files(tmp_path, include=["**/*"], exclude=[])

    assert files == ["small.txt"]


def test_discover_respects_root_gitignore(tmp_path: Path):
    _write(tmp_path, ".gitignore", "ignored.txt\n")
    _write(tmp_path, "ignored.txt")
    _write(tmp_path, "kept.txt")

    files = discover_files(tmp_path, include=["**/*"], exclude=[], respect_gitignore=True)

    # hidden dotfiles themselves aren't auto-skipped (only hidden dirs are); the
    # .gitignore file's own pattern only matches "ignored.txt".
    assert files == [".gitignore", "kept.txt"]


def test_discover_respects_nested_gitignore_scoped_to_its_subtree(tmp_path: Path):
    # A subproject's .gitignore (e.g. a vendored repo's generated docs) must keep its
    # junk out of the index, but must not leak onto same-named files elsewhere.
    _write(tmp_path, "sub/.gitignore", "generated.md\n/anchored.md\n")
    _write(tmp_path, "sub/generated.md")
    _write(tmp_path, "sub/deep/generated.md")
    _write(tmp_path, "sub/anchored.md")
    _write(tmp_path, "sub/deep/anchored.md")
    _write(tmp_path, "generated.md")
    _write(tmp_path, "anchored.md")

    files = discover_files(tmp_path, include=["**/*.md"], exclude=[], respect_gitignore=True)

    # "/anchored.md" anchors to sub/, so only sub/anchored.md is dropped.
    assert files == ["anchored.md", "generated.md", "sub/deep/anchored.md"]


def test_discover_deeper_gitignore_overrides_root(tmp_path: Path):
    # Git precedence: the deeper file wins, so a subdir can re-include what root ignores.
    _write(tmp_path, ".gitignore", "*.log.md\n")
    _write(tmp_path, "notes/.gitignore", "!keep.log.md\n")
    _write(tmp_path, "notes/keep.log.md")
    _write(tmp_path, "notes/drop.log.md")
    _write(tmp_path, "keep.log.md")

    files = discover_files(tmp_path, include=["**/*.md"], exclude=[], respect_gitignore=True)

    assert files == ["notes/keep.log.md"]


def test_discover_prunes_gitignored_directories(tmp_path: Path):
    # Git can't re-include a file whose parent dir is ignored; ragx must not index it
    # either, or the index would contain files git itself never tracks.
    _write(tmp_path, ".gitignore", "out/\n")
    _write(tmp_path, "pkg/out/.gitignore", "!kept.md\n")
    _write(tmp_path, "pkg/out/kept.md")
    _write(tmp_path, "pkg/src.md")

    files = discover_files(tmp_path, include=["**/*.md"], exclude=[], respect_gitignore=True)

    assert files == ["pkg/src.md"]


def test_discover_ignores_nested_gitignore_when_disabled(tmp_path: Path):
    _write(tmp_path, "sub/.gitignore", "hidden.md\n")
    _write(tmp_path, "sub/hidden.md")

    files = discover_files(tmp_path, include=["**/*.md"], exclude=[], respect_gitignore=False)

    assert files == ["sub/hidden.md"]


def test_discover_ignores_gitignore_when_disabled(tmp_path: Path):
    _write(tmp_path, ".gitignore", "notignored.txt\n")
    _write(tmp_path, "notignored.txt")

    files = discover_files(tmp_path, include=["**/*"], exclude=[], respect_gitignore=False)

    assert "notignored.txt" in files


def test_hash_file_matches_xxhash_and_is_deterministic(tmp_path: Path):
    path = _write(tmp_path, "file.txt", "some content")

    digest = hash_file(path)

    assert digest == xxhash.xxh64(b"some content").hexdigest()
    assert digest == hash_file(path)


def test_hash_file_differs_on_content_change(tmp_path: Path):
    path = _write(tmp_path, "file.txt", "content a")
    digest_a = hash_file(path)
    path.write_text("content b", encoding="utf-8")
    digest_b = hash_file(path)

    assert digest_a != digest_b


def test_discover_skips_node_modules(tmp_path):
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "README.md").write_text("# pkg\n")
    (tmp_path / "real.md").write_text("# real\n")
    assert discover_files(tmp_path, ["**/*.md"], []) == ["real.md"]


def test_default_globs_index_source_and_docs_but_not_dependency_files(tmp_path: Path):
    """Lockfiles and dependency manifests are large, repetitive and match many queries
    lexically; indexing them crowds real code and prose out of the results."""
    from ragx.core.corpus_globs import DEFAULT_EXCLUDE, DEFAULT_INCLUDE

    for rel in [
        "README.md", "docs/guide.rst", "notes.txt", "src/app.py", "web/index.tsx",
        "cmd/main.go", "Dockerfile",
        "package-lock.json", "package.json", "yarn.lock", "uv.lock", "Cargo.lock",
        "go.sum", "requirements.txt", "requirements-dev.txt", "LICENSE.md",
        "static/app.min.js", "dist/bundle.js", "web/build/out.js", "data.csv",
    ]:
        _write(tmp_path, rel)

    files = discover_files(tmp_path, include=DEFAULT_INCLUDE, exclude=DEFAULT_EXCLUDE)

    assert files == [
        "Dockerfile", "README.md", "cmd/main.go", "docs/guide.rst", "notes.txt",
        "src/app.py", "web/index.tsx",
    ]
