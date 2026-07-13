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
