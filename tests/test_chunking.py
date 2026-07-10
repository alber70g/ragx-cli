from ragx.core.chunking import chunk_text


def _assert_slice_invariant(text: str, drafts) -> None:
    text_bytes = text.encode("utf-8")
    lines = text.split("\n")
    n_lines = len(lines)
    for d in drafts:
        assert text_bytes[d.byte_start : d.byte_end].decode("utf-8") == d.text
        assert d.text != ""
        assert 1 <= d.line_start <= d.line_end <= n_lines


def test_empty_and_whitespace_only_files_return_no_chunks():
    assert chunk_text("", "empty.txt") == []
    assert chunk_text("   \n\t\n  ", "whitespace.txt") == []


def test_slice_invariant_holds_for_multibyte_content():
    paragraph = "café ☕ mañana 日本語 résumé 北京 naïve\n\n"
    text = paragraph * 40

    drafts = chunk_text(text, "notes.txt", size_tokens=20, overlap=0.15)

    assert len(drafts) > 1
    _assert_slice_invariant(text, drafts)


def test_markdown_small_sections_merge():
    text = "\n\n".join(f"# Heading {i}\n\nShort paragraph {i}." for i in range(6))

    drafts = chunk_text(text, "doc.md", size_tokens=50, overlap=0.1)

    # 6 small sections should merge into fewer chunks than headings.
    assert 0 < len(drafts) < 6
    _assert_slice_invariant(text, drafts)


def test_markdown_oversized_section_falls_back_to_recursive_split():
    big_body = "This is a long sentence about ragx. " * 40
    text = f"# Intro\n\nshort\n\n# Big Section\n\n{big_body}"

    drafts = chunk_text(text, "doc.md", size_tokens=10, overlap=0.1)

    # the oversized section must have been split into more than one piece
    assert len(drafts) > 2
    _assert_slice_invariant(text, drafts)


def test_recursive_split_overlap_ranges_actually_overlap():
    text = ("word " * 400).strip()

    drafts = chunk_text(text, "plain.txt", size_tokens=20, overlap=0.2)

    assert len(drafts) > 1
    for prev, cur in zip(drafts, drafts[1:]):
        assert cur.byte_start < prev.byte_end
    _assert_slice_invariant(text, drafts)


def test_code_split_on_top_level_boundaries():
    text = (
        "import os\n\n"
        "def first():\n    return 1\n\n\n"
        "def second():\n    return 2\n\n\n"
        "class Third:\n    def method(self):\n        return 3\n"
    )

    drafts = chunk_text(text, "module.py", size_tokens=50, overlap=0.1)

    assert len(drafts) >= 3
    _assert_slice_invariant(text, drafts)


def test_code_oversized_function_falls_back_to_recursive_split():
    body_lines = "\n".join(f"    x{i} = {i}" for i in range(200))
    text = f"def big():\n{body_lines}\n"

    drafts = chunk_text(text, "big.py", size_tokens=10, overlap=0.1)

    assert len(drafts) > 1
    _assert_slice_invariant(text, drafts)


def test_no_whitespace_only_chunks():
    # heading sections separated by blank-only regions must not emit whitespace drafts
    text = "# A\n\n\n\n# B\n\ncontent here\n"
    drafts = chunk_text(text, "x.md", size_tokens=4, overlap=0.15)
    assert drafts
    assert all(d.text.strip() for d in drafts)
