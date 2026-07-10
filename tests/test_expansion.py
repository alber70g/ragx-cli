from __future__ import annotations

from ragx.core.expansion import Expansion, expand_query


class FakeGenerator:
    model = "fake"

    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = 0

    def generate(self, system, prompt, *, max_tokens=1024):
        self.calls += 1
        if self._exc is not None:
            raise self._exc
        return self._response


def test_valid_json():
    gen = FakeGenerator('{"queries": ["how to foo", "foo bar baz"], "hyde": "Foo is a thing."}')
    result = expand_query(gen, "what is foo", variants=3, hyde=True)
    assert gen.calls == 1
    assert result.variants == ["how to foo", "foo bar baz"]
    assert result.hyde == "Foo is a thing."


def test_fenced_json():
    gen = FakeGenerator('```json\n{"queries": ["alt query"]}\n```')
    result = expand_query(gen, "original", variants=3, hyde=False)
    assert result.variants == ["alt query"]
    assert result.hyde is None


def test_garbage_response_returns_empty():
    gen = FakeGenerator("not json at all")
    result = expand_query(gen, "q", variants=3, hyde=True)
    assert result == Expansion([], None)


def test_wrong_shape_returns_empty():
    gen = FakeGenerator('{"foo": "bar"}')
    result = expand_query(gen, "q", variants=3, hyde=True)
    assert result == Expansion([], None)


def test_generator_raises_returns_empty():
    gen = FakeGenerator(exc=RuntimeError("boom"))
    result = expand_query(gen, "q", variants=3, hyde=True)
    assert result == Expansion([], None)


def test_clamps_to_variants_and_drops_dupes_and_empties():
    gen = FakeGenerator(
        '{"queries": ["", "Original Query", "one", "two", "three", "four"]}'
    )
    result = expand_query(gen, "original query", variants=3, hyde=False)
    assert result.variants == ["one", "two", "three"]


def test_only_one_generate_call():
    gen = FakeGenerator('{"queries": ["x"]}')
    expand_query(gen, "q")
    assert gen.calls == 1
