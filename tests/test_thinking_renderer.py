"""Behavior tests for ThinkingRenderer — the show_thinking policy.

The renderer owns the off/terse/full state machine and the
thinking->content transition, so cli and tui only have to format and
write the pieces it emits.
"""

from agent99x.reasoning import ContentType, ThinkingRenderer

CONTENT = ContentType.CONTENT
THINKING = ContentType.THINKING


def test_full_passes_content_through():
    r = ThinkingRenderer("full")
    assert r.feed(CONTENT, "hello") == [(CONTENT, "hello")]


def test_off_drops_thinking_keeps_content():
    r = ThinkingRenderer("off")
    assert r.feed(THINKING, "pondering") == []
    assert r.feed(CONTENT, "answer") == [(CONTENT, "answer")]


def test_terse_shows_only_first_line_then_suppresses():
    r = ThinkingRenderer("terse")
    assert r.feed(THINKING, "first\nsecond\nthird") == [(THINKING, "first")]
    assert r.feed(THINKING, "more thinking") == []


def test_terse_first_line_spans_tokens_until_newline():
    r = ThinkingRenderer("terse")
    assert r.feed(THINKING, "abc") == [(THINKING, "abc")]
    assert r.feed(THINKING, "def\nghi") == [(THINKING, "def")]
    assert r.feed(THINKING, "jkl") == []


def test_shown_thinking_then_content_inserts_separator():
    r = ThinkingRenderer("full")
    assert r.feed(THINKING, "thinking") == [(THINKING, "thinking")]
    assert r.feed(CONTENT, "answer") == [(CONTENT, "\n"), (CONTENT, "answer")]


def test_off_thinking_then_content_has_no_separator():
    r = ThinkingRenderer("off")
    assert r.feed(THINKING, "thinking") == []
    assert r.feed(CONTENT, "answer") == [(CONTENT, "answer")]


def test_terse_resets_for_each_thinking_block():
    r = ThinkingRenderer("terse")
    assert r.feed(THINKING, "a\nb") == [(THINKING, "a")]
    assert r.feed(CONTENT, "x") == [(CONTENT, "\n"), (CONTENT, "x")]
    assert r.feed(THINKING, "c\nd") == [(THINKING, "c")]


def test_separator_inserted_only_on_first_transition():
    r = ThinkingRenderer("full")
    r.feed(THINKING, "t1")
    assert r.feed(CONTENT, "c1") == [(CONTENT, "\n"), (CONTENT, "c1")]
    assert r.feed(THINKING, "t2") == [(THINKING, "t2")]
    assert r.feed(CONTENT, "c2") == [(CONTENT, "c2")]
