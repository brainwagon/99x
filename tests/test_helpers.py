"""Tests for pure-helper functions."""

import importlib

from agent99x import core


def test_clip_below_threshold_unchanged():
    assert core.clip("hello", n=10) == "hello"


def test_clip_at_threshold_unchanged():
    assert core.clip("hello", n=5) == "hello"


def test_clip_above_threshold_truncates_with_ellipsis():
    assert core.clip("hello world", n=5) == "hello..."


def test_clip_custom_ellipsis():
    assert core.clip("hello world", n=5, ellipsis="…") == "hello…"


def test_clip_coerces_non_string():
    assert core.clip(12345, n=3) == "123..."


def test_linkify_wraps_url():
    nn = importlib.import_module("agent99x.tui")
    out = nn._linkify("see http://example.com for more")
    assert "[link=http://example.com]" in out
    assert "[/link]" in out


def test_linkify_strips_trailing_punctuation():
    nn = importlib.import_module("agent99x.tui")
    out = nn._linkify("visit http://example.com.")
    assert "[link=http://example.com]" in out
    assert out.endswith(".")


def test_linkify_no_url_passthrough():
    nn = importlib.import_module("agent99x.tui")
    assert nn._linkify("plain text") == "plain text"
