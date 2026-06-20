"""Tests for pure-helper functions."""

import importlib
import pytest

from agent99x import core
from agent99x import providers
from agent99x import tools
from agent99x import cli, commands, session
from agent99x.session import SessionConfig


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


def test_usage_bar_no_window_returns_token_count():
    session = SessionConfig(context_window=0)
    assert cli.usage_bar(1234, session) == "1234 tokens"


def test_usage_bar_with_window_renders_bar():
    session = SessionConfig(context_window=1000)
    out = cli.usage_bar(500, session, width=10)
    assert "500/1,000" in out
    assert "(50%)" in out
    assert out.startswith("[")
    assert out.count("█") + out.count("░") == 10


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
