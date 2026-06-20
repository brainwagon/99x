"""Tests for read_file / write_file / edit_file."""

import json

from agent99x import tools as tools_file


def test_read_file_missing_returns_error(in_tmp_cwd):
    result = tools_file.read_file("does-not-exist.txt")
    assert "error" in result
    assert "content" not in result


def test_read_file_returns_content(in_tmp_cwd):
    p = in_tmp_cwd / "hello.txt"
    p.write_text("hi there", encoding="utf-8")
    result = tools_file.read_file(str(p))
    assert result == {"content": "hi there"}


def test_write_file_creates_file_and_reports_bytes(in_tmp_cwd):
    p = in_tmp_cwd / "out.txt"
    result = tools_file.write_file(str(p), "hello")
    assert result == {"ok": True, "bytes": 5}
    assert p.read_text(encoding="utf-8") == "hello"


def test_write_file_overwrites(in_tmp_cwd):
    p = in_tmp_cwd / "out.txt"
    p.write_text("old", encoding="utf-8")
    tools_file.write_file(str(p), "new content")
    assert p.read_text(encoding="utf-8") == "new content"


def test_edit_file_happy_path(in_tmp_cwd):
    p = in_tmp_cwd / "f.txt"
    p.write_text("alpha beta gamma", encoding="utf-8")
    result = tools_file.edit_file(str(p), "beta", "BETA")
    assert result["ok"] is True
    assert result["replacements"] == 1
    assert p.read_text(encoding="utf-8") == "alpha BETA gamma"


def test_edit_file_missing_string_errors(in_tmp_cwd):
    p = in_tmp_cwd / "f.txt"
    p.write_text("alpha", encoding="utf-8")
    result = tools_file.edit_file(str(p), "missing", "x")
    assert "error" in result
    assert "not found" in result["error"]


def test_edit_file_ambiguous_match_errors(in_tmp_cwd):
    p = in_tmp_cwd / "f.txt"
    p.write_text("ab ab ab", encoding="utf-8")
    result = tools_file.edit_file(str(p), "ab", "X")
    assert "error" in result
    assert "matches 3 times" in result["error"]
    # File must remain untouched on error.
    assert p.read_text(encoding="utf-8") == "ab ab ab"


def test_edit_file_replace_all(in_tmp_cwd):
    p = in_tmp_cwd / "f.txt"
    p.write_text("ab ab ab", encoding="utf-8")
    result = tools_file.edit_file(str(p), "ab", "X", replace_all=True)
    assert result["ok"] is True
    assert result["replacements"] == 3
    assert p.read_text(encoding="utf-8") == "X X X"


def test_read_file_offset_limit_slice(in_tmp_cwd):
    p = in_tmp_cwd / "lines.txt"
    p.write_text("one\ntwo\nthree\nfour\nfive\n", encoding="utf-8")
    result = tools_file.read_file(str(p), offset=2, limit=2)
    assert result["content"] == "two\nthree\n"
    assert result["total_lines"] == 5
    assert result["start_line"] == 2
    assert result["end_line"] == 3


def test_read_file_offset_only(in_tmp_cwd):
    p = in_tmp_cwd / "lines.txt"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    result = tools_file.read_file(str(p), offset=2)
    assert result["content"] == "b\nc\n"
    assert result["start_line"] == 2
    assert result["end_line"] == 3


def test_read_file_limit_only(in_tmp_cwd):
    p = in_tmp_cwd / "lines.txt"
    p.write_text("a\nb\nc\n", encoding="utf-8")
    result = tools_file.read_file(str(p), limit=2)
    assert result["content"] == "a\nb\n"
    assert result["end_line"] == 2


def test_edit_file_missing_path_errors(in_tmp_cwd):
    result = tools_file.edit_file("nope.txt", "a", "b")
    assert "error" in result
