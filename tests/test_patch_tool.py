"""Tests for applying patches (diffs)."""

import pytest
import agent99x.tools  # noqa: F401 — fire all @tool registrations
from agent99x import tools as registry, tools as tools_file

def test_apply_patch_happy_path(in_tmp_cwd):
    p = in_tmp_cwd / "hello.txt"
    p.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")
    
    diff = """--- hello.txt
+++ hello.txt
@@ -1,3 +1,3 @@
 line 1
-line 2
+line TWO
 line 3
"""
    result = tools_file.apply_patch(str(p), diff)
    assert result["ok"] is True
    assert p.read_text(encoding="utf-8") == "line 1\nline TWO\nline 3\n"

def test_apply_patch_context_mismatch(in_tmp_cwd):
    p = in_tmp_cwd / "hello.txt"
    p.write_text("line 1\nWRONG LINE\nline 3\n", encoding="utf-8")
    
    diff = """--- hello.txt
+++ hello.txt
@@ -1,3 +1,3 @@
 line 1
-line 2
+line TWO
 line 3
"""
    result = tools_file.apply_patch(str(p), diff)
    assert "error" in result
    assert "context mismatch" in result["error"].lower()

def test_apply_patch_invalid_format(in_tmp_cwd):
    p = in_tmp_cwd / "hello.txt"
    p.write_text("content", encoding="utf-8")
    
    result = tools_file.apply_patch(str(p), "not a diff")
    assert "error" in result

def test_patch_tool_is_registered():
    # Verify the tool is in the registry
    patch_tool = next((t for t in registry.TOOLS if t["function"]["name"] == "patch"), None)
    assert patch_tool is not None
    assert "patch" in registry.TOOL_HANDLERS
    assert registry.TOOL_HANDLERS["patch"] == tools_file.apply_patch
