"""Plan mode blocks mutating tools, driven by registry traits.

These exercise the dispatcher's trait lookups (core consulting the
registry) rather than any hardcoded tool-name set.
"""

import agent99x.tools  # noqa: F401 — ensure all tools are registered
from agent99x import core


def test_plan_mode_blocks_all_mutating_file_tools(session):
    session.plan_mode = True
    for name in ("write_file", "edit_file", "replace_lines", "patch"):
        reason = core._plan_mode_block_reason(session, name, {})
        assert reason is not None, f"{name} should be blocked in plan mode"


def test_plan_mode_allows_read_only_tools(session):
    session.plan_mode = True
    assert core._plan_mode_block_reason(session, "read_file", {"path": "x"}) is None
    assert core._plan_mode_block_reason(session, "grep", {"pattern": "x"}) is None


def test_plan_mode_hides_mutating_tools_from_offered_list(session):
    session.plan_mode = True
    offered = {t["function"]["name"] for t in core._active_tools(session)}
    assert "write_file" not in offered
    assert "replace_lines" not in offered
    assert "read_file" in offered


def test_no_plan_mode_blocks_nothing(session):
    session.plan_mode = False
    assert core._plan_mode_block_reason(session, "write_file", {}) is None
