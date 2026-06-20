"""Behavior tests for tool-trait metadata on the registry.

The registry owns per-tool traits (needs_session, mutating, plan-safe) so
the dispatcher can ask the registry instead of hardcoding tool-name sets.
"""

from agent99x import tools as registry


def test_default_traits_are_safe(clean_registry):
    @registry.tool("dummy_default", "A plain tool.")
    def _handler():
        return {}

    assert registry.needs_session("dummy_default") is False
    assert registry.is_mutating("dummy_default") is False
    assert registry.is_plan_safe("dummy_default") is True


def test_needs_session_trait_recorded(clean_registry):
    @registry.tool("dummy_spawn", "Needs the session.", needs_session=True)
    def _handler(session=None):
        return {}

    assert registry.needs_session("dummy_spawn") is True
    assert registry.TOOL_HANDLERS["dummy_spawn"] is _handler


def test_mutating_and_plan_unsafe_traits_recorded(clean_registry):
    @registry.tool("dummy_write", "Writes files.", mutates=True, plan_safe=False)
    def _handler():
        return {}

    assert registry.is_mutating("dummy_write") is True
    assert registry.is_plan_safe("dummy_write") is False


def test_register_primitive_adds_tool_with_default_safe_traits(clean_registry):
    schema = {"type": "function", "function": {"name": "mcp_dummy", "description": "x"}}

    def handler():
        return {"ok": True}

    registry.register(schema, handler)

    assert registry.TOOL_HANDLERS["mcp_dummy"] is handler
    assert schema in registry.TOOLS
    assert registry.needs_session("mcp_dummy") is False
    assert registry.is_mutating("mcp_dummy") is False
    assert registry.is_plan_safe("mcp_dummy") is True


def test_register_is_idempotent_by_name(clean_registry):
    schema = {"type": "function", "function": {"name": "mcp_dup", "description": "x"}}

    def first():
        return 1

    def second():
        return 2

    registry.register(schema, first)
    registry.register(schema, second)

    names = [t["function"]["name"] for t in registry.TOOLS]
    assert names.count("mcp_dup") == 1
    assert registry.TOOL_HANDLERS["mcp_dup"] is second


def test_unknown_tool_defaults_to_safe():
    assert registry.needs_session("no_such_tool") is False
    assert registry.is_mutating("no_such_tool") is False
    assert registry.is_plan_safe("no_such_tool") is True


def test_real_tools_carry_expected_traits():
    import agent99x.tools  # noqa: F401 — triggers all @tool registrations

    for name in ("spawn_agent",):
        assert registry.needs_session(name) is True

    for name in ("write_file", "edit_file", "replace_lines", "patch"):
        assert registry.is_mutating(name) is True
        assert registry.is_plan_safe(name) is False

    assert registry.is_mutating("read_file") is False
    assert registry.is_plan_safe("read_file") is True
    assert registry.needs_session("read_file") is False
