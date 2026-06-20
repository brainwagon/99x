"""Behavior tests for the three-kind model: tools / skills / agents.

Covers the symmetric discovery surface (the 'list tools returns nothing' bug),
the terse-grouped tool catalog, meta-tool hiding, load_skill returning its dir,
and drop-in agentskills.io SKILL.md frontmatter parsing.
"""

import agent99x.tools as tools
import agent99x.skills as skills  # noqa: F401 — registers the meta/bridge tools
from agent99x.prompt import parse_frontmatter, _build_capabilities
from agent99x.providers import PROVIDERS
from agent99x.session import SessionConfig


# ── tool catalog & list_tools ───────────────────────────────────────

def test_tool_catalog_groups_primitives_terse():
    cat = tools.tool_catalog()
    assert "- Files: read_file" in cat
    assert "- Shell: run_bash" in cat
    assert "- Search: grep, glob" in cat
    # terse: names only, no descriptions repeated from the schemas
    assert "Read a file from disk" not in cat


def test_meta_tools_hidden_from_catalog():
    cat = tools.tool_catalog()
    for meta in ("spawn_agent", "load_skill", "list_tools", "list_skills", "list_agents"):
        assert meta not in cat


def test_list_tools_is_exhaustive_and_includes_primitives_and_meta():
    names = {t["name"] for t in tools.tool_listing()}
    # the original bug: real tools must be enumerable
    assert {"run_bash", "read_file", "grep"} <= names
    # the exhaustive list also surfaces the bridge tools
    assert {"list_tools", "spawn_agent", "load_skill"} <= names


def test_list_tools_handler_returns_listing():
    out = tools.TOOL_HANDLERS["list_tools"]()
    assert any(t["name"] == "run_bash" and t["group"] == "shell" for t in out["tools"])


def test_catalog_allowed_filter_restricts_tools():
    cat = tools.tool_catalog(allowed={"read_file", "grep"})
    assert "read_file" in cat and "grep" in cat
    assert "run_bash" not in cat
    assert "write_file" not in cat


def test_unknown_group_falls_into_other(clean_registry):
    tools.register(
        {"type": "function",
         "function": {"name": "mcp_thing", "description": "x",
                      "parameters": {"type": "object", "properties": {}, "required": []}}},
        lambda: None,
    )
    cat = tools.tool_catalog()
    assert "- Other: mcp_thing" in cat


# ── skills are pure prose ───────────────────────────────────────────

def test_run_skill_script_is_gone():
    assert "run_skill_script" not in tools.TOOL_HANDLERS


def test_load_skill_returns_dir(tmp_path, monkeypatch):
    from agent99x import scopes
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: A demo skill.\n---\n\nDo the thing.\n")
    monkeypatch.setattr(scopes, "PROJECT_DIR", str(tmp_path))
    monkeypatch.setattr(scopes, "AGENT_HOME", str(tmp_path / "nope"))
    out = tools.TOOL_HANDLERS["load_skill"]("demo")
    assert out["skill"] == "demo"
    assert "Do the thing." in out["content"]
    assert out["dir"] == str(skill_dir)


# ── agentskills.io frontmatter compatibility ────────────────────────

def test_frontmatter_parses_nested_metadata_map():
    meta, body = parse_frontmatter(
        "---\n"
        "name: pdf-processing\n"
        "description: Extract PDF text.\n"
        "metadata:\n"
        "  author: example-org\n"
        '  version: "1.0"\n'
        "---\n\nBody\n")
    assert meta["metadata"] == {"author": "example-org", "version": "1.0"}
    assert body.startswith("Body")


def test_frontmatter_keeps_allowed_tools_as_string_and_flat_keys():
    meta, _ = parse_frontmatter(
        "---\n"
        "name: x\n"
        "compatibility: Requires git and uv\n"
        "allowed-tools: Bash(git:*) Read\n"
        "---\nb")
    assert meta["allowed-tools"] == "Bash(git:*) Read"
    assert meta["compatibility"] == "Requires git and uv"


def test_frontmatter_flat_list_still_parses():
    meta, _ = parse_frontmatter("---\nallowed_tools: [read_file, grep]\n---\nb")
    assert meta["allowed_tools"] == ["read_file", "grep"]


# ── capabilities block ──────────────────────────────────────────────

def test_capabilities_block_has_primer_and_tools_section():
    s = SessionConfig(provider=PROVIDERS["ollama"], model="m")
    block = _build_capabilities(s)
    assert "# YOUR CAPABILITIES" in block
    assert "Tools" in block and "Skills" in block.replace("## Skills", "Skills")
    assert "do it yourself → tool" in block
    assert "## Tools" in block
