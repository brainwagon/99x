"""Behavior tests for the scope resolver (skills/agents precedence).

The resolver owns the rule "project shadows global" for both skills and
agents, so callers stop hand-rolling dir lists.
"""

from types import SimpleNamespace

import pytest

from agent99x import scopes


@pytest.fixture
def scope_dirs(tmp_path, monkeypatch):
    """Point the two scope tiers at isolated tmp dirs."""
    project = tmp_path / "project"
    glob = tmp_path / "global"
    monkeypatch.setattr(scopes, "PROJECT_DIR", str(project))
    monkeypatch.setattr(scopes, "AGENT_HOME", str(glob))
    return SimpleNamespace(project=project, glob=glob)


_MANIFEST = {"skills": "SKILL.md", "agents": "AGENT.md"}


def _make(tier, kind, name, body="desc"):
    """Create a kind/name manifest under a tier dir; return its path."""
    path = tier / kind / name / _MANIFEST[kind]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return str(path)


def test_resolve_finds_skill_in_project(scope_dirs):
    path = _make(scope_dirs.project, "skills", "weather")
    assert scopes.resolve("skills", "weather") == path


def test_resolve_priority_project_then_global(scope_dirs):
    gl = _make(scope_dirs.glob, "skills", "weather")
    assert scopes.resolve("skills", "weather") == gl

    pr = _make(scope_dirs.project, "skills", "weather")
    assert scopes.resolve("skills", "weather") == pr


def test_resolve_unknown_returns_none(scope_dirs):
    assert scopes.resolve("skills", "nope") is None


def test_discover_merges_tiers_project_shadows_global(scope_dirs):
    _make(scope_dirs.glob, "skills", "alpha")
    glob_beta = _make(scope_dirs.glob, "skills", "beta")
    proj_alpha = _make(scope_dirs.project, "skills", "alpha")

    found = scopes.discover("skills")

    assert found == [("alpha", proj_alpha), ("beta", glob_beta)]


def test_agents_use_agent_md(scope_dirs):
    gl = _make(scope_dirs.glob, "agents", "helper")
    assert scopes.resolve("agents", "helper") == gl
    assert scopes.discover("agents") == [("helper", gl)]
