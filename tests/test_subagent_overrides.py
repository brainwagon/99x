"""Per-subagent provider/host/model/effort overrides via AGENT.md frontmatter."""

import pytest

from agent99x import core, providers, skills as subagents
from agent99x.session import SessionConfig


@pytest.fixture
def captured_loop(monkeypatch):
    """Replace agent_loop with a recorder; returns the list of seen sub_sessions."""
    seen = []

    def fake_loop(sub_session, **_kw):
        seen.append(sub_session)
        return "ok"

    monkeypatch.setattr(core, "agent_loop", fake_loop)
    return seen


def _stub_build_agent_system(monkeypatch, meta):
    system = {"role": "system", "content": "x"}
    from agent99x import skills as subagents
    monkeypatch.setattr(subagents, "build_agent_system", lambda s, name=None: (system, meta))


def test_provider_override_creates_fresh_client(monkeypatch, session, captured_loop):
    _stub_build_agent_system(monkeypatch, {"provider": "ollama"})
    monkeypatch.setattr(providers, "list_models", lambda s: ["llama3"])

    parent_client = session.client
    res = subagents.spawn_agent("do it", name="x", session=session)
    assert "result" in res

    sub = captured_loop[0]
    assert sub.provider.name == "ollama"
    assert sub.base_url == "http://localhost:11434/v1"
    assert sub.client is not parent_client


def test_host_override_swaps_netloc(monkeypatch, session, captured_loop):
    _stub_build_agent_system(monkeypatch, {"host": "10.0.0.5:9999"})
    monkeypatch.setattr(providers, "list_models", lambda s: [session.model])

    subagents.spawn_agent("do it", name="x", session=session)

    sub = captured_loop[0]
    assert sub.base_url == "http://10.0.0.5:9999/v1"
    assert sub.provider == session.provider


def test_model_only_reuses_parent_client(monkeypatch, session, captured_loop):
    _stub_build_agent_system(monkeypatch, {"model": "alt-model"})

    parent_client = session.client
    subagents.spawn_agent("do it", name="x", session=session)

    sub = captured_loop[0]
    assert sub.model == "alt-model"
    assert sub.client is parent_client
    assert sub.base_url == session.base_url


def test_effort_override(monkeypatch, session, captured_loop):
    _stub_build_agent_system(monkeypatch, {"effort": "high"})

    subagents.spawn_agent("do it", name="x", session=session)

    assert captured_loop[0].effort == "high"


def test_no_overrides_inherits_everything(monkeypatch, session, captured_loop):
    _stub_build_agent_system(monkeypatch, {})

    subagents.spawn_agent("do it", name="x", session=session)

    sub = captured_loop[0]
    assert sub.provider == session.provider
    assert sub.model == session.model
    assert sub.client is session.client


def test_unknown_provider_returns_error(monkeypatch, session, captured_loop):
    _stub_build_agent_system(monkeypatch, {"provider": "made-up"})

    res = subagents.spawn_agent("do it", name="x", session=session)
    assert "error" in res
    assert "made-up" in res["error"]
    assert captured_loop == []
