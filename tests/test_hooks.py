import json
import threading
import time
from types import SimpleNamespace

import pytest

from agent99x import core
from agent99x.hooks import AgentHooks, HookEvent
from agent99x import tools
from agent99x.session import SessionConfig


def test_agent_loop_emits_metadata_only_run_and_model_events(session, fake_lm):
    fake_lm.queue([
        fake_lm.chunk(content="private answer"),
        fake_lm.chunk(finish_reason="stop"),
        fake_lm.usage_chunk(prompt_tokens=42),
    ])
    session.history = [{"role": "user", "content": "private prompt"}]
    events = []

    result = core.agent_loop(session, hooks=AgentHooks([events.append]))

    assert result == "private answer"
    assert [event.name for event in events] == [
        "run.started",
        "model_request.started",
        "model_request.finished",
        "run.finished",
    ]
    assert events[0].run_id == events[-1].run_id
    assert events[-1].metadata["status"] == "completed"
    assert events[2].metadata["input_tokens"] == 42
    assert "private prompt" not in repr(events)
    assert "private answer" not in repr(events)
    with pytest.raises(TypeError):
        events[0].metadata["unexpected"] = "value"


def test_agent_loop_reports_tool_metadata_without_arguments_or_results(
    session, fake_lm, in_tmp_cwd
):
    target = in_tmp_cwd / "secret.txt"
    target.write_text("private file contents", encoding="utf-8")
    fake_lm.queue([
        fake_lm.chunk(tool_calls=[
            fake_lm.tool_call_delta(
                0,
                id="call_1",
                name="read_file",
                arguments=json.dumps({"path": str(target)}),
            ),
        ]),
        fake_lm.chunk(finish_reason="tool_calls"),
    ])
    fake_lm.queue([
        fake_lm.chunk(content="read complete"),
        fake_lm.chunk(finish_reason="stop"),
    ])
    session.history = [{"role": "user", "content": "read the file"}]
    events = []

    result = core.agent_loop(session, hooks=AgentHooks([events.append]))

    assert result == "read complete"
    tool_events = [event for event in events if event.name.startswith("tool.")]
    assert [event.name for event in tool_events] == ["tool.started", "tool.finished"]
    assert tool_events[0].metadata == {"tool": "read_file"}
    assert tool_events[1].metadata["status"] == "completed"
    assert "private file contents" not in repr(events)
    assert str(target) not in repr(events)


def test_nested_agent_inherits_hooks_and_links_parent_run(
    session, fake_lm, clean_registry
):
    @tools.tool("nested_agent", "Run a nested agent.", needs_session=True)
    def nested_agent(session=None):
        child_session = SessionConfig(
            provider=session.provider,
            model=session.model,
            client=session.client,
            shared_state=session.shared_state,
            history=[{"role": "user", "content": "nested prompt"}],
        )
        return {"result": core.agent_loop(child_session)}

    fake_lm.queue([
        fake_lm.chunk(tool_calls=[
            fake_lm.tool_call_delta(0, id="call_1", name="nested_agent", arguments="{}"),
        ]),
        fake_lm.chunk(finish_reason="tool_calls"),
    ])
    fake_lm.queue([fake_lm.chunk(content="nested answer")])
    fake_lm.queue([fake_lm.chunk(content="outer answer")])
    session.history = [{"role": "user", "content": "outer prompt"}]
    events = []

    result = core.agent_loop(session, hooks=AgentHooks([events.append]))

    assert result == "outer answer"
    starts = [event for event in events if event.name == "run.started"]
    assert len(starts) == 2
    parent, child = starts
    assert child.parent_run_id == parent.run_id
    assert all(event.parent_run_id == parent.run_id for event in events
               if event.run_id == child.run_id)


def test_hook_failures_do_not_stop_other_handlers_or_the_agent(
    session, fake_lm, caplog
):
    fake_lm.queue([fake_lm.chunk(content="still works")])
    session.history = [{"role": "user", "content": "hello"}]
    seen = []

    def failing_handler(event):
        raise ValueError("handler failure")

    result = core.agent_loop(
        session,
        hooks=AgentHooks([failing_handler, seen.append]),
    )

    assert result == "still works"
    assert any(event.name == "run.finished" for event in seen)
    assert "hook handler" in caplog.text


def test_async_hook_is_rejected_without_affecting_agent(caplog):
    called = []

    async def async_handler(_event):
        called.append(True)

    hooks = AgentHooks([async_handler])
    hooks.emit(HookEvent("run.started", "run", None, time.time(), {}))

    assert called == []
    assert "synchronous" in caplog.text


def test_cancelled_run_emits_terminal_status(session):
    session.history = [{"role": "user", "content": "cancel me"}]
    cancel = threading.Event()
    cancel.set()
    events = []

    result = core.agent_loop(session, cancel=cancel, hooks=AgentHooks([events.append]))

    assert result == "(cancelled)"
    assert [event.name for event in events] == [
        "run.started",
        "run.cancelled",
        "run.finished",
    ]
    assert events[-1].metadata["status"] == "cancelled"


def test_model_failure_is_reported_and_re_raised(session, fake_lm):
    session.history = [{"role": "user", "content": "fail"}]
    events = []

    with pytest.raises(AssertionError, match="no streams queued"):
        core.agent_loop(session, hooks=AgentHooks([events.append]))

    failed_request = next(
        event for event in events if event.name == "model_request.finished"
    )
    assert failed_request.metadata["status"] == "failed"
    assert failed_request.metadata["error_type"] == "AssertionError"
    assert next(event for event in events if event.name == "error").metadata[
        "source"
    ] == "agent"
    assert events[-1].metadata["status"] == "failed"


def test_hook_events_are_delivered_serially_across_threads():
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def handler(event):
        if event.name == "first":
            first_entered.set()
            release_first.wait(timeout=2)
        elif event.name == "second":
            second_entered.set()

    hooks = AgentHooks([handler])

    def event(name):
        return HookEvent(name, "run", None, time.time(), {})

    first = threading.Thread(target=hooks.emit, args=(event("first"),))
    second = threading.Thread(target=hooks.emit, args=(event("second"),))
    first.start()
    assert first_entered.wait(timeout=1)
    second.start()
    assert not second_entered.wait(timeout=0.05)
    release_first.set()
    first.join(timeout=1)
    second.join(timeout=1)
    assert second_entered.is_set()


def test_one_shot_cli_forwards_registered_hooks(monkeypatch, session):
    from agent99x import cli

    hooks = AgentHooks()
    seen = {}
    monkeypatch.setattr(cli, "provider_ready", lambda _session: True)
    monkeypatch.setattr(cli, "_mcp_start", lambda _session: None)
    monkeypatch.setattr(cli, "_mcp_stop", lambda _session: None)
    monkeypatch.setattr(cli, "save_session", lambda _session: None)
    monkeypatch.setattr(
        cli.core,
        "agent_loop",
        lambda _session, **kwargs: seen.update(kwargs),
    )

    cli._run_oneshot(session, hooks=hooks)

    assert seen["hooks"] is hooks


def test_tui_worker_forwards_registered_hooks(monkeypatch, session):
    from agent99x import tui

    hooks = AgentHooks()
    seen = {}
    monkeypatch.setattr(
        tui.core,
        "agent_loop",
        lambda _session, **kwargs: (seen.update(kwargs) or "reply"),
    )
    monkeypatch.setattr(tui.compaction, "maybe_autocompact", lambda *_: None)

    app = tui.AgentApp(session, hooks=hooks)
    app._agent_workers = {1: tui.WorkerState(1, threading.Event())}
    app._agent_workers_lock = threading.Lock()
    app._last_usage = 0
    app.call_from_thread = lambda *_args: None
    app.query_one = lambda *_args: SimpleNamespace(write=lambda *_args: None)

    app._agent_worker(1)

    assert seen["hooks"] is hooks
