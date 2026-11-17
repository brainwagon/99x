"""Regression tests for the agent-loop code-review fixes."""

import json
import os
import shutil
import threading
from types import SimpleNamespace

import pytest

from agent99x import compaction, conversation, core, llm, providers, skills, tools, validation
from agent99x.config import MAX_AGENT_DEPTH
from agent99x.hooks import AgentHooks, HookEvent
from agent99x.reasoning import ContentType, ReasoningParser
from agent99x.session import SessionConfig

from conftest import FakeStream, _chunk, _tool_call_delta


# ── helpers ─────────────────────────────────────────────────────────

def _client(streams, seen=None):
    """Fake OpenAI client: each create() pops the next stream (or raises it)."""
    queue = list(streams)

    def create(**kwargs):
        if seen is not None:
            seen.append(kwargs)
        item = queue.pop(0) if queue else [_chunk(content="done")]
        if isinstance(item, BaseException):
            raise item
        return FakeStream(item)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _session(streams, seen=None, **kw):
    s = SessionConfig(provider=providers.PROVIDERS["ollama"], model="test-model",
                      context_window=131072, client=_client(streams, seen), **kw)
    s.history = [{"role": "user", "content": "go"}]
    return s


def _calls(*calls):
    """One model turn issuing the given (id, name, args) tool calls."""
    return [_chunk(tool_calls=[_tool_call_delta(i, id=cid, name=name, arguments=args)
                               for i, (cid, name, args) in enumerate(calls)]),
            _chunk(finish_reason="tool_calls")]


def _assert_paired(history):
    """Every assistant tool_call id is answered by a following tool message."""
    for i, msg in enumerate(history):
        ids = [tc["id"] for tc in msg.get("tool_calls") or ()]
        follow = []
        for m in history[i + 1:]:
            if m["role"] != "tool":
                break
            follow.append(m["tool_call_id"])
        assert sorted(follow) == sorted(ids) if ids else True, (ids, follow)
        if msg["role"] == "tool":
            assert any(msg["tool_call_id"] in [tc["id"] for tc in p.get("tool_calls") or ()]
                       for p in history[:i])


@pytest.fixture
def fresh_depth():
    tl = tools._thread_local
    saved = (getattr(tl, "depth", 0), getattr(tl, "cancel_event", None))
    yield
    tl.depth, tl.cancel_event = saved


# ── 1 + 6: spawn_agent inherits restrictions and cancel ─────────────

@pytest.fixture
def captured_loop(monkeypatch):
    seen = []

    def fake_loop(sub_session, **kw):
        seen.append((sub_session, kw))
        return "ok"

    monkeypatch.setattr(core, "agent_loop", fake_loop)
    return seen


def _stub_agent(monkeypatch, meta):
    monkeypatch.setattr(skills, "build_agent_system",
                        lambda s, name=None: ({"role": "system", "content": "x"}, meta))


def test_subagent_inherits_plan_mode_and_session_fields(monkeypatch, captured_loop, fresh_depth):
    _stub_agent(monkeypatch, {})
    parent = _session([], plan_mode=True, effort="high", effort_suppressed=True,
                      http_timeout=17.0)
    cancel = threading.Event()
    tools._thread_local.cancel_event = cancel
    skills.spawn_agent("t", session=parent)
    sub, kw = captured_loop[0]
    assert sub.plan_mode is True
    assert sub.effort_suppressed is True
    assert sub.http_timeout == 17.0
    assert sub.session_lock is parent.session_lock
    assert sub.shared_state is parent.shared_state
    assert kw["cancel"] is cancel


def test_subagent_allowed_tools_never_exceed_parent(monkeypatch, captured_loop):
    parent = _session([], allowed_tools=["read_file", "grep"])
    _stub_agent(monkeypatch, {"allowed_tools": ["read_file", "write_file"]})
    skills.spawn_agent("t", session=parent)
    _stub_agent(monkeypatch, {})
    skills.spawn_agent("t", session=parent)
    assert captured_loop[0][0].allowed_tools == ["read_file"]
    assert captured_loop[1][0].allowed_tools == ["read_file", "grep"]


def test_subagent_in_plan_mode_cannot_write(monkeypatch, in_tmp_cwd, fresh_depth):
    _stub_agent(monkeypatch, {})
    parent = _session([
        _calls(("p1", "spawn_agent", json.dumps({"task": "write"}))),
        _calls(("s1", "write_file", json.dumps({"path": "x.txt", "content": "hi"}))),
        [_chunk(content="sub done")],
        [_chunk(content="parent done")],
    ], plan_mode=True)
    core.agent_loop(parent)
    assert not (in_tmp_cwd / "x.txt").exists()


# ── 2: plan-mode gate never raises ──────────────────────────────────

@pytest.mark.parametrize("name,args", [
    ("run_bash", None), ("run_bash", ["ls"]), ("run_bash", {"command": None}),
    ("run_bash", {"command": 5}), ("http_request", {"method": None}),
    ("http_request", "GET"),
])
def test_plan_mode_block_reason_bad_shapes_are_blocked(name, args):
    s = SessionConfig(plan_mode=True)
    assert core._plan_mode_block_reason(s, name, args).startswith("Blocked")


# ── 3: tool_call / tool result pairing ──────────────────────────────

def test_repeat_early_stop_fires_on_third_identical_call_and_stays_paired():
    same = ("list_dir", '{"path": "."}')
    s = _session([_calls((f"c{i}",) + same) for i in range(5)])
    result = core.agent_loop(s)
    assert "Early-stop" in result
    assert sum(1 for m in s.history if m.get("tool_calls")) == 3
    assert s.history[-1]["role"] == "tool"
    _assert_paired(s.history)


def test_keyboard_interrupt_in_tool_leaves_paired_history(clean_registry):
    @tools.tool("boom_kbd", "raise ^C")
    def _boom():
        raise KeyboardInterrupt

    events = []
    s = _session([_calls(("c1", "boom_kbd", "{}"))])
    with pytest.raises(KeyboardInterrupt):
        core.agent_loop(s, hooks=AgentHooks([events.append]))
    assert s.history[-1]["role"] == "tool"
    assert "cancelled" in s.history[-1]["content"]
    _assert_paired(s.history)
    # 11: interrupt is reported as cancellation, consistently.
    names = [e.name for e in events]
    assert "run.cancelled" in names
    assert events[-1].name == "run.finished"
    assert events[-1].metadata["status"] == "cancelled"
    tool_done = [e for e in events if e.name == "tool.finished"]
    assert tool_done and tool_done[0].metadata["status"] == "cancelled"


def test_parallel_dispatch_exception_still_records_every_result():
    def on_tool_result(result, _elapsed):
        if "date" in result:
            raise RuntimeError("callback exploded")

    s = _session([_calls(("a", "current_datetime", "{}"), ("b", "read_todos", "{}"))])
    assert core.agent_loop(s, on_tool_result=on_tool_result) == "done"
    _assert_paired(s.history)
    assert "callback exploded" in s.history[2]["content"]


def test_repair_tool_pairs_inserts_placeholders_and_drops_orphans():
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "tool", "tool_call_id": "stray", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "b", "content": "rb"},
        {"role": "tool", "tool_call_id": "b", "content": "dup"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]},
    ]
    out = conversation.repair_tool_pairs(msgs)
    assert [m.get("tool_call_id") for m in out if m["role"] == "tool"] == ["b", "a", "c"]
    assert out[1]["role"] == "assistant"
    assert out[-1]["tool_call_id"] == "c" and "cancelled" in out[-1]["content"]
    assert len(msgs) == 7  # input untouched
    _assert_paired(out)
    assert conversation.repair_tool_pairs(out) == out


def test_load_repairs_broken_saved_context(session, in_tmp_cwd):
    path = in_tmp_cwd / "ctx.json"
    path.write_text(json.dumps([
        {"role": "system", "content": "old"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "a", "arguments": "{}"}}]},
    ]))
    assert conversation.load(session, str(path))
    assert session.history[-1] == {"role": "tool", "tool_call_id": "x",
                                   "content": session.history[-1]["content"]}
    _assert_paired(session.history)


def test_loop_repairs_history_before_model_request():
    seen = []
    s = _session([[_chunk(content="ok")]], seen)
    s.history = [{"role": "user", "content": "hi"},
                 {"role": "assistant", "content": "", "tool_calls": [{"id": "z"}]},
                 {"role": "user", "content": "again"}]
    core.agent_loop(s)
    sent = seen[0]["messages"]
    assert sent[2] == {"role": "tool", "tool_call_id": "z", "content": sent[2]["content"]}


# ── 4: minicpm XML tool parsing is minicpm-only ─────────────────────

_XML = '<function name="run_bash"><param name="command">echo hi</param></function>'


def test_xml_function_text_is_not_a_tool_call_for_other_models():
    s = _session([[_chunk(content="Example: " + _XML)]])
    _c, _r, calls, *_ = llm.stream_completion(s, s.history, [], None)
    assert calls == []


def test_xml_function_text_is_parsed_for_minicpm():
    s = _session([[_chunk(content=_XML)]])
    s.model = "minicpm-v"
    _c, _r, calls, *_ = llm.stream_completion(s, s.history, [], None)
    assert calls[0]["function"]["name"] == "run_bash"


# ── 5: generic "Thought:" only at the start ─────────────────────────

def _split(chunks):
    p = ReasoningParser()
    out = [r for c in chunks for r in p.feed_content(c)] + p.flush()
    return ("".join(t for c, t in out if c == ContentType.THINKING),
            "".join(t for c, t in out if c == ContentType.CONTENT))


def test_mid_answer_reasoning_label_stays_content():
    text = "The answer is 4.\n\n**Reasoning:** 2+2=4. Thought: easy."
    thinking, content = _split([text[:9], text[9:30], text[30:]])
    assert thinking == ""
    assert content == text


def test_leading_bold_thought_ends_at_blank_line():
    thinking, content = _split(["  **Tho", "ught:** plan it", "\n\nHere is the answer."])
    assert "plan it" in thinking
    assert content == "Here is the answer."


def test_leading_thought_ends_at_final_answer():
    thinking, content = _split(["Thought: think\nFinal Answer: 42"])
    assert "think" in thinking
    assert content == "42"


# ── 6: depth/cancel carried into pool workers; no nested-pool deadlock ──

def test_parallel_spawn_agent_respects_max_depth(monkeypatch, fresh_depth):
    _stub_agent(monkeypatch, {})
    tools._thread_local.depth = MAX_AGENT_DEPTH
    s = _session([_calls(("a", "spawn_agent", '{"task": "x"}'),
                         ("b", "spawn_agent", '{"task": "y"}'))])
    core.agent_loop(s)
    results = [json.loads(m["content"]) for m in s.history if m["role"] == "tool"]
    assert len(results) == 2
    assert all("Max agent depth" in r["error"] for r in results)


def test_submit_carries_thread_state_and_worker_dispatch_is_sequential(fresh_depth):
    cancel = threading.Event()
    tools._thread_local.depth = 2
    tools._thread_local.cancel_event = cancel
    got = core._submit(lambda: (core._in_pool_worker(), tools._thread_local.depth,
                                tools._thread_local.cancel_event)).result()
    assert got == (True, 2, cancel)
    assert not core._in_pool_worker()

    threads = []

    def run_inside_worker():
        s = _session([_calls(("a", "current_datetime", "{}"), ("b", "read_todos", "{}"))])
        core.agent_loop(s, on_tool_call=lambda *_: threads.append(threading.current_thread()))
        return threading.current_thread()

    worker = core._submit(run_inside_worker).result(timeout=10)
    assert threads == [worker, worker]


# ── 7: string results containing "error" are not failures ──────────

def test_string_result_mentioning_error_is_not_a_failure(clean_registry):
    @tools.tool("mcp_like", "returns text")
    def _mcp_like():
        return "no error here"

    events = []
    s = _session([_calls(("a", "mcp_like", "{}")), [_chunk(content="fine")]])
    assert core.agent_loop(s, hooks=AgentHooks([events.append])) == "fine"
    finished = [e for e in events if e.name == "tool.finished"]
    assert finished[0].metadata["status"] == "completed"
    assert core._is_error({"error": "x"}) and not core._is_error("error")


# ── 8: bad JSON arguments never reach the tool ──────────────────────

def test_invalid_json_arguments_return_error_without_calling_tool(clean_registry):
    called = []

    @tools.tool("probe_args", "records calls", params={"x": "string"})
    def _probe(**kw):
        called.append(kw)
        return {"ok": True}

    s = _session([_calls(("a", "probe_args", '{"x": ')), [_chunk(content="ok")]])
    core.agent_loop(s)
    assert called == []
    assert "Invalid JSON" in json.loads(s.history[2]["content"])["error"]


@pytest.mark.parametrize("raw", ["", "   ", "null"])
def test_empty_arguments_mean_no_arguments(raw):
    assert core._parse_args(raw) == ({}, None)
    assert core._parse_args("[1]")[0] is None


# ── 9: streamed tool-call delta merging ─────────────────────────────

def test_new_id_on_reused_index_starts_new_call():
    s = _session([[
        _chunk(tool_calls=[_tool_call_delta(0, id="a", name="f", arguments="{}")]),
        _chunk(tool_calls=[_tool_call_delta(0, id="b", name="g", arguments="{")]),
        _chunk(tool_calls=[_tool_call_delta(0, arguments="}")]),
    ]])
    _c, _r, calls, *_ = llm.stream_completion(s, s.history, [], None)
    assert [(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in calls] == [
        ("a", "f", "{}"), ("b", "g", "{}")]


def test_none_index_mixed_with_int_index():
    s = _session([[
        _chunk(tool_calls=[_tool_call_delta(None, id="a", name="f", arguments="{")]),
        _chunk(tool_calls=[_tool_call_delta(None, arguments="}")]),
        _chunk(tool_calls=[_tool_call_delta(0, name="g", arguments="{}")]),
    ]])
    _c, _r, calls, *_ = llm.stream_completion(s, s.history, [], None)
    assert [c["function"]["name"] for c in calls] == ["f", "g"]
    assert calls[0]["function"]["arguments"] == "{}"
    assert calls[1]["id"]  # synthesized when the backend sends none


# ── 10: compaction honours effort_suppressed ────────────────────────

def test_compaction_skips_suppressed_reasoning_effort(session, in_tmp_cwd):
    seen = []
    msg = SimpleNamespace(content="summary")
    session.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: seen.append(kw) or SimpleNamespace(
            choices=[SimpleNamespace(message=msg)], usage=None))))
    session.effort, session.effort_suppressed = "high", True
    session.history = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    compaction.do_compact(session)
    assert "reasoning_effort" not in seen[0]


# ── 11: hooks ───────────────────────────────────────────────────────

def test_interrupted_model_request_has_matching_finished_event():
    events = []
    s = _session([KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        core.agent_loop(s, hooks=AgentHooks([events.append]))
    started = [e for e in events if e.name == "model_request.started"]
    finished = [e for e in events if e.name == "model_request.finished"]
    assert len(started) == len(finished) == 1
    assert finished[0].metadata["status"] == "cancelled"
    assert events[-1].metadata["status"] == "cancelled"


def test_handler_that_emits_does_not_deadlock():
    seen = []
    hooks = AgentHooks()

    def handler(event):
        seen.append(event.name)
        if event.name == "run.started":
            hooks.emit(HookEvent("error", event.run_id, None, 0.0, {}))

    hooks._handlers = (handler,)
    t = threading.Thread(target=hooks.emit,
                         args=(HookEvent("run.started", "r", None, 0.0, {}),), daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()
    assert seen == ["run.started", "error"]


# ── 12: validation and rollback ─────────────────────────────────────

def _write(path, content, cid):
    return _calls((cid, "write_file", json.dumps({"path": path, "content": content})))


def test_already_invalid_file_is_not_rolled_back(in_tmp_cwd):
    (in_tmp_cwd / "bad.py").write_text("def (:\n")
    s = _session([_write("bad.py", "def (: still\n", "w1"), [_chunk(content="ok")]])
    assert core.agent_loop(s) == "ok"
    assert (in_tmp_cwd / "bad.py").read_text() == "def (: still\n"
    assert not any("Validation failed" in str(m.get("content")) for m in s.history)


def test_rollback_restores_crlf_bytes_and_removes_new_files(in_tmp_cwd):
    original = b"x = 1\r\ny = 2\r\n"
    (in_tmp_cwd / "a.py").write_bytes(original)
    s = _session([
        _calls(("w1", "write_file", json.dumps({"path": "a.py", "content": "def (:\n"})),
               ("w2", "write_file", json.dumps({"path": "new.py", "content": "x = 1\n"}))),
        [_chunk(content="ok")],
    ])
    core.agent_loop(s)
    assert (in_tmp_cwd / "a.py").read_bytes() == original
    assert not (in_tmp_cwd / "new.py").exists()
    assert any("Validation failed" in str(m.get("content")) for m in s.history)


def test_validation_retries_reset_after_success(in_tmp_cwd):
    streams = [_write("f.py", f"def (: {i}\n", f"b{i}") for i in range(3)]
    streams.append(_write("f.py", "x = 1\n", "good"))
    streams.append(_write("f.py", "def (: again\n", "b9"))
    streams.append([_chunk(content="finished")])
    s = _session(streams)
    assert core.agent_loop(s) == "finished"


def test_python_check_is_in_process_and_writes_no_pycache(in_tmp_cwd):
    (in_tmp_cwd / "ok.py").write_text("x = 1\n")
    (in_tmp_cwd / "bad.py").write_text("def (:\n")
    assert validation.validate_file("ok.py") is None
    assert "Syntax error in bad.py" in validation.validate_file("bad.py")
    assert not (in_tmp_cwd / "__pycache__").exists()


def test_html_css_are_not_linted(in_tmp_cwd):
    (in_tmp_cwd / "a.css").write_text("a { color: }")
    (in_tmp_cwd / "a.html").write_text("<p>")
    assert validation.validate_files(["a.css", "a.html"]) is None


def test_unavailable_or_hung_checker_is_not_a_syntax_error(monkeypatch):
    assert validation._run(["definitely-not-a-real-checker-xyz"]) is None
    monkeypatch.setattr(validation, "CHECK_TIMEOUT", 0.2)
    assert validation._run(["sleep", "5"]) is None


@pytest.mark.skipif(not shutil.which("gcc"), reason="gcc not installed")
def test_c_check_finds_local_includes_and_ignores_missing_ones(in_tmp_cwd):
    sub = in_tmp_cwd / "src"
    sub.mkdir()
    (sub / "local.h").write_text("int f(void);\n")
    (sub / "a.c").write_text('#include "local.h"\nint g(void) { return f(); }\n')
    (sub / "b.c").write_text('#include "nowhere_zz.h"\nint g(void) { return 1; }\n')
    (sub / "c.c").write_text("int g(void) { return 1 }\n")
    assert validation.validate_file(str(sub / "a.c")) is None
    assert validation.validate_file(str(sub / "b.c")) is None
    assert validation.validate_file(str(sub / "c.c")) is not None


# ── 13: ollama context window prefers num_ctx ───────────────────────

@pytest.mark.parametrize("params,expected", [
    ("temperature 0.7\nnum_ctx                        32768\n", 32768),
    ("temperature 0.7\n", 131072),
])
def test_ollama_context_window_prefers_num_ctx(monkeypatch, params, expected):
    payload = {"parameters": params, "model_info": {"llama.context_length": 131072}}

    class FakeClient:
        def __init__(self, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def post(self, *_a, **_kw):
            return SimpleNamespace(json=lambda: payload)

    monkeypatch.setattr(providers.httpx, "Client", FakeClient)
    s = _session([])
    assert providers.PROVIDERS["ollama"].fetch_context_window(s) == expected
