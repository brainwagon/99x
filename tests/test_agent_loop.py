"""Tests for agent_loop using a fake LM Studio client."""

import json

from agent99x import core


def test_agent_loop_returns_text_for_simple_completion(session, fake_lm):
    fake_lm.queue([
        fake_lm.chunk(content="hello there"),
        fake_lm.chunk(finish_reason="stop"),
        fake_lm.usage_chunk(prompt_tokens=42),
    ])
    session.history = [{"role": "user", "content": "hi"}]
    result = core.agent_loop(session)
    assert result == "hello there"
    assert session.history[-1] == {"role": "assistant", "content": "hello there"}


def test_agent_loop_reports_usage_via_callback(session, fake_lm):
    fake_lm.queue([
        fake_lm.chunk(content="ok"),
        fake_lm.usage_chunk(prompt_tokens=99),
    ])
    seen = []
    session.history = [{"role": "user", "content": "hi"}]
    core.agent_loop(session, on_usage=seen.append)
    assert seen == [99]


def test_agent_loop_dispatches_tool_call(session, fake_lm, in_tmp_cwd):
    target = in_tmp_cwd / "data.txt"
    target.write_text("file body", encoding="utf-8")

    args_json = json.dumps({"path": str(target)})
    fake_lm.queue([
        fake_lm.chunk(tool_calls=[
            fake_lm.tool_call_delta(0, id="call_1", name="read_file", arguments=args_json),
        ]),
        fake_lm.chunk(finish_reason="tool_calls"),
    ])
    fake_lm.queue([
        fake_lm.chunk(content="done"),
        fake_lm.chunk(finish_reason="stop"),
    ])

    session.history = [{"role": "user", "content": "read it"}]
    result = core.agent_loop(session)

    assert result == "done"
    # Trace: user, assistant(tool_calls), tool, assistant(text).
    assert session.history[1]["role"] == "assistant"
    assert session.history[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert session.history[2]["role"] == "tool"
    tool_payload = json.loads(session.history[2]["content"])
    assert tool_payload["content"] == "file body"
    assert session.history[3] == {"role": "assistant", "content": "done"}


def test_agent_loop_streams_tool_arguments_across_chunks(session, fake_lm, in_tmp_cwd):
    """Tool call deltas may arrive split across multiple chunks; the loop
    should accumulate id/name/arguments before dispatch."""
    target = in_tmp_cwd / "x.txt"
    target.write_text("body", encoding="utf-8")

    fake_lm.queue([
        fake_lm.chunk(tool_calls=[
            fake_lm.tool_call_delta(0, id="call_1", name="read_file"),
        ]),
        fake_lm.chunk(tool_calls=[
            fake_lm.tool_call_delta(0, arguments='{"path": "'),
        ]),
        fake_lm.chunk(tool_calls=[
            fake_lm.tool_call_delta(0, arguments=f'{target}"' + "}"),
        ]),
        fake_lm.chunk(finish_reason="tool_calls"),
    ])
    fake_lm.queue([
        fake_lm.chunk(content="all done"),
    ])

    session.history = [{"role": "user", "content": "go"}]
    result = core.agent_loop(session)
    assert result == "all done"
    assert json.loads(session.history[2]["content"])["content"] == "body"


def test_agent_loop_dispatches_patch_tool(session, fake_lm, in_tmp_cwd):
    target = in_tmp_cwd / "test.txt"
    target.write_text("old line\n", encoding="utf-8")

    diff = """--- test.txt
+++ test.txt
@@ -1,1 +1,1 @@
-old line
+new line
"""
    args_json = json.dumps({"path": str(target), "diff": diff})
    fake_lm.queue([
        fake_lm.chunk(tool_calls=[
            fake_lm.tool_call_delta(0, id="call_1", name="patch", arguments=args_json),
        ]),
        fake_lm.chunk(finish_reason="tool_calls"),
    ])
    fake_lm.queue([
        fake_lm.chunk(content="patched"),
        fake_lm.chunk(finish_reason="stop"),
    ])

    session.history = [{"role": "user", "content": "patch it"}]
    result = core.agent_loop(session)

    assert result == "patched"
    assert target.read_text(encoding="utf-8") == "new line\n"
    tool_payload = json.loads(session.history[2]["content"])
    assert tool_payload["ok"] is True
