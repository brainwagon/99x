"""Regression tests for UI/CLI/prompt/config review findings.

Everything here is headless: network calls are mocked, and the TUI is
driven through Textual's ``App.run_test``.
"""

import asyncio
import json
import math
import os
import threading
import time

import pytest

from agent99x import cli, commands, config_codec, config_io, prompt, providers
from agent99x import tui
from agent99x.commands import handle_slash_command
from agent99x.hooks import HookEvent
from agent99x.providers import PROVIDERS
from agent99x.reasoning import ContentType
from agent99x.session import SessionConfig


@pytest.fixture
def no_save(monkeypatch):
    monkeypatch.setattr(commands, "save_config", lambda s: None)


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "home" / "config.json"
    path.parent.mkdir()
    monkeypatch.setattr(config_io, "CONFIG_FILE", str(path))
    return path


# ── 1. global memory comes from AGENT_HOME, not the cwd ─────────────

def test_global_memory_and_user_md_load_from_agent_home(in_tmp_cwd, monkeypatch):
    home = in_tmp_cwd / "home"
    home.mkdir()
    monkeypatch.setattr("agent99x.session.AGENT_HOME", str(home))
    (home / "MEMORY.md").write_text("GLOBAL-MEMORY-FACT")
    (home / "USER.md").write_text("GLOBAL-USER-FACT")
    (in_tmp_cwd / "MEMORY.md").write_text("CWD-MEMORY")
    (in_tmp_cwd / "USER.md").write_text("CWD-USER")

    content = prompt.build_system(SessionConfig())["content"]

    assert "GLOBAL-MEMORY-FACT" in content
    assert "GLOBAL-USER-FACT" in content
    assert "CWD-MEMORY" not in content
    assert "CWD-USER" not in content


def test_subagent_inherit_memory_reads_agent_home(in_tmp_cwd, monkeypatch):
    home = in_tmp_cwd / "home"
    home.mkdir()
    monkeypatch.setattr("agent99x.session.AGENT_HOME", str(home))
    (home / "MEMORY.md").write_text("GLOBAL-MEMORY-FACT")
    (in_tmp_cwd / "MEMORY.md").write_text("CWD-MEMORY")
    agent_dir = in_tmp_cwd / ".99x" / "agents" / "helper"
    agent_dir.mkdir(parents=True)
    (agent_dir / "AGENT.md").write_text(
        "---\ninherit_memory: true\nallowed_tools:\n  - read_file\n  - grep\n---\nBody")

    system, meta = prompt.build_agent_system(SessionConfig(), "helper")

    assert "GLOBAL-MEMORY-FACT" in system["content"]
    assert "CWD-MEMORY" not in system["content"]
    assert meta["allowed_tools"] == ["read_file", "grep"]


# ── 2. provider/host switches rebuild the client on the new host ────

def _client_url(session):
    return str(session.client.base_url)


def test_host_clear_rebuilds_client_without_old_host(session, no_save):
    session.host = "10.9.9.9:1234"
    session.provider_hosts["ollama"] = "10.9.9.9:1234"
    session.update_client()
    assert "10.9.9.9" in _client_url(session)

    handle_slash_command(session, "/host ollama clear")

    assert session.host is None
    assert "10.9.9.9" not in _client_url(session)
    assert "localhost:11434" in _client_url(session)


def test_switch_provider_without_host_drops_old_host(session, no_save, monkeypatch):
    session.host = "10.9.9.9:1234"
    session.update_client()
    session.context_window = 999
    monkeypatch.setattr(providers, "list_models_for",
                        lambda name, host=None, timeout=30.0: ["a/one"])
    handle_slash_command(session, "/provider openrouter")
    handle_slash_command(session, "1")

    assert session.provider.name == "openrouter"
    assert session.host is None
    assert "openrouter.ai" in _client_url(session)
    assert session.context_window == 0 and not session.context_fetched


def test_host_change_without_model_makes_no_network_call(session, no_save, monkeypatch):
    session.model = None
    monkeypatch.setattr(providers, "list_models",
                        lambda s: pytest.fail("must not list models"))
    handle_slash_command(session, "/host ollama 10.0.0.7:1")
    assert "10.0.0.7" in _client_url(session)


# ── 3. non-streamed final text is shown exactly once ─────────────────

def test_reply_tracker_detects_unstreamed_text():
    t = cli.ReplyTracker()
    t.on_model_request()
    t.on_token(ContentType.THINKING, "hmm")
    t.on_token(ContentType.CONTENT, "Hello ")
    t.on_token(ContentType.CONTENT, "world")
    assert t.unstreamed("Hello world") is None
    assert t.unstreamed("(max iterations reached)") == "(max iterations reached)"
    assert t.unstreamed("(cancelled)") is None
    t.on_model_request()
    assert t.unstreamed("Hello world") == "Hello world"


def _fake_loop(stream_text, reply):
    def loop(session, on_model_request=None, on_token=None, **_kw):
        if on_model_request:
            on_model_request()
        if stream_text and on_token:
            on_token(ContentType.CONTENT, stream_text)
        return reply
    return loop


def test_oneshot_prints_unstreamed_reply(session, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_mcp_start", lambda s: None)
    monkeypatch.setattr(cli.core, "agent_loop",
                        _fake_loop("partial", "(global call budget of 3 exhausted)"))
    cli._run_oneshot(session)
    out = capsys.readouterr().out
    assert "(global call budget of 3 exhausted)" in out


def test_oneshot_does_not_double_print_streamed_reply(session, monkeypatch, capsys):
    monkeypatch.setattr(cli, "_mcp_start", lambda s: None)
    monkeypatch.setattr(cli.core, "agent_loop", _fake_loop("Hello there", "Hello there"))
    cli._run_oneshot(session)
    assert capsys.readouterr().out.count("Hello there") == 1


# ── 4. history/client-mutating commands are refused while busy ──────

@pytest.mark.parametrize("text", [
    "/clear", "/new", "/compact", "/model", "/provider ollama",
    "/host ollama 1.2.3.4:5", "/host ollama clear",
])
def test_mutating_commands_blocked_while_busy(session, no_save, text, monkeypatch):
    monkeypatch.setattr(providers, "list_models_for",
                        lambda *a, **k: pytest.fail("must not run"))
    result = handle_slash_command(session, text, busy=True)
    assert "while an agent is running" in result.message
    assert not result.clear_history and not result.compact
    assert result.background is None


@pytest.mark.parametrize("text", ["/help", "/provider", "/host", "/budget", "/todos"])
def test_read_only_commands_allowed_while_busy(session, text, in_tmp_cwd):
    result = handle_slash_command(session, text, busy=True)
    assert "while an agent is running" not in result.message


def test_pending_pick_refused_while_busy(session, no_save, monkeypatch):
    monkeypatch.setattr(providers, "list_models_for",
                        lambda name, host=None, timeout=30.0: ["a/one", "b/two"])
    handle_slash_command(session, "/model")
    result = handle_slash_command(session, "2", busy=True)
    assert "running" in result.message
    assert session.model == "test-model"
    assert session.pending_input_handler is not None  # still armed


# ── 5. --trace-hooks in the TUI writes to a file ────────────────────

def test_tui_trace_goes_to_file(in_tmp_cwd, capsys):
    session = SessionConfig()
    session.shared_state["_trace_hooks"] = True
    path = os.path.join(".99x", "hook-trace.log")
    hooks = cli._cli_hooks(session, None, trace_path=path)
    hooks.emit(HookEvent("tool.started", "abcdef123456", None, time.time(),
                         {"tool": "grep"}))
    text = (in_tmp_cwd / ".99x" / "hook-trace.log").read_text()
    assert "tool.started" in text and "tool=grep" in text
    assert capsys.readouterr().err == ""
    assert session.shared_state["hook_trace_path"] == os.path.abspath(path)


# ── 6/8. 99xsh needs a prompt and is stateless ──────────────────────

def test_99xsh_without_prompt_is_usage_error(config_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.init_from_argv(SessionConfig(), [], prog="99xsh", require_prompt=True)
    assert exc.value.code == 2
    assert "usage: 99xsh" in capsys.readouterr().err


def _write_cfg(path, **data):
    path.write_text(json.dumps(data))


def test_oneshot_does_not_save_config_or_context(config_path, in_tmp_cwd):
    _write_cfg(config_path, provider="ollama", model="m1", effort=None)
    before = config_path.read_text()
    session = SessionConfig()
    assert cli.init_from_argv(session, ["--model", "m2", "--effort", "high", "hi"])
    assert config_path.read_text() == before
    assert session.context_path is None
    assert session.model == "m2" and session.effort == "high"


def test_oneshot_with_context_flag_keeps_context_file(config_path, in_tmp_cwd):
    _write_cfg(config_path, provider="ollama", model="m1")
    session = SessionConfig()
    cli.init_from_argv(session, ["-c", "work", "hi"])
    assert session.context_path.endswith(os.path.join("contexts", "work.json"))


def test_effort_none_clears_saved_effort(config_path, in_tmp_cwd):
    _write_cfg(config_path, provider="ollama", model="m1", effort="high")
    session = SessionConfig()
    cli.init_from_argv(session, ["--effort", "none"])
    assert session.effort is None
    assert json.loads(config_path.read_text())["effort"] is None


# ── 7. --provider restores that provider's remembered model/host ────

def test_provider_flag_uses_remembered_model_and_host(config_path, in_tmp_cwd, monkeypatch):
    _write_cfg(config_path, provider="ollama", model="local", host=None,
               provider_models={"openrouter": "or/remembered"},
               provider_hosts={"openrouter": None})
    monkeypatch.setattr(providers, "list_models",
                        lambda s: pytest.fail("remembered model must be used"))
    session = SessionConfig()
    cli.init_from_argv(session, ["--provider", "openrouter"])
    assert session.model == "or/remembered"
    saved = json.loads(config_path.read_text())
    assert saved["model"] == "or/remembered"


def test_provider_flag_explicit_model_and_host_override(config_path, in_tmp_cwd):
    _write_cfg(config_path, provider_models={"ollama": "old"},
               provider_hosts={"ollama": "1.1.1.1:1"})
    session = SessionConfig()
    cli.init_from_argv(session, ["--provider", "ollama", "--model", "new",
                                 "--host", "2.2.2.2:2"])
    assert session.model == "new"
    assert "2.2.2.2" in str(session.client.base_url)


# ── 9. skill pseudo-commands send an explicit instruction ───────────

def test_skill_pseudo_command_inlines_skill(session, in_tmp_cwd):
    d = in_tmp_cwd / ".99x" / "skills" / "demo"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\ndescription: d\n---\n# Demo\nStep one.\n")
    result = handle_slash_command(session, "/demo Paris")
    assert result.handled is False
    assert "`demo` skill" in result.send_text
    assert "Step one." in result.send_text
    assert "Arguments: Paris" in result.send_text


# ── 10. frontmatter parser ──────────────────────────────────────────

def test_frontmatter_block_scalars_lists_and_continuations():
    meta, body = prompt.parse_frontmatter(
        "---\n"
        "description: >-\n"
        "  Folded text\n"
        "  with key: value inside\n"
        "notes: |\n"
        "  line1\n"
        "  line2\n"
        "keep: |+\n"
        "  k\n"
        "\n"
        "summary: starts here\n"
        "  and: continues\n"
        "allowed_tools:\n"
        "  - read_file\n"
        "  - grep\n"
        "inline: [a, \"b, c\"]\n"
        "quoted: 'it''s'\n"
        "---\nBody\n")
    assert meta["description"] == "Folded text with key: value inside"
    assert meta["notes"] == "line1\nline2\n"
    assert meta["keep"] == "k\n\n"
    assert meta["summary"] == "starts here and: continues"
    assert meta["allowed_tools"] == ["read_file", "grep"]
    assert meta["inline"] == ["a", "b, c"]
    assert meta["quoted"] == "it's"
    assert "and" not in meta and "with key" not in meta
    assert body == "Body\n"


# ── 11. budget: unlimited is distinct from provider default ─────────

def test_budget_shows_effective_default_before_first_run(session):
    session.provider = PROVIDERS["openrouter"]
    msg = handle_slash_command(session, "/budget").message
    assert "0/100" in msg and "provider default" in msg


def test_budget_unlimited_survives_provider_default(session, no_save):
    session.provider = PROVIDERS["openrouter"]
    handle_slash_command(session, "/budget unlimited")
    assert session.call_budget == math.inf
    assert session.effective_call_budget() == math.inf
    assert "unlimited" in handle_slash_command(session, "/budget").message
    data = config_codec.to_dict(session)
    assert data["call_budget"] == "unlimited"
    json.dumps(data, allow_nan=False)
    fresh = SessionConfig()
    config_codec.from_dict(fresh, data)
    assert fresh.call_budget == math.inf
    handle_slash_command(session, "/budget default")
    assert session.call_budget is None
    assert session.shared_state["call_budget"] == 100


# ── 12. bare slash ──────────────────────────────────────────────────

@pytest.mark.parametrize("text", ["/", "/ ", "/ x"])
def test_bare_slash_is_helpful(session, text):
    assert "/help" in handle_slash_command(session, text).message


# ── 14. config validation and atomic writes ─────────────────────────

@pytest.mark.parametrize("raw", [
    "[]", '"x"', '{"provider_models": null, "provider_hosts": 3}',
    '{"call_budget": "lots", "autocompact": "high", "effort": 5, "model": 7}',
    '{"show_thinking": "loud", "provider": 12}',
])
def test_wrong_shape_config_falls_back_to_defaults(config_path, raw):
    config_path.write_text(raw)
    session = SessionConfig()
    config_io.load_config(session)
    default = SessionConfig()
    assert session.provider_models == {} and session.provider_hosts == {}
    assert session.call_budget == default.call_budget
    assert session.autocompact_threshold == default.autocompact_threshold
    assert session.show_thinking == default.show_thinking
    assert session.model is None and session.effort is None


def test_concurrent_config_saves_do_not_collide(config_path):
    session = SessionConfig(provider=PROVIDERS["ollama"], model="m")
    errors = []

    def save():
        try:
            for _ in range(30):
                config_io.save_config(session)
        except Exception as e:  # pragma: no cover - the regression
            errors.append(e)

    threads = [threading.Thread(target=save) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert json.loads(config_path.read_text())["model"] == "m"
    assert os.listdir(config_path.parent) == ["config.json"]


# ── 17. /plan lists every plan-blocked tool ─────────────────────────

def test_plan_message_lists_blocked_tools(session):
    msg = handle_slash_command(session, "/plan").message
    for name in prompt.plan_blocked_tools():
        assert name in msg
    assert "edit_file" in msg and "write_file" in msg
    session.plan_mode = True
    content = prompt.build_system(session)["content"]
    assert "edit_file" in content.split("PLAN MODE")[1]


# ── TUI (headless) ──────────────────────────────────────────────────

@pytest.fixture
def tui_env(in_tmp_cwd, monkeypatch):
    saved = []
    monkeypatch.setattr(cli, "save_session", lambda s: saved.append(list(s.history)))
    monkeypatch.setattr(tui.compaction, "maybe_autocompact", lambda *a: None)
    monkeypatch.setattr(commands, "save_config", lambda s: None)
    return saved


def _log_text(app):
    log = app.query_one(tui.RichLog)
    return "\n".join("".join(seg.text for seg in line) for line in log.lines)


async def _wait_idle(app, pilot, timeout=3.0):
    deadline = time.monotonic() + timeout
    while app._any_busy() and time.monotonic() < deadline:
        await pilot.pause(0.02)
    await pilot.pause(0.05)


async def _submit(app, pilot, text):
    app.query_one(tui.Input).value = text
    await pilot.press("enter")
    await pilot.pause(0.02)


def test_tui_thinking_collapse_and_toggle(session, tui_env, monkeypatch):
    session.show_thinking = "full"

    def loop(s, on_model_request=None, on_token=None, **_kw):
        on_model_request()
        on_token(ContentType.THINKING, "pondering deeply here")
        on_token(ContentType.CONTENT, "the answer")
        return "the answer"

    monkeypatch.setattr(tui.core, "agent_loop", loop)

    async def run():
        app = tui.AgentApp(session)
        async with app.run_test() as pilot:
            await _submit(app, pilot, "question")
            await _wait_idle(app, pilot)
            assert len(app._thinking_entries) == 1
            assert "Thinking (3 words)" in _log_text(app)
            await pilot.press("ctrl+t")
            await pilot.pause(0.05)
            assert app._thinking_expanded
            assert "pondering deeply here" in _log_text(app)

    asyncio.run(run())


def test_tui_shows_unstreamed_reply(session, tui_env, monkeypatch):
    monkeypatch.setattr(tui.core, "agent_loop",
                        _fake_loop("", "(max iterations reached)"))

    async def run():
        app = tui.AgentApp(session)
        async with app.run_test() as pilot:
            await _submit(app, pilot, "go")
            await _wait_idle(app, pilot)
            assert _log_text(app).count("(max iterations reached)") == 1

    asyncio.run(run())


def test_tui_blocks_clear_while_busy_and_lists_models_off_ui_thread(
        session, tui_env, monkeypatch):
    release = threading.Event()
    seen_threads = []

    def loop(s, cancel=None, on_model_request=None, **_kw):
        release.wait(3)
        return "(cancelled)" if cancel.is_set() else "ok"

    def slow_list(name, host=None, timeout=30.0):
        seen_threads.append(threading.current_thread() is threading.main_thread())
        time.sleep(0.1)
        return ["m/one"]

    monkeypatch.setattr(tui.core, "agent_loop", loop)
    monkeypatch.setattr(providers, "list_models_for", slow_list)

    async def run():
        app = tui.AgentApp(session)
        async with app.run_test() as pilot:
            await _submit(app, pilot, "work")
            history_len = len(session.history)
            await _submit(app, pilot, "/clear")
            assert len(session.history) == history_len
            assert "while an agent is running" in _log_text(app)
            release.set()
            await _wait_idle(app, pilot)

            await _submit(app, pilot, "/model")
            await _wait_idle(app, pilot)
            assert seen_threads == [False]
            assert "m/one" in _log_text(app)
            assert session.pending_input_handler is not None

    asyncio.run(run())


def test_tui_exit_cancels_running_agent_before_saving(session, tui_env, monkeypatch):
    cancels = []

    def loop(s, cancel=None, **_kw):
        cancels.append(cancel)
        cancel.wait(5)
        s.history.append({"role": "assistant", "content": "late"})
        return "(cancelled)"

    monkeypatch.setattr(tui.core, "agent_loop", loop)

    async def run():
        app = tui.AgentApp(session)
        async with app.run_test() as pilot:
            await _submit(app, pilot, "work")
            await pilot.pause(0.05)
            await _submit(app, pilot, "/exit")
            await pilot.pause(0.3)

    asyncio.run(run())
    assert cancels and cancels[0].is_set()
    assert tui_env, "history must be saved on exit"
    assert tui_env[-1][-1]["content"] == "late"


def test_tui_esc_discards_compaction(session, tui_env, monkeypatch):
    release = threading.Event()

    def slow_compact(s):
        release.wait(3)
        s.history[:] = s.history[:1]
        return 5, 1, 10, 1

    monkeypatch.setattr(tui.compaction, "do_compact", slow_compact)

    async def run():
        app = tui.AgentApp(session)
        async with app.run_test() as pilot:
            session.history.extend([{"role": "user", "content": "a"},
                                    {"role": "assistant", "content": "b"}])
            before = list(session.history)
            await _submit(app, pilot, "/compact")
            assert app._any_busy()
            await pilot.press("escape")
            await pilot.pause(0.02)
            assert not app._any_busy()
            assert "compaction cancelled" in _log_text(app)
            release.set()
            await pilot.pause(0.2)
            assert session.history == before

    asyncio.run(run())


# ── subagent prompt: plan mode and the final (narrowed) tool list ───

def test_subagent_prompt_follows_parent_plan_mode_and_whitelist(in_tmp_cwd):
    agent_dir = in_tmp_cwd / ".99x" / "agents" / "narrow"
    agent_dir.mkdir(parents=True)
    (agent_dir / "AGENT.md").write_text(
        "---\nallowed_tools: [read_file, grep]\n---\nBody")
    parent = SessionConfig(plan_mode=True, allowed_tools=["read_file", "glob"])

    system, _meta = prompt.build_agent_system(parent, "narrow")
    content = system["content"]

    assert "PLAN MODE" in content
    tools_section = content.split("## Tools")[1].split("\n\n")[0]
    assert "read_file" in tools_section
    assert "grep" not in tools_section  # dropped: parent doesn't allow it
    assert prompt.effective_agent_tools(None, ["a"]) == ["a"]
    assert prompt.effective_agent_tools(["a", "b"], None) == ["a", "b"]
