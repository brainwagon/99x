"""Tests for the slash-command dispatch table."""

from agent99x.providers import PROVIDERS
from agent99x import core
from agent99x.commands import CommandResult, handle_slash_command


def test_non_slash_returns_unhandled(session):
    result = handle_slash_command(session, "hello")
    assert result.handled is False


def test_unknown_command_returns_error(session):
    result = handle_slash_command(session, "/bogus")
    assert result.handled is True
    assert "Unknown command" in result.message
    assert "/bogus" in result.message


def test_help_lists_known_commands(session):
    result = handle_slash_command(session, "/help")
    assert result.handled is True
    for cmd in ("clear", "help", "plan", "compact", "todos", "provider", "model", "host"):
        assert f"/{cmd}" in result.message


def test_clear_signals_history_reset(session):
    result = handle_slash_command(session, "/clear")
    assert result.clear_history is True


def test_exit_signals_app_exit(session):
    result = handle_slash_command(session, "/exit")
    assert result.exit_app is True


def test_compact_signals_compact(session):
    result = handle_slash_command(session, "/compact")
    assert result.compact is True


def test_todos_empty(in_tmp_cwd, session):
    result = handle_slash_command(session, "/todos")
    assert "(no todos)" in result.message


def test_command_argument_is_passed_through(in_tmp_cwd, monkeypatch, session):
    from agent99x import cli, commands

    captured = {}

    def fake_handler(sess, arg):
        captured["arg"] = arg
        return CommandResult(message="ok")

    monkeypatch.setitem(commands._COMMANDS, "spy", (fake_handler, "test"))
    handle_slash_command(session, "/spy hello world")
    assert captured["arg"] == "hello world"


def test_command_name_is_lowercased(in_tmp_cwd, monkeypatch, session):
    from agent99x import cli, commands

    captured = {}

    def fake_handler(sess, arg):
        captured["called"] = True
        return CommandResult(message="ok")

    monkeypatch.setitem(commands._COMMANDS, "spy", (fake_handler, "test"))
    handle_slash_command(session, "/SPY")
    assert captured.get("called") is True


def test_plan_command_toggles(session):
    session.plan_mode = False
    result = handle_slash_command(session, "/plan")
    assert "ON" in result.message
    assert session.plan_mode is True
    result = handle_slash_command(session, "/plan")
    assert "OFF" in result.message
    assert session.plan_mode is False


def test_provider_no_args_shows_current_and_known(session):
    result = handle_slash_command(session, "/provider")
    assert result.handled is True
    assert "ollama" in result.message
    assert "openrouter" in result.message


def test_provider_unknown_name(session):
    result = handle_slash_command(session, "/provider bogus-provider")
    assert "Unknown provider" in result.message
    assert session.pending_input_handler is None


def test_provider_no_cached_lists_models(session, monkeypatch):
    monkeypatch.setattr(
        "agent99x.providers.list_models_for",
        lambda name, host=None, timeout=30.0: ["only/model"],
    )
    result = handle_slash_command(session, "/provider openrouter")
    assert "only/model" in result.message
    assert session.pending_input_handler is not None


def test_provider_pick_by_index(session, monkeypatch):
    monkeypatch.setattr(
        "agent99x.providers.list_models_for",
        lambda name, host=None, timeout=30.0: ["a/one", "b/two", "c/three"],
    )
    switched = {}

    def fake_setup(s, name, model, host=None):
        switched["name"] = name
        switched["model"] = model
        s.provider = PROVIDERS[name]
        s.model = model

    monkeypatch.setattr("agent99x.providers.setup_provider", fake_setup)
    monkeypatch.setattr("agent99x.commands.save_config", lambda s: None)

    handle_slash_command(session, "/provider openrouter")
    result = handle_slash_command(session, "2")
    assert switched == {"name": "openrouter", "model": "b/two"}
    assert session.provider_models["openrouter"] == "b/two"
    assert session.pending_input_handler is None
    assert "b/two" in result.message


def test_provider_pick_out_of_range(session, monkeypatch):
    monkeypatch.setattr(
        "agent99x.providers.list_models_for",
        lambda name, host=None, timeout=30.0: ["a/one", "b/two"],
    )
    handle_slash_command(session, "/provider openrouter")
    result = handle_slash_command(session, "9")
    assert "Cancelled" in result.message
    assert session.pending_input_handler is None


def test_provider_pick_non_numeric(session, monkeypatch):
    monkeypatch.setattr(
        "agent99x.providers.list_models_for",
        lambda name, host=None, timeout=30.0: ["a/one", "b/two"],
    )
    handle_slash_command(session, "/provider openrouter")
    result = handle_slash_command(session, "garbage")
    assert "Cancelled" in result.message
    assert session.pending_input_handler is None


def test_pending_handler_cancelled_by_new_slash_command(session, monkeypatch):
    session.provider_models["openrouter"] = "openai/gpt-4o"
    monkeypatch.setattr(
        "agent99x.providers.list_models_for",
        lambda name, host=None, timeout=30.0: ["openai/gpt-4o"],
    )
    handle_slash_command(session, "/provider openrouter")
    assert session.pending_input_handler is not None
    result = handle_slash_command(session, "/help")
    assert session.pending_input_handler is None
    assert "/help" in result.message


def test_cancel_clears_pending_handler(session, monkeypatch):
    session.provider_models["openrouter"] = "openai/gpt-4o"
    monkeypatch.setattr(
        "agent99x.providers.list_models_for",
        lambda name, host=None, timeout=30.0: ["openai/gpt-4o"],
    )
    handle_slash_command(session, "/provider openrouter")
    assert session.pending_input_handler is not None
    result = handle_slash_command(session, "/cancel")
    assert session.pending_input_handler is None
    assert "Cancelled" in result.message


def test_model_lists_current_provider_models(session, monkeypatch):
    fetch_calls = []

    def fake_list(name, host=None, timeout=30.0):
        fetch_calls.append(name)
        return ["m1", "m2", "m3"]

    monkeypatch.setattr("agent99x.providers.list_models_for", fake_list)
    session.provider = PROVIDERS["openrouter"]
    result = handle_slash_command(session, "/model")
    assert fetch_calls == ["openrouter"]
    for m in ("m1", "m2", "m3"):
        assert m in result.message
    assert session.pending_input_handler is not None


def test_model_pick_persists(session, monkeypatch):
    monkeypatch.setattr(
        "agent99x.providers.list_models_for",
        lambda name, host=None, timeout=30.0: ["a/one", "b/two", "c/three"],
    )
    switched = {}

    def fake_setup(s, name, model, host=None):
        switched["name"] = name
        switched["model"] = model
        s.provider = PROVIDERS[name]
        s.model = model

    monkeypatch.setattr("agent99x.providers.setup_provider", fake_setup)
    monkeypatch.setattr("agent99x.commands.save_config", lambda s: None)

    session.provider = PROVIDERS["openrouter"]
    handle_slash_command(session, "/model")
    result = handle_slash_command(session, "2")
    assert switched == {"name": "openrouter", "model": "b/two"}
    assert session.provider_models["openrouter"] == "b/two"
    assert session.pending_input_handler is None
    assert "b/two" in result.message


def test_provider_no_saved_host_skips_host_prompt(session, monkeypatch):
    """First-time switch to a provider goes straight to the model step."""
    monkeypatch.setattr(
        "agent99x.providers.list_models_for",
        lambda name, host=None, timeout=30.0: ["m/only"],
    )
    result = handle_slash_command(session, "/provider openrouter")
    # No host prompt — message lists models directly.
    assert "Available models" in result.message
    assert "y/n" not in result.message.lower()


def test_host_no_args_with_no_saved_hosts(session):
    result = handle_slash_command(session, "/host")
    assert "No saved hosts" in result.message


def test_host_no_args_lists_saved(session):
    session.provider = PROVIDERS["ollama"]
    session.provider_hosts = {
        "ollama": "192.168.1.139:1234",
        "openrouter": None,
    }
    result = handle_slash_command(session, "/host")
    assert "ollama" in result.message
    assert "192.168.1.139:1234" in result.message
    assert "(current)" in result.message
    assert "openrouter" in result.message


def test_host_show_specific_provider(session):
    session.provider_hosts["ollama"] = "10.0.0.5:1234"
    result = handle_slash_command(session, "/host ollama")
    assert "10.0.0.5:1234" in result.message


def test_host_show_unknown_provider(session):
    result = handle_slash_command(session, "/host bogus")
    assert "Unknown provider" in result.message


def test_host_set_for_other_provider_persists_without_apply(session, monkeypatch):
    """Setting a host for a non-current provider must NOT call setup_provider —
    it just gets persisted for later restoration via /provider."""
    setup_calls = []

    def fake_setup(s, name, model, host=None):
        setup_calls.append((name, model, host))

    monkeypatch.setattr("agent99x.providers.setup_provider", fake_setup)
    monkeypatch.setattr("agent99x.commands.save_config", lambda s: None)

    session.provider = PROVIDERS["openrouter"]
    result = handle_slash_command(session, "/host ollama 192.168.1.139:1234")
    assert session.provider_hosts["ollama"] == "192.168.1.139:1234"
    assert setup_calls == []   # not the active provider, no client mutation
    assert "(applied now)" not in result.message


def test_host_set_for_current_provider_applies_and_persists(session, monkeypatch):
    setup_calls = []

    def fake_setup(s, name, model, host=None):
        setup_calls.append((name, model, host))
        s.host = host

    monkeypatch.setattr("agent99x.providers.setup_provider", fake_setup)
    monkeypatch.setattr("agent99x.commands.save_config", lambda s: None)

    session.provider = PROVIDERS["ollama"]
    session.model = "google/gemma-4-e4b"
    result = handle_slash_command(session, "/host ollama 10.0.0.99:1234")
    assert session.provider_hosts["ollama"] == "10.0.0.99:1234"
    assert setup_calls == [("ollama", "google/gemma-4-e4b", "10.0.0.99:1234")]
    assert session.host == "10.0.0.99:1234"
    assert "applied now" in result.message


def test_host_clear_for_other_provider(session, monkeypatch):
    monkeypatch.setattr("agent99x.commands.save_config", lambda s: None)
    session.provider = PROVIDERS["openrouter"]
    session.provider_hosts["ollama"] = "10.0.0.5:1234"
    result = handle_slash_command(session, "/host ollama clear")
    assert "ollama" not in session.provider_hosts
    assert "Cleared" in result.message
    assert "(applied now)" not in result.message


def test_host_clear_for_current_provider(session, monkeypatch):
    setup_calls = []

    def fake_setup(s, name, model, host=None):
        setup_calls.append((name, model, host))
        s.host = host

    monkeypatch.setattr("agent99x.providers.setup_provider", fake_setup)
    monkeypatch.setattr("agent99x.commands.save_config", lambda s: None)

    session.provider = PROVIDERS["ollama"]
    session.model = "google/gemma-4-e4b"
    session.host = "10.0.0.5:1234"
    session.provider_hosts["ollama"] = "10.0.0.5:1234"
    result = handle_slash_command(session, "/host ollama clear")
    assert "ollama" not in session.provider_hosts
    assert setup_calls == [("ollama", "google/gemma-4-e4b", None)]
    assert session.host is None
    assert "applied now" in result.message


