"""Tests for provider/host/model config initialization."""

import json
import os
import pytest

from agent99x import core
from agent99x import providers
from agent99x import cli, commands, session
from agent99x.session import SessionConfig


def _write_config(path, *, provider="ollama", model="test-model",
                  host="192.168.1.5:1234", effort=None, autocompact=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "provider": provider, "model": model,
            "host": host, "effort": effort, "autocompact": autocompact,
        }))


@pytest.fixture
def fake_provider_env(tmp_path, monkeypatch):
    """Set up a fake AGENT_HOME in tmp_path, with a config.json we control."""
    monkeypatch.setattr("agent99x.config_io.CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setattr(providers, "fetch_context_window", lambda s: 131072)
    return tmp_path


class TestProviderSwitchResetsModelAndHost:
    """When --provider is used, stale model/host from config must be dropped."""

    def test_provider_flag_resets_host(self, fake_provider_env, monkeypatch):
        """--provider openrouter → host is None (not stale LM Studio host)."""
        path = fake_provider_env / "config.json"
        _write_config(path, host="192.168.1.5:1234", model="some-old-model")

        monkeypatch.setattr(providers, "list_models",
                            lambda s: ["deepseek/deepseek-v4-pro"])

        session = SessionConfig()
        cli.init_from_argv(session, ["--provider", "openrouter"])
        assert session.provider.name == "openrouter"
        assert session.model == "deepseek/deepseek-v4-pro"
        # Host must be reset; OpenRouter's default base_url is used.
        assert session.base_url == "https://openrouter.ai/api/v1"

        # Config written must not contain the stale host.
        saved = json.loads(path.read_text())
        assert saved["host"] is None

    def test_provider_flag_preserves_explicit_host(self, fake_provider_env, monkeypatch):
        """--provider ollama --host 10.0.0.1:9999 preserves explicit host."""
        path = fake_provider_env / "config.json"
        _write_config(path, host="192.168.1.5:1234")

        monkeypatch.setattr(providers, "list_models", lambda s: ["my-model"])

        session = SessionConfig()
        cli.init_from_argv(session, ["--provider", "ollama", "--host", "10.0.0.1:9999"])

        assert session.base_url == "http://10.0.0.1:9999/v1"

        saved = json.loads(path.read_text())
        assert saved["host"] == "10.0.0.1:9999"

    def test_provider_flag_resets_model(self, fake_provider_env, monkeypatch):
        """--provider openrouter → model is None, auto-picked from list_models."""
        path = fake_provider_env / "config.json"
        _write_config(path, model="some-stale-local-model",
                      host="192.168.1.5:1234")

        monkeypatch.setattr(providers, "list_models", lambda s: ["fresh-model"])

        session = SessionConfig()
        cli.init_from_argv(session, ["--provider", "ollama"])

        # Model should be freshly picked, not stale.
        assert session.model == "fresh-model"

    def test_provider_flag_preserves_explicit_model(self, fake_provider_env, monkeypatch):
        """--provider ollama --model my-explicit keeps explicit model."""
        path = fake_provider_env / "config.json"
        _write_config(path, model="old-model", host="192.168.1.5:1234")

        monkeypatch.setattr(providers, "list_models", lambda s: ["my-explicit"])

        session = SessionConfig()
        cli.init_from_argv(session, ["--provider", "ollama", "--model", "my-explicit"])

        assert session.model == "my-explicit"


class TestNoProviderFlagUsesSavedConfig:
    """Without --provider, saved config values are applied as normal."""

    def test_saved_host_is_used(self, fake_provider_env, monkeypatch):
        """No --provider → saved host from config carries through."""
        path = fake_provider_env / "config.json"
        _write_config(path, host="10.0.0.99:4567", provider="ollama",
                      model="my-model")

        monkeypatch.setattr(providers, "list_models", lambda s: ["my-model"])

        session = SessionConfig()
        cli.init_from_argv(session, [])

        assert session.provider.name == "ollama"
        assert session.base_url == "http://10.0.0.99:4567/v1"
        assert session.model == "my-model"

    def test_saved_model_is_used(self, fake_provider_env, monkeypatch):
        """No --provider, no --model → saved model is used."""
        path = fake_provider_env / "config.json"
        _write_config(path, provider="ollama", model="saved-model",
                      host="192.168.1.5:1234")

        monkeypatch.setattr(providers, "list_models", lambda s: ["saved-model"])

        session = SessionConfig()
        cli.init_from_argv(session, [])

        assert session.model == "saved-model"

    def test_no_saved_provider_leaves_provider_unset(self, fake_provider_env, monkeypatch):
        """Absent config → no silent fallback; provider/model stay unset so the app can ask."""
        monkeypatch.setattr(providers, "list_models", lambda s: ["default-model"])

        session = SessionConfig()
        cli.init_from_argv(session, [])

        assert session.provider is None
        assert session.model is None
        assert not cli.provider_ready(session)


class TestSaveConfigWritesCorrectValues:
    """After init, saved config always reflects current state."""

    def test_after_switch_provider_saved_host_is_none(self, fake_provider_env, monkeypatch):
        """Switching to OpenRouter writes host=None to config."""
        path = fake_provider_env / "config.json"
        _write_config(path, host="192.168.1.5:1234", provider="ollama",
                      model="old-model")

        monkeypatch.setattr(providers, "list_models", lambda s: ["or-model"])

        session = SessionConfig()
        cli.init_from_argv(session, ["--provider", "openrouter"])

        saved = json.loads(path.read_text())
        assert saved["provider"] == "openrouter"
        assert saved["model"] == "or-model"
        assert saved["host"] is None

    def test_after_setting_host_saved_host_persists(self, fake_provider_env, monkeypatch):
        """Explicit --host is written to config."""
        path = fake_provider_env / "config.json"
        _write_config(path, provider="ollama", model="m", host=None)

        monkeypatch.setattr(providers, "list_models", lambda s: ["m"])

        session = SessionConfig()
        cli.init_from_argv(session, ["--host", "10.0.1.2:8888"])

        saved = json.loads(path.read_text())
        assert saved["host"] == "10.0.1.2:8888"


class TestModelFlagWithoutProvider:
    """--model without --provider should override saved config."""

    def test_model_flag_overrides_saved(self, fake_provider_env, monkeypatch):
        """--model explicit overrides saved model."""
        path = fake_provider_env / "config.json"
        _write_config(path, provider="ollama", model="saved-model", host=None)

        monkeypatch.setattr(providers, "list_models", lambda s: ["explicit-model"])

        session = SessionConfig()
        cli.init_from_argv(session, ["--model", "explicit-model"])

        assert session.model == "explicit-model"
