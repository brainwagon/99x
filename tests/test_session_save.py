import os
import json
import pytest
from agent99x import conversation
from agent99x.config_io import save_session, save_config, load_config
from agent99x.providers import PROVIDERS
from agent99x.session import SessionConfig, SESSION_FILE

def test_save_session_atomic(tmp_path, session):
    # Use a temporary directory for the session file
    session_file = tmp_path / ".99-session.json"
    
    session.history = [{"role": "user", "content": "hello"}]
    
    # Save session
    save_session(session, str(session_file))
    
    # Verify file exists and has content
    assert session_file.exists()
    with open(session_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == session.history
    
    # Verify .tmp file does not exist after completion
    assert not (tmp_path / ".99-session.json.tmp").exists()

def test_load_session_missing(tmp_path, session):
    session_file = tmp_path / "non-existent.json"
    assert conversation.load(session, str(session_file)) is False

def test_save_load_roundtrip(tmp_path, session):
    session_file = tmp_path / "session.json"
    session.history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi"}
    ]
    
    save_session(session, str(session_file))
    
    new_session = SessionConfig()
    conversation.load(new_session, str(session_file))
    
    # load_session might update the system message, so we check content
    assert len(new_session.history) == 2
    assert new_session.history[1]["content"] == "Hi"
    assert new_session.history[0]["role"] == "system"

def test_save_config_atomic(tmp_path, monkeypatch, session):
    config_file = tmp_path / "config.json"
    monkeypatch.setattr("agent99x.config_io.CONFIG_FILE", str(config_file))
    
    session.provider = PROVIDERS["openrouter"]
    session.model = "gpt-4"
    session.effort = "high"
    session.autocompact_threshold = 0.5
    session.host = "localhost"

    save_config(session)

    assert config_file.exists()
    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["provider"] == "openrouter"
    assert data["model"] == "gpt-4"
    assert data["effort"] == "high"
    assert data["autocompact"] == 0.5
    assert data["host"] == "localhost"

    assert not (tmp_path / "config.json.tmp").exists()


def test_config_roundtrip_preserves_provider_and_model(tmp_path, monkeypatch):
    """save_config + load_config must round-trip the per-provider
    selection so a restart picks the same provider/model without
    prompting the user."""
    config_file = tmp_path / "config.json"
    monkeypatch.setattr("agent99x.config_io.CONFIG_FILE", str(config_file))

    s1 = SessionConfig(provider=PROVIDERS["openrouter"], model="openai/gpt-4o")
    s1.provider_models = {
        "openrouter": "openai/gpt-4o",
        "ollama": "google/gemma-4-e4b",
    }
    s1.provider_hosts = {
        "openrouter": None,
        "ollama": "192.168.1.139:1234",
    }
    save_config(s1)

    s2 = SessionConfig()
    load_config(s2)
    assert s2.provider.name == "openrouter"
    assert s2.model == "openai/gpt-4o"
    assert s2.provider_models == {
        "openrouter": "openai/gpt-4o",
        "ollama": "google/gemma-4-e4b",
    }
    assert s2.provider_hosts == {
        "openrouter": None,
        "ollama": "192.168.1.139:1234",
    }
