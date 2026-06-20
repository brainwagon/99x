"""Behavior tests for the config codec (one field list, save = inverse of load).

The codec is the single source of truth for which fields persist to
config.json. ``to_dict`` and ``from_dict`` are inverse views off that list,
so save and load can no longer drift apart.
"""

from agent99x import config_codec
from agent99x.providers import PROVIDERS
from agent99x.session import SessionConfig


def _full_session():
    """A session with every persisted field set to a non-default value."""
    return SessionConfig(
        provider=PROVIDERS["ollama"],
        model="some-model",
        host="http://example:1234",
        effort="high",
        show_thinking="terse",
        autocompact_threshold=0.5,
        call_budget=7,
        provider_models={"ollama": "some-model"},
        provider_hosts={"ollama": "http://example:1234"},
    )


def test_roundtrip_preserves_all_fields():
    src = _full_session()
    dst = SessionConfig()
    config_codec.from_dict(dst, config_codec.to_dict(src))
    assert dst.provider is src.provider
    assert dst.model == src.model
    assert dst.host == src.host
    assert dst.effort == src.effort
    assert dst.show_thinking == src.show_thinking
    assert dst.autocompact_threshold == src.autocompact_threshold
    assert dst.call_budget == src.call_budget
    assert dst.provider_models == src.provider_models
    assert dst.provider_hosts == src.provider_hosts


def test_provider_encodes_to_name_and_decodes_by_lookup():
    src = SessionConfig(provider=PROVIDERS["ollama"])
    data = config_codec.to_dict(src)
    assert data["provider"] == "ollama"

    dst = SessionConfig()
    config_codec.from_dict(dst, data)
    assert dst.provider is PROVIDERS["ollama"]


def test_unknown_provider_leaves_session_provider_unchanged():
    dst = SessionConfig(provider=PROVIDERS["ollama"])
    config_codec.from_dict(dst, {"provider": "no-such-provider"})
    assert dst.provider is PROVIDERS["ollama"]


def test_show_thinking_legacy_bool_decodes_to_string():
    dst = SessionConfig()
    config_codec.from_dict(dst, {"show_thinking": True})
    assert dst.show_thinking == "full"
    config_codec.from_dict(dst, {"show_thinking": False})
    assert dst.show_thinking == "off"


def test_missing_key_keeps_current_value():
    dst = SessionConfig(model="keep-me", effort="low")
    config_codec.from_dict(dst, {"model": "changed"})
    assert dst.model == "changed"
    assert dst.effort == "low"  # untouched: absent from the dict


def test_provider_dicts_are_copied_not_aliased():
    data = config_codec.to_dict(_full_session())
    dst = SessionConfig()
    config_codec.from_dict(dst, data)
    dst.provider_models["ollama"] = "mutated"
    assert data["provider_models"]["ollama"] == "some-model"


def test_effective_call_budget_prefers_explicit():
    s = SessionConfig(provider=PROVIDERS["ollama"], call_budget=3)
    assert s.effective_call_budget() == 3


def test_effective_call_budget_falls_back_to_provider_default():
    s = SessionConfig(provider=PROVIDERS["ollama"], call_budget=None)
    assert s.effective_call_budget() == PROVIDERS["ollama"].default_call_budget


def test_effective_call_budget_defaults_to_100_without_provider():
    s = SessionConfig(provider=None, call_budget=None)
    assert s.effective_call_budget() == 100
