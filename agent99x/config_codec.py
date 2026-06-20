"""Config codec: one field list, save = inverse of load.

The set of fields persisted to config.json lives here exactly once.
``to_dict`` and ``from_dict`` are inverse views off that list, so
``save_config`` and ``load_config`` cannot drift. The codec is pure: no
file IO, no client rebuild.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from agent99x import providers
from agent99x.session import SessionConfig


def _identity(value: Any) -> Any:
    return value


def _decode_thinking(value: Any) -> Any:
    """Back-compat: legacy configs stored show_thinking as a bool."""
    if isinstance(value, bool):
        return "full" if value else "off"
    return value


@dataclass(frozen=True)
class _Field:
    """One persisted field: json ``key`` ↔ SessionConfig ``attr``."""
    key: str
    attr: str
    encode: Callable[[Any], Any] = _identity   # attr value → json
    decode: Callable[[Any], Any] = _identity   # json value → attr value


# The single field list. Provider is handled separately (name ↔ object).
_FIELDS: List[_Field] = [
    _Field("model", "model"),
    _Field("effort", "effort"),
    _Field("autocompact", "autocompact_threshold"),
    _Field("host", "host"),
    _Field("call_budget", "call_budget"),
    _Field("provider_models", "provider_models", decode=dict),
    _Field("provider_hosts", "provider_hosts", decode=dict),
    _Field("show_thinking", "show_thinking", decode=_decode_thinking),
]


def to_dict(session: SessionConfig) -> Dict[str, Any]:
    """Encode a session's persisted config as a plain JSON-able dict."""
    out: Dict[str, Any] = {
        "provider": session.provider.name if session.provider else None,
    }
    for f in _FIELDS:
        out[f.key] = f.encode(getattr(session, f.attr))
    return out


def from_dict(session: SessionConfig, data: Dict[str, Any]) -> None:
    """Apply a persisted config dict onto a session in place.

    Keys absent from ``data`` leave the session's current value untouched.
    An unknown provider name is ignored (provider left unchanged).
    """
    name = data.get("provider")
    if name in providers.PROVIDERS:
        session.provider = providers.PROVIDERS[name]
    for f in _FIELDS:
        if f.key in data:
            setattr(session, f.attr, f.decode(data[f.key]))
