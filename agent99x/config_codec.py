"""Config codec: one field list, save = inverse of load.

The set of fields persisted to config.json lives here exactly once.
``to_dict`` and ``from_dict`` are inverse views off that list, so
``save_config`` and ``load_config`` cannot drift. The codec is pure: no
file IO, no client rebuild.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from agent99x import logs
from agent99x import providers
from agent99x.session import SessionConfig

# config.json spelling of an explicitly unlimited call budget (math.inf in
# memory). ``null`` means "use the provider default", which is different.
UNLIMITED = "unlimited"


def _identity(value: Any) -> Any:
    return value


def _decode_thinking(value: Any) -> Any:
    """Back-compat: legacy configs stored show_thinking as a bool."""
    if isinstance(value, bool):
        return "full" if value else "off"
    if value not in ("off", "terse", "full"):
        raise ValueError("expected off|terse|full")
    return value


def _decode_optional_str(value: Any) -> Any:
    if value is not None and not isinstance(value, str):
        raise ValueError("expected a string or null")
    return value


def _decode_effort(value: Any) -> Any:
    if value not in (None, "low", "medium", "high"):
        raise ValueError("expected low|medium|high or null")
    return value


def _decode_fraction(value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a number or null")
    if not 0 <= value <= 1:
        raise ValueError("expected a value between 0 and 1")
    return float(value)


def _encode_budget(value: Any) -> Any:
    return UNLIMITED if value == math.inf else value


def _decode_budget(value: Any) -> Any:
    if value is None:
        return None
    if value == UNLIMITED or value == math.inf:
        return math.inf
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"expected a positive integer, {UNLIMITED!r}, or null")
    return value


def _decode_str_map(allow_none: bool) -> Callable[[Any], Dict[str, Any]]:
    def decode(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("expected an object")
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if isinstance(v, str) or (allow_none and v is None):
                out[str(k)] = v
            else:
                logs.log_warning(f"config.json: ignoring bad entry {k!r}: {v!r}")
        return out
    return decode


@dataclass(frozen=True)
class _Field:
    """One persisted field: json ``key`` ↔ SessionConfig ``attr``.

    ``decode`` validates as well as converts: it raises ValueError on a
    wrong-shaped value, and from_dict then keeps the session default.
    """
    key: str
    attr: str
    encode: Callable[[Any], Any] = _identity   # attr value → json
    decode: Callable[[Any], Any] = _identity   # json value → attr value


# The single field list. Provider is handled separately (name ↔ object).
_FIELDS: List[_Field] = [
    _Field("model", "model", decode=_decode_optional_str),
    _Field("effort", "effort", decode=_decode_effort),
    _Field("autocompact", "autocompact_threshold", decode=_decode_fraction),
    _Field("host", "host", decode=_decode_optional_str),
    _Field("call_budget", "call_budget", encode=_encode_budget, decode=_decode_budget),
    _Field("provider_models", "provider_models", decode=_decode_str_map(False)),
    _Field("provider_hosts", "provider_hosts", decode=_decode_str_map(True)),
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


def from_dict(session: SessionConfig, data: Any) -> None:
    """Apply a persisted config dict onto a session in place.

    Keys absent from ``data`` leave the session's current value untouched.
    An unknown provider name is ignored (provider left unchanged). A field
    with the wrong type or value is logged and skipped, as is a top-level
    value that is not an object, so a hand-edited config can't crash startup.
    """
    if not isinstance(data, dict):
        logs.log_warning("config.json is not a JSON object; using defaults")
        return
    name = data.get("provider")
    if isinstance(name, str) and name in providers.PROVIDERS:
        session.provider = providers.PROVIDERS[name]
    for f in _FIELDS:
        if f.key not in data:
            continue
        try:
            value = f.decode(data[f.key])
        except ValueError as e:
            logs.log_warning(
                f"config.json: ignoring {f.key}={data[f.key]!r} ({e})")
            continue
        setattr(session, f.attr, value)
