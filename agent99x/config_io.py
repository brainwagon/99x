"""Session and config serialization."""

import json
import os
from typing import Optional

from agent99x import config_codec
from agent99x.session import SessionConfig, SESSION_FILE, CONFIG_FILE


def load_config(session: SessionConfig) -> None:
    """Load saved config from AGENT_HOME/config.json into session."""
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return
    config_codec.from_dict(session, data)
    if session.provider is not None:
        session.update_client()


def save_config(session: SessionConfig) -> None:
    """Save session config to AGENT_HOME/config.json."""
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    tmp = CONFIG_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(config_codec.to_dict(session), indent=2))
    os.replace(tmp, CONFIG_FILE)


def save_session(session: SessionConfig, path: Optional[str] = None) -> None:
    """Persist the message history to the session's context file.

    Falls back to the legacy SESSION_FILE when no context has been selected.
    """
    if path is None:
        path = session.context_path or SESSION_FILE
    tmp = path + ".tmp"
    with session.session_lock:
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(session.history, indent=2))
        os.replace(tmp, path)
