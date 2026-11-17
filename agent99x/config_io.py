"""Session and config serialization."""

import json
import os
import tempfile
from typing import Optional

from agent99x import config_codec
from agent99x import logs
from agent99x.session import SessionConfig, SESSION_FILE, CONFIG_FILE


def load_config(session: SessionConfig) -> None:
    """Load saved config from AGENT_HOME/config.json into session."""
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        logs.log_warning(f"could not read {CONFIG_FILE} ({e}); using defaults")
        return
    config_codec.from_dict(session, data)
    if session.provider is not None:
        session.update_client()


def save_config(session: SessionConfig) -> None:
    """Save session config to AGENT_HOME/config.json."""
    _atomic_write(CONFIG_FILE, json.dumps(config_codec.to_dict(session), indent=2))


def _atomic_write(path: str, text: str) -> None:
    """Write ``text`` to ``path`` via a unique temp file in the same directory,
    so concurrent writers never share (and clobber) one temp name."""
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirpath, prefix=os.path.basename(path) + ".",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_session(session: SessionConfig, path: Optional[str] = None) -> None:
    """Persist the message history to the session's context file.

    Falls back to the legacy SESSION_FILE when no context has been selected.
    """
    if path is None:
        path = session.context_path or SESSION_FILE
    with session.session_lock:
        _atomic_write(path, json.dumps(session.history, indent=2))
