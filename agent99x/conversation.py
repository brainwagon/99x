"""Session-turn lifecycle: the one place that owns session.history.

Every public function here keeps the invariant that history[0] is a
freshly-built system prompt. Callers append turns, clear, and compact
through this module rather than splicing session.history by hand.
"""

import json

from agent99x import logs
from agent99x.prompt import build_system, build_user_message
from agent99x.session import SESSION_FILE, SessionConfig

_CONTEXT_ACK = "Got it. I have the context from our previous work."


def start(session: SessionConfig) -> None:
    """Seed the history with a fresh system prompt, discarding any turns."""
    session.history = [build_system(session)]


def load(session: SessionConfig, path: str = SESSION_FILE) -> bool:
    """Load saved history into the session, rebuilding the system prompt."""
    try:
        with open(path, encoding="utf-8") as f:
            history = json.load(f)
    except FileNotFoundError:
        return False
    except json.JSONDecodeError as e:
        logs.log_warning(f"session file at {path} is corrupt ({e}); starting fresh")
        return False
    if not isinstance(history, list):
        logs.log_warning(f"session file at {path} is not a message list; starting fresh")
        return False
    if history:
        history[0] = build_system(session)
    session.history = repair_tool_pairs(history)
    return True


_MISSING_RESULT = json.dumps(
    {"error": "cancelled: no result was recorded for this tool call"})


def repair_tool_pairs(messages: list) -> list:
    """Return messages with every assistant tool call paired with a result.

    The chat API rejects a transcript where an assistant ``tool_calls``
    entry has no following ``tool`` message (or a ``tool`` message answers
    no preceding call); an interrupted run or an old saved context can
    leave one behind.  This inserts a placeholder error result for each
    unanswered call id and drops tool messages whose id has no call
    immediately before them, or that answer an id twice.  Returns a new
    list that reuses the original message dicts; ``messages`` is untouched.
    """
    out: list = []
    pending: list = []  # call ids of the latest assistant turn still unanswered

    def close_pending() -> None:
        for call_id in pending:
            out.append({"role": "tool", "tool_call_id": call_id,
                        "content": _MISSING_RESULT})
        pending.clear()

    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else None
        if role == "tool":
            call_id = msg.get("tool_call_id")
            if call_id in pending:
                pending.remove(call_id)
                out.append(msg)
            # else: orphan or duplicate result -- drop it
            continue
        close_pending()
        out.append(msg)
        if role == "assistant":
            for tc in msg.get("tool_calls") or ():
                call_id = tc.get("id") if isinstance(tc, dict) else None
                if call_id and call_id not in pending:
                    pending.append(call_id)
    close_pending()
    return out


def clear(session: SessionConfig) -> None:
    """Discard all turns, leaving only a fresh system prompt."""
    session.history = [build_system(session)]


def replace_with(session: SessionConfig, context_text: str) -> None:
    """Replace all turns with a fresh system prompt and a context handoff.

    Used by compaction: the prior conversation collapses into `context_text`
    (already labelled by the caller), followed by an assistant acknowledgement.
    """
    session.history[:] = [
        build_system(session),
        {"role": "user", "content": context_text},
        {"role": "assistant", "content": _CONTEXT_ACK},
    ]


def add_user(session: SessionConfig, text: str) -> tuple[list[str], list[str]]:
    """Append a user turn, refreshing the system prompt first."""
    refresh_system(session)
    user_msg, media_paths, media_errors = build_user_message(text)
    session.history.append(user_msg)
    return media_paths, media_errors


def truncate_to(session: SessionConfig, index: int) -> None:
    """Drop turns from `index` onward, rolling back to an earlier mark."""
    del session.history[index:]


def refresh_system(session: SessionConfig) -> None:
    """Rebuild history[0] so the system prompt reflects current state."""
    if session.history and session.history[0].get("role") == "system":
        session.history[0] = build_system(session)
    else:
        session.history.insert(0, build_system(session))
