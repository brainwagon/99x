from typing import Any, Dict, Optional, Tuple

from agent99x import conversation, providers
from agent99x.session import SessionConfig


def do_compact(session: SessionConfig) -> Tuple[int, int, int, int]:
    """Compact conversation history into a summary.

    Returns (old_count, new_count, prompt_tokens, summary_tokens).
    """
    history = session.history
    old_count = len(history)
    messages = list(history)
    messages.append({
        "role": "user",
        "content": (
            "Summarize our conversation so far into a concise context block. "
            "Include: decisions made, files created or modified and their key contents, "
            "important facts discovered, and the current state of any work in progress. "
            "Omit redundant tool calls and exploratory dead ends."
        ),
    })
    compact_kwargs: Dict[str, Any] = dict(
        model=session.model, messages=messages,
        timeout=session.model_timeout, stream=False)
    if session.effort:
        compact_kwargs["reasoning_effort"] = session.effort
    response = session.client.chat.completions.create(**compact_kwargs)
    summary = response.choices[0].message.content or "(empty summary)"
    prompt_tok = response.usage.prompt_tokens if response.usage else 0
    summary_tok = response.usage.completion_tokens if response.usage else 0
    conversation.replace_with(
        session, f"[Conversation context — compacted]\n{summary}")
    return old_count, len(history), prompt_tok, summary_tok


def maybe_autocompact(session: SessionConfig, last_usage: int) -> Optional[Tuple[int, int, int, int]]:
    """Auto-compact if usage exceeds threshold. Returns (old, new, prompt_tok, summary_tok) or None."""
    cw = providers.ensure_context_window(session)
    if not (session.autocompact_threshold and cw and last_usage):
        return None
    if last_usage / cw < session.autocompact_threshold:
        return None
    if len(session.history) <= 2:
        return None
    return do_compact(session)
