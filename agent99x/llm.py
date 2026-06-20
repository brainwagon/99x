import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import openai

from agent99x.reasoning import ReasoningParser, ContentType
from agent99x.session import SessionConfig


def _fmt_connection_error(session: SessionConfig, exc: openai.APIConnectionError) -> str:
    """Return a user-friendly message for APIConnectionError, hinting at firewall issues."""
    cause = exc.__cause__
    cause_str = str(cause) if cause else str(exc)
    firewall_hints = ("connection refused", "no route to host", "network is unreachable",
                      "timed out", "connection timed out", "errno 111", "errno 113", "errno 101")
    is_likely_firewall = any(h in cause_str.lower() for h in firewall_hints)
    base = f"Cannot reach model server at {session.base_url}"
    if is_likely_firewall:
        return (f"{base} — connection refused or no route to host.\n"
                "Check that ollama (or your provider) is running on that address, "
                "the server is started, and any firewall allows the port.")
    return f"{base}: {cause_str}"


def stream_completion(
    session: SessionConfig,
    messages: List[Dict[str, Any]],
    active_tools: Optional[List[Dict[str, Any]]],
    cancel: Optional[threading.Event],
    on_token: Optional[Callable[[ContentType, str], None]] = None,
) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]], bool, int, Optional[str]]:
    """Stream a chat completion, polling cancel between chunks.

    Returns (content, reasoning, tool_calls, cancelled, prompt_tokens, finish_reason).
    """
    if session.client is None:
        raise RuntimeError(
            "No provider configured. Pass --provider on launch, or run /provider in the TUI."
        )
    if not session.model:
        provider_name = session.provider.name if session.provider else "the active provider"
        raise RuntimeError(
            f"No model selected for {provider_name}. "
            "Pass --model on launch, or run /model in the TUI to pick one. "
            "(A missing model is what makes the agent hang waiting on the server.)"
        )
    kwargs: Dict[str, Any] = dict(model=session.model, messages=messages,
                                  timeout=session.model_timeout,
                                  stream=True, stream_options={"include_usage": True})
    if active_tools is not None:
        kwargs["tools"] = active_tools
    if session.effort and not session.effort_suppressed:
        kwargs["reasoning_effort"] = session.effort
    try:
        stream = session.client.chat.completions.create(**kwargs)
    except openai.BadRequestError as exc:
        msg = str(exc)
        if session.effort and not session.effort_suppressed and (
            "reasoning_effort" in msg or "does not support thinking" in msg
        ):
            session.effort_suppressed = True
            kwargs.pop("reasoning_effort", None)
            stream = session.client.chat.completions.create(**kwargs)
        else:
            raise
    except openai.APIConnectionError as exc:
        raise RuntimeError(_fmt_connection_error(session, exc)) from None

    watcher_stop = threading.Event()

    def _watch() -> None:
        while not watcher_stop.is_set():
            if cancel is not None and cancel.is_set():
                try:
                    stream.close()
                except Exception:
                    pass
                return
            watcher_stop.wait(0.1)

    watcher = threading.Thread(target=_watch, daemon=True) if cancel is not None else None
    if watcher:
        watcher.start()

    parser = ReasoningParser()
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    tool_calls_acc: Dict[int, Dict[str, str]] = {}
    cancelled = False
    prompt_tokens = 0

    def handle_tokens(tokens: List[Tuple[ContentType, str]]) -> None:
        for ctype, text in tokens:
            if ctype == ContentType.CONTENT:
                content_parts.append(text)
            else:
                reasoning_parts.append(text)
            if on_token:
                on_token(ctype, text)

    finish_reason = None

    try:
        for chunk in stream:
            if cancel is not None and cancel.is_set():
                cancelled = True
                break
            usage = getattr(chunk, "usage", None)
            if usage and getattr(usage, "prompt_tokens", 0):
                prompt_tokens = usage.prompt_tokens
            if not chunk.choices:
                continue

            if getattr(chunk.choices[0], "finish_reason", None):
                finish_reason = chunk.choices[0].finish_reason

            delta = chunk.choices[0].delta
            if getattr(delta, "reasoning_content", None):
                handle_tokens(parser.feed_reasoning(delta.reasoning_content))
            if getattr(delta, "content", None):
                handle_tokens(parser.feed_content(delta.content))
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    slot = tool_calls_acc.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
        handle_tokens(parser.flush())
    except openai.APIConnectionError as exc:
        if cancel is not None and cancel.is_set():
            cancelled = True
        else:
            raise RuntimeError(_fmt_connection_error(session, exc)) from None
    except Exception:
        if cancel is not None and cancel.is_set():
            cancelled = True
        else:
            raise
    finally:
        watcher_stop.set()
        if watcher:
            watcher.join(timeout=0.5)
        try:
            stream.close()
        except Exception:
            pass

    content = "".join(content_parts) or None
    reasoning = "".join(reasoning_parts) or None
    tc_list: List[Dict[str, Any]] = [
        {"id": tc["id"], "type": "function",
         "function": {"name": tc["name"], "arguments": tc["arguments"]}}
        for _idx, tc in sorted(tool_calls_acc.items())
    ]
    return content, reasoning, tc_list, cancelled, prompt_tokens, finish_reason
