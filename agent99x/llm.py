import json
import re
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import openai

from agent99x.reasoning import ReasoningParser, ContentType
from agent99x.session import SessionConfig

# Gemma 4 enables step-by-step thinking via a control token at the start of
# the system prompt (there is no API parameter for it; ollama accepts
# reasoning_effort for these models but ignores it).
THINK_TOKEN = "<|think|>"
_GEMMA4_RE = re.compile(r"gemma[-_]?4", re.IGNORECASE)
# Gemma 3 family (incl. 3n) emits spurious mid-word newlines when streaming;
# gemma 4 does not, and the workaround corrupts its numbered lists.
_GEMMA3_RE = re.compile(r"gemma[-_]?3", re.IGNORECASE)
_MINICPM_RE = re.compile(r"minicpm", re.IGNORECASE)


def wants_think_token(session: SessionConfig) -> bool:
    """True when the active model takes Gemma 4's <|think|> control token."""
    return bool(session.effort and session.model and _GEMMA4_RE.search(session.model))


def with_think_token(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return messages with <|think|> prepended to the system prompt.

    Copies the system message rather than mutating it so the token never
    leaks into the persisted session history.
    """
    if messages and messages[0].get("role") == "system":
        first = dict(messages[0])
        content = first.get("content") or ""
        if content.startswith(THINK_TOKEN):
            return messages
        first["content"] = THINK_TOKEN + content
        return [first] + messages[1:]
    return [{"role": "system", "content": THINK_TOKEN}] + list(messages)


def wants_minicpm_tools(session: SessionConfig) -> bool:
    """True when the active model is minicpm."""
    return bool(session.model and _MINICPM_RE.search(session.model))


def with_minicpm_tools(messages: List[Dict[str, Any]], active_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return messages with tool instructions injected into the system prompt for minicpm."""
    if not active_tools:
        return messages
    tools_str = json.dumps(active_tools, indent=2)
    prompt = (
        "\n\nYou have access to the following tools:\n"
        f"{tools_str}\n\n"
        "To use a tool, you MUST respond using this exact XML format:\n"
        "<function name=\"tool_name\">\n"
        "  <param name=\"param1\">value1</param>\n"
        "</function>\n"
    )
    if messages and messages[0].get("role") == "system":
        first = dict(messages[0])
        first["content"] = str(first.get("content", "")) + prompt
        return [first] + messages[1:]
    return [{"role": "system", "content": prompt}] + list(messages)


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


def _new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:8]


def _merge_tool_call_delta(tc: Any, acc: List[Dict[str, str]],
                           slot_for_index: Dict[Any, Dict[str, str]]) -> None:
    """Fold one streamed tool-call delta into the accumulated calls.

    Deltas are matched to a call by ``index``.  A delta carrying a new,
    different id for an index already in use starts a new call (some
    backends reuse index 0 for every call).  A delta with no index joins
    the most recent call unless it brings a new id.
    """
    index = getattr(tc, "index", None)
    call_id = getattr(tc, "id", None) or ""
    slot = slot_for_index.get(index)
    if slot is None and index is None and not call_id and acc:
        slot = acc[-1]
    if slot is not None and call_id and slot["id"] and slot["id"] != call_id:
        slot = None
    if slot is None:
        slot = {"id": "", "name": "", "arguments": ""}
        acc.append(slot)
        slot_for_index[index] = slot
    if call_id:
        slot["id"] = call_id
    fn = getattr(tc, "function", None)
    if fn:
        if getattr(fn, "name", None):
            slot["name"] += fn.name
        if getattr(fn, "arguments", None):
            slot["arguments"] += fn.arguments


def stream_completion(
    session: SessionConfig,
    messages: List[Dict[str, Any]],
    active_tools: Optional[List[Dict[str, Any]]],
    cancel: Optional[threading.Event],
    on_token: Optional[Callable[[ContentType, str], None]] = None,
) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]], bool, Tuple[int, int], Optional[str]]:
    """Stream a chat completion, polling cancel between chunks.

    Returns (content, reasoning, tool_calls, cancelled, (prompt_tokens, completion_tokens), finish_reason).
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
    if wants_think_token(session):
        messages = with_think_token(messages)
    if wants_minicpm_tools(session) and active_tools is not None:
        messages = with_minicpm_tools(messages, active_tools)
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

    parser = ReasoningParser(
        strip_midword_newlines=bool(session.model and _GEMMA3_RE.search(session.model)))
    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    # Calls in arrival order; slot_for_index maps a delta's index to its call.
    tool_calls_acc: List[Dict[str, str]] = []
    slot_for_index: Dict[Any, Dict[str, str]] = {}
    cancelled = False
    prompt_tokens = 0
    completion_tokens = 0

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
            if usage:
                if getattr(usage, "prompt_tokens", 0):
                    prompt_tokens = usage.prompt_tokens
                if getattr(usage, "completion_tokens", 0):
                    completion_tokens = usage.completion_tokens
            if not chunk.choices:
                continue

            if getattr(chunk.choices[0], "finish_reason", None):
                finish_reason = chunk.choices[0].finish_reason

            delta = chunk.choices[0].delta
            # deepseek-style backends use reasoning_content; ollama and
            # OpenRouter stream Gemma 4 (and others) as reasoning.
            reasoning_delta = (getattr(delta, "reasoning_content", None)
                               or getattr(delta, "reasoning", None))
            if reasoning_delta:
                handle_tokens(parser.feed_reasoning(reasoning_delta))
            if getattr(delta, "content", None):
                handle_tokens(parser.feed_content(delta.content))
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    _merge_tool_call_delta(tc, tool_calls_acc, slot_for_index)
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
        {"id": tc["id"] or _new_call_id(), "type": "function",
         "function": {"name": tc["name"], "arguments": tc["arguments"]}}
        for tc in tool_calls_acc
    ]

    if (not tc_list and wants_minicpm_tools(session) and content
            and "<function" in content and "</function>" in content):
        func_pattern = re.compile(r'<function\s+name="([^"]+)">\s*(.*?)\s*</function>', re.DOTALL)
        param_pattern = re.compile(r'<param\s+name="([^"]+)">(.*?)</param>', re.DOTALL)

        new_content = content
        for match in func_pattern.finditer(content):
            func_name = match.group(1)
            params_str = match.group(2)
            args = {}
            for param_match in param_pattern.finditer(params_str):
                args[param_match.group(1)] = param_match.group(2).strip()

            call_id = _new_call_id()
            tc_list.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args)
                }
            })
            new_content = new_content.replace(match.group(0), "")

        content = new_content.strip() or None

    return content, reasoning, tc_list, cancelled, (prompt_tokens, completion_tokens), finish_reason
