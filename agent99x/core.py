"""Agent loop and tool dispatch."""

import concurrent.futures
import json
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from agent99x import logs
from agent99x import tools  # registry + primitives (import fires registration)
from agent99x import skills  # noqa: F401 — spawn_agent / skill tools registration
from agent99x.config import MAX_AGENT_ITERATIONS
from agent99x.reasoning import ContentType
from agent99x.session import SessionConfig
from agent99x.llm import stream_completion
from agent99x.validation import validate_files

# Plain-text replies that read as the model handing control back to the user
# (a question / stall) rather than reporting a finished task. We nudge once;
# if the model insists, we let the text through.
_HANDBACK_PATTERNS = re.compile(
    r"(would you like|shall i|should i|how (would|should) you|"
    r"which .*would you|let me know|do you want me to)",
    re.IGNORECASE,
)


def _looks_like_handback(text: str) -> bool:
    """Return True if plain-text output reads as a question back to the user."""
    stripped = text.strip()
    if not stripped:
        return False
    return stripped.endswith("?") or bool(_HANDBACK_PATTERNS.search(stripped))


# Shared pool for parallel tool-call dispatch.
_GLOBAL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="worker")


def _plan_mode_block_reason(session: SessionConfig, name: str,
                            args: Dict[str, Any]) -> Optional[str]:
    """Return an error string if plan mode forbids this tool/args combo, else None."""
    if not session.plan_mode:
        return None
    if not tools.is_plan_safe(name):
        return f"Blocked in plan mode: {name} is disabled."
    if name == "run_bash" and not tools.is_readonly_bash(args.get("command", "")):
        return ("Blocked in plan mode: mutating command. "
                "Only read-only commands (ls, cat, grep, find, git log, git diff, etc.) are permitted.")
    if name == "http_request" and args.get("method", "GET").upper() != "GET":
        return "Blocked in plan mode: only GET requests are permitted."
    return None


def _active_tools(session: SessionConfig) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = tools.TOOLS
    if session.plan_mode:
        pool = [t for t in pool if tools.is_plan_safe(t["function"]["name"])]
    if session.allowed_tools is not None:
        allowed = set(session.allowed_tools)
        pool = [t for t in pool if t["function"]["name"] in allowed]
    return pool


def _run_tool(
    session: SessionConfig,
    name: str,
    args: Dict[str, Any],
    cancel: Optional[threading.Event],
    on_tool_call: Optional[Callable[..., None]],
    on_tool_result: Optional[Callable[[Dict[str, Any], float], None]],
) -> Dict[str, Any]:
    tools._thread_local.cancel_event = cancel
    handler = tools.TOOL_HANDLERS.get(name, lambda **_: {"error": "unknown tool"})
    if on_tool_call:
        on_tool_call(name, args)
    logs.log_tool_call(name, args)
    t0 = time.monotonic()
    blocked = _plan_mode_block_reason(session, name, args)
    if blocked:
        result_obj = {"error": blocked}
    elif session.allowed_tools is not None and name not in set(session.allowed_tools):
        result_obj = {"error": f"tool '{name}' is not permitted for this agent"}
    else:
        try:
            if tools.needs_session(name):
                result_obj = handler(session=session, **args)
            else:
                result_obj = handler(**args)
        except Exception as e:
            logs.logger.warning("tool '%s' raised %s: %s", name, type(e).__name__, e,
                                exc_info=True)
            result_obj = {"error": f"tool invocation failed: {e}"}
    elapsed = time.monotonic() - t0
    if on_tool_result:
        on_tool_result(result_obj, elapsed)
    logs.log_tool_result(name, json.dumps(result_obj), elapsed)
    return result_obj


def agent_loop(
    session: SessionConfig,
    cancel: Optional[threading.Event] = None,
    on_model_request: Optional[Callable[[], None]] = None,
    on_tool_call: Optional[Callable[..., None]] = None,
    on_tool_result: Optional[Callable[..., None]] = None,
    on_usage: Optional[Callable[[int], None]] = None,
    on_token: Optional[Callable[[ContentType, str], None]] = None,
    on_iteration: Optional[Callable[[int, int], None]] = None,
) -> Optional[str]:
    """Run the agent's main chat-completion + tool-dispatch loop."""
    tools._thread_local.cancel_event = cancel
    active_tools = _active_tools(session)
    messages = session.history
    if "call_budget" not in session.shared_state:
        session.shared_state["call_budget"] = session.effective_call_budget()

    last_had_error = False
    auto_continues = 0
    nudged_handback = False
    validation_retries = 0
    MAX_VALIDATION_RETRIES = 3

    last_tool_call_signature = None
    tool_call_repeat_count = 0

    for iteration_num in range(MAX_AGENT_ITERATIONS):
        if on_iteration:
            on_iteration(iteration_num + 1, MAX_AGENT_ITERATIONS)
        if cancel and cancel.is_set():
            return "(cancelled)"
        with session.session_lock:
            calls = session.shared_state.get("total_calls", 0)
            budget = session.shared_state["call_budget"]
            if calls >= budget:
                logs.log_warning(f"Global call budget of {budget} exhausted.")
                return f"(global call budget of {budget} exhausted)"
            session.shared_state["total_calls"] = calls + 1

        if on_model_request:
            on_model_request()

        content, reasoning, tool_calls, cancelled, prompt_tokens, finish_reason = stream_completion(
            session, messages, active_tools, cancel, on_token=on_token)

        if cancelled:
            return "(cancelled)"
        if prompt_tokens and on_usage:
            on_usage(prompt_tokens)

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            if finish_reason == "length":
                messages.append({"role": "user", "content": "Your response was cut off due to length limits. Please continue where you left off."})
                continue
            if not content and reasoning:
                messages.append({"role": "user", "content": "You completed your thoughts but provided no text response or tool calls. Please provide your response or call a tool."})
                continue

            if last_had_error and auto_continues < 2 and content:
                auto_continues += 1
                messages.append({"role": "user", "content": "Please continue."})
                continue

            # Model stopped with question-shaped text and no error — nudge once.
            if content and not nudged_handback and _looks_like_handback(content):
                nudged_handback = True
                messages.append({
                    "role": "user",
                    "content": (
                        "If the task is complete, say so plainly. If work remains, "
                        "continue by calling a tool rather than asking me — only ask "
                        "if you are genuinely blocked on a decision that is mine to make."
                    ),
                })
                continue

            return content

        last_had_error = False
        auto_continues = 0

        current_signature = tuple((tc["function"]["name"], tc["function"]["arguments"]) for tc in tool_calls)
        if last_tool_call_signature == current_signature:
            tool_call_repeat_count += 1
            if tool_call_repeat_count >= 3:
                return "(User Escalation) Early-stop triggered: You have repeated the exact same tool calls 3 times in a row without making progress. Please rethink your approach."
        else:
            last_tool_call_signature = current_signature
            tool_call_repeat_count = 0

        snapshot = {}
        modified_paths = []
        for tc in tool_calls:
            if tools.is_mutating(tc["function"]["name"]):
                try:
                    args = json.loads(tc["function"]["arguments"])
                    filepath = args.get("path")
                    if filepath and filepath not in snapshot:
                        modified_paths.append(filepath)
                        if os.path.exists(filepath):
                            with open(filepath, "r", encoding="utf-8") as f:
                                snapshot[filepath] = f.read()
                        else:
                            snapshot[filepath] = None
                except Exception:
                    pass

        if len(tool_calls) == 1:
            tc = tool_calls[0]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError as e:
                args = {"error": f"Invalid JSON in arguments: {e}"}
            result_obj = _run_tool(session, tc["function"]["name"], args,
                                   cancel, on_tool_call, on_tool_result)
            if "error" in result_obj:
                last_had_error = True
            session.history.append({"role": "tool", "tool_call_id": tc["id"],
                                    "content": json.dumps(result_obj)})
        else:
            def _task(tc: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError as e:
                    args = {"error": f"Invalid JSON in arguments: {e}"}
                return _run_tool(session, tc["function"]["name"], args,
                                 cancel, on_tool_call, on_tool_result)

            futures = [(tc["id"], _GLOBAL_EXECUTOR.submit(_task, tc)) for tc in tool_calls]
            for tool_id, fut in futures:
                result = fut.result()
                if "error" in result:
                    last_had_error = True
                session.history.append({"role": "tool", "tool_call_id": tool_id,
                                        "content": json.dumps(result)})

        if snapshot:
            error_msg = validate_files(modified_paths)
            if error_msg:
                # Rollback files
                for fp, orig_content in snapshot.items():
                    if orig_content is None:
                        if os.path.exists(fp):
                            os.remove(fp)
                    else:
                        with open(fp, "w", encoding="utf-8") as f:
                            f.write(orig_content)

                validation_retries += 1
                if validation_retries > MAX_VALIDATION_RETRIES:
                    return f"(User Escalation) I attempted to edit the code {MAX_VALIDATION_RETRIES} times but kept hitting syntax errors. I've rolled back the files to keep the project safe. Here is the persistent error:\n\n{error_msg}\n\nHow would you like to proceed?"

                if validation_retries == 2:
                    session.history.append({
                        "role": "user",
                        "content": f"Validation failed:\n{error_msg}\n\nTool Escalation: Targeted edits are failing. You MUST use 'write_file' to rewrite the entire file from scratch to avoid context drift."
                    })
                else:
                    session.history.append({
                        "role": "user",
                        "content": f"Validation failed:\n{error_msg}\n\nI have rolled back the changes. Please fix the logic and try again."
                    })
                continue

    return "(max iterations reached)"


def clip(s: str, n: int = 50, ellipsis: str = "...") -> str:
    s = str(s)
    return s[:n] + ellipsis if len(s) > n else s
