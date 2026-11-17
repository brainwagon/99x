"""Agent loop and tool dispatch."""

import concurrent.futures
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from agent99x import logs
from agent99x import tools  # registry + primitives (import fires registration)
from agent99x import skills  # noqa: F401 — spawn_agent / skill tools registration
from agent99x.config import MAX_AGENT_ITERATIONS
from agent99x.hooks import AgentHooks, HookEvent
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
    hooks: Optional[AgentHooks] = None,
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    tools._thread_local.cancel_event = cancel
    handler = tools.TOOL_HANDLERS.get(name, lambda **_: {"error": "unknown tool"})
    if on_tool_call:
        on_tool_call(name, args)
    logs.log_tool_call(name, args)
    _emit_hook(hooks, "tool.started", run_id, parent_run_id, tool=name)
    started = time.monotonic()
    blocked = _plan_mode_block_reason(session, name, args)
    if blocked:
        result_obj = {"error": blocked}
    elif session.allowed_tools is not None and name not in set(session.allowed_tools):
        result_obj = {"error": f"tool '{name}' is not permitted for this agent"}
    else:
        previous_hook_context = getattr(tools._thread_local, "agent_hook_context", None)
        if hooks is not None and run_id is not None:
            tools._thread_local.agent_hook_context = (hooks, run_id)
        try:
            try:
                if tools.needs_session(name):
                    result_obj = handler(session=session, **args)
                else:
                    result_obj = handler(**args)
            except Exception as e:
                logs.logger.warning("tool '%s' raised %s: %s", name, type(e).__name__, e,
                                    exc_info=True)
                result_obj = {"error": f"tool invocation failed: {e}"}
        finally:
            if hooks is not None and run_id is not None:
                if previous_hook_context is None:
                    del tools._thread_local.agent_hook_context
                else:
                    tools._thread_local.agent_hook_context = previous_hook_context
    elapsed = time.monotonic() - started
    failed = "error" in result_obj
    _emit_hook(
        hooks,
        "tool.finished",
        run_id,
        parent_run_id,
        tool=name,
        status="failed" if failed else "completed",
        duration_seconds=elapsed,
    )
    if failed:
        _emit_hook(
            hooks,
            "error",
            run_id,
            parent_run_id,
            source="tool",
            tool=name,
            error_type="ToolError",
        )
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
    on_usage: Optional[Callable[[tuple[int, int]], None]] = None,
    on_token: Optional[Callable[[ContentType, str], None]] = None,
    hooks: Optional[AgentHooks] = None,
) -> Optional[str]:
    """Run one agent invocation, optionally emitting metadata-only hook events."""
    inherited = getattr(tools._thread_local, "agent_hook_context", None)
    parent_run_id = None
    if hooks is None and inherited is not None:
        hooks, parent_run_id = inherited

    run_id = uuid.uuid4().hex if hooks is not None else None
    started = time.monotonic()
    if hooks is not None and run_id is not None:
        _emit_hook(
            hooks,
            "run.started",
            run_id,
            parent_run_id,
            provider=session.provider.name if session.provider else None,
            model=session.model,
        )

    status = "failed"
    try:
        result = _agent_loop(
            session,
            cancel=cancel,
            on_model_request=on_model_request,
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
            on_usage=on_usage,
            on_token=on_token,
            hooks=hooks,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )
        status = _run_status(result)
        if status == "cancelled":
            _emit_hook(hooks, "run.cancelled", run_id, parent_run_id)
        return result
    except Exception as exc:
        _emit_hook(
            hooks,
            "error",
            run_id,
            parent_run_id,
            source="agent",
            error_type=type(exc).__name__,
        )
        raise
    finally:
        _emit_hook(
            hooks,
            "run.finished",
            run_id,
            parent_run_id,
            status=status,
            duration_seconds=time.monotonic() - started,
        )


def _run_status(result: Optional[str]) -> str:
    if result == "(cancelled)":
        return "cancelled"
    if result is None:
        return "completed"
    if result.startswith(("(global call budget", "(max iterations", "(User Escalation)")):
        return "stopped"
    return "completed"


def _emit_hook(
    hooks: Optional[AgentHooks],
    name: str,
    run_id: Optional[str],
    parent_run_id: Optional[str],
    **metadata: Any,
) -> None:
    if hooks is not None and run_id is not None:
        hooks.emit(HookEvent(
            name=name,
            run_id=run_id,
            parent_run_id=parent_run_id,
            timestamp=time.time(),
            metadata=metadata,
        ))


def _agent_loop(
    session: SessionConfig,
    cancel: Optional[threading.Event] = None,
    on_model_request: Optional[Callable[[], None]] = None,
    on_tool_call: Optional[Callable[..., None]] = None,
    on_tool_result: Optional[Callable[..., None]] = None,
    on_usage: Optional[Callable[[tuple[int, int]], None]] = None,
    on_token: Optional[Callable[[ContentType, str], None]] = None,
    hooks: Optional[AgentHooks] = None,
    run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
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
    skill_loaded_pending = False
    nudged_skill = False
    validation_retries = 0
    MAX_VALIDATION_RETRIES = 3

    last_tool_call_signature = None
    tool_call_repeat_count = 0

    for iteration_num in range(MAX_AGENT_ITERATIONS):
        if cancel and cancel.is_set():
            return "(cancelled)"
        with session.session_lock:
            calls = session.shared_state.get("total_calls", 0)
            budget = session.shared_state["call_budget"]
            if calls >= budget:
                logs.log_warning(f"Global call budget of {budget} exhausted.")
                return f"(global call budget of {budget} exhausted)"
            session.shared_state["total_calls"] = calls + 1

        _emit_hook(
            hooks,
            "model_request.started",
            run_id,
            parent_run_id,
            provider=session.provider.name if session.provider else None,
            model=session.model,
        )
        request_started = time.monotonic()
        try:
            if on_model_request:
                on_model_request()
            request_started = time.monotonic()
            content, reasoning, tool_calls, cancelled, usage_tokens, finish_reason = stream_completion(
                session, messages, active_tools, cancel, on_token=on_token)
        except Exception as exc:
            _emit_hook(
                hooks,
                "model_request.finished",
                run_id,
                parent_run_id,
                status="failed",
                duration_seconds=time.monotonic() - request_started,
                error_type=type(exc).__name__,
            )
            raise

        _emit_hook(
            hooks,
            "model_request.finished",
            run_id,
            parent_run_id,
            status="cancelled" if cancelled else "completed",
            duration_seconds=time.monotonic() - request_started,
            input_tokens=usage_tokens[0] if usage_tokens else None,
            output_tokens=usage_tokens[1] if usage_tokens else None,
        )

        if cancelled:
            return "(cancelled)"
        if usage_tokens and on_usage:
            on_usage(usage_tokens)

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

            # Model loaded a skill but stopped with prose instead of carrying
            # out its steps — nudge once, then let any further text through.
            if content and skill_loaded_pending and not nudged_skill:
                nudged_skill = True
                messages.append({
                    "role": "user",
                    "content": (
                        "You loaded a skill but haven't carried out its steps yet. "
                        "Execute the skill's steps now by calling the appropriate "
                        "tools. Only stop and reply in text once the steps are done "
                        "(or if you are genuinely blocked on a decision that is mine)."
                    ),
                })
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
                                   cancel, on_tool_call, on_tool_result,
                                   hooks, run_id, parent_run_id)
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
                                 cancel, on_tool_call, on_tool_result,
                                 hooks, run_id, parent_run_id)

            futures = [(tc["id"], _GLOBAL_EXECUTOR.submit(_task, tc)) for tc in tool_calls]
            for tool_id, fut in futures:
                result = fut.result()
                if "error" in result:
                    last_had_error = True
                session.history.append({"role": "tool", "tool_call_id": tool_id,
                                        "content": json.dumps(result)})

        # Track skills loaded but not yet acted on: a load_skill call arms the
        # flag; any other tool call (carrying out the steps) clears it.
        called = [tc["function"]["name"] for tc in tool_calls]
        if any(n != "load_skill" for n in called):
            skill_loaded_pending = False
        elif "load_skill" in called:
            skill_loaded_pending = True

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
