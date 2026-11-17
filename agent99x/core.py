"""Agent loop and tool dispatch."""

import concurrent.futures
import json
import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from agent99x import conversation
from agent99x import logs
from agent99x import tools  # registry + primitives (import fires registration)
# Not referenced directly: importing skills registers the meta tools
# (spawn_agent, load_skill, list_*) via @tool decorators.  Keep it.
from agent99x import skills  # noqa: F401
from agent99x.config import MAX_AGENT_ITERATIONS
from agent99x.hooks import AgentHooks, HookEvent
from agent99x.reasoning import ContentType
from agent99x.session import SessionConfig
from agent99x.llm import stream_completion
from agent99x.validation import validate_file

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


def _in_pool_worker() -> bool:
    """True when running inside a _GLOBAL_EXECUTOR task submitted by _submit."""
    return bool(getattr(tools._thread_local, "in_pool_worker", False))


def _submit(fn: Callable[..., Any], *args: Any) -> "concurrent.futures.Future[Any]":
    """Submit to the shared pool, carrying the caller's agent depth and
    cancel event into the worker thread (thread-locals don't follow)."""
    depth = getattr(tools._thread_local, "depth", 0)
    cancel = getattr(tools._thread_local, "cancel_event", None)

    def runner() -> Any:
        tl = tools._thread_local
        saved = (getattr(tl, "depth", 0), getattr(tl, "cancel_event", None),
                 getattr(tl, "in_pool_worker", False))
        tl.depth, tl.cancel_event, tl.in_pool_worker = depth, cancel, True
        try:
            return fn(*args)
        finally:
            tl.depth, tl.cancel_event, tl.in_pool_worker = saved

    return _GLOBAL_EXECUTOR.submit(runner)


def _is_error(result: Any) -> bool:
    """A tool result is a failure only if it is a dict carrying an "error" key.

    (MCP tools may return plain strings; "error" appearing in such text is
    not a failure.)
    """
    return isinstance(result, dict) and "error" in result


def _dump_result(result: Any) -> str:
    """Serialize a tool result for the transcript; never raises."""
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError) as e:
        return json.dumps({"error": f"tool result not serializable: {e}"})


def _parse_args(raw: Any) -> "tuple[Optional[Dict[str, Any]], Optional[str]]":
    """Parse a tool call's JSON arguments.

    Returns (args, None) on success or (None, error) when the arguments are
    not a JSON object.  Empty, whitespace-only, and "null" arguments mean {}.
    """
    if isinstance(raw, dict):
        return raw, None
    if raw is None or (isinstance(raw, str) and raw.strip() in ("", "null")):
        return {}, None
    if not isinstance(raw, str):
        return None, f"Tool arguments must be a JSON object string, got {type(raw).__name__}."
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, (f"Invalid JSON in tool arguments: {e}. The tool was not run; "
                      "re-send the call with arguments as a valid JSON object.")
    if args is None:
        return {}, None
    if not isinstance(args, dict):
        return None, (f"Tool arguments must be a JSON object, got {type(args).__name__}. "
                      "The tool was not run.")
    return args, None


def _plan_mode_block_reason(session: SessionConfig, name: str,
                            args: Any) -> Optional[str]:
    """Return an error string if plan mode forbids this tool/args combo, else None.

    Never raises: malformed arguments to a gated tool count as blocked.
    """
    if not session.plan_mode:
        return None
    if not tools.is_plan_safe(name):
        return f"Blocked in plan mode: {name} is disabled."
    if name in ("run_bash", "http_request") and not isinstance(args, dict):
        return "Blocked in plan mode: tool arguments must be a JSON object."
    if name == "run_bash":
        command = args.get("command", "")
        if not isinstance(command, str) or not tools.is_readonly_bash(command):
            return ("Blocked in plan mode: mutating command. "
                    "Only read-only commands (ls, cat, grep, find, git log, git diff, etc.) are permitted.")
    if name == "http_request":
        method = args.get("method", "GET")
        if not isinstance(method, str) or method.upper() != "GET":
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
) -> Any:
    tools._thread_local.cancel_event = cancel
    handler = tools.TOOL_HANDLERS.get(name, lambda **_: {"error": "unknown tool"})
    if on_tool_call:
        on_tool_call(name, args)
    logs.log_tool_call(name, args)
    _emit_hook(hooks, "tool.started", run_id, parent_run_id, tool=name)
    started = time.monotonic()
    blocked = _plan_mode_block_reason(session, name, args)
    if blocked:
        result_obj: Any = {"error": blocked}
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
            except BaseException:
                # KeyboardInterrupt etc.: report the tool as cancelled, then
                # let the interrupt propagate (the loop records a result).
                _emit_hook(hooks, "tool.finished", run_id, parent_run_id, tool=name,
                           status="cancelled",
                           duration_seconds=time.monotonic() - started)
                raise
        finally:
            if hooks is not None and run_id is not None:
                if previous_hook_context is None:
                    del tools._thread_local.agent_hook_context
                else:
                    tools._thread_local.agent_hook_context = previous_hook_context
    elapsed = time.monotonic() - started
    failed = _is_error(result_obj)
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
    logs.log_tool_result(name, _dump_result(result_obj), elapsed)
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
    except KeyboardInterrupt:
        status = "cancelled"
        _emit_hook(hooks, "run.cancelled", run_id, parent_run_id)
        raise
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
    previous_cancel = getattr(tools._thread_local, "cancel_event", None)
    tools._thread_local.cancel_event = cancel
    try:
        return _agent_loop_body(session, cancel, on_model_request, on_tool_call,
                                on_tool_result, on_usage, on_token, hooks,
                                run_id, parent_run_id)
    finally:
        tools._thread_local.cancel_event = previous_cancel


_NOT_RUN_CANCELLED = {"error": "cancelled: the tool call was interrupted before it returned a result"}
MAX_REPEATED_TOOL_CALLS = 3
MAX_VALIDATION_RETRIES = 3


def _agent_loop_body(session, cancel, on_model_request, on_tool_call,
                     on_tool_result, on_usage, on_token, hooks,
                     run_id, parent_run_id) -> Optional[str]:
    active_tools = _active_tools(session)
    # One list for the whole run: assistant turns and their tool results
    # must land in the same transcript to stay paired.
    messages = session.history
    with session.session_lock:
        if "call_budget" not in session.shared_state:
            session.shared_state["call_budget"] = session.effective_call_budget()

    last_had_error = False
    auto_continues = 0
    nudged_handback = False
    skill_loaded_pending = False
    nudged_skill = False
    validation_retries = 0

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

        # Never send a transcript with unpaired tool calls/results; the
        # API rejects it.  Repair in place so the saved context is clean too.
        repaired = conversation.repair_tool_pairs(messages)
        if repaired != messages:
            messages[:] = repaired

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
        except BaseException as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            _emit_hook(
                hooks,
                "model_request.finished",
                run_id,
                parent_run_id,
                status="cancelled" if interrupted else "failed",
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

        # Every tool_call id must get a tool message, on every exit path
        # (early stop, exceptions, KeyboardInterrupt): the finally below
        # fills in a "cancelled" result for any call that has none.
        results: List[Any] = [None] * len(tool_calls)
        snapshot: Dict[str, Optional[bytes]] = {}
        modified_paths: List[str] = []
        try:
            current_signature = tuple((tc["function"]["name"], tc["function"]["arguments"]) for tc in tool_calls)
            if last_tool_call_signature == current_signature:
                tool_call_repeat_count += 1
            else:
                last_tool_call_signature = current_signature
                tool_call_repeat_count = 1
            if tool_call_repeat_count >= MAX_REPEATED_TOOL_CALLS:
                stop = {"error": "not run: early stop after repeated identical tool calls"}
                results[:] = [stop] * len(tool_calls)
                return (f"(User Escalation) Early-stop triggered: You have repeated the exact same "
                        f"tool calls {MAX_REPEATED_TOOL_CALLS} times in a row without making "
                        "progress. Please rethink your approach.")

            parsed = [_parse_args(tc["function"]["arguments"]) for tc in tool_calls]

            for tc, (args, _err) in zip(tool_calls, parsed):
                if not (args and tools.is_mutating(tc["function"]["name"])):
                    continue
                filepath = args.get("path")
                if not isinstance(filepath, str) or not filepath or filepath in snapshot:
                    continue
                if tools.check_write_path(filepath):
                    continue  # the tool will refuse; never snapshot or roll back outside paths
                try:
                    if os.path.exists(filepath):
                        with open(filepath, "rb") as f:
                            snapshot[filepath] = f.read()
                    else:
                        snapshot[filepath] = None
                    modified_paths.append(filepath)
                except OSError:
                    pass

            def _task(tc: Dict[str, Any], args: Optional[Dict[str, Any]],
                      err: Optional[str]) -> Any:
                if err is not None:
                    return {"error": err}
                return _run_tool(session, tc["function"]["name"], args,
                                 cancel, on_tool_call, on_tool_result,
                                 hooks, run_id, parent_run_id)

            # Run sequentially for one call, and inside a pool worker (a
            # parallel subagent) so nested waits can't exhaust the pool.
            if len(tool_calls) == 1 or _in_pool_worker():
                for i, tc in enumerate(tool_calls):
                    if i > 0 and cancel is not None and cancel.is_set():
                        break
                    results[i] = _task(tc, *parsed[i])
            else:
                futures = [_submit(_task, tc, *parsed[i]) for i, tc in enumerate(tool_calls)]
                try:
                    for i, fut in enumerate(futures):
                        try:
                            results[i] = fut.result()
                        except Exception as e:
                            logs.logger.warning("tool dispatch raised %s: %s",
                                                type(e).__name__, e, exc_info=True)
                            results[i] = {"error": f"tool invocation failed: {e}"}
                finally:
                    for fut in futures:
                        fut.cancel()
        finally:
            for i, tc in enumerate(tool_calls):
                result = results[i] if results[i] is not None else _NOT_RUN_CANCELLED
                if _is_error(result):
                    last_had_error = True
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": _dump_result(result)})

        # Track skills loaded but not yet acted on: a load_skill call arms the
        # flag; any other tool call (carrying out the steps) clears it.
        called = [tc["function"]["name"] for tc in tool_calls]
        if any(n != "load_skill" for n in called):
            skill_loaded_pending = False
        elif "load_skill" in called:
            skill_loaded_pending = True

        if snapshot:
            error_msg = _validate_edits(modified_paths, snapshot)
            if not error_msg:
                validation_retries = 0
                continue
            _restore_snapshot(snapshot)

            validation_retries += 1
            if validation_retries > MAX_VALIDATION_RETRIES:
                return f"(User Escalation) I attempted to edit the code {MAX_VALIDATION_RETRIES} times but kept hitting syntax errors. I've rolled back the files to keep the project safe. Here is the persistent error:\n\n{error_msg}\n\nHow would you like to proceed?"

            if validation_retries == 2:
                messages.append({
                    "role": "user",
                    "content": f"Validation failed:\n{error_msg}\n\nTool Escalation: Targeted edits are failing. You MUST use 'write_file' to rewrite the entire file from scratch to avoid context drift."
                })
            else:
                messages.append({
                    "role": "user",
                    "content": f"Validation failed:\n{error_msg}\n\nI have rolled back the changes. Please fix the logic and try again."
                })
            continue

    return "(max iterations reached)"


def _write_bytes(path: str, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)


def _restore_snapshot(snapshot: Dict[str, Optional[bytes]]) -> None:
    """Put every snapshotted file back byte-for-byte (removing new files)."""
    for fp, orig in snapshot.items():
        try:
            if orig is None:
                if os.path.exists(fp):
                    os.remove(fp)
            else:
                _write_bytes(fp, orig)
        except OSError as e:
            logs.log_warning(f"rollback of {fp} failed: {e}")


def _validate_edits(paths: List[str], snapshot: Dict[str, Optional[bytes]]) -> Optional[str]:
    """Return syntax errors introduced by this batch of edits, or None.

    A file that already failed validation before the edit is not counted
    (rolling back would not make it valid); new files are checked normally.
    """
    errors: List[str] = []
    for fp in paths:
        err = validate_file(fp)
        if not err:
            continue
        orig = snapshot.get(fp)
        if orig is not None:
            try:
                with open(fp, "rb") as f:
                    edited = f.read()
                _write_bytes(fp, orig)
                try:
                    was_invalid = validate_file(fp) is not None
                finally:
                    _write_bytes(fp, edited)
            except OSError:
                was_invalid = False
            if was_invalid:
                continue
        errors.append(err)
    return "\n\n".join(errors) if errors else None


def clip(s: str, n: int = 50, ellipsis: str = "...") -> str:
    s = str(s)
    return s[:n] + ellipsis if len(s) > n else s
