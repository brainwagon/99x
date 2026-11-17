"""CLI argument parsing, entry points, one-shot runner, and the plain REPL."""

import argparse
import os
import sys
from typing import Optional, List

from agent99x import providers
from agent99x import core
from agent99x import compaction
from agent99x import conversation
from agent99x import contexts
from agent99x import commands
from agent99x.reasoning import ContentType, ThinkingRenderer
from agent99x.session import SessionConfig, AGENT_HOME, PROJECT_DIR
from agent99x.hooks import AgentHooks, HookEvent
from agent99x.config_io import load_config, save_config, save_session
from agent99x.commands import handle_slash_command


def _make_token_handler(session: SessionConfig):
    """Create a token callback that respects the show_thinking level."""
    renderer = ThinkingRenderer(session.show_thinking)

    def on_token(ctype, text):
        for kind, piece in renderer.feed(ctype, text):
            if kind == ContentType.THINKING:
                sys.stdout.write(f"\033[90m{piece}\033[0m")
            else:
                sys.stdout.write(piece)
        sys.stdout.flush()

    return on_token


class ReplyTracker:
    """Tells whether agent_loop's returned text was already streamed.

    agent_loop returns either the final model content (streamed token by
    token through on_token during the last model request) or a loop-made
    message that never streams: "(global call budget … exhausted)",
    "(max iterations reached)", "(User Escalation) …". Feed it the
    on_model_request and on_token callbacks; ``unstreamed(reply)`` returns
    the reply if the user hasn't seen it yet, else None.
    """

    def __init__(self) -> None:
        self._parts: List[str] = []

    def on_model_request(self) -> None:
        self._parts.clear()

    def on_token(self, ctype: ContentType, text: str) -> None:
        if ctype == ContentType.CONTENT:
            self._parts.append(text)

    def unstreamed(self, reply: Optional[str]) -> Optional[str]:
        if not reply or reply == "(cancelled)":
            return None
        if reply.strip() == "".join(self._parts).strip():
            return None
        return reply


# Where `99x --trace-hooks` writes in TUI mode: Textual owns the terminal
# (and stderr) while the app runs, so the trace goes to a file instead.
TRACE_FILE = os.path.join(PROJECT_DIR, "hook-trace.log")


def _format_hook_event(event: HookEvent) -> str:
    parent = f" parent={event.parent_run_id[:8]}" if event.parent_run_id else ""
    details = " ".join(
        f"{key}={value}" for key, value in event.metadata.items()
        if value is not None
    )
    return (f"[hook] {event.timestamp:.3f} {event.name} "
            f"run={event.run_id[:8]}{parent} {details}")


def _trace_hook_event(event: HookEvent) -> None:
    """Print a concise, privacy-safe hook event to stderr."""
    print(_format_hook_event(event), file=sys.stderr, flush=True)


def _file_trace_handler(path: str):
    """A trace handler that appends each event line to ``path``."""
    def handler(event: HookEvent) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(_format_hook_event(event) + "\n")
    return handler


def _cli_hooks(session: SessionConfig, hooks: Optional[AgentHooks],
               trace_path: Optional[str] = None) -> Optional[AgentHooks]:
    """Add the optional CLI trace handler to any programmatic handlers.

    The trace goes to stderr, or appended to ``trace_path`` when given (the
    TUI passes TRACE_FILE and records the path in
    ``session.shared_state["hook_trace_path"]`` for its startup banner).
    """
    if not session.shared_state.pop("_trace_hooks", False):
        return hooks
    if trace_path is None:
        trace = _trace_hook_event
    else:
        os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
        session.shared_state["hook_trace_path"] = os.path.abspath(trace_path)
        trace = _file_trace_handler(trace_path)
    handlers = [trace]
    if hooks is not None:
        handlers.extend(hooks.handlers)
    return AgentHooks(handlers)


# ── MCP gating ─────────────────────────────────────────────────────

def _mcp_config_path(session: SessionConfig) -> Optional[str]:
    """Return an existing mcp-config.json path (project first, then global), or None."""
    for path in ("mcp-config.json", session.agent_path("mcp-config.json")):
        if os.path.exists(path):
            return path
    return None


def _mcp_start(session: SessionConfig) -> None:
    """Start MCP only when a config file exists and the optional dep is installed."""
    if _mcp_config_path(session) is None:
        return
    try:
        from agent99x import mcp
    except ImportError:
        print("Warning: mcp-config.json found but the 'mcp' package is not installed "
              "(pip install -e .[mcp]); skipping MCP.", file=sys.stderr)
        return
    mcp.initialize_mcp(session)


def _mcp_stop(session: SessionConfig) -> None:
    if not session.mcp_clients:
        return
    from agent99x import mcp
    mcp.cleanup_mcp(session)


# ── guards ─────────────────────────────────────────────────────────

def _check_not_home_dir() -> None:
    """Exit if cwd/.99x would resolve to AGENT_HOME, colliding project state with global config."""
    cwd = os.getcwd()
    project_path = os.path.realpath(os.path.join(cwd, PROJECT_DIR))
    if project_path == os.path.realpath(AGENT_HOME):
        print(
            f"Error: running 99x from {cwd} would collide the project {PROJECT_DIR}/ directory\n"
            f"with AGENT_HOME ({AGENT_HOME}). cd to a project directory first.",
            file=sys.stderr,
        )
        sys.exit(1)


# ── init ───────────────────────────────────────────────────────────

def _build_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="99x: a streamlined Python AI agent.")
    parser.add_argument("prompt", nargs="*", help="Initial prompt (triggers one-shot mode).")
    parser.add_argument("--provider", choices=list(providers.PROVIDERS), help="LLM provider.")
    parser.add_argument("--model", help="Model ID.")
    parser.add_argument("--host", help="Host override (e.g. 10.0.0.5:11434).")
    parser.add_argument("--effort", choices=["low", "medium", "high", "none"],
                        help="Reasoning effort ('none' clears it).")
    parser.add_argument("--autocompact", type=float, help="Autocompact threshold (0 to 1).")
    parser.add_argument("--plan", action="store_true", help="Start in plan mode.")
    parser.add_argument(
        "--trace-hooks",
        action="store_true",
        help="Trace metadata-only lifecycle hook events (to stderr for one-shot "
             f"runs, to {TRACE_FILE} in the TUI).",
    )
    parser.add_argument(
        "-c", "--context", nargs="?", const=contexts.LATEST, default=None, metavar="NAME",
        help="Restore a saved context. With NAME, restore (or create) that named context; "
             "with no value, restore the latest context for this project. "
             "Omit -c entirely to start with a fresh context.")
    return parser


def init_from_argv(session: SessionConfig, argv: Optional[List[str]] = None, *,
                   prog: str = "99x", require_prompt: bool = False) -> bool:
    """Parse CLI args and set up session. Returns True if one-shot mode.

    One-shot runs (a prompt was given) are stateless: flags apply to this
    run only (config.json is not written), and no context file is written
    unless ``-c`` names one to continue. Interactive runs persist the
    provider/model/host/effort choices, as the TUI's slash commands do.
    With ``require_prompt`` (99xsh), a missing prompt is a usage error.
    """
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_parser(prog)
    args = parser.parse_args(argv)
    if require_prompt and not args.prompt:
        parser.error("a prompt is required")
    one_shot = bool(args.prompt)
    persist = not one_shot

    load_config(session)

    if args.provider:
        # Like /provider: restore what was last used with this provider;
        # explicit --model/--host override it.
        name = args.provider
        session.provider = providers.PROVIDERS[name]
        session.model = args.model or session.provider_models.get(name)
        session.host = args.host or session.provider_hosts.get(name)
    else:
        if args.model:
            session.model = args.model
        if args.host:
            session.host = args.host
    if args.effort:
        session.effort = None if args.effort == "none" else args.effort
    if args.autocompact is not None:
        session.autocompact_threshold = args.autocompact
    if args.plan:
        session.plan_mode = True
    if args.trace_hooks:
        session.shared_state["_trace_hooks"] = True

    if session.provider:
        provider_name = session.provider.name
        try:
            commands.activate_provider(session, provider_name, session.model,
                                       session.host, autopick=True)
        except ValueError as e:
            print(
                f"Warning: {e}\n"
                f"  No model is selected. Pass --model <id>, or use /model in the TUI.",
                file=sys.stderr,
            )
            session.model = None
        if session.model:
            session.provider_models[provider_name] = session.model
        if session.host is not None:
            session.provider_hosts[provider_name] = session.host
        if persist:
            save_config(session)

    _setup_context(session, args.context, persist)

    if one_shot:
        prompt = " ".join(args.prompt)
        conversation.add_user(session, prompt)

    return one_shot


def _setup_context(session: SessionConfig, context_arg: Optional[str],
                   write_new: bool = True) -> None:
    """Pick the context file for this run and seed session.history.

    ``context_arg`` is the parsed ``-c/--context`` value:
      * None            — flag absent: start fresh in a new timestamped context
                          (or, with ``write_new=False``, in no context file at
                          all: session.context_path stays None).
      * contexts.LATEST — ``-c`` with no value: restore the newest context.
      * "<name>"        — restore the named context (or start fresh under it).
    """
    if context_arg is None:
        session.context_path = contexts.new_path() if write_new else None
        conversation.start(session)
        return

    if context_arg == contexts.LATEST:
        path = contexts.latest_path()
        if path is None:
            print("No saved context to restore; starting fresh.", file=sys.stderr)
            session.context_path = contexts.new_path() if write_new else None
            conversation.start(session)
            return
    else:
        path = contexts.path_for(context_arg)

    session.context_path = path
    if not conversation.load(session, path):
        conversation.start(session)


def provider_ready(session: SessionConfig) -> bool:
    """True when a provider and model are both configured (a request can be made)."""
    return bool(session.provider and session.model)


def repl(session: SessionConfig, hooks: Optional[AgentHooks] = None) -> None:
    """A plain terminal REPL."""
    print(f"99xsh — {session.provider.name if session.provider else '(no provider)'} / {session.model}")
    if session.plan_mode:
        print("Plan mode is ON.")
    hint = commands.model_hint(session)
    if hint:
        print(hint)

    if not provider_ready(session):
        print(commands.prompt_for_provider(session).message)

    import readline  # noqa: F401  enables line editing for input()

    while True:
        try:
            text = input("> ").strip()
            if not text:
                continue

            result = handle_slash_command(session, text)
            if result.handled:
                if result.message:
                    print(result.message)
                if result.clear_history:
                    conversation.clear(session)
                if result.exit_app:
                    break
                if result.compact:
                    print("Compacting...")
                    compaction.do_compact(session)
                continue

            conversation.add_user(session, result.send_text or text)
            _run_turn(session, hooks)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            continue
        except EOFError:
            print("\nExit.")
            break
        except Exception as e:
            print(f"\nError: {e}")

    save_session(session)


def _run_turn(session: SessionConfig, hooks: Optional[AgentHooks]) -> Optional[str]:
    """Run agent_loop, streaming to stdout; print the reply if it didn't stream."""
    tracker = ReplyTracker()
    show = _make_token_handler(session)

    def on_token(ctype, text):
        tracker.on_token(ctype, text)
        show(ctype, text)

    print()
    reply = core.agent_loop(session, on_model_request=tracker.on_model_request,
                            on_token=on_token, hooks=hooks)
    unseen = tracker.unstreamed(reply)
    if unseen:
        print(unseen, end="")
    print()
    return reply


def _run_oneshot(
    session: SessionConfig,
    hooks: Optional[AgentHooks] = None,
) -> None:
    if not provider_ready(session):
        known = ", ".join(sorted(providers.PROVIDERS))
        print(
            "Error: no provider/model configured. One-shot mode can't prompt.\n"
            f"  Pass --provider <name> --model <id> (known providers: {known}),\n"
            "  or launch the interactive TUI with `99x` to choose one.",
            file=sys.stderr,
        )
        sys.exit(1)
    _mcp_start(session)
    try:
        _run_turn(session, hooks)
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        # Stateless unless -c chose a context to continue.
        if session.context_path is not None:
            save_session(session)
        _mcp_stop(session)


def _run_tui(
    session: SessionConfig,
    hooks: Optional[AgentHooks] = None,
) -> None:
    from agent99x.tui import AgentApp
    _mcp_start(session)
    try:
        AgentApp(session, hooks=hooks).run()
    finally:
        _mcp_stop(session)


def main_oneshot(hooks: Optional[AgentHooks] = None) -> None:
    """Entry point for `99xsh`: one-shot query, print reply, exit."""
    _check_not_home_dir()
    session = SessionConfig()
    init_from_argv(session, prog="99xsh", require_prompt=True)
    _run_oneshot(session, hooks=_cli_hooks(session, hooks))


def main_tui(hooks: Optional[AgentHooks] = None) -> None:
    """Entry point for `99x`: interactive TUI."""
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        cmd_init()
        return
    _check_not_home_dir()
    session = SessionConfig()
    one_shot = init_from_argv(session)
    hooks = _cli_hooks(session, hooks, trace_path=None if one_shot else TRACE_FILE)
    if one_shot:
        _run_oneshot(session, hooks=hooks)
    else:
        _run_tui(session, hooks=hooks)


def cmd_init() -> None:
    """Seed ~/.99x (AGENT_HOME) with default files if they are absent."""
    import shutil

    def _seed_file(dest_path, src_path):
        if os.path.exists(dest_path):
            print(f"  skip  {dest_path}")
            return
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)
        print(f"  copy  {src_path} -> {dest_path}")

    defaults_dir = os.path.join(os.path.dirname(__file__), "_defaults")
    if not os.path.isdir(defaults_dir):
        print(f"Error: bundled defaults not found at {defaults_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(AGENT_HOME, exist_ok=True)
    for dirpath, _dirs, files in os.walk(defaults_dir):
        rel = os.path.relpath(dirpath, defaults_dir)
        for fname in files:
            src = os.path.join(dirpath, fname)
            dst = os.path.join(AGENT_HOME, rel, fname) if rel != "." else os.path.join(AGENT_HOME, fname)
            _seed_file(dst, src)

    print(f"Initialized {AGENT_HOME}")
