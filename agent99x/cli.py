"""CLI argument parsing, REPL, slash commands, session save/load."""

import argparse
import os
import sys
from typing import Optional, List

from agent99x import providers
from agent99x import core
from agent99x import compaction
from agent99x import conversation
from agent99x import contexts
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


def _trace_hook_event(event: HookEvent) -> None:
    """Print a concise, privacy-safe hook event to stderr."""
    parent = f" parent={event.parent_run_id[:8]}" if event.parent_run_id else ""
    details = " ".join(
        f"{key}={value}" for key, value in event.metadata.items()
        if value is not None
    )
    print(
        f"[hook] {event.timestamp:.3f} {event.name} "
        f"run={event.run_id[:8]}{parent} {details}",
        file=sys.stderr,
        flush=True,
    )


def _cli_hooks(session: SessionConfig, hooks: Optional[AgentHooks]) -> Optional[AgentHooks]:
    """Add the optional CLI trace handler to any programmatic handlers."""
    if not session.shared_state.pop("_trace_hooks", False):
        return hooks
    handlers = [_trace_hook_event]
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

def init_from_argv(session: SessionConfig, argv: Optional[List[str]] = None) -> bool:
    """Parse CLI args and set up session. Returns True if one-shot mode."""
    if argv is None:
        argv = sys.argv[1:]
    load_config(session)

    parser = argparse.ArgumentParser(prog="99x", description="99x: a streamlined Python AI agent.")
    parser.add_argument("prompt", nargs="*", help="Initial prompt (triggers one-shot mode).")
    parser.add_argument("--provider", choices=list(providers.PROVIDERS), help="LLM provider.")
    parser.add_argument("--model", help="Model ID.")
    parser.add_argument("--host", help="Host override (e.g. 10.0.0.5:11434).")
    parser.add_argument("--effort", choices=["low", "medium", "high"], help="Reasoning effort.")
    parser.add_argument("--autocompact", type=float, help="Autocompact threshold (0 to 1).")
    parser.add_argument("--plan", action="store_true", help="Start in plan mode.")
    parser.add_argument(
        "--trace-hooks",
        action="store_true",
        help="Print metadata-only lifecycle hook events to stderr.",
    )
    parser.add_argument(
        "-c", "--context", nargs="?", const=contexts.LATEST, default=None, metavar="NAME",
        help="Restore a saved context. With NAME, restore (or create) that named context; "
             "with no value, restore the latest context for this project. "
             "Omit -c entirely to start with a fresh context.")
    args = parser.parse_args(argv)

    if args.provider:
        session.provider = providers.PROVIDERS[args.provider]
        if not args.host:
            session.host = None
        if not args.model:
            session.model = None

    if args.model:
        session.model = args.model
    if args.host:
        session.host = args.host
    if args.effort:
        session.effort = args.effort
    if args.autocompact is not None:
        session.autocompact_threshold = args.autocompact
    if args.plan:
        session.plan_mode = True
    if args.trace_hooks:
        session.shared_state["_trace_hooks"] = True

    if session.provider:
        provider_name = session.provider.name
        try:
            providers.setup_provider(session, provider_name, session.model, session.host)
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
        save_config(session)

    _setup_context(session, args.context)

    one_shot = bool(args.prompt)
    if one_shot:
        prompt = " ".join(args.prompt)
        conversation.add_user(session, prompt)

    return one_shot


def _setup_context(session: SessionConfig, context_arg: Optional[str]) -> None:
    """Pick the context file for this run and seed session.history.

    ``context_arg`` is the parsed ``-c/--context`` value:
      * None            — flag absent: start fresh in a new timestamped context.
      * contexts.LATEST — ``-c`` with no value: restore the newest context.
      * "<name>"        — restore the named context (or start fresh under it).
    """
    if context_arg is None:
        session.context_path = contexts.new_path()
        conversation.start(session)
        return

    if context_arg == contexts.LATEST:
        path = contexts.latest_path()
        if path is None:
            print("No saved context to restore; starting fresh.", file=sys.stderr)
            session.context_path = contexts.new_path()
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
    from agent99x import commands as _commands
    hint = _commands.model_hint(session)
    if hint:
        print(hint)

    if not provider_ready(session):
        from agent99x import commands
        print(commands.prompt_for_provider(session).message)

    import readline  # noqa: F401 — enables input() history if available

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

            conversation.add_user(session, text)
            on_token = _make_token_handler(session)
            print()
            core.agent_loop(session, on_token=on_token, hooks=hooks)
            print()

        except KeyboardInterrupt:
            print("\nInterrupted.")
            continue
        except EOFError:
            print("\nExit.")
            break
        except Exception as e:
            print(f"\nError: {e}")

    save_session(session)


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
        on_token = _make_token_handler(session)
        print()
        core.agent_loop(session, on_token=on_token, hooks=hooks)
        print()
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
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
    init_from_argv(session)
    _run_oneshot(session, hooks=_cli_hooks(session, hooks))


def main_tui(hooks: Optional[AgentHooks] = None) -> None:
    """Entry point for `99x`: interactive TUI."""
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        cmd_init()
        return
    _check_not_home_dir()
    session = SessionConfig()
    one_shot = init_from_argv(session)
    hooks = _cli_hooks(session, hooks)
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
