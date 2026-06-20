"""CLI argument parsing, REPL, slash commands, session save/load."""

import argparse
import os
import sys
from typing import Optional, List

from agent99x import providers
from agent99x import core
from agent99x import compaction
from agent99x import conversation
from agent99x.reasoning import ContentType, ThinkingRenderer
from agent99x.session import SessionConfig, AGENT_HOME, PROJECT_DIR
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

    one_shot = bool(args.prompt)
    if one_shot:
        prompt = " ".join(args.prompt)
        conversation.add_user(session, prompt)

    return one_shot


def provider_ready(session: SessionConfig) -> bool:
    """True when a provider and model are both configured (a request can be made)."""
    return bool(session.provider and session.model)


def repl(session: SessionConfig) -> None:
    """A plain terminal REPL."""
    print(f"99xsh — {session.provider.name if session.provider else '(no provider)'} / {session.model}")
    if session.plan_mode:
        print("Plan mode is ON.")

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
            core.agent_loop(session, on_token=on_token)
            print()
            save_session(session)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            continue
        except EOFError:
            print("\nExit.")
            break
        except Exception as e:
            print(f"\nError: {e}")


def _run_oneshot(session: SessionConfig) -> None:
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
        core.agent_loop(session, on_token=on_token)
        print()
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        _mcp_stop(session)


def _run_tui(session: SessionConfig) -> None:
    from agent99x.tui import AgentApp
    _mcp_start(session)
    try:
        AgentApp(session).run()
    finally:
        _mcp_stop(session)


def main_oneshot() -> None:
    """Entry point for `99xsh`: one-shot query, print reply, exit."""
    _check_not_home_dir()
    session = SessionConfig()
    init_from_argv(session)
    _run_oneshot(session)


def main_tui() -> None:
    """Entry point for `99x`: interactive TUI."""
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        cmd_init()
        return
    _check_not_home_dir()
    session = SessionConfig()
    one_shot = init_from_argv(session)
    if one_shot:
        _run_oneshot(session)
    else:
        _run_tui(session)


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


def usage_bar(tokens: int, session: SessionConfig, width: int = 20) -> str:
    """Return a color-coded usage bar as a string."""
    cw = providers.ensure_context_window(session)
    if not cw:
        return f"{tokens} tokens"
    pct = tokens / cw
    filled = int(width * pct)
    bar = "█" * min(filled, width) + "░" * max(0, width - filled)
    color = "green" if pct < 0.7 else "yellow" if pct < 0.9 else "red"
    return f"[{color}]{bar}[/] {tokens:,}/{cw:,} ({pct:.0%})"
