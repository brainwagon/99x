"""Slash command routing for 99x."""

import math
from dataclasses import dataclass
from typing import Optional, List

from agent99x import providers
from agent99x import todos as todo_md
from agent99x.session import SessionConfig
from agent99x.config_io import save_config


@dataclass
class CommandResult:
    message: str = ""
    clear_history: bool = False
    clear_display: bool = False
    exit_app: bool = False
    compact: bool = False
    handled: bool = True


def _cmd_clear(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    return CommandResult(message="History cleared.", clear_history=True, clear_display=True)


def _cmd_exit(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    return CommandResult(exit_app=True)


def _cmd_help(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    from agent99x import scopes
    from agent99x.prompt import _skill_meta
    skills = []
    for name, path in scopes.discover("skills"):
        desc, _ = _skill_meta(path)
        skills.append((name, desc or "Run this skill."))

    width = max([len(name) for name in _COMMANDS] + [len(name) for name, _ in skills] + [4])

    lines = ["Available commands:"]
    for name, (_, desc) in _COMMANDS.items():
        lines.append(f"  /{name:<{width}}  {desc}")

    if skills:
        lines.append("\nAvailable skills:")
        for name, desc in skills:
            lines.append(f"  /{name:<{width}}  {desc}")

    return CommandResult(message="\n".join(lines))


def _cmd_plan(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    session.plan_mode = not session.plan_mode
    return CommandResult(
        message=f"Plan mode {'ON — write_file blocked; run_bash limited to read-only commands.' if session.plan_mode else 'OFF.'}")


def _cmd_compact(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    return CommandResult(compact=True)


def _cmd_todos(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    todos = todo_md.load()
    if not todos:
        return CommandResult(message="(no todos)")
    lines = ["Current Todos:"]
    for t in todos:
        lines.append(f"  {t.get('id', '?')}. [{todo_md.marker(t['status'])}] {t['text']}")
    return CommandResult(message="\n".join(lines))


def _cmd_todo(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    if not arg:
        return CommandResult(message="Usage: /todo [add|done|rm] text_or_id")

    parts = arg.split(None, 1)
    sub = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""

    todos = todo_md.load()
    msg = ""

    if sub == "add":
        if not rest:
            return CommandResult(message="Usage: /todo add text")
        todos.append({"text": rest, "status": "pending"})
        msg = f"Added: {rest}"
    elif sub in ("done", "rm"):
        if not rest:
            return CommandResult(message=f"Usage: /todo {sub} id_or_text")
        target_idx = -1
        try:
            target_idx = int(rest) - 1
        except ValueError:
            for i, t in enumerate(todos):
                if rest.lower() in t["text"].lower():
                    target_idx = i
                    break

        if 0 <= target_idx < len(todos):
            if sub == "done":
                todos[target_idx]["status"] = "done"
                msg = f"Marked done: {todos[target_idx]['text']}"
            else:
                msg = f"Removed: {todos[target_idx]['text']}"
                todos.pop(target_idx)
        else:
            return CommandResult(message=f"Could not find todo: {rest}")
    else:
        todos.append({"text": arg, "status": "pending"})
        msg = f"Added: {arg}"

    todo_md.save(todos)
    return CommandResult(message=msg)


# ── provider / model selection (simplified) ────────────────────────

def _switch_provider(session: SessionConfig, name: str, model: str, host: Optional[str]) -> None:
    """Activate a provider/model/host triple and remember all three."""
    providers.setup_provider(session, name, model, host=host)
    session.effort_suppressed = False
    if not host:
        session.host = None
    session.context_window = 0
    session.context_fetched = False
    session.provider_models[name] = model
    if host is not None:
        session.provider_hosts[name] = host
    session.shared_state["total_calls"] = 0
    session.shared_state["call_budget"] = session.effective_call_budget()
    save_config(session)


def _make_pick_handler(name: str, models: List[str], host: Optional[str]):
    """Return a pending-input handler that selects a model by index."""
    def _pick(session: SessionConfig, response: str) -> CommandResult:
        raw = response.strip()
        try:
            idx = int(raw) - 1
        except ValueError:
            return CommandResult(message=f"Cancelled: '{raw}' is not a number.")
        if not 0 <= idx < len(models):
            return CommandResult(message=f"Cancelled: pick 1-{len(models)}, got {raw}.")
        picked = models[idx]
        _switch_provider(session, name, picked, host)
        return CommandResult(message=f"Switched to {name} / {picked}.")
    return _pick


def _prompt_for_model(session: SessionConfig, name: str, *, host: Optional[str],
                      note: str = "") -> CommandResult:
    """List models for `name` and arm the picker that selects one."""
    models = providers.list_models_for(name, host=host, timeout=session.http_timeout)
    if not models:
        return CommandResult(
            message=f"{note}No models reported by provider '{name}'. "
                    f"Check the server is running (and OPENROUTER_API_KEY for openrouter)."
        )
    lines = [f"Available models for {name}:"]
    for i, m in enumerate(models, 1):
        lines.append(f"  {i}. {m}")
    lines.append(f"Select [1-{len(models)}] (or /cancel):")
    session.pending_input_handler = _make_pick_handler(name, models, host)
    return CommandResult(message=note + "\n".join(lines))


def _make_provider_pick_handler():
    """Return a pending-input handler that routes a typed provider name into /provider."""
    def _pick(session: SessionConfig, response: str) -> CommandResult:
        name = response.strip().lower()
        if name not in providers.PROVIDERS:
            known = ", ".join(sorted(providers.PROVIDERS))
            session.pending_input_handler = _make_provider_pick_handler()
            return CommandResult(
                message=f"Unknown provider '{name}'. Known: {known}\nWhich provider? (or /cancel):")
        return _cmd_provider(session, name)
    return _pick


def prompt_for_provider(session: SessionConfig) -> CommandResult:
    """List known providers and arm the picker, asking which to use (startup path)."""
    known = ", ".join(sorted(providers.PROVIDERS))
    session.pending_input_handler = _make_provider_pick_handler()
    return CommandResult(
        message=f"No provider/model configured. Known providers: {known}\n"
                f"Which provider? (or /cancel):")


def _cmd_provider(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    if not arg:
        known = ", ".join(sorted(providers.PROVIDERS))
        current_model = session.model or "(unset)"
        host_note = f" (host {session.host})" if session.host else ""
        current_name = session.provider.name if session.provider else "(unset)"
        return CommandResult(
            message=(f"Current: {current_name} / {current_model}{host_note}\n"
                     f"Known providers: {known}\n"
                     f"Usage: /provider <name>"))
    name = arg.strip().lower()
    if name not in providers.PROVIDERS:
        known = ", ".join(sorted(providers.PROVIDERS))
        return CommandResult(message=f"Unknown provider '{name}'. Known: {known}")
    host = session.provider_hosts.get(name)
    return _prompt_for_model(session, name, host=host)


def _cmd_model(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    """List models for the current provider and arm the picker."""
    if not session.provider:
        return CommandResult(message="No provider set. Use /provider <name> first.")
    return _prompt_for_model(session, session.provider.name, host=session.host)


def _cmd_host(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    """Manage the saved per-provider host override.

    /host                        list all saved hosts
    /host <provider> <addr>      set the host (applies now if it's the current provider)
    /host <provider> clear       remove the saved host (applies now if current)
    """
    if not arg:
        if not session.provider_hosts:
            return CommandResult(message="No saved hosts. Use /host <provider> <addr> to set one.")
        lines = ["Saved hosts:"]
        current_name = session.provider.name if session.provider else None
        for name in sorted(session.provider_hosts):
            host = session.provider_hosts[name]
            marker = " (current)" if name == current_name else ""
            lines.append(f"  {name}: {host or '(none)'}{marker}")
        return CommandResult(message="\n".join(lines))

    parts = arg.split(None, 1)
    name = parts[0].lower()
    if name not in providers.PROVIDERS:
        known = ", ".join(sorted(providers.PROVIDERS))
        return CommandResult(message=f"Unknown provider '{name}'. Known: {known}")
    current_name = session.provider.name if session.provider else None
    is_current = (name == current_name)

    if len(parts) == 1:
        host = session.provider_hosts.get(name)
        return CommandResult(message=f"Saved host for {name}: {host or '(none)'}")

    op = parts[1].strip()
    if op.lower() == "clear":
        session.provider_hosts.pop(name, None)
        if is_current:
            providers.setup_provider(session, name, session.model, host=None)
            session.host = None
            session.context_window = 0
            session.context_fetched = False
        save_config(session)
        return CommandResult(message=f"Cleared saved host for {name}"
                                     f"{' (applied now)' if is_current else ''}.")

    session.provider_hosts[name] = op
    if is_current:
        providers.setup_provider(session, name, session.model, host=op)
        session.context_window = 0
        session.context_fetched = False
    save_config(session)
    return CommandResult(message=f"Saved host for {name}: {op}"
                                 f"{' (applied now)' if is_current else ''}.")


def _cmd_effort(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    if not arg:
        if session.effort and session.effort_suppressed:
            return CommandResult(message=f"Effort: {session.effort} (disabled — model does not support it)")
        return CommandResult(message=f"Effort: {session.effort or 'default'}")
    val = arg.strip().lower()
    if val in ("none", "default", "off"):
        session.effort = None
    elif val in ("low", "medium", "high"):
        session.effort = val
    else:
        return CommandResult(message="Usage: /effort [low|medium|high|none]")
    save_config(session)
    return CommandResult(message=f"Effort set to {session.effort or 'default'}.")


def _cmd_budget(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    if not arg:
        used = session.shared_state.get("total_calls", 0)
        budget = session.shared_state.get("call_budget", session.call_budget)
        budget_str = "unlimited" if budget in (None, math.inf) else str(int(budget))
        return CommandResult(message=f"Budget: {used}/{budget_str} calls used.")
    val = arg.strip().lower()
    if val in ("none", "off", "inf", "infinite", "unlimited"):
        session.call_budget = None
        session.shared_state["call_budget"] = math.inf
    else:
        try:
            n = int(val)
            if n <= 0:
                raise ValueError
        except ValueError:
            return CommandResult(message="Usage: /budget [<positive int>|unlimited]")
        session.call_budget = n
        session.shared_state["call_budget"] = n
    save_config(session)
    label = "unlimited" if session.call_budget is None else session.call_budget
    return CommandResult(message=f"Call budget set to {label}.")


def _cmd_thinking(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    if arg:
        val = arg.lower().strip()
        if val in ("off", "false", "no", "0"):
            session.show_thinking = "off"
        elif val in ("terse", "brief", "short"):
            session.show_thinking = "terse"
        elif val in ("on", "full", "true", "yes", "1"):
            session.show_thinking = "full"
        else:
            return CommandResult(message="Invalid argument. Use /thinking [off|terse|full].")
    else:
        if session.show_thinking == "off":
            session.show_thinking = "terse"
        elif session.show_thinking == "terse":
            session.show_thinking = "full"
        else:
            session.show_thinking = "off"

    save_config(session)
    return CommandResult(message=f"Thinking display is now {session.show_thinking}.")


def _cmd_cancel(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    return CommandResult(message="Cancelled.")


_COMMANDS = {
    "clear":    (_cmd_clear,    "Clear conversation history and screen."),
    "new":      (_cmd_clear,    "Alias for /clear."),
    "exit":     (_cmd_exit,     "Exit the agent."),
    "quit":     (_cmd_exit,     "Alias for /exit."),
    "help":     (_cmd_help,     "Show this list of commands."),
    "plan":     (_cmd_plan,     "Toggle plan mode (blocks writes; restricts shell to read-only)."),
    "compact":  (_cmd_compact,  "Summarize history to free context."),
    "todos":    (_cmd_todos,    "List current todos."),
    "todo":     (_cmd_todo,     "Manage todos: /todo [add|done|rm] text_or_id."),
    "provider": (_cmd_provider, "Switch LLM provider mid-conversation: /provider <name>."),
    "model":    (_cmd_model,    "Pick a different model on the current provider."),
    "host":     (_cmd_host,     "Manage per-provider host: /host <provider> [<addr>|clear]."),
    "effort":   (_cmd_effort,   "Set reasoning effort: /effort [low|medium|high|none]."),
    "thinking": (_cmd_thinking, "Toggle thinking display: /thinking [off|terse|full] (or cycle with no arg)."),
    "budget":   (_cmd_budget,   "Show or set the global call budget: /budget [<n>|unlimited]."),
    "cancel":   (_cmd_cancel,   "Cancel a pending interactive prompt."),
}


def handle_slash_command(session: SessionConfig, text: str) -> CommandResult:
    """Parse and dispatch slash commands."""
    pending = session.pending_input_handler
    if pending is not None and not text.startswith("/"):
        session.pending_input_handler = None
        return pending(session, text)
    if pending is not None:
        session.pending_input_handler = None
    if not text.startswith("/"):
        return CommandResult(handled=False)
    parts = text[1:].split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None
    entry = _COMMANDS.get(cmd)
    if entry:
        handler, _desc = entry
        return handler(session, arg)

    from agent99x import scopes
    skills = {name for name, _ in scopes.discover("skills")}
    if cmd in skills:
        return CommandResult(handled=False)

    return CommandResult(message=f"Unknown command: /{cmd}. Type /help for list.")
