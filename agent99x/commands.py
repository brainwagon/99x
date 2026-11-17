"""Slash command routing for 99x."""

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Callable, Optional, List

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
    # With handled=False: the text to send to the model in place of what the
    # user typed (skill pseudo-commands expand to an explicit instruction).
    send_text: Optional[str] = None
    # Slow (network) follow-up work. The TUI runs it on a thread worker and
    # shows its result when done; other callers run it inline (see
    # handle_slash_command's ``defer``). ``background_blocks`` marks it busy
    # meanwhile (new messages and mutating commands wait for it).
    background: Optional[Callable[[], "CommandResult"]] = None
    background_label: str = ""
    background_blocks: bool = True


def run_background(result: CommandResult) -> CommandResult:
    """Run any deferred follow-up work inline, merging the messages."""
    while result.background is not None:
        follow = result.background()
        message = "\n".join(m for m in (result.message, follow.message) if m)
        result = dataclasses.replace(follow, message=message)
    return result


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
    from agent99x.prompt import plan_blocked_tools
    session.plan_mode = not session.plan_mode
    if not session.plan_mode:
        return CommandResult(message="Plan mode OFF.")
    blocked = ", ".join(plan_blocked_tools()) or "file writes"
    return CommandResult(
        message=f"Plan mode ON — {blocked} blocked; "
                f"run_bash limited to read-only commands.")


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

def model_hint(session: SessionConfig) -> Optional[str]:
    """One-line advisory for known model quirks, or None."""
    model = (session.model or "").lower()
    hints = []

    if "gemma" in model and "e2b" in model:
        from agent99x import tools
        n = len(tools.TOOLS)
        hints.append(f"Note: E2B-class Gemma models get unreliable at tool selection with "
                     f"10+ tools (this agent exposes {n}). Prefer an E4B model for agentic work.")

    if session.provider:
        pricing = session.provider.fetch_pricing(session)
        if pricing:
            prompt, completion = pricing
            hints.append(f"Cost per 1M tokens: ${prompt:g} input / ${completion:g} output")

    return "\n".join(hints) if hints else None


def activate_provider(session: SessionConfig, name: str, model: Optional[str],
                      host: Optional[str], *, autopick: bool = False) -> None:
    """Point the session at provider ``name`` with ``model`` and ``host``.

    The one place a provider/host/model triple is applied. ``session.host``
    is set (or cleared) *before* the client is rebuilt, so the client never
    keeps a stale host. With no model, the client is rebuilt without any
    network call unless ``autopick`` asks setup_provider to pick the
    server's only model (which lists models, and may raise ValueError).
    Resets per-model state (context window, effort suppression).
    """
    session.host = host or None
    if model or autopick:
        providers.setup_provider(session, name, model, host=session.host)
    else:
        session.provider = providers.PROVIDERS[name]
        session.update_client()
    session.effort_suppressed = False
    session.context_window = 0
    session.context_fetched = False


def _switch_provider(session: SessionConfig, name: str, model: str, host: Optional[str]) -> None:
    """Activate a provider/model/host triple and remember all three."""
    activate_provider(session, name, model, host)
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
        # model_hint may fetch pricing over the network: run it deferred.
        return CommandResult(
            message=f"Switched to {name} / {picked}.",
            background=lambda: CommandResult(message=model_hint(session) or ""),
            background_label="fetching model info…",
            background_blocks=False,
        )
    return _pick


def _prompt_for_model(session: SessionConfig, name: str, *,
                      host: Optional[str]) -> CommandResult:
    """List models for `name` (deferred: it's a network call) and arm the
    picker that selects one."""
    def fetch() -> CommandResult:
        models = providers.list_models_for(name, host=host, timeout=session.http_timeout)
        if not models:
            return CommandResult(
                message=f"No models reported by provider '{name}'. "
                        f"Check the server is running (and OPENROUTER_API_KEY for openrouter)."
            )
        lines = [f"Available models for {name}:"]
        for i, m in enumerate(models, 1):
            lines.append(f"  {i}. {m}")
        lines.append(f"Select [1-{len(models)}] (or /cancel):")
        session.pending_input_handler = _make_pick_handler(name, models, host)
        return CommandResult(message="\n".join(lines))
    return CommandResult(
        message=f"Fetching models for {name}…",
        background=fetch,
        background_label=f"listing {name} models…",
    )


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
            activate_provider(session, name, session.model, None)
        save_config(session)
        return CommandResult(message=f"Cleared saved host for {name}"
                                     f"{' (applied now)' if is_current else ''}.")

    session.provider_hosts[name] = op
    if is_current:
        activate_provider(session, name, session.model, op)
    save_config(session)
    return CommandResult(message=f"Saved host for {name}: {op}"
                                 f"{' (applied now)' if is_current else ''}.")


def _cmd_effort(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    if not arg:
        current = session.effort or "default"
        if session.effort and session.effort_suppressed:
            current = f"{session.effort} (disabled — model does not support it)"
        return CommandResult(message=f"Effort: low|medium|high|none (current: {current})")
    val = arg.strip().lower()
    if val in ("none", "default", "off"):
        session.effort = None
    elif val in ("low", "medium", "high"):
        session.effort = val
    else:
        return CommandResult(message="Usage: /effort [low|medium|high|none]")
    save_config(session)
    return CommandResult(message=f"Effort set to {session.effort or 'default'}.")


def _budget_label(budget: float) -> str:
    return "unlimited" if budget == math.inf else str(int(budget))


def _cmd_budget(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    """Show or set the call budget.

    session.call_budget: None = provider default, math.inf = unlimited
    (an explicit choice that survives provider switches), n = n calls.
    """
    if not arg:
        used = session.shared_state.get("total_calls", 0)
        budget = session.shared_state.get("call_budget", session.effective_call_budget())
        source = ("provider default" if session.call_budget is None
                  else "set with /budget")
        return CommandResult(
            message=f"Budget: {used}/{_budget_label(budget)} calls used ({source}).")
    val = arg.strip().lower()
    if val in ("none", "off", "inf", "infinite", "unlimited"):
        session.call_budget = math.inf
    elif val == "default":
        session.call_budget = None
    else:
        try:
            n = int(val)
            if n <= 0:
                raise ValueError
        except ValueError:
            return CommandResult(message="Usage: /budget [<positive int>|unlimited|default]")
        session.call_budget = n
    session.shared_state["call_budget"] = session.effective_call_budget()
    save_config(session)
    label = _budget_label(session.effective_call_budget())
    if session.call_budget is None:
        label += " (provider default)"
    return CommandResult(message=f"Call budget set to {label}.")


def _cmd_thinking(session: SessionConfig, arg: Optional[str] = None) -> CommandResult:
    if not arg:
        return CommandResult(message=f"Thinking: off|terse|full (current: {session.show_thinking})")

    val = arg.lower().strip()
    if val in ("off", "false", "no", "0"):
        session.show_thinking = "off"
    elif val in ("terse", "brief", "short"):
        session.show_thinking = "terse"
    elif val in ("on", "full", "true", "yes", "1"):
        session.show_thinking = "full"
    else:
        return CommandResult(message="Invalid argument. Use /thinking [off|terse|full].")

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
    "thinking": (_cmd_thinking, "Set thinking display: /thinking [off|terse|full] (no arg shows options)."),
    "budget":   (_cmd_budget,   "Show or set the global call budget: /budget [<n>|unlimited|default]."),
    "cancel":   (_cmd_cancel,   "Cancel a pending interactive prompt."),
}


# Commands that replace session.history or the client. They must not run
# while an agent (or compaction) is using them.
_BUSY_BLOCKED = {"clear", "new", "compact", "provider", "model"}


def _blocked_while_busy(cmd: str, arg: Optional[str]) -> bool:
    if cmd == "provider":
        return bool(arg)  # bare /provider only shows the current setting
    if cmd == "host":
        return bool(arg) and len(arg.split(None, 1)) == 2  # set/clear
    return cmd in _BUSY_BLOCKED


def skill_invocation(name: str, arg: Optional[str]) -> str:
    """The user message a ``/<skill> args`` pseudo-command sends.

    The skill's instructions are inlined so the model needn't decide to call
    load_skill first; if they can't be read, fall back to telling it to.
    """
    from agent99x import scopes
    from agent99x.prompt import _skill_meta
    args = (arg or "").strip() or "(none)"
    path = scopes.resolve("skills", name)
    body = _skill_meta(path)[1].strip() if path else ""
    if not body:
        return (f"Use the `{name}` skill: call load_skill(\"{name}\") and "
                f"follow its instructions.\n\nArguments: {args}")
    return (
        f"Use the `{name}` skill for this request. Its instructions are "
        f"included below (already loaded; no need to call load_skill). Carry "
        f"out its steps now with your tools. Paths to bundled scripts are "
        f"relative to the skill directory `{os.path.dirname(path)}`.\n\n"
        f"Arguments: {args}\n\n"
        f"<skill name=\"{name}\">\n{body}\n</skill>"
    )


def _dispatch(session: SessionConfig, text: str, busy: bool) -> CommandResult:
    pending = session.pending_input_handler
    if pending is not None and not text.startswith("/"):
        if busy:
            return CommandResult(
                message="An agent is running; wait for it (or press Esc) "
                        "before answering the pending prompt.")
        session.pending_input_handler = None
        return pending(session, text)
    if pending is not None:
        session.pending_input_handler = None
    if not text.startswith("/"):
        return CommandResult(handled=False)
    parts = text[1:].split(None, 1) if text[1:2].strip() else []
    if not parts:
        return CommandResult(message="Type /help for the list of commands.")
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else None
    entry = _COMMANDS.get(cmd)
    if entry:
        if busy and _blocked_while_busy(cmd, arg):
            return CommandResult(
                message=f"Cannot run /{cmd} while an agent is running — "
                        f"wait for it to finish or press Esc to cancel.")
        handler, _desc = entry
        return handler(session, arg)

    from agent99x import scopes
    skills = {name for name, _ in scopes.discover("skills")}
    if cmd in skills:
        return CommandResult(handled=False, send_text=skill_invocation(cmd, arg))

    return CommandResult(message=f"Unknown command: /{cmd}. Type /help for list.")


def handle_slash_command(session: SessionConfig, text: str, *,
                         busy: bool = False, defer: bool = False) -> CommandResult:
    """Parse and dispatch slash commands (and pending-prompt answers).

    ``busy``: an agent run is in progress; commands that would replace the
    history or client are refused. ``defer``: leave slow follow-up work in
    ``result.background`` for the caller to run off the UI thread; otherwise
    it runs inline before returning.
    """
    result = _dispatch(session, text, busy)
    return result if defer else run_background(result)
