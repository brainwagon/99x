"""TUI application for 99x — Textual-based chat interface."""

import asyncio
import dataclasses
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from rich.markdown import Markdown
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input, RichLog, Static

from agent99x import core
from agent99x import logs
from agent99x import cli
from agent99x import conversation
from agent99x import compaction
from agent99x import commands
from agent99x.hooks import AgentHooks
from agent99x.reasoning import ContentType, ThinkingRenderer
from agent99x.session import SessionConfig


class TuiLogHandler(logging.Handler):
    def __init__(self, app: "AgentApp"):
        super().__init__()
        self.app = app

    def _dispatch(self, method, *args):
        """Route a UI mutation to the app from any thread."""
        try:
            self.app.call_from_thread(method, *args)
        except RuntimeError:
            method(*args)

    def emit(self, record: logging.LogRecord):
        try:
            self._emit(record)
        except Exception:
            # Never let a UI problem (e.g. the app has exited) surface in the
            # code that logged.
            self.handleError(record)

    def _emit(self, record: logging.LogRecord):
        msg = self.format(record)
        log_type = getattr(record, "type", None)

        if log_type == "tool_call":
            name = getattr(record, "tool_name", "unknown")
            args = getattr(record, "tool_args", {})
            self._dispatch(self.app.handle_log_tool_call, name, args)
        elif log_type == "tool_result":
            name = getattr(record, "tool_name", "unknown")
            result = getattr(record, "result", "")
            elapsed = getattr(record, "elapsed", 0.0)
            self._dispatch(self.app.handle_log_tool_result, name, result, elapsed)
        else:
            if record.levelno >= logging.WARNING:
                self._dispatch(self.app._write, f"[bold red]{escape(msg)}[/bold red]")
            else:
                self._dispatch(self.app._write, f"[dim]{escape(msg)}[/dim]")


_URL_RE = re.compile(r'(https?://\S+)')


def _linkify(text: str) -> str:
    parts = _URL_RE.split(text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            url = part.rstrip("`.,;'\"\\)>]")
            suffix = part[len(url):]
            out.append(f"[link={url}]{escape(url)}[/link]{escape(suffix)}")
        else:
            out.append(escape(part))
    return "".join(out)


_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_BANNER = "\n".join([
    "        ██        ██        ",
    "          ██    ██          ",
    "        ██████████████      ",
    "      ████  ██████  ████    ",
    "    ██████████████████████  ",
    "    ██  ██████████████  ██  ",
    "    ██  ██          ██  ██  ",
    "          ████  ████        ",
])

_BANNER_COLORS = ["#3aff3a"] * 8
_BANNER_CREDIT = [
    "",
    "",
    "  a streamlined Python AI agent",
    "",
    "  Created by Mark VandeWettering",
    "  <mvandewettering@gmail.com>  ·  2026",
    "",
    "",
]


def _banner_markup() -> str:
    banner_lines = _BANNER.splitlines()
    width = max(len(line) for line in banner_lines)
    lines = []
    for i, line in enumerate(banner_lines):
        color = _BANNER_COLORS[i % len(_BANNER_COLORS)]
        padded = line.ljust(width + 2)
        credit = _BANNER_CREDIT[i] if i < len(_BANNER_CREDIT) else ""
        lines.append(f"[bold {color}]{escape(padded)}[/][dim]{escape(credit)}[/]")
    return "\n".join(lines)


@dataclass
class WorkerState:
    worker_id: int
    cancel: threading.Event
    state: str = "starting…"
    start_time: float = field(default_factory=time.monotonic)
    last_reply: str = ""
    # "agent" (agent_loop), "compact", or "command" (a slash command's
    # deferred network work). Esc drops compact/command workers at once
    # (their results are discarded); agent workers stop at the next check.
    kind: str = "agent"
    # Set by the worker thread once it no longer touches session state.
    done: threading.Event = field(default_factory=threading.Event)


# How long exiting waits for cancelled workers before saving anyway.
_EXIT_WAIT_SECONDS = 2.0


@dataclass
class _StreamCtx:
    """Mutable streaming state shared between agent_loop callbacks."""
    token_buffer: list = field(default_factory=list)
    thinking_capture: list = field(default_factory=list)
    renderer: Optional[ThinkingRenderer] = None


class AgentApp(App):
    BINDINGS = [
        Binding("ctrl+r", "toggle_markdown", "Render markdown", priority=True),
        Binding("ctrl+t", "toggle_thinking", "Toggle thinking", priority=True),
        Binding("up", "history_up", "Previous command", show=False),
        Binding("down", "history_down", "Next command", show=False),
        Binding("ctrl+p", "history_up", "Previous command", show=False),
        Binding("ctrl+n", "history_down", "Next command", show=False),
    ]

    CSS = """
    Screen { layout: vertical; }
    #header { height: 1; padding: 0 1; color: $text; background: $primary-darken-2; }
    #log  { height: 1fr; }
    #status { height: 1; padding: 0 1; color: $text-muted; background: $surface-darken-1; }
    """

    def __init__(
        self,
        session: SessionConfig,
        hooks: Optional[AgentHooks] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.session = session
        self.agent_hooks = hooks
        self._exiting = False
        self._pricing_cache: Dict[tuple, Optional[tuple]] = {}
        self._pricing_pending: set = set()

    def compose(self) -> ComposeResult:
        yield Static("", id="header")
        yield RichLog(id="log", markup=True, wrap=True)
        yield Static("", id="status")
        yield Input(placeholder="Message…", id="input")

    def on_mount(self) -> None:
        # History was already seeded by cli._setup_context (fresh or restored);
        # only fall back to a fresh start if something left it empty.
        if not self.session.history:
            conversation.start(self.session)

        self._agent_workers: Dict[int, WorkerState] = {}
        self._agent_workers_lock = threading.Lock()
        self._worker_id_counter = 0
        self._last_usage = 0
        self._cmd_history: List[str] = []
        self._cmd_idx = -1
        self._saved_draft = ""
        self._frame = 0
        self._notice_until = 0.0
        self._notice_msg = ""
        self._log_entries: list = []
        self._render_md = False
        self._thinking_entries: Dict[int, str] = {}
        self._thinking_expanded: set = set()

        # Setup structured logging
        logs.setup_logging(level=logging.INFO)
        self._log_handler = TuiLogHandler(self)
        logs.logger.addHandler(self._log_handler)
        for h in logs.logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, TuiLogHandler):
                logs.logger.removeHandler(h)

        self._write(_banner_markup())
        self._write("[dim]run_bash is UNSANDBOXED[/dim]")
        trace_path = self.session.shared_state.get("hook_trace_path")
        if trace_path:
            self._write(f"[dim]Hook trace: {escape(trace_path)}[/dim]")
        # model_hint may fetch OpenRouter pricing: keep it off the UI thread.
        self.run_worker(self._startup_hint_worker, thread=True,
                        group="startup-hint", exclusive=False)
        if len(self.session.history) > 1:
            self._write(f"[dim](resuming session, {len(self.session.history) - 1} messages)[/dim]")
        self._update_header()
        if not cli.provider_ready(self.session):
            result = commands.prompt_for_provider(self.session)
            self._write(f"[dim]{escape(result.message)}[/dim]")
        self.set_interval(1 / 12, self._tick)
        self.query_one(Input).focus()

    def _post(self, method, *args) -> None:
        """call_from_thread that tolerates the app having already exited."""
        try:
            self.call_from_thread(method, *args)
        except RuntimeError:
            pass

    def _startup_hint_worker(self) -> None:
        try:
            hint = commands.model_hint(self.session)
        except Exception:
            hint = None
        if hint:
            self._post(self._write, f"[dim]{escape(hint)}[/dim]")

    def handle_log_tool_call(self, name: str, args: dict) -> None:
        preview = ", ".join(f"{k}={core.clip(str(v), 40, '…')}" for k, v in args.items())
        self._write(f"[dim yellow]  ⚙ {escape(name)}({escape(preview)})[/dim yellow]")

    def handle_log_tool_result(self, name: str, result: str, elapsed: float) -> None:
        self._write(f"[dim]    → {escape(core.clip(result, 80, '…'))}  ({elapsed:.2f}s)[/dim]")

    def _write(self, payload) -> None:
        """Write a renderable to the log and remember it for redraws."""
        self.query_one(RichLog).write(payload)
        self._log_entries.append(payload)

    def _write_agent(self, reply: str, wid: Optional[int] = None,
                     run_id: Optional[int] = None) -> None:
        """Write an agent reply, respecting the current render mode.

        ``wid`` labels the reply (only shown with concurrent workers);
        ``run_id`` is the worker id its thinking is recorded under.
        """
        log = self.query_one(RichLog)
        header = f"\n[bold green][{wid}][/bold green]" if wid is not None else "\n"

        if run_id is not None and run_id in self._thinking_entries:
            thinking_text = self._thinking_entries[run_id]
            word_count = len(thinking_text.split())
            if run_id in self._thinking_expanded:
                log.write(f"[dim]▼ Thinking ({word_count} words) — Ctrl+T to collapse[/dim]")
                log.write("\n".join(f"[dim]{line}[/dim]" for line in escape(thinking_text).split("\n")))
            else:
                log.write(f"[dim]▶ Thinking ({word_count} words) — Ctrl+T to expand[/dim]")

        if self._render_md:
            log.write(header)
            log.write(Markdown(reply or ""))
        else:
            log.write(f"{header}{_linkify(reply or '')}")
        self._log_entries.append(("agent", reply or "", wid, run_id))

    def _redraw_log(self) -> None:
        log = self.query_one(RichLog)
        log.clear()
        entries, self._log_entries = self._log_entries, []
        for entry in entries:
            if isinstance(entry, tuple) and entry and entry[0] == "agent":
                self._write_agent(*entry[1:])
            else:
                self._write(entry)

    def action_toggle_markdown(self) -> None:
        self._render_md = not self._render_md
        self._redraw_log()
        self._notice_msg = f"Markdown {'ON' if self._render_md else 'OFF'}"
        self._notice_until = time.monotonic() + 1.5

    def action_history_up(self) -> None:
        input_widget = self.query_one(Input)
        if not input_widget.has_focus:
            return
        if self._cmd_idx < len(self._cmd_history) - 1:
            if self._cmd_idx == -1:
                self._saved_draft = input_widget.value
            self._cmd_idx += 1
            input_widget.value = self._cmd_history[self._cmd_idx]
            input_widget.cursor_position = len(input_widget.value)

    def action_history_down(self) -> None:
        input_widget = self.query_one(Input)
        if not input_widget.has_focus:
            return
        if self._cmd_idx >= 0:
            self._cmd_idx -= 1
            if self._cmd_idx == -1:
                input_widget.value = self._saved_draft
            else:
                input_widget.value = self._cmd_history[self._cmd_idx]
            input_widget.cursor_position = len(input_widget.value)

    def action_toggle_thinking(self) -> None:
        if not self._thinking_entries:
            return
        last_wid = max(self._thinking_entries.keys())
        if last_wid in self._thinking_expanded:
            self._thinking_expanded.discard(last_wid)
            msg = "Thinking collapsed"
        else:
            self._thinking_expanded.add(last_wid)
            msg = "Thinking expanded"
        self._notice_msg = msg
        self._notice_until = time.monotonic() + 1.5
        self._redraw_log()

    def _any_busy(self) -> bool:
        with self._agent_workers_lock:
            return bool(self._agent_workers)

    def _update_header(self) -> None:
        cw = self.session.context_window
        if self._last_usage and cw:
            pct = self._last_usage / cw
            ctx = f"  ·  {self._last_usage:,}/{cw:,} ({pct:.0%})"
        elif self._last_usage:
            ctx = f"  ·  {self._last_usage:,} tokens"
        else:
            ctx = f"  ·  {cw:,} ctx" if cw else ""
        if self.session.effort and self.session.effort_suppressed:
            effort = f"  ·  effort:{self.session.effort} (disabled)"
        elif self.session.effort:
            effort = f"  ·  effort:{self.session.effort}"
        else:
            effort = ""
        thinking = f"  ·  thinking:{self.session.show_thinking}"
        cost_str = ""
        if self.session.provider and self.session.provider.name == "openrouter":
            pricing = self._cached_pricing()
            if pricing:
                in_cost, out_cost = pricing
                total_in = self.session.shared_state.get("total_input_tokens", 0)
                total_out = self.session.shared_state.get("total_output_tokens", 0)
                cost = (total_in * in_cost / 1_000_000) + (total_out * out_cost / 1_000_000)
                if cost > 0:
                    cost_str = f"  ·  ${cost:.4f}"

        self.query_one("#header", Static).update(
            f"{self.session.provider.name if self.session.provider else '(no provider)'}  ·  {self.session.model}{ctx}{cost_str}{effort}{thinking}"
        )

    def _cached_pricing(self) -> Optional[tuple]:
        """Pricing for the current model without blocking the UI thread.

        Returns the cached value, or None while a thread worker fetches it.
        """
        key = (self.session.provider.name, self.session.model)
        if key in self._pricing_cache:
            return self._pricing_cache[key]
        if key not in self._pricing_pending:
            self._pricing_pending.add(key)
            provider = self.session.provider

            def fetch() -> None:
                try:
                    value = provider.fetch_pricing(self.session)
                except Exception:
                    value = None
                # Drop a result that raced a model switch; it will be refetched.
                if (self.session.provider is provider
                        and self.session.model == key[1]):
                    self._pricing_cache[key] = value
                self._pricing_pending.discard(key)

            self.run_worker(fetch, thread=True, group="pricing", exclusive=False)
        return None

    def _tick(self) -> None:
        self._update_header()
        now = time.monotonic()
        status = self.query_one("#status", Static)

        git_branch = ""
        try:
            with open(".git/HEAD", "r") as f:
                head = f.read().strip()
                if head.startswith("ref: refs/heads/"):
                    git_branch = f"git:{head[16:]}"
                else:
                    git_branch = f"git:{head[:7]}"
        except Exception:
            pass

        with self._agent_workers_lock:
            workers = list(self._agent_workers.values())
        if not workers:
            if now < self._notice_until:
                status.update(f"{self._notice_msg}  ·  {git_branch}" if git_branch else self._notice_msg)
            else:
                status.update(git_branch)
            return
        ch = _SPINNER[self._frame % len(_SPINNER)]
        self._frame += 1
        parts = []
        for worker_state in workers:
            elapsed = now - worker_state.start_time
            parts.append(f"{ch} [{worker_state.worker_id}] {worker_state.state}  [{elapsed:.1f}s]")
        status_text = "   ".join(parts)
        if git_branch:
            status_text += f"  ·  {git_branch}"
        status.update(status_text)

    def on_key(self, event) -> None:
        if event.key == "escape":
            input_widget = self.query_one(Input)
            if not input_widget.has_focus or not input_widget.value.strip():
                for dropped in self._cancel_all():
                    label = "compaction" if dropped.kind == "compact" else "command"
                    self._write(f"[dim yellow]  ✗ {label} cancelled[/dim yellow]")
        elif event.key == "ctrl+y":
            with self._agent_workers_lock:
                replies = [ws.last_reply for ws in self._agent_workers.values() if ws.last_reply]
            reply = replies[-1] if replies else getattr(self, "_last_completed_reply", "")
            if reply:
                self.copy_to_clipboard(reply)
                self._notice_msg = "Copied to clipboard"
                self._notice_until = time.monotonic() + 2.0

    def _cancel_all(self) -> List[WorkerState]:
        """Signal every worker to stop. Compaction and command workers are
        dropped at once (their late results are discarded) and returned;
        agent workers show "cancelling…" until agent_loop notices."""
        dropped = []
        with self._agent_workers_lock:
            for wid, worker_state in list(self._agent_workers.items()):
                worker_state.cancel.set()
                if worker_state.kind == "agent":
                    worker_state.state = "cancelling…"
                else:
                    dropped.append(self._agent_workers.pop(wid))
        return dropped

    def _register_worker(self, kind: str, state: str) -> WorkerState:
        worker_state = WorkerState(worker_id=self._next_worker_id(),
                                   cancel=threading.Event(), state=state, kind=kind)
        with self._agent_workers_lock:
            self._agent_workers[worker_state.worker_id] = worker_state
        return worker_state

    async def _graceful_exit(self) -> None:
        """Cancel running work, give it a bounded moment to stop, then exit
        (on_unmount saves the context)."""
        if self._exiting:
            return
        self._exiting = True
        self._cancel_all()
        deadline = time.monotonic() + _EXIT_WAIT_SECONDS
        while time.monotonic() < deadline:
            with self._agent_workers_lock:
                running = [ws for ws in self._agent_workers.values()
                           if not ws.done.is_set()]
            if not running:
                break
            await asyncio.sleep(0.05)
        self.exit()

    async def action_quit(self) -> None:
        await self._graceful_exit()

    def _start_background(self, result: "commands.CommandResult") -> None:
        """Run a slash command's deferred (network) work on a thread worker."""
        job = result.background
        worker_state = None
        if result.background_blocks:
            worker_state = self._register_worker(
                "command", result.background_label or "working…")
        self.run_worker(lambda: self._background_worker(worker_state, job),
                        thread=True, group="command", exclusive=False)

    def _background_worker(self, worker_state: Optional[WorkerState], job) -> None:
        pending_before = self.session.pending_input_handler
        error: Optional[str] = None
        follow = None
        try:
            follow = job()
        except Exception as e:
            error = str(e)
        finally:
            if worker_state is not None:
                worker_state.done.set()

        def finish() -> None:
            if worker_state is not None:
                with self._agent_workers_lock:
                    still = self._agent_workers.pop(worker_state.worker_id, None)
                if still is None:
                    # Cancelled with Esc: discard, and un-arm any picker it set.
                    self.session.pending_input_handler = pending_before
                    return
            if error is not None:
                self._write(f"[red]Error:[/red] {escape(error)}")
            elif follow is not None:
                if follow.message:
                    self._write(f"[dim]{escape(follow.message)}[/dim]")
                if follow.background is not None:
                    self._start_background(follow)
            self.query_one(Input).focus()

        self._post(finish)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        try:
            text = event.value.strip()
            if not text:
                return
            if not self._cmd_history or self._cmd_history[0] != text:
                self._cmd_history.insert(0, text)
            self._cmd_idx = -1
            self._saved_draft = ""

            self.query_one(Input).clear()
            busy = self._any_busy()
            result = cli.handle_slash_command(self.session, text, busy=busy, defer=True)
            if result.handled:
                if result.message:
                    self._write(f"[dim]{escape(result.message)}[/dim]")
                if result.clear_history:
                    conversation.clear(self.session)
                    self._last_usage = 0
                if result.clear_display:
                    self.query_one(RichLog).clear()
                    self._log_entries = []
                    self._thinking_entries.clear()
                    self._thinking_expanded.clear()
                if result.compact:
                    if self._any_busy():
                        self._write("[dim]Cannot compact while an agent is running.[/dim]")
                    elif len(self.session.history) > 2:
                        worker_state = self._register_worker("compact", "compacting…")
                        self.run_worker(lambda: self._compact_worker(worker_state.worker_id),
                                        thread=True, group=f"agent-{worker_state.worker_id}",
                                        exclusive=False)
                    else:
                        self._write("[dim]Nothing to compact.[/dim]")
                if result.background is not None:
                    self._start_background(result)
                if result.exit_app:
                    self.run_worker(self._graceful_exit(), group="exit", exclusive=True)
                return

            if busy:
                self._write("[dim yellow]Busy — press Esc to cancel.[/dim yellow]")
                return

            snapshot = len(self.session.history)
            media_paths, media_errors = conversation.add_user(
                self.session, result.send_text or text)
            clean_text = re.sub(r'attach:\S+', '', text).strip()
            media_tags = "  ".join(
                f"[dim cyan]{escape('[attachment: ' + os.path.basename(p) + ']')}[/dim cyan]"
                for p in media_paths
            )
            display = escape(clean_text) + (f"  {media_tags}" if media_tags else "")
            self._write(f"\n[bold cyan]You:[/bold cyan] {display}")
            for err in media_errors:
                self._write(f"[red]{escape(err)}[/red]")

            worker_state = self._register_worker("agent", "starting…")
            worker_id = worker_state.worker_id
            self.run_worker(lambda: self._agent_worker(worker_id, user_snapshot=snapshot),
                            thread=True, group=f"agent-{worker_id}", exclusive=False)
        except Exception as e:
            self._write(f"[red]Error:[/red] {escape(str(e))}")

    def _next_worker_id(self) -> int:
        self._worker_id_counter += 1
        return self._worker_id_counter

    def _flush_lines(self, lines: list) -> None:
        """Write accumulated lines to the log (runs on UI thread)."""
        log = self.query_one(RichLog)
        for line in lines:
            log.write(line)

    def _on_token(self, ctx: _StreamCtx, ctype: ContentType, text: str) -> None:
        """Handle a streaming token from agent_loop (runs on worker thread)."""
        if ctype == ContentType.THINKING:
            ctx.thinking_capture.append(text)
        for kind, piece in ctx.renderer.feed(ctype, text):
            if kind == ContentType.THINKING:
                escaped = escape(piece)
                markup = "\n".join(f"[dim]{line}[/dim]" for line in escaped.split("\n"))
            else:
                markup = escape(piece)
            ctx.token_buffer.append(markup)
        full_text = "".join(ctx.token_buffer)
        if "\n" in full_text:
            lines = full_text.split("\n")
            ctx.token_buffer.clear()
            if lines[-1]:
                ctx.token_buffer.append(lines[-1])
            self.call_from_thread(self._flush_lines, lines[:-1])

    def _finish_agent_run(
        self,
        worker_id: int,
        reply: str,
        agent_error: Optional[str],
        user_snapshot: Optional[int],
        ctx: _StreamCtx,
        autocompact_result,
        unstreamed: Optional[str] = None,
    ) -> None:
        """Clean up after agent_loop completes (runs on UI thread via call_from_thread).

        ``unstreamed`` is the reply when agent_loop returned text that never
        streamed (budget/iteration limits, escalations), so it must be shown.
        """
        with self._agent_workers_lock:
            busy_count = len(self._agent_workers)
            worker_state = self._agent_workers.pop(worker_id, None)

        prefix = f"[dim cyan][{worker_id}][/dim cyan] " if busy_count > 1 else ""
        if agent_error is not None:
            self._write(f"[red]  {prefix}Agent error:[/red] {escape(agent_error)}")
            if user_snapshot is not None:
                conversation.truncate_to(self.session, user_snapshot)
        elif reply == "(cancelled)":
            self._write(f"[dim yellow]  {prefix}✗ cancelled[/dim yellow]")
        else:
            if worker_state:
                worker_state.last_reply = reply or ""
            self._last_completed_reply = reply or ""
            label = worker_id if prefix else None
            self._log_entries.append(("agent", reply or "", label, worker_id))
            captured = "".join(ctx.thinking_capture)
            should_redraw = self._render_md
            if captured and self.session.show_thinking == "full":
                # Record it so the redraw collapses the streamed thinking to
                # a summary line that Ctrl+T expands.
                self._thinking_entries[worker_id] = captured
                should_redraw = True
            if should_redraw:
                self._redraw_log()
            elif unstreamed:
                self.query_one(RichLog).write(_linkify(unstreamed))

            if reply:
                match = re.search(r"\[RECOMMENDED:\s*(.+?)\]", reply)
                if match:
                    recommended = match.group(1).strip()
                    input_widget = self.query_one(Input)
                    if not input_widget.value:
                        input_widget.value = recommended
                        input_widget.cursor_position = len(recommended)
        if autocompact_result:
            old, new, _prompt_tokens, summary_tokens = autocompact_result
            self._last_usage = summary_tokens
            self.session.shared_state["total_input_tokens"] = self.session.shared_state.get("total_input_tokens", 0) + _prompt_tokens
            self.session.shared_state["total_output_tokens"] = self.session.shared_state.get("total_output_tokens", 0) + summary_tokens
            self._write(f"[dim]  auto-compacted: {old} → {new} messages[/dim]")
        # Context is persisted on exit (on_unmount), not per turn.
        self.query_one(Input).focus()

    def _agent_worker(self, worker_id: int, user_snapshot: Optional[int] = None) -> None:
        with self._agent_workers_lock:
            worker_state = self._agent_workers[worker_id]

        ctx = _StreamCtx(renderer=ThinkingRenderer(self.session.show_thinking))
        tracker = cli.ReplyTracker()

        def on_model_request() -> None:
            tracker.on_model_request()
            worker_state.state = "waiting for model…"

        def on_token(ctype: ContentType, text: str) -> None:
            tracker.on_token(ctype, text)
            self._on_token(ctx, ctype, text)

        def on_tool_call(name: str, args: dict) -> None:
            preview = ", ".join(f"{k}={core.clip(str(v), 40, '…')}" for k, v in args.items())
            worker_state.state = f"tool: {name}({core.clip(preview, 55, '…')})"

        def on_usage(usage_tokens: tuple[int, int]) -> None:
            p_tok, c_tok = usage_tokens
            def _update():
                self._last_usage = p_tok + c_tok
                self.session.shared_state["total_input_tokens"] = self.session.shared_state.get("total_input_tokens", 0) + p_tok
                self.session.shared_state["total_output_tokens"] = self.session.shared_state.get("total_output_tokens", 0) + c_tok
            self.call_from_thread(_update)

        worker_state.state = "waiting for model…"
        with self._agent_workers_lock:
            busy_count = len(self._agent_workers)
        wid_label = worker_id if busy_count > 1 else None
        header = f"\n[bold green][{wid_label}][/bold green] " if wid_label is not None else "\n"
        self.call_from_thread(self.query_one(RichLog).write, header)

        agent_error: Optional[str] = None
        try:
            reply = core.agent_loop(
                self.session,
                cancel=worker_state.cancel,
                on_model_request=on_model_request,
                on_tool_call=on_tool_call,
                on_usage=on_usage,
                on_token=on_token,
                hooks=self.agent_hooks,
            )
            if ctx.token_buffer:
                self.call_from_thread(self._flush_lines, ["".join(ctx.token_buffer)])
        except Exception as e:
            reply = ""
            agent_error = str(e)

        autocompact_result = None
        if agent_error is None and reply != "(cancelled)":
            try:
                worker_state.state = "auto-compacting…"
                autocompact_result = compaction.maybe_autocompact(self.session, self._last_usage)
            except Exception:
                autocompact_result = None

        unstreamed = tracker.unstreamed(reply) if agent_error is None else None
        worker_state.done.set()
        self._post(
            self._finish_agent_run,
            worker_id, reply, agent_error, user_snapshot, ctx, autocompact_result,
            unstreamed,
        )

    def _compact_worker(self, worker_id: int) -> None:
        """Compact a copy of the history; adopt it only if not cancelled.

        do_compact is one blocking model call that Esc can't interrupt, so
        Esc drops this worker at once (see _cancel_all) and the late result
        is discarded instead of replacing the live history.
        """
        with self._agent_workers_lock:
            worker_state = self._agent_workers.get(worker_id)
        work = dataclasses.replace(self.session, history=list(self.session.history))
        error: Optional[str] = None
        stats = None
        try:
            stats = compaction.do_compact(work)
        except Exception as e:
            error = str(e)
        finally:
            if worker_state is not None:
                worker_state.done.set()

        def finish() -> None:
            with self._agent_workers_lock:
                still = self._agent_workers.pop(worker_id, None)
            if still is None or still.cancel.is_set():
                return  # cancelled: discard the result
            if error is not None:
                self._write(f"[red]Compact failed: {escape(error)}[/red]")
            else:
                old, new, prompt_tokens, summary_tokens = stats
                self.session.history[:] = work.history
                self._last_usage = summary_tokens
                self.session.shared_state["total_input_tokens"] = self.session.shared_state.get("total_input_tokens", 0) + prompt_tokens
                self.session.shared_state["total_output_tokens"] = self.session.shared_state.get("total_output_tokens", 0) + summary_tokens
                self._write(f"[dim]Compacted: {old} → {new} messages  (~{prompt_tokens:,} → ~{summary_tokens:,} tokens)[/dim]")
            self.query_one(Input).focus()

        self._post(finish)

    def on_unmount(self) -> None:
        # Covers exits that bypass _graceful_exit: stop workers, give them a
        # brief bounded chance to finish, then save what we have.
        self._cancel_all()
        deadline = time.monotonic() + 0.5
        with self._agent_workers_lock:
            running = list(self._agent_workers.values())
        for worker_state in running:
            worker_state.done.wait(max(0.0, deadline - time.monotonic()))
        handler = getattr(self, "_log_handler", None)
        if handler is not None:
            logs.logger.removeHandler(handler)
        cli.save_session(self.session)
