"""TUI application for 99x — Textual-based chat interface."""

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
    " ██████   ██████ ",
    "██    ██ ██    ██",
    "██    ██ ██    ██",
    " ███████  ███████",
    "      ██       ██",
    "      ██       ██",
    " ██████   ██████ ",
])

_BANNER_COLORS = ["#ff5f87", "#ff8700", "#ffd700", "#5fd7af", "#5fafff", "#af87ff", "#ff5fd7"]
_BANNER_CREDIT = [
    "",
    "",
    "  a streamlined Python AI agent",
    "",
    "  Created by Mark VandeWettering",
    "  <mvandewettering@gmail.com>  ·  2026",
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

    def __init__(self, session: SessionConfig, **kwargs):
        super().__init__(**kwargs)
        self.session = session

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
        logs.logger.addHandler(TuiLogHandler(self))
        for h in logs.logger.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, TuiLogHandler):
                logs.logger.removeHandler(h)

        self._write(_banner_markup())
        self._write("[dim]run_bash is UNSANDBOXED[/dim]")
        if len(self.session.history) > 1:
            self._write(f"[dim](resuming session, {len(self.session.history) - 1} messages)[/dim]")
        self._update_header()
        if not cli.provider_ready(self.session):
            from agent99x import commands
            result = commands.prompt_for_provider(self.session)
            self._write(f"[dim]{escape(result.message)}[/dim]")
        self.set_interval(1 / 12, self._tick)
        self.query_one(Input).focus()

    def handle_log_tool_call(self, name: str, args: dict) -> None:
        preview = ", ".join(f"{k}={core.clip(str(v), 40, '…')}" for k, v in args.items())
        self._write(f"[dim yellow]  ⚙ {escape(name)}({escape(preview)})[/dim yellow]")

    def handle_log_tool_result(self, name: str, result: str, elapsed: float) -> None:
        self._write(f"[dim]    → {escape(core.clip(result, 80, '…'))}  ({elapsed:.2f}s)[/dim]")

    def _write(self, payload) -> None:
        """Write a renderable to the log and remember it for redraws."""
        self.query_one(RichLog).write(payload)
        self._log_entries.append(payload)

    def _write_agent(self, reply: str, wid: Optional[int] = None) -> None:
        """Write an agent reply, respecting the current render mode."""
        log = self.query_one(RichLog)
        header = f"\n[bold green][{wid}][/bold green]" if wid is not None else "\n"

        if wid is not None and wid in self._thinking_entries:
            thinking_text = self._thinking_entries[wid]
            word_count = len(thinking_text.split())
            if wid in self._thinking_expanded:
                log.write(f"[dim]▼ Thinking ({word_count} words) — Ctrl+T to collapse[/dim]")
                log.write("\n".join(f"[dim]{line}[/dim]" for line in escape(thinking_text).split("\n")))
            else:
                log.write(f"[dim]▶ Thinking ({word_count} words) — Ctrl+T to expand[/dim]")

        if self._render_md:
            log.write(header)
            log.write(Markdown(reply or ""))
        else:
            log.write(f"{header}{_linkify(reply or '')}")
        self._log_entries.append(("agent", reply or "", wid))

    def _redraw_log(self) -> None:
        log = self.query_one(RichLog)
        log.clear()
        entries, self._log_entries = self._log_entries, []
        for entry in entries:
            if isinstance(entry, tuple) and entry and entry[0] == "agent":
                self._write_agent(entry[1], entry[2])
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
        self.query_one("#header", Static).update(
            f"{self.session.provider.name if self.session.provider else '(no provider)'}  ·  {self.session.model}{ctx}{effort}{thinking}"
        )

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
                with self._agent_workers_lock:
                    workers = list(self._agent_workers.values())
                if workers:
                    for worker_state in workers:
                        worker_state.cancel.set()
                        worker_state.state = "cancelling…"
        elif event.key == "ctrl+y":
            with self._agent_workers_lock:
                replies = [ws.last_reply for ws in self._agent_workers.values() if ws.last_reply]
            reply = replies[-1] if replies else getattr(self, "_last_completed_reply", "")
            if reply:
                self.copy_to_clipboard(reply)
                self._notice_msg = "Copied to clipboard"
                self._notice_until = time.monotonic() + 2.0

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
            result = cli.handle_slash_command(self.session, text)
            if result.handled:
                if result.message:
                    self._write(f"[dim]{escape(result.message)}[/dim]")
                if result.clear_history:
                    conversation.clear(self.session)
                    self._last_usage = 0
                if result.clear_display:
                    self.query_one(RichLog).clear()
                    self._log_entries = []
                if result.compact:
                    if self._any_busy():
                        self._write("[dim]Cannot compact while an agent is running.[/dim]")
                    elif len(self.session.history) > 2:
                        worker_id = self._next_worker_id()
                        worker_state = WorkerState(worker_id=worker_id, cancel=threading.Event(), state="compacting…")
                        with self._agent_workers_lock:
                            self._agent_workers[worker_id] = worker_state
                        self.run_worker(lambda: self._compact_worker(worker_id),
                                        thread=True, group=f"agent-{worker_id}", exclusive=False)
                    else:
                        self._write("[dim]Nothing to compact.[/dim]")
                if result.exit_app:
                    self.exit()
                return

            if self._any_busy():
                self._write("[dim yellow]Busy — press Esc to cancel.[/dim yellow]")
                return

            snapshot = len(self.session.history)
            media_paths, media_errors = conversation.add_user(self.session, text)
            clean_text = re.sub(r'attach:\S+', '', text).strip()
            media_tags = "  ".join(
                f"[dim cyan]{escape('[attachment: ' + os.path.basename(p) + ']')}[/dim cyan]"
                for p in media_paths
            )
            display = escape(clean_text) + (f"  {media_tags}" if media_tags else "")
            self._write(f"\n[bold cyan]You:[/bold cyan] {display}")
            for err in media_errors:
                self._write(f"[red]{escape(err)}[/red]")

            worker_id = self._next_worker_id()
            worker_state = WorkerState(worker_id=worker_id, cancel=threading.Event())
            with self._agent_workers_lock:
                self._agent_workers[worker_id] = worker_state
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
    ) -> None:
        """Clean up after agent_loop completes (runs on UI thread via call_from_thread)."""
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
            wid_for_log = worker_id if prefix else None
            self._log_entries.append(("agent", reply or "", wid_for_log))
            captured = "".join(ctx.thinking_capture)
            should_redraw = self._render_md
            if captured and self.session.show_thinking == "full" and wid_for_log is not None:
                self._thinking_entries[wid_for_log] = captured
                should_redraw = True
            if should_redraw:
                self._redraw_log()

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
            self._write(f"[dim]  auto-compacted: {old} → {new} messages[/dim]")
        # Context is persisted on exit (on_unmount), not per turn.
        self.query_one(Input).focus()

    def _agent_worker(self, worker_id: int, user_snapshot: Optional[int] = None) -> None:
        with self._agent_workers_lock:
            worker_state = self._agent_workers[worker_id]

        ctx = _StreamCtx(renderer=ThinkingRenderer(self.session.show_thinking))

        def on_tool_call(name: str, args: dict) -> None:
            preview = ", ".join(f"{k}={core.clip(str(v), 40, '…')}" for k, v in args.items())
            worker_state.state = f"tool: {name}({core.clip(preview, 55, '…')})"

        def on_usage(tokens: int) -> None:
            self.call_from_thread(setattr, self, "_last_usage", tokens)

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
                on_model_request=lambda: setattr(worker_state, "state", "waiting for model…"),
                on_tool_call=on_tool_call,
                on_usage=on_usage,
                on_token=lambda ctype, text: self._on_token(ctx, ctype, text),
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

        self.call_from_thread(
            self._finish_agent_run,
            worker_id, reply, agent_error, user_snapshot, ctx, autocompact_result,
        )

    def _compact_worker(self, worker_id: int) -> None:
        try:
            old, new, prompt_tokens, summary_tokens = compaction.do_compact(self.session)

            def finish() -> None:
                with self._agent_workers_lock:
                    self._agent_workers.pop(worker_id, None)
                self._last_usage = summary_tokens
                self._write(f"[dim]Compacted: {old} → {new} messages  (~{prompt_tokens:,} → ~{summary_tokens:,} tokens)[/dim]")
                self.query_one(Input).focus()

            self.call_from_thread(finish)
        except Exception as e:
            err = str(e)

            def finish_err() -> None:
                with self._agent_workers_lock:
                    self._agent_workers.pop(worker_id, None)
                self._write(f"[red]Compact failed: {escape(err)}[/red]")
                self.query_one(Input).focus()

            self.call_from_thread(finish_err)

    def on_unmount(self) -> None:
        cli.save_session(self.session)
