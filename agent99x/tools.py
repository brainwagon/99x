"""Tool registry and the built-in tool primitives.

This single module owns:
  - the registry surface (``TOOLS``, ``TOOL_HANDLERS``, traits, ``@tool``,
    ``register``)
  - the primitive tools the model can always call: file r/w/edit, bash,
    grep/glob, todos, current_datetime, http_request.

Skill/agent discovery and ``spawn_agent`` live in ``skills.py`` (which imports
``tool`` from here). Importing ``tools`` fires registration of these primitives;
importing ``skills`` fires registration of the rest.
"""

import datetime
import os
import pathlib
import re
import select
import signal
import subprocess
import threading
import time
import zoneinfo
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx

from agent99x.config import MAX_GREP_RESULTS, MAX_GLOB_MATCHES
from agent99x import todos as todo_md

# ── registry ───────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = []
TOOL_HANDLERS: Dict[str, Callable[..., Any]] = {}
TOOL_GROUPS: Dict[str, str] = {}  # tool name -> family ("files", "shell", ..., "meta")

# Catalog rendering order and labels. "meta" is the sentinel for "hide from the
# eager catalog"; anything unmapped (e.g. MCP tools) falls into "other".
GROUP_ORDER = ["files", "shell", "search", "net", "plan", "info"]
GROUP_LABELS = {
    "files": "Files", "shell": "Shell", "search": "Search",
    "net": "Network", "plan": "Planning", "info": "Info", "other": "Other",
}


@dataclass(frozen=True)
class ToolTraits:
    """Dispatch-relevant facts about a tool, recorded at registration."""
    needs_session: bool = False
    plan_safe: bool = True
    mutates: bool = False


TOOL_TRAITS: Dict[str, ToolTraits] = {}
_DEFAULT_TRAITS = ToolTraits()


def needs_session(name: str) -> bool:
    return TOOL_TRAITS.get(name, _DEFAULT_TRAITS).needs_session


def is_mutating(name: str) -> bool:
    return TOOL_TRAITS.get(name, _DEFAULT_TRAITS).mutates


def is_plan_safe(name: str) -> bool:
    return TOOL_TRAITS.get(name, _DEFAULT_TRAITS).plan_safe


def tool(
    name: str,
    desc: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    required: Optional[List[str]] = None,
    doc: Optional[str] = None,
    group: str = "other",
    needs_session: bool = False,
    plan_safe: bool = True,
    mutates: bool = False,
) -> Callable:
    """Register a function as a tool callable by the model."""
    props: Dict[str, Any] = {}
    auto_required: List[str] = []
    for pname, spec in (params or {}).items():
        props[pname] = {"type": spec} if isinstance(spec, str) else spec
        auto_required.append(pname)
    req = list(required) if required is not None else auto_required
    description = desc if doc is None else f"{desc}\n\n{doc}"
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": req},
        },
    }
    traits = ToolTraits(needs_session=needs_session, plan_safe=plan_safe, mutates=mutates)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        register(schema, fn, traits=traits, group=group)
        return fn

    return decorator


def register(schema: Dict[str, Any], handler: Callable[..., Any], *,
             traits: Optional[ToolTraits] = None, group: str = "other") -> None:
    """Record a tool's schema, handler, and traits. The single registration path
    for both the @tool decorator and runtime registrars (e.g. MCP)."""
    name = schema["function"]["name"]
    existing = next((i for i, t in enumerate(TOOLS) if t["function"]["name"] == name), None)
    if existing is None:
        TOOLS.append(schema)
    else:
        TOOLS[existing] = schema
    TOOL_HANDLERS[name] = handler
    TOOL_TRAITS[name] = traits or _DEFAULT_TRAITS
    TOOL_GROUPS[name] = group


def tool_catalog(allowed: Optional[set] = None) -> str:
    """Terse, grouped catalog of callable tools (names only).

    Descriptions live in the JSON schemas already sent to the model, so this
    only anchors which tools exist and their families. Hides group="meta"
    (bridge tools, covered by the capabilities primer). If ``allowed`` is given,
    only those tool names are shown (used for tool-restricted subagents).
    """
    buckets: Dict[str, List[str]] = {}
    for schema in TOOLS:
        name = schema["function"]["name"]
        if allowed is not None and name not in allowed:
            continue
        grp = TOOL_GROUPS.get(name, "other")
        if grp == "meta":
            continue
        buckets.setdefault(grp, []).append(name)
    order = GROUP_ORDER + [g for g in buckets if g not in GROUP_ORDER]
    lines: List[str] = []
    for g in order:
        if g in buckets:
            label = GROUP_LABELS.get(g, g.capitalize())
            lines.append(f"- {label}: {', '.join(buckets[g])}")
    return "\n".join(lines)


def tool_listing() -> List[Dict[str, str]]:
    """Exhaustive list of every registered tool: name, description, group.

    Includes meta/bridge tools and runtime-added (MCP) tools — the full truth
    behind the curated eager catalog. Backs the ``list_tools`` tool.
    """
    out: List[Dict[str, str]] = []
    for schema in TOOLS:
        fn = schema["function"]
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "group": TOOL_GROUPS.get(fn["name"], "other"),
        })
    return out


# ── bash ───────────────────────────────────────────────────────────

_thread_local = threading.local()  # per-thread state: cancel_event, depth

_BASH_BLOCK_RE = re.compile(
    r"""
      [^|&<]>[>]?               # > or >> not preceded by |, &, <
    | ^\s*>                     # bare redirect at start of command
    | `                         # backticks (command substitution)
    | \$\(                      # $( command substitution
    | <\(                       # <( process substitution
    | >\(                       # >( process substitution
    | \b(rm|rmdir|mv|dd|mkfs|shred|truncate|chmod|chown|chattr|tee)\b
    | \b(kill|pkill|killall|sudo|su|eval|exec)\b
    | \b(curl|wget)\b
    | \bgit\s+(commit|push|pull|merge|rebase|reset|clean|fetch
              |tag|branch\s+-[dD]|checkout\s+-b|switch\s+-c
              |restore|stash\s+(pop|drop|clear))\b
    | \bgh\s+(pr|issue|release|repo)\s+(create|edit|close|merge|delete|comment|review)\b
    | \b(pip|pip3)\s+(install|uninstall|download)\b
    | \b(apt|apt-get|dnf|yum|brew)\s+(install|remove|purge|upgrade|update)\b
    | \bnpm\s+(install|uninstall|ci)\b
    | \bpython3?\s+-c\b
    | \bfind\b[^|]*\s-delete\b
    | \bxargs\b[^|]*\b(rm|mv|kill)\b
    """,
    re.VERBOSE | re.IGNORECASE,
)


def is_readonly_bash(command: str) -> bool:
    """Return True if the command contains no known mutating patterns."""
    return _BASH_BLOCK_RE.search(command) is None


MAX_BASH_OUTPUT = 1_000_000  # bytes per stream; tests monkeypatch this


def _bash_decode(buf: bytes, truncated: bool, max_output: int) -> str:
    text = buf.decode("utf-8", errors="replace")
    if truncated:
        text += f"\n[truncated at {max_output} bytes]"
    return text


def _bash_result(bufs: Dict[str, bytearray], truncated: Dict[str, bool], *,
                 max_output: int = 1_000_000, returncode: Optional[int] = None,
                 error: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "stdout": _bash_decode(bufs["stdout"], truncated["stdout"], max_output),
        "stderr": _bash_decode(bufs["stderr"], truncated["stderr"], max_output),
    }
    if error is not None:
        out["error"] = error
    if returncode is not None:
        out["returncode"] = returncode
    return out


@tool("run_bash", "Run a bash command (30s timeout).", params={"command": "string"},
      group="shell")
def run_bash(command: str) -> Dict[str, Any]:
    proc = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    t0 = time.monotonic()
    fd_to_kind = {proc.stdout.fileno(): "stdout", proc.stderr.fileno(): "stderr"}
    bufs: Dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    truncated: Dict[str, bool] = {"stdout": False, "stderr": False}
    open_fds = set(fd_to_kind)

    def kill_group() -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    while open_fds:
        ready, _, _ = select.select(list(open_fds), [], [], 0.25)
        cancel = getattr(_thread_local, "cancel_event", None)
        if cancel is not None and cancel.is_set():
            kill_group()
            return _bash_result(bufs, truncated, error="cancelled")
        if time.monotonic() - t0 > 30:
            kill_group()
            return _bash_result(bufs, truncated, error="timeout")
        for fd in ready:
            chunk = os.read(fd, 65536)
            kind = fd_to_kind[fd]
            if not chunk:
                open_fds.discard(fd)
                continue
            buf = bufs[kind]
            remaining = MAX_BASH_OUTPUT - len(buf)
            if remaining > 0:
                buf.extend(chunk[:remaining])
            if len(chunk) > remaining:
                truncated[kind] = True

    proc.wait()
    return _bash_result(bufs, truncated, max_output=MAX_BASH_OUTPUT, returncode=proc.returncode)


# ── file read/write/edit ───────────────────────────────────────────

def _detect_eol(s: str) -> str:
    crlf = s.count("\r\n")
    lf = s.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _normalize_eol(s: str, target: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s if target == "\n" else s.replace("\n", target)


def _rstrip_replace(content: str, old: str, new: str, replace_all: bool,
                    eol: str) -> Tuple[Optional[str], int]:
    """Match old against content line-by-line after rstripping each line."""
    content_lines = content.splitlines(keepends=True)
    old_lines = old.splitlines()
    if not old_lines:
        return None, 0
    n = len(old_lines)
    old_stripped = [ln.rstrip() for ln in old_lines]
    matches: List[int] = []
    i = 0
    limit = len(content_lines) - n
    while i <= limit:
        if all(content_lines[i + j].rstrip() == old_stripped[j] for j in range(n)):
            matches.append(i)
            i += n
        else:
            i += 1
    if not matches:
        return None, 0
    if len(matches) > 1 and not replace_all:
        return None, len(matches)
    targets = matches if replace_all else matches[:1]
    new_raw = new.splitlines()
    out: List[str] = []
    target_set = set(targets)
    i = 0
    while i < len(content_lines):
        if i in target_set:
            last_line = content_lines[i + n - 1]
            last_had_eol = last_line.endswith("\n") or last_line.endswith("\r")
            if not new_raw:
                block = ""
            else:
                block = eol.join(new_raw)
                if last_had_eol and not block.endswith(("\n", "\r")):
                    block += eol
            out.append(block)
            i += n
        else:
            out.append(content_lines[i])
            i += 1
    return "".join(out), len(targets)


@tool("read_file", "Read a file from disk.", group="files",
      params={
          "path": "string",
          "offset": {"type": "integer", "description": "Line number to start reading from (1-based)."},
          "limit": {"type": "integer", "description": "Max lines to read (default all)."},
          "with_line_numbers": {"type": "boolean",
                                "description": "Prefix each line with its 1-based line number and a tab."},
      },
      required=["path"])
def read_file(path: str, offset: Optional[int] = None, limit: Optional[int] = None,
              with_line_numbers: bool = False) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        total = len(lines)
        if offset is not None or limit is not None or with_line_numbers:
            start = max((offset or 1) - 1, 0)
            end = total if limit is None else start + limit
            end = min(end, total)
            slice_ = lines[start:end]
            if with_line_numbers:
                content = "".join(f"{start + i + 1}\t{ln}" for i, ln in enumerate(slice_))
            else:
                content = "".join(slice_)
            return {"content": content, "total_lines": total,
                    "start_line": start + 1 if slice_ else 0, "end_line": end}
        return {"content": "".join(lines)}
    except OSError as e:
        return {"error": str(e)}
    except UnicodeDecodeError as e:
        return {"error": f"encoding error: {e}"}


@tool("write_file",
      "Write (overwrite) a file on disk. Prefer edit_file for targeted changes.",
      group="files",
      params={"path": "string", "content": "string"},
      mutates=True, plan_safe=False)
def write_file(path: str, content: str) -> Dict[str, Any]:
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "bytes": len(content)}
    except OSError as e:
        return {"error": str(e)}


@tool("edit_file",
      "Edit a file by replacing exact-match text. old_string must match uniquely "
      "(or pass replace_all=true). Tolerates CRLF/LF differences and trailing-whitespace drift.",
      group="files",
      params={
          "path": "string",
          "old_string": {"type": "string",
                         "description": "Exact text to find. Must be unique unless replace_all=true."},
          "new_string": {"type": "string", "description": "Text to replace it with."},
          "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
      },
      required=["path", "old_string", "new_string"],
      mutates=True, plan_safe=False)
def edit_file(path: str, old_string: str, new_string: str,
              replace_all: bool = False) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            content = f.read()
        eol = _detect_eol(content) if content else "\n"
        if old_string.startswith("﻿"):
            old_string = old_string[1:]
        old_norm = _normalize_eol(old_string, eol)
        new_norm = _normalize_eol(new_string, eol)
        count = content.count(old_norm)
        if count == 0:
            fallback_content, fallback_count = _rstrip_replace(
                content, old_norm, new_norm, replace_all, eol)
            if fallback_content is not None:
                with open(path, "w", encoding="utf-8", newline="") as f:
                    f.write(fallback_content)
                return {"ok": True, "replacements": fallback_count,
                        "bytes": len(fallback_content),
                        "note": "matched after rstripping each line (whitespace-tolerant)"}
            if fallback_count > 1:
                return {"error": (f"old_string matches {fallback_count} times after rstripping "
                                  "per-line; pass replace_all=true or include more context")}
            stripped = old_string.strip()
            if stripped and stripped in content and stripped != old_string:
                return {"error": ("old_string not found exactly, but matched after stripping "
                                  "whitespace. Check leading/trailing whitespace and indentation.")}
            first_line = old_string.splitlines()[0] if old_string else ""
            if first_line and first_line in content:
                idx = content.index(first_line)
                snippet = content[max(0, idx - 50):idx + 200]
                return {"error": (f"old_string not found. First line matched but full block did not. "
                                  f"Nearby content: {snippet!r}")}
            return {"error": "old_string not found in file"}
        if count > 1 and not replace_all:
            return {"error": (f"old_string matches {count} times; "
                              "pass replace_all=true or include more context to make it unique")}
        new_content = (content.replace(old_norm, new_norm) if replace_all
                       else content.replace(old_norm, new_norm, 1))
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new_content)
        return {"ok": True, "replacements": count if replace_all else 1, "bytes": len(new_content)}
    except OSError as e:
        return {"error": str(e)}


@tool("replace_lines",
      "Replace a contiguous range of lines in a file by line number. Lines are 1-based, end is inclusive. "
      "More robust than edit_file for small models since there is no string to match. "
      "Use end=start-1 to insert before start without deleting anything.",
      group="files",
      params={
          "path": "string",
          "start": {"type": "integer", "description": "First line to replace (1-based, inclusive)."},
          "end": {"type": "integer",
                  "description": "Last line to replace (1-based, inclusive). Use start-1 for pure insertion."},
          "new_content": {"type": "string",
                          "description": "Replacement text. Trailing newline added automatically."},
      },
      required=["path", "start", "end", "new_content"],
      mutates=True, plan_safe=False)
def replace_lines(path: str, start: int, end: int, new_content: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8", newline="") as f:
            content = f.read()
        lines = content.splitlines(keepends=True)
        total = len(lines)
        if start < 1 or start > total + 1:
            return {"error": f"start={start} out of range (file has {total} lines)"}
        if end < start - 1 or end > total:
            return {"error": f"end={end} out of range for start={start} (file has {total} lines)"}
        eol = _detect_eol(content) if content else "\n"
        is_insertion = end < start
        if not new_content:
            new_block = ""
        else:
            new_raw = _normalize_eol(new_content, "\n").split("\n")
            if new_raw and new_raw[-1] == "":
                new_raw.pop()
            new_block = eol.join(new_raw)
            need_trailing = is_insertion or lines[end - 1].endswith(("\n", "\r"))
            if need_trailing and not new_block.endswith(("\n", "\r")):
                new_block += eol
        before = "".join(lines[:start - 1])
        after = "".join(lines[start - 1:]) if is_insertion else "".join(lines[end:])
        new_full = before + new_block + after
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new_full)
        replaced = 0 if is_insertion else (end - start + 1)
        inserted = len(new_block.splitlines()) if new_block else 0
        return {"ok": True, "lines_replaced": replaced, "lines_inserted": inserted, "bytes": len(new_full)}
    except OSError as e:
        return {"error": str(e)}


@tool("patch",
      "Apply a unified diff (patch) to a file. Strict context matching is used.",
      group="files",
      params={
          "path": "string",
          "diff": {"type": "string", "description": "The unified diff content to apply."},
      },
      mutates=True, plan_safe=False)
def apply_patch(path: str, diff: str) -> Dict[str, Any]:
    """Apply a unified diff to a file. Fails if any context/deletion mismatches."""
    try:
        target_path = pathlib.Path(path).resolve()
        cwd = pathlib.Path.cwd().resolve()
        if not str(target_path).startswith(str(cwd)):
            return {"error": f"Security violation: path {path} is outside the current working directory"}
        if not target_path.exists():
            return {"error": f"File not found: {path}"}
        with open(target_path, encoding="utf-8", newline="") as f:
            content = f.read()
        eol = _detect_eol(content) if content else "\n"
        lines = content.splitlines(keepends=True)
        diff_lines = diff.splitlines()
        if not diff_lines:
            return {"error": "Empty diff"}
        idx = 0
        while idx < len(diff_lines) and (
            diff_lines[idx].startswith("---") or diff_lines[idx].startswith("+++")):
            idx += 1
        new_lines = list(lines)
        offset = 0
        hunks_found = 0
        while idx < len(diff_lines):
            line = diff_lines[idx]
            if not line.startswith("@@"):
                idx += 1
                continue
            hunks_found += 1
            m = re.match(r"^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", line)
            if not m:
                return {"error": f"Invalid hunk header at line {idx + 1}: {line}"}
            orig_start = int(m.group(1))
            orig_len = int(m.group(2)) if m.group(2) else 1
            current_pos = orig_start - 1 + offset
            idx += 1
            hunk_idx = 0
            while idx < len(diff_lines) and hunk_idx < (orig_len + 100):
                if diff_lines[idx].startswith("@@"):
                    break
                dl = diff_lines[idx]
                if dl.startswith(" "):
                    expected = dl[1:]
                    if (current_pos >= len(new_lines)
                            or new_lines[current_pos].rstrip("\r\n") != expected.rstrip("\r\n")):
                        return {"error": (f"Context mismatch at hunk {line}, pos {current_pos + 1}. "
                                          f"Expected: {expected!r}")}
                    current_pos += 1
                elif dl.startswith("-"):
                    expected = dl[1:]
                    if (current_pos >= len(new_lines)
                            or new_lines[current_pos].rstrip("\r\n") != expected.rstrip("\r\n")):
                        return {"error": (f"Context mismatch (deletion) at hunk {line}, "
                                          f"pos {current_pos + 1}. Expected: {expected!r}")}
                    new_lines.pop(current_pos)
                    offset -= 1
                elif dl.startswith("+"):
                    to_add = dl[1:]
                    if not to_add.endswith(("\n", "\r")):
                        to_add += eol
                    new_lines.insert(current_pos, to_add)
                    current_pos += 1
                    offset += 1
                idx += 1
                hunk_idx += 1
        if hunks_found == 0:
            return {"error": "No valid hunks found in diff"}
        new_content = "".join(new_lines)
        with open(target_path, "w", encoding="utf-8", newline="") as f:
            f.write(new_content)
        return {"ok": True, "bytes": len(new_content)}
    except OSError as e:
        return {"error": str(e)}


# ── search: grep / glob ────────────────────────────────────────────

_IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
                ".pytest_cache", ".ruff_cache"}


def _grep_python(pattern: str, path: str, glob_pat: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    regex = re.compile(pattern)
    root = pathlib.Path(path)

    def _walk(p: pathlib.Path) -> None:
        if p.name in _IGNORE_DIRS:
            return
        if p.is_file():
            if not p.match(glob_pat):
                return
            try:
                with open(p, encoding="utf-8", errors="replace") as f:
                    for n, line in enumerate(f, 1):
                        if regex.search(line):
                            out.append({"path": str(p), "line": n, "text": line.rstrip("\n")[:300]})
                            if len(out) >= MAX_GREP_RESULTS:
                                return
            except OSError:
                pass
        elif p.is_dir():
            for child in p.iterdir():
                _walk(child)
                if len(out) >= MAX_GREP_RESULTS:
                    return

    _walk(root)
    return out


@tool("grep",
      "Search file contents for a regex. Returns up to 200 matches as {path,line,text}.",
      group="search",
      params={
          "pattern": "string",
          "path": {"type": "string", "description": "Directory or file (default '.')."},
          "glob": {"type": "string", "description": "Filename glob filter, e.g. '*.py' (default '*')."},
      },
      required=["pattern"])
def grep(pattern: str, path: str = ".", glob: str = "*") -> Dict[str, Any]:
    import shutil
    try:
        if shutil.which("rg"):
            cmd = ["rg", "--line-number", "--no-heading", "--max-count", str(MAX_GREP_RESULTS),
                   "--glob", glob, "-e", pattern, path]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            results: List[Dict[str, Any]] = []
            for line in r.stdout.splitlines()[:MAX_GREP_RESULTS]:
                parts = line.split(":", 2)
                if len(parts) == 3:
                    try:
                        results.append({"path": parts[0], "line": int(parts[1]), "text": parts[2][:300]})
                    except ValueError:
                        continue
            return {"results": results, "tool": "rg"}
        return {"results": _grep_python(pattern, path, glob), "tool": "python"}
    except (OSError, re.error, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


@tool("glob",
      "Find files by glob pattern (e.g. '**/*.py'). Returns up to 500 paths.",
      group="search",
      params={"pattern": "string", "path": "string"},
      required=["pattern"])
def glob_files(pattern: str, path: str = ".") -> Dict[str, Any]:
    try:
        root = pathlib.Path(path)
        matches: List[str] = []

        def _walk(p: pathlib.Path) -> None:
            if p.name in _IGNORE_DIRS:
                return
            if p.is_file():
                if p.match(pattern):
                    matches.append(str(p))
            elif p.is_dir():
                try:
                    for child in p.iterdir():
                        _walk(child)
                        if len(matches) >= MAX_GLOB_MATCHES:
                            return
                except OSError:
                    pass

        _walk(root)
        return {"matches": matches}
    except OSError as e:
        return {"error": str(e)}


# ── todos ──────────────────────────────────────────────────────────

@tool("write_todos",
      "Replace the current todo list. Use to plan and track multi-step work; rewrite the whole list each call.",
      group="plan",
      params={"todos": {
          "type": "array",
          "items": {
              "type": "object",
              "properties": {
                  "text": {"type": "string"},
                  "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
              },
              "required": ["text"],
          },
      }})
def write_todos(todos: List[Union[Dict[str, Any], str]]) -> Dict[str, Any]:
    saved = todo_md.save(todos)
    return {"ok": True, "count": len(saved), "todos": saved}


@tool("read_todos", "Read the current todo list.", group="plan")
def read_todos() -> Dict[str, Any]:
    return {"todos": todo_md.load()}


# ── datetime ───────────────────────────────────────────────────────

@tool("current_datetime",
      "Return the current date, time, and timezone. Pass an IANA timezone name "
      "(e.g. 'America/Los_Angeles', 'Europe/London', 'UTC') to get the time there; "
      "omit for local time. Use for date/time math instead of computing it yourself.",
      group="info",
      params={"timezone": {"type": "string",
                           "description": "Optional IANA timezone name, e.g. 'America/Los_Angeles'."}},
      required=[])
def current_datetime(timezone: Optional[str] = None) -> Dict[str, Any]:
    if timezone:
        try:
            tz = zoneinfo.ZoneInfo(timezone)
        except zoneinfo.ZoneInfoNotFoundError:
            return {"error": f"unknown timezone: {timezone!r} (use an IANA name like 'America/Los_Angeles')"}
        now = datetime.datetime.now(tz)
    else:
        now = datetime.datetime.now().astimezone()
    tzinfo = now.tzinfo
    return {
        "date":       now.strftime("%Y-%m-%d"),
        "time":       now.strftime("%H:%M:%S"),
        "datetime":   now.isoformat(timespec="seconds"),
        "timezone":   tzinfo.tzname(now) if tzinfo else "",
        "utc_offset": now.strftime("%z"),
        "weekday":    now.strftime("%A"),
        "epoch":      int(now.timestamp()),
    }


# ── http ───────────────────────────────────────────────────────────

_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}
_HTTP_BODY_CAP = 10_000_000


@tool("http_request",
      "Make an HTTP request. Returns {status, headers, body, truncated}; body is capped at max_bytes (default 10MB).",
      group="net",
      params={
          "url": "string",
          "method": {"type": "string", "description": "GET, POST, PUT, PATCH, DELETE, HEAD (default GET)."},
          "headers": {"type": "object", "description": "Optional header dict."},
          "body": {"type": "string", "description": "Raw request body. Set Content-Type header for JSON/form."},
          "timeout": {"type": "number", "description": "Seconds (default 30)."},
          "max_bytes": {"type": "integer", "description": "Max response body chars to return (default 10000000)."},
      },
      required=["url"])
def http_request(url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None,
                 body: Optional[str] = None, timeout: float = 30,
                 max_bytes: int = _HTTP_BODY_CAP) -> Dict[str, Any]:
    try:
        method = method.upper()
        if method not in _HTTP_METHODS:
            return {"error": f"unsupported method: {method}"}
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.request(method, url, headers=headers or {}, content=body)
            text = r.text
            truncated = len(text) > max_bytes
            if truncated:
                text = text[:max_bytes]
            return {"status": r.status_code, "headers": dict(r.headers),
                    "body": text, "truncated": truncated}
    except httpx.HTTPError as e:
        return {"error": str(e)}
