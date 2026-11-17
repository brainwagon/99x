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

import codecs
import datetime
import os
import pathlib
import re
import select
import shutil
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

# Plan mode lets run_bash through only when is_readonly_bash() can show the
# command is read-only.  The check is an allowlist: the command is lexed the
# way bash would split it, every pipeline segment must start with a known
# read-only program (after optional harmless VAR=val prefixes), dangerous
# options of those programs are refused, and the only redirections allowed are
# reads, fd duplications, and writes to /dev/null.  Anything the lexer cannot
# account for (command/process substitution, heredocs, subshells, unbalanced
# quotes, ...) makes the command "not read-only".


class _Word:
    """One shell word after quote removal.

    ``expands`` is set when bash would expand part of the word ($VAR, $'..'),
    so its runtime value is not the literal ``value``.  ``unquoted`` mirrors
    ``value`` with quoted/expanded characters replaced by NUL, for checks that
    only apply to unquoted text (assignments, brace expansion).
    """
    __slots__ = ("value", "expands", "unquoted")

    def __init__(self, value: str, expands: bool, unquoted: str):
        self.value = value
        self.expands = expands
        self.unquoted = unquoted


_SAFE_REDIRECT_TARGETS = {"/dev/null", "/dev/stdout", "/dev/stderr"}
_SIMPLE_PARAM_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
_BRACE_EXPANSION_RE = re.compile(r"\{[^{}]*(?:,|\.\.)[^{}]*\}")
_ASSIGNMENT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\+?=")
# Variables a read-only command may set inline (VAR=val cmd). Anything else
# (PATH, LD_PRELOAD, GIT_*, PAGER, BASH_ENV, ...) can change what runs.
_SAFE_ENV_RE = re.compile(
    r"LC_[A-Z]+|LANG|LANGUAGE|TZ|COLUMNS|LINES|NO_COLOR|CLICOLOR(?:_FORCE)?|FORCE_COLOR|TERM")


def _bash_segments(command: str) -> Optional[List[List[_Word]]]:
    """Split ``command`` into simple-command segments of words.

    Segments are separated by | || |& && ; & and newlines.  Returns None when
    the command uses a construct the read-only check refuses to reason about,
    or has a redirection that could write somewhere other than /dev/null.
    """
    segs: List[List[_Word]] = [[]]
    buf: List[str] = []
    unq: List[str] = []
    state = {"started": False, "expands": False, "redirect": None}
    n = len(command)

    def add(text: str, quoted: bool) -> None:
        buf.append(text)
        unq.append("\0" * len(text) if quoted else text)
        state["started"] = True

    def end_word() -> bool:
        if not state["started"]:
            return True
        word = _Word("".join(buf), state["expands"], "".join(unq))
        buf.clear()
        unq.clear()
        state["started"] = False
        state["expands"] = False
        kind = state["redirect"]
        if kind is None:
            if _BRACE_EXPANSION_RE.search(word.unquoted):
                return False
            segs[-1].append(word)
            return True
        state["redirect"] = None
        if kind == "in":
            return True
        if word.expands:
            return False
        if kind == "dup" and re.fullmatch(r"\d*-?", word.value):
            return True
        return word.value in _SAFE_REDIRECT_TARGETS

    def end_segment() -> bool:
        if not end_word() or state["redirect"] is not None:
            return False
        segs.append([])
        return True

    def start_redirect(kind: str) -> bool:
        # A word of bare digits directly before the operator is its fd number.
        if state["started"] and "".join(unq).isdigit():
            buf.clear()
            unq.clear()
            state["started"] = False
        elif not end_word():
            return False
        if state["redirect"] is not None:
            return False
        state["redirect"] = kind
        return True

    def dollar(i: int, quoted: bool) -> Optional[int]:
        """Handle a '$' at command[i]; return the index after it, or None."""
        nxt = command[i + 1] if i + 1 < n else ""
        if nxt in ("(", "["):
            return None  # $(...) $((...)) $[...]
        if nxt == "{":
            m = _SIMPLE_PARAM_RE.match(command, i + 1)
            if not m:
                return None  # ${x@P}, ${a[$(cmd)]}, ... can run code
            add(command[i:m.end()], True)
            state["expands"] = True
            return m.end()
        if nxt and (nxt.isalnum() or nxt in "_@*#?-$!"):
            state["expands"] = True
        add("$", quoted)
        return i + 1

    i = 0
    while i < n:
        c = command[i]
        nxt = command[i + 1] if i + 1 < n else ""
        if c == "'":
            j = command.find("'", i + 1)
            if j < 0:
                return None
            add(command[i + 1:j], True)
            i = j + 1
        elif c == '"' or (c == "$" and nxt == '"'):
            j = i + (2 if c == "$" else 1)
            state["started"] = True
            while True:
                if j >= n:
                    return None
                d = command[j]
                if d == '"':
                    break
                if d == "`":
                    return None
                if d == "\\" and j + 1 < n:
                    e = command[j + 1]
                    if e != "\n":
                        add(e if e in '"\\$`' else "\\" + e, True)
                    j += 2
                elif d == "$":
                    j = dollar(j, True)
                    if j is None:
                        return None
                else:
                    add(d, True)
                    j += 1
            i = j + 1
        elif c == "$" and nxt == "'":
            # ANSI-C quoting: the escapes are not decoded, so mark it expanded.
            j = i + 2
            while j < n and command[j] != "'":
                j += 2 if command[j] == "\\" else 1
            if j >= n:
                return None
            add(command[i + 2:j], True)
            state["expands"] = True
            i = j + 1
        elif c == "$":
            i = dollar(i, False)
            if i is None:
                return None
        elif c == "`" or c in "()":
            return None
        elif c == "\\":
            if nxt == "\n":
                i += 2
                continue
            if nxt:
                add(nxt, True)
                i += 2
            else:
                add("\\", False)
                i += 1
        elif c in " \t":
            if not end_word():
                return None
            i += 1
        elif c == "#" and not state["started"]:
            j = command.find("\n", i)
            i = n if j < 0 else j
        elif c in "\n;":
            if not end_segment():
                return None
            i += 1
        elif c == "&":
            if nxt == ">":
                if not start_redirect("out"):
                    return None
                i += 3 if command.startswith("&>>", i) else 2
            else:
                if not end_segment():
                    return None
                i += 2 if nxt == "&" else 1
        elif c == "|":
            if not end_segment():
                return None
            i += 2 if nxt in "|&" and nxt else 1
        elif c == "<":
            if command.startswith("<<<", i):
                kind, width = "in", 3
            elif nxt in ("<", ">", "("):
                return None  # heredoc, read-write open, process substitution
            elif nxt == "&":
                kind, width = "dup", 2
            else:
                kind, width = "in", 1
            if not start_redirect(kind):
                return None
            i += width
        elif c == ">":
            if nxt == "(":
                return None
            if nxt == "&":
                kind, width = "dup", 2
            elif nxt in (">", "|"):
                kind, width = "out", 2
            else:
                kind, width = "out", 1
            if not start_redirect(kind):
                return None
            i += width
        else:
            add(c, False)
            i += 1
    if not end_segment():
        return None
    return [s for s in segs if s]


def _opt_hit(args: List[str], short: str, long_: Tuple[str, ...],
             short_with_arg: str = "") -> bool:
    """True if any option in ``args`` is one of the given short chars or long
    names. Long names match by prefix too, since GNU getopt accepts
    unambiguous abbreviations (``--out=f`` is ``--output=f``)."""
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a == "--":
            break
        if a.startswith("--"):
            name = a[2:].split("=", 1)[0]
            if name and any(lg.startswith(name) for lg in long_):
                return True
        elif a.startswith("-") and len(a) > 1:
            for j, ch in enumerate(a[1:], 1):
                if ch in short:
                    return True
                if ch in short_with_arg:
                    skip = j == len(a) - 1
                    break
    return False


def _positional(args: List[str], short_with_arg: str = "",
                exact_with_arg: Tuple[str, ...] = ()) -> List[str]:
    """Non-option arguments, skipping the values of options that take one."""
    out: List[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            out.extend(args[i + 1:])
            break
        if a in exact_with_arg:
            i += 2
            continue
        if a.startswith("-") and len(a) > 1:
            if not a.startswith("--"):
                for j, ch in enumerate(a[1:], 1):
                    if ch in short_with_arg:
                        if j == len(a) - 1:
                            i += 1
                        break
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _sed_script_ok(s: str) -> bool:
    """Accept a sed script only if it uses no command that writes files or
    runs programs (w, W, e, and the s///w and s///e flags)."""
    n = len(s)
    i = 0

    def delimited(i: int, delim: str) -> Optional[int]:
        while i < n:
            if s[i] == "\\":
                i += 2
            elif s[i] == delim:
                return i + 1
            elif s[i] == "\n":
                return None
            else:
                i += 1
        return None

    def address(i: int) -> Optional[int]:
        if i < n and s[i] in "+~" and i + 1 < n and s[i + 1].isdigit():
            i += 1
        if i < n and s[i].isdigit():
            while i < n and (s[i].isdigit() or s[i] == "~"):
                i += 1
        elif i < n and s[i] == "$":
            i += 1
        elif i < n and s[i] in "/\\":
            if s[i] == "\\":
                if i + 1 >= n:
                    return None
                i += 1
            r = delimited(i + 1, s[i])
            if r is None:
                return None
            i = r
            while i < n and s[i] in "IM":
                i += 1
        return i

    while True:
        while i < n and s[i] in " \t\n;":
            i += 1
        if i >= n:
            return True
        r = address(i)
        if r is None:
            return False
        if r != i and r < n and s[r] == ",":
            r = address(r + 1)
            if r is None:
                return False
        i = r
        while i < n and s[i] in " \t!":
            i += 1
        if i >= n:
            return False
        c = s[i]
        i += 1
        if c in "{}":
            continue
        if c in "=dDgGhHlnNpPxzFqQ":
            while i < n and (s[i] in " \t" or s[i].isdigit()):
                i += 1
        elif c in "#btT:aicrR":
            # label / text / filename-to-read runs to end of line (or ';' for
            # branch labels, which is GNU's rule)
            stop = "\n" if c in "#:aicrR" else "\n;"
            while i < n and s[i] not in stop:
                i += 2 if s[i] == "\\" else 1
        elif c in "sy":
            if i >= n or s[i] in "\n\\":
                return False
            delim = s[i]
            for _ in range(2):
                r = delimited(i + 1, delim)
                if r is None:
                    return False
                i = r - 1
            i += 1
            if c == "s":
                while i < n and s[i] not in " \t\n;}":
                    if s[i] not in "gpiImM0123456789":
                        return False  # e (execute), w (write), or unknown
                    i += 1
        else:
            return False  # w W e and anything unrecognised
    # unreachable


def _sed_ok(args: List[str]) -> bool:
    scripts: List[str] = []
    positional: List[str] = []
    i = 0
    seen_dd = False
    flags_long = {"quiet", "silent", "regexp-extended", "separate", "unbuffered",
                  "null-data", "posix", "debug", "sandbox", "binary"}
    while i < len(args):
        a = args[i]
        if seen_dd or a == "-" or not a.startswith("-"):
            positional.append(a)
        elif a == "--":
            seen_dd = True
        elif a.startswith("--"):
            name, eq, val = a[2:].partition("=")
            if name == "expression":
                if not eq:
                    i += 1
                    if i >= len(args):
                        return False
                    val = args[i]
                scripts.append(val)
            elif name == "line-length":
                i += 0 if eq else 1
            elif name not in flags_long:
                return False  # --in-place, --file, abbreviations, unknown
        else:
            j = 1
            while j < len(a):
                ch = a[j]
                if ch in "nErsuzb":
                    j += 1
                    continue
                if ch in "el":
                    rest = a[j + 1:]
                    if not rest:
                        i += 1
                        if i >= len(args):
                            return False
                        rest = args[i]
                    if ch == "e":
                        scripts.append(rest)
                    break
                return False  # -i (in place), -f (script file), unknown
        i += 1
    if not scripts:
        if not positional:
            return False
        scripts.append(positional[0])
    return all(_sed_script_ok(s) for s in scripts)


def _awk_ok(args: List[str]) -> bool:
    prog = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("-F", "-v"):
            i += 2
            continue
        if a.startswith(("-F", "-v")):
            i += 1
            continue
        if a == "--" and i + 1 < len(args):
            prog = args[i + 1]
            break
        if a.startswith("-") and a != "-":
            return False  # -f progfile, -i/-l (gawk includes), -W exec, ...
        prog = a
        break
    if prog is None:
        return False
    # system(), pipes to/from commands, output redirection, getline, @load.
    return re.search(r"system|getline|[|>@]", prog) is None


def _git_branch_ok(args: List[str]) -> bool:
    with_arg = ("--contains", "--no-contains", "--merged", "--no-merged",
                "--points-at", "--sort", "--format", "--abbrev")
    flags = {"--all", "--remotes", "--verbose", "--list", "--show-current",
             "--no-color", "--color", "--column", "--no-column", "--ignore-case",
             "--omit-empty", "--no-abbrev"}
    listing = False
    positional = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in with_arg:
            i += 2
            continue
        if a.startswith("--"):
            if a.split("=", 1)[0] not in flags | set(with_arg):
                return False
            listing = listing or a == "--list"
        elif a.startswith("-") and len(a) > 1:
            if not set(a[1:]) <= set("arvli"):
                return False
            listing = listing or "l" in a
        else:
            positional.append(a)
        i += 1
    return not positional or listing


def _git_tag_ok(args: List[str]) -> bool:
    with_arg = ("--contains", "--no-contains", "--points-at", "--merged",
                "--no-merged", "--sort", "--format")
    flags = {"--list", "--ignore-case", "--column", "--no-column", "--color",
             "--no-color", "--omit-empty"}
    listing = False
    positional = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in with_arg:
            i += 2
            continue
        if a.startswith("--"):
            if a.split("=", 1)[0] not in flags | set(with_arg):
                return False
            listing = listing or a == "--list"
        elif a == "-l" or a == "-i" or re.fullmatch(r"-n\d*", a):
            listing = listing or a == "-l"
        elif a.startswith("-"):
            return False
        else:
            positional.append(a)
        i += 1
    return not positional or listing


def _git_config_ok(args: List[str]) -> bool:
    read_modes = {"--get", "--get-all", "--get-regexp", "--get-urlmatch",
                  "--list", "-l", "--get-color", "--get-colorbool"}
    flags = {"--global", "--system", "--local", "--worktree", "--show-origin",
             "--show-scope", "--name-only", "--null", "-z", "--includes",
             "--no-includes", "--bool", "--int", "--path", "--bool-or-int",
             "--expiry-date", "--all", "--regexp"}
    with_arg = ("--file", "-f", "--blob", "--type", "--default")
    reading = False
    positional = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in with_arg:
            i += 2
            continue
        if a.startswith("-"):
            base = a.split("=", 1)[0]
            if a in read_modes:
                reading = True
            elif a not in flags and not (base in with_arg and "=" in a):
                return False
        else:
            positional.append(a)
        i += 1
    if positional and positional[0] in ("get", "list"):
        reading = True  # git >= 2.46 subcommand syntax
    return reading


_GIT_READ_SUBCMDS = frozenset("""
    status log diff show blame ls-files ls-tree rev-parse describe grep shortlog
    cat-file rev-list name-rev merge-base show-ref for-each-ref check-ignore
    diff-tree diff-index diff-files count-objects ls-remote version whatchanged
""".split())


def _git_ok(args: List[str]) -> bool:
    i = 0
    while i < len(args) and args[i].startswith("-"):
        g = args[i]
        if g == "-C":
            i += 2
        elif g in ("--no-pager", "-P", "--no-optional-locks", "--literal-pathspecs",
                   "--no-replace-objects"):
            i += 1
        else:
            return False  # -c k=v, --git-dir, --exec-path, ... can run code
    if i >= len(args):
        return True  # bare `git` prints usage
    sub, rest = args[i], args[i + 1:]
    # --output writes a file; external diff / pager options run programs.
    if _opt_hit(rest, "", ("output", "ext-diff", "open-files-in-pager")):
        return False
    if sub in _GIT_READ_SUBCMDS:
        if sub == "grep" and _opt_hit(rest, "O", (), short_with_arg="efABCm"):
            return False
        return True
    if sub == "branch":
        return _git_branch_ok(rest)
    if sub == "tag":
        return _git_tag_ok(rest)
    if sub == "config":
        return _git_config_ok(rest)
    if sub == "remote":
        rest = [a for a in rest if a not in ("-v", "--verbose")]
        return not rest or rest[0] in ("show", "get-url")
    if sub == "reflog":
        pos = _positional(rest)
        return (not pos or pos[0] in ("show", "exists", "list")
                or re.search(r"[A-Z@^~/{]", pos[0]) is not None)
    if sub == "stash":
        return bool(rest) and rest[0] in ("list", "show")
    if sub == "worktree":
        return bool(rest) and rest[0] == "list"
    return False


# Programs with no file-writing or program-running options: any arguments OK.
_READONLY_SIMPLE = frozenset("""
    ls cat head tail wc grep egrep fgrep cut tr diff cmp stat du df pwd echo
    which type whoami uname basename dirname realpath readlink jq column nl od
    hexdump md5sum sha1sum sha224sum sha256sum sha384sum sha512sum b2sum cksum
    true false test [ cd printenv id
""".split())

# Programs that are read-only unless given particular options/arguments.
_READONLY_CHECKED: Dict[str, Callable[[List[str]], bool]] = {
    "find": lambda a: not set(a) & {"-delete", "-exec", "-execdir", "-ok", "-okdir",
                                    "-fprint", "-fprint0", "-fprintf", "-fls"},
    "sed": _sed_ok,
    "awk": _awk_ok, "gawk": _awk_ok, "mawk": _awk_ok,
    "sort": lambda a: not _opt_hit(a, "o", ("output", "compress-program"),
                                   short_with_arg="ktST"),
    "uniq": lambda a: len(_positional(a, "fsw", ("--skip-fields", "--skip-chars",
                                                  "--check-chars"))) <= 1,
    "xxd": lambda a: len(_positional(a, "", ("-c", "-cols", "-g", "-groupsize", "-l",
                                             "-len", "-o", "-offset", "-s", "-seek",
                                             "-n", "-name"))) <= 1,
    "tree": lambda a: not _opt_hit(a, "oR", ("output",), short_with_arg="LPIH"),
    "file": lambda a: not _opt_hit(a, "C", ("compile",), short_with_arg="emfFP"),
    "date": lambda a: not _opt_hit(a, "s", ("set",), short_with_arg="dfrI"),
    "rg": lambda a: not _opt_hit(a, "", ("pre",)),
    "printf": lambda a: not (a and a[0].startswith("-v")),
    "env": lambda a: all(x in ("-0", "--null") for x in a),
    "git": _git_ok,
}


def _segment_readonly(words: List[_Word]) -> bool:
    i = 0
    while i < len(words):
        m = _ASSIGNMENT_RE.match(words[i].unquoted)
        if not m:
            break
        if not _SAFE_ENV_RE.fullmatch(m.group(1)):
            return False
        i += 1
    if i == len(words):
        return True  # assignments only
    name = words[i].value
    if "/" in name or words[i].expands:
        return False
    if name in _READONLY_SIMPLE:
        return True
    check = _READONLY_CHECKED.get(name)
    if check is None:
        return False
    args = words[i + 1:]
    # An expansion could smuggle in an option (e.g. ${IFS}-delete), so the
    # option checks only trust literal arguments.
    if any(w.expands for w in args):
        return False
    return check([w.value for w in args])


def is_readonly_bash(command: str) -> bool:
    """Return True only if ``command`` is recognisably read-only.

    Allowlist-based: every segment of every pipeline/list must run a known
    read-only program with no option that writes files or runs other
    programs, and output may only be redirected to /dev/null (or duplicated
    onto another fd). Unparseable or unrecognised commands return False.
    """
    segs = _bash_segments(command)
    if segs is None:
        return False
    return all(_segment_readonly(seg) for seg in segs)


MAX_BASH_OUTPUT = 1_000_000  # bytes per stream; tests monkeypatch this
BASH_TIMEOUT = 30.0  # seconds; tests monkeypatch this
# After the shell exits, keep draining its pipes this long: background jobs it
# started (``cmd &``) may hold them open indefinitely.
_BASH_EXIT_GRACE = 0.25
_BASH = shutil.which("bash")


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


def _discard_output(proc: subprocess.Popen, fds: set) -> None:
    try:
        while fds:
            ready, _, _ = select.select(list(fds), [], [])
            for fd in ready:
                if not os.read(fd, 65536):
                    fds.discard(fd)
    except (OSError, ValueError):
        pass
    finally:
        proc.stdout.close()
        proc.stderr.close()


@tool("run_bash", "Run a bash command (30s timeout).", params={"command": "string"},
      group="shell")
def run_bash(command: str) -> Dict[str, Any]:
    proc = subprocess.Popen(
        command, shell=True, executable=_BASH, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    max_output = MAX_BASH_OUTPUT
    t0 = time.monotonic()
    fd_to_kind = {proc.stdout.fileno(): "stdout", proc.stderr.fileno(): "stderr"}
    bufs: Dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
    truncated: Dict[str, bool] = {"stdout": False, "stderr": False}
    open_fds = set(fd_to_kind)
    drain_until: Optional[float] = None  # set once the shell has exited

    def kill_group() -> None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()

    try:
        while True:
            now = time.monotonic()
            if drain_until is None:
                cancel = getattr(_thread_local, "cancel_event", None)
                if cancel is not None and cancel.is_set():
                    kill_group()
                    return _bash_result(bufs, truncated, max_output=max_output,
                                        error="cancelled")
                if now - t0 > BASH_TIMEOUT:
                    kill_group()
                    return _bash_result(bufs, truncated, max_output=max_output,
                                        error="timeout")
                if proc.poll() is not None:
                    drain_until = now + _BASH_EXIT_GRACE
            if drain_until is not None and (not open_fds or now >= drain_until):
                break
            if not open_fds:
                # Pipes closed but the shell is still running (e.g. it closed
                # stdout/stderr itself): keep honouring timeout and cancel.
                try:
                    proc.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    pass
                continue
            wait = 0.25 if drain_until is None else max(0.0, min(0.25, drain_until - now))
            ready, _, _ = select.select(list(open_fds), [], [], wait)
            for fd in ready:
                chunk = os.read(fd, 65536)
                kind = fd_to_kind[fd]
                if not chunk:
                    open_fds.discard(fd)
                    continue
                buf = bufs[kind]
                remaining = max_output - len(buf)
                if remaining > 0:
                    buf.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated[kind] = True
    finally:
        if open_fds and drain_until is not None:
            # A background job still holds the pipes. Keep draining them so it
            # does not die of SIGPIPE on its next write; close when it is done.
            threading.Thread(target=_discard_output, args=(proc, set(open_fds)),
                             daemon=True, name="run_bash-drain").start()
        else:
            proc.stdout.close()
            proc.stderr.close()

    return _bash_result(bufs, truncated, max_output=max_output, returncode=proc.returncode)


# ── file read/write/edit ───────────────────────────────────────────

def _detect_eol(s: str) -> str:
    crlf = s.count("\r\n")
    lf = s.count("\n") - crlf
    return "\r\n" if crlf > lf else "\n"


def _normalize_eol(s: str, target: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s if target == "\n" else s.replace("\n", target)


# Split only at real line breaks (\n, \r\n, \r), the same ones read_file's
# universal-newline mode honours. str.splitlines() also splits at \x0b, \x0c,
# \x1c-\x1e, \x85 and \u2028/\u2029, which would shift line numbers.
_LINE_SPLIT_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+\Z")


def _split_lines(s: str) -> List[str]:
    """Split into lines, keeping each line's ending."""
    return _LINE_SPLIT_RE.findall(s)


def _line_body(line: str) -> str:
    return line.rstrip("\r\n")


def _line_eol(line: str) -> str:
    return line[len(_line_body(line)):]


def _rstrip_replace(content: str, old: str, new: str, replace_all: bool,
                    eol: str) -> Tuple[Optional[str], int]:
    """Match old against content line-by-line after rstripping each line.

    The matched lines are replaced by ``new`` verbatim. As with an exact
    match, if ``old`` does not end with a line break the last matched line's
    break is kept after ``new``.
    """
    content_lines = _split_lines(content)
    old_lines = [_line_body(ln) for ln in _split_lines(old)]
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
    old_has_eol = old.endswith(("\n", "\r"))
    out: List[str] = []
    target_set = set(targets)
    i = 0
    while i < len(content_lines):
        if i in target_set:
            last_eol = _line_eol(content_lines[i + n - 1])
            block = new
            if not old_has_eol:
                block += last_eol
            elif not last_eol and block.endswith(eol):
                # old's final break matched the end of a file lacking one
                block = block[:-len(eol)]
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
        if not old_string:
            # An empty old_string "matches" between every character. Only an
            # empty file has an unambiguous meaning: write new_string into it.
            if content:
                return {"error": ("old_string is empty; give the exact text to replace "
                                  "(use replace_lines or write_file to insert or overwrite)")}
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(new_string)
            return {"ok": True, "replacements": 1, "bytes": len(new_string)}
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
            first_line = _line_body(_split_lines(old_string)[0]) if old_string else ""
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
        lines = _split_lines(content)
        total = len(lines)
        if start < 1 or start > total + 1:
            return {"error": f"start={start} out of range (file has {total} lines)"}
        if end < start - 1 or end > total:
            return {"error": f"end={end} out of range for start={start} (file has {total} lines)"}
        eol = _detect_eol(content) if content else "\n"
        is_insertion = end < start
        appending = is_insertion and start == total + 1
        if appending and total and not _line_eol(lines[-1]):
            # The old last line had no line break; give it one so the new
            # text starts on its own line, and let the new last line inherit
            # the missing final newline.
            lines[-1] += eol
            need_trailing = False
        elif is_insertion:
            need_trailing = True
        else:
            need_trailing = bool(_line_eol(lines[end - 1]))
        if not new_content:
            new_block = ""
        else:
            new_raw = _normalize_eol(new_content, "\n").split("\n")
            if new_raw and new_raw[-1] == "":
                new_raw.pop()
            new_block = eol.join(new_raw)
            if need_trailing and not new_block.endswith(("\n", "\r")):
                new_block += eol
        before = "".join(lines[:start - 1])
        after = "".join(lines[start - 1:]) if is_insertion else "".join(lines[end:])
        new_full = before + new_block + after
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(new_full)
        replaced = 0 if is_insertion else (end - start + 1)
        inserted = len(_split_lines(new_block))
        return {"ok": True, "lines_replaced": replaced, "lines_inserted": inserted, "bytes": len(new_full)}
    except OSError as e:
        return {"error": str(e)}


_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _parse_hunks(diff: str) -> Union[str, List[Dict[str, Any]]]:
    """Parse the hunks of a single-file unified diff (error string on failure).

    Each hunk's body runs to the next @@ header (or a new file header, or the
    end of the diff), regardless of the counts in its header, so hunks with
    miscounted headers still apply in full; count mismatches are reported as
    warnings. A blank body line is taken as a blank context line (editors and
    models often strip the lone space), and trailing blank lines are dropped.
    """
    raw = diff.split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    raw = [ln[:-1] if ln.endswith("\r") else ln for ln in raw]
    hunks: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(raw):
        line = raw[idx]
        if not line.startswith("@@"):
            idx += 1
            continue
        m = _HUNK_HEADER_RE.match(line)
        if not m:
            return f"Invalid hunk header at line {idx + 1}: {line}"
        idx += 1
        body: List[Tuple[str, str]] = []
        while idx < len(raw):
            dl = raw[idx]
            if dl.startswith("@@") or dl.startswith("diff ") or (
                    dl.startswith("--- ") and idx + 1 < len(raw)
                    and raw[idx + 1].startswith("+++ ")):
                break
            if dl == "":
                body.append((" ", ""))
            elif dl[0] in " +-":
                body.append((dl[0], dl[1:]))
            elif dl[0] == "\\":
                body.append(("\\", ""))  # "\ No newline at end of file"
            else:
                break  # trailing junk / prose after the hunk
            idx += 1
        while body and body[-1] == (" ", ""):
            body.pop()
        old_len = sum(1 for op, _ in body if op in " -")
        new_len = sum(1 for op, _ in body if op in " +")
        hunks.append({
            "header": line,
            "orig_start": int(m.group(1)),
            "orig_len": None if m.group(2) is None else int(m.group(2)),
            "new_len": None if m.group(4) is None else int(m.group(4)),
            "old_count": old_len, "new_count": new_len, "body": body,
        })
    return hunks


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
        if not target_path.is_relative_to(cwd):
            return {"error": f"Security violation: path {path} is outside the current working directory"}
        if not target_path.exists():
            return {"error": f"File not found: {path}"}
        with open(target_path, encoding="utf-8", newline="") as f:
            content = f.read()
        eol = _detect_eol(content) if content else "\n"
        if not diff.strip():
            return {"error": "Empty diff"}
        hunks = _parse_hunks(diff)
        if isinstance(hunks, str):
            return {"error": hunks}
        if not hunks:
            return {"error": "No valid hunks found in diff"}
        new_lines = _split_lines(content)
        warnings: List[str] = []
        offset = 0
        for h in hunks:
            line = h["header"]
            for label, declared, counted in (("old", h["orig_len"], h["old_count"]),
                                             ("new", h["new_len"], h["new_count"])):
                if (1 if declared is None else declared) != counted:
                    warnings.append(f"hunk {line}: header {label} length "
                                    f"{1 if declared is None else declared}, body has {counted}")
            # "-N,0" means "insert after line N"; otherwise N is the first line.
            start = h["orig_start"] if h["orig_len"] == 0 else h["orig_start"] - 1
            current_pos = max(start, 0) + offset
            body = h["body"]
            for k, (op, text) in enumerate(body):
                if op == " " or op == "-":
                    if (current_pos >= len(new_lines)
                            or _line_body(new_lines[current_pos]) != text):
                        kind = "" if op == " " else " (deletion)"
                        return {"error": (f"Context mismatch{kind} at hunk {line}, "
                                          f"pos {current_pos + 1}. Expected: {text!r}")}
                    if op == " ":
                        current_pos += 1
                    else:
                        new_lines.pop(current_pos)
                        offset -= 1
                elif op == "+":
                    if current_pos > len(new_lines):
                        return {"error": (f"Hunk {line} inserts at line {current_pos + 1}, "
                                          f"past the end of the file ({len(new_lines)} lines)")}
                    prev = current_pos - 1
                    if prev >= 0 and not _line_eol(new_lines[prev]):
                        new_lines[prev] += eol  # was the last line, lacking a newline
                    no_eol = k + 1 < len(body) and body[k + 1][0] == "\\"
                    new_lines.insert(current_pos, text if no_eol else text + eol)
                    current_pos += 1
                    offset += 1
        new_content = "".join(new_lines)
        with open(target_path, "w", encoding="utf-8", newline="") as f:
            f.write(new_content)
        result: Dict[str, Any] = {"ok": True, "bytes": len(new_content)}
        if warnings:
            result["warnings"] = warnings
        return result
    except OSError as e:
        return {"error": str(e)}


# ── search: grep / glob ────────────────────────────────────────────

_IGNORE_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv",
                ".pytest_cache", ".ruff_cache"}


def _glob_regex(pattern: str) -> "re.Pattern[str]":
    """Compile a path glob: ``*`` and ``?`` stay within one path component,
    ``[...]`` is a character class, and ``**`` as a whole component matches
    zero or more directories (so ``**/*.py`` also matches ``a.py``)."""
    parts = [p for p in pattern.replace("\\", "/").split("/") if p not in ("", ".")]
    out = ""
    for k, part in enumerate(parts):
        last = k == len(parts) - 1
        if part == "**":
            out += ".*" if last else "(?:[^/]+/)*"
            continue
        i = 0
        while i < len(part):
            c = part[i]
            if c == "*":
                out += "[^/]*"
            elif c == "?":
                out += "[^/]"
            elif c == "[":
                j = part.find("]", i + 2 if part[i + 1:i + 2] in ("!", "^") else i + 1)
                if j < 0:
                    out += r"\["
                else:
                    body = part[i + 1:j]
                    if body[:1] in ("!", "^"):
                        body = "^" + body[1:]
                    out += "[" + body.replace("\\", r"\\") + "]"
                    i = j
            else:
                out += re.escape(c)
            i += 1
        if not last:
            out += "/"
    return re.compile(out + r"\Z", re.DOTALL)


def _walk_files(root: str):
    """Yield (path, path-relative-to-root) for files under root, pruning
    _IGNORE_DIRS and not following directory symlinks."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _IGNORE_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            yield full, os.path.relpath(full, root).replace(os.sep, "/")


def _grep_python(pattern: str, path: str, glob_pat: str) -> List[Dict[str, Any]]:
    """Fallback grep. Mirrors rg: an explicit file is always searched, and a
    glob without '/' matches the file name at any depth."""
    out: List[Dict[str, Any]] = []
    regex = re.compile(pattern)
    if os.path.isfile(path):
        candidates = [path]
    else:
        gre = _glob_regex(glob_pat)
        by_name = "/" not in glob_pat
        candidates = [full for full, rel in _walk_files(path)
                      if gre.match(os.path.basename(rel) if by_name else rel)]
    for p in candidates:
        try:
            with open(p, encoding="utf-8", errors="replace", newline="") as f:
                for n, line in enumerate(f, 1):
                    if regex.search(line):
                        out.append({"path": p, "line": n, "text": _line_body(line)[:300]})
                        if len(out) >= MAX_GREP_RESULTS:
                            return out
        except OSError:
            pass
    return out


def _parse_rg_output(stdout: str) -> List[Dict[str, Any]]:
    # rg --null prints "path\0line:text", so paths containing ':' parse fine.
    results: List[Dict[str, Any]] = []
    for line in stdout.split("\n"):
        fname, sep, rest = line.partition("\0")
        if not sep:
            continue
        lineno, sep, text = rest.partition(":")
        if not sep or not lineno.isdigit():
            continue
        results.append({"path": fname, "line": int(lineno), "text": text.rstrip("\r")[:300]})
        if len(results) >= MAX_GREP_RESULTS:
            break
    return results


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
    try:
        if shutil.which("rg"):
            cmd = ["rg", "--with-filename", "--line-number", "--no-heading", "--null",
                   "--color=never", "--max-count", str(MAX_GREP_RESULTS),
                   "--glob", glob, "-e", pattern, "--", path]
            r = subprocess.run(cmd, capture_output=True, timeout=10)
            results = _parse_rg_output(r.stdout.decode("utf-8", errors="replace"))
            out: Dict[str, Any] = {"results": results, "tool": "rg"}
            if r.returncode == 2:  # 1 just means "no matches"
                err = r.stderr.decode("utf-8", errors="replace").strip() or "rg failed"
                if not results:
                    return {"error": err}
                out["errors"] = err[:2000]
            return out
        if not os.path.exists(path):
            return {"error": f"{path}: No such file or directory"}
        return {"results": _grep_python(pattern, path, glob), "tool": "python"}
    except (OSError, re.error, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


@tool("glob",
      "Find files by glob pattern, relative to path (default '.'). '*' and '?' "
      "match within one directory level, '**' matches any number of levels: "
      "'*.py' lists only path's own .py files, '**/*.py' finds them at any depth. "
      "Returns up to 500 paths.",
      group="search",
      params={"pattern": "string", "path": "string"},
      required=["pattern"])
def glob_files(pattern: str, path: str = ".") -> Dict[str, Any]:
    try:
        if not os.path.isdir(path):
            return {"error": f"{path}: not a directory"}
        gre = _glob_regex(pattern)
        matches: List[str] = []
        for full, rel in _walk_files(path):
            if gre.match(rel):
                matches.append(os.path.normpath(full))
                if len(matches) >= MAX_GLOB_MATCHES:
                    break
        return {"matches": matches}
    except (OSError, re.error) as e:
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
# The body is returned inside the tool result, i.e. straight into the model's
# context, so the default cap is sized for that (~25k tokens), not for disk.
_HTTP_BODY_CAP = 100_000


@tool("http_request",
      "Make an HTTP request. Returns {status, headers, body, truncated}; body is capped at "
      "max_bytes (default 100KB) and the rest of the response is not downloaded.",
      group="net",
      params={
          "url": "string",
          "method": {"type": "string", "description": "GET, POST, PUT, PATCH, DELETE, HEAD (default GET)."},
          "headers": {"type": "object", "description": "Optional header dict."},
          "body": {"type": "string", "description": "Raw request body. Set Content-Type header for JSON/form."},
          "timeout": {"type": "number", "description": "Seconds (default 30)."},
          "max_bytes": {"type": "integer",
                        "description": "Max response body bytes to return (default 100000)."},
      },
      required=["url"])
def http_request(url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None,
                 body: Optional[str] = None, timeout: float = 30,
                 max_bytes: int = _HTTP_BODY_CAP) -> Dict[str, Any]:
    try:
        method = method.upper()
        if method not in _HTTP_METHODS:
            return {"error": f"unsupported method: {method}"}
        max_bytes = max(0, int(max_bytes))
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream(method, url, headers=headers or {}, content=body) as r:
                data = bytearray()
                truncated = False
                for chunk in r.iter_bytes():
                    room = max_bytes - len(data)
                    if len(chunk) > room:
                        data.extend(chunk[:room])
                        truncated = True
                        break
                    data.extend(chunk)
                try:
                    decoder = codecs.getincrementaldecoder(r.encoding or "utf-8")(errors="replace")
                except LookupError:
                    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                # final=False when cut short drops a partial multi-byte char.
                text = decoder.decode(bytes(data), final=not truncated)
                return {"status": r.status_code, "headers": dict(r.headers),
                        "body": text, "truncated": truncated}
    except httpx.HTTPError as e:
        return {"error": str(e)}
