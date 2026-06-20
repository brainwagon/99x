"""Prompt assembly: AGENT.md parsing, system prompts (from AGENT.md, memory,
skills) and user messages (with attach:path media expansion)."""

import base64
import mimetypes
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from agent99x import logs
from agent99x import scopes
from agent99x import tools
from agent99x.session import SessionConfig, PROJECT_DIR

MEMORY_FILE = "MEMORY.md"
USER_MD_FILE = "USER.md"
AGENT_MD_FILE = "AGENT.md"
PROJECT_AGENT_MD = os.path.join(PROJECT_DIR, AGENT_MD_FILE)
PROJECT_MEMORY_MD = os.path.join(PROJECT_DIR, MEMORY_FILE)


# ── AGENT.md frontmatter parsing (was agent_md.py) ──────────────────

_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}


def _parse_scalar(raw: str) -> Any:
    s = raw.strip()
    if not s:
        return ""
    low = s.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item) for item in inner.split(",")]
    return s


def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Split YAML-ish frontmatter from body. Returns ({}, text) when none.

    Handles flat ``key: value`` pairs and one level of nested map (an empty
    value followed by indented ``key: value`` lines), so agentskills.io's
    ``metadata:`` block parses into a dict. Deeper nesting is not supported.
    """
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return {}, text
    first_nl = text.find("\n") + 1
    end = text.find("\n---", first_nl)
    if end == -1:
        return {}, text
    block = text[first_nl:end]
    after = end + len("\n---")
    if after < len(text) and text[after] == "\r":
        after += 1
    if after < len(text) and text[after] == "\n":
        after += 1
    meta: Dict[str, Any] = {}
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        val = value.strip()
        if val in (">", "|"):
            # YAML block scalar: gather the indented lines below it. Folded
            # (">") joins them with spaces; literal ("|") keeps newlines.
            body_lines: List[str] = []
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() and len(nxt) - len(nxt.lstrip()) == 0:
                    break  # back to top level
                body_lines.append(nxt.strip())
                i += 1
            joined = " " if val == ">" else "\n"
            meta[key] = joined.join(body_lines).strip()
        elif val == "":
            # Possibly a nested map: gather following indented "k: v" lines.
            submap: Dict[str, Any] = {}
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    i += 1
                    continue
                if len(nxt) - len(nxt.lstrip()) == 0:
                    break  # back to top level
                sub = nxt.strip()
                if ":" in sub:
                    sk, _, sv = sub.partition(":")
                    submap[sk.strip()] = _parse_scalar(sv)
                i += 1
            meta[key] = submap if submap else ""
        else:
            meta[key] = _parse_scalar(val)
    return meta, text[after:].lstrip("\n")


# ── system-prompt suffixes ─────────────────────────────────────────

_SYSTEM_SUFFIX = (
    "Use read_file, write_file, edit_file, grep, glob, and run_bash to complete the user's task. "
    "Prefer edit_file over write_file for targeted changes; write_file is for new files or full rewrites. "
    "Prefer grep/glob over shelling out to grep/find via run_bash. "
    "For tasks with more than ~3 steps, call write_todos first to plan, then update statuses as you go. "
    "Never describe a tool call you are about to make — just make it. "
    "If more work remains, your reply MUST contain a tool call; "
    "only return plain text with no tool call when the task is fully complete. "
    "If your final text reply presents the user with choices and you have a recommendation, "
    "format the recommended choice exactly like this: [RECOMMENDED: <choice>]. "
    "The UI will parse this and automatically preload the input buffer for the user."
)

# Prescriptive primer that teaches the three-kind capability model. Prepended to
# the catalogs in the "# YOUR CAPABILITIES" section. Keep it short — it's
# always-on prompt.
_CAPABILITIES_PRIMER = (
    "You have three kinds of capability. Pick the right kind:\n"
    "- **Tools** — code you run yourself, right now (read_file, run_bash, grep, …). "
    "Call them directly. `list_tools` shows them all.\n"
    "- **Skills** — written instructions for a procedure. When a task matches a skill below, "
    "call `load_skill(name)` to load its steps, then carry them out in this conversation. "
    "A skill may bundle scripts; `load_skill` returns its `dir`, so run any bundled script with "
    "`run_bash` from that dir.\n"
    "- **Agents** — fresh specialist workers. For a large, self-contained sub-task, "
    "call `spawn_agent(task, name)`; the agent works in its own context and returns a result.\n"
    "Decide by: do it yourself → tool; need the recipe → skill; hand it off → agent."
)
_PLAN_SUFFIX = (
    "You are in PLAN MODE. "
    "You may use read_file and run_bash with read-only commands "
    "(ls, cat, grep, find, git log, git status, git diff, etc.) to explore the codebase. "
    "You MUST NOT write files (write_file is disabled) or run mutating bash commands "
    "(no rm, mv, git commit, git push, pip install, output redirections, etc.). "
    "Describe what you would do rather than doing it."
)


def _read_with_mtime(path: str) -> Optional[Tuple[float, str]]:
    """Return (mtime, text) for path, or None on any error (logs non-ENOENT warnings)."""
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return None
    except OSError as e:
        logs.log_warning(f"could not stat {path}: {e}")
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return mtime, f.read()
    except FileNotFoundError:
        return None
    except OSError as e:
        logs.log_warning(f"could not read {path}: {e}")
        return None


def _load_memory(session: SessionConfig, path: str = MEMORY_FILE) -> str:
    cache = session.shared_state.setdefault("memory_cache", {})
    result = _read_with_mtime(path)
    if result is None:
        return ""
    mtime, text = result
    entry = cache.get(path)
    if entry is not None and entry[0] == mtime:
        return entry[1]
    cache[path] = (mtime, text)
    return text


def _load_agent_md(session: SessionConfig,
                   path: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """Read an AGENT.md file and return (frontmatter, body)."""
    cache = session.shared_state.setdefault("agent_md_cache", {})
    if path is None:
        path = session.agent_path(AGENT_MD_FILE)
    result = _read_with_mtime(path)
    if result is None:
        return {}, ""
    mtime, text = result
    entry = cache.get(path)
    if entry is not None and entry[0] == mtime:
        return entry[1], entry[2]
    meta, body = parse_frontmatter(text)
    cache[path] = (mtime, meta, body)
    return meta, body


def _skill_meta(path: str) -> Tuple[str, str]:
    """Return (description, body) for a SKILL.md file."""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        logs.log_warning(f"could not read {path}: {e}")
        return "", ""
    meta, body = parse_frontmatter(text)
    if meta.get("description"):
        return str(meta["description"]), body
    title = ""
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            if not title:
                title = s.lstrip("# ").strip()
            continue
        return s, body
    return title, body


def _skill_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _skills_catalog_lines(session: SessionConfig) -> List[str]:
    """Full (name — description) catalog lines for available skills, cached."""
    cache = session.shared_state.setdefault("skills_catalog_cache", {})
    entries = scopes.discover("skills")
    if not entries:
        return []
    sig = tuple((n, _skill_mtime(p)) for n, p in entries)
    cached = cache.get("merged")
    if cached and cached[0] == sig:
        return cached[1]
    lines: List[str] = []
    for name, path in entries:
        desc, _body = _skill_meta(path)
        if not desc and not _body:
            continue
        lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    cache["merged"] = (sig, lines)
    return lines


def _agents_catalog_lines() -> List[str]:
    """Full (name — description) catalog lines for available agents."""
    lines: List[str] = []
    for name, path in scopes.discover("agents"):
        try:
            with open(path, encoding="utf-8") as f:
                meta, _body = parse_frontmatter(f.read())
        except OSError:
            continue
        desc = str(meta.get("description") or "")
        lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    return lines


def _build_capabilities(session: SessionConfig,
                        allowed: Optional[set] = None) -> str:
    """The '# YOUR CAPABILITIES' section: primer + three catalogs.

    Tools render terse (grouped names); skills/agents render full. ``allowed``
    restricts the tools catalog for tool-restricted subagents.
    """
    parts: List[str] = ["# YOUR CAPABILITIES", _CAPABILITIES_PRIMER]
    tool_cat = tools.tool_catalog(allowed)
    if tool_cat:
        parts.append("## Tools\n" + tool_cat)
    skill_lines = _skills_catalog_lines(session)
    if skill_lines:
        parts.append("## Skills\n" + "\n".join(skill_lines))
    agent_lines = _agents_catalog_lines()
    if agent_lines:
        parts.append("## Agents\n" + "\n".join(agent_lines))
    return "\n\n".join(parts)


def _runtime_paths_section(session: SessionConfig) -> str:
    """Tell the agent where its own files live, so read/edit calls don't have to guess."""
    return (
        "# YOUR FILES\n\n"
        "Files are scoped to two levels:\n\n"
        "**Global** (fixed absolute paths, use verbatim):\n"
        f"- `{session.agent_path('AGENT.md')}` — your persona and base instructions\n"
        f"- `{session.agent_path('USER.md')}` — facts about the user\n"
        f"- `{session.agent_path('MEMORY.md')}` — facts about the user (write user-preference facts here)\n"
        f"- `{session.agent_path('diary')}` — daily notes (YYYY-MM-DD.md)\n\n"
        "**Project** (relative to CWD, created on demand):\n"
        f"- `{PROJECT_DIR}/AGENT.md` — project context loaded into your system prompt (user-written)\n"
        f"- `{PROJECT_DIR}/MEMORY.md` — facts about this project (write project-specific facts here)\n"
        f"- `{PROJECT_DIR}/todos.md` — task list\n\n"
        "**Memory writing rule:** write facts about the *user* "
        f"(preferences, style, habits) to `{session.agent_path('MEMORY.md')}`. "
        "Write facts about the *project* (conventions, decisions, architecture) "
        f"to `{PROJECT_DIR}/MEMORY.md`."
    )


# ── user-message construction ──────────────────────────────────────

ATTACH_PATH_RE = re.compile(r'attach:(\S+)')


def _encode_media(path: str) -> Dict[str, Any]:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    mime = mimetypes.guess_type(path)[0]
    if not mime:
        raise ValueError(f"Unknown file type for {path}")

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()

    if mime.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
    elif mime.startswith("audio/"):
        subtype = mime.split("/")[1]
        if subtype == "mpeg":
            subtype = "mp3"
        return {"type": "input_audio", "input_audio": {"data": data, "format": subtype}}
    else:
        raise ValueError(f"File is not an image or audio (detected mime: {mime})")


def build_user_message(text: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Parse attach:path media refs from text. Returns (message_dict, media_paths, [error_str, ...])."""
    paths = ATTACH_PATH_RE.findall(text)
    if not paths:
        return {"role": "user", "content": text}, [], []
    clean = ATTACH_PATH_RE.sub("", text).strip()
    content: List[Dict[str, Any]] = []
    errors: List[str] = []
    if clean:
        content.append({"type": "text", "text": clean})
    media_paths: List[str] = []
    for p in paths:
        try:
            media_obj = _encode_media(p)
            content.append(media_obj)
            media_paths.append(p)
        except (FileNotFoundError, ValueError, OSError) as e:
            errors.append(str(e))
    if not content:
        return {"role": "user", "content": text}, [], errors
    return {"role": "user", "content": content}, media_paths, errors


# ── system-prompt assembly ─────────────────────────────────────────

def build_system(session: SessionConfig) -> Dict[str, Any]:
    """Build the system message from AGENT.md, memory, and mode suffix."""
    _meta, body = _load_agent_md(session)
    _meta_proj, proj_body = _load_agent_md(session, PROJECT_AGENT_MD)
    memory = _load_memory(session)
    proj_memory = _load_memory(session, PROJECT_MEMORY_MD)
    user_md = _load_memory(session, USER_MD_FILE)
    suffix = _PLAN_SUFFIX if session.plan_mode else _SYSTEM_SUFFIX
    parts = []
    if body:
        parts.append(body)
    if proj_body:
        parts.append("# PROJECT\n\n" + proj_body.strip())
    parts.append(_build_capabilities(session))
    if user_md:
        parts.append("# USER\n\n" + user_md.strip())
    mem_sections = []
    if memory.strip():
        mem_sections.append("## User\n\n" + memory.strip())
    if proj_memory.strip():
        mem_sections.append("## Project\n\n" + proj_memory.strip())
    if mem_sections:
        parts.append("# MEMORY\n\n" + "\n\n".join(mem_sections))

    todos_path = os.path.join(PROJECT_DIR, "todos.md")
    todos = _load_memory(session, todos_path)
    if todos.strip():
        parts.append("# ACTIVE PLAN / TODOS\n\n" + todos.strip())

    parts.append(_runtime_paths_section(session))
    parts.append(suffix)
    return {"role": "system", "content": "\n\n".join(parts)}


def build_agent_system(
    session: SessionConfig,
    name: Optional[str] = None,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Build a subagent's system message and return (system_msg, frontmatter_meta).

    Returns None if a named agent has no AGENT.md on disk.
    """
    if name:
        path = scopes.resolve("agents", name)
        if path is None:
            return None
        meta, body = _load_agent_md(session, path)
    else:
        meta, body = _load_agent_md(session)
    parts: List[str] = []
    if body:
        parts.append(body)
    allowed = meta.get("allowed_tools")
    allowed_set = set(allowed) if isinstance(allowed, list) else None
    parts.append(_build_capabilities(session, allowed_set))
    if meta.get("inherit_memory"):
        memory = _load_memory(session)
        if memory:
            parts.append("# MEMORY\n\n" + memory.strip())
    parts.append(_SYSTEM_SUFFIX)
    return {"role": "system", "content": "\n\n".join(parts)}, meta
