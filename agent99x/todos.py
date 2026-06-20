"""Todo list persistence: the todos.md on-disk format behind one interface.

Owns parsing, serialization, the status markers, and the "preserve unknown
lines" rule, so callers (the write_todos/read_todos tools and the /todo slash
commands) never touch the file format or the passthrough bookkeeping.
"""

import os
import re
import threading
from typing import Any, Dict, List, Tuple, Union

from agent99x.session import PROJECT_DIR

TODOS_FILE = os.path.join(PROJECT_DIR, "todos.md")

Todo = Dict[str, Any]  # {"id": int, "text": str, "status": str}

_LINE_RE = re.compile(r'^\s*-\s*\[([ ~xX])\]\s*(.+?)\s*$')
_STATUS = {' ': 'pending', '~': 'in_progress', 'x': 'done', 'X': 'done'}
_MARKER = {'pending': ' ', 'in_progress': '~', 'done': 'x'}
_DEFAULT_HEADER = [
    "<!-- Managed by 99x. Edit freely; the agent will preserve unknown lines.",
    "     [ ] pending   [~] in progress   [x] done -->",
    "",
]

_lock = threading.RLock()


def marker(status: str) -> str:
    """The single-character checkbox marker for a status."""
    return _MARKER.get(status, ' ')


def load(path: str = TODOS_FILE) -> List[Todo]:
    """Return the current todos (ids assigned in file order)."""
    todos, _passthrough = _read(path)
    return todos


def save(items: List[Union[Todo, str]], path: str = TODOS_FILE) -> List[Todo]:
    """Persist a full todo list, preserving unknown lines; return the normalized list."""
    _existing, passthrough = _read(path)
    normalized: List[Todo] = []
    for i, item in enumerate(items, 1):
        if isinstance(item, str):
            item = {"text": item}
        normalized.append({"id": i, "text": item["text"],
                           "status": item.get("status", "pending")})
    _write(normalized, passthrough, path)
    return normalized


def _read(path: str) -> Tuple[List[Todo], List[Tuple[int, str]]]:
    """Return (todos, passthrough) where passthrough is [(insert_before_index, raw_line), ...]."""
    todos: List[Todo] = []
    passthrough: List[Tuple[int, str]] = []
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return todos, passthrough
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if m:
            todos.append({"id": len(todos) + 1,
                          "text": m.group(2),
                          "status": _STATUS.get(m.group(1), 'pending')})
        else:
            passthrough.append((len(todos), line))
    return todos, passthrough


def _write(todos: List[Todo], passthrough: List[Tuple[int, str]], path: str) -> None:
    if not passthrough and not os.path.exists(path):
        passthrough = [(0, line) for line in _DEFAULT_HEADER]
    out: List[str] = []
    pt_idx = 0
    n = len(todos)
    for i in range(n):
        while pt_idx < len(passthrough) and passthrough[pt_idx][0] <= i:
            out.append(passthrough[pt_idx][1])
            pt_idx += 1
        out.append(f"- [{marker(todos[i]['status'])}] {todos[i]['text']}")
    while pt_idx < len(passthrough):
        out.append(passthrough[pt_idx][1])
        pt_idx += 1
    text = "\n".join(out)
    if text and not text.endswith("\n"):
        text += "\n"
    tmp = path + ".tmp"
    with _lock:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
