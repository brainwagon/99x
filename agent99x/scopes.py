"""Scope resolution for skills and agents.

Owns the precedence rule "project shadows global" so callers (skill
catalog/loader, agent catalog/loader) stop hand-rolling directory lists.
"""

import os
from typing import List, Optional, Tuple

from agent99x.session import AGENT_HOME, PROJECT_DIR

_MANIFEST = {"skills": "SKILL.md", "agents": "AGENT.md"}


def search_dirs(kind: str) -> List[str]:
    """Return the scope dirs for a kind in priority order (project first)."""
    return [
        os.path.join(PROJECT_DIR, kind),
        os.path.join(AGENT_HOME, kind),
    ]


def resolve(kind: str, name: str) -> Optional[str]:
    """Return the winning manifest path for a named skill/agent, or None."""
    manifest = _MANIFEST[kind]
    for base in search_dirs(kind):
        path = os.path.join(base, name, manifest)
        if os.path.exists(path):
            return path
    return None


def discover(kind: str) -> List[Tuple[str, str]]:
    """Return (name, manifest path) for every skill/agent, sorted by name.

    Project-local entries shadow global ones.
    """
    manifest = _MANIFEST[kind]
    seen: dict[str, str] = {}
    for base in reversed(search_dirs(kind)):  # lowest priority first; project overwrites
        try:
            names = os.listdir(base)
        except OSError:
            continue
        for name in names:
            path = os.path.join(base, name, manifest)
            if os.path.exists(path):
                seen[name] = path
    return sorted(seen.items())
