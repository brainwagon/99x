"""Context (saved conversation) file resolution.

A "context" is a saved message history. Contexts live as JSON files under
PROJECT_DIR/contexts/. Each run writes to exactly one context file
(session.context_path), persisted when the session ends.

Naming: a context may have a friendly name (``refactor-auth.json``) chosen via
``-c NAME``; runs started without a name get a timestamped file. ``-c`` with no
value restores the most recently modified context for the project.
"""

import datetime
import glob
import os
from typing import Optional

from agent99x.session import CONTEXTS_DIR

# Sentinel for ``-c`` given with no value: restore the latest context.
LATEST = "\x00latest"


def path_for(name: str) -> str:
    """Path to the named context file (may not exist yet)."""
    name = name[:-5] if name.endswith(".json") else name
    return os.path.join(CONTEXTS_DIR, name + ".json")


def latest_path() -> Optional[str]:
    """The most recently modified context file, or None if there are none."""
    files = glob.glob(os.path.join(CONTEXTS_DIR, "*.json"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def new_path() -> str:
    """A fresh timestamped context path for a new conversation."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return os.path.join(CONTEXTS_DIR, stamp + ".json")
