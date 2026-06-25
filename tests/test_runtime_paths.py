"""Tests for the runtime-paths section of the system prompt.

The bug under test: the system prompt told the model where its *global* files
live (absolute paths under AGENT_HOME / $HOME) but never stated the absolute
current working directory. With no project anchor and only home-rooted absolute
paths to imitate, the model would create files in $HOME instead of the CWD.
"""

import os
import re
from datetime import datetime, timezone

from agent99x.prompt import build_system


def test_system_prompt_states_absolute_cwd(session, in_tmp_cwd):
    content = build_system(session)["content"]

    # The model must be told the absolute working directory so it has a
    # project anchor and doesn't default to home-rooted paths.
    assert str(in_tmp_cwd) == os.getcwd()  # sanity: fixture chdir'd us here
    assert os.getcwd() in content


def test_system_prompt_states_current_time(session):
    content = build_system(session)["content"]

    # The environment section must carry a timezone-aware ISO 8601 timestamp so
    # the model can ground 'recent'/'today' reasoning instead of falling back to
    # its training cutoff.
    match = re.search(r"Current time: `([^`]+)`", content)
    assert match, "system prompt is missing the current-time line"

    stamp = datetime.fromisoformat(match.group(1))
    assert stamp.tzinfo is not None, "timestamp must carry a UTC offset"

    # The timestamp is built per call, so it should be ~now (generous window to
    # avoid flaking on slow CI).
    now = datetime.now(timezone.utc)
    assert abs((now - stamp).total_seconds()) < 300
