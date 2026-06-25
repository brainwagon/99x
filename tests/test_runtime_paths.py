"""Tests for the runtime-paths section of the system prompt.

The bug under test: the system prompt told the model where its *global* files
live (absolute paths under AGENT_HOME / $HOME) but never stated the absolute
current working directory. With no project anchor and only home-rooted absolute
paths to imitate, the model would create files in $HOME instead of the CWD.
"""

import os

from agent99x.prompt import build_system


def test_system_prompt_states_absolute_cwd(session, in_tmp_cwd):
    content = build_system(session)["content"]

    # The model must be told the absolute working directory so it has a
    # project anchor and doesn't default to home-rooted paths.
    assert str(in_tmp_cwd) == os.getcwd()  # sanity: fixture chdir'd us here
    assert os.getcwd() in content
