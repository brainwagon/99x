"""Tests for the loop nudging the model to finish a loaded skill's actions.

When the model calls ``load_skill`` it receives the skill's *steps* as a tool
result and is expected to carry them out (more tool calls) before reporting
back. The bug under test: the loop would return the model's plain-text reply to
the user even though the skill's actions never ran.
"""

import json

from agent99x import core


def _make_skill(root, name="demo", body="# Demo\n\nDo the thing."):
    """Create a project-local skill at ./.99x/skills/<name>/SKILL.md."""
    skill_dir = root / ".99x" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\ndescription: a demo skill\n---\n{body}\n", encoding="utf-8"
    )
    return name


def test_loop_nudges_when_skill_loaded_but_not_carried_out(session, fake_lm, in_tmp_cwd):
    name = _make_skill(in_tmp_cwd)

    # Turn 1: model loads the skill.
    fake_lm.queue([
        fake_lm.chunk(tool_calls=[
            fake_lm.tool_call_delta(0, id="call_1", name="load_skill",
                                    arguments=json.dumps({"name": name})),
        ]),
        fake_lm.chunk(finish_reason="tool_calls"),
    ])
    # Turn 2: model stops with plain prose instead of carrying out the steps.
    fake_lm.queue([
        fake_lm.chunk(content="Here are the steps: 1, 2, 3."),
        fake_lm.chunk(finish_reason="stop"),
    ])
    # Turn 3: after being nudged, the model reports genuine completion.
    fake_lm.queue([
        fake_lm.chunk(content="Done — carried out the steps."),
        fake_lm.chunk(finish_reason="stop"),
    ])

    session.history = [{"role": "user", "content": "do the demo skill"}]
    result = core.agent_loop(session)

    # The premature prose must not be what the user gets back.
    assert result == "Done — carried out the steps."
    # A nudge (user turn) must have been injected after the skill was loaded.
    nudges = [m for m in session.history
              if m["role"] == "user" and "skill" in m["content"].lower()
              and m["content"] != "do the demo skill"]
    assert nudges, "expected a nudge prompting the model to carry out the skill"


def test_loop_nudges_at_most_once_for_a_loaded_skill(session, fake_lm, in_tmp_cwd):
    name = _make_skill(in_tmp_cwd)

    fake_lm.queue([
        fake_lm.chunk(tool_calls=[
            fake_lm.tool_call_delta(0, id="call_1", name="load_skill",
                                    arguments=json.dumps({"name": name})),
        ]),
        fake_lm.chunk(finish_reason="tool_calls"),
    ])
    # The model stubbornly returns prose twice in a row.
    fake_lm.queue([
        fake_lm.chunk(content="First prose."),
        fake_lm.chunk(finish_reason="stop"),
    ])
    fake_lm.queue([
        fake_lm.chunk(content="Second prose, still not acting."),
        fake_lm.chunk(finish_reason="stop"),
    ])

    session.history = [{"role": "user", "content": "do the demo skill"}]
    result = core.agent_loop(session)

    # After one nudge, the second prose reply is allowed through to the user.
    assert result == "Second prose, still not acting."
    nudges = [m for m in session.history
              if m["role"] == "user" and "carried out its steps" in m["content"]]
    assert len(nudges) == 1, "the skill nudge must fire at most once"


def test_loop_does_not_nudge_when_skill_steps_were_carried_out(session, fake_lm, in_tmp_cwd):
    name = _make_skill(in_tmp_cwd)
    target = in_tmp_cwd / "out.txt"
    target.write_text("data", encoding="utf-8")

    # Turn 1: load the skill.
    fake_lm.queue([
        fake_lm.chunk(tool_calls=[
            fake_lm.tool_call_delta(0, id="call_1", name="load_skill",
                                    arguments=json.dumps({"name": name})),
        ]),
        fake_lm.chunk(finish_reason="tool_calls"),
    ])
    # Turn 2: actually carry out a step (a real action tool clears the flag).
    fake_lm.queue([
        fake_lm.chunk(tool_calls=[
            fake_lm.tool_call_delta(0, id="call_2", name="read_file",
                                    arguments=json.dumps({"path": str(target)})),
        ]),
        fake_lm.chunk(finish_reason="tool_calls"),
    ])
    # Turn 3: legitimately report completion in prose.
    fake_lm.queue([
        fake_lm.chunk(content="Skill complete."),
        fake_lm.chunk(finish_reason="stop"),
    ])

    session.history = [{"role": "user", "content": "do the demo skill"}]
    result = core.agent_loop(session)

    assert result == "Skill complete."
    nudges = [m for m in session.history
              if m["role"] == "user" and "carried out its steps" in m["content"]]
    assert not nudges, "no nudge expected once the skill's steps were carried out"
