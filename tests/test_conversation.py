"""Behavior tests for the session-turn (conversation) module.

The conversation module is the only code that writes history[0], so the
invariant "message 0 is a freshly-built system prompt" is asserted in
nearly every test here.
"""

import json

from agent99x import conversation


def test_start_seeds_history_with_system_message(session):
    conversation.start(session)

    assert len(session.history) == 1
    assert session.history[0]["role"] == "system"


def test_add_user_appends_turn_after_system(session):
    conversation.start(session)

    media_paths, media_errors = conversation.add_user(session, "hello there")

    assert len(session.history) == 2
    assert session.history[0]["role"] == "system"
    assert session.history[1]["role"] == "user"
    assert "hello there" in str(session.history[1]["content"])
    assert media_paths == []
    assert media_errors == []


def test_add_user_rebuilds_system_prompt_each_turn(session):
    """history[0] reflects current session state, not the state at start()."""
    conversation.start(session)
    conversation.add_user(session, "first")
    before = session.history[0]["content"]

    # Flip a state that build_system bakes into the system prompt.
    session.plan_mode = True
    conversation.add_user(session, "second")
    after = session.history[0]["content"]

    assert session.history[0]["role"] == "system"
    assert before != after


def test_clear_drops_turns_to_fresh_system(session):
    conversation.start(session)
    conversation.add_user(session, "one")
    conversation.add_user(session, "two")

    conversation.clear(session)

    assert len(session.history) == 1
    assert session.history[0]["role"] == "system"


def test_replace_with_builds_compact_shape(session):
    conversation.start(session)
    conversation.add_user(session, "lots of prior work")

    conversation.replace_with(session, "[compacted]\nthe summary")

    roles = [m["role"] for m in session.history]
    assert roles == ["system", "user", "assistant"]
    assert session.history[1]["content"] == "[compacted]\nthe summary"
    assert session.history[2]["content"].strip() != ""


def test_load_reads_history_and_rebuilds_system(session, tmp_path):
    path = tmp_path / "session.json"
    path.write_text(json.dumps([
        {"role": "system", "content": "STALE SYSTEM"},
        {"role": "user", "content": "earlier question"},
    ]))

    loaded = conversation.load(session, str(path))

    assert loaded is True
    assert session.history[0]["role"] == "system"
    assert session.history[0]["content"] != "STALE SYSTEM"
    assert session.history[1]["content"] == "earlier question"


def test_load_missing_file_returns_false(session, tmp_path):
    assert conversation.load(session, str(tmp_path / "absent.json")) is False


def test_refresh_system_rebuilds_in_place(session):
    conversation.start(session)
    before = session.history[0]["content"]
    session.history.append({"role": "user", "content": "x"})

    session.plan_mode = True
    conversation.refresh_system(session)

    assert len(session.history) == 2
    assert session.history[0]["role"] == "system"
    assert session.history[0]["content"] != before


def test_truncate_to_rolls_back_to_mark(session):
    conversation.start(session)
    conversation.add_user(session, "keep me")
    mark = len(session.history)
    conversation.add_user(session, "roll me back")

    conversation.truncate_to(session, mark)

    assert len(session.history) == mark
    assert "keep me" in str(session.history[-1]["content"])
