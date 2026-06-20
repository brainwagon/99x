"""Behavior tests for the todos module — the todos.md format behind one interface.

Callers (the write_todos/read_todos tools and the /todo slash commands) go
through load/save/marker and never touch the file format, the status markers,
or the passthrough bookkeeping for unknown lines.
"""

from agent99x import todos as todo_md


def test_load_missing_file_is_empty(tmp_path):
    assert todo_md.load(str(tmp_path / "todos.md")) == []


def test_save_then_load_roundtrips_and_numbers_ids(tmp_path):
    path = str(tmp_path / "todos.md")
    todo_md.save([{"text": "A", "status": "pending"},
                  {"text": "B", "status": "done"}], path)
    loaded = todo_md.load(path)
    assert [t["text"] for t in loaded] == ["A", "B"]
    assert [t["id"] for t in loaded] == [1, 2]
    assert loaded[1]["status"] == "done"


def test_save_accepts_plain_strings_as_pending(tmp_path):
    path = str(tmp_path / "todos.md")
    saved = todo_md.save(["A", {"text": "B", "status": "in_progress"}], path)
    assert saved[0] == {"id": 1, "text": "A", "status": "pending"}
    assert saved[1]["status"] == "in_progress"


def test_save_preserves_unknown_lines(tmp_path):
    path = tmp_path / "todos.md"
    path.write_text("Extra 1\n- [ ] Task A\nExtra 2\n", encoding="utf-8")
    todo_md.save([{"text": "Task A", "status": "done"}], str(path))
    text = path.read_text(encoding="utf-8")
    assert "Extra 1" in text
    assert "Extra 2" in text
    assert "- [x] Task A" in text


def test_save_writes_default_header_for_new_file(tmp_path):
    path = tmp_path / "todos.md"
    todo_md.save([{"text": "A"}], str(path))
    assert "Managed by 99" in path.read_text(encoding="utf-8")


def test_marker_maps_statuses():
    assert todo_md.marker("pending") == " "
    assert todo_md.marker("in_progress") == "~"
    assert todo_md.marker("done") == "x"
    assert todo_md.marker("bogus") == " "
