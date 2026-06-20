"""Tests for the todo list tool."""

from agent99x import tools as tools_state
from agent99x import todos as todo_md


def test_todo_roundtrip(in_tmp_cwd):
    todos = [
        {"text": "Task 1", "status": "pending"},
        {"text": "Task 2", "status": "done"},
    ]
    tools_state.write_todos(todos)

    result = tools_state.read_todos()
    assert len(result["todos"]) == 2
    assert result["todos"][0]["text"] == "Task 1"
    assert result["todos"][1]["status"] == "done"


def test_todo_markdown_preserves_extra_lines(in_tmp_cwd):
    path = in_tmp_cwd / ".99x" / "todos.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("Extra line 1\n- [ ] Task A\nExtra line 2\n", encoding="utf-8")

    assert len(todo_md.load(str(path))) == 1

    tools_state.write_todos([{"text": "Task A", "status": "done"}])

    text = path.read_text(encoding="utf-8")
    assert "Extra line 1" in text
    assert "Extra line 2" in text
    assert "- [x] Task A" in text


def test_write_todos_normalizes_string_input(in_tmp_cwd):
    tools_state.write_todos(["Task 1", {"text": "Task 2", "status": "in_progress"}])
    result = tools_state.read_todos()
    assert result["todos"][0]["text"] == "Task 1"
    assert result["todos"][0]["status"] == "pending"
    assert result["todos"][1]["status"] == "in_progress"
