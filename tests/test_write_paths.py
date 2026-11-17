"""All four writing tools share one path check (cwd or AGENT_HOME)."""

import os

import pytest

from agent99x import session, tools

DIFF = "--- a\n+++ b\n@@ -1 +1 @@\n-old\n+new\n"


def _call(name, path):
    if name == "write_file":
        return tools.write_file(path, "new")
    if name == "edit_file":
        return tools.edit_file(path, "old", "new")
    if name == "replace_lines":
        return tools.replace_lines(path, 1, 1, "new")
    return tools.apply_patch(path, DIFF)


NAMES = ["write_file", "edit_file", "replace_lines", "patch"]


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    work = tmp_path / "w"
    work.mkdir()
    (tmp_path / "w-evil").mkdir()
    (tmp_path / "outside").mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(session, "AGENT_HOME", str(home))
    return tmp_path


def _target(d, rel):
    p = d / rel
    p.write_text("old\n")
    return p


@pytest.mark.parametrize("name", NAMES)
def test_absolute_outside_rejected(dirs, name):
    p = _target(dirs, "outside/f.txt")
    assert "error" in _call(name, str(p))
    assert p.read_text() == "old\n"


@pytest.mark.parametrize("name", NAMES)
def test_dotdot_escape_rejected(dirs, name):
    p = _target(dirs, "outside/f.txt")
    assert "error" in _call(name, "../outside/f.txt")
    assert p.read_text() == "old\n"


@pytest.mark.parametrize("name", NAMES)
def test_symlink_file_out_rejected(dirs, name):
    p = _target(dirs, "outside/f.txt")
    os.symlink(p, dirs / "w" / "link.txt")
    assert "error" in _call(name, "link.txt")
    assert p.read_text() == "old\n"


@pytest.mark.parametrize("name", NAMES)
def test_new_file_under_symlinked_dir_rejected(dirs, name):
    os.symlink(dirs / "outside", dirs / "w" / "d")
    r = _call(name, "d/new.txt")
    assert "error" in r
    assert not (dirs / "outside" / "new.txt").exists()


@pytest.mark.parametrize("name", NAMES)
def test_sibling_prefix_dir_rejected(dirs, name):
    p = _target(dirs, "w-evil/f.txt")
    assert "error" in _call(name, str(p))
    assert p.read_text() == "old\n"


@pytest.mark.parametrize("name", NAMES)
def test_inside_cwd_ok(dirs, name):
    p = _target(dirs, "w/f.txt")
    assert "error" not in _call(name, "f.txt")
    assert p.read_text().startswith("new")


@pytest.mark.parametrize("name", NAMES)
def test_inside_agent_home_ok(dirs, name):
    p = _target(dirs, "home/MEMORY.md")
    assert "error" not in _call(name, str(p))
    assert p.read_text().startswith("new")


def test_write_file_creates_new_file_in_cwd(dirs):
    assert "error" not in tools.write_file("sub_new.txt", "x")
    assert (dirs / "w" / "sub_new.txt").read_text() == "x"


def test_blocked_path_not_snapshotted(dirs):
    """core must not snapshot (and so never roll back) outside paths."""
    assert tools.check_write_path(str(dirs / "outside" / "x.py"))
