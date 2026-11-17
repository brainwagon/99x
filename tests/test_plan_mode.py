"""Plan mode blocks mutating tools, driven by registry traits.

These exercise the dispatcher's trait lookups (core consulting the
registry) rather than any hardcoded tool-name set.
"""

import agent99x.tools  # noqa: F401 — ensure all tools are registered
from agent99x import core


def test_plan_mode_blocks_all_mutating_file_tools(session):
    session.plan_mode = True
    for name in ("write_file", "edit_file", "replace_lines", "patch"):
        reason = core._plan_mode_block_reason(session, name, {})
        assert reason is not None, f"{name} should be blocked in plan mode"


def test_plan_mode_allows_read_only_tools(session):
    session.plan_mode = True
    assert core._plan_mode_block_reason(session, "read_file", {"path": "x"}) is None
    assert core._plan_mode_block_reason(session, "grep", {"pattern": "x"}) is None


def test_plan_mode_hides_mutating_tools_from_offered_list(session):
    session.plan_mode = True
    offered = {t["function"]["name"] for t in core._active_tools(session)}
    assert "write_file" not in offered
    assert "replace_lines" not in offered
    assert "read_file" in offered


def test_no_plan_mode_blocks_nothing(session):
    session.plan_mode = False
    assert core._plan_mode_block_reason(session, "write_file", {}) is None


# ── is_readonly_bash: allowlist classifier for plan-mode run_bash ─────

import pytest

from agent99x.tools import is_readonly_bash

_MUTATING = [
    # holes in the old blocklist
    "sed -i s/a/b/ f.py", "cp a b", "touch x", "perl -pi -e s/a/b/ f",
    "echo hi &>f", "git checkout -- .", "git apply p.diff", "ln -sf /etc/passwd f",
    "python3.10 -c 'open(\"f\",\"w\")'", "bash -c 'r''m -rf x'",
    "find . -name '*.pyc' -exec rm {} +", "mkdir d; ln -sf /etc/passwd f",
    # separators, substitutions, redirections
    "ls; rm x", "ls\nrm x", "ls && rm x", "ls || rm x", "ls & rm x", "ls | xargs rm",
    "echo $(rm x)", "echo `rm x`", "echo \"$(rm x)\"", "cat <(rm x)", "echo hi >(cat)",
    "cat <<EOF > f\nhi\nEOF", "echo hi > f", "echo hi >> f", "echo hi 2> f", "echo >|f",
    "ls >&f", "cat 0<>f", "(ls)", "{ ls; }", "f() { rm x; }; f", "echo 'unterminated",
    "echo ${x@P}", "echo $((1))",
    # dangerous options of allowlisted programs
    "find . -delete", "find . -fprint f", "find . -{delete,true}", "find . ${IFS}-delete",
    "sed -n 'w out' f", "sed 's/a/b/w out' f", "sed 's/a/b/e' f", "sed 1e f", "sed -ni p f",
    "sed --in-place s/a/b/ f", "sed -f s.sed f",
    "awk '{system(\"rm x\")}' f", "awk '{print > \"f\"}' x", "awk -f p.awk f",
    "awk 'BEGIN{\"ls\" | getline l}'",
    "sort -o f x", "sort --out=f x", "sort -uo f x", "uniq a b", "xxd a b", "tree -o f",
    "date -s now", "rg --pre sh x", "file -C -m x", "printf -v PATH x", "env rm x",
    # git: only read-only subcommands/forms
    "git -c core.pager=sh log", "git branch -D x", "git branch newb", "git tag v1",
    "git config user.name x", "git remote add o u", "git reflog expire --all",
    "git diff --output=f", "git log --ext-diff", "git grep -O vi x", "git stash",
    "git stash pop", "git commit -m x", "git push",
    # environment and command names
    "GIT_EXTERNAL_DIFF=x git diff", "PATH=. ls", "LD_PRELOAD=x ls", "./ls", "/bin/rm x",
    "x=1; ls", "rm -rf x", "tee f", "dd if=x of=y", "chmod +x f", "pip install x",
]

_READONLY = [
    # former false positives
    "ls 2>/dev/null", "grep x f >/dev/null", "grep -c x f >/dev/null",
    "ls >/dev/null 2>&1", "ls &>/dev/null", "echo hi 2>&1",
    # ordinary read-only use
    "ls -la", "cat a | grep b | wc -l", "cd src && ls", "grep -rn 'foo|bar' .",
    "grep '$(' f", "echo '`'", "head -n 5 f; tail -3 f", "wc -l < f", "echo $HOME",
    "LANG=C sort f", "test -f x && echo y", "[ -d x ] || echo no", "ls # > comment",
    "find . -name '*.py' -type f", "sed -n '10,20p' f", "sed 's/a/b/g' f",
    "sed -n '/x/{p;q}' f", "sed -E 's/(a|b)+/x/' f", "awk '{print $1}' f",
    "sort -k2,2 f | uniq -c", "jq .a f.json", "date +%Y-%m-%d", "date -Iseconds",
    "du -sh .", "pwd", "which python", "env", "uniq -c f", "xxd f", "tree -L 2",
    "cut -d: -f1 /etc/passwd | sort | uniq", "printf '%s\\n' a b", "rg -n foo src",
    "git status", "git log --oneline -5", "git diff HEAD~1", "git show HEAD@{1}",
    "git branch -a", "git branch --list 'f*'", "git remote -v", "git tag",
    "git tag -l 'v*'", "git config --get user.name", "git config --list", "git reflog",
    "git -C sub log -1", "git --no-pager diff --stat", "git ls-files | head",
    "git stash list", "git rev-parse --show-toplevel", "git blame f",
]


@pytest.mark.parametrize("command", _MUTATING)
def test_is_readonly_bash_rejects(command):
    assert not is_readonly_bash(command)


@pytest.mark.parametrize("command", _READONLY)
def test_is_readonly_bash_accepts(command):
    assert is_readonly_bash(command)


def test_plan_mode_run_bash_uses_classifier(session):
    session.plan_mode = True
    assert core._plan_mode_block_reason(session, "run_bash", {"command": "ls 2>/dev/null"}) is None
    assert core._plan_mode_block_reason(session, "run_bash", {"command": "sed -i s/a/b/ f"}) is not None
