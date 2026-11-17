"""Post-edit syntax checks for files the agent modifies.

A check reports only genuine syntax errors.  A checker that is missing,
times out, or fails for reasons unrelated to the file's syntax (e.g. a C
file whose includes can't be found) counts as "unavailable" and passes.
"""

import configparser
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional

CHECK_TIMEOUT = 20.0  # seconds per external checker

# gcc/clang diagnostics meaning "couldn't resolve the environment", not
# "this file is malformed".
_CC_UNAVAILABLE = re.compile(
    r"fatal error: [^\n]*(No such file or directory|file not found)"
    r"|cannot execute|error trying to exec",
)


def _run(cmd: List[str]) -> Optional[str]:
    """Run an external checker; return its output on failure, None if it
    passed or could not run."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=CHECK_TIMEOUT, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None  # not installed, not executable, or timed out
    if proc.returncode == 0:
        return None
    return ((proc.stderr or "") + (proc.stdout or "")).strip() or (
        f"{cmd[0]} exited with status {proc.returncode}")


def _check_python(fp: str) -> Optional[str]:
    # In-process compile: same interpreter as the agent, no __pycache__.
    with open(fp, "rb") as f:
        source = f.read()
    try:
        compile(source, fp, "exec", dont_inherit=True)
    except SyntaxError as e:
        return f"{type(e).__name__}: {e.msg} (line {e.lineno})"
    except ValueError as e:  # e.g. source contains null bytes
        return str(e)
    except (RecursionError, MemoryError):
        return None
    return None


def _check_c(compiler: str) -> Callable[[str], Optional[str]]:
    def check(fp: str) -> Optional[str]:
        here = os.path.dirname(os.path.abspath(fp))
        out = _run([compiler, "-fsyntax-only", "-I", here, "-I", os.getcwd(), fp])
        if out and _CC_UNAVAILABLE.search(out):
            return None
        return out
    return check


def _check_json(fp: str) -> Optional[str]:
    try:
        with open(fp, encoding="utf-8-sig") as f:
            json.load(f)
    except (ValueError, UnicodeDecodeError) as e:
        return str(e)
    return None


def _check_yaml(fp: str) -> Optional[str]:
    try:
        import yaml
    except ImportError:
        return None
    try:
        with open(fp, encoding="utf-8") as f:
            for _doc in yaml.safe_load_all(f):
                pass
    except yaml.YAMLError as e:
        return str(e)
    except UnicodeDecodeError as e:
        return str(e)
    return None


def _check_ini(fp: str) -> Optional[str]:
    try:
        with open(fp, encoding="utf-8") as f:
            configparser.ConfigParser().read_file(f)
    except (configparser.Error, UnicodeDecodeError) as e:
        return str(e)
    return None


def _check_toml(fp: str) -> Optional[str]:
    try:
        import tomllib
    except ImportError:  # Python < 3.11
        return None
    try:
        with open(fp, "rb") as f:
            tomllib.load(f)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        return str(e)
    return None


def _check_xml(fp: str) -> Optional[str]:
    try:
        ET.parse(fp)
    except ET.ParseError as e:
        return str(e)
    return None


# HTML and CSS are deliberately unchecked: the available tools (htmlhint,
# stylelint) report lint and missing-config problems, not syntax errors.
_CHECKERS: Dict[str, Callable[[str], Optional[str]]] = {
    ".py": _check_python,
    ".sh": lambda fp: _run(["bash", "-n", fp]),
    ".bash": lambda fp: _run(["bash", "-n", fp]),
    ".c": _check_c("gcc"),
    ".cpp": _check_c("g++"),
    ".cc": _check_c("g++"),
    ".cxx": _check_c("g++"),
    ".h": _check_c("g++"),
    ".hpp": _check_c("g++"),
    ".js": lambda fp: _run(["node", "--check", fp]),
    ".json": _check_json,
    ".yaml": _check_yaml,
    ".yml": _check_yaml,
    ".ini": _check_ini,
    ".toml": _check_toml,
    ".xml": _check_xml,
    ".rss": _check_xml,
}


def validate_file(fp: str) -> Optional[str]:
    """Return a syntax-error report for one file, or None if it is valid,
    missing, of an unchecked type, or its checker is unavailable."""
    if not os.path.isfile(fp):
        return None
    check = _CHECKERS.get(os.path.splitext(fp)[1].lower())
    if check is None:
        return None
    try:
        err = check(fp)
    except OSError:
        return None
    return f"Syntax error in {fp}:\n{err.strip()}" if err else None


def validate_files(filepaths: List[str]) -> Optional[str]:
    """Run syntax validation on modified files. Return error string if failed."""
    errors = [e for e in (validate_file(fp) for fp in filepaths) if e]
    return "\n\n".join(errors) if errors else None
