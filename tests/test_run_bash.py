"""Tests for the run_bash tool."""

import os
import signal
import threading
import time
from agent99x import tools as tools_bash


def test_bash_result_success_shape():
    bufs = {"stdout": bytearray(b"out"), "stderr": bytearray(b"err")}
    trunc = {"stdout": False, "stderr": False}
    res = tools_bash._bash_result(bufs, trunc, returncode=0)
    assert res == {"stdout": "out", "stderr": "err", "returncode": 0}


def test_bash_result_error_shape():
    bufs = {"stdout": bytearray(), "stderr": bytearray()}
    trunc = {"stdout": False, "stderr": False}
    res = tools_bash._bash_result(bufs, trunc, error="timeout")
    assert res == {"stdout": "", "stderr": "", "error": "timeout"}


def test_run_bash_normal(in_tmp_cwd):
    result = tools_bash.run_bash("echo 'hello world'")
    assert result["stdout"].strip() == "hello world"
    assert result["returncode"] == 0


def test_run_bash_captures_nonzero_returncode(in_tmp_cwd):
    result = tools_bash.run_bash("exit 42")
    assert result["returncode"] == 42


def test_run_bash_truncates_large_output(in_tmp_cwd, monkeypatch):
    monkeypatch.setattr(tools_bash, "MAX_BASH_OUTPUT", 10)
    # Generate 20 bytes
    result = tools_bash.run_bash("echo '01234567890123456789'")
    assert len(result["stdout"]) > 10
    assert "[truncated" in result["stdout"]


def test_run_bash_timeout_kills_process_group(in_tmp_cwd, monkeypatch):
    monkeypatch.setattr(tools_bash, "BASH_TIMEOUT", 0.5)
    t0 = time.monotonic()
    result = tools_bash.run_bash("sleep 10 & sleep 10; wait")
    assert result["error"] == "timeout"
    assert time.monotonic() - t0 < 5


def _clear_thread_local():
    if hasattr(tools_bash._thread_local, "cancel_event"):
        del tools_bash._thread_local.cancel_event


def test_run_bash_cancel_via_thread_local(in_tmp_cwd):
    _clear_thread_local()
    cancel = threading.Event()
    tools_bash._thread_local.cancel_event = cancel

    def _trip():
        time.sleep(0.5)
        cancel.set()

    threading.Thread(target=_trip, daemon=True).start()
    t0 = time.monotonic()
    result = tools_bash.run_bash("sleep 30")
    elapsed = time.monotonic() - t0
    
    assert result["error"] == "cancelled"
    assert elapsed < 5  # Should have been killed quickly
