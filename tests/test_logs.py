import logging
import pytest
from agent99x import logs

def test_log_tool_call_no_crash():
    """Verify that log_tool_call does not raise KeyError due to reserved attribute clash."""
    logs.log_tool_call("test_tool", {"arg1": "val1"})

def test_log_tool_result_no_crash():
    """Verify that log_tool_result does not raise KeyError due to reserved attribute clash."""
    logs.log_tool_result("test_tool", "success", 1.23)

def test_log_attributes(caplog):
    """Verify that log records contain the expected custom attributes."""
    caplog.set_level(logging.INFO, logger="agent99x")
    
    # Test tool call
    logs.log_tool_call("my_tool", {"a": 1})
    record = caplog.records[-1]
    assert getattr(record, "tool_name") == "my_tool"
    assert getattr(record, "tool_args") == {"a": 1}
    assert getattr(record, "type") == "tool_call"

    # Test tool result
    logs.log_tool_result("my_tool", "ok", 0.5)
    record = caplog.records[-1]
    assert getattr(record, "tool_name") == "my_tool"
    assert getattr(record, "result") == "ok"
    assert getattr(record, "elapsed") == 0.5
    assert getattr(record, "type") == "tool_result"
