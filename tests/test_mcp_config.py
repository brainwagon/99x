import json
import os
import pytest
from agent99x import mcp

def test_load_mcp_config_missing_file(in_tmp_cwd):
    # Should return empty config or handle gracefully
    config = mcp.load_config("non-existent.json")
    assert config == {"mcpServers": {}}

def test_load_mcp_config_valid(in_tmp_cwd):
    config_data = {
        "mcpServers": {
            "test-server": {
                "command": "echo",
                "args": ["hello"],
                "env": {"FOO": "BAR"}
            }
        }
    }
    p = in_tmp_cwd / "mcp-config.json"
    p.write_text(json.dumps(config_data), encoding="utf-8")
    
    config = mcp.load_config(str(p))
    assert config == config_data

def test_load_mcp_config_invalid_json(in_tmp_cwd):
    p = in_tmp_cwd / "mcp-config.json"
    p.write_text("{ invalid json }", encoding="utf-8")
    
    with pytest.raises(mcp.MCPConfigError):
        mcp.load_config(str(p))

def test_mcp_server_config_validation(in_tmp_cwd):
    # Test that missing required fields raise error
    invalid_config = {
        "mcpServers": {
            "bad-server": {
                "args": ["only args, no command"]
            }
        }
    }
    p = in_tmp_cwd / "mcp-config.json"
    p.write_text(json.dumps(invalid_config), encoding="utf-8")
    
    with pytest.raises(mcp.MCPConfigError, match="Missing 'command'"):
        mcp.load_config(str(p))
