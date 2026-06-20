import sys
from unittest.mock import MagicMock, patch
import pytest
from agent99x import cli, commands, session
from agent99x import core
from agent99x.session import SessionConfig
from agent99x.reasoning import ContentType

def test_cli_on_token_streams_to_stdout_and_stderr():
    session = SessionConfig()
    session.model = "test-model"
    session.client = MagicMock()
    
    # Mock chunks
    def make_chunk(content=None, reasoning=None):
        delta = MagicMock()
        delta.content = content
        delta.reasoning_content = reasoning
        delta.tool_calls = None
        choice = MagicMock(delta=delta)
        return MagicMock(choices=[choice], usage=None)

    mock_chunks = [
        make_chunk(reasoning="Thinking... "),
        make_chunk(content="Hello world!"),
    ]
    session.client.chat.completions.create.return_value = mock_chunks

    captured_stdout = []
    captured_stderr = []

    with patch("sys.stdout.write", side_effect=captured_stdout.append), \
         patch("sys.stderr.write", side_effect=captured_stderr.append), \
         patch("sys.stdout.flush"), \
         patch("sys.stderr.flush"), \
         patch("agent99x.config_io.save_session"):
        
        # Test one-shot path logic (extracted from main)
        def on_token(ctype, text):
            if ctype == core.ContentType.THINKING:
                sys.stderr.write(f"THINK:{text}")
            else:
                sys.stdout.write(text)

        reply = core.agent_loop(session, on_token=on_token)
        
    assert reply == "Hello world!"
    assert "".join(captured_stdout) == "Hello world!"
    assert "".join(captured_stderr) == "THINK:Thinking... "

@patch("agent99x.cli.input", create=True)
def test_repl_streams_reasoning(mock_input, session, monkeypatch):
    # Setup session
    from agent99x.providers import PROVIDERS
    session.provider = PROVIDERS["ollama"]
    session.model = "test-model"
    session.client = MagicMock()
    
    # Mock input to run once then exit
    mock_input.side_effect = ["Hi", EOFError()]
    
    def make_chunk(content=None, reasoning=None):
        delta = MagicMock()
        delta.content = content
        delta.reasoning_content = reasoning
        delta.tool_calls = None
        choice = MagicMock(delta=delta)
        return MagicMock(choices=[choice], usage=None)

    mock_chunks = [
        make_chunk(reasoning="Reasoning"),
        make_chunk(content="Response"),
    ]
    session.client.chat.completions.create.return_value = mock_chunks

    captured_stdout = []
    captured_stderr = []

    monkeypatch.setattr("sys.stdout.write", captured_stdout.append)
    monkeypatch.setattr("sys.stderr.write", captured_stderr.append)
    monkeypatch.setattr("sys.stdout.flush", lambda: None)
    monkeypatch.setattr("sys.stderr.flush", lambda: None)
    monkeypatch.setattr("agent99x.config_io.save_session", lambda s: None)
    
    # Run REPL
    try:
        cli.repl(session)
    except EOFError:
        pass
    
    # Verify reasoning was printed to stdout (with ANSI codes)
    full_stdout = "".join(captured_stdout)
    assert "Reasoning" in full_stdout
    assert "\033[90m" in full_stdout # Dim code
    
    # Verify response was printed to stdout
    assert "Response" in full_stdout
