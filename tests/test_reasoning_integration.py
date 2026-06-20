import pytest
from unittest.mock import MagicMock, patch
from agent99x import core
from agent99x.session import SessionConfig
from agent99x.reasoning import ContentType

@pytest.mark.asyncio
async def test_agent_loop_collects_reasoning():
    session = SessionConfig()
    session.model = "test-model"
    session.client = MagicMock()
    
    # Mock chunks with actual strings
    def make_chunk(content=None, reasoning=None):
        delta = MagicMock()
        delta.content = content
        delta.reasoning_content = reasoning
        delta.tool_calls = None
        choice = MagicMock(delta=delta)
        return MagicMock(choices=[choice], usage=None)

    mock_chunks = [
        make_chunk(content="<think>I am thinking</think>"),
        make_chunk(content=" Hello!"),
        MagicMock(choices=[], usage=MagicMock(prompt_tokens=10))
    ]
    
    session.client.chat.completions.create.return_value = mock_chunks
    
    tokens = []
    def on_token(ctype, text):
        tokens.append((ctype, text))
        
    reply = core.agent_loop(session, on_token=on_token)
    
    assert reply == " Hello!"
    thinking = "".join(t for c, t in tokens if c == ContentType.THINKING)
    content = "".join(t for c, t in tokens if c == ContentType.CONTENT)
    
    assert "I am thinking" in thinking
    assert " Hello!" in content
    
    # Verify reasoning NOT in history
    assert session.history[-1]["role"] == "assistant"
    assert session.history[-1]["content"] == " Hello!"
    assert "<think>" not in session.history[-1]["content"]

@pytest.mark.asyncio
async def test_agent_loop_explicit_reasoning():
    session = SessionConfig()
    session.model = "test-model"
    session.client = MagicMock()
    
    def make_chunk(content=None, reasoning=None):
        delta = MagicMock()
        delta.content = content
        delta.reasoning_content = reasoning
        delta.tool_calls = None
        choice = MagicMock(delta=delta)
        return MagicMock(choices=[choice], usage=None)

    mock_chunks = [
        make_chunk(reasoning="Reasoning part."),
        make_chunk(content="Final answer."),
    ]
    
    session.client.chat.completions.create.return_value = mock_chunks
    
    tokens = []
    def on_token(ctype, text):
        tokens.append((ctype, text))
        
    reply = core.agent_loop(session, on_token=on_token)
    
    assert reply == "Final answer."
    thinking = "".join(t for c, t in tokens if c == ContentType.THINKING)
    content = "".join(t for c, t in tokens if c == ContentType.CONTENT)
    
    assert "Reasoning part." in thinking
    assert "Final answer." in content
    
    assert session.history[-1]["content"] == "Final answer."
