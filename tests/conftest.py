"""Shared fixtures for the 99 test suite."""

import os
import sys
from types import SimpleNamespace

import pytest

# Make the repo root importable so `import agent99x` works regardless of pytest's cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    """Run the test inside an empty tmp directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _delta(**kw):
    return SimpleNamespace(**kw)


def _chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    """Build a minimal chat-completion stream chunk shaped like the OpenAI SDK."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _usage_chunk(prompt_tokens):
    """A trailing usage-only chunk with no choices, matching include_usage=True."""
    return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=prompt_tokens))


def _tool_call_delta(index, *, id=None, name=None, arguments=None):
    """Shape mirrors what _stream_completion reads off delta.tool_calls."""
    fn = SimpleNamespace(name=name, arguments=arguments) if (name or arguments) else None
    return SimpleNamespace(index=index, id=id, function=fn)


class FakeStream:
    """Iterable of chunks, with a no-op close()."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        pass


@pytest.fixture
def fake_lm(monkeypatch):
    """Replace session.client with a fake whose chat.completions.create
    returns pre-programmed FakeStreams.

    Tests append streams via fake_lm.queue(chunks). Each call to
    client.chat.completions.create pops the next queued stream.
    """
    from agent99x.session import SessionConfig

    queued = []

    def create(**_kwargs):
        if not queued:
            raise AssertionError("fake_lm: no streams queued")
        return FakeStream(queued.pop(0))

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    
    # Add helper methods to the fake
    fake.queue = lambda chunks: queued.append(list(chunks))
    fake.chunk = _chunk
    fake.usage_chunk = _usage_chunk
    fake.tool_call_delta = _tool_call_delta
    
    # We will need to inject this into the session during tests
    return fake

@pytest.fixture
def session(fake_lm):
    """Return a SessionConfig pre-configured with a fake_lm client."""
    from agent99x.session import SessionConfig
    from agent99x.providers import PROVIDERS
    s = SessionConfig(
        provider=PROVIDERS["ollama"],
        model="test-model",
        context_window=131072,
        client=fake_lm
    )
    return s
