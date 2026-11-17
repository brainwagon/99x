"""Gemma 4 support: <|think|> injection, channel-tag parsing, reasoning field."""
from unittest.mock import MagicMock

from agent99x import core
from agent99x.llm import THINK_TOKEN, wants_think_token, with_think_token
from agent99x.reasoning import ReasoningParser, ContentType
from agent99x.session import SessionConfig


# ── <|think|> injection ────────────────────────────────────────────

def _session(model, effort):
    s = SessionConfig()
    s.model = model
    s.effort = effort
    return s


def test_wants_think_token_gemma4_with_effort():
    assert wants_think_token(_session("gemma4:e4b", "medium"))
    assert wants_think_token(_session("google/gemma-4-E2B-it", "high"))
    assert wants_think_token(_session("GEMMA_4-26B", "low"))


def test_wants_think_token_negative_cases():
    assert not wants_think_token(_session("gemma4:e4b", None))  # effort unset
    assert not wants_think_token(_session("gemma3:12b", "high"))  # wrong family
    assert not wants_think_token(_session("qwen3:4b", "high"))
    assert not wants_think_token(_session(None, "high"))


def test_with_think_token_prepends_to_system():
    msgs = [{"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"}]
    out = with_think_token(msgs)
    assert out[0]["content"] == THINK_TOKEN + "Be helpful."
    assert out[1] is msgs[1]
    # Original history must not be mutated.
    assert msgs[0]["content"] == "Be helpful."


def test_with_think_token_idempotent():
    msgs = [{"role": "system", "content": THINK_TOKEN + "Be helpful."}]
    assert with_think_token(msgs) is msgs


def test_with_think_token_no_system_message():
    msgs = [{"role": "user", "content": "Hi"}]
    out = with_think_token(msgs)
    assert out[0] == {"role": "system", "content": THINK_TOKEN}
    assert out[1] is msgs[0]


# ── channel-tag fallback parsing ───────────────────────────────────

def _run_parser(chunks):
    parser = ReasoningParser()
    results = []
    for chunk in chunks:
        results.extend(parser.feed_content(chunk))
    results.extend(parser.flush())
    thinking = "".join(t for c, t in results if c == ContentType.THINKING)
    content = "".join(t for c, t in results if c == ContentType.CONTENT)
    return thinking, content


def test_channel_tags_basic():
    thinking, content = _run_parser(
        ["<|channel>thought\nLet me work this out.<channel|>The answer is 4."])
    assert thinking == "Let me work this out."
    assert content == "The answer is 4."
    assert "<|channel>" not in content
    assert "<channel|>" not in content


def test_channel_tags_split_across_chunks():
    thinking, content = _run_parser(
        ["<|chan", "nel>thought\nStep 1. ", "Step 2.<chan", "nel|>Done."])
    assert thinking == "Step 1. Step 2."
    assert content == "Done."


def test_channel_tags_empty_thought():
    # Non-thinking gemma4 output: tags present, thought section blank.
    thinking, content = _run_parser(["<|channel>thought\n<channel|>Just the answer."])
    assert thinking == ""
    assert content == "Just the answer."


def test_think_tags_still_work():
    thinking, content = _run_parser(["<think>hmm</think>ok"])
    assert thinking == "hmm"
    assert content == "ok"


# ── mid-word newline hack scoped to gemma3 ─────────────────────────

def test_numbered_lists_survive_by_default():
    parser = ReasoningParser()
    text = "1. Albert Einstein\n2. Marie Curie\n"
    results = parser.feed_content(text)
    results.extend(parser.flush())
    assert "".join(t for _, t in results) == text


def test_midword_strip_active_when_enabled():
    parser = ReasoningParser(strip_midword_newlines=True)
    results = parser.feed_content("Ein\nstein was here.")
    results.extend(parser.flush())
    assert "".join(t for _, t in results) == "Einstein was here."


# ── delta.reasoning field (ollama / OpenRouter) ────────────────────

def test_stream_reads_delta_reasoning_field():
    session = SessionConfig()
    session.model = "gemma4:e4b"
    session.client = MagicMock()

    def make_chunk(content=None, reasoning=None):
        delta = MagicMock()
        delta.content = content
        delta.reasoning_content = None
        delta.reasoning = reasoning
        delta.tool_calls = None
        choice = MagicMock(delta=delta)
        return MagicMock(choices=[choice], usage=None)

    session.client.chat.completions.create.return_value = [
        make_chunk(reasoning="I should say hi."),
        make_chunk(content="Hi!"),
    ]

    tokens = []
    reply = core.agent_loop(session, on_token=lambda c, t: tokens.append((c, t)))

    assert reply == "Hi!"
    thinking = "".join(t for c, t in tokens if c == ContentType.THINKING)
    assert "I should say hi." in thinking
    # Reasoning must not be fed back into history.
    assert session.history[-1]["content"] == "Hi!"
