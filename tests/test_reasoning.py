import pytest
from agent99x.reasoning import ReasoningParser, ContentType

def test_basic_think_tags():
    parser = ReasoningParser()
    chunks = [
        "Hello, let me think. <think>",
        "I should say hello.",
        "</think> Hi there!"
    ]
    
    all_results = []
    for chunk in chunks:
        all_results.extend(parser.feed_content(chunk))
    all_results.extend(parser.flush())
    
    # Expected sequence:
    # (CONTENT, "Hello, let me think. ")
    # (THINKING, "I should say hello.")
    # (CONTENT, " Hi there!")
    
    full_content = "".join(t for c, t in all_results if c == ContentType.CONTENT)
    full_thinking = "".join(t for c, t in all_results if c == ContentType.THINKING)
    
    assert "Hello, let me think. " in full_content
    assert "Hi there!" in full_content
    assert "I should say hello." in full_thinking
    assert "<think>" not in full_content
    assert "</think>" not in full_content

def test_split_tags():
    parser = ReasoningParser()
    # Split <think> across chunks
    chunks = ["Part 1 <th", "ink> Inside </th", "ink> Part 2"]
    
    all_results = []
    for chunk in chunks:
        all_results.extend(parser.feed_content(chunk))
    all_results.extend(parser.flush())
    
    full_content = "".join(t for c, t in all_results if c == ContentType.CONTENT)
    full_thinking = "".join(t for c, t in all_results if c == ContentType.THINKING)
    
    assert "Part 1 " in full_content
    assert "Part 2" in full_content
    assert " Inside " in full_thinking

def test_generic_pattern():
    parser = ReasoningParser()
    chunks = ["Thought: I should do this. ", "Actually, let's do that."]
    
    all_results = []
    for chunk in chunks:
        all_results.extend(parser.feed_content(chunk))
    all_results.extend(parser.flush())
    
    full_thinking = "".join(t for c, t in all_results if c == ContentType.THINKING)
    assert "Thought: I should do this." in full_thinking
    assert "Actually, let's do that." in full_thinking

def test_explicit_reasoning():
    parser = ReasoningParser()
    all_results = parser.feed_reasoning("Reasoning from o1.")
    all_results.extend(parser.flush())
    
    assert all_results == [(ContentType.THINKING, "Reasoning from o1.")]

def test_mixed_tags_and_reasoning():
    parser = ReasoningParser()
    res1 = parser.feed_reasoning("Reasoning A. ")
    res2 = parser.feed_content("Content B. <think>Thinking C.</think> Content D.")
    res3 = parser.flush()
    
    all_res = res1 + res2 + res3
    
    thinking = "".join(t for c, t in all_res if c == ContentType.THINKING)
    content = "".join(t for c, t in all_res if c == ContentType.CONTENT)
    
    assert "Reasoning A." in thinking
    assert "Thinking C." in thinking
    assert "Content B." in content
    assert "Content D." in content

def test_no_tags():
    parser = ReasoningParser()
    res = parser.feed_content("Just plain old content.")
    res.extend(parser.flush())
    
    full = "".join(t for c, t in res)
    assert full == "Just plain old content."
    assert all(c == ContentType.CONTENT for c, t in res)
