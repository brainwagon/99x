import re
from enum import Enum
from typing import List, Tuple

_WORD_CHAR = re.compile(r'\w')


class ContentType(Enum):
    CONTENT = "content"
    THINKING = "thinking"


class ReasoningParser:
    """Parses streaming LLM output to separate thinking/reasoning from content."""

    def __init__(self):
        self.in_think_tag = False
        self.buffer = ""
        self.think_start_tag = "<think>"
        self.think_end_tag = "</think>"

        # Generic pattern state
        self.generic_patterns = ["Thought:", "Reasoning:"]
        self.in_generic_reasoning = False
        self.pending_newline = False
        self.last_was_word = False

    def _process_content(self, text: str) -> str:
        """Strip mid-word newlines from streamed content (a Gemma tokenizer artifact)."""
        if not text:
            return ""

        out = []

        if self.pending_newline:
            self.pending_newline = False
            if self.last_was_word and _WORD_CHAR.match(text[0]):
                pass  # mid-word — drop
            else:
                out.append("\n")
                self.last_was_word = False

        text = re.sub(r'(?<=\w)\n(?=\w)', '', text)

        if text.endswith("\n") and not text.endswith("\n\n"):
            self.pending_newline = True
            text = text[:-1]

        if text:
            self.last_was_word = bool(_WORD_CHAR.match(text[-1]))

        out.append(text)
        return "".join(out)

    def _emit(self, ctype: ContentType, text: str, results: List[Tuple[ContentType, str]]):
        if ctype == ContentType.CONTENT:
            text = self._process_content(text)
        if text:
            results.append((ctype, text))

    def feed_content(self, chunk: str) -> List[Tuple[ContentType, str]]:
        """Feed a chunk of regular content (which might contain think tags)."""
        results = []
        self.buffer += chunk

        while self.buffer:
            if not self.in_think_tag:
                if not self.in_generic_reasoning:
                    found_pattern = None
                    pattern_idx = -1
                    for p in self.generic_patterns:
                        idx = self.buffer.find(p)
                        if idx != -1 and (pattern_idx == -1 or idx < pattern_idx):
                            pattern_idx = idx
                            found_pattern = p

                    if found_pattern:
                        if pattern_idx > 0:
                            self._emit(ContentType.CONTENT, self.buffer[:pattern_idx], results)
                        self.in_generic_reasoning = True
                        self.buffer = self.buffer[pattern_idx:]

                start_idx = self.buffer.find(self.think_start_tag)
                if start_idx != -1:
                    if start_idx > 0:
                        ctype = ContentType.THINKING if self.in_generic_reasoning else ContentType.CONTENT
                        self._emit(ctype, self.buffer[:start_idx], results)
                    self.in_think_tag = True
                    self.buffer = self.buffer[start_idx + len(self.think_start_tag):]
                else:
                    max_tag_len = max(len(self.think_start_tag), len(self.think_end_tag))
                    safe_len = max_tag_len - 1
                    if len(self.buffer) > safe_len:
                        ctype = ContentType.THINKING if self.in_generic_reasoning else ContentType.CONTENT
                        self._emit(ctype, self.buffer[:-safe_len], results)
                        self.buffer = self.buffer[-safe_len:]
                    break
            else:
                end_idx = self.buffer.find(self.think_end_tag)
                if end_idx != -1:
                    if end_idx > 0:
                        self._emit(ContentType.THINKING, self.buffer[:end_idx], results)
                    self.in_think_tag = False
                    self.buffer = self.buffer[end_idx + len(self.think_end_tag):]
                else:
                    safe_len = len(self.think_end_tag) - 1
                    if len(self.buffer) > safe_len:
                        self._emit(ContentType.THINKING, self.buffer[:-safe_len], results)
                        self.buffer = self.buffer[-safe_len:]
                    break
        return results

    def feed_reasoning(self, chunk: str) -> List[Tuple[ContentType, str]]:
        """Feed a chunk of explicit reasoning content (e.g. from a reasoning model)."""
        return [(ContentType.THINKING, chunk)]

    def flush(self) -> List[Tuple[ContentType, str]]:
        """Flush remaining buffer."""
        results = []
        if self.buffer:
            ctype = ContentType.THINKING if (self.in_think_tag or self.in_generic_reasoning) else ContentType.CONTENT
            self._emit(ctype, self.buffer, results)
            self.buffer = ""
        if self.pending_newline:
            self.pending_newline = False
            results.append((ContentType.CONTENT, "\n"))
        return results


class ThinkingRenderer:
    """Applies the show_thinking policy (off/terse/full) to a token stream."""

    def __init__(self, show_thinking: str) -> None:
        self.show_thinking = show_thinking
        self._terse_shown = False
        self._had_thinking = False
        self._content_started = False

    def feed(self, ctype: ContentType, text: str) -> List[Tuple[ContentType, str]]:
        """Return the pieces to render for one classified token."""
        if ctype != ContentType.THINKING:
            pieces: List[Tuple[ContentType, str]] = []
            self._terse_shown = False
            if self._had_thinking:
                self._had_thinking = False
                if not self._content_started:
                    pieces.append((ContentType.CONTENT, "\n"))
            self._content_started = True
            pieces.append((ctype, text))
            return pieces
        if self.show_thinking == "off":
            return []
        if self.show_thinking == "terse":
            if self._terse_shown:
                return []
            if "\n" in text:
                text = text.split("\n")[0]
                self._terse_shown = True
        self._had_thinking = True
        return [(ctype, text)]
