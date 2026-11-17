import re
from enum import Enum
from typing import List, Tuple

_WORD_CHAR = re.compile(r'\w')


class ContentType(Enum):
    CONTENT = "content"
    THINKING = "thinking"


class ReasoningParser:
    """Parses streaming LLM output to separate thinking/reasoning from content."""

    # (start, end) pairs that wrap inline reasoning. Most backends strip
    # these server-side; this is the fallback for templates that leak them.
    # The second pair is Gemma 4's channel format: <|channel>thought\n...<channel|>
    THINK_TAG_PAIRS = [
        ("<think>", "</think>"),
        ("<|channel>thought", "<channel|>"),
    ]

    def __init__(self, strip_midword_newlines: bool = False):
        self.in_think_tag = False
        self.think_end_tag = ""  # end tag of the pair we're inside
        self.strip_thinking_lead_nl = False  # swallow the \n after <|channel>thought
        self.buffer = ""
        self._max_tag_len = max(len(t) for pair in self.THINK_TAG_PAIRS for t in pair)
        # Gemma-3-family tokenizer artifact workaround; corrupts legitimate
        # single newlines (e.g. numbered lists) on models that don't need it.
        self.strip_midword_newlines = strip_midword_newlines

        # Generic "Thought:"/"Reasoning:" preamble: recognised only at the
        # very start of a response, and ended by a blank line or an
        # "Answer:"/"Final Answer:" line so the answer is never hidden.
        self.generic_patterns = ["Thought:", "Reasoning:"]
        self.generic_end_markers = ["\n\n", "\nFinal Answer:", "\nAnswer:"]
        self.in_generic_reasoning = False
        self.generic_start_checked = False
        self.pending_newline = False
        self.last_was_word = False

    def _process_content(self, text: str) -> str:
        """Strip mid-word newlines from streamed content (a Gemma 3 tokenizer artifact)."""
        if not self.strip_midword_newlines or not text:
            return text

        out = []

        if self.pending_newline:
            self.pending_newline = False
            if self.last_was_word and _WORD_CHAR.match(text[0]):
                pass  # mid-word — drop
            else:
                out.append("\n")
                self.last_was_word = False

        # A chunk can also *start* with the artifact newline (the previous
        # chunk ended mid-word); drop it when both neighbours are word chars.
        if (text.startswith("\n") and not text.startswith("\n\n")
                and self.last_was_word
                and len(text) > 1 and _WORD_CHAR.match(text[1])):
            text = text[1:]

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
        elif self.strip_thinking_lead_nl:
            self.strip_thinking_lead_nl = False
            text = text[1:] if text.startswith("\n") else text
        if text:
            results.append((ctype, text))

    def _check_generic_start(self) -> bool:
        """Decide whether the response opens with a generic reasoning label.

        Returns False while the buffer is still too short to tell.
        """
        head = self.buffer.lstrip(" \t\r\n*_")
        if not head:
            return False
        for p in self.generic_patterns:
            if head.startswith(p):
                self.in_generic_reasoning = True
                break
            if p.startswith(head):
                return False  # could still become this label
        self.generic_start_checked = True
        return True

    def feed_content(self, chunk: str) -> List[Tuple[ContentType, str]]:
        """Feed a chunk of regular content (which might contain think tags)."""
        results = []
        self.buffer += chunk

        while self.buffer:
            if not self.in_think_tag:
                if not self.generic_start_checked and not self._check_generic_start():
                    break
                ctype = ContentType.THINKING if self.in_generic_reasoning else ContentType.CONTENT

                # Earliest of: a think start tag, or (in a generic preamble)
                # the marker that ends it.
                best_idx, best_tag, best_end = -1, None, None
                for start_tag, end_tag in self.THINK_TAG_PAIRS:
                    idx = self.buffer.find(start_tag)
                    if idx != -1 and (best_idx == -1 or idx < best_idx):
                        best_idx, best_tag, best_end = idx, (start_tag, end_tag), None
                if self.in_generic_reasoning:
                    for marker in self.generic_end_markers:
                        idx = self.buffer.find(marker)
                        if idx != -1 and (best_idx == -1 or idx < best_idx):
                            best_idx, best_tag, best_end = idx, None, marker

                if best_tag:
                    if best_idx > 0:
                        self._emit(ctype, self.buffer[:best_idx], results)
                    self.in_think_tag = True
                    self.think_end_tag = best_tag[1]
                    self.strip_thinking_lead_nl = best_tag[0] == "<|channel>thought"
                    self.buffer = self.buffer[best_idx + len(best_tag[0]):]
                elif best_end:
                    if best_idx > 0:
                        self._emit(ContentType.THINKING, self.buffer[:best_idx], results)
                    self.in_generic_reasoning = False
                    rest = self.buffer[best_idx + len(best_end):]
                    self.buffer = rest.lstrip(" ") if best_end != "\n\n" else rest.lstrip("\n")
                else:
                    safe_len = self._max_tag_len - 1
                    if self.in_generic_reasoning:
                        safe_len = max(safe_len, max(len(m) for m in self.generic_end_markers) - 1)
                    if len(self.buffer) > safe_len:
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
