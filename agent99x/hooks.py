"""Opt-in, metadata-only observation hooks for agent runs."""

import inspect
import logging
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterable, Literal, Mapping, Optional, Union


logger = logging.getLogger(__name__)

HookValue = Union[str, int, float, bool, None, tuple]
HookEventName = Literal[
    "run.started",
    "run.finished",
    "run.cancelled",
    "model_request.started",
    "model_request.finished",
    "tool.started",
    "tool.finished",
    "error",
]
HookHandler = Callable[["HookEvent"], None]


@dataclass(frozen=True)
class HookEvent:
    """An immutable metadata snapshot emitted during an agent run.

    ``metadata`` contains event-specific scalar values only; prompts,
    completions, tool arguments, and tool results are never included.
    Timestamps are seconds since the Unix epoch.
    """

    name: HookEventName
    run_id: str
    parent_run_id: Optional[str]
    timestamp: float
    metadata: Mapping[str, HookValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class AgentHooks:
    """A per-run collection of synchronous event handlers.

    Events are dispatched inline. A lock serializes delivery across parallel
    tool workers and nested agent runs. Handlers run in registration order;
    exceptions are logged and do not affect the agent or other handlers.
    """

    def __init__(self, handlers: Iterable[HookHandler] = ()) -> None:
        self._handlers = tuple(handlers)
        self._lock = threading.Lock()

    def emit(self, event: HookEvent) -> None:
        """Deliver an event to every registered handler."""
        with self._lock:
            for handler in self._handlers:
                handler_name = getattr(
                    handler, "__qualname__", type(handler).__qualname__
                )
                try:
                    result = handler(event)
                    if inspect.isawaitable(result):
                        close = getattr(result, "close", None)
                        if close is not None:
                            close()
                        logger.warning(
                            "hook handler %s returned an awaitable; handlers must "
                            "be synchronous for event %s",
                            handler_name,
                            event.name,
                        )
                except Exception as exc:
                    logger.warning(
                        "hook handler %s raised %s for event %s",
                        handler_name,
                        type(exc).__name__,
                        event.name,
                    )
