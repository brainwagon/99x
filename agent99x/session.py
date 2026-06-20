"""Session and configuration state management for agent99x."""

import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import httpx
from openai import OpenAI

if TYPE_CHECKING:
    from agent99x.providers import Provider

# ── constants ──────────────────────────────────────────────────────

AGENT_HOME: str = os.path.expanduser(os.environ.get("AGENT_HOME", "~/.99x"))
PROJECT_DIR: str = ".99x"
SESSION_FILE: str = os.path.join(PROJECT_DIR, "session.json")
CONTEXTS_DIR: str = os.path.join(PROJECT_DIR, "contexts")
CONFIG_FILE: str = os.path.join(AGENT_HOME, "config.json")

# ── models ─────────────────────────────────────────────────────────

@dataclass
class SessionConfig:
    """Encapsulates the configuration and state of an agent session."""
    # Provider is set via providers.setup_provider; None until configured.
    provider: Optional["Provider"] = None
    model: Optional[str] = None
    # Host override applied on top of provider.base_url when building the client.
    host: Optional[str] = None

    effort: Optional[str] = None  # None | "low" | "medium" | "high"
    effort_suppressed: bool = False  # True when the current model rejected reasoning_effort
    show_thinking: str = "full"  # "off" | "terse" | "full"
    autocompact_threshold: float = 0.85  # fraction of context_window
    call_budget: Optional[int] = None  # None = provider.default_call_budget

    context_window: int = 0
    model_timeout: float = 120.0
    http_timeout: float = 5.0

    plan_mode: bool = False

    # When set, restrict callable tools to this whitelist (per-agent capability gate).
    allowed_tools: Optional[List[str]] = None

    context_fetched: bool = False  # Track if lazy fetch was attempted

    done_grace: float = 10.0  # seconds to show finished agents in TUI

    # Shared state across subagent trees (e.g., call budget)
    shared_state: Dict[str, Any] = field(default_factory=dict, repr=False)

    client: Optional[OpenAI] = field(default=None, repr=False)

    # Last model used per provider, persisted in config.json.
    provider_models: Dict[str, str] = field(default_factory=dict)

    # Last host override used per provider (None = provider default base_url).
    provider_hosts: Dict[str, Optional[str]] = field(default_factory=dict)

    # Transient: when set, the next user input is routed here instead of being
    # dispatched as a slash command or sent to the model. Used for multi-step
    # prompts like /provider's model picker.
    pending_input_handler: Optional[Callable[["SessionConfig", str], Any]] = field(
        default=None, repr=False
    )

    # Message history
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Path to the context file this run reads from and persists to on exit.
    # None means no context selected yet (fall back to the legacy SESSION_FILE).
    context_path: Optional[str] = None

    # Active MCP clients (only populated when mcp-config.json exists)
    mcp_clients: List[Any] = field(default_factory=list, repr=False)
    mcp_loop: Optional[Any] = field(default=None, repr=False)

    # Locks for thread-safety
    session_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def base_url(self) -> str:
        """Effective base URL: provider.base_url with host override applied."""
        if self.provider is None:
            return ""
        from agent99x.providers import _apply_host
        return _apply_host(self.provider.base_url, self.host)

    @property
    def api_key(self) -> str:
        return self.provider.api_key if self.provider else ""

    def effective_call_budget(self) -> float:
        """The call budget to enforce: explicit override, else provider default."""
        if self.call_budget is not None:
            return self.call_budget
        return self.provider.default_call_budget if self.provider else 100

    def agent_path(self, *parts: str) -> str:
        """Return a path under AGENT_HOME."""
        return os.path.join(AGENT_HOME, *parts)

    def update_client(self):
        """Re-initialize the OpenAI client with current config."""
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(self.model_timeout, connect=30.0),
            # Surface connection failures immediately rather than retrying with
            # exponential backoff — a silent retry storm is what makes a
            # misconfigured/unreachable provider feel like an eternal hang.
            max_retries=0,
        )
