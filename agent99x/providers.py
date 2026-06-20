"""Provider definitions: each backend is a concrete subclass of `Provider`
carrying its own quirks (base URL, API key, default call budget, context-window
probe).

agent99x ships two providers: ollama (local default) and openrouter (remote).
"""

import math
import os
import urllib.parse
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

from agent99x import logs
from agent99x.session import SessionConfig


# ── base ───────────────────────────────────────────────────────────

class Provider:
    """A model backend. Subclasses set class attributes and may override
    `fetch_context_window`. Instances are immutable singletons in PROVIDERS."""

    name: str = ""
    base_url: str = ""
    api_key: str = ""
    default_call_budget: float = 100

    def fetch_context_window(self, session: "SessionConfig") -> int:
        """Return the active model's context-window size, or 0 if unknown."""
        return 0


def _apply_host(base_url: str, host: Optional[str]) -> str:
    """Substitute `host` for the netloc of `base_url`. Returns base_url unchanged if host is falsy."""
    if not host:
        return base_url
    parsed = urllib.parse.urlparse(base_url)
    default_port = 443 if parsed.scheme == "https" else 80
    port = parsed.port or default_port
    return parsed._replace(netloc=host if ":" in host else f"{host}:{port}").geturl()


# ── concrete providers ─────────────────────────────────────────────

class OllamaProvider(Provider):
    name = "ollama"
    base_url = "http://localhost:11434/v1"
    api_key = "ollama"
    default_call_budget = math.inf  # local inference is free

    def fetch_context_window(self, session):
        base = session.base_url.rsplit("/v1", 1)[0]
        with httpx.Client(timeout=session.http_timeout) as c:
            data = c.post(f"{base}/api/show", json={"model": session.model}).json()
        for k, v in data.get("model_info", {}).items():
            if k.endswith(".context_length"):
                return int(v)
        return 0


class OpenRouterProvider(Provider):
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    def fetch_context_window(self, session):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=session.http_timeout) as c:
            data = c.get(f"{session.base_url}/models", headers=headers).json()
        for m in data.get("data", []):
            if m["id"] == session.model:
                return int(m.get("context_length", 0))
        return 0


PROVIDERS: Dict[str, Provider] = {
    p.name: p for p in (
        OllamaProvider(),
        OpenRouterProvider(),
    )
}


# ── module-level helpers ───────────────────────────────────────────

def list_models(session: SessionConfig) -> List[str]:
    """Return sorted list of available model IDs from the active session client."""
    if not session.client:
        return []
    try:
        return sorted(m.id for m in session.client.models.list())
    except Exception as e:
        logs.log_warning(f"list_models failed: {e}")
        return []


def list_models_for(provider: str, host: Optional[str] = None, timeout: float = 30.0) -> List[str]:
    """Return sorted model IDs for a named provider without touching session state."""
    p = PROVIDERS.get(provider)
    if p is None:
        return []
    client = OpenAI(
        base_url=_apply_host(p.base_url, host),
        api_key=p.api_key,
        timeout=httpx.Timeout(timeout, connect=min(timeout, 30.0)),
        max_retries=0,
    )
    try:
        return sorted(m.id for m in client.models.list())
    except Exception as e:
        logs.log_warning(f"list_models_for({provider}, host={host}) failed: {e}")
        return []


def fetch_context_window(session: SessionConfig) -> int:
    """Try to determine the context-window size for the active provider/model."""
    if not session.provider:
        return 0
    try:
        return session.provider.fetch_context_window(session)
    except Exception as e:
        logs.log_warning(f"could not fetch context window: {e}")
        return 0


def ensure_context_window(session: SessionConfig) -> int:
    """Ensure session.context_window is populated, fetching it if necessary."""
    if session.context_fetched:
        return session.context_window
    if session.context_window > 0:
        session.context_fetched = True
        return session.context_window
    session.context_window = fetch_context_window(session)
    session.context_fetched = True
    return session.context_window


def setup_provider(session: SessionConfig, provider: str, model: Optional[str] = None, host: Optional[str] = None) -> None:
    """Configure the session provider, model, and client."""
    if provider not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"unknown provider {provider!r}; known: {known}")
    session.provider = PROVIDERS[provider]
    if host:
        session.host = host
    session.update_client()
    if model:
        session.model = model
        return
    available = list_models(session)
    if len(available) == 1:
        session.model = available[0]
    elif not available:
        raise ValueError(f"no models available from provider {provider!r}")
    else:
        raise ValueError(
            f"provider {provider!r} has {len(available)} models; "
            f"specify one with --model. Available: {', '.join(available)}"
        )
