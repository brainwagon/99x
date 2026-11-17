"""MCP (Model Context Protocol) stdio client — optional, gated behind a config file.

Only imported and started when an ``mcp-config.json`` exists (see cli._mcp_start).
Requires the optional ``mcp`` dependency: ``pip install -e .[mcp]``.
"""

import json
import os
import asyncio
import hashlib
import re
import threading
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED


class MCPConfigError(Exception):
    """Raised when MCP configuration is invalid or missing required fields."""
    pass


# ── Background Loop Management ──────────────────────────────────────

_mcp_loop: Optional[asyncio.AbstractEventLoop] = None
_mcp_thread: Optional[threading.Thread] = None


def get_mcp_loop() -> asyncio.AbstractEventLoop:
    """Get or create a dedicated background event loop for MCP operations."""
    global _mcp_loop, _mcp_thread
    if _mcp_loop is None:
        _mcp_loop = asyncio.new_event_loop()

        def run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        _mcp_thread = threading.Thread(target=run_loop, args=(_mcp_loop,), daemon=True, name="MCPEventLoop")
        _mcp_thread.start()
    return _mcp_loop


def run_in_mcp_loop(coro):
    """Run a coroutine in the dedicated MCP loop and wait for the result.

    Must be called from outside the MCP loop thread: blocking that thread on
    its own future would deadlock, so code already on the loop must ``await``
    the coroutine instead.
    """
    loop = get_mcp_loop()
    try:
        on_loop = asyncio.get_running_loop() is loop
    except RuntimeError:
        on_loop = False
    if on_loop or threading.current_thread() is _mcp_thread:
        coro.close()
        raise RuntimeError("run_in_mcp_loop() called from the MCP event loop thread; "
                           "await the coroutine directly instead")
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


# ── Config and Client ───────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    """Load and validate MCP configuration from a JSON file."""
    if not os.path.exists(path):
        return {"mcpServers": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise MCPConfigError(f"Invalid JSON in MCP config: {e}")
    except OSError as e:
        raise MCPConfigError(f"Could not read MCP config: {e}")

    if not isinstance(config, dict):
        raise MCPConfigError("MCP config must be a JSON object")

    servers = config.get("mcpServers", {})
    if not isinstance(servers, dict):
        raise MCPConfigError("'mcpServers' must be a JSON object")

    for name, server_cfg in servers.items():
        if not isinstance(server_cfg, dict):
            raise MCPConfigError(f"Configuration for server '{name}' must be a JSON object")
        if "command" not in server_cfg:
            raise MCPConfigError(f"Missing 'command' for MCP server '{name}'")

    return config


class MCPClient:
    """A client for communicating with an MCP server via stdio transport."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.server_params = StdioServerParameters(
            command=config["command"],
            args=config.get("args", []),
            env={**os.environ, **config.get("env", {})},
        )
        self.session: Optional[ClientSession] = None
        self.initialize_result: Optional[Any] = None
        self.timeout = config.get("timeout", 30)

        self._lifecycle_task: Optional[asyncio.Task] = None
        self._exit_event: Optional[asyncio.Event] = None
        self._init_done_event: Optional[asyncio.Event] = None
        self._init_error: Optional[Exception] = None
        self.disconnect_reason: Optional[str] = None

    @property
    def connected(self) -> bool:
        return self.session is not None and self.disconnect_reason is None

    def mark_disconnected(self, reason: str) -> None:
        """Record that the server is gone and tear down the session.

        Called on the MCP loop. Later calls fail fast with a clear message
        instead of waiting on a dead transport.
        """
        if self.disconnect_reason is None:
            self.disconnect_reason = reason
        if self._exit_event is not None:
            self._exit_event.set()

    async def start(self):
        self._init_done_event = asyncio.Event()
        self._exit_event = asyncio.Event()
        self._lifecycle_task = asyncio.create_task(self._run_lifecycle())
        await self._init_done_event.wait()
        if self._init_error:
            raise self._init_error

    async def _run_lifecycle(self):
        try:
            async with stdio_client(self.server_params) as (read, write):
                self.session = ClientSession(
                    read, write, read_timeout_seconds=timedelta(seconds=self.timeout))
                async with self.session:
                    self.initialize_result = await self.session.initialize()
                    self._init_done_event.set()
                    keep_alive_task = asyncio.create_task(self._keep_alive_loop())
                    await self._exit_event.wait()
                    keep_alive_task.cancel()
                    try:
                        await keep_alive_task
                    except asyncio.CancelledError:
                        pass
        except Exception as e:
            if self._init_done_event and not self._init_done_event.is_set():
                self._init_error = e
                self._init_done_event.set()
            else:
                if self._exit_event and not self._exit_event.is_set():
                    print(f"MCP Server '{self.name}' encountered an error: {e}")
                if self.disconnect_reason is None:
                    self.disconnect_reason = str(e) or type(e).__name__
        finally:
            self.session = None
            if self._init_done_event and not self._init_done_event.is_set():
                self._init_done_event.set()

    async def stop(self):
        if self._exit_event:
            self._exit_event.set()
        if self._lifecycle_task:
            try:
                await asyncio.wait_for(self._lifecycle_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._lifecycle_task.cancel()

    async def _keep_alive_loop(self):
        try:
            while self.session:
                await asyncio.sleep(30)
                await self.ping()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Warning: MCP keep-alive failed for '{self.name}': {e}")
            self.mark_disconnected(f"keep-alive ping failed: {e}")

    async def ping(self) -> None:
        if not self.session:
            return
        await self.session.send_ping()

    async def list_tools(self) -> List[Any]:
        if not self.session:
            return []
        result = await self.session.list_tools()
        return result.tools

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if not self.connected:
            raise RuntimeError(f"Client '{self.name}' is not connected")
        return await self.session.call_tool(name, arguments)

    async def list_resources(self) -> List[Any]:
        if not self.session:
            return []
        result = await self.session.list_resources()
        return result.resources

    async def read_resource(self, uri: str) -> Any:
        if not self.session:
            raise RuntimeError(f"Client '{self.name}' is not connected")
        return await self.session.read_resource(uri)

    async def list_prompts(self) -> List[Any]:
        if not self.session:
            return []
        result = await self.session.list_prompts()
        return result.prompts

    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Any:
        if not self.session:
            raise RuntimeError(f"Client '{self.name}' is not connected")
        return await self.session.get_prompt(name, arguments)


def is_connection_lost(exc: BaseException) -> bool:
    """True if ``exc`` means the server's transport is gone."""
    if isinstance(exc, (anyio.ClosedResourceError, anyio.BrokenResourceError,
                        anyio.EndOfStream, BrokenPipeError, ConnectionError)):
        return True
    return isinstance(exc, McpError) and exc.error.code == CONNECTION_CLOSED


def disconnected_error(client: "MCPClient") -> Dict[str, str]:
    reason = f" ({client.disconnect_reason})" if client.disconnect_reason else ""
    return {"error": (f"MCP server '{client.name}' is disconnected{reason}; its tools are "
                      "unavailable until the agent is restarted.")}


# OpenAI-compatible function names must match ^[a-zA-Z0-9_-]{1,64}$.
_TOOL_NAME_MAX = 64
_INVALID_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")

# Registered function name -> (server name, real MCP tool name).
MCP_TOOL_NAMES: Dict[str, Tuple[str, str]] = {}


def openai_tool_name(server_name: str, tool_name: str) -> str:
    """A valid, collision-free function name for an MCP tool.

    Invalid characters become '_'. Names that are too long, or that would
    collide with a different server/tool after sanitising, get a short hash
    of the real name as suffix. ``MCP_TOOL_NAMES`` maps back to the real name.
    """
    key = (server_name, tool_name)
    raw = f"mcp__{server_name}__{tool_name}"
    name = _INVALID_NAME_CHARS.sub("_", raw)
    if len(name) > _TOOL_NAME_MAX or MCP_TOOL_NAMES.get(name, key) != key:
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
        n = 0
        while True:
            suffix = "_" + (digest[:8] if n == 0 else f"{digest[:6]}{n}")
            name = _INVALID_NAME_CHARS.sub("_", raw)[:_TOOL_NAME_MAX - len(suffix)] + suffix
            if MCP_TOOL_NAMES.get(name, key) == key:
                break
            n += 1
    MCP_TOOL_NAMES[name] = key
    return name


def mcp_to_openai_tool(mcp_tool: Any, server_name: str,
                       name: Optional[str] = None) -> Dict[str, Any]:
    """Convert an MCP tool definition to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": name or openai_tool_name(server_name, mcp_tool.name),
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema,
        },
    }


def mcp_tool_traits(mcp_tool: Any) -> Any:
    """Registry traits for an MCP tool: plan-safe only when the server says
    the tool is read-only (annotations.readOnlyHint). It may mutate state we
    cannot see, but it has no file paths to track, so mutates stays False."""
    from agent99x.tools import ToolTraits
    annotations = getattr(mcp_tool, "annotations", None)
    read_only = bool(getattr(annotations, "readOnlyHint", False))
    return ToolTraits(plan_safe=read_only, mutates=False)


def make_tool_handler(client: "MCPClient", tool_name: str):
    """Synchronous registry handler that calls ``tool_name`` on ``client``."""
    def sync_handler(**kwargs):
        async def run_and_format():
            if not client.connected:
                return disconnected_error(client)
            try:
                result = await client.call_tool(tool_name, kwargs)
            except Exception as e:
                if is_connection_lost(e) or not client.connected:
                    client.mark_disconnected(str(e) or type(e).__name__)
                    return disconnected_error(client)
                return {"error": f"MCP tool '{tool_name}' on server '{client.name}' failed: {e}"}
            texts = []
            for item in result.content:
                if hasattr(item, "text"):
                    texts.append(item.text)
                elif hasattr(item, "data"):
                    texts.append(f"[Image Data: {item.mimeType}]")
                elif hasattr(item, "resource"):
                    texts.append(f"[Embedded Resource: {item.resource.uri}]")
            combined = "\n".join(texts)
            if result.isError:
                return {"error": combined}
            return combined
        return run_in_mcp_loop(run_and_format())
    return sync_handler


# ── Lifecycle ───────────────────────────────────────────────────────

def initialize_mcp(session: Any):
    """Initialize MCP servers from configuration and register their tools."""
    from agent99x import tools

    loop = get_mcp_loop()

    config_paths = ["mcp-config.json", session.agent_path("mcp-config.json")]
    config = {"mcpServers": {}}
    for path in config_paths:
        if os.path.exists(path):
            try:
                config = load_config(path)
                break
            except MCPConfigError as e:
                print(f"Warning: Failed to load MCP config at {path}: {e}")

    servers = config.get("mcpServers", {})
    _register_global_mcp_tools(session)
    if not servers:
        return

    async def _init_server(name, cfg):
        client = MCPClient(name, cfg)
        await client.start()
        session.mcp_clients.append(client)

        mcp_tools = await client.list_tools()
        for t in mcp_tools:
            openai_tool = mcp_to_openai_tool(t, name)
            tools.register(openai_tool, make_tool_handler(client, t.name),
                           traits=mcp_tool_traits(t))

    for name, server_cfg in servers.items():
        try:
            future = asyncio.run_coroutine_threadsafe(_init_server(name, server_cfg), loop)
            future.result()
        except Exception as e:
            print(f"Failed to initialize MCP server '{name}': {e}")


def _looks_not_found(exc: BaseException) -> bool:
    """Best-effort: does this server error mean "no such resource"?"""
    if isinstance(exc, McpError) and exc.error.code in (-32002, -32602):
        return True  # MCP's RESOURCE_NOT_FOUND, or the common INVALID_PARAMS
    text = str(exc).lower()
    return "not found" in text or "unknown resource" in text or "no such" in text


def _register_global_mcp_tools(session: Any):
    """Register global tools for interacting with MCP resources and prompts."""
    from agent99x import tools

    list_resources_schema = {
        "type": "function",
        "function": {
            "name": "mcp_list_resources",
            "description": "List all available resources from all connected MCP servers.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    def list_resources_handler():
        async def run():
            all_resources = []
            for client in session.mcp_clients:
                try:
                    resources = await client.list_resources()
                    for r in resources:
                        all_resources.append({"server": client.name, "uri": str(r.uri),
                                              "name": r.name, "description": getattr(r, "description", "")})
                except Exception as e:
                    all_resources.append({"server": client.name, "error": str(e)})
            return all_resources
        return run_in_mcp_loop(run())

    tools.register(list_resources_schema, list_resources_handler)

    read_resource_schema = {
        "type": "function",
        "function": {
            "name": "mcp_read_resource",
            "description": "Read the content of an MCP resource by URI.",
            "parameters": {
                "type": "object",
                "properties": {"uri": {"type": "string", "description": "The URI of the resource to read."}},
                "required": ["uri"],
            },
        },
    }

    def read_resource_handler(uri: str):
        async def run():
            clients = [c for c in session.mcp_clients if c.connected]
            if not clients:
                return {"error": "No connected MCP servers."}
            # Ask the server(s) that list this URI first; servers with
            # resource templates may still serve URIs they do not list.
            listed = []
            for client in clients:
                try:
                    if any(str(r.uri) == uri for r in await client.list_resources()):
                        listed.append(client)
                except Exception:
                    pass
            ordered = listed + [c for c in clients if c not in listed]
            failures = []
            for client in ordered:
                try:
                    result = await client.read_resource(uri)
                except Exception as e:
                    if is_connection_lost(e):
                        client.mark_disconnected(str(e) or type(e).__name__)
                    failures.append((client, e))
                    continue
                texts = []
                for item in result.contents:
                    if hasattr(item, "text"):
                        texts.append(item.text)
                    elif hasattr(item, "blob"):
                        texts.append(f"[Binary Data: {item.mimeType}]")
                return "\n".join(texts)
            if listed:
                # A server that lists the URI failed to read it: that error is
                # the real one, not "not found".
                errs = [f"{c.name}: {e}" for c, e in failures if c in listed]
                return {"error": f"Resource '{uri}' could not be read: " + "; ".join(errs)}
            if all(_looks_not_found(e) for _, e in failures):
                return {"error": f"Resource with URI '{uri}' not found on any connected MCP server."}
            errs = [f"{c.name}: {e}" for c, e in failures]
            return {"error": f"Resource '{uri}' could not be read: " + "; ".join(errs)}
        return run_in_mcp_loop(run())

    tools.register(read_resource_schema, read_resource_handler)

    list_prompts_schema = {
        "type": "function",
        "function": {
            "name": "mcp_list_prompts",
            "description": "List all available prompt templates from all connected MCP servers.",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    def list_prompts_handler():
        async def run():
            all_prompts = []
            for client in session.mcp_clients:
                try:
                    prompts = await client.list_prompts()
                    for p in prompts:
                        all_prompts.append({
                            "server": client.name, "name": p.name,
                            "description": getattr(p, "description", ""),
                            "arguments": [
                                {k: (str(v) if k == "uri" else v) for k, v in vars(a).items()}
                                for a in (getattr(p, "arguments", None) or [])
                            ],
                        })
                except Exception as e:
                    all_prompts.append({"server": client.name, "error": str(e)})
            return all_prompts
        return run_in_mcp_loop(run())

    tools.register(list_prompts_schema, list_prompts_handler)

    get_prompt_schema = {
        "type": "function",
        "function": {
            "name": "mcp_get_prompt",
            "description": "Get a prompt template by name from an MCP server.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name of the prompt template."},
                    "arguments": {"type": "object", "description": "Arguments for the prompt template."},
                },
                "required": ["name"],
            },
        },
    }

    def get_prompt_handler(name: str, arguments: Optional[Dict[str, Any]] = None):
        args = {k: str(v) for k, v in (arguments or {}).items()}

        async def run():
            clients = [c for c in session.mcp_clients if c.connected]
            if not clients:
                return {"error": "No connected MCP servers."}
            owner = None
            list_errors = []
            for client in clients:
                try:
                    if any(p.name == name for p in await client.list_prompts()):
                        owner = client
                        break
                except Exception as e:
                    list_errors.append(f"{client.name}: {e}")
            if owner is None:
                detail = f" (listing failed on {'; '.join(list_errors)})" if list_errors else ""
                return {"error": f"Prompt '{name}' not found on any connected MCP server{detail}."}
            try:
                result = await owner.get_prompt(name, args)
            except Exception as e:
                if is_connection_lost(e):
                    owner.mark_disconnected(str(e) or type(e).__name__)
                    return disconnected_error(owner)
                return {"error": f"Prompt '{name}' on server '{owner.name}' failed: {e}"}
            messages = []
            for msg in result.messages:
                content = msg.content
                text = getattr(content, "text", None)
                if text is None:
                    resource = getattr(content, "resource", None)
                    text = (getattr(resource, "text", None)
                            or f"[{getattr(content, 'type', 'content')}]")
                messages.append(f"[{msg.role}] {text}")
            return "\n".join(messages)
        return run_in_mcp_loop(run())

    tools.register(get_prompt_schema, get_prompt_handler)


def cleanup_mcp(session: Any):
    """Gracefully close all active MCP clients."""
    loop = get_mcp_loop()

    async def _cleanup():
        for client in session.mcp_clients:
            try:
                await client.stop()
            except Exception as e:
                print(f"Error closing MCP client '{client.name}': {e}")
        session.mcp_clients = []

    asyncio.run_coroutine_threadsafe(_cleanup(), loop).result()
