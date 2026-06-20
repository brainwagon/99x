"""MCP (Model Context Protocol) stdio client — optional, gated behind a config file.

Only imported and started when an ``mcp-config.json`` exists (see cli._mcp_start).
Requires the optional ``mcp`` dependency: ``pip install -e .[mcp]``.
"""

import json
import os
import asyncio
import threading
from datetime import timedelta
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


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
    """Run a coroutine in the dedicated MCP loop and wait for the result."""
    loop = get_mcp_loop()
    try:
        current_loop = asyncio.get_running_loop()
        if current_loop == loop:
            return coro
    except RuntimeError:
        pass
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
        if not self.session:
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


def mcp_to_openai_tool(mcp_tool: Any, server_name: str) -> Dict[str, Any]:
    """Convert an MCP tool definition to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": f"mcp__{server_name}__{mcp_tool.name}",
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema,
        },
    }


# ── Lifecycle ───────────────────────────────────────────────────────

def initialize_mcp(session: Any):
    """Initialize MCP servers from configuration and register their tools."""
    from agent99x import tools

    loop = get_mcp_loop()
    session.mcp_loop = loop

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

            def make_sync_handler(c: MCPClient, tool_name: str):
                def sync_handler(**kwargs):
                    async def run_and_format():
                        result = await c.call_tool(tool_name, kwargs)
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

            tools.register(openai_tool, make_sync_handler(client, t.name))

    for name, server_cfg in servers.items():
        try:
            future = asyncio.run_coroutine_threadsafe(_init_server(name, server_cfg), loop)
            future.result()
        except Exception as e:
            print(f"Failed to initialize MCP server '{name}': {e}")


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
            for client in session.mcp_clients:
                try:
                    result = await client.read_resource(uri)
                    texts = []
                    for item in result.contents:
                        if hasattr(item, "text"):
                            texts.append(item.text)
                        elif hasattr(item, "blob"):
                            texts.append(f"[Binary Data: {item.mimeType}]")
                    return "\n".join(texts)
                except Exception:
                    continue
            return {"error": f"Resource with URI '{uri}' not found or could not be read."}
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
                                for a in getattr(p, "arguments", [])
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

    def get_prompt_handler(name: str, arguments: Dict[str, Any] = {}):
        async def run():
            for client in session.mcp_clients:
                try:
                    result = await client.get_prompt(name, arguments)
                    messages = []
                    for msg in result.messages:
                        messages.append(f"[{msg.role}] {msg.content.text}")
                    return "\n".join(messages)
                except Exception:
                    continue
            return {"error": f"Prompt '{name}' not found or could not be retrieved."}
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
