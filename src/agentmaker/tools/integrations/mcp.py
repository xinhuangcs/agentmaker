"""agentmaker.tools.integrations.mcp: adapt an MCP server's tools into this framework's Tool (a thin adapter).

MCP (Model Context Protocol, Anthropic's open standard, the "USB-C for AI"): someone writes a server to the MCP
standard (exposing a set of tools); we connect with the official mcp SDK, list its tools, adapt each into this
framework's Tool, register them in ToolRegistry, and the agent calls them like ordinary tools, the same idea as
LangChain's load_mcp_tools. We do not roll our own protocol, only calling the official SDK's public API.

MCP calls are inherently async, so MCPTool overrides arun directly (native async) and does not implement a
synchronous run (calling it raises per the base-class contract, guiding the caller to arun). mcp is an optional
dependency (uv add mcp), imported lazily: if you do not use MCP it is not installed and does not affect the rest
of the framework.

Two transports are supported, pick one (the rest, handshake / listing tools / calling / security boundary, is fully shared):
    - stdio (a local server as a subprocess):
        async with MCPClient(command="python", args=["my_server.py"], namespace="calc") as client:
            tools = await client.load_tools()      # [MCPTool, ...], one per server tool
            for t in tools: registry.register(t)
            ...use inside the async with block (call tools while the connection is alive)...
    - Streamable HTTP (a remote / SaaS-hosted server):
        async with MCPClient(url="https://mcp.example.com/mcp", namespace="calendar", auth=my_oauth) as client:
            ...same as above...
      OAuth: pass an httpx.Auth (such as the mcp SDK's OAuthClientProvider) via auth=; the framework only forwards
      it. Token acquisition / refresh / persistence all live in the app (mechanism belongs to agentmaker,
      credential policy belongs to the app). The SSE transport is deprecated and not supported.
"""

import asyncio
import hashlib
import json
import re
from typing import List, Optional

from ...core.exceptions import ToolError
from ...prompts import DEFAULT_PROMPTS
from ..base import Tool, ToolParameter
from ..registry import sanitize_tool_name
from ..response import ToolResponse

# Sanitizing text from external sources: strip control characters (keep \t \n \r) to prevent MCP descriptions
# from hiding control characters or overwhelming the context with overly long descriptions.
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize_text(text: str, limit: int) -> str:
    """Sanitize text from an external source: strip control characters (keep \\t \\n \\r) and truncate if too long. Note that truncation only prevents context flooding, not the poisoning itself."""
    cleaned = _CTRL_CHARS.sub("", text or "")
    return cleaned[:limit] + "…(truncated)" if len(cleaned) > limit else cleaned


def _fingerprint(remote_name: str, description: str, input_schema: dict) -> str:
    """Compute a sha256 over (remote name + description + canonicalized inputSchema) as the tool-definition fingerprint (tool pinning):

    if the server later swaps out the description / schema (a rug-pull), the fingerprint changes and can be caught by an app that pins it.
    """
    canonical = json.dumps(input_schema or {}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(f"{remote_name}\x00{description}\x00{canonical}".encode("utf-8")).hexdigest()


def _dedupe_name(name: str, taken: set) -> str:
    """If the display name is already taken, add a _2/_3... suffix (keeping length <=64) to avoid collisions after sanitize normalization."""
    if name not in taken:
        return name
    i = 2
    while True:
        suffix = f"_{i}"
        candidate = name[:64 - len(suffix)] + suffix
        if candidate not in taken:
            return candidate
        i += 1


class MCPClient:
    """Manage a connection to one MCP server (local stdio subprocess or remote Streamable HTTP) and load its tools as MCPTool.

    Use async with to manage the connection lifecycle (on entry, start the subprocess / open the HTTP session and
    handshake; on exit, clean up). Tools must be called while the async with block is alive (they cannot be called
    after the connection closes).
    """

    def __init__(self, command: Optional[str] = None, args: Optional[List[str]] = None, *,
                 namespace: str, env: Optional[dict] = None,
                 url: Optional[str] = None, headers: Optional[dict] = None, auth=None,
                 requires_confirmation: bool = True, max_desc_chars: int = 4096,
                 expected_fingerprints: Optional[dict] = None, timeout: Optional[float] = 30.0,
                 prompts=None):
        """Configure an MCP server connection over exactly one transport.

        Pick one transport: for a local subprocess pass command (plus optional args/env), for a remote server pass
        url (plus optional headers/auth); the two are mutually exclusive, and giving neither or both is an error.

        Args:
            command: [stdio] Command to start the server (e.g. "python" / "uvx" / "npx"). Mutually exclusive with url.
            args: [stdio] Command arguments (e.g. ["my_server.py"]).
            env: [stdio] Optional environment variables passed to the server subprocess.
            url: [HTTP] The remote server's Streamable HTTP endpoint (e.g. "https://mcp.example.com/mcp"). Mutually exclusive with command.
            headers: [HTTP] Optional extra request headers (e.g. a custom API key header).
            auth: [HTTP] Optional httpx.Auth (such as the mcp SDK's OAuthClientProvider); the framework only forwards it, with token acquisition/refresh/persistence all in the app.
            namespace: Tool namespace prefix (**required**, the trust root): the tool display name becomes
                "{namespace}_{original name}" and the origin is stamped "mcp:{namespace}". Required and specified
                explicitly by the integrator, **not derived from the server's self-reported serverInfo.name** (which
                is attacker-controlled, equivalent to letting a malicious server pick its own trusted prefix).
                Prevents collisions across servers (two servers both having search -> "calendar_search" /
                "web_search"); remote calls still use the original name. Use underscores, not dots (must match ^[a-zA-Z0-9_-]{1,64}$).
            requires_confirmation: Whether the loaded MCPTools require human confirmation by default; default **True**,
                since remote tools are untrusted and the default posture is even stricter than our own CLITool; only
                after the app has vetted the server should it explicitly pass False to downgrade. A remote HTTP server is only less trustworthy and even more deserving of default confirmation.
            max_desc_chars: Length limit for tool / parameter descriptions, truncated beyond it (to prevent overly long descriptions from flooding the context); default 4096.
            expected_fingerprints: Optional {display name: sha256} pinning list (tool pinning): if given, load_tools
                verifies each tool-definition fingerprint and raises ToolError refusing to load on mismatch (to prevent
                the server later swapping the description / schema, a rug-pull); if not given, the fingerprint is only
                exposed via MCPTool.fingerprint for the app to record and compare itself (mechanism in agentmaker, policy in the app).
            timeout: Timeout in seconds for the handshake (initialize) and tool calls (call_tool), default 30; a stuck
                server will not make the agent wait forever. For the HTTP transport it also serves as the per-request
                timeout (streamablehttp_client's timeout). Pass None to disable the timeout (wait forever, not recommended).
            prompts: Optional prompt registry (PromptRegistry); the text call_tool feeds back to the model (session error / no output / tool-error prefix) comes from it, defaulting to DEFAULT_PROMPTS.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive or None, got {timeout}")
        # Pick exactly one transport: command (stdio) or url (HTTP). Giving both or neither is a configuration error, fail loud.
        if (command is None) == (url is None):
            raise ValueError("MCPClient transport must be exactly one: local uses command=..., remote uses url=... (not both, not neither)")
        if url is not None and (args or env is not None):
            raise ValueError("Remote HTTP mode (url) does not accept args/env, those are stdio subprocess parameters; remote uses headers/auth")
        if command is not None and (headers is not None or auth is not None):
            raise ValueError("stdio mode (command) does not accept headers/auth, those are remote HTTP transport parameters; local uses env")
        self.command = command
        self.args = args or []
        self.env = env
        self.url = url
        self.headers = headers
        self.auth = auth
        self.namespace = namespace
        self.requires_confirmation = requires_confirmation
        self.max_desc_chars = max_desc_chars
        self.expected_fingerprints = expected_fingerprints
        self.timeout = timeout
        self.prompts = prompts or DEFAULT_PROMPTS   # Text call_tool feeds back to the model comes from the registry (whole language swappable).
        self._session = None
        self._stack = None

    async def __aenter__(self):
        """Establish the connection (stdio subprocess or remote HTTP), create the ClientSession, and handshake (initialize)."""
        try:
            from contextlib import AsyncExitStack

            from mcp import ClientSession
        except ImportError as e:
            raise ToolError("Using MCP requires installing first: uv add mcp") from e
        self._stack = AsyncExitStack()
        try:
            read, write = await self._open_transport()   # Branch by command / url; the two transports are fully shared afterward.
            self._session = await self._stack.enter_async_context(ClientSession(read, write))
            try:
                await asyncio.wait_for(self._session.initialize(), timeout=self.timeout)
            except asyncio.TimeoutError as e:
                raise ToolError(f"MCP server handshake timed out (>{self.timeout}s): {self.url or self.command}") from e
        except BaseException:            # If initialization fails partway, clean up the started subprocess / HTTP session to avoid leaks.
            await self._stack.aclose()
            self._stack = None
            self._session = None
            raise
        return self

    async def _open_transport(self):
        """Establish the underlying (read, write) streams per configuration: with a url, use remote Streamable HTTP, otherwise a local stdio subprocess.

        After the handshake, both transports expose an identically shaped (read, write) consumed by the same
        ClientSession, so the upper layer has zero branching.
        """
        if self.url is not None:
            from mcp.client.streamable_http import streamablehttp_client
            kwargs = {"url": self.url, "headers": self.headers, "auth": self.auth}
            if self.timeout is not None:            # streamablehttp_client's timeout takes float seconds (when None, use the SDK default rather than forcing None).
                kwargs["timeout"] = self.timeout
            # streamablehttp_client yields a triple (read, write, get_session_id); take only the first two to align with stdio.
            read, write, _get_session_id = await self._stack.enter_async_context(streamablehttp_client(**kwargs))
            return read, write
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(command=self.command, args=self.args, env=self.env)
        read, write = await self._stack.enter_async_context(stdio_client(params))
        return read, write

    async def __aexit__(self, *exc):
        """Close the session and underlying transport (stdio subprocess / remote HTTP connection, all cleaned up uniformly by AsyncExitStack)."""
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None

    async def load_tools(self) -> List["MCPTool"]:
        """List the tools the server exposes, adapting each into an MCPTool (sharing this connection).

        The display name is prefixed "{namespace}_{original name}" to prevent collisions across servers, then run
        through sanitize_tool_name to normalize it to the character set function calling allows, deduplicating with
        _2/_3 on collision; remote calls always use the original name. Each tool description is sanitized (strip
        control characters plus truncate), the origin is stamped "mcp:{namespace}", and requires_confirmation is on
        by default; a tool-definition fingerprint is computed, and if expected_fingerprints was passed it is
        verified with a mismatch refusing to load. Raises ToolError if the session is not established or is closed.
        """
        if self._session is None:
            raise ToolError("MCP session not established or already closed: load_tools() must be called while the async with MCPClient(...) block is alive")
        resp = await self._session.list_tools()
        tools = []
        taken = set()
        for t in resp.tools:
            display = _dedupe_name(sanitize_tool_name(f"{self.namespace}_{t.name}"), taken)   # Normalize to a valid name and deduplicate.
            taken.add(display)
            raw_desc = t.description or ""
            schema = getattr(t, "inputSchema", None) or {}
            fp = _fingerprint(t.name, raw_desc, schema)
            if self.expected_fingerprints is not None and self.expected_fingerprints.get(display) != fp:
                raise ToolError(
                    f"MCP tool '{display}' fingerprint mismatch (expected {self.expected_fingerprints.get(display)}, actual {fp}): "
                    "the tool definition may have been tampered with by the server (rug-pull), refused to load")
            tools.append(MCPTool(self, display, _sanitize_text(raw_desc, self.max_desc_chars), schema,
                                 remote_name=t.name, origin=f"mcp:{self.namespace}",
                                 requires_confirmation=self.requires_confirmation,
                                 fingerprint=fp, max_desc_chars=self.max_desc_chars))
        return tools

    async def call_tool(self, name: str, arguments: dict) -> ToolResponse:
        """Call one of the server's tools and return a ToolResponse.

        text concatenates the text blocks in content (for the model to read); data preserves the **complete raw
        result**, all content blocks (including non-text image / audio / resource) and newer MCP's
        structuredContent, for programmatic use without losing information. A tool error (isError) yields
        status="error"; a connection not established / already closed yields a clear error (not a bare AttributeError).
        """
        if self._session is None:
            return ToolResponse.error(self.prompts.text("tool.msg.mcp.no_session"))
        try:
            result = await asyncio.wait_for(self._session.call_tool(name, arguments), timeout=self.timeout)
        except asyncio.TimeoutError:
            return ToolResponse.error(self.prompts.render("tool.msg.mcp.timeout", timeout=self.timeout))
        content = getattr(result, "content", None) or []
        text = "\n".join(t for t in (getattr(c, "text", "") for c in content) if t) or self.prompts.text("tool.msg.mcp.no_text")
        data = {
            "content": [c.model_dump(mode="json") if hasattr(c, "model_dump") else str(c) for c in content],
            "structured": getattr(result, "structuredContent", None),
        }
        if getattr(result, "isError", False):
            return ToolResponse.error(self.prompts.render("tool.msg.mcp.error", text=text), data=data)
        return ToolResponse.ok(text, data=data)


class MCPTool(Tool):
    """Adapt a single MCP server tool: present it as this framework's Tool (native async, overriding arun)."""

    external_content = True   # The result is external content returned by a third-party server: wrap it in an injection-defense delimiting guardrail before feeding back to the model (OWASP LLM01).

    def __init__(self, client: "MCPClient", name: str, description: str, input_schema: dict,
                 *, remote_name: Optional[str] = None, origin: str = "mcp",
                 requires_confirmation: bool = True, fingerprint: Optional[str] = None,
                 max_desc_chars: int = 4096):
        """Build an MCPTool wrapping one server tool.

        Args:
            client: The owning MCPClient (which provides call_tool).
            name: Tool display name (used by the registry / agent, with the namespace prefix).
            description: Tool description (from the server, already sanitized, for the LLM to decide when to call).
            input_schema: The tool's JSON Schema (from the server's inputSchema), converted into ToolParameter.
            remote_name: The real tool name on the server side (used when calling); defaults to name.
            origin: Origin marker, stamped "mcp:{namespace}" by MCPClient (the trust root, not overridable by the tool definition).
            requires_confirmation: Whether human confirmation is required (instance attribute shadowing the class attribute); default True, since untrusted remote tools require confirmation by default.
            fingerprint: Tool-definition fingerprint (sha256 of remote_name + description + schema), for the app to pin and compare (tool pinning).
            max_desc_chars: Length limit for parameter-description sanitization (same source as client).
        """
        super().__init__(name=name, description=description, origin=origin)
        self.requires_confirmation = requires_confirmation   # Instance attribute shadows the class attribute: one setting per server.
        self.fingerprint = fingerprint
        self._client = client
        self._input_schema = input_schema
        self._remote_name = remote_name or name
        self._max_desc_chars = max_desc_chars

    async def arun(self, parameters: dict) -> ToolResponse:
        """Run natively async: forward the call to the MCP server using the remote real name (name may carry a namespace prefix and cannot be used)."""
        return await self._client.call_tool(self._remote_name, parameters)

    def get_parameters(self) -> List[ToolParameter]:
        """Convert MCP's inputSchema (JSON Schema) into this framework's ToolParameter; sub-schema descriptions are
        likewise sanitized (strip control characters plus truncate, to prevent injection text / control characters
        hidden in nested descriptions), with the remaining schema fields carried over faithfully."""
        props = (self._input_schema or {}).get("properties", {}) or {}
        required = set((self._input_schema or {}).get("required", []) or [])
        out = []
        for k, v in props.items():
            v = dict(v)                                      # Copy before sanitizing, do not alter the original schema.
            if isinstance(v.get("description"), str):
                v["description"] = _sanitize_text(v["description"], self._max_desc_chars)
            out.append(ToolParameter(name=k, type=v.get("type", "string"),
                                     description=v.get("description", ""), required=k in required, schema=v))
        return out


