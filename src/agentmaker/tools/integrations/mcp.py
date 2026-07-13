"""agentmaker.tools.integrations.mcp: adapt an MCP server's tools into this framework's Tool (a thin adapter).

MCP (Model Context Protocol, Anthropic's open standard, the "USB-C for AI"): someone writes a server to the MCP
standard (exposing a set of tools); we connect with the official mcp SDK, list its tools, adapt each into this
framework's Tool, register them in ToolRegistry, and the agent calls them like ordinary tools, the same idea as
LangChain's load_mcp_tools. We do not roll our own protocol, only calling the official SDK's public API.

MCP calls are inherently async, so MCPTool overrides arun directly (native async) and does not implement a
synchronous run (calling it raises per the base-class contract, guiding the caller to arun). mcp is an optional
dependency (`uv add "agentmaker[mcp]"`), imported lazily: if you do not use MCP it is not installed and does not affect the rest
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
import base64
import hashlib
import ipaddress
import json
import math
import re
import unicodedata
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, List, Optional, cast
from urllib.parse import urlsplit

from ...core.exceptions import ToolError
from ...prompts import DEFAULT_PROMPTS
from ..base import Tool, ToolParameter
from ..registry import sanitize_tool_name
from ..response import ToolResponse

if TYPE_CHECKING:
    import httpx
    from ...prompts import PromptRegistry

# Sanitizing text from external sources prevents hidden controls and context flooding.
_MAX_SCHEMA_BYTES = 128 * 1024
_MAX_SCHEMA_DEPTH = 32
_MAX_SCHEMA_NODES = 4096
_MAX_SCHEMA_PROPERTIES = 512
_MAX_RESULT_BLOCKS = 512
_NAMESPACE_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class _PayloadTooLarge(Exception):
    """Internal signal for an invalid or oversized JSON payload."""


# ZWNJ / ZWJ / soft hyphen: legitimate in emoji sequences and Persian/Indic text.
_KEPT_FORMAT_CHARS = frozenset({"\u200c", "\u200d", "\u00ad"})


def _clean_text(text: str, limit: int) -> tuple[str, bool]:
    """Sanitize at most a bounded prefix of external text."""
    if not text:
        return "", False
    kept = []
    scan_limit = max(limit * 8 + 1024, 2048)
    truncated = False
    for index, char in enumerate(text):
        if index >= scan_limit:
            truncated = True
            break
        code = ord(char)
        if (code < 32 and char not in "\t\n\r") or (
                unicodedata.category(char) == "Cf" and char not in _KEPT_FORMAT_CHARS):
            continue
        if len(kept) >= limit:
            truncated = True
            break
        kept.append(char)
    return "".join(kept), truncated


def _sanitize_text(text: str, limit: int) -> str:
    """Sanitize text from an external source: strip control characters (keep \\t \\n \\r) and truncate if too long. Note that truncation only prevents context flooding, not the poisoning itself."""
    cleaned, truncated = _clean_text(text or "", limit)
    return cleaned + "…(truncated)" if truncated else cleaned


def _result_text(content, limit: int, empty: str, *, content_truncated: bool = False) -> str:
    """Collect bounded text blocks without constructing an unbounded joined string."""
    parts = []
    remaining = limit
    truncated = content_truncated
    for block in content:
        value = getattr(block, "text", "")
        if not isinstance(value, str) or not value:
            continue
        separator = 1 if parts else 0
        if separator >= remaining:
            truncated = True
            break
        value, value_truncated = _clean_text(value, remaining - separator)
        if value:
            remaining -= separator
            parts.append(value)
            remaining -= len(value)
        if value_truncated:
            truncated = True
            break
    if not parts:
        return "…(truncated)" if truncated else empty
    result = "\n".join(parts)
    return result + "…(truncated)" if truncated else result


def _bounded_json_copy(value: Any, budget: int) -> tuple[Any, int]:
    """Copy JSON-compatible data while enforcing byte, depth, and node limits."""
    remaining = budget
    nodes = 0
    active = set()

    def consume(size: int) -> None:
        nonlocal remaining
        if size > remaining:
            raise _PayloadTooLarge
        remaining -= size

    def consume_string(value: str) -> None:
        consume(2)
        for char in value:
            code = ord(char)
            if char in ('"', "\\"):
                consume(2)
            elif code < 32:
                consume(6)
            else:
                consume(len(char.encode("utf-8")))

    def visit(item: Any, depth: int) -> Any:
        nonlocal nodes
        if depth > _MAX_SCHEMA_DEPTH:
            raise _PayloadTooLarge
        nodes += 1
        if nodes > _MAX_SCHEMA_NODES:
            raise _PayloadTooLarge
        if item is None:
            consume(4)
            return None
        if item is True:
            consume(4)
            return True
        if item is False:
            consume(5)
            return False
        if isinstance(item, str):
            consume_string(item)
            return item
        if isinstance(item, bytes):
            encoded_size = 4 * ((len(item) + 2) // 3)
            if encoded_size + 2 > remaining:
                raise _PayloadTooLarge
            encoded = base64.b64encode(item).decode("ascii")
            consume(encoded_size + 2)
            return encoded
        if isinstance(item, int):
            try:
                encoded = str(item)
            except (ValueError, OverflowError) as error:
                raise _PayloadTooLarge from error
            consume(len(encoded))
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise _PayloadTooLarge
            consume(len(repr(item)))
            return item

        if isinstance(item, (list, tuple)):
            identity = id(item)
            if identity in active:
                raise _PayloadTooLarge
            active.add(identity)
            consume(2)
            out = []
            try:
                for index, child in enumerate(item):
                    if index:
                        consume(1)
                    out.append(visit(child, depth + 1))
            finally:
                active.remove(identity)
            return out

        mapping = item if isinstance(item, dict) else getattr(item, "__dict__", None)
        if not isinstance(mapping, dict):
            raise _PayloadTooLarge
        identity = id(item)
        if identity in active:
            raise _PayloadTooLarge
        active.add(identity)
        consume(2)
        out = {}
        written = 0
        try:
            for key, child in mapping.items():
                if not isinstance(key, str) or key.startswith("_"):
                    if isinstance(key, str) and key.startswith("_"):
                        continue
                    raise _PayloadTooLarge
                if written:
                    consume(1)
                consume_string(key)
                consume(1)
                out[key] = visit(child, depth + 1)
                written += 1
        finally:
            active.remove(identity)
        return out

    copied = visit(value, 0)
    return copied, budget - remaining


def _content_block_view(block: Any) -> Any:
    """Expose public MCP block fields without invoking an unbounded serializer."""
    if isinstance(block, (dict, list, tuple, str, bytes, int, float, bool)) or block is None:
        return block
    values = getattr(block, "__dict__", None)
    if isinstance(values, dict) and values:
        return block
    fields = {}
    for name in ("type", "text", "data", "blob", "audio", "mimeType", "uri", "name",
                 "description", "resource", "annotations", "meta"):
        if hasattr(block, name):
            fields[name] = getattr(block, name)
    return fields


def _result_data(content, structured: Any, limit: int, *, content_truncated: bool = False) -> dict:
    """Retain result data only when a bounded traversal fits the configured budget."""
    if content_truncated:
        return {"truncated": True}
    budget = max(limit * 4, 64)
    retained = []
    used = 32
    try:
        for block in content:
            item, size = _bounded_json_copy(_content_block_view(block), budget - used)
            retained.append(item)
            used += size + 1
        structured_copy, size = _bounded_json_copy(structured, budget - used)
        used += size
        if used > budget:
            raise _PayloadTooLarge
    except (_PayloadTooLarge, TypeError, ValueError, RecursionError, UnicodeError):
        return {"truncated": True}
    return {"content": retained, "structured": structured_copy}


def _bounded_utf8_size(value: str, current: int, limit: int) -> int:
    """Add a string's UTF-8 size without allocating its complete encoded form."""
    for char in value:
        current += len(char.encode("utf-8"))
        if current > limit:
            raise ToolError(f"MCP inputSchema is too large (>{limit} bytes)")
    return current


def _prepare_input_schema(schema: dict, description_limit: int) -> dict:
    """Validate, bound, and copy an MCP root input schema without flattening it."""
    if not isinstance(schema, dict):
        raise ToolError(f"MCP inputSchema must be a JSON object, got {type(schema).__name__}")
    nodes = 0
    properties = 0
    raw_bytes = 0

    def visit(value, depth: int) -> Any:
        nonlocal nodes, properties, raw_bytes
        if depth > _MAX_SCHEMA_DEPTH:
            raise ToolError(f"MCP inputSchema exceeds the maximum depth of {_MAX_SCHEMA_DEPTH}")
        nodes += 1
        if nodes > _MAX_SCHEMA_NODES:
            raise ToolError(f"MCP inputSchema exceeds the maximum node count of {_MAX_SCHEMA_NODES}")
        if isinstance(value, dict):
            for key in value:
                if not isinstance(key, str):
                    raise ToolError("MCP inputSchema object keys must be strings")
                raw_bytes = _bounded_utf8_size(key, raw_bytes, _MAX_SCHEMA_BYTES)
            props = value.get("properties")
            if props is not None:
                if not isinstance(props, dict):
                    raise ToolError("MCP inputSchema properties must be an object")
                properties += len(props)
                if properties > _MAX_SCHEMA_PROPERTIES:
                    raise ToolError(
                        f"MCP inputSchema exceeds the maximum property count of {_MAX_SCHEMA_PROPERTIES}")
            for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
                ref = value.get(keyword)
                if isinstance(ref, str) and not ref.startswith("#"):
                    raise ToolError(f"MCP inputSchema may only use local {keyword} values")
            out = {}
            for key, item in value.items():
                if key == "description" and isinstance(item, str):
                    raw_bytes = _bounded_utf8_size(item, raw_bytes, _MAX_SCHEMA_BYTES)
                    out[key] = _sanitize_text(item, description_limit)
                else:
                    out[key] = visit(item, depth + 1)
            return out
        if isinstance(value, list):
            return [visit(item, depth + 1) for item in value]
        if isinstance(value, str):
            raw_bytes = _bounded_utf8_size(value, raw_bytes, _MAX_SCHEMA_BYTES)
        else:
            raw_bytes += 8
        if raw_bytes > _MAX_SCHEMA_BYTES:
            raise ToolError(f"MCP inputSchema is too large (>{_MAX_SCHEMA_BYTES} bytes)")
        return value

    prepared = cast(dict, visit(schema, 0))
    try:
        encoded = json.dumps(prepared, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ToolError(f"MCP inputSchema is not valid JSON: {e}") from e
    size = len(encoded.encode("utf-8"))
    if size > _MAX_SCHEMA_BYTES:
        raise ToolError(f"MCP inputSchema is too large ({size} bytes > {_MAX_SCHEMA_BYTES})")
    root_type = prepared.get("type")
    if root_type not in (None, "object"):
        raise ToolError(f"MCP inputSchema root type must be 'object', got {root_type!r}")
    prepared.setdefault("type", "object")
    prepared.setdefault("properties", {})
    required = prepared.get("required")
    if required is not None and (
            not isinstance(required, list) or any(not isinstance(name, str) for name in required)):
        raise ToolError("MCP inputSchema required must be a list of property names")
    try:
        from jsonschema import validators
        validator = validators.validator_for(prepared)
        validator.check_schema(prepared)
    except Exception as e:
        raise ToolError(f"MCP inputSchema is not a valid JSON Schema: {e}") from e
    return prepared


def _fingerprint(remote_name: str, description: str, input_schema: dict) -> str:
    """Compute a sha256 over (remote name + description + canonicalized inputSchema) as the tool-definition fingerprint (tool pinning):

    if the server later swaps out the description / schema (a rug-pull), the fingerprint changes and can be caught by an app that pins it.
    """
    digest = hashlib.sha256()
    canonical = json.dumps(input_schema or {}, sort_keys=True, ensure_ascii=False)
    for index, value in enumerate((remote_name, description, canonical)):
        if index:
            digest.update(b"\x00")
        for offset in range(0, len(value), 4096):
            digest.update(value[offset:offset + 4096].encode("utf-8"))
    return digest.hexdigest()


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
                 url: Optional[str] = None, headers: Optional[dict] = None, auth: "Optional[httpx.Auth]" = None,
                 requires_confirmation: bool = True, max_desc_chars: int = 4096,
                 expected_fingerprints: Optional[dict] = None, timeout: Optional[float] = 30.0,
                 max_tools: int = 128, max_result_chars: int = 50_000,
                 allow_insecure_http: bool = False,
                 prompts: "Optional[PromptRegistry]" = None):
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
            expected_fingerprints: Optional exact {display name: sha256} pin set. When supplied,
                load_tools rejects changed, added, or omitted tool definitions. Without it, each
                MCPTool exposes its fingerprint for application-managed pinning.
            timeout: Timeout in seconds for the handshake (initialize) and tool calls (call_tool), default 30; a stuck
                server will not make the agent wait forever. For the HTTP transport it also serves as the per-request
                timeout (streamablehttp_client's timeout). Pass None to disable the timeout (wait forever, not recommended).
            max_tools: Maximum tools accepted from one server.
            max_result_chars: Maximum text characters retained from one tool result.
            allow_insecure_http: Permit cleartext HTTP for non-loopback URLs. Defaults to False.
            prompts: Optional prompt registry (PromptRegistry); the text call_tool feeds back to the model (session error / no output / tool-error prefix) comes from it, defaulting to DEFAULT_PROMPTS.
        """
        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be positive or None, got {timeout}")
        if not _NAMESPACE_RE.fullmatch(namespace or ""):
            raise ValueError("namespace must match ^[a-zA-Z0-9_-]{1,64}$")
        if max_desc_chars <= 0:
            raise ValueError(f"max_desc_chars must be positive, got {max_desc_chars}")
        if max_tools < 1:
            raise ValueError(f"max_tools must be >= 1, got {max_tools}")
        if max_result_chars < 1:
            raise ValueError(f"max_result_chars must be >= 1, got {max_result_chars}")
        # Pick exactly one transport: command (stdio) or url (HTTP). Giving both or neither is a configuration error, fail loud.
        if (command is None) == (url is None):
            raise ValueError("MCPClient transport must be exactly one: local uses command=..., remote uses url=... (not both, not neither)")
        if url is not None and (args or env is not None):
            raise ValueError("Remote HTTP mode (url) does not accept args/env, those are stdio subprocess parameters; remote uses headers/auth")
        if command is not None and (headers is not None or auth is not None):
            raise ValueError("stdio mode (command) does not accept headers/auth, those are remote HTTP transport parameters; local uses env")
        if url is not None:
            parsed = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                raise ValueError("MCP URL must be an absolute http:// or https:// endpoint")
            if parsed.username is not None or parsed.password is not None:
                raise ValueError("MCP URL must not contain embedded credentials")
            loopback = parsed.hostname == "localhost"
            try:
                loopback = loopback or ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                pass
            if parsed.scheme == "http" and not loopback and not allow_insecure_http:
                raise ValueError(
                    "cleartext MCP HTTP is refused for non-loopback hosts; use https:// or pass "
                    "allow_insecure_http=True explicitly")
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
        self.max_tools = max_tools
        self.max_result_chars = max_result_chars
        self.allow_insecure_http = allow_insecure_http
        self.prompts = prompts or DEFAULT_PROMPTS   # Text call_tool feeds back to the model comes from the registry (whole language swappable).
        self._session = None
        self._stack: Optional[AsyncExitStack] = None

    async def __aenter__(self):
        """Establish the connection (stdio subprocess or remote HTTP), create the ClientSession, and handshake (initialize)."""
        if self._stack is not None:
            raise ToolError("MCPClient is already entered")
        try:
            from mcp import ClientSession
        except ImportError as e:
            raise ToolError("Using MCP requires installing first: uv add 'agentmaker[mcp]'") from e
        stack = AsyncExitStack()
        self._stack = stack
        try:
            read, write = await self._open_transport()   # Branch by command / url; the two transports are fully shared afterward.
            self._session = await stack.enter_async_context(ClientSession(read, write))
            try:
                await asyncio.wait_for(self._session.initialize(), timeout=self.timeout)
            except asyncio.TimeoutError as e:
                raise ToolError(f"MCP server handshake timed out (>{self.timeout}s): {self.url or self.command}") from e
        except BaseException as error:
            try:
                await stack.aclose()
            except Exception as cleanup_error:  # noqa: BLE001
                error.add_note(f"MCP transport cleanup also failed: {cleanup_error}")
            finally:
                self._stack = None
                self._session = None
            raise
        return self

    async def _open_transport(self):
        """Establish the underlying (read, write) streams per configuration: with a url, use remote Streamable HTTP, otherwise a local stdio subprocess.

        After the handshake, both transports expose an identically shaped (read, write) consumed by the same
        ClientSession, so the upper layer has zero branching.
        """
        stack = self._stack
        if stack is None:
            raise ToolError("MCP transport cannot open before entering the client context")
        if self.url is not None:
            from mcp.client.streamable_http import streamablehttp_client
            kwargs = {"url": self.url, "headers": self.headers, "auth": self.auth}
            if self.timeout is not None:            # streamablehttp_client's timeout takes float seconds (when None, use the SDK default rather than forcing None).
                kwargs["timeout"] = self.timeout
            # streamablehttp_client yields a triple (read, write, get_session_id); take only the first two to align with stdio.
            read, write, _get_session_id = await stack.enter_async_context(streamablehttp_client(**kwargs))
            return read, write
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        command = self.command
        if command is None:
            raise ToolError("MCP stdio transport requires a command")
        params = StdioServerParameters(command=command, args=self.args, env=self.env)
        read, write = await stack.enter_async_context(stdio_client(params))
        return read, write

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Close the session and underlying transport (stdio subprocess / remote HTTP connection, all cleaned up uniformly by AsyncExitStack)."""
        stack = self._stack
        self._stack = None
        self._session = None
        if stack is not None:
            try:
                await stack.aclose()
            except BaseException as cleanup_error:
                if exc_value is None:
                    raise
                exc_value.add_note(f"MCP transport cleanup also failed: {cleanup_error}")
        return False

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
        try:
            resp = await asyncio.wait_for(self._session.list_tools(), timeout=self.timeout)
        except asyncio.TimeoutError as e:
            raise ToolError(f"MCP tool listing timed out (>{self.timeout}s)") from e
        if len(resp.tools) > self.max_tools:
            raise ToolError(f"MCP server returned too many tools ({len(resp.tools)} > {self.max_tools})")
        tools = []
        taken = set()
        for t in resp.tools:
            display = _dedupe_name(sanitize_tool_name(f"{self.namespace}_{t.name}"), taken)   # Normalize to a valid name and deduplicate.
            taken.add(display)
            raw_desc = t.description or ""
            schema = getattr(t, "inputSchema", None)
            if schema is None:
                schema = {}
            description = _sanitize_text(raw_desc, self.max_desc_chars)
            tool = MCPTool(self, display, description, schema,
                           remote_name=t.name, origin=f"mcp:{self.namespace}",
                           requires_confirmation=self.requires_confirmation,
                           max_desc_chars=self.max_desc_chars)
            fp = _fingerprint(t.name, description, tool.get_input_schema())
            if self.expected_fingerprints is not None and self.expected_fingerprints.get(display) != fp:
                raise ToolError(
                    f"MCP tool '{display}' fingerprint mismatch (expected {self.expected_fingerprints.get(display)}, actual {fp}): "
                    "the tool definition may have been tampered with by the server (rug-pull), refused to load")
            tool.fingerprint = fp
            tools.append(tool)
        if self.expected_fingerprints is not None:
            missing = sorted(set(self.expected_fingerprints) - taken)
            if missing:
                raise ToolError(f"MCP server omitted pinned tools: {missing}")
        return tools

    async def call_tool(self, name: str, arguments: dict) -> ToolResponse:
        """Call one of the server's tools and return a ToolResponse.

        Text concatenates bounded, sanitized text blocks for the model. Data contains the content blocks and
        structuredContent when their serialized representation fits the configured result bound; oversized or
        invalid payloads are replaced by a truncation marker. A tool error (isError) yields status="error"; a
        closed or unestablished connection yields a readable ToolResponse error.
        """
        if self._session is None:
            return ToolResponse.error(self.prompts.text("tool.msg.mcp.no_session"))
        try:
            result = await asyncio.wait_for(self._session.call_tool(name, arguments), timeout=self.timeout)
        except asyncio.TimeoutError:
            return ToolResponse.error(self.prompts.render("tool.msg.mcp.timeout", timeout=self.timeout))
        content = []
        content_truncated = False
        for index, block in enumerate(getattr(result, "content", None) or ()):
            if index >= _MAX_RESULT_BLOCKS:
                content_truncated = True
                break
            content.append(block)
        text = _result_text(content, self.max_result_chars,
                            self.prompts.text("tool.msg.mcp.no_text"),
                            content_truncated=content_truncated)
        data = _result_data(content, getattr(result, "structuredContent", None),
                            self.max_result_chars, content_truncated=content_truncated)
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
            input_schema: The tool's root JSON Schema (from the server's inputSchema).
            remote_name: The real tool name on the server side (used when calling); defaults to name.
            origin: Origin marker, stamped "mcp:{namespace}" by MCPClient (the trust root, not overridable by the tool definition).
            requires_confirmation: Whether human confirmation is required (instance attribute shadowing the class attribute); default True, since untrusted remote tools require confirmation by default.
            fingerprint: Tool-definition fingerprint (sha256 of remote_name + description + schema), for the app to pin and compare (tool pinning).
            max_desc_chars: Length limit for parameter-description sanitization (same source as client).
        """
        if max_desc_chars <= 0:
            raise ValueError(f"max_desc_chars must be positive, got {max_desc_chars}")
        super().__init__(name=name, description=description, origin=origin)
        self.requires_confirmation = requires_confirmation   # Instance attribute shadows the class attribute: one setting per server.
        self.fingerprint = fingerprint
        self._client = client
        self._input_schema = _prepare_input_schema({} if input_schema is None else input_schema,
                                                   max_desc_chars)
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
            sub = v if isinstance(v, dict) else {}
            out.append(ToolParameter(name=k, type=sub.get("type", "any"),
                                     description=sub.get("description", ""), required=k in required,
                                     schema=sub or None))
        return out

    def get_input_schema(self) -> dict:
        """Return the bounded, sanitized root inputSchema unchanged in structure."""
        return self._input_schema
