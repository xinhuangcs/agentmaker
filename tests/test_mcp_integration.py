import asyncio
from types import SimpleNamespace

import pytest

from agentmaker.core.adapters.gemini import GeminiAdapter
from agentmaker.core.exceptions import ToolError
from agentmaker.tools.integrations.mcp import MCPClient, MCPTool, _fingerprint


@pytest.mark.parametrize("namespace", ["", "bad space", "bad.dot", "x" * 65])
def test_mcp_client_rejects_invalid_namespace(namespace):
    with pytest.raises(ValueError, match="namespace"):
        MCPClient(command="server", namespace=namespace)


@pytest.mark.parametrize("kwargs", [
    {"max_desc_chars": 0},
    {"max_tools": 0},
    {"max_result_chars": 0},
])
def test_mcp_client_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        MCPClient(command="server", namespace="safe", **kwargs)


def test_mcp_remote_http_requires_transport_security():
    with pytest.raises(ValueError, match="cleartext"):
        MCPClient(url="http://example.com/mcp", namespace="safe")
    assert MCPClient(url="http://127.0.0.1:8000/mcp", namespace="safe").url
    assert MCPClient(url="http://example.com/mcp", namespace="safe",
                     allow_insecure_http=True).url
    with pytest.raises(ValueError, match="embedded credentials"):
        MCPClient(url="https://user:secret@example.com/mcp", namespace="safe")


def test_mcp_tool_rejects_invalid_direct_description_bound():
    with pytest.raises(ValueError, match="max_desc_chars"):
        MCPTool(None, "remote", "description", {}, max_desc_chars=0)


@pytest.mark.parametrize("schema", [[], "", False, 0])
def test_mcp_tool_rejects_falsey_non_object_schema(schema):
    with pytest.raises(ToolError, match="JSON object"):
        MCPTool(None, "remote", "description", schema)


def test_mcp_load_tools_bounds_count_before_schema_processing():
    class Session:
        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(name="a", description="", inputSchema={}),
                                          SimpleNamespace(name="b", description="", inputSchema={})])

    client = MCPClient(command="server", namespace="safe", max_tools=1)
    client._session = Session()
    with pytest.raises(ToolError, match="too many tools"):
        asyncio.run(client.load_tools())


def test_mcp_load_tools_rejects_falsey_non_object_schema():
    class Session:
        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(
                name="invalid", description="", inputSchema=[])])

    client = MCPClient(command="server", namespace="safe")
    client._session = Session()
    with pytest.raises(ToolError, match="JSON object"):
        asyncio.run(client.load_tools())


def test_mcp_load_tools_times_out():
    class Session:
        async def list_tools(self):
            await asyncio.sleep(10)

    client = MCPClient(command="server", namespace="safe", timeout=0.01)
    client._session = Session()
    with pytest.raises(ToolError, match="listing timed out"):
        asyncio.run(client.load_tools())


def test_mcp_load_tools_rejects_missing_pinned_tool():
    schema = {"type": "object", "properties": {}}

    class Session:
        async def list_tools(self):
            return SimpleNamespace(tools=[SimpleNamespace(
                name="present", description="", inputSchema=schema)])

    expected = {
        "safe_present": _fingerprint("present", "", schema),
        "safe_missing": "0" * 64,
    }
    client = MCPClient(command="server", namespace="safe", expected_fingerprints=expected)
    client._session = Session()
    with pytest.raises(ToolError, match="omitted pinned tools"):
        asyncio.run(client.load_tools())


def test_mcp_client_reentry_is_rejected():
    client = MCPClient(command="server", namespace="safe")
    client._stack = object()
    with pytest.raises(ToolError, match="already entered"):
        asyncio.run(client.__aenter__())


def test_mcp_client_exit_clears_state_when_cleanup_fails():
    class BrokenStack:
        async def aclose(self):
            raise RuntimeError("cleanup failed")

    client = MCPClient(command="server", namespace="safe")
    client._stack = BrokenStack()
    client._session = object()
    with pytest.raises(RuntimeError, match="cleanup failed"):
        asyncio.run(client.__aexit__(None, None, None))
    assert client._stack is None
    assert client._session is None


def test_mcp_client_cleanup_does_not_replace_business_error():
    class BrokenStack:
        async def aclose(self):
            raise RuntimeError("cleanup failed")

    client = MCPClient(command="server", namespace="safe")
    client._stack = BrokenStack()
    error = ValueError("business failed")
    assert asyncio.run(client.__aexit__(ValueError, error, None)) is False
    assert any("cleanup also failed" in note for note in error.__notes__)


def test_mcp_result_text_and_data_are_bounded():
    class Content:
        text = "x" * 100

        def model_dump(self, *, mode):
            assert mode == "json"
            return {"type": "text", "text": self.text}

    class Session:
        async def call_tool(self, name, arguments):
            return SimpleNamespace(content=[Content()], structuredContent=None, isError=False)

    client = MCPClient(command="server", namespace="safe", max_result_chars=10)
    client._session = Session()
    response = asyncio.run(client.call_tool("remote", {}))
    assert response.status == "success"
    assert response.text.startswith("x" * 10)
    assert response.data == {"truncated": True}


def test_mcp_huge_structured_and_text_results_are_replaced_early():
    class Content:
        text = "x" * 2_000_000

        def model_dump_json(self):
            raise AssertionError("unbounded block serializer must not be called")

    class Session:
        async def call_tool(self, name, arguments):
            return SimpleNamespace(
                content=[Content()],
                structuredContent={"items": ["y" * 2_000_000]},
                isError=False)

    client = MCPClient(command="server", namespace="safe", max_result_chars=64)
    client._session = Session()
    response = asyncio.run(client.call_tool("remote", {}))
    assert response.text.startswith("x" * 64)
    assert response.data == {"truncated": True}


def test_mcp_result_block_count_is_bounded():
    class Content:
        text = "x"

    class Session:
        async def call_tool(self, name, arguments):
            return SimpleNamespace(content=(Content() for _ in range(10_000)),
                                   structuredContent=None, isError=False)

    client = MCPClient(command="server", namespace="safe", max_result_chars=10_000)
    client._session = Session()
    response = asyncio.run(client.call_tool("remote", {}))
    assert response.data == {"truncated": True}
    assert response.text.endswith("…(truncated)")


def test_mcp_sanitizer_removes_unicode_format_controls():
    from agentmaker.tools.integrations.mcp import _sanitize_text
    assert _sanitize_text("safe\u202eevil\u2066", 100) == "safeevil"
    assert _sanitize_text("tag\U000e0041smuggle\ufeff", 100) == "tagsmuggle"


def test_mcp_sanitizer_keeps_joiners_and_soft_hyphen():
    from agentmaker.tools.integrations.mcp import _sanitize_text
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
    assert _sanitize_text(family, 100) == family
    assert _sanitize_text("\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645", 100) == "\u0645\u06cc\u200c\u062e\u0648\u0627\u0647\u0645"
    assert _sanitize_text("Zeit\u00adplan", 100) == "Zeit\u00adplan"


def test_gemini_preserves_json_schema_references_for_tools():
    pytest.importorskip("google.genai")
    schema = {
        "type": "object",
        "$defs": {"Payload": {"type": "object", "properties": {"value": {"type": "string"}}}},
        "properties": {"payload": {"$ref": "#/$defs/Payload"}},
    }
    converted = GeminiAdapter._tools_to_gemini([
        {"type": "function", "function": {"name": "lookup", "description": "", "parameters": schema}}
    ])
    declaration = converted[0].function_declarations[0]
    assert declaration.parameters is None
    assert declaration.parameters_json_schema == schema
