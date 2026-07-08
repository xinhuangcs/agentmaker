"""Hermetic tests for the tool-call translation shim (ToolEmulationAdapter) used with non-function-calling models (no key / no network)."""
import asyncio

import pytest

from agentmaker import Agent, CalculatorTool, LLMClient
from agentmaker.core.adapters.tool_emulation import ToolEmulationAdapter
from agentmaker.core.llm_response import LLMResponse


class _FakeDelegate:
    """Fakes a non-function-calling model's underlying adapter: records the messages it receives and returns scripted LLMResponses."""

    def __init__(self, script):
        self._script = list(script)
        self.seen = []
        self.last_stream_stats = None

    async def chat(self, messages, *, temperature=None, max_tokens=None, output_schema=None, **kwargs):
        self.seen.append((messages, kwargs))
        return self._script.pop(0)

    async def stream(self, messages, **kwargs):
        self.seen.append((messages, kwargs))
        for piece in ["a", "b"]:
            yield piece


_SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "搜索网络",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    },
}]


def test_emulation_parses_and_strips_directive():
    d = _FakeDelegate([LLMResponse(content='先想想。{"tool": "search", "arguments": {"q": "猫"}}', model="m")])
    r = asyncio.run(ToolEmulationAdapter(d).chat([{"role": "user", "content": "查猫"}], tools=_SEARCH_TOOL))
    assert r.tool_calls and r.tool_calls[0]["function"]["name"] == "search"
    assert r.tool_calls[0]["function"]["arguments"] == '{"q": "猫"}'
    assert r.content == "先想想。"                            # directive JSON stripped from content, thinking text kept
    # the underlying adapter received no native tools param, and a tool catalog was injected into system
    messages, kwargs = d.seen[0]
    assert "tools" not in kwargs
    assert any(m["role"] == "system" and "search" in m["content"] for m in messages)


def test_emulation_parses_tool_call_with_braces_in_arguments():
    """Tool argument values containing braces (common in code/text/query) still parse correctly: brace matching skips over the interior of JSON strings."""
    for content in ['{"tool": "search", "arguments": {"q": "a{b}c"}}',       # value contains { and }
                    '想想。{"tool": "search", "arguments": {"q": "x}y"}}',    # leading text + value contains }
                    '{"tool": "search", "arguments": {"q": "带反斜杠\\\\和\\"引号"}}']:  # escapes
        d = _FakeDelegate([LLMResponse(content=content, model="m")])
        r = asyncio.run(ToolEmulationAdapter(d).chat([{"role": "user", "content": "q"}], tools=_SEARCH_TOOL))
        assert r.tool_calls and r.tool_calls[0]["function"]["name"] == "search", content


def test_emulation_strips_correct_directive_when_duplicate_substring_earlier():
    """When the directive string appears earlier (escaped) inside another JSON string, the **real** directive is excised precisely at the index _extract located, without deleting the preceding text."""
    note = '{"note": "示例是 {\\"tool\\": \\"search\\", \\"arguments\\": {}}"}'   # contains an escaped copy of the same directive
    directive = '{"tool": "search", "arguments": {}}'                          # the real (unescaped) directive at the end
    content = f"{note} 然后我决定 {directive}"
    d = _FakeDelegate([LLMResponse(content=content, model="m")])
    r = asyncio.run(ToolEmulationAdapter(d).chat([{"role": "user", "content": "q"}], tools=_SEARCH_TOOL))
    assert r.tool_calls and r.tool_calls[0]["function"]["name"] == "search"
    assert note in r.content and directive not in r.content   # the leading note is kept verbatim, the real directive is excised precisely


def test_emulation_flattens_tool_trace():
    """History with a tool trace (assistant.tool_calls + role:tool) is flattened to plain text a non-function-calling model can read."""
    d = _FakeDelegate([LLMResponse(content="好的", model="m")])
    history = [
        {"role": "user", "content": "查猫"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q": "猫"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "找到 3 条"},
    ]
    asyncio.run(ToolEmulationAdapter(d).chat(history, tools=_SEARCH_TOOL))
    flat = d.seen[0][0]
    assert all(m["role"] in ("system", "user", "assistant") for m in flat)   # no role:tool (flattened into user)
    joined = "\n".join(m["content"] for m in flat)
    assert "search" in joined and "找到 3 条" in joined                       # both the call and the result appear as text


def test_emulation_passthrough_without_tools():
    """Without tools -> passes straight through to the underlying adapter (messages unchanged, response as-is)."""
    resp = LLMResponse(content="直接回答", model="m")
    d = _FakeDelegate([resp])
    msgs = [{"role": "user", "content": "你好"}]
    r = asyncio.run(ToolEmulationAdapter(d).chat(msgs))
    assert r is resp and d.seen[0][0] == msgs                 # messages not flattened, response as-is


def test_emulation_plain_answer_when_no_directive():
    """No tool-call directive in the model reply -> treated as a plain-text answer (tool_calls is None)."""
    d = _FakeDelegate([LLMResponse(content="答案是 42", model="m")])
    r = asyncio.run(ToolEmulationAdapter(d).chat([{"role": "user", "content": "?"}], tools=_SEARCH_TOOL))
    assert r.tool_calls is None and r.content == "答案是 42"


def test_llmclient_emulate_tools_wires_shim_and_sets_fc():
    """LLMClient(emulate_tools=True): wraps a ToolEmulationAdapter and sets supports_function_calling to True (construction does not hit the network)."""
    client = LLMClient(provider="openai_compatible", model="local-model", api_key="x",
                       base_url="http://localhost:1234/v1", supports_function_calling=False, emulate_tools=True)
    assert isinstance(client._adapter, ToolEmulationAdapter)
    assert client.supports_function_calling is True           # emulated via the shim, so a tool-enabled Agent no longer fails loud


class _EmuLLM:
    """Wraps a ToolEmulationAdapter into an LLM the Agent can use (duck: chat/stream/supports_function_calling/protocol)."""
    supports_function_calling = True
    protocol = "openai"
    model = "no-fc-model"
    context_window = None

    def __init__(self, delegate):
        self._emu = ToolEmulationAdapter(delegate)

    async def chat(self, messages, **kwargs):
        return await self._emu.chat(messages, **kwargs)

    async def stream(self, messages, **kwargs):
        async for piece in self._emu.stream(messages, **kwargs):
            yield piece

    @property
    def last_stream_stats(self):
        return self._emu.last_stream_stats


def test_emulation_end_to_end_agent_loop():
    """Full chain: a non-function-calling model "text-calls" calculator through the shim -> Agent executes -> result fed back -> model gives the final answer."""
    delegate = _FakeDelegate([
        LLMResponse(content='算一下。{"tool": "calculator", "arguments": {"expression": "2+3"}}', model="m"),
        LLMResponse(content="答案是 5", model="m"),           # after receiving the tool result (no directive) -> plain-text answer
    ])
    agent = Agent("助手", _EmuLLM(delegate), tools=[CalculatorTool()])
    assert agent.run("2+3 等于几").final_output == "答案是 5"
    # second call (the tool-result turn): the underlying adapter receives the flattened text trace, including the calculator result "5"
    second_turn_msgs = delegate.seen[1][0]
    assert any("5" in m["content"] for m in second_turn_msgs if m["role"] == "user")
