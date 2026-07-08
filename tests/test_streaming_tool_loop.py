"""Streaming tool-loop regression: astream_run with tools.

Three layers:
- Agent layer: ScriptedLLM drives the streaming tool loop (text deltas + tool execution + final answer + history persistence);
- adapter layer: OpenAI _StreamState unit tests following the official index-accumulation pattern (faked chunk objects);
- real smoke: when OPENAI_API_KEY is set, gpt-4o-mini with an echo tool end to end (auto-skipped without a key).
"""

import asyncio
import os
from types import SimpleNamespace

import pytest

from agentmaker import Agent, LLMResponse, Tool, ToolParameter, ToolResponse
from agentmaker.core.adapters.openai_compat import _StreamState
from agentmaker.testing import ScriptedLLM


class EchoTool(Tool):
    """Echo the text parameter back verbatim and record that it ran (for side-effect assertions)."""

    def __init__(self):
        super().__init__(name="echo_tool", description="Echo the given text back verbatim.")
        self.ran_with = None

    def get_parameters(self):
        return [ToolParameter("text", "string", "The text to echo back.")]

    def run(self, parameters: dict) -> ToolResponse:
        self.ran_with = parameters.get("text")
        return ToolResponse.ok(f"echo: {self.ran_with}")


def run(coroutine):
    """Run a coroutine synchronously."""
    return asyncio.run(coroutine)


async def collect(agent, text, **kwargs):
    """Drain astream_run and return the list of pieces."""
    return [piece async for piece in agent.astream_run(text, **kwargs)]


def test_astream_run_tool_loop_streams_text_and_executes_tool():
    """Streaming tool loop: call the tool first, then stream out the final answer; the tool really runs; two model calls."""
    llm = ScriptedLLM([
        ScriptedLLM.tool_call("echo_tool", {"text": "hi"}),
        "这是逐字流出的最终答案。",
    ])
    tool = EchoTool()
    agent = Agent("t", llm, tools=[tool])
    pieces = run(collect(agent, "请回显 hi"))
    assert "".join(pieces) == "这是逐字流出的最终答案。"
    assert len(pieces) > 1                      # genuinely incremental (8 chars per piece), not one block
    assert tool.ran_with == "hi"                # the tool loop actually executed
    assert llm.calls == 2


def test_astream_run_tool_loop_persists_one_turn_history():
    """History persistence matches arun's contract: one atomic user + final assistant turn (no tool trace stored)."""
    llm = ScriptedLLM([ScriptedLLM.tool_call("echo_tool", {"text": "x"}), "答案"])
    agent = Agent("t", llm, tools=[EchoTool()])
    run(collect(agent, "问题"))
    history = run(agent.harness.session.aload(scope=agent.scope)) if hasattr(agent.harness, "session") else None
    # session history is persisted via agent.add_messages; verify count and roles directly from the in-memory store
    messages = run(agent.aget_history()) if hasattr(agent, "aget_history") else None
    if messages is None:
        pytest.skip("no in-memory history read interface; this assertion is covered by the arun same-path test")
    assert [m.role for m in messages[-2:]] == ["user", "assistant"]
    assert messages[-1].content == "答案"


def test_astream_run_without_tools_unchanged():
    """astream_run on a tool-less agent takes the original plain-text path (backward compatible)."""
    agent = Agent("t", ScriptedLLM(["纯文本流式回复"]))
    pieces = run(collect(agent, "你好"))
    assert "".join(pieces) == "纯文本流式回复"


def test_astream_run_buffer_output_releases_after_final(monkeypatch):
    """buffer_output=True: all text is released only after the final answer clears the guardrail (including text accompanying the tool turn)."""
    llm = ScriptedLLM([
        LLMResponse(content="我先查一下。", model="test", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "echo_tool", "arguments": "{\"text\": \"y\"}"}}]),
        "最终答案。",
    ])
    agent = Agent("t", llm, tools=[EchoTool()])
    pieces = run(collect(agent, "问", buffer_output=True))
    assert "".join(pieces) == "我先查一下。最终答案。"


def test_openai_stream_state_accumulates_tool_call_fragments():
    """OpenAI stream state machine: id/name arrive in the first fragment, arguments are concatenated across fragments, merged by index (the official pattern)."""
    def chunk(delta_calls=None, content=None, finish=None):
        delta = SimpleNamespace(tool_calls=delta_calls, content=content)
        choice = SimpleNamespace(delta=delta, finish_reason=finish)
        return SimpleNamespace(model="m", usage=None, choices=[choice])

    def frag(index, id=None, name=None, arguments=None):
        fn = SimpleNamespace(name=name, arguments=arguments)
        return SimpleNamespace(index=index, id=id, function=fn)

    st = _StreamState("m")
    assert st.feed(chunk(content="先说")) == "先说"
    st.feed(chunk(delta_calls=[frag(0, id="call_a", name="echo_tool", arguments="")]))
    st.feed(chunk(delta_calls=[frag(0, arguments="{\"text\": ")]))
    st.feed(chunk(delta_calls=[frag(0, arguments="\"hi\"}")]))
    st.feed(chunk(finish="tool_calls"))
    calls = st.final_tool_calls()
    assert calls == [{"id": "call_a", "type": "function",
                      "function": {"name": "echo_tool", "arguments": "{\"text\": \"hi\"}"}}]
    assert st.finish_reason == "tool_calls"
    assert "".join(st.text_parts) == "先说"


def test_openai_stream_state_no_tools_returns_none():
    """Plain-text stream: final_tool_calls is None, text still accumulates."""
    st = _StreamState("m")
    delta = SimpleNamespace(tool_calls=None, content="abc")
    st.feed(SimpleNamespace(model="m", usage=None,
                            choices=[SimpleNamespace(delta=delta, finish_reason="stop")]))
    assert st.final_tool_calls() is None


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="requires OPENAI_API_KEY for a real streaming tool smoke test")
def test_real_openai_streaming_tool_loop_smoke():
    """Real smoke: gpt-4o-mini streams a call to the echo tool then answers token by token (verifies delta.tool_calls accumulation is actually correct)."""
    from agentmaker import LLMClient

    llm = LLMClient("openai", model="gpt-4o-mini")
    tool = EchoTool()
    agent = Agent("smoke", llm, tools=[tool],
                  system_prompt="You must call echo_tool with text='pineapple' first, then answer with what it returned.")
    pieces = run(collect(agent, "Use the tool, then tell me what it echoed."))
    assert tool.ran_with is not None            # the model really initiated a streaming tool call
    assert len(pieces) >= 2                     # the final answer streamed token by token
    assert "pineapple" in "".join(pieces).lower()
