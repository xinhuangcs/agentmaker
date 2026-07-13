"""Streaming tool-loop behavior for astream_run with tools.

Three layers:
- Agent layer: ScriptedLLM drives the streaming tool loop (text deltas + tool execution + final answer + history persistence);
- adapter layer: OpenAI _StreamState unit tests following the official index-accumulation pattern (faked chunk objects);
- real smoke: when OPENAI_API_KEY is set, gpt-4o-mini with an echo tool end to end (auto-skipped without a key).
"""

import asyncio
import os
from types import SimpleNamespace

import pytest

from agentmaker import Agent, LLMResponse, RunPolicy, Tool, ToolParameter, ToolResponse
from agentmaker.core.adapters.openai_compat import _StreamState
from agentmaker.core.exceptions import (GuardrailTripwireError, LLMConfigError, LLMResponseError,
                                        RunLimitExceeded)
from agentmaker.runtime.guardrails import CallableGuardrail, GuardrailResult
from agentmaker.testing import ScriptedLLM
from agentmaker.testing import RecordingHook


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


def test_streaming_tool_loop_forwards_adapter_assistant_state():
    """Streaming tool turns preserve adapter continuation state for the next model call."""
    class _CaptureLLM(ScriptedLLM):
        def __init__(self, script):
            super().__init__(script)
            self.seen = []

        async def stream(self, messages, *, tools=None, **kwargs):
            self.seen.append([dict(message) for message in messages])
            async for item in super().stream(messages, tools=tools, **kwargs):
                yield item

    turn = ScriptedLLM.tool_call("echo_tool", {"text": "hi"})
    turn.assistant_message = {"reasoning_content": "stream-reasoning"}
    llm = _CaptureLLM([turn, "done"])
    agent = Agent("t", llm, tools=[EchoTool()])
    assert "".join(run(collect(agent, "question"))) == "done"
    assistant = next(message for message in llm.seen[1] if message.get("role") == "assistant")
    assert assistant["reasoning_content"] == "stream-reasoning"


def test_astream_run_tool_loop_persists_one_turn_history():
    """History persistence matches arun's contract: one atomic user + final assistant turn (no tool trace stored)."""
    llm = ScriptedLLM([ScriptedLLM.tool_call("echo_tool", {"text": "x"}), "答案"])
    agent = Agent("t", llm, tools=[EchoTool()])
    run(collect(agent, "问题"))
    messages = agent.get_history()
    assert [m.role for m in messages[-2:]] == ["user", "assistant"]
    assert messages[-1].content == "答案"


def test_astream_run_without_tools_unchanged():
    """astream_run on a tool-less agent uses the plain-text path."""
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


def test_buffered_tool_stream_guards_every_released_turn():
    """Buffered mode blocks untrusted text from any streamed tool-loop turn."""
    llm = ScriptedLLM([
        LLMResponse(content="BLOCKED", model="test", tool_calls=[{
            "id": "c1", "type": "function",
            "function": {"name": "echo_tool", "arguments": "{\"text\": \"y\"}"}}]),
        "safe final",
    ])
    agent = Agent(
        "t", llm, tools=[EchoTool()],
        output_guardrails=[CallableGuardrail(lambda text: "BLOCKED" not in text, message="blocked")],
    )
    received = []

    async def consume():
        async for piece in agent.astream_run("问", buffer_output=True):
            received.append(piece)

    with pytest.raises(GuardrailTripwireError, match="blocked"):
        run(consume())
    assert received == []
    assert agent.get_history() == []


def test_buffered_stream_checks_deadline_after_output_guardrail():
    """Buffered text stays private when an async output guardrail crosses the run deadline."""
    class SlowPass:
        async def acheck(self, text):
            await asyncio.sleep(0.03)
            return GuardrailResult(True)

    agent = Agent(
        "t", ScriptedLLM(["late"]), output_guardrails=[SlowPass()],
        run_policy=RunPolicy(deadline_seconds=0.01),
    )
    received = []

    async def consume():
        async for piece in agent.astream_run("问", buffer_output=True):
            received.append(piece)

    with pytest.raises(RunLimitExceeded, match="wall-clock time limit"):
        run(consume())
    assert received == []


def test_delivered_stream_is_persisted_despite_deadline():
    """A non-buffered stream the consumer already saw is committed to history even when the post-hoc guardrail crosses the deadline."""
    class SlowPass:
        async def acheck(self, text):
            await asyncio.sleep(0.03)
            return GuardrailResult(True)

    agent = Agent(
        "t", ScriptedLLM(["seen by user"]), output_guardrails=[SlowPass()],
        run_policy=RunPolicy(deadline_seconds=0.01),
    )
    received = []

    async def consume():
        async for piece in agent.astream_run("问"):
            received.append(piece)

    run(consume())
    assert "".join(received) == "seen by user"
    assert [m.content for m in agent.get_history()] == ["问", "seen by user"]


def test_undelivered_fallback_tail_still_obeys_deadline():
    """The tool loop's fallback tail was never streamed, so a deadline crossing before it is released aborts the turn instead of committing it."""
    class SlowPass:
        async def acheck(self, text):
            await asyncio.sleep(0.03)
            return GuardrailResult(True)

    class EmptyReplyLLM:
        provider = "test"
        context_window = None
        supports_function_calling = True

        async def stream(self, messages, *, tools=None, **kwargs):
            yield LLMResponse(content="")   # no text, no tool calls -> nudge then invalid-reply fallback tail

    agent = Agent(
        "t", EmptyReplyLLM(), tools=[EchoTool()], output_guardrails=[SlowPass()],
        run_policy=RunPolicy(deadline_seconds=0.01),
    )
    received = []

    async def consume():
        async for piece in agent.astream_run("问"):
            received.append(piece)

    with pytest.raises(RunLimitExceeded, match="wall-clock time limit"):
        run(consume())
    assert received == []                       # the tail was gated by the deadline, never released
    assert agent.get_history() == []            # and the turn is not committed


def test_streaming_tool_loop_capability_opt_out_fails_fast():
    """An explicit supports_streaming_tools=False is rejected before a tool-loop request starts."""
    class OptOutLLM:
        provider = "test"
        context_window = None
        supports_function_calling = True
        supports_streaming_tools = False

        async def stream(self, messages, **kwargs):
            raise AssertionError("stream must not start")
            yield ""

    agent = Agent("t", OptOutLLM(), tools=[EchoTool()])
    with pytest.raises(LLMConfigError, match="does not support streaming tool calls"):
        run(collect(agent, "问"))


def test_streaming_tool_loop_accepts_kwargs_duck_stream():
    """A duck client whose stream takes tools via **kwargs is not misread as incapable."""
    class KwargsDuckLLM:
        provider = "test"
        context_window = None
        supports_function_calling = True

        def __init__(self):
            self.saw_tools = False

        async def stream(self, messages, **kwargs):
            self.saw_tools = kwargs.get("tools") is not None
            yield "答"
            yield LLMResponse(content="答")

    llm = KwargsDuckLLM()
    agent = Agent("t2", llm, tools=[EchoTool()])
    assert "".join(run(collect(agent, "问"))) == "答"
    assert llm.saw_tools


def test_streaming_tool_loop_requires_terminal_response():
    """A tool-bearing stream must end with an LLMResponse carrying the turn state."""
    class BrokenStreamLLM:
        provider = "test"
        context_window = None
        supports_function_calling = True

        async def stream(self, messages, *, tools=None, **kwargs):
            yield "partial"

    agent = Agent("t", BrokenStreamLLM(), tools=[EchoTool()])
    with pytest.raises(LLMResponseError, match="without the terminal LLMResponse"):
        run(collect(agent, "问", buffer_output=True))
    assert agent.get_history() == []


def test_after_model_fires_for_streaming_tool_turns():
    """Each terminal streaming tool-loop response fires after_model."""
    hook = RecordingHook()
    agent = Agent(
        "t", ScriptedLLM([ScriptedLLM.tool_call("echo_tool", {"text": "x"}), "done"]),
        tools=[EchoTool()], hooks=[hook],
    )
    assert "".join(run(collect(agent, "问"))) == "done"
    assert [name for name, *_ in hook.events].count("after_model") == 2


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
