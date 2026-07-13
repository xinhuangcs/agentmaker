"""Hook dispatch and guardrail behavior.

Hooks: nine lifecycle events fire via Agent.run with correctly destructured args (captured with agentmaker.testing.RecordingHook).
Guardrails: CallableGuardrail sync / async dispatch, tripping raises GuardrailTripwireError, input vs output stages.
"""

import asyncio

import pytest

from agentmaker import Agent
from agentmaker.core.exceptions import GuardrailTripwireError
from agentmaker.runtime.guardrails import CallableGuardrail, GuardrailResult
from agentmaker.runtime.observability import Tracer
from agentmaker.runtime.harness import Harness, _validate_structured
from agentmaker.prompts import DEFAULT_PROMPTS
from agentmaker.testing import MemoryCheckpointStore, RecordingHook, ScriptedLLM
from agentmaker.tools import Tool, ToolParameter, ToolResponse
from pydantic import RootModel


class _EchoTool(Tool):
    def __init__(self, *, danger=False):
        super().__init__("echo", "回显")
        self.requires_confirmation = danger

    def get_parameters(self):
        return [ToolParameter("text", "string", "内容")]

    def run(self, parameters):
        return ToolResponse.ok(f"echo:{parameters.get('text')}")


# ---------- Hooks: 9-event dispatch ----------

def test_hooks_run_lifecycle_order():
    """Pure Q&A run: on_run_start -> before_model -> after_model -> on_run_end, with correct args."""
    hook = RecordingHook()
    Agent("t", ScriptedLLM(["答案"]), hooks=[hook]).run("问题")
    assert [e[0] for e in hook.events] == ["on_run_start", "before_model", "after_model", "on_run_end"]
    assert hook.events[0][1] == "问题" and hook.events[-1][1] == "答案"


def test_hooks_tool_events():
    """Tool run: before_tool / after_tool also fire, with the correct name."""
    hook = RecordingHook()
    llm = ScriptedLLM([ScriptedLLM.tool_call("echo", {"text": "hi"}), "完成"])
    Agent("t", llm, tools=[_EchoTool()], hooks=[hook]).run("回显")
    names = [e[0] for e in hook.events]
    assert "before_tool" in names and "after_tool" in names
    assert ("before_tool", "echo") in hook.events and ("after_tool", "echo") in hook.events


def test_hooks_on_interrupt():
    """A HITL suspension fires on_interrupt with the suspended tool's name."""
    hook = RecordingHook()
    llm = ScriptedLLM([ScriptedLLM.tool_call("echo", {"text": "x"})])
    Agent("t", llm, tools=[_EchoTool(danger=True)], checkpoint_store=MemoryCheckpointStore(), hooks=[hook]).run("删")
    assert ("on_interrupt", "echo") in hook.events


def test_hooks_on_error():
    """A non-guardrail exception in run fires on_error (with the exception type name) before propagating."""
    class _BoomLLM:
        model = "b"
        provider = "t"
        supports_function_calling = True
        context_window = None

        async def chat(self, messages, *, tools=None, **kw):
            raise ValueError("boom")

    hook = RecordingHook()
    tracer = Tracer()
    with pytest.raises(ValueError):
        Agent("t", _BoomLLM(), hooks=[hook], tracer=tracer).run("x")
    assert ("on_error", "ValueError") in hook.events
    event = next(e for e in tracer.events if e["type"] == "run_error")
    assert event["error_type"] == "ValueError" and event["message"] == "boom"
    assert event["run_id"] and event["step_index"] > 0


def test_nested_agent_failure_absorbed_by_parent_emits_no_run_error():
    """run_error is terminal: an AgentTool child whose LLM raises becomes a tool error, the parent completes, and no run_error event appears."""
    from agentmaker.agents.multi_agent.agent_tool import AgentTool

    class _ChildBoomLLM:
        model = "b"
        provider = "t"
        supports_function_calling = True
        context_window = None

        async def chat(self, messages, *, tools=None, **kw):
            raise ValueError("child boom")

    tracer = Tracer()
    child = Agent("worker", _ChildBoomLLM(), tracer=tracer)   # shared tracer, as in from_config setups
    llm = ScriptedLLM([ScriptedLLM.tool_call("worker", {"task": "干活"}), "汇总完成"])
    parent = Agent("parent", llm, tools=[AgentTool(child)], tracer=tracer)
    assert parent.run("去").final_output == "汇总完成"
    assert [e for e in tracer.events if e["type"] == "run_error"] == []


def test_run_error_trace_failure_preserves_original_exception():
    """A strict trace exporter cannot replace the run exception it was reporting."""
    class BrokenExporter:
        def export(self, event):
            raise RuntimeError("export failed")

        def close(self):
            pass

    class BrokenLLM:
        provider = "test"
        context_window = None
        supports_function_calling = True

        async def chat(self, messages, **kwargs):
            raise ValueError("model failed")

    hook = RecordingHook()
    tracer = Tracer(exporters=[BrokenExporter()], strict=True)
    with pytest.raises(ValueError, match="model failed") as exc_info:
        Agent("t", BrokenLLM(), tracer=tracer, hooks=[hook]).run("x")
    assert ("on_error", "ValueError") in hook.events
    assert any("Failed to emit run_error trace" in note for note in exc_info.value.__notes__)


def test_stream_trace_failure_preserves_original_exception():
    class BrokenExporter:
        def export(self, event):
            raise RuntimeError("export failed")

        def close(self):
            pass

    class BrokenStreamLLM:
        model = "broken"

        async def stream(self, messages, **kwargs):
            yield "partial"
            raise ValueError("stream failed")

    async def consume():
        harness = Harness(BrokenStreamLLM(), tracer=Tracer(
            exporters=[BrokenExporter()], strict=True))
        return [piece async for piece in harness.astream_llm([])]

    with pytest.raises(ValueError, match="stream failed") as exc_info:
        asyncio.run(consume())
    assert any("trace cleanup also failed" in note for note in exc_info.value.__notes__)


def test_structured_validation_accepts_root_json_array():
    """Pydantic root models accept array JSON, including a fenced response."""
    schema = RootModel[list[int]]
    plain, plain_error = _validate_structured(schema, "[1, 2]", DEFAULT_PROMPTS)
    fenced, fenced_error = _validate_structured(schema, "```json\n[3, 4]\n```", DEFAULT_PROMPTS)
    assert plain_error is None and plain is not None and plain.root == [1, 2]
    assert fenced_error is None and fenced is not None and fenced.root == [3, 4]


# ---------- Guardrails ----------

def test_input_guardrail_trips():
    """An input guardrail tripping (sync CallableGuardrail returns False) -> GuardrailTripwireError with message; on_guardrail_trip fires."""
    hook = RecordingHook()
    guard = CallableGuardrail(lambda t: "禁词" not in t, message="命中禁词")
    agent = Agent("t", ScriptedLLM(["不该到这"]), input_guardrails=[guard], hooks=[hook])
    with pytest.raises(GuardrailTripwireError, match="命中禁词"):
        agent.run("含禁词的输入")
    assert ("on_guardrail_trip", "input") in hook.events


def test_input_guardrail_passes():
    """An input guardrail that passes -> normal output (no false trip)."""
    guard = CallableGuardrail(lambda t: True)
    assert Agent("t", ScriptedLLM(["正常"]), input_guardrails=[guard]).run("ok").final_output == "正常"


def test_async_callable_guardrail():
    """CallableGuardrail given an async fn -> dispatched via acheck (no blocking in the event loop, not rejected)."""
    async def afn(text):
        return "bad" not in text
    guard = CallableGuardrail(afn, message="异步拦截")
    with pytest.raises(GuardrailTripwireError, match="异步拦截"):
        Agent("t", ScriptedLLM(["x"]), input_guardrails=[guard]).run("bad input")


def test_acheck_awaits_sync_signature_returning_awaitable():
    """acheck awaits a lambda or callable object whose synchronous signature returns an awaitable."""
    async def moderate(text):
        return "bad" not in text                                    # contains bad -> trips (passed=False)

    lam = CallableGuardrail(lambda t: moderate(t), message="lambda 异步拦截")   # lambda is sync, returns a coroutine
    assert asyncio.run(lam.acheck("bad input")).passed is False
    assert asyncio.run(lam.acheck("clean input")).passed is True

    class AsyncModerator:                                           # object with async __call__; its signature is judged sync too
        async def __call__(self, text):
            return "bad" not in text

    obj = CallableGuardrail(AsyncModerator(), message="对象异步拦截")
    assert asyncio.run(obj.acheck("bad input")).passed is False
    assert asyncio.run(obj.acheck("clean input")).passed is True


def test_guardrail_returning_result_object():
    """An fn returning a GuardrailResult directly (with its own message) -> that message is used."""
    guard = CallableGuardrail(lambda t: GuardrailResult(passed=False, message="自带说明"))
    with pytest.raises(GuardrailTripwireError, match="自带说明"):
        Agent("t", ScriptedLLM(["x"]), input_guardrails=[guard]).run("any")


def test_output_guardrail_checks_output():
    """An output guardrail checks the model output (not the input): a sensitive word in the output trips it."""
    guard = CallableGuardrail(lambda t: "密码" not in t, message="输出含敏感词")
    with pytest.raises(GuardrailTripwireError, match="输出含敏感词"):
        Agent("t", ScriptedLLM(["你的密码是 123"]), output_guardrails=[guard]).run("问")
