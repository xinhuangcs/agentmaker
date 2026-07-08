"""RunPolicy run-governance regression (hermetic): the five limits + cooperative cancellation + __post_init__ validation + no-policy runs unaffected.

Fills a coverage gap (previously only the max_llm_calls dimension was tested). A loop LLM that always requests a tool keeps the run cycling until RunPolicy halts it.
"""

import time

import pytest

from agentmaker import Agent, RunPolicy
from agentmaker.core.exceptions import RunCancelled, RunLimitExceeded
from agentmaker.core.llm_response import LLMResponse
from agentmaker.tools import Tool, ToolResponse


class _LoopLLM:
    """Returns a noop tool call every time (keeps the agent looping) until RunPolicy halts it. Optional usage. Duck-types LLMClient."""
    model = "loop"
    provider = "test"
    supports_function_calling = True
    context_window = None

    def __init__(self, usage=None):
        self._usage = usage
        self.calls = 0

    async def chat(self, messages, *, tools=None, **kw):
        self.calls += 1
        return LLMResponse(content="", model="loop", usage=self._usage, tool_calls=[
            {"id": f"c{self.calls}", "type": "function", "function": {"name": "noop", "arguments": "{}"}}])


class _NoopTool(Tool):
    def __init__(self, *, sleep: float = 0.0):
        super().__init__("noop", "空操作")
        self._sleep = sleep
        self.calls = 0

    def get_parameters(self):
        return []

    def run(self, parameters):
        self.calls += 1
        if self._sleep:
            time.sleep(self._sleep)
        return ToolResponse.ok("ok")


def _agent(llm, policy, tool=None):
    return Agent("t", llm, tools=[tool or _NoopTool()], max_turns=50, run_policy=policy)


# ---------- The five limits ----------

def test_max_llm_calls():
    """Exceeding the LLM-call limit raises RunLimitExceeded (the loop halts at the (N+1)th check)."""
    llm = _LoopLLM()
    with pytest.raises(RunLimitExceeded, match="LLM call limit"):
        _agent(llm, RunPolicy(max_llm_calls=3)).run("go")
    assert llm.calls == 3                                   # halts after exactly 3 calls (the 4th check raises)


def test_max_tool_calls():
    """Exceeding the tool-call limit raises RunLimitExceeded."""
    tool = _NoopTool()
    with pytest.raises(RunLimitExceeded, match="tool call limit"):
        _agent(_LoopLLM(), RunPolicy(max_tool_calls=2), tool).run("go")
    assert tool.calls == 2                                  # halts after exactly 2 executions


def test_max_tool_calls_zero_readonly_mode():
    """max_tool_calls=0: halts the moment the model wants to run a tool (read-only / safe mode); pure Q&A is unaffected."""
    tool = _NoopTool()
    with pytest.raises(RunLimitExceeded, match="tool call limit"):
        _agent(_LoopLLM(), RunPolicy(max_tool_calls=0), tool).run("go")
    assert tool.calls == 0                                  # not executed even once
    # pure Q&A (no tool calls) + max_tool_calls=0 -> completes normally
    out = Agent("t", _ScriptOnce("直接答"), max_turns=5, run_policy=RunPolicy(max_tool_calls=0)).run("hi")
    assert out.final_output == "直接答"


def test_max_tokens():
    """Cumulative tokens (sum of each usage.total_tokens) over the limit raises RunLimitExceeded."""
    with pytest.raises(RunLimitExceeded, match="token limit"):
        _agent(_LoopLLM(usage={"total_tokens": 60}), RunPolicy(max_tokens=100)).run("go")


def test_deadline_seconds():
    """Exceeding the wall-time limit raises RunLimitExceeded (the tool sleep pushes time past the deadline)."""
    slow = _NoopTool(sleep=0.08)
    with pytest.raises(RunLimitExceeded, match="wall-clock time limit"):
        _agent(_LoopLLM(), RunPolicy(deadline_seconds=0.04), slow).run("go")


def test_cancel_hook():
    """Cooperative cancellation: cancel() returning True raises RunCancelled."""
    flag = {"n": 0}

    def cancel():
        flag["n"] += 1
        return flag["n"] >= 2                               # cancel on the 2nd check

    with pytest.raises(RunCancelled):
        _agent(_LoopLLM(), RunPolicy(cancel=cancel)).run("go")


def test_no_policy_unaffected():
    """No RunPolicy: the loop is bounded only by max_turns and never raises RunLimitExceeded (governance doesn't touch policy-less runs)."""
    out = Agent("t", _LoopLLM(), tools=[_NoopTool()], max_turns=3).run("go")
    assert out.final_output                                 # hitting max_turns returns the fallback text normally, no limit exception


# ---------- __post_init__ validation ----------

@pytest.mark.parametrize("kwargs", [
    {"max_llm_calls": 0}, {"max_llm_calls": -1}, {"max_llm_calls": True},   # must be >=1; bool isn't a valid int
    {"max_tokens": 0}, {"max_tokens": 1.5},                                 # must be an int >=1
    {"max_tool_calls": -1},                                                 # must be >=0
    {"deadline_seconds": 0}, {"deadline_seconds": -1}, {"deadline_seconds": True},   # must be >0
    {"cancel": 123},                                                        # must be callable
])
def test_post_init_rejects_invalid(kwargs):
    """Meaningless config raises ValueError at construction (not surfaced mid-run as a cryptic error)."""
    with pytest.raises(ValueError):
        RunPolicy(**kwargs)


def test_post_init_accepts_valid():
    """Valid config (including max_tool_calls=0 read-only semantics, None = unlimited) constructs fine."""
    RunPolicy(max_llm_calls=1, max_tool_calls=0, max_tokens=1, deadline_seconds=0.5, cancel=lambda: False)
    RunPolicy()                                             # all None = nothing limited


class _ScriptOnce:
    """Minimal fake LLM that returns one fixed string (pure Q&A, no tool calls)."""
    model = "test"
    provider = "test"
    supports_function_calling = True
    context_window = None

    def __init__(self, text):
        self._text = text

    async def chat(self, messages, *, tools=None, **kw):
        return LLMResponse(content=self._text, model="test")
