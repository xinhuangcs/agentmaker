"""RunPolicy: cap how much a single run may consume.

Bound the number of LLM calls, tool calls, total tokens, or wall-clock seconds; exceeding any
limit raises RunLimitExceeded and aborts the run (useful as a budget / safety guard). Hermetic
via a ScriptedLLM that keeps asking to call a tool.

    uv run python examples/14_run_policy.py
"""
from agentmaker import Agent, RunLimitExceeded, RunPolicy, tool
from agentmaker.testing import ScriptedLLM


@tool
def noop() -> str:
    """A tool that does nothing."""
    return "ok"


# The model keeps requesting the tool; the policy stops the run after 3 LLM calls.
looping = ScriptedLLM([ScriptedLLM.tool_call("noop", {}) for _ in range(10)])
agent = Agent("assistant", looping, tools=[noop], max_turns=50,
              run_policy=RunPolicy(max_llm_calls=3))

try:
    agent.run("keep going forever")
except RunLimitExceeded as e:
    print("stopped by policy:", e)
