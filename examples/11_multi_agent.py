"""Multi-agent: expose a sub-agent as a tool (orchestrator-worker).

AgentTool wraps an Agent so a coordinator can call it like any other tool and delegate a
sub-task to a specialist. Hermetic via ScriptedLLM.

    uv run python examples/11_multi_agent.py
"""
from agentmaker import Agent, AgentTool
from agentmaker.testing import ScriptedLLM

# The worker: a specialist sub-agent.
translator = Agent("translator", ScriptedLLM(["Bonjour le monde"]))

# The coordinator calls the worker through AgentTool, then composes the final answer.
coordinator = Agent("coordinator", ScriptedLLM([
    ScriptedLLM.tool_call("translate", {"task": "translate 'hello world' to French"}),
    "In French, 'hello world' is: Bonjour le monde.",
]), tools=[AgentTool(translator, name="translate", description="Translate text to French")])

print(coordinator.run("How do you say 'hello world' in French?").final_output)
