"""Declarative agents: describe an agent with AgentSpec, build it with build_agent.

AgentSpec is a plain config object (name / strategy / model / tools / ...); build_agent turns it
into a ready-to-run agent. Because build_agent resolves `model` to a real client, actually
building and running it needs an API key, so here we only construct and inspect the spec.

    uv run python examples/15_declarative_spec.py
"""
from agentmaker import AgentSpec, tool


@tool
def get_time() -> str:
    """Return the current time."""
    return "12:00"


# strategy is one of: "chat" / "react" / "plan" / "reflection".
spec = AgentSpec(name="helper", strategy="react", model="deepseek", tools=[get_time])
print(f"spec: name={spec.name!r} strategy={spec.strategy!r} "
      f"model={spec.model!r} tools={[t.name for t in spec.tools]}")

# To build and run it (needs the provider's API key in your environment):
#     from agentmaker import build_agent
#     agent = build_agent(spec)              # resolves model="deepseek" to a real LLMClient
#     print(agent.run("what time is it?").final_output)
print("build with: agent = build_agent(spec)  # needs DEEPSEEK_API_KEY to run")
