"""Quickstart: a single-loop Agent that calls a tool, then answers.

Runs with zero setup (no API key, no network): it uses ScriptedLLM, a test double that
returns preset responses in order. To use a real model, replace ScriptedLLM(...) with
LLMClient("deepseek") (or "openai" / "anthropic" / "gemini") and set the matching API key
in your environment; then the model itself decides when to call the tool.

    uv run python examples/01_quickstart.py
"""
from agentmaker import Agent, tool
from agentmaker.testing import ScriptedLLM


@tool
def get_weather(city: str) -> str:
    """Return today's weather for a city.

    Args:
        city: The city name.
    """
    return f"{city}: sunny, 24C"


# With a real model the LLM decides when to call the tool. Here we script that decision:
# first it asks to call get_weather(city="Copenhagen"), then it writes the final answer.
llm = ScriptedLLM([
    ScriptedLLM.tool_call("get_weather", {"city": "Copenhagen"}),
    "It's sunny and 24C in Copenhagen today.",
])

agent = Agent("assistant", llm, tools=[get_weather])
result = agent.run("What's the weather in Copenhagen?")
print(result.final_output)
