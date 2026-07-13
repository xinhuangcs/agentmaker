"""Built-in tools plus a custom @tool, wired through a ToolRegistry.

A ToolRegistry holds the tools an agent may call. Built-ins (CalculatorTool, SearchTool,
CLITool, NotesTool, ...) and your own @tool functions register the same way. Hermetic
(no key / no network) via ScriptedLLM.

    uv run python examples/02_tools_and_registry.py
"""
from agentmaker import Agent, CalculatorTool, ToolRegistry, tool
from agentmaker.testing import ScriptedLLM


@tool
def to_upper(text: str) -> str:
    """Uppercase a string.

    Args:
        text: The input text.
    """
    return text.upper()


registry = ToolRegistry()
registry.register(CalculatorTool())   # built-in: safe arithmetic evaluation
registry.register(to_upper)           # your custom tool

# Script the model's decision to call the calculator, then its final answer.
llm = ScriptedLLM([
    ScriptedLLM.tool_call("calculator", {"expression": "(3 + 4) * 5"}),
    "The result is 35.",
])
agent = Agent("assistant", llm, tool_registry=registry)
print(agent.run("Compute (3 + 4) * 5").final_output)
