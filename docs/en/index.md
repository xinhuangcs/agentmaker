# agentmaker

A general-purpose Python framework for building LLM agents and multi-agent systems, with tools, memory, retrieval / RAG, context engineering, guardrails, human-in-the-loop, and observability built in.

## Install

```bash
pip install agentmaker            # core batteries, works out of the box
pip install "agentmaker[all]"     # every optional capability
```

Requires Python 3.12+. See [Installation](installation.md) for the full extras matrix.

## 30-second example

Runs with zero setup (no API key, no network), exactly as shipped in [`examples/01_quickstart.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/01_quickstart.py):

```python
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
```

To use a real model, replace `ScriptedLLM(...)` with `LLMClient("deepseek")` (or `"openai"` / `"anthropic"` / `"gemini"`) and set the matching API key in your environment; the model itself then decides when to call the tool.

## Where to go next

- New here? Start with the [Quickstart](guides/quickstart.md).
- Pick a capability from the **Guides**: [LLM clients](guides/llm-clients.md), [Agents & workflows](guides/agents.md), [Tools](guides/tools.md), [Structured output](guides/structured-output.md), [Memory](guides/memory.md), [Retrieval & RAG](guides/retrieval-and-rag.md), [Context engineering](guides/context-engineering.md), [Guardrails & HITL](guides/guardrails-and-hitl.md), [Observability](guides/observability.md), [Prompt registry](guides/prompts.md), [Skills](guides/skills.md).
- Look up an exact signature in the [API Reference](reference/core.md) (generated from source docstrings).

## Versioning

Pre-1.0: minor versions may introduce breaking changes, patch versions only fix. Pin `agentmaker>=0.1,<0.2`. See the [changelog](https://github.com/xinhuangcs/agentmaker/blob/main/CHANGELOG.md).

## License

[MIT](https://github.com/xinhuangcs/agentmaker/blob/main/LICENSE).
