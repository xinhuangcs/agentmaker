"""Observability: capture structured trace events (LLM calls, tool calls, timings).

Attach a Tracer with one or more exporters. MemoryExporter collects events in a list (for
in-process inspection); other exporters persist to JSONL / SQLite / OpenTelemetry. Hermetic.

    uv run python examples/13_observability.py
"""
from agentmaker import Agent, MemoryExporter, Tracer, tool
from agentmaker.testing import ScriptedLLM


@tool
def double(x: int) -> int:
    """Double a number.

    Args:
        x: The number to double.
    """
    return x * 2


exporter = MemoryExporter()
tracer = Tracer(exporters=[exporter])

agent = Agent("assistant", ScriptedLLM([
    ScriptedLLM.tool_call("double", {"x": 21}),
    "The answer is 42.",
]), tools=[double], tracer=tracer)
agent.run("double 21")

print("captured trace events:")
for event in exporter.events:
    print("  -", event.get("type"))
