# Testing

`agentmaker.testing` gives you deterministic test doubles that stand in for the parts of an agent that would otherwise cost money or reach the network: the LLM, the embedder, the checkpoint store, and lifecycle hooks. Swap them in and your agent tests run hermetically (no API key, no network, no flakiness), so you can assert on exactly what your agent does with a scripted set of model responses. Reach for this module whenever you write unit tests for agents, tools, human-in-the-loop flows, or retrieval wiring built on this framework.

These utilities are not re-exported from the top-level `agentmaker` namespace. Import them directly from the submodule:

```python
from agentmaker.testing import ScriptedLLM, FakeEmbedder, MemoryCheckpointStore, RecordingHook
```

The module defines exactly four public doubles:

| Double | Replaces | Use it to test |
| --- | --- | --- |
| `ScriptedLLM` | the LLM client | plain chat, tool loops, streaming, all decisions scripted |
| `FakeEmbedder` | an `Embedder` | retrieval wiring with deterministic vectors |
| `MemoryCheckpointStore` | a `CheckpointStore` | HITL suspend / resume and crash recovery, in memory |
| `RecordingHook` | a `Hook` | that lifecycle events fired in the expected order |

## ScriptedLLM

`ScriptedLLM` emits preset responses in call order. It is duck-typed to the LLM client contract (it exposes `provider`, `model`, `supports_function_calling`, `context_window`, `chat`, and `stream`) but does not inherit the real client, so constructing it triggers no API-key validation and no network call. Because the script fully determines behavior, `chat` ignores the incoming `messages` and `tools`. `stream` ignores `messages` too, but when `tools` are passed it additionally yields one terminal `LLMResponse` after the text, mirroring how the real adapters signal the tool-loop channel.

Pass a list of script entries to the constructor. Each entry is either a plain `str` (a text reply) or a ready-made `LLMResponse` (for precise control over tool calls, usage, or finish reason):

```python
from agentmaker import Agent
from agentmaker.testing import ScriptedLLM

agent = Agent("assistant", ScriptedLLM(["Hello.", "Goodbye."]))
assert agent.run("hi").final_output == "Hello."
assert agent.run("bye").final_output == "Goodbye."
```

`agent.run(...)` returns a `RunResult`. Its `final_output` field holds the completed answer (see [Asserting on a run](#asserting-on-a-run) below).

The constructor signature is:

```python
ScriptedLLM(script=None, *, model="test", provider="test",
            supports_function_calling=True, context_window=None)
```

- `supports_function_calling` is the model capability flag. A tool-enabled agent validates against it at construction time, so pass `False` to test the no-function-calling path (constructing an `Agent` with tools against such a client raises `ValueError`).
- `context_window` is left as `None` (unknown) by default, which triggers no window-budget reduction. Pass a concrete integer to exercise context-window budgeting.

### Scripting a tool call

To make the scripted model "decide" to call a tool, build the tool-call response with the `ScriptedLLM.tool_call` helper instead of hand-writing the underlying tool-calls structure:

```python
ScriptedLLM.tool_call(name, arguments=None, *, call_id="call_1", content="") -> LLMResponse
```

A typical tool loop scripts two entries: first the model asks to call the tool, then it writes the final answer. This is the shipped [`examples/01_quickstart.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/01_quickstart.py):

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

The same pattern works when the tools come from a [`ToolRegistry`](tools.md), including built-ins. Note that the name you pass to `tool_call` is the tool's registered name (here the built-in calculator is `"calculator"`), taken from [`examples/02_tools_and_registry.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/02_tools_and_registry.py):

```python
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
```

### Counting calls

`ScriptedLLM` tracks how many script entries have been consumed in its `calls` attribute, which is handy for asserting that a tool loop took exactly the number of model turns you expect (one to request the tool, one to answer):

```python
llm = ScriptedLLM([ScriptedLLM.tool_call("get_weather", {"city": "Oslo"}), "Sunny."])
agent = Agent("assistant", llm, tools=[get_weather])
agent.run("weather in Oslo?")
assert llm.calls == 2
```

### Running out of script

If the agent asks for one more response than the script provides, `ScriptedLLM` raises `AssertionError` (reporting which call number ran dry and how many entries the script had), rather than silently returning something unexpected. This turns "my loop iterated more than I thought" into a loud, immediate test failure:

```python
import pytest

agent = Agent("assistant", ScriptedLLM(["only one reply"]))
agent.run("a")
with pytest.raises(AssertionError, match="script exhausted"):
    agent.run("b")
```

### Streaming

`ScriptedLLM.stream` yields the next response's content in small slices, mirroring the real adapter contract. Drive it through the agent's synchronous streaming facade and reassemble the pieces:

```python
agent = Agent("assistant", ScriptedLLM(["A streamed reply."]))
pieces = list(agent.stream_run("hi"))
assert "".join(pieces) == "A streamed reply."
```

Empty content yields no chunk at all (an empty string produces an empty stream, not a single `""`). On completion the double records the response's usage and finish reason on `last_stream_stats`, matching how real adapters report streaming statistics. To assert on streamed usage or finish reason, script a full `LLMResponse` instead of a bare string:

```python
from agentmaker.core.llm_response import LLMResponse

llm = ScriptedLLM([LLMResponse(content="hello world", model="test",
                               finish_reason="stop", usage={"total_tokens": 5})])
```

!!! note
    `LLMResponse` lives in `agentmaker.core.llm_response`. You only need it when you want to pin usage, `finish_reason`, or a precise `tool_calls` payload; for ordinary text replies, a plain string script entry is enough.

## Asserting on a run

Every scripted run comes back as a `RunResult`, so your assertions read the same fields you would read in production. The ones you assert on most:

- `final_output`: the completed answer (a `str`, or a structured instance when the agent was given an output schema); `None` when the run is suspended for approval.
- `status`: `"completed"` or `"interrupted"`.
- `usage`: a `RunUsage` snapshot with `llm_calls`, `tool_calls`, and `total_tokens`.

See [Agents & workflows](agents.md) for the full field list (`interrupt`, `new_messages`, `run_id`, and the rest).

```python
result = agent.run("What's the weather in Copenhagen?")
assert not result.interrupted
assert result.final_output == "It's sunny and 24C in Copenhagen today."
assert result.usage.tool_calls == 1
```

## FakeEmbedder

`FakeEmbedder` is a deterministic, offline `Embedder`: the same text always maps to the same vector, and different texts map to different vectors (each vector is derived from a SHA-256 hash and L2-normalized, so cosine similarity stays meaningful and retrieval can genuinely tell entries apart). Use it to test retrieval and RAG wiring without calling a real embedding API.

```python
from agentmaker.testing import FakeEmbedder

emb = FakeEmbedder(dim=8)
assert emb.dim == 8
assert emb.model_id == "fake-embedder-8"

# Same text yields the same vector; different text yields a different one.
assert emb.embed(["cat"]) == emb.embed(["cat"])
assert emb.embed(["dog"]) != emb.embed(["cat"])
```

The constructor takes the vector width, `FakeEmbedder(dim=8)`. `embed(texts)` returns one vector per input text, and the double exposes the `dim` and `model_id` properties the retrieval stack expects, so you can drop it straight into the components described in [Retrieval & RAG](retrieval-and-rag.md).

## MemoryCheckpointStore

`MemoryCheckpointStore` is an in-process `CheckpointStore` that keeps each run's execution state in a plain dictionary instead of on disk. It exists so you can test human-in-the-loop suspend / resume and crash recovery without touching the filesystem. Construct it with no arguments and pass it to the agent as `checkpoint_store`.

When the agent tries to run a tool that requires confirmation, the run suspends and returns a `RunResult` with `interrupted == True`. Approve it by calling `resume(True, scope=...)` with the scope carried on the interrupt:

```python
from agentmaker import Agent, Tool, ToolParameter, ToolResponse
from agentmaker.testing import MemoryCheckpointStore, ScriptedLLM


class DeleteTool(Tool):
    requires_confirmation = True   # high-risk: routes through the HITL confirmation gate

    def __init__(self):
        super().__init__("delete", "Delete a path")

    def get_parameters(self):
        return [ToolParameter("path", "string", "Target path")]

    def run(self, parameters):
        return ToolResponse.ok(f"deleted {parameters.get('path')}")


llm = ScriptedLLM([ScriptedLLM.tool_call("delete", {"path": "/tmp/a"}), "Done."])
agent = Agent("assistant", llm, tools=[DeleteTool()],
              checkpoint_store=MemoryCheckpointStore())

result = agent.run("delete /tmp/a")
assert result.interrupted
assert result.interrupt.pending.tool_name == "delete"

resumed = agent.resume(True, scope=result.interrupt.scope)
assert resumed.final_output == "Done."
```

See [Guardrails & HITL](guardrails-and-hitl.md) for the full approval model, including rejecting actions and approving several pending calls at once.

## RecordingHook

`RecordingHook` is a `Hook` that appends every lifecycle event it receives to its `events` list, as `(event_name, key_param)` tuples. Use it to assert that hook dispatch happened in the order you expect. Pass it to the agent via `hooks=[...]`:

```python
from agentmaker import Agent
from agentmaker.testing import RecordingHook, ScriptedLLM

hook = RecordingHook()
Agent("assistant", ScriptedLLM(["Answer."]), hooks=[hook]).run("Question?")

names = [name for name, _ in hook.events]
assert names[0] == "on_run_start"
assert "before_model" in names
assert "after_model" in names
assert names[-1] == "on_run_end"
assert hook.events[-1][1] == "Answer."   # on_run_end carries the final output
```

`RecordingHook` records the run-level events (`on_run_start`, `on_run_end`, `on_interrupt`, `on_error`), the model events (`before_model`, `after_model`), the tool events (`before_tool`, `after_tool`), and `on_guardrail_trip`. The second element of each tuple is a small key parameter for that event (for example the input text for `on_run_start`, the tool name for `before_tool`, or the final output for `on_run_end`), which lets you assert on what the event carried without wiring up a full mock. For what these events observe in production, see [Observability](observability.md).

## Running your tests

Because all four doubles are hermetic, a suite built on them needs no API key and no network access, so it runs the same on a laptop or in CI. Start from the [Quickstart](quickstart.md) if you have not built an agent yet, then wrap each behavior you care about in an assertion using the doubles above.
