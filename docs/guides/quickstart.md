# Quickstart

This guide builds a working agent in a dozen lines: a function becomes a tool, a scripted test model stands in for a real LLM (so it runs with no API key and no network), and the agent runs one "model calls tools in a loop" turn and hands back a final answer. Read it first if you are new to agentmaker; every other guide assumes you have this shape in your head. It walks through [`examples/01_quickstart.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/01_quickstart.py) line by line, then shows how to swap the test model for a real provider.

## The whole program

This is the example verbatim. It has zero setup: no API key, no network. You can run it with:

```bash
uv run python examples/01_quickstart.py
```

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

Running it prints:

```text
It's sunny and 24C in Copenhagen today.
```

The rest of this page explains each piece.

## Define a tool with `@tool`

```python
@tool
def get_weather(city: str) -> str:
    """Return today's weather for a city.

    Args:
        city: The city name.
    """
    return f"{city}: sunny, 24C"
```

`@tool` turns a type-annotated function into a `Tool` object in one line. After decoration `get_weather` is no longer a plain function, it is a `Tool` you can hand to an agent. The decorator reads the function to build the schema the model needs (function calling is the mechanism that lets the LLM emit a structured "call this tool with these arguments" instruction):

- **Parameters, types, defaults, and required-ness** come from the signature. Here `city: str` becomes a required string parameter.
- **The tool description** comes from the first line of the docstring.
- **Parameter descriptions** come from the `Args:` section (or from `Annotated[...]` metadata if you use it).

Every parameter must have a type annotation. A missing annotation, a `*args` / `**kwargs` parameter, or a type that does not map to JSON (`str`, `int`, `float`, `bool`, `list`, `dict`, and their `Optional` / `Annotated` wrappers) raises `ToolRegistrationError` at registration time rather than failing silently later.

`@tool` also accepts keyword options for special tools, for example `@tool(requires_confirmation=True)` for a high-risk action that must pass a confirmation gate, or `@tool(supports_parallel=True)` for a read-only tool that may run concurrently with others in the same turn. See [Tools](tools.md) for the full set.

## Script the model with `ScriptedLLM`

```python
llm = ScriptedLLM([
    ScriptedLLM.tool_call("get_weather", {"city": "Copenhagen"}),
    "It's sunny and 24C in Copenhagen today.",
])
```

`ScriptedLLM` is a test double: it emits preset responses in call order instead of contacting a provider, so agent tests run with no cost and no network. It lives in `agentmaker.testing`, which is not part of the top-level public surface, so import it explicitly with `from agentmaker.testing import ScriptedLLM`.

Each entry in the script list is either:

- a plain `str`, which becomes a text reply, or
- an `LLMResponse`, which lets you control tool calls and other fields precisely.

`ScriptedLLM.tool_call(name, arguments)` is a helper that builds the second kind: an `LLMResponse` representing "the model requests calling `name(arguments)`", so you do not have to hand-craft the tool-call structure yourself.

So this script says: on the first turn, ask to call `get_weather(city="Copenhagen")`; on the second turn, once the tool result is back, answer with the final sentence. Each call consumes the next entry in order. If the agent asks for one more response than the script provides, `ScriptedLLM` raises `AssertionError` telling you how many entries are missing, which usually means the loop took a turn you did not expect.

!!! note "Why script the tool call?"
    With a real model, the LLM itself decides when to call `get_weather`. `ScriptedLLM` just lets you pin that decision so the test is deterministic. The agent's loop behavior is identical either way, which is what makes the test meaningful.

## Construct the `Agent`

```python
agent = Agent("assistant", llm, tools=[get_weather])
```

`Agent` is the framework's core execution primitive: one input goes in, the model runs a tool loop, and a reply comes out. The three arguments here are:

- `"assistant"`: the agent's name.
- `llm`: the LLM client (here the `ScriptedLLM` double; later a real `LLMClient`).
- `tools=[get_weather]`: the list of tools the model may call. This is the one-line convenience entry point; it accepts a list of `Tool` objects (including `@tool`-decorated functions).

There is no separate registration step: passing `tools` is enough. If you omit `tools`, the agent does plain question-answering with no tool loop. Useful extra keyword arguments (all optional) include `system_prompt=` to set the persona and `max_turns=` to cap how many model turns the loop may take (default `10`), which guards against a tool loop that never terminates. The full parameter list is in the [Agents & workflows](agents.md) guide.

## Run it and read `final_output`

```python
result = agent.run("What's the weather in Copenhagen?")
print(result.final_output)
```

`agent.run(...)` executes the loop and returns a `RunResult`. Behind that one call, the loop does this:

1. Send the user message to the model. The model replies with the scripted tool call `get_weather(city="Copenhagen")`.
2. The framework executes the tool, then feeds its result back to the model as a tool message.
3. Send again. This time the model replies with plain text and no tool call, so the loop ends and that text is the answer.

`RunResult` is a single envelope for the outcome rather than a bare string. Its main field is `final_output`, the completed run's answer (a string here, or a structured instance if you asked for [structured output](structured-output.md)). Other fields let you inspect the run:

- `result.status`: `"completed"` or `"interrupted"`.
- `result.interrupted`: a convenience boolean, `True` when the run suspended awaiting human approval (see [Guardrails & human-in-the-loop](guardrails-and-hitl.md)).
- `result.usage`: a `RunUsage` snapshot with `llm_calls`, `tool_calls`, and `total_tokens`.
- `result.new_messages`: the user and assistant messages added to history this turn.
- `result.run_id`: this run's trace correlation id.

For most simple cases you read `final_output` and move on:

```python
result = agent.run("What's the weather in Copenhagen?")
print(result.final_output)                 # the answer text
print(result.usage.tool_calls)             # 1 (get_weather ran once)
```

!!! note "Async counterpart"
    `agent.run(...)` is the synchronous entry point. The framework is async-first, so `await agent.arun(...)` is the async version and returns the same `RunResult`. Use it inside `async def` code; use `run` in plain scripts.

## Swap in a real model

The only line that changes is the LLM. Replace `ScriptedLLM(...)` with an `LLMClient`, and now the model itself decides when to call `get_weather`:

```python
from agentmaker import Agent, LLMClient, tool


@tool
def get_weather(city: str) -> str:
    """Return today's weather for a city.

    Args:
        city: The city name.
    """
    return f"{city}: sunny, 24C"


llm = LLMClient("deepseek")                 # reads DEEPSEEK_API_KEY from the environment
agent = Agent("assistant", llm, tools=[get_weather])
result = agent.run("What's the weather in Copenhagen?")
print(result.final_output)
```

`LLMClient(provider)` resolves that provider's configuration and reads its API key from your environment. The provider defaults to `"deepseek"`, and each cloud vendor has a default model filled in, so `LLMClient("deepseek")` needs nothing else. Set the matching key for whichever provider you pick:

| Call | Reads env var |
| --- | --- |
| `LLMClient("deepseek")` | `DEEPSEEK_API_KEY` |
| `LLMClient("openai")` | `OPENAI_API_KEY` |
| `LLMClient("anthropic")` | `ANTHROPIC_API_KEY` |
| `LLMClient("gemini")` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |

Pass `model=` to pick a specific model, for example `LLMClient("openai", model="gpt-4.1-nano")`. See [LLM clients & providers](llm-clients.md) for the full provider list, self-hosted and OpenAI-compatible endpoints, and per-call options.

Everything else stays the same: the `@tool` definition, the `Agent` construction, `run(...)`, and `result.final_output` all behave identically. That is the point of `ScriptedLLM`, your test and your production code exercise the same loop.

## Mount more capabilities

The agent above is deliberately minimal. Every other capability is a few more arguments to the same constructor, each one optional. Here is that agent given semantic long-term memory, a model-invoked skill library, retrieved context, and an input guardrail:

```python
from agentmaker import (Agent, LLMClient, Memory, MemoryStore, ContextBuilder,
                          CallableSource, SkillLoader, CallableGuardrail)
from agentmaker.retrieval import build_sqlite_hybrid, OpenAIEmbedder

llm = LLMClient("openai")
memory = Memory(build_sqlite_hybrid(OpenAIEmbedder()), MemoryStore())
skills = SkillLoader("./skills")

agent = Agent(
    "assistant", llm,
    tools=[get_weather],  # function calling
    sources=[CallableSource("memory", memory.search)],  # memory pulled into context each turn
    context_builder=ContextBuilder(),  # assemble context under a token budget
    system_prompt=f"You are a helpful assistant.\nSkills:\n{skills.catalog()}",  # model-invoked skills
    input_guardrails=[CallableGuardrail(lambda t: len(t) < 4000, message="input too long")],  # validate input
)
print(agent.run("Plan a day in Copenhagen, and remember I'm vegetarian.").final_output)
```

You add these one at a time, not all at once, and the same pattern reaches the rest of the framework:

- **More `sources=`**: RAG retrieval over an ingested corpus (chunking, query transforms, rank fusion, source citations) sits beside memory. See [Retrieval & RAG](retrieval-and-rag.md).
- **More `tools=`**: MCP servers via `MCPClient`, sub-agents via `AgentTool` for orchestrator-worker setups, and `ToolRetriever` to pick from a large toolset. See [Tools](tools.md).
- **Smarter memory**: `SmartWriter` extracts facts from a conversation and diffs them against what is already stored, then adds, updates, or deletes, instead of saving raw text. See [Memory](memory.md).
- **Other run modes**: [structured output](structured-output.md) via `run(..., output_type=Model)`, streaming with `async for` over `agent.stream(...)`, and the `PlanAgent` / `ReflectionAgent` recipes in [Agents & workflows](agents.md).
- **Persistence and safety**: sessions, checkpoints (human-in-the-loop), tool permissions, and history compaction. See [Guardrails & human-in-the-loop](guardrails-and-hitl.md) and [Context engineering](context-engineering.md).

## Debug it with an agent

For development, attach the trace-based agent debugger. A `Tracer` records every step of a run, and `DoctorHook` turns a failed run into an LLM-written diagnosis (first bad step, root cause, suggested fix) printed straight to your terminal:

```python
from agentmaker import Agent, Tracer
from agentmaker.devtools import DoctorHook

tracer = Tracer()
agent = Agent("assistant", llm, tools=[get_weather], tracer=tracer, hooks=[DoctorHook(tracer)])
print(agent.run("What's the weather in Copenhagen?").final_output)
```

`DoctorHook` and the standalone Trace Detective (`python -m agentmaker.devtools`, a local web UI over recorded runs) are themselves agentmaker agents, so the framework debugs its own runs. See [Observability](observability.md) for tracing, exporters, and the Trace Detective UI.

## Where to go next

- [LLM clients & providers](llm-clients.md): every provider, model selection, and streaming.
- [Agents & workflows](agents.md): the full `Agent` surface, plus plan-and-execute and reflection recipes.
- [Tools](tools.md): richer tools, confirmation gates, parallel execution, and tool registries.
- [Structured output](structured-output.md): return a validated object instead of text.
- [Guardrails & human-in-the-loop](guardrails-and-hitl.md): input and output guardrails, and approving high-risk actions.
- [Testing](testing.md): `ScriptedLLM` and the other test doubles for hermetic agent tests.
