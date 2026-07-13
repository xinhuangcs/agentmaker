# Agents & workflows

An agent takes one input, optionally calls tools in a loop, and returns a reply. agentmaker gives you a single execution primitive for that (`Agent`), two workflow recipes for tasks that need a fixed shape (`PlanAgent`, `ReflectionAgent`), and a declarative way to describe any of them (`AgentSpec` + `build_agent`). Every agent strategy returns the same envelope, `RunResult`, so the calling code reads the outcome the same way regardless of which one produced it. A separate adapter, `AgentTool`, hands one agent to another as a tool; because it is a tool, it returns a `ToolResponse` to the orchestrator rather than a `RunResult`.

## The unified loop

`Agent` is the core "model calls tools in a loop" primitive. Each turn the messages go to the model with the tool schemas attached, and the model either answers with text (the loop ends) or asks to call tools. The framework runs the tools, feeds the results back, and calls the model again, until it answers or the turn budget runs out. With no tools registered, the first turn is terminal, which is plain question answering.

This one loop covers both the "chat" and "react" usages. ReAct (reason then act, the model writes its reasoning before each tool call) is just a preset of the same loop, described later under [Declarative construction](#declarative-construction).

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

`ScriptedLLM` is a test double from `agentmaker.testing`: it replays a fixed list of replies so agent code runs with no API key and no network. Swap it for `LLMClient("deepseek")` (or `"openai"` / `"anthropic"` / `"gemini"`) to use a real model, which then decides on its own when to call the tool. See [LLM clients](llm-clients.md).

### Constructing an Agent

The two most common arguments beyond `name` and `llm`:

| Argument | What it does |
| --- | --- |
| `system_prompt` | The persona. Optional; when omitted, no system message is sent. |
| `tools` | Convenience entry point: a `list[Tool]` (including `@tool`-decorated functions) or a `ToolRegistry`. Passing it enables the tool loop. |
| `tool_registry` | The advanced form of `tools` (reuse or customize a registry). Mutually exclusive with `tools`; passing both raises `ValueError`. |
| `max_turns` | Maximum model turns in the loop (default 10, must be a positive integer). Caps runaway tool calling. |

`Agent` keeps multi-turn history automatically, isolated per `scope`, so one instance can serve many sessions. Attach a `session_store` to persist that history across restarts. See [Scope & async](scope-and-async.md) for how scope keys sessions, and [Tools](tools.md) for building tools and registries.

Many more optional arguments (`confirm`, `permissions`, `checkpoint_store`, `context_builder`, `tool_retriever`, guardrails, hooks, and so on) wire in cross-cutting capabilities that have their own guides: [Guardrails & HITL](guardrails-and-hitl.md), [Context engineering](context-engineering.md), [Memory](memory.md), and [Observability](observability.md).

### Structured output

Pass an `output_schema` (a Pydantic model) to `run` / `arun` and the returned `RunResult.final_output` holds a validated instance instead of text. This runs without tools. See [Structured output](structured-output.md) for the full pattern.

### Streaming

`Agent.stream_run(input_text, ...)` yields the reply piece by piece (its async counterpart is `astream_run`, an `async for`). When the agent has tools and the model adapter supports native streaming tool calls, the streaming loop runs the same turn structure and streams each turn's text as it arrives. Text tool emulation is non-streaming: an agent built with `LLMClient(..., emulate_tools=True)` raises `LLMConfigError` if asked to run a streaming tool loop, so use `run` / `arun` for that configuration.

By default, pieces are emitted immediately and the output guardrail runs after the stream completes. Set `buffer_output=True` to hold the complete output until it passes the guardrail.

!!! note
    The streaming loop does not support human-in-the-loop suspend/resume or checkpoints. A tool that requires confirmation falls back to its synchronous confirm callback. Use `run` / `arun` when you need suspend semantics (see [Return types](#return-types) and [Guardrails & HITL](guardrails-and-hitl.md)).

## Workflow recipes

Where `Agent` lets the model decide each next step, the workflow recipes fix the order of stages in code. Both are built on the same single-loop `Agent` under the hood, so they take the same LLM and the same tools.

The example below is hermetic (`ScriptedLLM` stands in for what a real model would generate) and runs as shipped:

```python
from agentmaker import PlanAgent, ReflectionAgent
from agentmaker.testing import ScriptedLLM

# Reflection: draft -> critique -> refine, looping until the critic replies "GOOD ENOUGH"
# (the default English pass signal; the Chinese pack uses a Chinese one).
reflection = ReflectionAgent("writer", ScriptedLLM([
    "The Earth orbits the Sun.",                              # draft
    "Add that one orbit takes about 365 days.",              # critique
    "The Earth orbits the Sun once every ~365 days.",        # refine
    "GOOD ENOUGH",                                            # critique -> pass, stop
]), max_turns=3)
print("Reflection:", reflection.run("Explain Earth's orbit in one sentence.").final_output)

# Plan-and-Solve: break the task into an ordered plan, execute each step, then synthesize.
plan = PlanAgent("solver", ScriptedLLM([
    '{"steps": ["Name the capital of Denmark", "State its approximate population"]}',  # plan (structured)
    "The capital of Denmark is Copenhagen.",                  # step 1 execution
    "Copenhagen has roughly 660,000 residents.",             # step 2 execution
    "Copenhagen is Denmark's capital, home to about 660,000 people.",  # synthesis
]))
print("Plan:", plan.run("Tell me about Denmark's capital.").final_output)
```

### PlanAgent

Plan-and-Solve works out the whole plan first and then executes it, which suits long-range, multi-step tasks that need strong goal consistency. It runs three stages in a fixed order: the model breaks the problem into an ordered list of sub-task steps (via structured output), each step is executed in turn (delegated to an internal single-loop `Agent`), and the per-step results are synthesized into the final answer.

```python
PlanAgent(name, llm, system_prompt=None, *, tool_registry=None, max_turns=3, ...)
```

Pass a `tool_registry` and every execution step can call tools; without it, each step is pure reasoning. Note that `max_turns` here (default 3) is the tool-loop cap for each sub-step executor, not the number of plan steps.

### ReflectionAgent

Reflection repeatedly polishes an answer: the model writes a draft, then loops `reflect -> refine` (self-critique, then revise based on the critique) until the critique signals that no further change is needed or the round cap is reached.

```python
ReflectionAgent(name, llm, system_prompt=None, *, max_turns=3, tool_registry=None, ...)
```

Here `max_turns` (default 3) is the maximum number of reflect-refine rounds. The default pass signal is `GOOD ENOUGH` (the critique writes it when the answer is good enough to stop). If you pass a `tool_registry`, the critique step can call tools to verify facts or arithmetic; without one, critique is pure self-judgment.

!!! note
    Unlike `Agent`, `PlanAgent` and `ReflectionAgent` accept only `tool_registry` (keyword-only), not the `tools` list convenience. Build a registry first (see [Tools](tools.md)) and pass it in.

When a step or critique calls a high-risk tool and a `checkpoint_store` is attached, that inner run suspends for approval and the interrupt propagates up through the recipe; resuming continues from that point. See [Guardrails & HITL](guardrails-and-hitl.md).

## Declarative construction

Instead of calling a constructor, you can describe an agent with `AgentSpec` (a plain config dataclass) and build it with `build_agent`. The two forms coexist; the declarative one is a convenience on top of the imperative constructors.

```python
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
```

`strategy` selects the paradigm and `build_agent` returns the matching instance:

- `"chat"` builds the single-loop `Agent` (default, `max_turns` default 10).
- `"react"` is the same `Agent` as a ReAct preset: it requires at least one tool (building it with none raises `ValueError`), defaults `max_turns` to 5, and adds a think-before-acting system prompt.
- `"reflection"` builds a `ReflectionAgent` (`max_turns` default 3).
- `"plan"` builds a `PlanAgent` (`max_turns` default 3).

Key fields:

- `model`: a `"provider:model"` string (for example `"deepseek:deepseek-v4-flash"`), a bare provider name like `"deepseek"` (uses that provider's default model), an `LLMClient` instance, a duck-typed client exposing chat/stream (such as `ScriptedLLM`, so a declaratively-built agent can be tested hermetically), or `None` for the default client.
- `instructions`: becomes the agent's `system_prompt`.
- `tools`: a `list[Tool]` or a `ToolRegistry`.
- `max_turns`: one unified turn limit mapped onto each strategy's cap; `None` uses that strategy's default.

`AgentSpec` fields are a superset across strategies. A field that a strategy does not support, set to a non-default value, makes `build_agent` raise `ValueError` rather than silently ignoring it (for example `compactor` is rejected by the `plan` and `reflection` strategies).

## Multi-agent orchestration

The orchestrator-worker pattern has a main agent delegate sub-tasks to specialist agents and keep control of the conversation. `AgentTool` implements it by adapting an agent into a `Tool`, so the main agent delegates a sub-task exactly like calling any other tool. The sub-agent carries its own independent history and tools, so its context stays isolated.

```python
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
```

`AgentTool(agent, *, name=None, description=None, scope=None, prompts=None)` wraps any agent. `name` defaults to `agent.name`; `description` tells the orchestrating model what the sub-agent is good at and when to delegate to it. The tool exposes a single `task` string parameter, the self-contained sub-task handed to the sub-agent.

With the default `scope=None`, delegation inherits the parent run's current scope, so one `AgentTool` instance can serve multiple parent sessions without mixing the sub-agent's history. Pass an explicit `scope` only to pin the sub-agent to a fixed ownership scope. Parallel branches should still use separate sub-agent instances because one agent object has one mutable execution/history surface.

!!! note
    A sub-agent invoked through `AgentTool` cannot suspend for human approval mid-delegation. If it hits a high-risk action, the delegation returns an error result telling the orchestrator the sub-task needs human approval and cannot be completed this way, so the orchestrator can reroute. Keep high-risk actions in the main flow, or use `PlanAgent`, which does propagate nested suspends. See [Guardrails & HITL](guardrails-and-hitl.md).

## Return types

`run`, `arun`, `resume`, and `aresume` on every agent return one `RunResult`. A run has two terminal states, captured by `RunStatus = Literal["completed", "interrupted"]`: it either produced a final output, or a high-risk action is suspended awaiting human approval (human-in-the-loop, HITL). The interrupted state is explicit rather than a bare value you might mistake for the answer.

```python
r = agent.run("...")
if r.interrupted:
    handle(r.interrupt)            # HITL: take the suspended state to resume
else:
    use(r.final_output)            # completed: take the final output (str or structured instance)
```

`RunResult` is a frozen dataclass with these fields:

- `final_output`: the completed output (a `str`, or a structured instance when `output_schema` was passed); `None` when suspended.
- `status`: `"completed"` or `"interrupted"`.
- `interrupt`: the suspended state's `Interrupt` (the pending action plus the resume scope); `None` when completed.
- `usage`: a `RunUsage` snapshot of this run.
- `new_messages`: the messages added to history this turn (user + assistant); empty when suspended.
- `run_id`: this run's trace correlation id.

`RunResult.interrupted` is a convenience property (`status == "interrupted"`). Calling `str(result)` gives the final output text directly (a suspended result shows a readable note instead of a bare `None`).

`RunUsage` is a frozen snapshot for cost accounting and limit observability, with three fields: `llm_calls`, `tool_calls`, and `total_tokens` (all accumulated over the run).

For the `Interrupt` object and the `resume(decision)` flow that continues a suspended run, see [Guardrails & HITL](guardrails-and-hitl.md).
