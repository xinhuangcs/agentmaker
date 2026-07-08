# Scope & async

Two framework-wide capabilities that shape how you run agentmaker in production. **Scope** is a label attached to every read and write that isolates retrieval, memory, and conversation sessions across users, agents, and apps that all share one backend. **Async-first** means agentmaker is asynchronous to the core: the coroutine is the real implementation of each capability, every outward capability has an `a*` counterpart, streaming is an `async for`, and the plain synchronous methods are thin facades over the async body. You will reach for Scope the moment more than one user (or more than one agent) shares a store, and for the async API whenever you run inside a web server or any other event loop.

## Scope

A `Scope` is an ownership label with five optional dimensions. Every store and index column carries these dimensions, so a read or write tagged with a Scope only ever touches rows that match it. `Scope` is imported straight from the top level:

```python
from agentmaker import Scope
```

It is a frozen (immutable, hashable) dataclass. Every field is optional and defaults to `None`:

| Field | Meaning |
| --- | --- |
| `base` | Subsystem distinction (memory / rag, etc.); leave empty to not restrict. |
| `user` | User identifier (the key to multi-user isolation, the minimal security boundary). |
| `agent` | Agent identifier (in a multi-agent system each agent keeps its own records). |
| `session` | Session identifier (equals `run_id`, the transient context of a single conversation). |
| `app` | Application / organization identifier (shared context). |

### Isolate memory and retrieval

This is the whole point of Scope: many tenants can share one backend and one index, yet each retrieves only its own data. The following runs with no API key and no network, exactly as shipped in [`examples/10_scope_isolation.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/10_scope_isolation.py):

```python
from agentmaker import Memory, MemoryStore, Scope
from agentmaker.retrieval import build_sqlite_hybrid
from agentmaker.testing import FakeEmbedder

# One shared store + index; the only thing separating the two users is their Scope.
store = MemoryStore()
index = build_sqlite_hybrid(FakeEmbedder())
alice = Memory(retriever=index, store=store, scope=Scope(base="memory", user="alice"))
bob = Memory(retriever=index, store=store, scope=Scope(base="memory", user="bob"))

alice.add("Alice loves tea")
bob.add("Bob loves coffee")

print("alice sees:", [h.content for h in alice.search("favorite drink", top_k=5)])
print("bob sees:  ", [h.content for h in bob.search("favorite drink", top_k=5)])
```

`alice` and `bob` write into the same `MemoryStore` and query the same hybrid index, but because each `Memory` was constructed with a different `Scope`, `alice.search(...)` never surfaces Bob's note and vice versa. There are no separate databases involved. See the [Memory](memory.md) guide for the `Memory` API itself.

### Filter semantics

Scope filters on non-empty dimensions only. A read adds a constraint for each dimension you gave a value, and leaves unset dimensions completely unrestricted. So `Scope(user="alice")` returns all of Alice's records regardless of which agent or session produced them, while an empty dimension simply means "match anything here." Every dimension, including `base`, is treated the same way: empty means unrestricted.

`base` distinguishes subsystems such as memory and RAG (retrieval-augmented generation, feeding retrieved documents into the prompt). By convention each upper layer passes it explicitly, which is why the example above uses `Scope(base="memory", user="alice")`: `Memory` works under `base="memory"` so its data never collides with a RAG store sharing the same file.

!!! note "The all-scopes guardrail"
    A fully empty `Scope()` restricts nothing, so it would match the entire database. For destructive or global operations the retrieval layer refuses a bare `Scope()` unless the caller explicitly opts in. If you build a custom retrieval backend, the helpers `scope_is_empty` and `require_explicit_scope` (both importable from `agentmaker.retrieval`) implement this check: `require_explicit_scope(scope, all_scopes, action)` raises unless `all_scopes=True` is passed for a scope that restricts no dimension. The framework's built-in memory and RAG always carry a non-empty `base`, so they are never blocked.

### Scope in agents (sessions)

The same label isolates conversation history. An `Agent` takes a default `scope=` at construction, and `run` / `arun` / `resume` / `stream_run` each accept a per-call `scope=` that overrides it. History is loaded and saved by scope, so a single `Agent` instance can serve many independent sessions:

```python
from agentmaker import Agent, Scope
from agentmaker.testing import ScriptedLLM

agent = Agent("assistant", ScriptedLLM(["Hi Alice.", "Hi Bob."]), scope=Scope(user="alice"))

agent.run("hello")                            # recorded under Scope(user="alice")
agent.run("hello", scope=Scope(user="bob"))   # a separate session on the same instance
```

When an `Agent` is given a `session_store`, that history is persisted per scope and reloaded per scope on each run, so a long-running daemon does not lose conversations on restart. Checkpoints for HITL (human-in-the-loop, pausing a run for approval) and crash recovery are likewise keyed by scope. Orchestration recipes that delegate to internal sub-agents run each sub-agent under a derived child scope, so a sub-agent's history and checkpoints never collide with its parent's. See [Guardrails & HITL](guardrails-and-hitl.md) for the suspend and resume flow.

## Async

Every outward capability exposes an `a*` twin of its synchronous method: agents expose `arun`, memory exposes `asearch` / `aadd` / `aupdate`, RAG exposes `aingest_text` / `aingest_file`, and so on. Token streaming lives one layer down on the LLM client as an async generator you consume with `async for`. The plain synchronous methods (`run`, `resume`, `stream_run`) are one-line facades that drive the async body to completion, so scripts and notebooks stay simple.

### Run an agent asynchronously

This runs with no API key and no network, exactly as shipped in [`examples/09_async.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/09_async.py):

```python
import asyncio

from agentmaker import Agent
from agentmaker.testing import ScriptedLLM


async def main():
    # The async twin of agent.run(...).
    agent = Agent("assistant", ScriptedLLM(["Hello from an async run."]))
    result = await agent.arun("hi")
    print("arun:", result.final_output)

    # Token streaming is exposed on the LLM client as an async generator.
    llm = ScriptedLLM(["streamed piece by piece"])
    chunks = [chunk async for chunk in llm.stream([{"role": "user", "content": "hi"}])]
    print("stream chunks:", chunks)


asyncio.run(main())
```

`await agent.arun("hi")` returns the same `RunResult` that the synchronous `agent.run("hi")` returns; `.final_output` holds the reply text. The streaming call yields text deltas piece by piece, which is why the list comprehension uses `async for`.

### The `a*` map

| Capability | Async form | Synchronous facade |
| --- | --- | --- |
| Run an agent | `agent.arun(...)` | `agent.run(...)` |
| Resume after HITL / crash | `agent.aresume(...)` | `agent.resume(...)` |
| Stream an agent's reply | `agent.astream_run(...)` | `agent.stream_run(...)` |
| Append to session history | `agent.add_messages(...)` | (async only) |
| Read session history | (async body is internal) | `agent.get_history(...)` |

The [LLM client](llm-clients.md) is async-native: `chat` is an async call and `stream` is an async generator, and there is no separate synchronous method on the client itself.

```python
resp = await llm.chat([{"role": "user", "content": "Hello"}])
print(resp.content)
async for piece in llm.stream([{"role": "user", "content": "Tell a joke"}]):
    print(piece, end="")
```

`llm.stream(...)` is an async generator of text deltas. Without tools it yields strings only; when tools are passed it additionally yields one final `LLMResponse` after the text drains (the channel the agent's streaming tool loop consumes). After a stream finishes you can read `llm.last_stream_stats` for that call's usage, latency, and finish reason.

Memory and RAG follow the same shape: the synchronous `search` / `add` shown in the [Scope example](#isolate-memory-and-retrieval) have async twins `asearch` / `aadd` / `aupdate`, and RAG ingestion has `aingest_text` / `aingest_file`. Use them from inside an event loop; use the synchronous forms from scripts. See [Memory](memory.md) and [Retrieval & RAG](retrieval-and-rag.md).

!!! note "Inside a running event loop, await the async form"
    The synchronous facades (`run`, `resume`, `stream_run`, `get_history`) drive the async body to completion, which cannot be done from within an already-running event loop. In an async application, Jupyter, or a FastAPI handler, call the `a*` method directly (`await agent.arun(...)`, `async for piece in agent.astream_run(...)`) instead of the synchronous facade.
