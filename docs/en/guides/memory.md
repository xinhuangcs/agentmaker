# Memory

Memory gives an agent durable facts that survive across conversations: what the user is allergic to, where they live, how they like their coffee. agentmaker ships two complementary memory types. `Memory` is semantic memory, where you write free-form facts and recall them by meaning. `KVMemory` is key-value memory, where you write structured facts under an exact key and read them back verbatim. Reach for `Memory` when a fact is fuzzy and you want the most relevant ones back ("what food should I avoid"); reach for `KVMemory` when a fact is definite and single-valued (`location = Beijing`).

## Quickstart

`Memory` pairs a source-of-truth store (`MemoryStore`) with a retrieval index built by [Retrieval & RAG](retrieval-and-rag.md). An embedder turns text into a vector so that similar meanings sit close together; the snippet below uses `FakeEmbedder`, a deterministic offline stand-in, so it runs with no API key and no network. This is [`examples/04_memory.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/04_memory.py) verbatim:

```python
from agentmaker import Memory, MemoryStore
from agentmaker.retrieval import build_sqlite_hybrid
from agentmaker.testing import FakeEmbedder

memory = Memory(retriever=build_sqlite_hybrid(FakeEmbedder()), store=MemoryStore())

memory.add("I am allergic to peanuts")
memory.add("I like oat milk in the evening")
memory.add("I work as a backend engineer")

# Note: FakeEmbedder is a deterministic hash-based stand-in, so ranking is stable but NOT
# semantic. With a real embedder (OpenAIEmbedder), the allergy fact would rank on top here.
print("Top matches for 'what food should I avoid':")
for hit in memory.search("what food should I avoid", top_k=2):
    print("  -", hit.content)
```

In production, swap `FakeEmbedder()` for `OpenAIEmbedder()` (which needs `OPENAI_API_KEY`) and the ranking becomes genuinely semantic.

`search` returns a list of `RetrievalResult`, ordered best-first. Each carries the `content`, a combined `score`, the source `id`, and a `metadata` dict:

```python
for hit in memory.search("what food should I avoid", top_k=2):
    print(hit.content, hit.score, hit.metadata["final"])
```

## How search ranks

`search` does not rank by relevance alone. Inspired by the Generative Agents retrieval model, the final score is a weighted sum of three components, each normalized to a 0..1 range:

- **relevance**: how well the memory matches the query, from hybrid retrieval (vector similarity plus keyword search, fused).
- **recency**: how fresh the memory is, decayed by a half-life (a more recent memory approaches 1, an older one approaches 0).
- **importance**: the memory's own `importance` value (0..1), set when you write it.

The three weights, the half-life, and the default result count all live in `MemoryConfig` and default to a sensible baseline (all weights `1.0`, `recency_halflife_hours=72.0`, `search_top_k=5`). Pass a config at construction to retune globally, or override per call as keyword arguments to `search`:

```python
from agentmaker import Memory, MemoryStore, MemoryConfig
from agentmaker.retrieval import build_sqlite_hybrid
from agentmaker.testing import FakeEmbedder

memory = Memory(
    retriever=build_sqlite_hybrid(FakeEmbedder()),
    store=MemoryStore(),
    config=MemoryConfig(recency_halflife_hours=24, importance_weight=2.0),
)

# per-call override wins over the config for this one search
hits = memory.search("coffee", top_k=3, recency_weight=0.0)
```

Setting all three weights to `0` degrades to pure relevance ranking. Each returned result exposes its component scores under `hit.metadata` as `relevance`, `recency`, `importance`, and `final`, which is useful when tuning weights.

!!! note "Hard filters vs soft ranking"
    The three weights are a soft ranking: they reorder, they do not exclude. To hard-filter candidates before ranking (for example by a metadata field), pass `filters=` a list of `MetadataFilter`. See [Retrieval & RAG](retrieval-and-rag.md) for the filter contract and the columns a backend must declare.

## A single memory: MemoryItem

`add` returns the `MemoryItem` it stored, and `search` results map back to one. Its fields:

| Field | Meaning |
| --- | --- |
| `content` | The memory body text. |
| `id` | Identifier within one complete ownership scope; auto-generated as a uuid unless you set it. |
| `type` | A free-form label (defaults to `"semantic"`); not enforced, purely for your own grouping. |
| `importance` | Importance in 0..1 (defaults to `0.5`); feeds the importance score and `forget`. |
| `created_at` | When the memory was recorded. |
| `updated_at` | Time of the last content edit; recency decays from here once edited. |
| `last_accessed_at` | Last retrieval-hit time (an optional recency anchor). |
| `invalid_at` | Soft-invalidation time; `None` means valid. |
| `superseded_by` | The id of the newer memory that replaced this one. |
| `metadata` | An attached dict, defaults to empty. |

The same explicit id may exist in sibling scopes. `MemoryStore.get` and `MemoryStore.replace` are point operations: if a coarse scope such as `Scope(base="memory", user="alice")` matches more than one row with that id under different agent or session dimensions, they raise `RetrievalError` instead of choosing an arbitrary sibling. Pass a scope narrowed to one ownership footprint. Collection operations such as search and `all` retain the normal non-empty-dimension filter semantics.

You control `type`, `importance`, and `metadata` at write time:

```python
memory.add("Ships to production on Fridays", type="procedural", importance=0.9,
           metadata={"team": "backend"})
```

## Writing smart: SmartWriter

Calling `add` for every incoming message quickly fills memory with duplicates and stale contradictions. `SmartWriter` is a Mem0-style smart-write layer that keeps memory clean. For each piece of input it:

1. uses the LLM to **extract** atomic facts from the text,
2. **searches** existing memory for each fact,
3. asks the LLM to **decide** one of `ADD`, `UPDATE`, `DELETE`, or `NOOP`, and
4. **executes** that decision.

`write` is asynchronous and needs an LLM (a cheap model such as DeepSeek is a good fit). It returns one record per fact, so you can see exactly what happened:

```python
import asyncio
from agentmaker import Memory, MemoryStore, SmartWriter, LLMClient
from agentmaker.retrieval import build_sqlite_hybrid, OpenAIEmbedder

memory = Memory(retriever=build_sqlite_hybrid(OpenAIEmbedder()), store=MemoryStore())
writer = SmartWriter(memory, LLMClient("deepseek"))

records = asyncio.run(writer.write("I moved from Shanghai to Beijing last month"))
for r in records:
    print(r["op"], r["fact"])   # each record has: fact, op, id, content
```

`UPDATE` and `DELETE` do not physically erase the old fact. They **soft-invalidate** it: the old row stays in the store (still visible for audit) with an `invalid_at` timestamp, and an `UPDATE` links it to its successor through `superseded_by`. So "moved from Shanghai to Beijing" supersedes the old location without wiping the history.

`SmartWriter` is deliberately fail-safe. If fact extraction cannot be parsed, `fail_open=True` (the default) degrades to storing the whole input as one fact so nothing is lost; set `fail_open=False` for chit-chat or sensitive text you would rather drop than store wholesale. If the reconcile step returns anything invalid, it falls back to `ADD`, so a confused model never triggers a wrong delete. To change the extraction language or categories, pass your own `extract_prompt` / `reconcile_prompt`.

## Updating and forgetting

Beyond `add` and `search`, `Memory` exposes the full lifecycle:

- `update(id, content)` replaces a memory's body in a single atomic transaction and re-dates its recency to the edit time.
- `invalidate(id, superseded_by=...)` soft-invalidates a memory: the source record is marked invalid, index cleanup is requested, and recall excludes it even if a stale index row remains. The record is kept for audit. This is what `SmartWriter` uses.
- `delete(id)` / `delete_many(ids)` physically remove memories from the authoritative store and request cleanup from the derived index. Index cleanup is best-effort and may require `rebuild_index()` / reconciliation, so this API alone does not guarantee immediate physical erasure from every backend.
- `forget(strategy=...)` bulk-prunes and returns the deleted ids. Strategies: `"importance"` (drop items below `threshold`), `"age"` (drop items older than `max_age_days`), and `"capacity"` (keep only the top-N most important and recent).
- `stats()` returns `{"total": ..., "by_type": {...}}`, a pure count with no LLM call.

Two lifecycle operations use the LLM and are therefore asynchronous coroutines (require passing `llm=` at construction):

- `summary(query=None)` folds the matching memories into one coherent paragraph.
- `consolidate()` hands all memories to the LLM to merge duplicates and keep the latest of any contradictions, then rewrites the store. It returns `{"before": ..., "after": ...}`. Like `SmartWriter`, it soft-invalidates the old entries rather than deleting them.

```python
paragraph = await memory.summary()
result = await memory.consolidate()   # {"before": 12, "after": 8}
```

!!! note "Async surface"
    Read and write basics (`add`, `search`, `update`, `delete`, `forget`, ...) each have an `a*` counterpart (`aadd`, `asearch`, ...) that runs the blocking database and embedding work off the event loop. `summary` and `consolidate` are async by nature because they call the model.

## Key-value memory

For definite, single-valued facts, semantic recall is overkill and imprecise. `KVMemory` stores one value per key and reads it back exactly, with no guessing. `KVStore` is the underlying SQLite table (string values); `KVMemory` is a facade over it that JSON-encodes and decodes, so values can be strings, numbers, lists, or dicts. It carries a fixed [scope](retrieval-and-rag.md) for ownership:

```python
from agentmaker import KVStore, KVMemory, Scope

kv = KVMemory(KVStore(), scope=Scope(base="kv", user="alice"))

kv.set("location", "Beijing")
kv.set("allergies", ["peanuts"])

print(kv.get("location"))        # "Beijing"
print(kv.get("theme", "light"))  # default when the key is missing
print(kv.as_dict())              # {"location": "Beijing", "allergies": ["peanuts"]}
```

`set` overwrites in place, `get(key, default=None)` returns the default when a key is absent, `delete(key)` removes it, and `as_dict()` returns the whole set decoded.

## Giving memory to an agent

`MemoryTool` wraps a `Memory` (optionally with a `SmartWriter`) as a [tool](tools.md), so the agent can decide on its own to remember and recall mid-conversation. Register it like any other tool:

```python
from agentmaker import Agent, Memory, MemoryStore, MemoryTool, LLMClient
from agentmaker.retrieval import build_sqlite_hybrid, OpenAIEmbedder

memory = Memory(retriever=build_sqlite_hybrid(OpenAIEmbedder()), store=MemoryStore())
agent = Agent("assistant", LLMClient("deepseek"), tools=[MemoryTool(memory)])
```

The tool takes an `action` plus a `content` or `query`, and dispatches to: `remember`, `recall`, `summary`, `stats`, `forget`, and `consolidate`. Pass a `writer=` to route `remember` through `SmartWriter` for automatic dedup and rewrite instead of a plain `add`.

Because some actions modify or delete stored data, `MemoryTool` gates them behind human confirmation: `forget` and `consolidate` always require confirmation, and `remember` does too when a writer is attached (since `SmartWriter` may update or delete existing memories). Read actions (`recall`, `summary`, `stats`) and a plain add pass through. This writer-gated confirmation is on by default; pass `MemoryTool(memory, writer, confirm_writer_edits=False)` to turn it off. See [Guardrails & HITL](guardrails-and-hitl.md) (human-in-the-loop) for how the confirmation gate is wired.

The tool uses the Memory's fixed scope by default. For one Agent instance serving multiple tenants, use `MemoryTool(memory, scope_policy="merge_run")`; this fills empty `user`, `agent`, and `app` dimensions from the current run and rejects conflicts. Session inheritance is opt-in through `inherit_dimensions`. `recall` and `summary` results are marked as external content before they are returned to the model; this delimiter-based prompt-injection defense reduces risk but is not a security sandbox.

## Persistence and isolation

The two constructor arguments above default to in-memory SQLite, which is wiped on exit. To persist, give both the store and the retrieval backend the same file path:

```python
memory = Memory(
    retriever=build_sqlite_hybrid(FakeEmbedder(), db_path="memory.db"),
    store=MemoryStore(db_path="memory.db"),
)
```

`MemoryStore` is the authoritative source of truth: it holds the complete `MemoryItem` records, while the retrieval index is a rebuildable derivative. If the index is ever lost or you swap backends, `rebuild_index()` re-embeds every stored memory back into the index.

Resources assembled by `Memory.from_config()` are owned by that `Memory` and are released by `close()` or its context manager. Objects passed to the direct constructor remain caller-owned, which allows a retriever to be shared safely with RAG or another manager.

`summary(top_k=N)` limits both queried and unfiltered summaries to at most `N` memories, preventing an unbounded prompt when the store grows.

For multi-user isolation, give each user its own scope. Pass `scope=` at construction so every write and read stays within that owner's data, and keep one `Memory` (and one `SmartWriter`) per user:

```python
from agentmaker import Scope

alice = Scope(base="memory", user="alice")
memory = Memory(retriever=..., store=..., scope=alice)
```

Scopes are shared across the retrieval, memory, and RAG subsystems (RAG is retrieval-augmented generation, answering from your own documents). See [Retrieval & RAG](retrieval-and-rag.md) for the full scope model.
