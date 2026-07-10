# Context engineering

Context engineering is the final-assembly and quality-control stage that decides what actually reaches the
model window. Retrieval and memory are the suppliers: they return candidate passages ranked by relevance. This
subsystem takes those candidates, picks the ones that most deserve a place, orders them, keeps everything inside
an explicit token budget, and compresses when things overflow. Reach for it whenever you are stitching together
retrieval-augmented generation (RAG, feeding retrieved text to the model), long conversation history, or a
multi-step agent trajectory, and you want the prompt to stay within a known budget instead of growing until it
overruns the window.

The design principle throughout is explicit allocation rather than passive accumulation: every stream that
competes for the window draws its quota from a single ledger, so no half of the prompt can quietly eat the
other half.

All symbols on this page are importable from the top level (`from agentmaker import ContextBuilder`), except
the trajectory reducers, which live in `agentmaker.context`.

## The pipeline at a glance

[`ContextBuilder`](#assembling-the-prompt-contextbuilder) runs a fixed four-stage pipeline:

```
Gather      collect candidates from each source
   -> MMR   per-source de-duplication + diversity selection
   -> Budget   three-region budget with two-round quota borrowing
   -> Structure   fixed layout: system -> memory -> rag -> history -> tool -> question
```

It does not run a second relevance rerank. The baseline ordering comes from the retrieval foundation (see
[Retrieval & RAG](retrieval-and-rag.md)); this layer only does what assembly alone can do: de-duplicate,
select within budget, allocate the budget across sources, and lay out the sections.

## Estimating tokens: `count_tokens`

Every budget decision starts from a token estimate. `count_tokens` is a zero-dependency estimator for mixed
CJK / Western text (each Chinese, Japanese, or Korean character counts as one token; everything else at roughly
four characters per token).

```python
from agentmaker import count_tokens

count_tokens("hello world")   # 3
```

!!! note "Estimate, not billing"
    `count_tokens` is a pre-send budget estimate; it never feeds cost or quota accounting (those always use the
    real token usage the model returns). It deliberately does not split on whitespace, so long runs with no
    spaces (base64, long URLs) are still measured by the four-chars-per-token rule rather than counted as one
    token.

The estimator is a pluggable seam. `ContextBuilder` and `HistoryCompactor` accept a `token_counter` argument of
type `TokenCounter`, which is just `Callable[[str], int]`; the reducers take the same callable as `counter`.
Inject a more precise counter (for example a `tiktoken`-based one) in production if you need tighter accounting.

## Selecting non-redundant candidates: `mmr_select`

`mmr_select` implements MMR (maximal marginal relevance): it picks a subset that is both relevant and
mutually distinct. Stuffing every candidate into the window wastes tokens and dilutes the signal (context rot);
MMR selects one item at a time, weighing at each step how relevant a candidate is against how similar it is to
what has already been selected.

```python
from agentmaker import mmr_select, RetrievalResult

candidates = [
    RetrievalResult(content="Cats are great pets.", score=0.9, source="rag", embedding=[1.0, 0.0]),
    RetrievalResult(content="Cats make wonderful pets.", score=0.8, source="rag", embedding=[0.99, 0.01]),
    RetrievalResult(content="The Eiffel Tower is in Paris.", score=0.6, source="rag", embedding=[0.0, 1.0]),
]

selected = mmr_select(candidates, top_k=2, lambda_=0.7)
for r in selected:
    print(r.content)
# Cats are great pets.
# The Eiffel Tower is in Paris.
```

The near-duplicate second sentence is dropped, and a topically distinct item takes its place. The signature is:

```python
mmr_select(candidates, *, top_k=None, lambda_=0.7, dedup_threshold=0.95)
```

- `top_k`: maximum number to select; `None` means no cap on count (rely only on de-duplication to remove
  near-duplicates).
- `lambda_`: the relevance-versus-diversity trade-off in `[0, 1]`. `1.0` is pure relevance (no de-dup penalty);
  lower values emphasize diversity. The default `0.7` reflects that retrieval is already ranked, so moderate
  de-dup is enough.
- `dedup_threshold`: a candidate whose cosine similarity to any already-selected item is at or above this value
  is treated as a near-duplicate and dropped outright. `0.95` means two items must be nearly identical to count
  as duplicates; a value above `1` effectively disables near-duplicate removal.

MMR reuses the `embedding` vectors that retrieval already carries back on each `RetrievalResult`, so nothing is
recomputed. A candidate with no embedding (for example a keyword-only hit) is treated as similarity `0`: if
redundancy cannot be judged, diversity is not penalized. Exact byte-for-byte duplicate content is collapsed
first, keeping the highest-scoring copy.

## Sources: `ContextSource` and `CallableSource`

The builder consumes candidates through a uniform supplier interface. `ContextSource` is the abstract base:
each source has a `name` (which quota it draws from) and a `fetch(query, scope=None)` method returning a list
of `RetrievalResult`, plus an async `afetch` counterpart.

Most of the time you do not write a subclass. `CallableSource` adapts any `(query)` or `(query, scope)` callable
into a source, so `memory.search`, `rag.retrieve`, or your own function can be plugged in directly:

```python
from agentmaker import CallableSource, RetrievalResult

def search_docs(query: str) -> list[RetrievalResult]:
    return [
        RetrievalResult(content="Meals are capped at 80 per day, no receipt needed.", score=0.9, source="rag"),
        RetrievalResult(content="Hotels are capped at 500 per night, receipt required.", score=0.7, source="rag"),
    ]

source = CallableSource("rag", search_docs)
```

The `name` (here `"rag"`) selects which budget quota the source consumes; it must be a key of the config's
`source_ratios` (see below).

### Threading scope

`scope` is the session identifier threaded through a run (see [Scope isolation](retrieval-and-rag.md)). How it
reaches your callable is controlled by `pass_scope`:

```python
CallableSource("memory", memory.search)                                             # keyword-only scope, uses its own
CallableSource("memory", lambda q, s: memory.search(q, scope=Scope(user=s.user)))   # positional, by the run's user
CallableSource("rag", rag.retrieve, pass_scope=True)                                # force pass by keyword scope=
CallableSource("rag", lambda q: rag.retrieve(q, top_k=8))                           # custom top_k, no scope
```

By default (`pass_scope=None`) the mode is auto-detected by positional-parameter count: a callable with two or
more positional parameters receives `scope` as the second positional argument, otherwise it does not receive it.

!!! warning "Auto-detection only counts positional parameters"
    A callable that takes scope keyword-only (`def f(query, *, scope=None)`, as `memory.search` and
    `rag.retrieve` do) will not be auto-recognized and will not receive the run scope. This is intentional: bind
    those methods directly to use their own scope. To force the run scope into a keyword-only parameter, pass
    `pass_scope=True`; to force it off, pass `pass_scope=False`.

## Assembling the prompt: `ContextBuilder`

`ContextBuilder` runs the full pipeline and returns assembled text. There are two entry points.

`build` produces one flat string, `system -> sections -> current question`, suitable for single-shot or
RAG-style calls:

```python
from agentmaker import CallableSource, ContextBuilder, ContextConfig, RetrievalResult

def search_docs(query: str) -> list[RetrievalResult]:
    return [
        RetrievalResult(content="Meals are capped at 80 per day, no receipt needed.", score=0.9, source="rag"),
        RetrievalResult(content="Hotels are capped at 500 per night, receipt required.", score=0.7, source="rag"),
    ]

builder = ContextBuilder(ContextConfig.for_window(None, fallback_window=8000))
context = builder.build(
    "how much can I spend on meals?",
    sources=[CallableSource("rag", search_docs)],
    system_prompt="You are a finance assistant.",
)
print(context)
```

```text
You are a finance assistant.

[Knowledge]
- Meals are capped at 80 per day, no receipt needed.
- Hotels are capped at 500 per night, receipt required.

[Current question]
how much can I spend on meals?
```

Section headers (`[Knowledge]`, `[Current question]`, and so on) come from the prompt registry; a custom source
name that has no registered header falls back to `[name]`.

`build_block` assembles only the dynamic-source block (memory / RAG / ...), with no system prompt and no current
question. Use it for multi-turn conversations: inject the block as a system message and pass the conversation
history separately as role-carrying messages, so the user / assistant roles are not flattened away. It returns
an empty string when there are no candidates.

```python
build(query, *, sources, system_prompt="", scope=None) -> str
build_block(query, *, sources, scope=None, budget=None) -> str
abuild_block(query, *, sources, scope=None, budget=None) -> str   # async; fans out over sources concurrently
```

The async `abuild_block` shares the same budget convention.

### Budgeting knobs: `ContextConfig`

`ContextConfig` is a frozen, immutable budget configuration. It expresses the budget as ratios rather than
absolute numbers, so switching to a larger-window model is a matter of scaling, not re-tuning.

| Field | Default | What it controls |
| --- | --- | --- |
| `max_tokens` | `None` | Total context budget. No hard-coded default: set it from the model's real window. |
| `output_reserve_ratio` | `0.2` | Fraction reserved for output plus the current question (does not compete for candidates). |
| `source_ratios` | `{"history": 0.35, "rag": 0.30, "memory": 0.20, "tool": 0.15}` | Each source's share of the dynamic region. Keys are source names. |
| `mmr_lambda` | `0.7` | Passed to `mmr_select` as `lambda_`. |
| `dedup_threshold` | `0.95` | Passed to `mmr_select` as `dedup_threshold`. |
| `allow_borrow` | `True` | Whether a source's idle quota is redistributed, in a second round, to sources that still have candidates to place. |
| `min_chunk_tokens` | `64` | Smallest single candidate a quota must be able to hold, used for sanity checking. |

Set `max_tokens` from the model window with `for_window`:

```python
ContextConfig.for_window(context_window, *, use_ratio=0.5, fallback_window=None, **kwargs)
```

```python
ContextConfig.for_window(LLMClient("deepseek").context_window)   # 1M window -> max_tokens = 500,000
ContextConfig.for_window(None, fallback_window=8000)             # unknown local model, explicit fallback
```

`use_ratio` defaults to `0.5`: the context takes only half the window, leaving ample room for output and safety
margin. `fallback_window` has no default and must be supplied explicitly when the window is unknown, forcing you
to pin a conservative value rather than silently picking one.

!!! note "Fail-loud validation"
    Both a source name absent from `source_ratios` and two sources sharing a name are rejected before any fetch
    runs: the first would silently get a zero quota and never appear, the second would overwrite candidates
    during assembly. The config also validates at construction that each source's quota can hold at least one
    complete candidate block; a quota too small to fit even the most relevant item raises immediately.

The two rounds of allocation give each source its ratio-based quota first, then (when `allow_borrow` is on)
hand out any idle quota to sources that still have candidates to place, sharing the surplus by how much each
still wants rather than by input order.

## The window budget: `WindowBudgetConfig` and `WindowBudget`

When several streams compete for one window (system prompt, tool schemas, the retrieval block, the agent
trajectory, and the output reserve), each deciding its own size independently can push the total past the
window. The window budget funnels the whole allocation into a single ledger:

```
whole window = output reserve + fixed overhead (system + tool schemas) + retrieval block + trajectory
```

`WindowBudgetConfig` holds the serializable knobs:

| Field | Default | What it controls |
| --- | --- | --- |
| `desired_output_tokens` | `4096` | How many tokens at most you want the model to generate this run (the main output-reserve knob). |
| `max_output_fraction` | `0.5` | Small-window guardrail: the output reserve takes at most this fraction of the window. |
| `rag_ratio` | `0.35` | The retrieval block's share of the allocatable balance; the trajectory gets the rest. |

The output reserve is the smallest of `desired_output_tokens`, `window * max_output_fraction`, and (when the model's per-call output cap is known) that cap. Each clamp guards one failure mode: not reserving more than you asked for, not
leaving a dead zone the model can never fill on a large window, and not eating up input on a small window. Only
one ratio (`rag_ratio`) is configured; the trajectory takes the remainder, which structurally rules out two
ratios that sum past the window.

`WindowBudget` is the value object computed once per run from the real window plus the measured fixed overhead.
Build it with `for_run`, which returns `None` when the window is unknown (the caller then falls back to no cap):

```python
from agentmaker import WindowBudget, WindowBudgetConfig
from agentmaker.testing import ScriptedLLM

llm = ScriptedLLM(context_window=200_000)
budget = WindowBudget.for_run(llm=llm, cfg=WindowBudgetConfig(), system_tokens=800, tool_tokens=1200)

budget.rag_budget                              # retrieval block cap
budget.trajectory_budget(rag_in_scope=True)    # trajectory trimming budget
```

```python
WindowBudget.for_run(*, llm, cfg, system_tokens=0, tool_tokens=0, rag_ratio=None) -> Optional[WindowBudget]
```

Its read-only accounting:

- `fixed`: total fixed overhead, `system_tokens + tool_tokens`.
- `spendable`: the balance divisible between retrieval block and trajectory after subtracting the output reserve
  and fixed overhead (never negative).
- `rag_budget`: the retrieval block cap, `spendable * rag_ratio`. This is the value passed as `build_block(...,
  budget=...)` so the retrieval block draws from the shared ledger instead of re-reserving output on its own.
- `trajectory_budget(*, rag_in_scope)`: the trimming budget for the paradigm trajectory, branching on whether
  the data being trimmed already includes the retrieval block.

Tool schemas ride in the request's `tools=` payload rather than in `messages`, so trajectory trimming cannot see
them; the ledger subtracts them separately so a growing trajectory cannot push the tool schemas out of the
window.

## Compacting conversation history: `HistoryCompactor`

Conversations grow without bound: dozens of turns eventually exceed the budget and dilute the signal.
`HistoryCompactor` summarizes the older turns into a single recap with an LLM and keeps the most recent turns
verbatim. Recent conversation must stay precise (the model continues answering from it); distant history only
needs a summary.

```python
from agentmaker import HistoryCompactor, Message
from agentmaker.testing import ScriptedLLM

llm = ScriptedLLM(["The user asked how to get a refund and was told to check Settings > Billing."])
compactor = HistoryCompactor(llm, keep_recent=2, trigger_tokens=10)

history = [
    Message(content="How do I get a refund?", role="user"),
    Message(content="Open Settings then Billing.", role="assistant"),
    Message(content="I did that but I am still stuck.", role="user"),
    Message(content="Let me escalate this for you.", role="assistant"),
]

compacted = compactor.compact(history)
for m in compacted:
    print(m.role, "::", m.content)
```

```text
system :: [Recap] The user asked how to get a refund and was told to check Settings > Billing.
user :: I did that but I am still stuck.
assistant :: Let me escalate this for you.
```

The constructor:

```python
HistoryCompactor(llm, *, keep_recent=4, trigger_tokens=2000, max_summary_tokens=1000,
                 summary_prompt=None, prompts=None, token_counter=count_tokens)
```

- `keep_recent` (default `4`, must be `>= 1`): how many recent turns to keep verbatim.
- `trigger_tokens` (default `2000`, must be `>= 0`): compress only when the total history exceeds this,
  otherwise return the history unchanged and spend no LLM call.
- `max_summary_tokens` (default `1000`, must be `>= 1`): hard cap on the recap, truncated if exceeded. This
  keeps the incrementally merged summary from growing without bound across hundreds of turns, since the cached
  summary is fed back as input on the next turn.
- `summary_prompt`: override the default summary instruction (for example to switch language); if omitted the
  framework default is used.

`compact(history, *, summarize=None)` returns the compacted list of `Message`; `acompact(history, *,
asummarize=None)` is the async counterpart. When the history is at or below the threshold, or has at most
`keep_recent` turns, the original history is returned untouched. If summarization fails, the compactor keeps the
original history rather than losing it.

`CompactionConfig` is the serializable slice that feeds `HistoryCompactor.from_config(llm, config)`:

| Field | Default | Meaning |
| --- | --- | --- |
| `keep_recent` | `4` | Recent turns kept verbatim. |
| `trigger_tokens` | `2000` | History token count that triggers compression. |

!!! note "Compaction is not assembly-time compression"
    History compaction summarizes one large object (the conversation) with a single LLM call. It is distinct
    from trimming scattered retrieval candidates at assembly time, which the builder does not do; candidate size
    is instead controlled upstream by chunking.

## Trimming paradigm trajectories: the reducers

Complementary to history compaction, the reducer layer trims an agent's own working trajectory when it
approaches the window budget, dropping the least important signals first while preserving each paradigm's
lifeline. A generic summary would strip out a reflection loop's past critique points or a plan's exact step
numbers, which would break the paradigm, so each has its own loss-aware policy. These live in
`agentmaker.context`:

```python
from agentmaker.context import reduce_agent, reduce_plan, reduce_reflection, tokens_of, REDUCERS
```

- `reduce_agent` trims a unified-loop tool-call trajectory: the most recent atomic units (an assistant message
  plus its tool results) are kept verbatim, earlier ones are summarized into a single system entry.
- `reduce_plan` trims plan step results, keeping the most recent steps verbatim and preserving key numbers,
  dates, and conclusions.
- `reduce_reflection` trims a reflection trajectory, keeping the latest answer plus a de-duplicated list of past
  critique points and dropping superseded drafts.

All three are async and take a caller-supplied `summarize(text, instruction) -> str` async callback plus the
token `budget` from `WindowBudget.trajectory_budget`. `REDUCERS` maps `"agent"` / `"plan"` / `"reflection"` to
these functions, and `tokens_of(*texts, counter=count_tokens)` estimates the total tokens of several texts. If
the parts that must be kept already exceed the budget, a reducer raises `ContextWindowExceeded` rather than
silently truncating.

`ReducerConfig` holds the serializable knobs for how much recent text to keep uncompressed:

| Field | Default | Meaning |
| --- | --- | --- |
| `agent_keep_recent_steps` | `3` | Trailing tool-trajectory units kept verbatim. |
| `plan_keep_recent` | `3` | Trailing plan step results kept verbatim. |

The trajectory's token budget itself is not in this config; it comes from the window ledger, so the two ratios
can never sum past the window.

## Configuring it all together

`AgentmakerConfig` aggregates these sub-configs (`context`, `reducer`, `compaction`, `window_budget`, among
others) into one holder you set once at your assembly root and pass down. `to_dict` / `from_dict` serialize it,
and `for_window(context_window)` derives an instance with `context.max_tokens` set from the model window. When a
builder or compactor is wired into an agent, the retrieval-block and trajectory budgets are supplied by the
shared `WindowBudget`, so `max_tokens` on `ContextConfig` may be left unset; only standalone `build` /
`build_block(budget=None)` calls require it.

See [Retrieval & RAG](retrieval-and-rag.md) for where the candidates come from, and [Memory](memory.md) for
the memory source you most often plug in.
