# Guardrails & human-in-the-loop

This guide covers the safety and control layer around an agent run: **guardrails** that screen input and output and trip the run when a rule is violated, and **human-in-the-loop** (HITL, a person approving an action before it executes) that suspends a run at a high-risk tool call and waits for a decision. It also covers the supporting machinery those two rely on: lifecycle **hooks** for observing a run, **session** persistence for conversation history, **checkpoints** for suspend/resume and crash recovery, and **run policies** for global per-run limits. Reach for this page when you need to block certain inputs, require approval before a dangerous action, keep an audit trail, or cap what a single run is allowed to do.

Everything on this page is grounded in [`examples/07_guardrails_and_hitl.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/07_guardrails_and_hitl.py), which is hermetic (runs with no API key and no network via `ScriptedLLM`, the LLM test double):

```bash
uv run python examples/07_guardrails_and_hitl.py
```

## Guardrails

A guardrail checks a piece of text (an agent's input or its final output) and returns a verdict. A failing verdict is a *tripwire*: the run stops and `GuardrailTripwireError` is raised. You attach guardrails to an `Agent` with `input_guardrails=[...]` (checked against the user input before the model runs) and `output_guardrails=[...]` (checked against the final output before it is returned).

The quickest way to make one is `CallableGuardrail`, which wraps any function `fn(text)` into a guardrail. The function returns a bool (True lets the text through, False is a tripwire) and the `message=` you pass at construction becomes the block explanation:

```python
from agentmaker import Agent, CallableGuardrail, GuardrailTripwireError, tool
from agentmaker.testing import MemoryCheckpointStore, ScriptedLLM

# 1) Guardrail: reject any input that mentions a password.
no_secrets = CallableGuardrail(lambda text: "password" not in text.lower(),
                               message="input mentions a password")
guarded = Agent("assistant", ScriptedLLM(["ok"]), input_guardrails=[no_secrets])
try:
    guarded.run("my password is 1234")
except GuardrailTripwireError as e:
    print("Guardrail blocked the input:", e)
```

The `str` of a `GuardrailTripwireError` is the readable block explanation shown to the user, so `print(..., e)` above prints the `message` you configured.

### The guardrail interface

`CallableGuardrail`, `Guardrail`, and `GuardrailResult` are all part of the public API.

`GuardrailResult` is the verdict a guardrail returns. It has two fields:

- `passed: bool` (True lets the text through, False signals a tripwire).
- `message: str` (a human-readable explanation of the block; defaults to `""` and may be empty when `passed=True`).

A `CallableGuardrail` wraps a function that returns either a bool or a `GuardrailResult`. With a bool, `False` trips the guardrail using the message you gave at construction; returning a `GuardrailResult` instead lets the function carry its own message. For example:

```python
from agentmaker import CallableGuardrail

# Bool form: False trips, using the message given at construction.
length_limit = CallableGuardrail(lambda t: len(t) < 4000, message="input too long")
```

For anything more than a one-liner, subclass `Guardrail` and implement `check`:

```python
from agentmaker import Guardrail, GuardrailResult


class BlocklistGuardrail(Guardrail):
    def __init__(self, words):
        self._words = [w.lower() for w in words]

    def check(self, text: str) -> GuardrailResult:
        hit = next((w for w in self._words if w in text.lower()), None)
        if hit is not None:
            return GuardrailResult(passed=False, message=f"input contains a blocked word: {hit}")
        return GuardrailResult(passed=True)
```

`Guardrail` is an abstract base class with one abstract method, `check(self, text) -> GuardrailResult`. There is also an async counterpart `acheck(self, text) -> GuardrailResult`, which the framework's execution layer actually calls; by default it inlines a direct call to `check`. Most guardrails are pure computation (length, regex, blocklist checks) so the default is right. Override `acheck` only when the guardrail does blocking I/O or wants to call an LLM to moderate the text. `CallableGuardrail` accepts an async function too (an `async def`, or a lambda wrapping an async call), which it awaits through `acheck`.

!!! note
    agentmaker ships the guardrail *interface* plus `CallableGuardrail`; the concrete rules are your business logic. There are no built-in content policies to configure. Write the checks your app needs and attach them as `input_guardrails` / `output_guardrails`.

## Human-in-the-loop (HITL)

When a tool is marked as a high-risk action, an agent run does not execute it silently. Instead the run *suspends* at that call, hands you a description of the pending action, and waits until you approve or reject it. Two pieces enable this:

1. Mark the tool with `@tool(requires_confirmation=True)`.
2. Give the agent a **checkpoint store** so the suspended state can be saved and resumed. For a real deployment use `SqliteCheckpointStore`; in tests use `MemoryCheckpointStore`.

```python
# 2) HITL: a high-risk tool suspends the run until a human approves.
@tool(requires_confirmation=True)
def delete_file(path: str) -> str:
    """Delete a file (high-risk, requires confirmation).

    Args:
        path: File path to delete.
    """
    return f"deleted {path}"


ops = Agent("ops", ScriptedLLM([
    ScriptedLLM.tool_call("delete_file", {"path": "/tmp/old.log"}),
    "Done, the file was deleted.",
]), tools=[delete_file], checkpoint_store=MemoryCheckpointStore())

result = ops.run("please delete /tmp/old.log")
if result.interrupt:                                    # run paused, awaiting approval
    pending = result.interrupt.pendings[0]
    print(f"Approval needed for: {pending.tool_name}({pending.arguments})")
    approved = ops.resume(True, scope=result.interrupt.scope)   # resume(False) would reject
    print("After approval:", approved.final_output)
```

### Reading the interrupt

`run` (and `resume`) always return a `RunResult`. When a high-risk tool suspends the run, `RunResult.status` is `"interrupted"`, the convenience property `RunResult.interrupted` is `True`, and `RunResult.interrupt` holds an `Interrupt`. When the run instead finishes normally, `status` is `"completed"`, `interrupt` is `None`, and `final_output` holds the answer. See [Agents & workflows](agents.md) for the full `RunResult` shape.

An `Interrupt` describes what is waiting:

- `pendings`: the list of `PendingAction` awaiting approval. One suspend can contain more than one action (a single turn requested several high-risk tools, or parallel sub-agents each suspended).
- `pending`: a convenience property returning the first pending action, or `None` if there are none.
- `scope`: the resume credential. Pass it back to `resume` (this is required to reload the suspended state across sessions).

Each `PendingAction` carries `tool_name`, `arguments` (the call arguments), and `call_id` (the unique id of this tool call; `resume` matches decisions against it).

### Resuming

Call `resume(decision, *, scope=...)` to continue a suspended run. The `decision` can be:

- A bool: `resume(True, scope=...)` approves and executes the suspended action; `resume(False, scope=...)` rejects it (the rejection is fed back to the model so it can reroute, not treated as an error). A single bool decides all pending actions of the turn uniformly.
- A dict `{call_id: bool}`: decide per action, keyed by each `PendingAction.call_id`. Use this when one suspend holds several pending actions.
- Omitted (`None`): a crash-recovery resume. No decision is injected; the run simply continues from the last checkpoint, and a still-pending high-risk action re-suspends and returns a fresh `Interrupt` rather than being mistaken for a rejection.

`resume` returns a `RunResult` just like `run`: `completed` with a `final_output`, or `interrupted` again if another high-risk action is now waiting. Pass `scope=result.interrupt.scope` so `resume` loads the correct suspended state; it defaults to the agent's own `scope` if you omit it.

!!! note
    `ApprovalRequired` is a public name but an internal control-flow signal, not an error you catch. The harness raises it when it reaches a high-risk tool with no decision this turn; the run loop catches it, packages the state, and converts it into the `Interrupt` you receive. You interact with the `Interrupt` and `resume`, never with `ApprovalRequired` directly.

### Per-call confirmation

`requires_confirmation=True` on the decorator marks the whole tool as high-risk. The gate actually reads `Tool.needs_confirmation(parameters)`, which defaults to returning `requires_confirmation`. A tool that subclasses `Tool` can override `needs_confirmation` to decide per call (for example, confirm only when deleting outside a safe directory).

### Synchronous confirmation with `cli_confirm`

HITL suspend/resume is the right model for a server: the request returns an interrupted `RunResult` whose `interrupt` describes the pending action, a human decides out of band, and a later request resumes. For a command-line or teaching setting you may instead want a blocking y/n prompt inline. `cli_confirm` is that battery: pass it as `confirm=cli_confirm` and a high-risk tool prints its action and asks on stdin.

```python
from agentmaker import Agent, cli_confirm

agent = Agent("ops", llm, tools=[delete_file], confirm=cli_confirm)
```

`cli_confirm(tool, parameters) -> bool` prints the tool name and parameters and returns whether the user typed `y`. It is not the default: if you pass no `confirm` and no `checkpoint_store`, the safe choice is to deny (so a headless server never hangs on `input()`). Use `cli_confirm` for interactive CLIs and `checkpoint_store` + `resume` for servers.

## Hooks

A `Hook` is an observe-only lifecycle callback. Subclass `Hook`, override just the events you care about (the rest are no-ops), and attach a list with `hooks=[...]`. Hooks are for side effects such as logging, metrics, auditing, and cost tracking. They cannot intercept or modify the run; interception belongs to guardrails, permissions, and HITL.

```python
from agentmaker import Agent, Hook


class AuditHook(Hook):
    def before_tool(self, name: str, parameters: dict):
        print(f"about to run tool {name} with {parameters}")

    def on_guardrail_trip(self, stage: str, message: str):
        print(f"guardrail tripped at {stage}: {message}")


agent = Agent("assistant", llm, tools=[delete_file], hooks=[AuditHook()])
```

The full set of events (all no-ops by default):

| Method | Fires when |
| --- | --- |
| `on_run_start(input_text, *, scope=None)` | A run begins, before the input guardrails. |
| `before_model(messages)` | Before each LLM call (also for streaming). |
| `after_model(response)` | After each non-streaming LLM call, and after the terminal `LLMResponse` of a stream invoked with tools. A plain-text stream has no single response object and does not fire this event. |
| `before_tool(name, parameters)` | Just before a tool executes (already past the permission and approval gates). |
| `after_tool(name, parameters, result)` | After a tool executes; `result` is a `ToolResponse`. |
| `on_guardrail_trip(stage, message)` | On a guardrail tripwire; `stage` is `"input"` or `"output"`. |
| `on_interrupt(pendings, *, scope=None)` | On a HITL suspend; `pendings` is the list of `PendingAction`. |
| `on_error(error)` | Just before a non-guardrail exception propagates. |
| `on_run_end(output, *, scope=None)` | When a run produces its final result normally. |

Every return value is ignored (hooks are pure side effects), and an exception raised inside a hook propagates upward (fail loud), so wrap risky I/O yourself. Run-level events (`on_run_start`, `on_interrupt`, `on_guardrail_trip`, `on_error`, `on_run_end`) are fired by the agent; model and tool events are fired by the underlying harness. An event method may be written as `async def` when the framework runs it on its async path.

## Sessions

By default an agent keeps conversation history in process, so a restart loses it. Attach a `SessionStore` to persist history and survive restarts. `SqliteSessionStore` is the built-in backend; give it a file path in production (the default `":memory:"` is for tests only). History is isolated by `Scope`, the same isolation label used across retrieval and memory (see [Retrieval & RAG](retrieval-and-rag.md)).

```python
from agentmaker import Agent, Scope, SqliteSessionStore

store = SqliteSessionStore("daemon.db")
agent = Agent("assistant", llm, session_store=store, scope=Scope(user="alice", session="chat-1"))
```

`SessionStore` is append-only: each message is one row, only appended and never rewritten. The interface is `append` / `append_many` / `load` / `clear`, each taking a keyword `scope`. `load` and `clear` match all scope dimensions exactly by default (an empty scope reads only the default bucket, never crossing into another session); pass `all_scopes=True` for a deliberate cross-session operation. `SqliteSessionStore` additionally offers `prune(...)` to truncate old history (`keep_last=N` or `before=time`) and `list_scopes(along="session")` to enumerate which sessions exist (each returned `ScopeSummary` carries a `message_count` and first/last timestamps, handy for building a conversation list). `append` / `append_many` / `load` / `clear` / `list_scopes` each have an `a*` async counterpart; `prune` is sync-only.

### Searching past conversations

`ConversationSearch` wraps any `SessionStore` to make past conversations semantically searchable (episodic recall: "what did we discuss before"). It is itself a `SessionStore`, so you attach it in place of the plain store; on top of the usual methods it adds `search(query, *, top_k=5, scope=None)` returning a list of `RetrievalResult`. It needs a shared retrieval backbone (a `HybridRetriever`) to index into:

```python
from agentmaker import ConversationSearch, SqliteSessionStore

searchable = ConversationSearch(SqliteSessionStore("daemon.db"), retriever)
agent = Agent("assistant", llm, session_store=searchable, scope=scope)
```

To let the *model* search past turns on its own, wrap it as a tool with `ConversationSearchTool(searchable, scope=scope)` and hand that to the agent's tools. Writes land in the source-of-truth store first and are fed into the index best-effort, so an index hiccup never loses a message.

`clear` is deliberately fail-closed in the other direction: it first deletes the matching derived-index entries with strict physical-error propagation, and clears the authoritative session rows only after that succeeds. For one scope it uses the exact ownership footprint; `all_scopes=True` uses the explicit conversation range. A custom `IndexSync` or retriever written against the pre-exact interfaces keeps working: the exact drop degrades to a ranged delete, and an implementation without the `strict` flag keeps its best-effort drop contract. Call `close()` when finished; the wrapper owns and closes the supplied session store and its index-sync seam.

## Execution state, checkpoints, and resume

Under the hood, a run's trajectory lives in an `ExecutionState`: the message list, the pending HITL actions, the decision table, the remaining iteration budget, and per-paradigm resume metadata. A `CheckpointStore` persists a serialized `ExecutionState` by scope so a run can be paused and resumed. It backs three uses:

- **HITL**: save at the suspend point, then `resume(decision)` to continue.
- **Crash recovery**: unfinished state is saved every step, so after a process restart `resume()` (no decision) continues from the last recoverable checkpoint.
- **Long-task resume**: same mechanism as crash recovery.

Unlike a session store, a checkpoint is the single current restorable state: `save` overwrites one point per scope. During finalization, an agent with a checkpoint marks execution completed before it writes conversation history, then clears the checkpoint after history succeeds. If the process stops or history persistence fails inside that window, the completed marker remains. A later `resume()` clears it and raises `SessionError` instead of replaying tools or appending the turn twice; the application should reconcile session history because the final turn may already be present or may be missing. This is an at-most-once preference for framework-controlled replay, not an exactly-once guarantee for distributed side effects.

`CheckpointStore` is the abstract interface (`save` / `load` / `clear` by scope, plus `a*` async forms); `SqliteCheckpointStore` is the built-in backend and can share a database file with sessions and memory.

```python
from agentmaker import Agent, SqliteCheckpointStore

agent = Agent("ops", llm, tools=[delete_file],
              checkpoint_store=SqliteCheckpointStore("daemon.db"))
```

You normally do not construct `ExecutionState` yourself; you pass a checkpoint store and let the agent manage it. `ExecutionState` is public so you can inspect or implement a custom `CheckpointStore` backend.

## Run policies and limits

A `RunPolicy` sets global limits for a single run and an optional cancellation hook. Attach it with `run_policy=...`. When a limit is exceeded the run aborts with `RunLimitExceeded`; when the cancel hook returns `True` the run aborts with `RunCancelled`. Both are framework exceptions (subclasses of `AgentmakerError`).

```python
from agentmaker import Agent, RunPolicy, RunLimitExceeded

policy = RunPolicy(max_llm_calls=8, max_tool_calls=20, deadline_seconds=30)
agent = Agent("assistant", llm, tools=[delete_file], run_policy=policy)
try:
    result = agent.run("do a long multi-step task")
except RunLimitExceeded as e:
    print("run hit a limit:", e)
```

The fields (each `None` means unlimited):

- `max_llm_calls`: maximum LLM calls in the run (including streaming and nested child executors); must be `>= 1`.
- `max_tool_calls`: maximum tools *actually executed* (calls blocked by permission or confirmation do not count); must be `>= 0`. Setting `0` disables tools for the run: the LLM can still be called, and the run aborts the moment the model tries to execute a tool (a hard "read-only / safe mode").
- `max_tokens`: cumulative token limit (the sum of `usage.total_tokens` across LLM responses); must be `>= 1`.
- `deadline_seconds`: wall-time limit measured from the start of the run; must be `> 0`. Enforcement is cooperative before model/tool calls, after model calls, and before the final history commit. It does not interrupt an in-flight SDK or tool call, but a run that has exceeded the deadline is not committed as a successful final result. A rejection raised at the history-commit step clears that turn's checkpoint so the scope is reusable; a deadline that trips earlier leaves a recoverable checkpoint for `resume()`. Against a non-buffered stream the deadline is best-effort: text already delivered to the consumer is not rolled back, but a deadline that elapses mid-stream still aborts the run.
- `cancel`: a fast, non-blocking callback `() -> bool`, checked before each LLM and tool call; returning `True` aborts. When one agent serves multiple sessions, the callback can call `current_run_id()` to tell which run it is looking at.

When any numeric call/token cap is configured, parallel tool batches are disabled for that run. Serial admission keeps both direct tool counts and nested LLM/token accounting within their exact limits.

Limits are validated at construction, so a meaningless value (a negative count, `max_llm_calls=0`) raises `ValueError` immediately rather than surfacing mid-run.

!!! note
    Limits are counted against the outermost run globally. A nested child agent's own `RunPolicy` does not take effect inside a parent run (it warns). A resume is a new run, so limits reset for it. To limit subtasks, set the limits on the parent agent's `run_policy`.

## Related guides

- [Tools](tools.md) for `@tool`, `requires_confirmation`, and tool permissions.
- [Agents & workflows](agents.md) for `RunResult`, `run`, and `resume`.
- [Observability](observability.md) for tracing runs (hooks observe; tracers record structured events).
- [Retrieval & RAG](retrieval-and-rag.md) for `Scope` and the `HybridRetriever` that `ConversationSearch` indexes into.
