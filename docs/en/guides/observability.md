# Observability

Every agent run can emit a structured trace: one record per LLM call, tool call, and context operation, complete with timings and token usage. Attach a `Tracer` when you want to debug a run, audit cost, or ship events to a backend like SQLite or OpenTelemetry. Nothing is attached by default, so an agent with no tracer pays zero overhead. When you do want to observe, you inject a `Tracer`, and where its events land is decided by pluggable exporters. Later on this page, [Trace Detective](#trace-detective-devtools) turns a recorded trace into an LLM-written diagnosis of what went wrong.

## Attach a tracer

Construct a `Tracer` and pass it to the agent. The tracer collects the events the agent emits during the run; you read them back from the exporter afterward. This example is hermetic (no API key, no network), copied from [`examples/13_observability.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/13_observability.py):

```python
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
```

This run captures a `llm_call` (the model decides to call the tool), a `tool_call` (the tool runs), and a final `llm_call` (the model writes the answer).

`Agent(..., tracer=None)` is the default, so leave the argument off in production paths where you do not want tracing.

## What gets recorded

The framework emits one event dict per operation. Each event has a `type` plus type-specific fields, and every event is stamped with correlation fields so you can group by run and order by step. Most events come from the agent's harness; the memory and RAG subsystems emit their own `memory_search` and `rag_retrieve` events the same way:

| `type` | Key fields |
| --- | --- |
| `llm_call` | `model`, `latency_ms`, `usage` |
| `tool_call` | `tool`, `params`, `status`, `latency_ms`, `result` |
| `context_block` | `query`, `block_chars` |
| `memory_search` | `query`, `hits` |
| `rag_retrieve` | `query`, `hits`, `latency_ms` |

Every event also carries `run_id` and `step_index` (added by the framework's correlation step), so events from the same run share one id and increment in order.

!!! note "Secrets never reach a sink"
    Before an event fans out, the tracer redacts it: values whose key name looks like a secret (`api_key`, `token`, `password`, and similar) are masked to `***`, secret-looking strings (an `sk-` key, a `Bearer` token, a long token run) are masked in place, and home-directory usernames in paths (`/Users/<name>/`) are masked. Long string values are always truncated to `max_value_len` (default 200 characters) even when redaction is off. The framework knows no business concepts, so declare app-specific sensitive fields yourself with `extra_secret_keys=[...]` (key-name substrings) or `extra_secret_patterns=[...]` (value regexes) on the `Tracer` constructor. The `run_id` and `step_index` fields are exempt, so correlation is never broken by masking.

## Exporters

An exporter decides where events go. All four subclass `TraceExporter` (interface: `export(event)` plus a `close()` that releases resources), and a single `Tracer` can drive several at once. Redaction happens once, before the fan-out, so every exporter receives already-cleaned events.

| Exporter | Signature | Where events go |
| --- | --- | --- |
| `MemoryExporter` | `MemoryExporter(max_events=2048)` | An in-memory list (ring buffer, drops oldest past the cap). The default sink; lost on restart. |
| `JsonlExporter` | `JsonlExporter(path)` | One JSON line appended per event (JSON Lines), flushed immediately. |
| `SqliteExporter` | `SqliteExporter(db_path=":memory:")` | One row per event in a `traces` table (`type`, `run_id`, `event`, `created_at`), indexed on `run_id`. |
| `OTelExporter` | `OTelExporter(tracer_name="agentmaker", *, carrier_provider=None)` | One OpenTelemetry (OTel, the vendor-neutral tracing standard) span per event, for Jaeger / Grafana / Datadog. |

If you pass no `exporters`, the tracer defaults to `[MemoryExporter()]`. To persist while still reading events in-process, include a `MemoryExporter()` alongside the persistent one:

```python
from agentmaker import JsonlExporter, MemoryExporter, Tracer

tracer = Tracer(exporters=[MemoryExporter(), JsonlExporter("run.jsonl")])
```

Call `tracer.close()` before the process exits to flush and release file / database handles.

### OpenTelemetry

`OTelExporter` maps each event to a span. It uses the event's `latency_ms` to give the span a real width in a waterfall view (rather than a zero-width point), and always attaches `run_id` as a span attribute so a backend can filter per run. It lazily imports `opentelemetry`, so install the `otel` extra:

```bash
pip install "agentmaker[otel]"
```

To make agent spans join an upstream request trace, pass `carrier_provider=current_trace_carrier`. See [Run-level context](#run-level-context) below for how the carrier is supplied.

## Reading the trace back

The `Tracer` exposes convenience readers over its in-memory events (the first `MemoryExporter` in its exporter list):

- `tracer.events` returns the collected event list.
- `tracer.summary()` returns a dict with `events` (total count), `by_type` (count per type), `total_tokens`, `total_latency_ms`, plus `dropped` (events lost to exporter failures, per exporter) and `dropped_uncleanable` (events dropped because cleaning itself raised).
- `str(tracer)` renders a readable one-line-per-event timeline.
- `tracer.clear()` empties the in-memory events (file / database sinks are untouched).

An exporter that throws is swallowed by default so a side-channel failure (disk full, database lock, unreachable collector) never takes down the run. Construct the tracer with `strict=True` to make exporter and cleaning failures re-raise instead, which is useful in tests.

## Run-level context

The framework propagates a run's identity and governance state through `contextvars`, so async tasks and thread pools stay isolated. These accessors let an app, tool, or hook read the current run's context. All are importable from the top level:

```python
from agentmaker import (
    current_run_id, current_scope, current_step, current_trace_carrier,
)
```

- `current_run_id()` returns this run's `run_id` (or `None` outside a run), so you can correlate your own logs with the trace.
- `current_step()` returns the step number the run has reached.
- `current_scope()` returns the run's session scope (used, for example, by a delegating tool to isolate a child agent's history by the parent session).
- `current_trace_carrier()` returns the run's upstream W3C trace carrier (a dict like `{"traceparent": ...}`), or `None` if none was supplied.

You supply the carrier when you start the run. `agent.run(...)` and `agent.arun(...)` accept `trace_carrier`, so a web handler can pass the inbound request's `traceparent` header:

```python
result = agent.run(user_text, trace_carrier={"traceparent": request_header})
```

With `OTelExporter(carrier_provider=current_trace_carrier)` attached, each of this run's spans then becomes a child of the app's cross-service trace instead of a new root.

### governed_chat

Most LLM and tool calls run through the harness, which applies run limits and tracing for free. A few framework paths call the model directly, bypassing the harness. If you hand-write a recipe that calls an LLM directly and want it to respect the same run governance, route the call through `governed_chat` (async):

```python
from agentmaker import governed_chat

response = await governed_chat(llm, messages, tracer=tracer, origin="my.recipe")
```

It checks the run's limits, awaits `llm.chat(messages, ...)`, records the call's count and token usage, optionally emits a trace event tagged with `origin`, then enforces the hard token limit. Outside a run context the governance (limit checks and usage accounting) is a zero-overhead no-op — the LLM call itself, and the trace event if a tracer is passed, still happen. The `tracer` argument is optional; extra keyword arguments pass through to `llm.chat`.

## Trace Detective (devtools)

Trace Detective is an optional developer tool that consumes a recorded trace and returns an LLM-written diagnosis: the earliest step that went wrong, the root cause, and the smallest fix. It lives in the `agentmaker.devtools` subpackage, which the framework core never imports, so the native tracing described above works with or without it. It ships behind the `devtools` extra:

```bash
pip install "agentmaker[devtools]"
```

Because it is not part of the top-level namespace, import it on demand:

```python
from agentmaker.devtools import diagnose_trace, DoctorHook
```

### Diagnose from the library

Record a run to a JSONL file (attach a `JsonlExporter` as shown above), then hand the file to `diagnose_trace`. It parses the whole trace, picks one run (by `run_id`, or the most recent), and diagnoses it with any LLM client. It returns the parsed run and the verdict:

```python
from agentmaker import LLMClient
from agentmaker.devtools import diagnose_trace

run, verdict = diagnose_trace(open("run.jsonl").read(), LLMClient("deepseek"))
```

The verdict is a `TraceDiagnosis` with these fields: `healthy` (bool), `first_bad_step` (the earliest failing step number, or `None`), `what_went_wrong`, `root_cause`, `suggested_fix`, and `confidence` (`"low"` / `"medium"` / `"high"`). Diagnosis runs through a normal agentmaker agent with structured output, so any LLM client the framework supports works here unchanged.

### Diagnose in the web UI

Start the local web server:

```bash
python -m agentmaker.devtools
```

It binds `127.0.0.1:8765` by default (a local debugging tool, not something to expose). Paste or load a trace to see the deterministic timeline plus findings, then request an LLM diagnosis. The server builds its diagnosis client from environment API keys; if no key is available it still starts in parse-only mode so the timeline stays usable. Useful flags: `--host`, `--port`, `--provider` (default `deepseek`), `--model`, and `--no-llm` (parse-only, skip the LLM).

### DoctorHook: diagnose on the spot

For the zero-friction path while developing, attach a `DoctorHook` and every troubled run diagnoses itself in the terminal, with no file to export or web UI to open. Pass the same `Tracer` to both the agent and the hook (the hook reads this run's events back from the tracer's `MemoryExporter`):

```python
tracer = Tracer()
agent = Agent("bot", llm, tools=[...], tracer=tracer, hooks=[DoctorHook(tracer)])
agent.run("...")   # a failed tool / truncation / exception now prints a three-part diagnosis
```

A run that raises always triggers a diagnosis; a run that finishes normally triggers only when its trace carries findings at or above the hook's `severity` threshold (`"error"` by default, which covers failed tools and truncation; `"warn"` widens it to include empty retrievals and other degradations). The diagnosis LLM is built lazily from environment keys, or you can hand it a ready client with `llm=` (or choose the paying vendor with `provider=` / `model=`). Every failure inside the hook is caught and reported as a single console line, so a broken diagnosis never affects the run's own outcome.

!!! note
    `DoctorHook` is a lifecycle `Hook`, the same extension point covered in [Guardrails & HITL](guardrails-and-hitl.md). It runs the diagnosis in a worker thread under a fresh context, so it never consumes the host run's limits: even a run that died of a run-limit error can still be diagnosed.
