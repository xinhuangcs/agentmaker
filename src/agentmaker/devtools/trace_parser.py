"""agentmaker.devtools.trace_parser: deterministic parsing of agentmaker traces (the free half of Trace Detective).

Turns a JSONL trace (JsonlExporter output) or an in-memory event list (MemoryExporter.events) into per-run
timelines: group events by run_id preserving order, attach a one-line summary plus rule-based findings
(tool failures / truncation / empty retrieval / degradation events) to each step, and aggregate run stats.
Everything here is LLM-free, cheap and unit-testable; the LLM half lives in diagnose.py. Field semantics
come from the event conventions documented in agentmaker.core.trace_events (the single source of truth);
events are assumed already redacted + truncated by the Tracer.
"""

import json
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Union

from ..core.trace_events import (ALL_EVENT_TYPES, EVENT_CONTEXT_BLOCK, EVENT_CONTEXT_COMPACT,
                                 EVENT_CONTEXT_REDUCE, EVENT_INDEX_SYNC_PENDING, EVENT_INDEX_SYNC_RECONCILE,
                                 EVENT_LLM_CALL, EVENT_MEMORY_SEARCH, EVENT_RAG_CONTEXTUALIZE_FAILED,
                                 EVENT_RAG_QUERY_TRANSFORM_FAILED, EVENT_RAG_RETRIEVE,
                                 EVENT_SUMMARIZE_FAILED, EVENT_TOOL_CALL)

# Mirrors harness._TRUNCATION_REASONS (private there; duplicated to keep devtools decoupled from the runtime internals).
_TRUNCATION_REASONS = frozenset({"length", "max_tokens", "model_context_window_exceeded"})
# tool_call statuses that mean "the tool itself failed" vs "the call was stopped by a gate" (see ToolResponse / trace_tool_gate).
_TOOL_FAILED = frozenset({"error", "invalid_args"})
_TOOL_BLOCKED = frozenset({"denied", "rejected"})

Severity = Literal["error", "warn"]


class TraceParseError(ValueError):
    """The input is not a readable agentmaker trace (bad JSONL line, non-dict event, missing type, unknown run_id...)."""


@dataclass(frozen=True)
class Finding:
    """One deterministic suspicion on a step, produced by static rules (no LLM).

    Attributes:
        code: Machine-readable rule name (e.g. "tool_error", "llm_truncated", "empty_retrieval").
        severity: "error" (the step itself failed) or "warn" (degraded / suspicious but not fatal).
        detail: Human-readable one-liner with the concrete values behind the verdict.
    """
    code: str
    severity: Severity
    detail: str


@dataclass
class TraceStep:
    """One event of a run, enriched for display and diagnosis.

    Attributes:
        index: Ordinal within the run (0-based, file order); the "#N" that diagnoses cite.
        type: Event type (one of trace_events.ALL_EVENT_TYPES, or an unknown string from a drifted producer).
        step_index: The framework's own agent-loop step correlation (may be None for bare calls).
        summary: Deterministic one-line rendering of the event's key fields.
        findings: Static-rule suspicions attached to this event (empty = nothing noteworthy).
        event: The full (already redacted) event dict, kept verbatim for drill-down.
    """
    index: int
    type: str
    step_index: Optional[int]
    summary: str
    findings: list[Finding]
    event: dict


@dataclass
class RunStats:
    """Aggregates over one run: call counts, token / latency totals, finding counts."""
    steps: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0
    errors: int = 0
    warnings: int = 0


@dataclass
class TraceRun:
    """One agent run reconstructed from the trace: ordered steps + aggregate stats.

    run_id is None for events emitted outside a run context (e.g. a bare retriever.retrieve call);
    they are grouped together so nothing in the file is silently dropped.
    """
    run_id: Optional[str]
    steps: list[TraceStep] = field(default_factory=list)
    stats: RunStats = field(default_factory=RunStats)


def parse_trace(source: Union[str, Iterable[dict]]) -> list[TraceRun]:
    """Parse a trace into runs: group by run_id (first-seen order), keep event order, attach summaries/findings/stats.

    Args:
        source: Either the text of a JSONL trace file (one JSON event per line, JsonlExporter output),
            or an iterable of event dicts (e.g. MemoryExporter.events / tracer.events).

    Returns:
        list[TraceRun]: One entry per run_id in first-seen order; events without run_id form a final
        TraceRun with run_id=None (appended last, after all identified runs).

    Raises:
        TraceParseError: Empty input, a line that is not valid JSON, an event that is not a dict,
            or an event missing the "type" field.
    """
    events = _read_jsonl(source) if isinstance(source, str) else list(source)
    if not events:
        raise TraceParseError("trace is empty: no events found")
    runs: dict[Optional[str], TraceRun] = {}
    order: list[Optional[str]] = []
    for position, event in enumerate(events):
        if not isinstance(event, dict):
            raise TraceParseError(f"event {position}: expected a JSON object, got {type(event).__name__}")
        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            raise TraceParseError(f"event {position}: missing \"type\" field (not an agentmaker trace event)")
        run_id = event.get("run_id")
        if run_id not in runs:
            runs[run_id] = TraceRun(run_id=run_id)
            order.append(run_id)
        run = runs[run_id]
        step = TraceStep(index=len(run.steps), type=event_type, step_index=event.get("step_index"),
                         summary=_summarize(event), findings=_findings(event), event=event)
        run.steps.append(step)
        _accumulate(run.stats, step)
    # Bare events (run_id=None) go last: they are background noise relative to identified runs.
    ordered = [runs[rid] for rid in order if rid is not None]
    if None in runs:
        ordered.append(runs[None])
    return ordered


def load_trace(path: str) -> list[TraceRun]:
    """Read a JSONL trace file from disk and parse it (thin convenience over parse_trace)."""
    with open(path, encoding="utf-8") as f:
        return parse_trace(f.read())


def pick_run(runs: list[TraceRun], run_id: Optional[str] = None) -> TraceRun:
    """Select the run to diagnose: by run_id when given, otherwise the last run (the most recent one in the file).

    Raises:
        TraceParseError: runs is empty, or run_id was given but no run in the list carries it.
    """
    if not runs:
        raise TraceParseError("no runs to pick from")
    if run_id is None:
        return runs[-1]
    for run in runs:
        if run.run_id == run_id:
            return run
    raise TraceParseError(f"run_id {run_id!r} not found in trace ({len(runs)} runs)")


def render_run(run: TraceRun, *, max_chars: int = 20_000) -> str:
    """Render one run as a compact text timeline (the exact input the diagnosis LLM sees; also fine for terminals).

    One line per step ("#N type key=value ..."), findings indented below as "!! severity code: detail".
    Over budget, middle steps WITHOUT findings are elided ("... N steps omitted ...") keeping the head,
    the tail and every step that has findings; as a last resort the text is hard-cut at max_chars.

    Args:
        run: The run to render.
        max_chars: Character budget for the whole rendering (keeps the LLM prompt bounded).

    Returns:
        str: The timeline text, always starting with a one-line stats header.
    """
    header = (f"run {run.run_id or '(no run id)'}: {run.stats.steps} steps, {run.stats.llm_calls} llm_calls, "
              f"{run.stats.tool_calls} tool_calls, {run.stats.total_tokens} tokens, "
              f"{run.stats.total_latency_ms} ms total, {run.stats.errors} errors, {run.stats.warnings} warnings")
    lines_per_step = [_step_lines(step) for step in run.steps]
    full = "\n".join([header] + [line for lines in lines_per_step for line in lines])
    if len(full) <= max_chars:
        return full
    # Elide middle steps without findings: keep the first 12 / last 8 (run start and end carry the most
    # context) plus every step with findings (they are the likely diagnosis targets).
    keep = set(range(min(12, len(run.steps)))) | set(range(max(0, len(run.steps) - 8), len(run.steps)))
    keep |= {step.index for step in run.steps if step.findings}
    lines, omitted = [header], 0
    for step in run.steps:
        if step.index in keep:
            if omitted:
                lines.append(f"... {omitted} steps omitted ...")
                omitted = 0
            lines.extend(lines_per_step[step.index])
        else:
            omitted += 1
    if omitted:
        lines.append(f"... {omitted} steps omitted ...")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text
    suffix = "\n... [truncated]"
    return text[:max_chars - len(suffix)] + suffix   # Hard cut sized so the total stays within max_chars.


def _read_jsonl(text: str) -> list[dict]:
    """Parse JSONL text into a list of events; blank lines are skipped, a bad line raises with its 1-based number."""
    events = []
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise TraceParseError(f"line {number}: not valid JSON ({e.msg}); expected one JSON event per line "
                                  "(JsonlExporter output)") from None
    return events


def _short(value, limit: int = 120) -> str:
    """Render a field value on one line, truncated to limit chars (tracer already caps strings, this is a second belt)."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


def _summarize(event: dict) -> str:
    """One deterministic line with the event's key fields (per-type shapes follow trace_events conventions)."""
    event_type = event.get("type")
    if event_type == EVENT_LLM_CALL:
        usage = event.get("usage")
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        parts = [f"model={event.get('model')}", f"latency={event.get('latency_ms')}ms"]
        if tokens is not None:
            parts.append(f"tokens={tokens}")
        if event.get("finish_reason") is not None:
            parts.append(f"finish={event.get('finish_reason')}")
        if event.get("has_tool_calls"):
            parts.append("requested_tools=yes")
        if event.get("streamed"):
            parts.append("streamed=yes")
        if event.get("origin"):
            parts.append(f"origin={event.get('origin')}")
        return " ".join(parts)
    if event_type == EVENT_TOOL_CALL:
        parts = [f"tool={event.get('tool')}", f"status={event.get('status')}"]
        if event.get("latency_ms") is not None:
            parts.append(f"latency={event.get('latency_ms')}ms")
        parts.append(f"params={_short(event.get('params'))}")
        parts.append(f"result={_short(event.get('result'))}")
        return " ".join(parts)
    if event_type in (EVENT_MEMORY_SEARCH, EVENT_RAG_RETRIEVE):
        parts = [f"query={_short(event.get('query'), 80)}", f"hits={event.get('hits')}"]
        if event.get("latency_ms") is not None:
            parts.append(f"latency={event.get('latency_ms')}ms")
        return " ".join(parts)
    if event_type == EVENT_CONTEXT_BLOCK:
        return f"query={_short(event.get('query'), 80)} block_chars={event.get('block_chars')}"
    if event_type in (EVENT_CONTEXT_REDUCE, EVENT_CONTEXT_COMPACT):
        prefix = f"paradigm={event.get('paradigm')} " if event.get("paradigm") else ""
        return f"{prefix}before={event.get('before')} after={event.get('after')}"
    if event_type == EVENT_INDEX_SYNC_PENDING:
        return f"op={event.get('op')} count={event.get('count')}"
    if event_type == EVENT_INDEX_SYNC_RECONCILE:
        return f"items={event.get('items')} pending_after={event.get('pending_after')}"
    # Degradation markers carry little payload; generic fallback also covers unknown (drifted) event types.
    extras = {k: v for k, v in event.items() if k not in ("type", "run_id", "step_index") and v is not None}
    return " ".join(f"{k}={_short(v, 60)}" for k, v in extras.items()) or "(no fields)"


def _findings(event: dict) -> list[Finding]:
    """Static checks over one event; each hit becomes a Finding the timeline highlights and the LLM can trust as fact."""
    event_type = event.get("type")
    found: list[Finding] = []
    if event_type == EVENT_TOOL_CALL:
        status, tool = event.get("status"), event.get("tool")
        if status in _TOOL_FAILED:
            label = "was called with invalid arguments" if status == "invalid_args" else "failed"
            found.append(Finding("tool_error", "error",
                                 f"tool '{tool}' {label} (status={status}): {_short(event.get('result'), 160)}"))
        elif status in _TOOL_BLOCKED:
            found.append(Finding("tool_blocked", "warn",
                                 f"tool '{tool}' was blocked before running (status={status}): {_short(event.get('result'), 160)}"))
        elif status == "partial":
            found.append(Finding("tool_partial", "warn",
                                 f"tool '{tool}' succeeded only partially: {_short(event.get('result'), 160)}"))
    elif event_type == EVENT_LLM_CALL and event.get("finish_reason") in _TRUNCATION_REASONS:
        found.append(Finding("llm_truncated", "error",
                             f"LLM output was cut off (finish_reason={event.get('finish_reason')}): the reply is "
                             "incomplete; raise max_tokens / desired_output_tokens or shrink the context"))
    elif event_type in (EVENT_MEMORY_SEARCH, EVENT_RAG_RETRIEVE) and event.get("hits") == 0:
        kind = "memory search" if event_type == EVENT_MEMORY_SEARCH else "RAG retrieval"
        found.append(Finding("empty_retrieval", "warn",
                             f"{kind} returned 0 hits for query={_short(event.get('query'), 80)}: downstream "
                             "answers were generated without this evidence"))
    elif event_type == EVENT_SUMMARIZE_FAILED:
        found.append(Finding("compaction_degraded", "warn",
                             "history/trace compaction LLM failed: the run continued with degraded context management"))
    elif event_type in (EVENT_RAG_QUERY_TRANSFORM_FAILED, EVENT_RAG_CONTEXTUALIZE_FAILED):
        origin = f" ({event.get('origin')})" if event.get("origin") else ""
        found.append(Finding("rag_degraded", "warn",
                             f"a RAG enhancement step failed{origin}: retrieval quality degraded for this run"))
    elif event_type == EVENT_INDEX_SYNC_PENDING:
        found.append(Finding("index_sync_pending", "warn",
                             f"a derived-index write failed and was marked pending (op={event.get('op')}, "
                             f"count={event.get('count')}): retrieval may serve stale results until reconciled"))
    elif event_type == EVENT_INDEX_SYNC_RECONCILE and (event.get("pending_after") or 0) > 0:
        found.append(Finding("index_not_converged", "warn",
                             f"index reconciliation left {event.get('pending_after')} rows still pending"))
    elif event_type not in ALL_EVENT_TYPES:
        found.append(Finding("unknown_event", "warn",
                             f"unknown event type '{event_type}': the trace may come from a different "
                             "agentmaker version than this parser"))
    return found


def _accumulate(stats: RunStats, step: TraceStep) -> None:
    """Fold one step into the run stats (same usage/latency reading as Tracer.summary, kept consistent)."""
    stats.steps += 1
    if step.type == EVENT_LLM_CALL:
        stats.llm_calls += 1
    elif step.type == EVENT_TOOL_CALL:
        stats.tool_calls += 1
    usage = step.event.get("usage")
    if isinstance(usage, dict):
        stats.total_tokens += usage.get("total_tokens") or 0
    stats.total_latency_ms += step.event.get("latency_ms") or 0
    stats.errors += sum(1 for f in step.findings if f.severity == "error")
    stats.warnings += sum(1 for f in step.findings if f.severity == "warn")


def _step_lines(step: TraceStep) -> list[str]:
    """Timeline lines for one step: the "#N type summary" line plus one indented "!!" line per finding."""
    lines = [f"#{step.index} {step.type} {step.summary}"]
    lines.extend(f"   !! {f.severity} {f.code}: {f.detail}" for f in step.findings)
    return lines
