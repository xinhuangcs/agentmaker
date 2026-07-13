"""agentmaker.runtime.execution.run_context: run-level context (trace correlation + run governance), propagated via contextvars.

A run's "environment state" is collected into this single contextvar, carrying two things:
    - trace correlation: `run_id` + `step_index`, so the events emitted by the Harness can be grouped / ordered by run (`correlation()`).
    - run governance: RunPolicy's counting / timing + limit checks (`check_limits` / `record_llm` / `record_tool`, see execution/run_policy).
It propagates via contextvars, so async tasks / thread pools (to_thread) are each isolated and concurrent sessions do
not leak into each other (aligned with the stateless Agent design). Rules:
    - Nested inheritance: a nested run (e.g. Plan's child executor) inherits the outer context (same run_id / step sequence / limit counts) rather than starting a new one.
    - Continuity across suspend: suspend / resume continues the same run_id + step (stored in ExecutionState); limits are independent per run (reset on resume).
Trace only does flat correlation (no span tree, left to OTel); governance only does narrow limits (no retry / circuit breaker, see run_policy).
"""

import contextvars
import logging
import uuid
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any, Optional

from ...core.exceptions import RunCancelled, RunLimitExceeded
from ...retrieval.scope import Scope          # Governance lives in execution, so the real type is available (retrieval does not import runtime, no cycle; checkpoint.py set the precedent).
from .run_policy import RunPolicy
from ...core.trace_events import EVENT_LLM_CALL

if TYPE_CHECKING:
    from ...core.llm_clients import LLMClient
    from ...core.llm_response import LLMResponse
    from ..observability import Tracer

_log = logging.getLogger(__name__)   # run_context deliberately does not hold a tracer (to avoid a reverse observability->execution dependency); warnings go through the stdlib logging.


@dataclass
class _RunContext:
    """A single run's context (stored in the contextvar): trace correlation + run-governance counts + this run's scope."""
    run_id: str
    step: int = 0
    policy: Optional[RunPolicy] = None   # This run's limit / cancellation configuration.
    scope: Optional[Scope] = None        # This run's session ownership; used by tools (e.g. AgentTool) to isolate by the parent run's scope.
    start: float = 0.0                   # This run's starting perf_counter (for computing wall time).
    llm_calls: int = 0
    tool_calls: int = 0
    tokens: int = 0
    trace_carrier: Optional[dict] = None  # Optional upstream W3C trace carrier ({"traceparent":..., "tracestate":...}); lets OTelExporter attach spans into the app's cross-service trace.


_current: contextvars.ContextVar[Optional[_RunContext]] = contextvars.ContextVar(
    "agentmaker_run_context", default=None)


def new_run_id() -> str:
    """Generate a new run_id (uuid4 hex)."""
    return uuid.uuid4().hex


def start_run(run_id: str, *, step: int = 0, policy: Optional[RunPolicy] = None, scope: Optional[Scope] = None,
              trace_carrier: Optional[dict] = None):
    """Start a run context (only when there is no active context); nested calls inherit the outer one instead of creating a new one.

    Args:
        run_id: The identifier of this run.
        step: Starting step (on resume, pass the previous step to continue; 0 on first start).
        policy: This run's RunPolicy (limits / cancellation); None means unlimited. Nested runs inherit the outer policy (a child executor's own policy is not counted separately, it is ignored with a warning).
        scope: This run's session ownership (Scope); stored in the context for tools to isolate by the parent run's scope (see current_scope). Nested runs inherit the outer scope.
        trace_carrier: Optional upstream W3C trace carrier (e.g. {"traceparent": "..."}); stored in the context for OTelExporter to attach spans into
            the app's cross-service trace (see current_trace_carrier). Nested runs inherit the outer one. NOT persisted across suspend/resume: a resume is a new request,
            and the app passes the current carrier via resume(trace_carrier=...) (so the resume segment's spans belong to the new request's trace, which is more accurate than reusing the old one).

    Returns:
        token: non-None means a new context was created (must call reset_run(token) at the end); None means the outer one was inherited (no reset needed).
    """
    existing = _current.get()
    if existing is not None:
        # Already inside a run (e.g. Plan's executor / AgentTool delegation) -> inherit the outer one (run_id / step / limit counts / scope / trace_carrier).
        # A nested child Agent's own run_policy does not take effect inside the parent run (it is only bound by the outermost policy's global counts). Silently ignoring is the most hidden failure, so warn for observability.
        if policy is not None and policy is not existing.policy:
            _log.warning("Nested run's run_policy is ignored: an inner Agent's own run_policy does not take effect inside the parent run, "
                         "it is only bound by the outermost policy's global counts (this happens for AgentTool delegation / Plan/Reflection child executors).")
        return None
    return _current.set(_RunContext(run_id=run_id, step=step, policy=policy, scope=scope, start=perf_counter(),
                                    trace_carrier=trace_carrier))


def reset_run(token) -> None:
    """End the run context (paired with start_run); a no-op if token is None (the outer context was inherited at the time)."""
    if token is not None:
        _current.reset(token)


def current_run_id() -> Optional[str]:
    """The current run's run_id; None if not inside a run context. An app / hook can read it to correlate its own logs."""
    ctx = _current.get()
    return ctx.run_id if ctx is not None else None


def current_step() -> int:
    """The step number the current run has reached (for persistence and resume continuation); 0 if not in a context."""
    ctx = _current.get()
    return ctx.step if ctx is not None else 0


def current_scope() -> Optional[Scope]:
    """The current run's scope (Scope); None if not inside a run context.

    Lets a tool fetch "the parent Agent's scope for the current run" during execution. The typical case is AgentTool
    delegating to a child Agent isolating history by the parent session, avoiding "multiple sessions sharing one tool
    instance -> child Agent history leaking across sessions". The Tool.run(parameters) signature cannot obtain scope,
    so it is passed through via this contextvar.
    """
    ctx = _current.get()
    return ctx.scope if ctx is not None else None


def current_trace_carrier() -> Optional[dict]:
    """The current run's upstream W3C trace carrier ({"traceparent": ...}); None if not in a run context or not passed.

    Typical use: `OTelExporter(carrier_provider=current_trace_carrier)`. Export and run are in the same async context
    (emit is an inline synchronous call), so what is read is the current run's carrier, making each AB span a child of
    the app's request span (rather than each becoming its own root).
    """
    ctx = _current.get()
    return ctx.trace_carrier if ctx is not None else None


def correlation() -> dict:
    """Get an event's correlation fields `{run_id, step_index}` (incrementing step); empty dict if not inside a run context.

    The Harness merges it when assembling each trace event (`{**event, **correlation()}`), so events can be grouped by run and ordered by step.
    """
    ctx = _current.get()
    if ctx is None:
        return {}
    ctx.step += 1
    return {"run_id": ctx.run_id, "step_index": ctx.step}


def snapshot_usage() -> dict:
    """Get a usage snapshot of the current run context {llm_calls, tool_calls, total_tokens}; all 0 if not in a run context.

    Used by RunResult during finalization to read this run's accumulated usage (purely read-only, does not increment step, does not change state). A nested run shares the outer counts (a child recipe reads the accumulated total
    including child steps); a resume is a new ctx with counts starting at 0, so usage reflects only this resume segment (consistent with RunPolicy's "reset on resume").
    The keys must match the RunUsage field names exactly (_RunContext.tokens maps to total_tokens).
    """
    ctx = _current.get()
    if ctx is None:
        return {"llm_calls": 0, "tool_calls": 0, "total_tokens": 0}
    return {"llm_calls": ctx.llm_calls, "tool_calls": ctx.tool_calls, "total_tokens": ctx.tokens}


# Run governance (RunPolicy limits / cancellation; the Harness calls check before each LLM / tool call and record after).

def check_limits(kind: str) -> None:
    """Check whether this run has exceeded a RunPolicy limit / been cancelled; if so, raise RunCancelled / RunLimitExceeded (aborting this run).

    Call it BEFORE each LLM call / tool execution. Not in a run context or no policy attached -> no-op (zero overhead).

    Args:
        kind: "llm" (before calling the model, checks the LLM call count) or "tool" (before executing a tool, checks the tool call count);
            token / cancellation / deadline are checked in both cases, to avoid continuing to execute tools after the token budget is exhausted.
    """
    ctx = _current.get()
    if ctx is None or ctx.policy is None:
        return
    p = ctx.policy
    if p.cancel is not None and p.cancel():
        raise RunCancelled("run was cancelled")
    check_deadline()
    if p.max_tokens is not None and ctx.tokens >= p.max_tokens:
        raise RunLimitExceeded(f"exceeded token limit {p.max_tokens}")
    if kind == "llm":
        if p.max_llm_calls is not None and ctx.llm_calls >= p.max_llm_calls:
            raise RunLimitExceeded(f"exceeded LLM call limit {p.max_llm_calls}")
    elif kind == "tool":
        if p.max_tool_calls is not None and ctx.tool_calls >= p.max_tool_calls:
            raise RunLimitExceeded(f"exceeded tool call limit {p.max_tool_calls}")


def check_deadline() -> None:
    """Cooperatively reject a run whose wall-clock deadline has elapsed; no-op when unbounded."""
    ctx = _current.get()
    if ctx is None or ctx.policy is None or ctx.policy.deadline_seconds is None:
        return
    if (perf_counter() - ctx.start) > ctx.policy.deadline_seconds:
        raise RunLimitExceeded(f"exceeded wall-clock time limit {ctx.policy.deadline_seconds}s")


def has_parallel_sensitive_limits() -> bool:
    """Whether tool batching could race with an active numeric run limit."""
    ctx = _current.get()
    if ctx is None or ctx.policy is None:
        return False
    policy = ctx.policy
    return any(limit is not None for limit in (
        policy.max_llm_calls, policy.max_tool_calls, policy.max_tokens,
    ))


def record_llm(usage=None) -> None:
    """Record one LLM call (count +1, accumulate usage.total_tokens); no-op if not in a run context. The Harness calls it after an LLM call."""
    ctx = _current.get()
    if ctx is None:
        return
    ctx.llm_calls += 1
    if isinstance(usage, dict):
        ctx.tokens += usage.get("total_tokens") or 0


def enforce_token_limit_after_llm() -> None:
    """Check the hard token limit immediately after the LLM returns and its usage is recorded; abort if exceeded, to avoid continuing to execute subsequent tools.

    The comparison operator is deliberately different from check_limits (not a typo): check_limits runs BEFORE the call
    with `>=` (do not open a new call once the limit is reached); this function runs AFTER recording this call's usage
    with `>` (counting this call's usage in, exactly equal to the limit is treated as "fully used, allowed", and only a
    true overshoot is blocked).
    """
    ctx = _current.get()
    if ctx is None or ctx.policy is None or ctx.policy.max_tokens is None:
        return
    if ctx.tokens > ctx.policy.max_tokens:
        raise RunLimitExceeded(f"exceeded token limit {ctx.policy.max_tokens}")


def record_tool() -> None:
    """Record one tool call that was ACTUALLY executed (count +1); no-op if not in a run context. The Harness calls it after tool execution."""
    ctx = _current.get()
    if ctx is not None:
        ctx.tool_calls += 1


def _governed_emit(tracer, resp, origin: str, start: float) -> None:
    """Assemble and emit one llm_call event (same shape as Harness._llm_event: also carries has_tool_calls / finish_reason, plus an origin field marking the bypass source)."""
    tracer.emit({"type": EVENT_LLM_CALL, "origin": origin,
                 "model": getattr(resp, "model", None),
                 "usage": getattr(resp, "usage", None),
                 "latency_ms": int((perf_counter() - start) * 1000),
                 "has_tool_calls": bool(getattr(resp, "tool_calls", None)),
                 "finish_reason": getattr(resp, "finish_reason", None),  # A bypass truncation is observable too (consistent with _llm_event).
                 **correlation()})


async def governed_chat(llm: "LLMClient", messages: list[dict], *, tracer: "Optional[Tracer]" = None,
                        origin: str = "", **kwargs: Any) -> "LLMResponse":
    """Governed llm.chat (async): check_limits -> await chat -> record_llm (count + tokens) -> optional trace -> hard token limit.

    Args:
        llm: LLMClient (or an object of the same shape).
        messages: Message list, passed to chat as-is.
        tracer: Optional tracer (duck-typed emit); zero overhead if not attached.
        origin: Origin label for the event (e.g. "memory.summary" / "rag.mqe"), to distinguish bypass calls in the trace.
        **kwargs: Passed through to llm.chat.

    Returns:
        LLMResponse.
    """
    check_limits("llm")
    start = perf_counter() if tracer is not None else 0.0
    resp = await llm.chat(messages, **kwargs)
    record_llm(getattr(resp, "usage", None))
    if tracer is not None:
        _governed_emit(tracer, resp, origin, start)
    enforce_token_limit_after_llm()
    check_deadline()
    return resp
