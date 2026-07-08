"""agentmaker.runtime.execution.run_policy: run governance (RunPolicy): a single run's limits and cancellation.

A general runtime governance layer (not a business capability): sets GLOBAL limits for a single run (wall time, LLM
call count, tool call count, tokens) plus a cooperative cancellation hook. It gates at each point where the harness
calls an LLM / executes a tool (count + check), and raises `RunLimitExceeded` / `RunCancelled` to abort the run on
over-limit / cancellation. Counts / timing live in the run context (contextvars, no cross-run leakage) and are
independent per run (a resume is a new run, limits reset).

Deliberately narrowed:
    - No retry / backoff: that belongs to `LLMClient` (which already has a timeout) / an individual Tool.
    - No circuit breaker: that mostly belongs to the app / ops.
    - No per-step limit: each paradigm's `max_turns` / `max_steps` already covers it (this class is the global cross-paradigm / cross-nesting limit).
    - No cost limit: cost = tokens x unit price, and unit price is app configuration; use `max_tokens` as the framework-side proxy, with cost mapped by the app.
"""

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class RunPolicy:
    """A single run's limits and cancellation (attached to an Agent, counted independently per run; a None field = that item is unlimited).

    Nested semantics: limits are counted against the OUTERMOST run globally. A nested child Agent's own RunPolicy
    (AgentTool delegation / Plan/Reflection child executor) does not take effect inside the parent run (it warns). To
    limit subtasks, set the global limits on the parent Agent's run_policy.

    Fields:
        max_llm_calls: Maximum LLM calls in a run (including streaming, including nesting such as Plan child executors); must be >= 1 (a run must call the LLM at least once).
        max_tool_calls: Maximum tools ACTUALLY executed in a run (those blocked by permission / confirmation are not counted); must be >= 0 (0 = tools disabled this run:
            the LLM can still be called, and the run aborts the moment the model wants to execute a tool: a hard limit for a "read-only / safe mode").
        max_tokens: Cumulative token limit for a run (the sum of usage.total_tokens across LLM responses); must be >= 1.
        deadline_seconds: A run's wall-time limit (measured from the start of the run; a resume is a new run and re-times); must be > 0.
        cancel: Cooperative cancellation hook `() -> bool`, checked before each LLM / tool call, returning True aborts. Must be fast and non-blocking;
            when one instance serves multiple sessions, the callback can call `current_run_id()` to identify which run it is and decide accordingly (cancel per run).
    """
    max_llm_calls: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_tokens: Optional[int] = None
    deadline_seconds: Optional[float] = None
    cancel: Optional[Callable[[], bool]] = None

    def __post_init__(self):
        """Validate at construction, catching meaningless configuration on the spot (otherwise it drags into mid-run and surfaces as confusing errors like "exceeded ... limit N").

        Per-field lower bounds, rather than a blanket "non-negative":
            - max_llm_calls / max_tokens must be >= 1: 0 would abort at the run's first check (`0 >= 0`) and never produce a result, which is a pure misconfiguration
              to be caught on the spot just like a negative value (this is exactly the intent of this validation).
            - max_tool_calls must be >= 0: 0 is meaningful (tools disabled this run, LLM still callable), a natural extension of the "limit" semantics, so it is not blocked.
            - deadline_seconds must be > 0; cancel must be callable. A bool is not treated as a valid integer / numeric config.
        """
        for name, lo in (("max_llm_calls", 1), ("max_tool_calls", 0), ("max_tokens", 1)):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < lo):
                raise ValueError(f"RunPolicy.{name} must be an integer >= {lo} or None, got {v!r}")
        d = self.deadline_seconds
        if d is not None and (isinstance(d, bool) or not isinstance(d, (int, float)) or d <= 0):
            raise ValueError(f"RunPolicy.deadline_seconds must be a positive number or None, got {d!r}")
        if self.cancel is not None and not callable(self.cancel):
            raise ValueError(f"RunPolicy.cancel must be a callable or None, got {self.cancel!r}")
