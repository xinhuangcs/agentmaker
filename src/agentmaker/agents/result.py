"""agentmaker.agents.result: RunResult, the unified return envelope for run.

run / arun / resume / aresume return one RunResult containing final output, interruption state, usage,
new messages, and run_id. A suspension is represented by `status == "interrupted"`, so callers cannot
confuse it with a final answer. This mirrors OpenAI Agents SDK's RunResult and Pydantic AI's
AgentRunResult.

Layering: RunResult is only wrapped at the outermost run boundary; inside a strategy (`_arun` /
`_adrive` / `_absorb_child`) the bare `str | instance | Interrupt` is still used as the control-flow
signal, with no awareness of RunResult.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from ..core.message import Message
    from ..runtime.hitl import Interrupt

# The two terminal states of a run: completed (produced a final output) / interrupted (a high-risk HITL action is suspended awaiting approval and needs resume to continue).
RunStatus = Literal["completed", "interrupted"]


@dataclass(frozen=True)
class RunUsage:
    """Usage snapshot of a single run (for cost accounting / limit observability).

    Attributes:
        llm_calls: total LLM calls accumulated over this run.
        tool_calls: tool calls actually executed during this run.
        total_tokens: total tokens accumulated over this run (sum of usage.total_tokens across each LLM response).
    """
    llm_calls: int = 0
    tool_calls: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class RunResult:
    """Unified return envelope for run / arun / resume / aresume: bundles final output + whether a
    human-approval interrupt occurred + usage + new messages this turn into one object (rather than
    returning a bare string), turning a suspension from a silent accident into an explicitly
    checkable state.

    Consumption template:
        r = agent.run("...")
        if r.interrupted:
            handle(r.interrupt)            # HITL: take the suspended state to resume
        else:
            use(r.final_output)            # completed: take the final output (str or structured instance)

    Attributes:
        final_output: the completed state's final output (str, or a structured instance when output_schema was passed); None when suspended.
        status: "completed" or "interrupted".
        interrupt: the suspended state's Interrupt (the pending action + the resume scope); None when completed.
        usage: usage snapshot of this run (RunUsage).
        new_messages: messages newly added and stored in history this turn (user + assistant); empty when suspended.
        run_id: this run's trace correlation id.
    """
    final_output: Any = None
    status: RunStatus = "completed"
    interrupt: "Optional[Interrupt]" = None
    usage: RunUsage = field(default_factory=RunUsage)
    new_messages: "tuple[Message, ...]" = ()
    run_id: Optional[str] = None

    @property
    def interrupted(self) -> bool:
        """Whether this is a HITL suspended state (status == "interrupted")."""
        return self.status == "interrupted"

    def __str__(self) -> str:
        """Show final output text directly and a readable note for interrupted results."""
        if self.status == "interrupted":
            pendings = self.interrupt.pendings if self.interrupt is not None else []
            if not pendings:
                return "<RunResult interrupted: awaiting human approval>"
            name = pendings[0].tool_name
            extra = f" and {len(pendings)} actions" if len(pendings) > 1 else ""
            return f"<RunResult interrupted: {name}{extra} awaiting human approval>"
        from .base import BaseAgent
        return BaseAgent._output_text(self.final_output)
