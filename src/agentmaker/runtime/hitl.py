"""agentmaker.runtime.hitl: data structures and internal signals for HITL async interrupts.

A suspend means a high-risk action is awaiting asynchronous human approval: when the executor
(harness.exec_tool) hits a tool that needs confirmation (tool.needs_confirmation(parameters), which defaults
to requires_confirmation; multi-action tools can decide per action) and there is no decision for that action
this turn, it raises ApprovalRequired (a control-flow signal, not an error). The policy loop catches it,
packages the state, and returns an Interrupt to the caller; the caller later calls resume(decision) to continue.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:                       # For type annotations only, not imported at runtime (interrupt does not hard-depend on retrieval).
    from ..retrieval.scope import Scope


@dataclass
class PendingAction:
    """A high-risk action awaiting human approval.

    Attributes:
        tool_name: The name of the tool to execute.
        arguments: The call arguments.
        call_id: The unique identifier of this tool call (the function-calling tool_call id); resume matches decisions against it.
    """
    tool_name: str
    arguments: dict
    call_id: str


@dataclass
class Interrupt:
    """The suspend result of run / resume: the pending actions plus the credentials needed to resume.

    Receiving it means "this turn is suspended, awaiting your approval": read `pendings` (there may be more than
    one), show them to a human, and resume once decided.
    - Single action: `agent.resume(True, scope=interrupt.scope)` (True approves / False rejects; pass scope by keyword).
    - Multiple actions (one turn requested several high-risk tools, or parallel sub-agents each suspended):
      `agent.resume({call_id: True/False, ...}, scope=...)` approves / rejects per call_id; you may also pass a
      single bool to uniformly approve / reject all pending actions.

    Attributes:
        pendings: The list of actions awaiting approval (one suspend may contain several).
        scope: The resume credential; resume uses it to load the suspended state back from the CheckpointStore (required for multi-session, may be empty for single-session).

    Convenience: `.pending` returns the first pending action (single-action case), or None if there are none.
    """
    pendings: list  # list[PendingAction]
    scope: "Optional[Scope]" = None

    def __post_init__(self):
        """Accept a single PendingAction passed directly (Interrupt(pending, scope)): normalize it into a list."""
        if isinstance(self.pendings, PendingAction):
            self.pendings = [self.pendings]

    @property
    def pending(self) -> "Optional[PendingAction]":
        """The first pending action (convenient access for the single-action case); None if there are none."""
        return self.pendings[0] if self.pendings else None


class ApprovalRequired(Exception):
    """An internal control-flow signal (not an error): harness.exec_tool raises it in HITL mode when it hits a
    high-risk tool with no decision. The policy loop catches it, packages the state, and turns it into an
    Interrupt. Analogous to the OpenAI Agents SDK's RunToolApprovalItem.
    """

    def __init__(self, pending: PendingAction):
        """
        Args:
            pending: The pending action that triggered the suspend (tool name / arguments / call_id).
        """
        self.pending = pending
        super().__init__(f"tool '{pending.tool_name}' requires human approval before it can execute")
