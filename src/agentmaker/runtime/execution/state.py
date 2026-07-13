"""agentmaker.runtime.execution.state: the unified execution state ExecutionState.

A single representation of "this run's execution trajectory" shared by all four paradigms (Chat/ReAct/Reflection/Plan),
replacing each paradigm's private messages / scratchpad / trajectory. Cross-cutting concerns (HITL / compaction / trace
/ persistence) are all built on it and handled in the unified layer without touching per-paradigm control flow. Aligned
with LangGraph's unified State / OpenAI's items list.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from ...core.exceptions import SessionError
from ..hitl import PendingAction

# Increment for incompatible ExecutionState serialization changes.
CHECKPOINT_FORMAT_VERSION = 2


def checkpoint_format_version(raw: str) -> Optional[int]:
    """Probe the checkpoint format version without fully deserializing; a missing version returns None."""
    try:
        v = json.loads(raw).get("v")
    except (json.JSONDecodeError, AttributeError):
        return None
    return v if isinstance(v, int) else None


@dataclass
class ExecutionState:
    """The unified execution state of one run.

    Unified fields (shared by all paradigms, the only ones cross-cutting concerns recognize):
        messages: The execution trajectory (OpenAI dicts: system / user / assistant[with tool_calls] / tool).
        input_text: This run's user input (stored into conversation history on completion).
        remaining: How many more LLM calls are allowed (guards against exceeding the iteration limit).
        decisions: The HITL decision table {call_id: bool}.
        pending: The LIST of HITL actions currently suspended awaiting approval (shown to the caller); an empty list when not suspended. One suspend can contain multiple
            (multiple high-risk tools requested in one turn, or parallel sub-agents each suspending), and resume injects decisions by call_id.
        completed: Whether this run has finished (during finalization, mark first, then clear the checkpoint). A leftover checkpoint with completed=True means "finished, only cleanup remains";
            _guard_pending / _load_execution_state use this to clear it directly rather than re-run, resolving the double-execution / double-bookkeeping window from "persisting history and clearing the checkpoint are not atomic".
    Paradigm-specific:
        meta: Each paradigm's own resume state (e.g. fc's pending_calls, ReAct's step index, Plan's steps/cursor).
            This is where the "per-paradigm snapshot/restore hook" lands: the paradigm-private bit of state beyond the unified fields.
    Trace correlation:
        run_id: This run's trace correlation id; saved on suspend, restored on resume (continuing the same run_id across suspend).
        step: The trace step number this run has reached; same, used for resume continuation (see execution/run_context).
    """

    messages: list[dict]
    input_text: str
    remaining: int = 0
    decisions: dict = field(default_factory=dict)
    pending: list = field(default_factory=list)   # list[PendingAction]; an empty list when not suspended.
    meta: dict = field(default_factory=dict)
    run_id: Optional[str] = None
    step: int = 0
    completed: bool = False

    def to_json(self) -> str:
        """Serialize to a JSON string (stored into CheckpointStore); pending is flattened into plain fields.

        The unified fields are all JSON-friendly types; the paradigm-private meta / messages / pending.arguments are
        filled by each paradigm, and if they store a non-serializable object, this wraps the bare TypeError into a
        clear SessionError (pointing to where to look), making it easy to locate which paradigm wrote dirty state.
        """
        try:
            return json.dumps({
                "v": CHECKPOINT_FORMAT_VERSION,
                "messages": self.messages, "input_text": self.input_text, "remaining": self.remaining,
                "decisions": self.decisions, "meta": self.meta, "run_id": self.run_id, "step": self.step,
                "completed": self.completed,
                "pending": [{"tool_name": p.tool_name, "arguments": p.arguments, "call_id": p.call_id}
                            for p in self.pending],
            }, ensure_ascii=False)
        except (TypeError, ValueError) as e:   # ValueError: a circular reference raises ValueError, which would otherwise escape as a bare exception.
            raise SessionError(
                f"ExecutionState contains content that cannot be JSON-serialized (check that meta / messages / pending.arguments are basic types and free of circular references): {e}") from e

    @classmethod
    def from_json(cls, raw: str) -> "ExecutionState":
        """Restore from a JSON string (used when CheckpointStore loads it back)."""
        d = json.loads(raw)
        return cls(messages=d["messages"], input_text=d["input_text"], remaining=d.get("remaining", 0),
                   decisions=d.get("decisions", {}), meta=d.get("meta", {}),
                   run_id=d.get("run_id"), step=d.get("step", 0), completed=d.get("completed", False),
                   pending=[PendingAction(**p) for p in (d.get("pending") or [])])
