"""agentmaker.runtime: the agent runtime machinery (orchestrated / used cross-cuttingly by agents).

Houses the harness (the cross-cutting choke point for LLM / tool calls) plus 6 cross-cutting
capabilities: guardrails / hitl / hooks / execution / sessions / observability. They depend only on
the foundation layer (core / retrieval / tools) and together form the middle layer of the
foundation -> runtime -> agents stack. The harness is the coordinator among them (not a container):
it orchestrates these capabilities at runtime, but the others are used directly by `agents` as well,
so they live in runtime as peers of the harness.
"""

from .harness import Harness, cli_confirm
from .guardrails import CallableGuardrail, Guardrail, GuardrailResult
from .hitl import ApprovalRequired, Interrupt, PendingAction
from .hooks import Hook
from .sessions import ConversationSearch, ConversationSearchTool, ScopeSummary, SessionStore, SqliteSessionStore
from .execution import (
    CheckpointStore, ExecutionState, RunPolicy, SqliteCheckpointStore,
    current_run_id, current_scope, current_step, current_trace_carrier, governed_chat,
)
from .observability import (
    JsonlExporter, MemoryExporter, OTelExporter, SqliteExporter, Tracer, TraceExporter,
)

__all__ = [
    "Harness", "cli_confirm",
    "CallableGuardrail", "Guardrail", "GuardrailResult",
    "ApprovalRequired", "Interrupt", "PendingAction",
    "Hook",
    "SessionStore", "SqliteSessionStore", "ConversationSearch", "ConversationSearchTool", "ScopeSummary",
    "CheckpointStore", "ExecutionState", "RunPolicy", "SqliteCheckpointStore",
    # Run-level context: current_run_id/current_scope/current_step are for correlation; current_trace_carrier
    # lets OTelExporter attach to an upstream trace; governed_chat is the governance entry point for bypass
    # LLM calls that skip the Harness (the normal path gets governance for free via Harness.acall_llm /
    # aexec_tool; only hand-written recipes or direct bypass calls need governed_chat).
    "current_run_id", "current_scope", "current_step", "current_trace_carrier", "governed_chat",
    "JsonlExporter", "MemoryExporter", "OTelExporter", "SqliteExporter", "Tracer", "TraceExporter",
]
