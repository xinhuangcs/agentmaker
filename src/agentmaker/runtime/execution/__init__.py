"""agentmaker.runtime.execution: unified execution state, its persistence, and resume mechanism (shared by all four paradigms).

`ExecutionState` replaces each paradigm's private messages / scratchpad / trajectory; cross-cutting concerns (HITL /
compaction / trace) are built on it and handled in the unified layer without touching per-paradigm control flow.
`CheckpointStore` persists ExecutionState by scope, shared across HITL / crash recovery / long-task resume (aligned
with LangGraph's checkpointer).
"""

from .checkpoint import CheckpointStore, SqliteCheckpointStore
from .run_policy import RunPolicy
from .state import ExecutionState
# Import run_context after run_policy / checkpoint / state (it depends on .run_policy) to avoid an import cycle during package init.
from .run_context import (check_limits, correlation, current_run_id, current_scope, current_step,
                          current_trace_carrier, enforce_token_limit_after_llm, governed_chat, new_run_id,
                          record_llm, record_tool, reset_run, start_run)

__all__ = ["ExecutionState", "CheckpointStore", "SqliteCheckpointStore", "RunPolicy",
           # Run-level context (trace correlation + RunPolicy governance); governed_chat is the governance entry point for bypass LLM calls that skip the Harness.
           "governed_chat", "current_run_id", "current_scope", "current_step", "current_trace_carrier",
           "correlation", "new_run_id", "start_run", "reset_run", "check_limits", "record_llm", "record_tool",
           "enforce_token_limit_after_llm"]
