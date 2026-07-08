"""agentmaker.runtime.observability: observability (tracing / trace).

Records each step's LLM call and tool call of an Agent as a structured trace, for debugging and cost auditing.
Secrets and sensitive paths are automatically redacted before writing (CLAUDE.md §8 red line). Not attached by
default (zero overhead); inject a Tracer to observe. Where events go is decided by a pluggable TraceExporter
(memory / JSONL / SQLite / OTel).
"""

from .exporters import (
    JsonlExporter, MemoryExporter, OTelExporter, SqliteExporter, TraceExporter,
)
from ..execution.run_context import correlation, current_run_id   # The governance context lives in execution; the trace side re-exports the correlation API back from there (observability -> execution is the forward direction).
from .tracer import Tracer

__all__ = [
    "Tracer",
    "TraceExporter", "MemoryExporter", "JsonlExporter", "SqliteExporter", "OTelExporter",
    "correlation", "current_run_id",
]
