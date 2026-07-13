"""agentmaker.core.trace_events: trace event type constants (single source of truth).

Trace events are schema-less free-form dicts, but the `type` field is the contract between producers,
third-party exporters, and alerting. This module defines the supported event types as constants and
provides a complete registry for consumers.

Common event field conventions (redacted before reaching each exporter; nullable fields per the table below):
    All events: type (always present). run_id / step_index exist only within a run context (injected by
        correlation(); a bare call such as a standalone retriever.retrieve has neither).
    EVENT_LLM_CALL: model, latency_ms, usage (dict or None, often missing when streaming),
        finish_reason (truncation observability: length / max_tokens etc.; carried by all three
        production paths: non-streaming / streaming / bypass), has_tool_calls (bool, non-streaming only),
        streamed (bool, streaming only), origin (present only for the governed_chat bypass).
    EVENT_TOOL_CALL: tool, params, status, latency_ms, result (tool result text).
    EVENT_RUN_ERROR: error_type, message.
    EVENT_CONTEXT_BLOCK: query, block_chars.  EVENT_CONTEXT_REDUCE: paradigm, before, after.
    EVENT_CONTEXT_COMPACT: before, after (message counts before/after history compaction; same units as
        REDUCE, emitted only when compaction actually happens).
    EVENT_SUMMARIZE_FAILED: (correlation only): observable signal that history/trace compaction LLM keeps failing.
    EVENT_MEMORY_SEARCH: query, hits.  EVENT_RAG_RETRIEVE: query, hits, latency_ms.
    EVENT_RAG_QUERY_TRANSFORM_FAILED: origin (rag.mqe / rag.hyde).  EVENT_RAG_CONTEXTUALIZE_FAILED: (correlation only).
    EVENT_INDEX_SYNC_PENDING: op (index/drop/reconcile), count.  EVENT_INDEX_SYNC_RECONCILE: items
        (rows re-indexed), pending_after (rows still pending after reconciliation, 0 = fully converged).
"""

# LLM / tools (Harness cross-cutting)
EVENT_LLM_CALL = "llm_call"                              # one LLM call (including the governed_chat bypass)
EVENT_TOOL_CALL = "tool_call"                            # one tool execution
EVENT_RUN_ERROR = "run_error"                            # one uncaught run/resume exception

# Context engineering (Harness)
EVENT_CONTEXT_BLOCK = "context_block"                    # assembling memory/RAG retrieval blocks
EVENT_CONTEXT_REDUCE = "context_reduce"                  # loss-aware trace reduction happened
EVENT_CONTEXT_COMPACT = "context_compact"               # history compaction actually happened (summarizing old turns, symmetric with REDUCE)
EVENT_SUMMARIZE_FAILED = "summarize_failed"              # history/trace compaction LLM failed, degraded

# memory / RAG
EVENT_MEMORY_SEARCH = "memory_search"                    # memory three-dimensional retrieval
EVENT_RAG_RETRIEVE = "rag_retrieve"                      # RAG retrieval
EVENT_RAG_QUERY_TRANSFORM_FAILED = "rag_query_transform_failed"   # multi-query expansion / HyDE failed, degraded
EVENT_RAG_CONTEXTUALIZE_FAILED = "rag_contextualize_failed"       # per-chunk contextual enrichment at ingest failed, degraded

# Index sync bookkeeping
EVENT_INDEX_SYNC_PENDING = "index_sync_pending"          # derived index write failed, marked pending
EVENT_INDEX_SYNC_RECONCILE = "index_sync_reconcile"      # full reconciliation (delete orphans + re-index)

# All event types (third-party exporters / alerting enumerate over this; always register new events to keep the single source of truth)
ALL_EVENT_TYPES = frozenset({
    EVENT_LLM_CALL, EVENT_TOOL_CALL, EVENT_RUN_ERROR, EVENT_CONTEXT_BLOCK, EVENT_CONTEXT_REDUCE,
    EVENT_CONTEXT_COMPACT, EVENT_SUMMARIZE_FAILED, EVENT_MEMORY_SEARCH, EVENT_RAG_RETRIEVE,
    EVENT_RAG_QUERY_TRANSFORM_FAILED, EVENT_RAG_CONTEXTUALIZE_FAILED, EVENT_INDEX_SYNC_PENDING,
    EVENT_INDEX_SYNC_RECONCILE,
})
