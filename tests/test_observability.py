"""Hermetic observability tests with no keys or network access.

Locks down the trace event contract (the events constants as the single source of truth),
SqliteExporter's dedicated run_id column and old-DB column backfill, and OTelExporter's
real-duration spans with optional upstream trace attachment.
"""

import sqlite3

import pytest

from agentmaker.core import trace_events as ev
from agentmaker.runtime.observability.exporters import OTelExporter, SqliteExporter


# ---------- event type contract (single source of truth) ----------

def test_event_constants_are_single_source():
    """ALL_EVENT_TYPES holds exactly every EVENT_* constant with unique values; producers all reference them (no scattered literals)."""
    consts = {v for k, v in vars(ev).items() if k.startswith("EVENT_")}
    assert consts == set(ev.ALL_EVENT_TYPES)                # constant set == registry (forgetting to register a new event fails here)
    assert len(consts) == len(ev.ALL_EVENT_TYPES) == 13     # no duplicate values
    assert ev.EVENT_LLM_CALL == "llm_call" and ev.EVENT_RUN_ERROR == "run_error"
    assert ev.EVENT_INDEX_SYNC_RECONCILE == "index_sync_reconcile"


# ---------- SqliteExporter: dedicated run_id column + old-DB backfill ----------

def test_sqlite_exporter_run_id_column_queryable():
    """After SqliteExporter persists an event, run_id lands in its own column and is queryable directly (no digging it out of JSON)."""
    sx = SqliteExporter(":memory:")
    sx.export({"type": "llm_call", "run_id": "abc123", "model": "m"})
    sx.export({"type": "tool_call", "run_id": "abc123", "tool": "calc"})
    sx.export({"type": "llm_call", "run_id": "other", "model": "m"})
    rows = sx._db.execute("SELECT type FROM traces WHERE run_id=? ORDER BY type", ("abc123",)).fetchall()
    assert [r[0] for r in rows] == ["llm_call", "tool_call"]
    sx.close()


def test_sqlite_exporter_adds_missing_run_id_column(tmp_path):
    """A traces table without run_id receives the safe additive column on open."""
    p = str(tmp_path / "traces.db")
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE traces(type TEXT, event TEXT, created_at TEXT)")
    c.commit()
    c.close()
    sx = SqliteExporter(p)                                  # doesn't raise: ensure_columns backfills the run_id column
    assert "run_id" in {r[1] for r in sx._db.execute("PRAGMA table_info(traces)")}
    sx.export({"type": "llm_call", "run_id": "r1"})         # writes normally after backfill
    assert sx._db.execute("SELECT run_id FROM traces").fetchone()[0] == "r1"
    sx.close()


# ---------- OTelExporter: real duration + optional upstream trace attachment ----------

def _otel_mem():
    """A fresh InMemorySpanExporter attached to the process-level TracerProvider (one per test, collecting only spans emitted after it attaches).

    set_tracer_provider is set-once (all OTel tests share one provider), so tests can't each build their own;
    instead we append a test-local processor + mem to the existing provider, and each test reads only its own,
    with no cross-talk.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    prov = trace.get_tracer_provider()
    if not isinstance(prov, TracerProvider):               # no real provider set yet (default is a Proxy) -> set one once
        prov = TracerProvider()
        trace.set_tracer_provider(prov)
    mem = InMemorySpanExporter()
    prov.add_span_processor(SimpleSpanProcessor(mem))
    return mem


def test_otel_exporter_self_roots_by_default():
    """By default (no carrier_provider): each event span is its own root (trace_ids differ, not derived from run_id), with run_id as an attribute;
    a span carrying latency has a real duration (not a zero-width point)."""
    pytest.importorskip("opentelemetry")
    mem = _otel_mem()
    ox = OTelExporter()
    run_id = "0123456789abcdef0123456789abcdef"             # 32 hex
    ox.export({"type": "llm_call", "run_id": run_id, "latency_ms": 100})
    ox.export({"type": "tool_call", "run_id": run_id, "latency_ms": 50})
    spans = mem.get_finished_spans()
    assert len(spans) == 2
    # each its own root: the two trace_ids differ and neither equals run_id (no parent synthesized from run_id)
    assert spans[0].context.trace_id != spans[1].context.trace_id
    assert spans[0].context.trace_id != int(run_id, 16)
    # run_id remains a span attribute so a backend can aggregate on it
    assert spans[0].attributes["run_id"] == run_id
    # latency_ms=100 produces a span about 100 ms wide.
    dur_ms = (spans[0].end_time - spans[0].start_time) / 1e6
    assert 90 <= dur_ms <= 110


def test_otel_exporter_attaches_to_upstream_traceparent():
    """With a carrier_provider (returning a carrier that contains traceparent): each span attaches to that upstream trace (trace_id == the carrier's trace-id segment)."""
    pytest.importorskip("opentelemetry")
    mem = _otel_mem()
    trace_hex = "11112222333344445555666677778888"          # upstream trace_id (32 hex)
    carrier = {"traceparent": f"00-{trace_hex}-1111222233334444-01"}
    ox = OTelExporter(carrier_provider=lambda: carrier)      # mirrors current_trace_carrier
    ox.export({"type": "llm_call", "run_id": "r1", "latency_ms": 10})
    ox.export({"type": "tool_call", "run_id": "r1", "latency_ms": 10})
    spans = mem.get_finished_spans()
    assert len(spans) == 2
    # both spans attach to the upstream trace: same trace_id, equal to the carrier's trace-id segment (children of the app request span)
    assert spans[0].context.trace_id == int(trace_hex, 16) == spans[1].context.trace_id


def test_otel_exporter_no_latency_non_negative_duration():
    """Events with no latency_ms / latency_ms=0 -> instantaneous but non-negative duration (you can't set only end_time, or start>end yields a negative duration that pollutes Jaeger)."""
    pytest.importorskip("opentelemetry")
    mem = _otel_mem()
    ox = OTelExporter()
    ox.export({"type": "summarize_failed", "run_id": "abcd"})    # no latency_ms field
    ox.export({"type": "context_block", "latency_ms": 0, "query": "q"})   # latency=0 (falsy)
    spans = mem.get_finished_spans()
    assert len(spans) == 2
    for sp in spans:
        assert sp.end_time - sp.start_time >= 0              # never negative
