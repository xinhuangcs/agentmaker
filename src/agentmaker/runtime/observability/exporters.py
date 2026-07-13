"""agentmaker.runtime.observability.exporters: pluggable export backends for trace events (TraceExporter).

`Tracer` receives events, and after redaction fans them out to a set of exporters; each backend decides where
events go: memory / JSONL file / SQLite / OTel. Each backend lazily imports its optional dependency instead of
binding it at module top level (e.g. OTel's opentelemetry is used only if installed). Everything exported is
already redacted (§8 red line: secrets / PII never land in any sink). Compared with the pluggable
TracingExporter of the OpenAI Agents SDK: our events are already structured, point-in-time records, so the
interface is simpler than its "span lifecycle": each event calls `export` once, and `close` releases resources.
"""

import json
import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional

from ...core.clock import now_utc
from ...core.sqlite_util import ensure_columns, open_sqlite


class TraceExporter(ABC):
    """Trace export backend interface: export is called once per (already redacted) event; close releases resources (files / connections)."""

    @abstractmethod
    def export(self, event: dict) -> None:
        """Export one event (called by Tracer after redaction)."""

    def close(self) -> None:
        """Release resources (no-op by default; backends holding file / DB handles override this)."""


class MemoryExporter(TraceExporter):
    """Collects events into an in-memory list (Tracer's default backend; summary / str / tests read it). Lost on restart; for in-process observation / debugging only."""

    def __init__(self, max_events: Optional[int] = 2048):
        """Initialize the event list.

        Args:
            max_events: How many events to keep at most (ring buffer: drops the oldest when exceeded), default 2048
                (aligned with OTel convention). Prevents unbounded memory growth for a long-running process with
                tracing on, and prevents summary / str full scans from getting slower. Pass None explicitly to
                retain everything, or attach a JsonlExporter / SqliteExporter.
        """
        self.events: list[dict] = []
        self._max = max_events

    def export(self, event: dict) -> None:
        """Append the event to the in-memory list; drop the oldest when exceeding max_events (ring buffer, keeps only the most recent)."""
        self.events.append(event)
        if self._max is not None and len(self.events) > self._max:
            del self.events[:len(self.events) - self._max]


class JsonlExporter(TraceExporter):
    """Appends each event as one JSON line to a file (JSON Lines): convenient for streaming appends, line-by-line reads, and feeding into log pipelines."""

    def __init__(self, path: str):
        """Open the file (append mode).

        Args:
            path: Output file path; opened in append mode. Call close before the process exits (or close the whole Tracer).
        """
        self._f = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()  # Serialize write+flush across concurrent emits to avoid interleaved writes corrupting lines (consistent with SqliteExporter).

    def export(self, event: dict) -> None:
        """Write one JSON line and flush (readable immediately, no need to wait for close)."""
        with self._lock:
            self._f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self._f.flush()

    def close(self) -> None:
        """Close the file handle."""
        with self._lock:
            self._f.close()


class SqliteExporter(TraceExporter):
    """Inserts each event as one SQLite row (type + full event JSON + created_at): can share a DB with sessions / memory, and is SQL-queryable for auditing."""

    def __init__(self, db_path: str = ":memory:"):
        """Open the connection and create the table if needed.

        Args:
            db_path: SQLite file path; defaults to ":memory:" for self-test only. Provide a file path in production to persist.
        """
        self._lock = threading.Lock()  # Serialize cross-thread writes (single connection with check_same_thread=False; concurrent emits never drop events or clash on locks).
        self._db = open_sqlite(db_path)
        try:
            # Dedicated run_id column + index: run_id is a high-frequency query key (aggregate events for one run); pulling it out of JSON is too slow.
            self._db.execute("CREATE TABLE IF NOT EXISTS traces(type TEXT, run_id TEXT, event TEXT, created_at TEXT)")
            ensure_columns(self._db, "traces", {"run_id": "TEXT"})   # Old DBs (without a run_id column) get the column auto-added: trace is derived data, so adding a column is safe.
            self._db.execute("CREATE INDEX IF NOT EXISTS idx_traces_run_id ON traces(run_id)")
            self._db.commit()
        except Exception:
            self._db.close()  # On table-creation failure: close the already-opened connection before re-raising to avoid leaking connection / file handles.
            raise

    def export(self, event: dict) -> None:
        """Insert one row: type / run_id as dedicated columns for query-by-type / query-by-run, event stores the full event JSON."""
        with self._lock:
            self._db.execute("INSERT INTO traces(type, run_id, event, created_at) VALUES (?, ?, ?, ?)",
                             (event.get("type", "?"), event.get("run_id"),
                              json.dumps(event, ensure_ascii=False, default=str), now_utc().isoformat()))
            self._db.commit()

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._db.close()


class OTelExporter(TraceExporter):
    """Maps events to OpenTelemetry spans, connecting to standard backends like Jaeger / Grafana / Datadog.

    Lazily imports opentelemetry (raises a clear error at construction if not installed, without dragging down
    other exporters). One span per event:
    - Real duration: the event already carries `latency_ms`, so the span start is rewound to `now - latency` and
      the end set to now, giving Jaeger's waterfall a real width (rather than a zero-width isolated point).
      Events without latency get an instantaneous span.
    - Parent span attribution: with the default `carrier_provider=None`, no context is passed, so OTel uses the
      current active context (if the app wraps `await agent.arun(...)` in
      `with tracer.start_as_current_span(...)`, the AB span naturally attaches; otherwise each becomes a root).
      Pass a `carrier_provider` (e.g. `current_trace_carrier`) to explicitly extract and attach the parent
      context from an upstream W3C carrier, for the case where the app holds a traceparent but has not set it as
      the current OTel context (e.g. only a header was passed across processes). The carrier is obtained
      out-of-band via a callback (not part of the event dict), which both avoids Tracer redaction accidentally
      masking traceparent's 32-hex segment and avoids changing the event schema.
    - run_id is always attached as a span attribute, so backends can filter by it to aggregate per run.
    Event fields become span attributes (non-primitive types are JSON-encoded, None is skipped: OTel attributes
    only accept primitive types).
    """

    def __init__(self, tracer_name: str = "agentmaker", *,
                 carrier_provider: Optional[Callable[[], Optional[dict]]] = None):
        """Obtain an OTel tracer.

        Args:
            tracer_name: OTel tracer name (instrumentation scope). Span export / sampling is decided by the
                app-configured OTel SDK TracerProvider (this class only produces spans, not bound to a specific backend).
            carrier_provider: Optional zero-arg callback `() -> Optional[dict]` returning the current run's
                upstream W3C trace carrier (e.g. agentmaker.current_trace_carrier). If given, each span uses the
                parent context parsed from that carrier as its parent; if it returns None / is not given, no
                explicit parent is set (falls back to OTel's current context, behaviorally identical to before).
        """
        try:
            from opentelemetry import trace
        except ImportError as e:
            raise ImportError(
                "OTelExporter requires opentelemetry (install opentelemetry-api / opentelemetry-sdk)") from e
        self._otel = trace.get_tracer(tracer_name)
        self._carrier_provider = carrier_provider

    def _parent_context(self):
        """Take the upstream W3C carrier from carrier_provider and parse it into an OTel parent context; None if there is no callback / no carrier (OTel then uses the current context)."""
        if self._carrier_provider is None:
            return None
        carrier = self._carrier_provider()
        if not carrier:
            return None
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        return TraceContextTextMapPropagator().extract(carrier)

    def export(self, event: dict) -> None:
        """Produce a span with real duration (parent context per the class docstring), event fields as attributes; run_id as an attribute so backends can aggregate per run."""
        latency_ms = event.get("latency_ms")
        if latency_ms:                       # Has duration: rewind the start to get real width (end=now, start=now-latency).
            end_ns = time.time_ns()
            start_ns = end_ns - int(latency_ms * 1_000_000)
        else:                                # No latency / latency=0: leave both ends to OTel's own clock (instantaneous, non-negative).
            start_ns = end_ns = None         # Must NOT set only end_ns: start would be taken by OTel at start_span time (later than the earlier-sampled end), yielding a negative duration.
        span = self._otel.start_span(event.get("type", "trace"), context=self._parent_context(), start_time=start_ns)
        try:
            for k, v in event.items():
                if k == "type" or v is None:
                    continue
                span.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False, default=str))
        finally:
            span.end(end_time=end_ns)   # End even if set_attribute raised, to avoid leaking an unclosed span.


