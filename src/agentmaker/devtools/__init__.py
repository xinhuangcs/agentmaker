"""agentmaker.devtools: OPTIONAL developer tools offered alongside the framework. Current tool: Trace Detective.

Strictly an opt-in add-on, not part of the framework proper: the framework core never imports this
package, and the native observability story is unchanged (Harness emits events -> Tracer redacts ->
exporters sink them; all of that works exactly the same whether or not this package is ever touched).
Trace Detective merely CONSUMES that output downstream, for developers who choose to use it.

It debugs agent runs recorded by the framework's Tracer: the deterministic half (trace_parser) turns a
JSONL trace / event list into per-run timelines with rule-based findings; the LLM half (diagnose) returns
a validated three-part verdict (earliest failure / root cause / fix); webapp wraps both into a local web
UI (`python -m agentmaker.devtools`, requires the `agentmaker[devtools]` extra).

Like agentmaker.testing, this package is not re-exported from the top-level `agentmaker` namespace;
import it on demand:

    from agentmaker.devtools import diagnose_trace, load_trace

    run, verdict = diagnose_trace(open("run.jsonl").read(), LLMClient("deepseek"))
"""

from .diagnose import LANGUAGES, TraceDiagnosis, diagnose, diagnose_trace
from .doctor import DoctorHook
from .trace_parser import (Finding, RunStats, TraceParseError, TraceRun, TraceStep, load_trace,
                           parse_trace, pick_run, render_run)
from .webapp import create_app

__all__ = [
    "Finding", "RunStats", "TraceParseError", "TraceRun", "TraceStep",
    "load_trace", "parse_trace", "pick_run", "render_run",
    "LANGUAGES", "TraceDiagnosis", "diagnose", "diagnose_trace",
    "DoctorHook",
    "create_app",
]
