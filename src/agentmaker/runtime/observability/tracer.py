"""agentmaker.runtime.observability.tracer: minimal tracer (Tracer).

Collects structured events emitted by the Harness during an Agent run (llm_call / tool_call ...) and threads them
into a readable trace, for debugging and cost auditing. Secrets and sensitive paths are automatically redacted
before writing (CLAUDE.md §8 red line: never write secrets into logs / traces).

The Harness's tracer defaults to None (zero overhead); to observe, construct a Tracer and inject it:
Harness(llm, tracer=Tracer()). Where events go is decided by pluggable TraceExporter backends (defaults to
collecting into the in-memory `MemoryExporter`; can be swapped for JSONL / SQLite / OTel, or attach several at
once). Redaction happens once before the fan-out, so all sinks receive already-redacted events. See exporters.py.
"""

import logging
import re
from typing import Any, Iterable, Optional

from .exporters import MemoryExporter, TraceExporter

_logger = logging.getLogger(__name__)   # Exporter-failure warnings go through this (the library configures no handler, leaving it to the host; see the NullHandler in agentmaker/__init__).

# A key name that matches is treated as a secret value: match exact first (to avoid "total_tokens" being caught by "token"),
# then substring-match the few with negligible false positives (e.g. "openai_api_key" contains "api_key").
_SECRET_EXACT = frozenset({"key", "token", "secret", "password", "passwd", "auth", "authorization",
                           "bearer", "credential", "credentials", "api_key", "apikey",
                           "access_token", "refresh_token"})
_SECRET_SUBSTR = ("api_key", "apikey", "secret", "password", "passwd", "access_token", "refresh_token")
# Secret-looking values: OpenAI sk-, Bearer xxx, or a long run without whitespace (hex/base64 token).
_SECRET_VALUE = re.compile(r"sk-[A-Za-z0-9_-]{12,}|Bearer\s+\S+|\b[A-Za-z0-9_-]{32,}\b")
# Username (PII) in a home-directory path: /Users/<name>/ or /home/<name>/ -> mask the username, keep the path structure.
_HOME_PATH = re.compile(r"(/Users/|/home/)([^/]+)")
_MASK = "***"
# Framework-generated correlation fields: non-secret, excluded from value redaction; otherwise run_id (32-hex) would be masked to *** by the "secret-looking long run" rule, breaking correlation.
_SAFE_KEYS = frozenset({"run_id", "step_index"})


class Tracer:
    """Minimal tracer that collects structured events; on emit it redacts + truncates long values, then fans out to each exporter.

    The Harness sends events via emit(event); `tracer.events` reads all of them from memory (already redacted, kept
    by the default MemoryExporter), `str(tracer)` shows the timeline, `tracer.summary()` shows event count / usage /
    latency totals, and `tracer.close()` closes file / DB backends.
    """

    def __init__(self, *, redact: bool = True, max_value_len: int = 200,
                 exporters: Optional[list[TraceExporter]] = None, strict: bool = False,
                 extra_secret_keys: Optional[Iterable[str]] = None,
                 extra_secret_patterns: Optional[Iterable[Any]] = None):
        """Construct a tracer.

        Args:
            redact: Whether to redact (default True; §8 red line, normally leave it on: turn it off only in tests
                that are known to carry no sensitive data). Only controls "secret / PII masking", and does not
                affect truncation: with redaction off, long values are still truncated to max_value_len.
            max_value_len: Maximum characters kept for a single string value, truncated beyond that (prevents the
                trace from being blown up by long text / tool results); always in effect.
            exporters: Where events go (a list of TraceExporter); defaults to `[MemoryExporter()]` (collect in
                memory, same behavior as before). To persist / connect to OTel, pass the corresponding exporter; to
                also keep memory, remember to include `MemoryExporter()`.
            strict: Whether to re-raise when a single exporter throws (default False = fault-tolerant, swallow). A
                trace is side-channel observation; one sink failing (disk full / DB lock / OTel collector
                unreachable) should not take down the main flow; pass True when debugging / testing wants fail-loud.
            extra_secret_keys: Additional secret key names contributed by the app (substring match, case-insensitive):
                if a key name contains any of these, its value is masked to `***`. The built-in set
                (key/token/secret/password/api_key... + sk-/Bearer/long-run values + home-directory PII) is always
                present; this only unions on top. Example: `extra_secret_keys=["ssn", "session_id"]` -> values whose
                key name contains ssn / session_id are masked. The framework knows no business concepts, so
                business-specific sensitive fields are declared here (upholding §8 "business concepts do not enter agentmaker").
            extra_secret_patterns: Additional secret-value regexes contributed by the app (str or a compiled
                re.Pattern): matched fragments are masked to `***`. Example: `extra_secret_patterns=[r"cus_[A-Za-z0-9]+"]`
                -> a custom token prefix is masked. Applied together with the built-in value rules.
        """
        self._redact = redact
        self._max = max_value_len
        self._strict = strict
        self.exporters: list[TraceExporter] = exporters if exporters is not None else [MemoryExporter()]
        # App-added secret key-name substrings (unioned into the built-in _SECRET_SUBSTR) and value regexes (unioned into the built-in _SECRET_VALUE); the built-in defaults are always present, only added to, never replaced.
        extra_keys = tuple(k.lower() for k in (extra_secret_keys or ()))
        if any(not k.strip() for k in extra_keys):   # A blank key matched as a substring would match every key name -> the whole trace silently masked to ***.
            raise ValueError("extra_secret_keys contains an empty / blank entry: an empty string would make every key name be judged secret, please remove it")
        self._secret_substr = _SECRET_SUBSTR + extra_keys
        self._value_patterns = [_SECRET_VALUE, *(
            re.compile(p) if isinstance(p, str) else p for p in (extra_secret_patterns or ()))]
        self._export_errors: dict[str, int] = {}   # Cumulative export-failure count per exporter (by class name); exposed via summary().dropped.
        self._clean_errors = 0                      # Count of events dropped because cleaning itself raised; exposed via summary().dropped_uncleanable.

    def emit(self, event: dict) -> None:
        """Receive one event: after redaction (optional) + long-value truncation (always), fan out to each exporter. Called by the Harness.

        A single exporter throwing is swallowed by default (fault tolerance: side-channel observation does not take
        down the main flow); with strict=True it re-raises (fail-loud). Cleaning itself throwing (e.g. an odd
        object's str() raising) also does not bubble up and kill the run: the event is dropped and counted (only
        re-raised when strict).
        """
        try:
            cleaned = self._clean(event)
        except Exception as e:  # noqa: BLE001
            if self._strict:
                raise
            self._clean_errors += 1
            if self._clean_errors == 1:
                _logger.warning("trace event cleaning failed, event dropped (only the first failure is logged; see dropped_uncleanable in tracer.summary): %r",
                                e, exc_info=True)
            return
        for exporter in self.exporters:
            try:
                exporter.export(cleaned)
            except Exception as e:  # noqa: BLE001
                if self._strict:
                    raise
                name = type(exporter).__name__
                if name not in self._export_errors:
                    _logger.warning("trace exporter %s export failed; only the first failure is logged (see dropped in tracer.summary): %r",
                                    name, e, exc_info=True)
                self._export_errors[name] = self._export_errors.get(name, 0) + 1

    @property
    def events(self) -> list[dict]:
        """Convenience read of events collected in memory: returns the list of the first MemoryExporter (the default config has one); empty list if none."""
        for exporter in self.exporters:
            if isinstance(exporter, MemoryExporter):
                return exporter.events
        return []

    def _is_secret_key(self, k: str) -> bool:
        """Whether the key name should be masked as a secret in full (built-in exact table + built-in / app-added substrings)."""
        kl = k.lower()
        return kl in _SECRET_EXACT or any(s in kl for s in self._secret_substr)

    def _clean(self, obj: Any) -> Any:
        """Recursively process: dict masks secret values by key, str redacts + truncates, list/tuple/set each element (restoring the original type), any other unknown object is str()'d then redacted + truncated,
        guaranteeing the invariant that any value reaching an exporter / persisted has passed redaction (including tuple / custom objects, plugging the gap where §8's red line could be bypassed)."""
        if isinstance(obj, dict):
            out: dict = {}
            for k, v in obj.items():
                if k in _SAFE_KEYS:
                    out[k] = v                       # Correlation field: key and value kept as-is (otherwise run_id would be wrongly masked by the "secret-looking long run" rule, breaking correlation).
                    continue
                # Key names are redacted too (not truncated: a key is a field name, and truncating it would break structure / lookup): secrets / home-directory paths appearing in a key-name position are not missed either.
                ck = self._clean_key(k) if isinstance(k, str) else k
                if self._redact and isinstance(k, str) and self._is_secret_key(k):
                    out[ck] = _MASK                  # Key name indicates a secret (api_key etc.) -> mask its value.
                else:
                    out[ck] = self._clean(v)
            return out
        if isinstance(obj, (list, tuple, set)):
            cleaned = [self._clean(x) for x in obj]
            return cleaned if isinstance(obj, list) else type(obj)(cleaned)   # Restore original type for tuple/set.
        if isinstance(obj, str):
            return self._clean_str(obj)
        if isinstance(obj, (bool, int, float)) or obj is None:
            return obj                                   # Number / bool / None: not a string, no redaction / truncation needed.
        return self._clean_str(str(obj))                 # Unknown object (dataclass / custom): str() then redact + truncate, otherwise it would bypass redaction when persisted via json.dumps(default=str).

    def _clean_key(self, k: str) -> str:
        """Redact secret-looking fragments and home-directory usernames in a key name (only when redact); never truncate: a key is a field name, and truncating would break structure / lookup."""
        if not self._redact:
            return k
        for pat in self._value_patterns:
            k = pat.sub(_MASK, k)
        return _HOME_PATH.sub(rf"\1{_MASK}", k)

    def _clean_str(self, s: str) -> str:
        """Mask secret-looking fragments in a string (built-in + app-added value regexes) and home-directory usernames (only when redact), then always truncate to max_value_len.

        Redaction and truncation are decoupled: turning off redaction only skips masking, long values are still
        truncated, avoiding "turning off redaction" incidentally blowing up the trace.
        """
        if self._redact:
            for pat in self._value_patterns:
                s = pat.sub(_MASK, s)
            s = _HOME_PATH.sub(rf"\1{_MASK}", s)
        if len(s) > self._max:
            s = s[: self._max] + f"…(+{len(s) - self._max})"
        return s

    def summary(self) -> dict:
        """Statistics: total event count, count by type, cumulative token usage, cumulative latency (ms)."""
        by_type: dict[str, int] = {}
        tokens = latency = 0
        for e in self.events:
            t = e.get("type", "?")
            by_type[t] = by_type.get(t, 0) + 1
            usage = e.get("usage")
            if isinstance(usage, dict):
                tokens += usage.get("total_tokens") or 0
            latency += e.get("latency_ms") or 0
        return {"events": len(self.events), "by_type": by_type,
                "total_tokens": tokens, "total_latency_ms": latency,
                "dropped": dict(self._export_errors),     # Events silently dropped per exporter (export failures); empty = no drops.
                "dropped_uncleanable": self._clean_errors}   # Events dropped because cleaning itself raised; 0 = none.

    def clear(self) -> None:
        """Clear the events collected in memory (only affects MemoryExporter; file / DB sinks are untouched)."""
        for exporter in self.exporters:
            if isinstance(exporter, MemoryExporter):
                exporter.events.clear()

    def close(self) -> None:
        """Close all exporters (release file / DB handles); call before the process exits."""
        for exporter in self.exporters:
            exporter.close()

    def __str__(self) -> str:
        """Readable timeline: one event per line (type + remaining fields)."""
        if not self.events:
            return "(trace is empty)"
        lines = []
        for i, e in enumerate(self.events, 1):
            rest = {k: v for k, v in e.items() if k != "type"}
            lines.append(f"{i:>2}. {e.get('type', '?'):<10} {rest}")
        return "\n".join(lines)
