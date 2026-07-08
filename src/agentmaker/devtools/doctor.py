"""agentmaker.devtools.doctor: DoctorHook, print a diagnosis in the terminal the moment a run goes wrong.

The zero-friction way to use Trace Detective while developing: attach one hook and every troubled run
diagnoses itself on the spot, no exporting files or opening the web UI. Like the rest of devtools this is
strictly opt-in; the framework core never references it.

    tracer = Tracer()
    agent = Agent("bot", llm, tools=[...], tracer=tracer, hooks=[DoctorHook(tracer)])
    agent.run("...")   # a failed tool / truncation / exception now prints a three-part diagnosis

Triggers: a run that raises (on_error), or a run whose trace carries error-severity findings
(severity="warn" widens that to warnings, e.g. empty retrievals). The diagnosis LLM is built lazily from
environment keys (pick the paying vendor via provider= / model=, or pass a ready client via llm=); if the
build fails the hook prints one notice and stays dormant, never breaking the run.
"""

import asyncio
import contextvars
import functools
from typing import Optional

from ..core.llm_clients import LLMClient
from ..runtime.execution.run_context import current_run_id
from ..runtime.hooks import Hook
from .diagnose import TraceDiagnosis, diagnose
from .trace_parser import TraceParseError, TraceRun, parse_trace

_MARK = "🩺"          # Console prefix: every DoctorHook line is recognizable and grep-able.
_SEVERITIES = ("error", "warn")


class DoctorHook(Hook):
    """Lifecycle hook that auto-diagnoses troubled runs and prints the verdict to the terminal.

    Reads this run's events back from the attached Tracer (via its MemoryExporter, present in the default
    Tracer config), so pass the SAME tracer to the Agent and to this hook. Diagnosis runs in a worker
    thread under a fresh contextvars context: it never consumes the host run's RunPolicy budget, so even a
    run that died of RunLimitExceeded can still be diagnosed. All failures inside the hook are caught and
    reported as a single console line; they never affect the run's own outcome.
    """

    def __init__(self, tracer, llm=None, *, provider: Optional[str] = None, model: Optional[str] = None,
                 severity: str = "error", language: Optional[str] = None,
                 max_chars: int = 20_000, prompts=None):
        """
        Args:
            tracer: The Tracer attached to the Agent (must keep a MemoryExporter, as the default config does).
            llm: LLM for the diagnosis (an already-built client; wins over provider/model). None builds an
                LLMClient lazily on first trigger; if that fails, one notice is printed and the hook stays dormant.
            provider: Vendor for the lazily-built client (e.g. "anthropic", "zhipu"); its API key is read
                from that vendor's environment variables, exactly like LLMClient(provider). None = the
                LLMClient default vendor. Projects holding several vendors' keys pick which one pays for
                diagnosis here, without building a client themselves.
            model: Model name for the lazily-built client; None = the chosen vendor's default model.
            severity: Minimum finding severity that triggers a diagnosis on a normally-finished run:
                "error" (default: failed tools / truncation) or "warn" (also empty retrievals /
                degradations). Runs that raise always trigger.
            language: Language of the printed verdict (a LANGUAGES key or a free-form language name).
                None (default) follows the active prompt catalog / language pack (see diagnose).
            max_chars: Character budget for the rendered timeline fed to the LLM.
            prompts: Optional PromptRegistry for the diagnosis prompt (see diagnose); defaults to the
                global DEFAULT_PROMPTS, so a process-wide language-pack override applies automatically.
        """
        if severity not in _SEVERITIES:
            raise ValueError(f"severity must be one of {_SEVERITIES}, got {severity!r}")
        self._tracer = tracer
        self._llm = llm
        self._provider = provider
        self._model = model
        self._llm_unavailable = False      # Lazy-build failed once: stay dormant instead of retrying every run.
        self._warned_no_events = False     # "tracer has no in-memory events" notice printed once.
        self._severity = severity
        self._language = language
        self._max_chars = max_chars
        self._prompts = prompts

    async def on_run_end(self, output, *, scope=None):
        """Normal completion: diagnose only when the trace carries findings at/above the severity threshold."""
        await self._maybe_diagnose(error=None)

    async def on_error(self, error: Exception):
        """Run raised: always diagnose (the exception itself is handed to the LLM as extra context)."""
        await self._maybe_diagnose(error=error)

    async def _maybe_diagnose(self, error) -> None:
        """Shared trigger path; swallows every internal failure (a broken diagnosis must not break the run)."""
        try:
            run = self._current_run()
            if run is None:
                return
            if error is None and not self._triggered(run):
                return
            llm = self._ensure_llm()
            if llm is None:
                return
            self._print_header(run, llm, error)
            extra = f"{type(error).__name__}: {error}" if error is not None else None
            # Fresh contextvars context in a worker thread: the diagnosis Agent gets its own run context
            # (own counters, no RunPolicy), instead of inheriting and draining the host run's budget.
            call = functools.partial(diagnose, run, llm, language=self._language,
                                     max_chars=self._max_chars, extra_context=extra, prompts=self._prompts)
            verdict = await asyncio.to_thread(contextvars.Context().run, call)
            self._print_verdict(verdict)
        except Exception as e:
            print(f"{_MARK} Trace Detective: diagnosis failed ({type(e).__name__}: {e})")

    def _current_run(self):
        """Pick this run's timeline out of the tracer's in-memory events (falls back to the most recent run).

        Filters to the current run's events BEFORE parsing: a long dev session accumulates many runs in the
        MemoryExporter, and re-parsing them all on every trigger would grow linearly (and let a malformed
        event from some old run break this run's diagnosis).
        """
        events = self._tracer.events
        if not events:
            if not self._warned_no_events:
                self._warned_no_events = True
                print(f"{_MARK} Trace Detective: no in-memory trace events to diagnose "
                      "(keep a MemoryExporter on the Tracer, as the default config does)")
            return None
        run_id = current_run_id()
        if run_id is not None:
            events = [e for e in events if isinstance(e, dict) and e.get("run_id") == run_id]
            if not events:                 # This run emitted nothing (e.g. it died before the first LLM call): no timeline to diagnose.
                return None
        try:
            runs = parse_trace(events)
        except TraceParseError:
            return None
        return runs[-1]                    # Filtered: the only run. Unfiltered fallback: the most recent one.

    def _triggered(self, run: TraceRun) -> bool:
        """Whether a normally-finished run crosses the severity threshold."""
        if self._severity == "warn":
            return run.stats.errors + run.stats.warnings > 0
        return run.stats.errors > 0

    def _ensure_llm(self):
        """Return the diagnosis LLM, lazily building the default client; on failure notify once and stay dormant."""
        if self._llm is not None:
            return self._llm
        if self._llm_unavailable:
            return None
        try:
            self._llm = (LLMClient(self._provider, model=self._model) if self._provider is not None
                         else LLMClient(model=self._model))
        except Exception as e:
            self._llm_unavailable = True
            print(f"{_MARK} Trace Detective: no LLM available ({e}); DoctorHook disabled "
                  "(pass llm=LLMClient(...) or set the provider's API key)")
            return None
        return self._llm

    def _print_header(self, run: TraceRun, llm, error) -> None:
        """One line saying what is being diagnosed and with which model (so the developer knows a paid call starts)."""
        run_label = (run.run_id or "(no run id)")[:12]
        cause = (f"raised {type(error).__name__}: {error}" if error is not None
                 else f"finished with {run.stats.errors} error / {run.stats.warnings} warning findings")
        model = f"{getattr(llm, 'provider', '?')}/{getattr(llm, 'model', '?')}"
        print(f"{_MARK} Trace Detective: run {run_label} {cause}; diagnosing with {model} ...")

    def _print_verdict(self, verdict: TraceDiagnosis) -> None:
        """The three-part verdict, one prefixed line per part."""
        state = "run looks healthy" if verdict.healthy else "failure found"
        step = f" · first bad step #{verdict.first_bad_step}" if verdict.first_bad_step is not None else ""
        print(f"{_MARK} verdict: {state} · confidence {verdict.confidence}{step}")
        print(f"{_MARK} what went wrong: {verdict.what_went_wrong}")
        print(f"{_MARK} root cause: {verdict.root_cause}")
        print(f"{_MARK} suggested fix: {verdict.suggested_fix}")
