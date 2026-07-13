"""agentmaker.agents.base: BaseAgent, the shared base for all agents.

Defines the common "run boundary" and mechanisms shared by every Agent, without dictating how
answers are actually generated:
    - run / resume / template: input guardrails -> generation (subclass `_arun`) -> output
      guardrails -> atomic history persistence -> checkpoint clearing;
    - HITL suspend / resume, per-step checkpointing, per-scope multi-session history;
    - the three sub-agent delegation methods (_derive_scope / _child_decision / _absorb_child):
      orchestration strategies get "suspend propagates upward + parent commits before child cleanup"
      ordering contracts for free by hanging sub-agents off them.

Async is the sole real implementation: `arun` / `aresume` are the template bodies; `run` / `resume`
/ `stream_run` (provided by concrete classes) are one-line synchronous facades over `core.aio`.
A subclass implements at least one of `_arun` (async-native) or `_run` (synchronous, defaulting to
a thread pool); `__init_subclass__` enforces this at definition time.
"""

import asyncio
import copy
import threading
import weakref
from contextlib import asynccontextmanager
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any, Awaitable, Optional, cast

from ..core.exceptions import GuardrailTripwireError, RunLimitExceeded, SessionError
from ..core.llm_clients import LLMClient
from ..core.message import Message
from ..core.trace_events import EVENT_RUN_ERROR
from ..prompts import DEFAULT_PROMPTS
from ..core.aio import run_sync
from ..runtime.harness import Harness, HarnessConfig
from ..runtime.hooks import afire
from ..runtime.execution.run_context import (check_deadline, correlation, current_run_id, current_step,
                                             new_run_id, reset_run, snapshot_usage, start_run)

if TYPE_CHECKING:                       # Type annotations only; not imported at runtime (cross-subsystem, avoids potential cycles)
    from .result import RunResult
    from ..prompts import PromptRegistry
    from ..runtime.execution.checkpoint import CheckpointStore
    from ..runtime.execution.run_policy import RunPolicy
    from ..runtime.guardrails import Guardrail
    from ..runtime.hooks import Hook
    from ..retrieval.scope import Scope
    from ..runtime.sessions import SessionStore

# Lets __init_subclass__ reference this class (BaseAgent does not yet exist at definition time; a
# module-level sentinel provides late binding).
_BASE = None


class _ScopeGate:
    """Serialize one scope within an event loop and reject concurrent use from another loop."""

    def __init__(self) -> None:
        """Initialize an unowned gate with a fresh loop-local lock."""
        self._meta_lock = threading.Lock()
        self._loop = None
        self._participants = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Join the owning loop and wait in FIFO order, failing fast on cross-loop contention."""
        loop = asyncio.get_running_loop()
        with self._meta_lock:
            if self._loop is not None and self._loop is not loop:
                raise SessionError(
                    "This scope is already running on another event loop (usually two synchronous threads "
                    "called run/resume concurrently on the same Agent). Cross-loop waiting on asyncio.Lock "
                    "can deadlock, so the second call was rejected. Serialize same-scope synchronous calls "
                    "in the application, or use the async API from one event loop.")
            self._loop = loop
            self._participants += 1
            lock = self._lock
        try:
            await lock.acquire()
        except BaseException:
            self._leave(lock)
            raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Release the loop-local lock and reset loop affinity after the final participant leaves."""
        lock = self._lock
        lock.release()
        self._leave(lock)

    def _leave(self, lock: asyncio.Lock) -> None:
        """Remove one holder/waiter and rebuild the asyncio lock once the gate becomes idle."""
        with self._meta_lock:
            self._participants -= 1
            if self._participants == 0:
                self._loop = None
                if self._lock is lock:
                    self._lock = asyncio.Lock()


class BaseAgent:
    """Shared base for all agents. Subclasses must implement at least one of `_arun` (async-native) or `_run` (synchronous)."""

    def __init__(self, name: str, llm: LLMClient, system_prompt: Optional[str] = None, *,
                 session_store: "Optional[SessionStore]" = None, scope: "Optional[Scope]" = None,
                 checkpoint_store: "Optional[CheckpointStore]" = None,
                 input_guardrails: "Optional[list[Guardrail]]" = None,
                 output_guardrails: "Optional[list[Guardrail]]" = None,
                 hooks: "Optional[list[Hook]]" = None,
                 run_policy: "Optional[RunPolicy]" = None,
                 harness_config: "Optional[HarnessConfig]" = None,
                 prompts: "Optional[PromptRegistry]" = None, on_pending: str = "error", as_child: bool = False):
        """
        Store the agent's core dependencies and initialize conversation history (loading and resuming from session_store if one is attached).

        Args:
            name: Agent name, used for identification / debugging.
            llm: LLM client the agent uses to call the model.
            system_prompt: System prompt defining the agent's persona and behavior; optional.
            session_store: Optional session persistence store. When attached, history is persisted
                per scope and loaded per scope at each run to resume (the store is the single source
                of truth and supports cross-process access), so a long-running daemon does not lose
                conversations on restart. When absent, history is per-scope and in-process (lost on restart).
            scope: The default scope the session belongs to (identifies session / user via
                retrieval.Scope); run / resume can override it per call, so one Agent instance can
                serve multiple sessions (history is isolated per scope). Defaults to an empty scope.
            checkpoint_store: Optional execution-state checkpoint store (CheckpointStore,
                agentmaker.runtime.execution). Attaching one enables execution-state persistence,
                shared across three use cases: (1) HITL async approval (high-risk tools suspend, run
                returns an interrupted RunResult, resume(decision) continues); (2) crash recovery / (3) long-task
                resume (a checkpoint is saved each step, resume() continues after a process restart).
                Strategies that support resume implement `_adrive`.
            input_guardrails: Optional list of input guardrails. Checked against user input before
                run; a tripped guardrail raises GuardrailTripwireError.
            output_guardrails: Optional list of output guardrails. Checked against the final output
                before persisting history; a trip raises (the blocked turn is not recorded). The rules
                are app business logic.
            hooks: Optional list of lifecycle hooks (Hook, agentmaker.runtime.hooks); observe-only.
                Run-level events (run start / end, guardrail, suspend, error, etc.) are fired by this
                base; model/tool-level events (calling the model / executing tools) are fired by each
                strategy's Harness. Zero overhead when none are attached.
            run_policy: Optional run governance (RunPolicy, agentmaker.runtime.execution). When
                attached, a single run is bounded by wall-time / LLM-call count / tool-call count /
                token limits plus cooperative cancellation (gated at each call site in the Harness);
                exceeding a limit / cancellation raises RunLimitExceeded / RunCancelled to abort the
                turn. Counted independently per run (resume is a new turn). Unbounded when absent.
                Warning: nested runs (delegated via AgentTool, or a Plan / Reflection sub-executor)
                inherit the outermost policy's global counters; a run_policy attached to a sub-agent
                does not take effect within the parent run (a warning is emitted). To bound sub-tasks,
                set the global limits on the parent Agent's run_policy.
            prompts: Optional prompt registry (PromptRegistry). If not passed, a copy of
                DEFAULT_PROMPTS is made at construction (isolated per instance, so update_prompts only
                affects this agent and its harness / sub-agents, without mutating the global registry
                / other agents).
            on_pending: Policy for running again when the same scope already has a pending-approval
                checkpoint (requires checkpoint_store): "error" (default) raises SessionError to
                prevent a new run silently overwriting the pending action (making the approval request
                vanish); "discard" drops the old pending action and continues the new run.
            as_child: Set True when acting as an internal sub-agent of an orchestration strategy (Plan
                / Reflection, etc.). It declares two semantics at once:
                (1) run-level hooks do not fire at the child layer (the parent strategy fires them
                once, avoiding repetition on every sub-step); the passed hooks are still forwarded via
                `_harness_hooks` to the inner Harness to observe model/tool-level events;
                (2) checkpoints are not self-cleared (`_defer_checkpoint_clear`); the parent strategy
                clears them explicitly via `clear_checkpoint` after committing its own progress,
                guaranteeing "parent commits before child cleanup" and eliminating the unrecoverable
                state where the parent is awaiting but the child's checkpoint has been cleared.
        """
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.session_store = session_store
        self.scope = scope
        self.checkpoint_store = checkpoint_store
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        all_hooks = hooks or []
        self.hooks = [] if as_child else all_hooks   # Run-level hooks (fired by the parent strategy when as_child, so the child layer does not repeat)
        self._harness_hooks = all_hooks              # Model/tool-level hooks (used by concrete classes when assembling the Harness, unaffected by as_child)
        self.run_policy = run_policy
        self.prompts = prompts if prompts is not None else DEFAULT_PROMPTS.copy()
        self._harness_cfg = harness_config or HarnessConfig()
        self._sessions: dict = {}
        self._scope_locks: "weakref.WeakValueDictionary" = weakref.WeakValueDictionary()
        self._scope_locks_guard = threading.Lock()
        if on_pending not in ("error", "discard"):
            raise ValueError(f"on_pending must be 'error' or 'discard', got {on_pending!r}")
        self._on_pending = on_pending
        self._defer_checkpoint_clear = as_child      # See as_child semantics (2)
        self._as_child = as_child

    def _make_harness(self, *, tool_registry=None, **overrides) -> "Harness":
        """Assemble a Harness from self._harness_cfg, automatically injecting hooks=self._harness_hooks and prompts=self.prompts.

        Strategy authors get a correctly assembled Harness in one line, without hand-newing one and
        without risking forgetting to pass _harness_hooks (which would silently drop model/tool-level
        events under as_child) or prompts (which would silently fork the harness prompts from the
        agent's). tool_registry and other per-strategy differences are expressed via explicit
        parameters / overrides (e.g. Reflection's own harness calls _make_harness() without
        tool_registry, so it carries no tools).

        Args:
            tool_registry: The tool registry for this harness (None means a pure-reasoning harness).
            **overrides: Override certain HarnessConfig fields (as needed).
        """
        cfg = {f.name: getattr(self._harness_cfg, f.name) for f in fields(self._harness_cfg)}   # Fetched by reference (not asdict, to avoid deep-copying live objects like tracer/compactor)
        cfg.update(overrides)
        return Harness(self.llm, tool_registry=tool_registry, hooks=self._harness_hooks, prompts=self.prompts, **cfg)

    def __init_subclass__(cls, **kwargs):
        """Definition-time check: a subclass (including multi-level inheritance) must override at least one of `_run` or `_arun` (resolved via MRO, so intermediate bases are unconstrained)."""
        super().__init_subclass__(**kwargs)
        if _BASE is not None and cls._run is _BASE._run and cls._arun is _BASE._arun:
            raise TypeError(f"{cls.__name__} must implement at least one of _run (synchronous) or _arun (asynchronous)")

    # -- run / resume template (async is the real body, sync is an aio facade) --

    async def arun(self, input_text: str, *, scope: "Optional[Scope]" = None,
                   trace_carrier: Optional[dict[str, str]] = None, **kwargs: Any) -> "RunResult":
        """Process one input and return a reply (template method: the run boundary shared by all strategies):
        input guardrails -> strategy logic `_arun` -> output guardrails -> persist history.

        Subclasses only implement `_arun` (pure generation); this method funnels the "run boundary"
        cross-cutting concerns (guardrails and history persistence) so every strategy gets them for
        free, mirroring the OpenAI Agents SDK where guardrails are declared on the agent and enforced
        by a single execution layer (rather than reimplemented in each strategy). Returns a RunResult
        envelope: final_output carries `_arun`'s output (usually str; a validated model instance when
        output_schema is passed); on HITL suspend the result is interrupted (.interrupt carries the
        pending actions), history is not persisted, and the state has already been saved into the
        CheckpointStore by the strategy, awaiting resume.

        Args:
            input_text: The user's raw input text.
            scope: The session identifier for this call; defaults to self.scope. History is loaded /
                saved by it (one instance can serve multiple sessions).
            trace_carrier: Optional upstream W3C trace carrier (e.g. {"traceparent": request-header}).
                Useful only when an OTelExporter(carrier_provider=...) is attached, letting this run's
                spans join the app's cross-service trace. When absent, spans are attributed by the OTel
                current context.
            **kwargs: Passed through to `_arun` (e.g. verbose, output_schema, temperature).
        """
        scope = scope if scope is not None else self.scope
        # Serialize the run / resume lifecycle for the same scope, preventing two concurrent requests
        # from passing the gate and overwriting each other's suspended state (approving an action nobody saw).
        async with self._scope_lock(scope), self._scaffold(input_text, scope, trace_carrier):
            await self._check_guardrails(self.input_guardrails, input_text, "input")
            await self._guard_pending(scope)        # Pending-state gate: do not silently overwrite a pending action for the same scope (see on_pending)
            output = await self._arun(input_text, scope=scope, **kwargs)
            if self._is_interrupt(output):
                await afire(self.hooks, "on_interrupt", output.pendings, scope=scope)
                return self._pack_interrupted(output)   # HITL suspend: no history persisted (no final output), state already saved, awaiting resume
            check_deadline()
            await self._mark_checkpoint_completed(scope)
            await self._record_after_completion_marker(input_text, output, scope)
            await self._clear_checkpoint(scope)  # Normal completion: clear the per-step checkpoints saved this turn (otherwise a leftover is mistaken for an unfinished run)
            await afire(self.hooks, "on_run_end", output, scope=scope)   # Hooks still receive the bare output (observe-only)
            return self._pack_completed(output, input_text)

    def run(self, input_text: str, *, scope: "Optional[Scope]" = None,
            trace_carrier: Optional[dict[str, str]] = None, **kwargs: Any) -> "RunResult":
        """Synchronous facade over arun (driven by aio.run_sync). In environments with a running event loop (async / Jupyter / FastAPI), await arun instead."""
        return run_sync(self.arun(input_text, scope=scope, trace_carrier=trace_carrier, **kwargs))

    def _scope_lock(self, scope) -> _ScopeGate:
        """Return the loop-aware gate for this scope, serializing run/resume within one event loop.

        Concurrency anti-overwrite: `_guard_pending` is check-then-act and checkpoint writes are
        last-writer-wins, so when two same-scope requests both pass the gate, A suspends while B
        overwrites the pending with its own; approving A's Interrupt then resumes into B's state,
        approving an action nobody saw. With this lock, same-scope requests queue rather than interleave.

        The gate queues same-loop callers exactly like asyncio.Lock. A concurrent caller from another
        loop (typically another synchronous worker thread) fails fast instead of awaiting an asyncio
        Future owned by the wrong loop and hanging forever. Once all participants leave, the gate drops
        loop affinity, so later non-overlapping calls from another thread remain supported.

        Gates are created lazily per scope key and stored in a WeakValueDictionary (reclaimed when no
        caller holds one). The registry access itself is protected for cross-thread sync facades.
        Nested sub-agents use distinct derived child scopes. Cross-process coordination still requires
        optimistic locking at the checkpoint_store layer.
        """
        with self._scope_locks_guard:
            gate = self._scope_locks.get(scope)
            if gate is None:
                gate = _ScopeGate()
                self._scope_locks[scope] = gate
            return gate

    async def _guard_pending(self, scope) -> None:
        """Pending-state gate (called by the arun template before entering _arun): when this scope
        already has a suspended checkpoint with a non-empty pending (last turn's HITL awaiting
        approval, not yet resumed), handle it per on_pending. Default "error" raises SessionError (to
        prevent a new run overwriting it via each step's _checkpoint and then deleting the pending
        action via _clear, making the approval request vanish); "discard" drops the old pending and
        continues. Passes through when there is no checkpoint_store, no checkpoint, or the checkpoint
        is not suspended (a crash-recovery checkpoint whose pending is None): the latter should be
        resumed via resume(None) and is not blocked here.
        """
        if self.checkpoint_store is None:
            return
        raw = await self.checkpoint_store.aload(scope=scope)
        if raw is None:
            return
        from ..runtime.execution.state import CHECKPOINT_FORMAT_VERSION, ExecutionState, checkpoint_format_version
        if checkpoint_format_version(raw) != CHECKPOINT_FORMAT_VERSION:
            await self.checkpoint_store.aclear(scope=scope)
            return
        state = ExecutionState.from_json(raw)
        if state.completed:
            await self.checkpoint_store.aclear(scope=scope)   # Completed but not cleanly cleared (crashed between record and clear): clear it, pass the new run through
            return
        if not state.pending:
            return
        if self._on_pending == "discard":
            await self.checkpoint_store.aclear(scope=scope)
            return
        raise SessionError(
            "This scope has an unfinished pending action (an unhandled HITL suspend): starting a new "
            "run directly would overwrite it and lose the approval request. "
            "First handle it via resume(decision) or drop it via clear_checkpoint(); to auto-discard, "
            "set on_pending='discard' at construction.")

    @asynccontextmanager
    async def _scaffold(self, input_text: str, scope, trace_carrier=None):
        """Shared run boundary (used by arun and streaming strategies): opens the trace-correlation
        context (nested runs inherit the outer one rather than creating a new; scope enters the
        context so tools isolate per parent run; trace_carrier lets OTelExporter join spans into the
        app's cross-service trace) + on_run_start + exception routing (GuardrailTripwireError
        re-raises directly, everything else goes through on_error) + finally reset. The teardown
        responsibilities (on_run_end / persist history / clear checkpoint) are not here: the caller
        performs them in order after confirming it is not a suspend.
        """
        token = start_run(new_run_id(), policy=self.run_policy, scope=scope, trace_carrier=trace_carrier)
        try:
            await afire(self.hooks, "on_run_start", input_text, scope=scope)
            try:
                yield
            except GuardrailTripwireError:
                raise                          # A guardrail trip is already reported via on_guardrail_trip, so it does not also go through on_error
            except Exception as e:             # Covers the whole run lifecycle: _arun / _record (history persistence failure) / on_run_end hooks, etc.
                if token is not None:          # only the run-context owner emits the terminal run_error
                    self._trace_run_error(e)
                await afire(self.hooks, "on_error", e)
                raise
        finally:
            reset_run(token)

    async def _arun(self, input_text: str, *, scope, **kwargs):
        """The strategy's own generation logic (without guardrails / history persistence: that is the
        arun template's job). Async-native strategies override this; strategies that only write a
        synchronous `_run` use this default (dispatching to a thread pool). Returns a reply (str or a
        structured instance, an Interrupt on HITL suspend).

        scope is passed by the template: strategies that read history load by it, strategies that
        suspend save the suspended state by it.
        Note: this default only suits simple synchronous strategies that do not touch the harness (like
        test stubs); the harness is already an async single core, so calling its synchronous facade
        from a synchronous `_run` detours through a thread-local loop, so real strategies should
        implement `_arun`.
        """
        return await asyncio.to_thread(lambda: self._run(input_text, scope=scope, **kwargs))

    def _run(self, input_text: str, *, scope, **kwargs):
        """The synchronous counterpart to `_arun` (a simple synchronous strategy implements this; the default `_arun` dispatches it to a thread pool)."""
        raise NotImplementedError(f"{type(self).__name__} implements neither _run nor _arun (subclasses must implement one)")

    # -- Checkpoint resume (funneled in the base; strategies only implement _adrive: driving their own loop from ExecutionState) --

    async def aresume(self, decision: Optional[bool | dict[str, bool]] = None, *, scope: "Optional[Scope]" = None,
                      trace_carrier: Optional[dict[str, str]] = None, **kwargs: Any) -> "RunResult":
        """Resume an unfinished run from a checkpoint: reload the execution state -> (for HITL) inject the decision -> continue from the breakpoint.

        Two kinds of resume share this entry:
        - HITL: `decision` is a bool. Approve (True) executes the suspended high-risk action / reject
          (False) skips it and feeds the result back so the model reroutes.
        - Crash recovery / long-task resume: `decision` is omitted (None). No decision is injected;
          it simply continues from the last checkpoint (a suspended high-risk action re-suspends and
          is returned as an interrupted RunResult awaiting approval again, rather than being mistaken
          for a rejection).

        Funneled in the base: reload ExecutionState -> (as needed) inject the decision -> call the
        strategy's `_adrive` to continue -> teardown (on completion: output guardrails + persist
        history + clear checkpoint; on re-suspend: repack as an interrupted RunResult). Input guardrails are not
        re-checked (the first run already checked them); it does not go through `_scaffold`, so resume
        continues the run_id / step from before the suspend and does not re-fire on_run_start.

        Args:
            decision: The HITL decision. A bool (True approve / False reject) decides all pending
                actions of this turn uniformly; a dict {call_id: bool} decides per call_id (used when
                one suspend has multiple pending actions, see Interrupt.pendings); None is a
                crash-recovery resume (no decision injected).
            scope: The session identifier; defaults to self.scope.
            trace_carrier: Optional upstream W3C trace carrier (the traceparent of this resume
                request); not persisted with the checkpoint, passed by the app on each resume.
            **kwargs: Passed through to the strategy's `_adrive`.

        Returns:
            RunResult: completed (the final reply in final_output, history persisted) or interrupted
                (a high-risk action re-suspended during resume, still awaiting approval).
        """
        if decision is not None and not isinstance(decision, (bool, dict)):
            raise TypeError(
                f"resume's decision must be a bool (uniformly approve/reject all), a dict {{call_id: bool}} (decide per call), or None (crash-recovery resume), "
                f"got {type(decision).__name__}: did you accidentally pass scope as a positional arg? Correct usage: resume(True, scope=interrupt.scope).")
        if isinstance(decision, dict) and not all(isinstance(v, bool) for v in decision.values()):
            raise TypeError("The values of resume's decision dict must all be bool (True approve / False reject).")
        scope = scope if scope is not None else self.scope
        async with self._scope_lock(scope):    # Same lock as arun: same-scope resume is also serialized (reload state -> continue -> teardown/re-suspend)
            state = await self._load_execution_state(scope)
            token = start_run(state.run_id or new_run_id(), step=state.step, policy=self.run_policy, scope=scope,
                              trace_carrier=trace_carrier)
            try:
                try:
                    self._inject_decision(state, decision)
                    return await self._finish_resume(await self._adrive(state, scope=scope, **kwargs), state, scope)
                except GuardrailTripwireError:
                    raise                          # An output-guardrail trip is already reported via on_guardrail_trip, so it does not also go through on_error
                except Exception as e:             # The resume path is also inside the on_error boundary (consistent with run)
                    if token is not None:          # only the run-context owner emits the terminal run_error
                        self._trace_run_error(e)
                    await afire(self.hooks, "on_error", e)
                    raise
            finally:
                reset_run(token)

    def resume(self, decision: Optional[bool | dict[str, bool]] = None, *, scope: "Optional[Scope]" = None,
               trace_carrier: Optional[dict[str, str]] = None, **kwargs: Any) -> "RunResult":
        """Synchronous facade over aresume (driven by aio.run_sync)."""
        return run_sync(self.aresume(decision, scope=scope, trace_carrier=trace_carrier, **kwargs))

    def _trace_run_error(self, error: Exception) -> None:
        """Emit the terminal exception while the run correlation context is still active."""
        tracer = getattr(getattr(self, "harness", None), "tracer", self._harness_cfg.tracer)
        if tracer is not None:
            try:
                tracer.emit({"type": EVENT_RUN_ERROR, "error_type": type(error).__name__,
                             "message": str(error), **correlation()})
            except Exception as trace_error:  # noqa: BLE001
                error.add_note(f"Failed to emit run_error trace: {trace_error}")

    async def _adrive(self, state, *, scope, **kwargs):
        """Drive this strategy's loop from ExecutionState (the shared entry for the first run and
        resume). Async-native strategies that support resume override this; strategies that only write
        a synchronous `_drive` use this default (dispatching to a thread pool).

        scope is passed by run / resume (checkpoints are saved by it on suspend / per step). Returns
        str (complete) or Interrupt (suspended; before suspending it must persist via `self._suspend`).
        """
        return await asyncio.to_thread(lambda: self._drive(state, scope=scope, **kwargs))

    def _drive(self, state, *, scope, **kwargs):
        """The synchronous counterpart to `_adrive`. Neither overridden = this strategy does not support resume (HITL / crash recovery)."""
        raise NotImplementedError(f"{type(self).__name__} implements neither _drive nor _adrive, so it does not support resume (HITL / crash recovery)")

    # -- The three sub-agent delegation methods (shared by orchestration strategies; funnel the as_child pieces into structure) --

    @staticmethod
    def _derive_scope(scope, suffix: str):
        """Derive a sub-agent's child scope: append a `"::"+suffix` suffix to the agent dimension of the passed scope (distinguishing the sub-agent's checkpoint / history from the parent's).

        Args:
            scope: The parent scope; falls back to an empty Scope() if None.
            suffix: The sub-agent identifier (e.g. "plan_exec" / "reflect_crit").
        """
        from ..retrieval import Scope   # Deferred import: base does not depend on retrieval at the top level
        base = scope or Scope()
        return replace(base, agent=(base.agent or "") + "::" + suffix)

    @staticmethod
    def _child_decision(state):
        """Take the pending calls that already have a decision and translate them into a multi-decision dict for the sub-agent's resume (only decided items, values always bool):

        - If there are decided items -> `{call_id: bool, ...}` (only the decided calls; the sub-agent's
          aresume injects by call_id, and undecided ones naturally re-suspend);
        - If none are decided (parent-level resume(None) crash recovery / this sub-step is entirely
          undecided under a partial decision) -> None: the sub-agent lets these actions re-suspend and
          propagate upward to await approval again, consistent with a leaf Agent's resume(None)
          semantics (an approval is never silently swallowed by being mistaken for a "rejection").

        Only decided keys are taken (same convention as leaf _inject_decision): an undecided item's
        None is never put into the dict, otherwise the sub-agent's aresume would reject it via its "dict
        values must all be bool" check (crashing the partial-decision delegation path).
        """
        decided = {p.call_id: state.decisions[p.call_id]
                   for p in state.pending if p.call_id in state.decisions}
        return decided or None

    async def _absorb_child(self, result, state, scope, *, child, child_scope, on_complete):
        """Absorb the result of one sub-agent step (the sole entry for orchestration strategies, with the ordering contract fixed here):

        - Suspend (result is an Interrupt) -> `meta["awaiting"]=True` -> persist parent progress via
          `_suspend` and re-pack as a parent-scope Interrupt to return (the sub-scope one must not be
          passed through: the caller resumes the parent by Interrupt.scope, and passing it through
          would reload the wrong state);
        - Complete -> `meta["awaiting"]=False` (reset before commit: otherwise
          "suspend -> approve -> a later step crashes mid-way -> resume" would, because of a leftover
          awaiting=True in the checkpoint, resume the already-cleared child checkpoint and fall into a
          permanently unrecoverable state) -> `on_complete(result)` for parent bookkeeping (writing
          meta progress, etc.) -> parent `_checkpoint` -> `child.clear_checkpoint(child_scope)`
          ("parent commits before child cleanup") -> return None.

        Args:
            result: The RunResult returned by the sub-agent's run / resume (reads result.interrupted /
                result.interrupt / result.final_output).
            state / scope: The parent strategy's execution state and session identifier.
            child / child_scope: The sub-agent instance and its derived child scope (see _derive_scope).
            on_complete: The parent bookkeeping callback on completion, (final_output) -> None (takes
                the child's final output to write into meta progress, not the whole envelope).

        Layering: this method returns the bare parent-scope Interrupt (produced by _suspend, a resume
        signal for the strategy's _adrive) or None; the RunResult is wrapped again only at the outermost
        aresume / _finish_resume boundary (internal resume signals are not wrapped in an envelope).
        """
        if result.interrupted:                          # result is the sub-agent's RunResult
            state.meta["awaiting"] = True
            return await self._suspend(state, result.interrupt.pendings, scope)   # Upload all of this sub-turn's pending actions (possibly more than one) at once
        state.meta["awaiting"] = False
        on_complete(result.final_output)                # on_complete takes the child's final output (str), not the whole envelope
        await self._checkpoint(state, scope)
        await child.clear_checkpoint(child_scope)
        return None

    # -- Shared pieces of the run boundary --

    def _pack_completed(self, output, input_text) -> "RunResult":
        """Pack the completed state into a completed RunResult (pure read-only snapshot + object construction; history persistence / checkpoint clearing are done explicitly and in order by the caller).

        Called after `_record`: new_messages match the pair already persisted to history, and
        snapshot_usage covers all usage of this turn.
        """
        from .result import RunResult, RunUsage
        new_msgs = (Message(input_text, "user"), Message(self._output_text(output), "assistant"))
        return RunResult(final_output=output, status="completed", usage=RunUsage(**snapshot_usage()),
                         new_messages=new_msgs, run_id=current_run_id())

    def _pack_interrupted(self, interrupt) -> "RunResult":
        """Pack the suspended state (a bare Interrupt) into an interrupted RunResult (no history persisted, state already in the checkpoint)."""
        from .result import RunResult, RunUsage
        return RunResult(final_output=None, status="interrupted", interrupt=interrupt,
                         usage=RunUsage(**snapshot_usage()), run_id=current_run_id())

    async def _record(self, input_text: str, output, scope) -> None:
        """Output guardrails (a trip raises before persisting history -> the blocked turn is not recorded) + atomically persist this turn (input + output text) to history by scope."""
        text = self._output_text(output)
        await self._check_guardrails(self.output_guardrails, text, "output")
        check_deadline()
        await self.add_messages([Message(input_text, "user"), Message(text, "assistant")], scope)

    async def _record_after_completion_marker(self, input_text: str, output, scope) -> None:
        """Record history after the at-most-once marker, clearing a deterministically blocked run.

        The completion marker deliberately precedes session persistence so a crash cannot replay completed
        tool side effects. A guardrail trip or a commit-boundary deadline rejection is different from a
        storage failure: the run is terminally rejected while the process is alive, so its marker is
        removed and the scope stays clean for a fresh run. If cleanup itself fails, keep the original
        exception and attach the cleanup failure for diagnostics.
        """
        try:
            await self._record(input_text, output, scope)
        except (GuardrailTripwireError, RunLimitExceeded) as error:
            try:
                await self._clear_checkpoint(scope)
            except Exception as cleanup_error:  # noqa: BLE001
                error.add_note(
                    "Failed to clear the completed checkpoint after the run was terminally rejected: "
                    f"{cleanup_error}")
            raise

    async def _check_guardrails(self, guardrails, text: str, stage: str) -> None:
        """Run a set of guardrails serially; the first passed=False fires the on_guardrail_trip(stage)
        hook and then raises GuardrailTripwireError (carrying a readable message; no automatic retry).
        stage is "input" / "output", letting hooks distinguish which guardrail it was. Called via
        g.acheck (with duck-typed fallback: a guardrail that only implements synchronous check goes
        through check): synchronous guardrails call check inline by default with zero extra overhead;
        asynchronous guardrails (including LLM moderation) await natively.
        """
        for g in guardrails:
            acheck = getattr(g, "acheck", None)
            result = await acheck(text) if acheck is not None else g.check(text)   # Duck-typed guardrail (only check) compatibility
            if not result.passed:
                message = result.message or "Guardrail tripped, blocked"
                await afire(self.hooks, "on_guardrail_trip", stage, message)
                raise GuardrailTripwireError(message)

    @staticmethod
    def _output_text(output) -> str:
        """Convert `_arun`'s output into text for guardrails / history: str as-is; a pydantic model via model_dump_json(); everything else via str()."""
        if isinstance(output, str):
            return output
        dump = getattr(output, "model_dump_json", None)
        return cast(str, dump()) if callable(dump) else str(output)

    @staticmethod
    def _is_interrupt(output) -> bool:
        """Whether output is a HITL suspend result Interrupt (deferred import, avoiding a core -> hitl top-level circular dependency)."""
        from ..runtime.hitl import Interrupt
        return isinstance(output, Interrupt)

    # -- HITL / checkpoint mechanisms (async: via the CheckpointStore a* variants, not blocking the event loop) --

    async def _suspend(self, state, pending, scope):
        """Suspend: record the pending action(s) (a single PendingAction or a list) into the state,
        persist the whole ExecutionState into the CheckpointStore, and return an Interrupt (called by
        the strategy's `_adrive`). A single value is normalized into a one-element list.
        """
        from ..runtime.hitl import Interrupt, PendingAction
        pendings = [pending] if isinstance(pending, PendingAction) else list(pending)
        state.pending = pendings
        self._stamp_run(state)
        checkpoint_store = cast("CheckpointStore", self.checkpoint_store)
        await checkpoint_store.asave(state.to_json(), scope=scope)
        return Interrupt(pendings, scope)

    async def _checkpoint(self, state, scope) -> None:
        """Per-step save: persist the current ExecutionState into the CheckpointStore (for crash recovery / long-task resume), called each step by the strategy's `_adrive`.

        A no-op when there is no checkpoint_store (zero overhead). Saved "after a step completes" (the
        tool result is already in messages), so resume does not re-run side effects. Saving each step
        means "this step completed normally, currently no suspend", so it clears `state.pending` in
        passing: it is only valid during a `_suspend`, and if not cleared after resume consumes the
        suspended action, it would linger in the checkpoint as a stale value (misleading UI / recovery
        logic that reads pending).
        """
        if self.checkpoint_store is not None:
            state.pending = []
            self._stamp_run(state)
            await self.checkpoint_store.asave(state.to_json(), scope=scope)

    @staticmethod
    def _stamp_run(state) -> None:
        """Record the current trace correlation (run_id / step) into state, so resume continues the same run_id + step sequence."""
        state.run_id = current_run_id()
        state.step = current_step()

    async def _mark_checkpoint_completed(self, scope, *, state=None) -> None:
        """At completion teardown, before clearing the checkpoint, mark the lingering checkpoint completed=True and persist it.

        "Persisting history via `_record` and clearing the checkpoint via `_clear_checkpoint` belong
        to different connections and are not atomic": a crash in between leaves an uncleared
        checkpoint that `resume(None)` treats as an unfinished run and re-runs, appending a second
        user+assistant pair. Marking completed=True first: the lingering checkpoint is then recognized
        by `_guard_pending` / `_load_execution_state` as "completed, only awaiting cleanup" and simply
        cleared rather than re-run (the cost is that a crash in the tiny window between the mark and
        persisting history loses that turn's history, but it never re-executes / double-counts;
        distributed side effects are inherently not exactly-once).

        A no-op when there is no checkpoint_store, when as_child (cleanup is left to the parent
        strategy), or when there is no lingering checkpoint. When state is passed, use it directly
        (the resume path); otherwise reload the current checkpoint from storage before marking (the run
        path has no state reference).
        """
        if self.checkpoint_store is None or self._defer_checkpoint_clear:
            return
        if state is None:
            from ..runtime.execution.state import (CHECKPOINT_FORMAT_VERSION, ExecutionState,
                                                   checkpoint_format_version)
            raw = await self.checkpoint_store.aload(scope=scope)
            if raw is None or checkpoint_format_version(raw) != CHECKPOINT_FORMAT_VERSION:
                return
            state = ExecutionState.from_json(raw)
        state.completed = True
        await self.checkpoint_store.asave(state.to_json(), scope=scope)

    async def _clear_checkpoint(self, scope) -> None:
        """Clear the checkpoint (after a run completes normally); a no-op when there is no checkpoint_store.

        When `_defer_checkpoint_clear` is set (an as_child sub-agent), it does not self-clear: the
        parent strategy clears it explicitly via `clear_checkpoint` after committing its own progress,
        guaranteeing "parent commits before child cleanup" (see the as_child semantics (2) in __init__).
        """
        if self.checkpoint_store is not None and not self._defer_checkpoint_clear:
            await self.checkpoint_store.aclear(scope=scope)

    def _child_agents(self):
        """This agent's list of internal sub-agents `[(child, scope_suffix), ...]` (orchestration strategies override to return their executor / critic); empty by default.

        For `clear_checkpoint` cascading cleanup: after clearing itself, it clears each sub-agent's
        checkpoint by its derived child scope (`_derive_scope(scope, suffix)`), avoiding the case where
        a user, following an error prompt, only clears the parent and leaves child checkpoints orphaned,
        permanently jamming that scope (the next sub-agent arun / delegation hits _guard_pending).
        """
        return []

    async def clear_checkpoint(self, scope: "Optional[Scope]" = None) -> None:
        """Explicitly clear a scope's checkpoint (ignoring the defer flag), and cascade-clear the internal sub-agents' derived child-scope checkpoints.

        For a parent strategy to clean up sub-agent checkpoints after committing its own progress, and
        for a user to manually clear a stuck suspended task: cascading ensures it does not clear only
        the parent and leave child checkpoints orphaned (the next sub-agent arun / delegation hitting
        `_guard_pending` permanently jams). A sub-agent without a checkpoint_store is a no-op each.

        Args:
            scope: The session identifier to clear; defaults to self.scope.
        """
        sc = scope if scope is not None else self.scope
        if self.checkpoint_store is not None:
            await self.checkpoint_store.aclear(scope=sc)
        for child, suffix in self._child_agents():
            await child.clear_checkpoint(self._derive_scope(sc, suffix))

    async def _load_execution_state(self, scope):
        """Reload an unfinished ExecutionState from the CheckpointStore; raises SessionError when there
        is no store or no checkpoint; a checkpoint with a missing or mismatched format version is cleared
        and raises SessionError so the user can re-initiate safely.
        """
        from ..runtime.execution.state import CHECKPOINT_FORMAT_VERSION, ExecutionState, checkpoint_format_version
        if self.checkpoint_store is None:
            raise SessionError("checkpoint_store is not configured, cannot resume (pass one when constructing the Agent to enable HITL / crash recovery)")
        raw = await self.checkpoint_store.aload(scope=scope)
        if raw is None:
            raise SessionError("No recoverable checkpoint (scope does not match, or the run has completed / been resumed)")
        if checkpoint_format_version(raw) != CHECKPOINT_FORMAT_VERSION:
            await self.checkpoint_store.aclear(scope=scope)
            raise SessionError(
                "This suspended task's checkpoint format is incompatible; it has been cleared automatically. Please re-initiate the task.")
        state = ExecutionState.from_json(raw)
        if state.completed:
            await self.checkpoint_store.aclear(scope=scope)   # A completed lingering checkpoint (crashed between record and clear): nothing recoverable, clear it and fail loud, do not re-run / double-count
            raise SessionError(
                "This run's execution completed, so it cannot be resumed safely. Its history may already have "
                "been written, or finalization may have stopped before history persistence; reconcile session "
                "history if needed. The lingering checkpoint has been cleared.")
        return state

    @staticmethod
    def _inject_decision(state, decision) -> None:
        """HITL resume: inject the approve / reject decision into the pending calls' decision table.

        - `dict {call_id: bool}`: inject per call_id (each pending action decided separately); only
          honors call_ids belonging to the current pending.
        - `bool`: uniformly approve / reject all pending actions (in the single-action case, decides
          that one).
        - `None` (crash-recovery resume): no injection; pending actions will re-suspend and await
          approval again.
        """
        if decision is None:
            return
        valid_ids = {p.call_id for p in state.pending}
        if isinstance(decision, dict):
            for call_id, value in decision.items():
                if call_id in valid_ids:            # Only inject calls belonging to this suspend, ignoring unrelated keys (to prevent mis-injection)
                    state.decisions[call_id] = bool(value)
        else:
            for call_id in valid_ids:               # A single bool: decide all pending actions uniformly
                state.decisions[call_id] = bool(decision)

    async def _finish_resume(self, output, state, scope):
        """Repack a re-suspend as an interrupted RunResult, or finalize history and checkpoint state.

        The order matches run's at-most-once policy: mark execution completed before session
        persistence, then record and clear. A persistence failure keeps the completed marker so tool
        side effects are not replayed; an output guardrail trip clears it because blocked output is
        terminal and has no resumable result. Resume does not re-fire on_run_start.
        """
        if self._is_interrupt(output):
            await afire(self.hooks, "on_interrupt", output.pendings, scope=scope)
            return self._pack_interrupted(output)
        check_deadline()
        state.completed = True
        await self._mark_checkpoint_completed(scope, state=state)   # Mark completed and persist first: a crash between record and clear then leaves a lingering checkpoint that resume(None) does not treat as unfinished and re-run, not re-appending history
        await self._record_after_completion_marker(state.input_text, output, scope)
        await self._clear_checkpoint(scope)
        await afire(self.hooks, "on_run_end", output, scope=scope)
        return self._pack_completed(output, state.input_text)

    # -- Conversation history (multi-session by scope) --

    async def _history_for(self, scope) -> list[Message]:
        """Fetch this scope's conversation history (the async real body): with a session_store, read
        from it via aload each time (single source of truth, cross-process, not blocking the event
        loop); without one, use the in-process dict[scope] (multi-session capable, lost on restart).
        The synchronous public face get_history funnels to it via run_sync.
        """
        if self.session_store is not None:
            return await self.session_store.aload(scope=scope)
        return self._sessions.setdefault(scope, [])

    async def add_message(self, message: Message, scope: "Optional[Scope]" = None) -> None:
        """Append one message to the conversation history by scope (a single-message convenience over add_messages).

        Args:
            message: The message to append.
            scope: The session identifier; defaults to self.scope.
        """
        await self.add_messages([message], scope)

    async def add_messages(self, messages: list[Message], scope: "Optional[Scope]" = None) -> None:
        """Append multiple messages to the conversation history by scope atomically; with a
        session_store it goes through its aappend_many (single transaction, avoiding a half-turn),
        otherwise the in-process dict[scope] is extended once. This is the single funnel point for
        session persistence (the async real body, using a* so as not to block the event loop).

        Args:
            messages: The list of messages to append.
            scope: The session identifier; defaults to self.scope.
        """
        scope = scope if scope is not None else self.scope
        if self.session_store is not None:
            aappend_many = getattr(self.session_store, "aappend_many", None)
            if callable(aappend_many):
                await cast(Awaitable[None], aappend_many(messages, scope=scope))
            else:
                for m in messages:
                    await self.session_store.aappend(m, scope=scope)
        else:
            self._sessions.setdefault(scope, []).extend(messages)

    def clear_history(self, scope: "Optional[Scope]" = None) -> None:
        """Clear a scope's conversation history; with a session_store, clear its persisted record, otherwise clear the in-process dict[scope].

        Args:
            scope: The session identifier; defaults to self.scope.
        """
        scope = scope if scope is not None else self.scope
        if self.session_store is not None:
            self.session_store.clear(scope=scope)
        else:
            self._sessions.pop(scope, None)

    def get_history(self, scope: "Optional[Scope]" = None) -> list[Message]:
        """Return a deep copy of a scope's conversation history, so external mutations (including changing a Message's content / metadata) do not affect internal state.

        Args:
            scope: The session identifier; defaults to self.scope.

        Returns:
            list[Message]: A deep copy of the history message list (Message is a mutable dataclass, so
            a shallow copy would not prevent field mutation).

        The synchronous public facade (funneling to the async real body _history_for via run_sync): in
        environments with a running event loop (async / Jupyter / FastAPI), use
        `await agent._history_for(scope)` instead (same limitation as the run/resume synchronous facades).
        """
        return copy.deepcopy(run_sync(self._history_for(scope if scope is not None else self.scope)))

    def get_prompts(self) -> dict:
        """List all built-in prompts this agent currently uses, as {key: text} (from the shared self.prompts).

        For inspection / export: first see which prompts the framework builds in, what their keys are
        called, and what they look like, then decide which to override (see agentmaker/doc/prompts.md).
        """
        return self.prompts.as_dict()

    def update_prompts(self, updates) -> None:
        """Override built-in prompts: updates may be {key: new-text} (per key) or another PromptRegistry (a whole swap, e.g. changing language).

        Modifies self.prompts in place: this agent's harness and internal sub-agents share the same
        copy, so one change takes effect across the whole chain. Each override is validated to ensure
        "the placeholders + protocol tokens are still present", raising PromptError if any is missing.
        It is isolated per instance by default (a copy of DEFAULT_PROMPTS is made at construction when
        prompts= is not passed), so this method only changes this agent's chain and does not affect
        other agents / independently constructed tools / the process-wide global. To change language /
        wording process-wide, explicitly call `DEFAULT_PROMPTS.override(pack)` before creating any component.

        Boundaries: (1) an already-created tool's overall description is a construction-time snapshot
        and does not change with this override (the parameter descriptions do change live), so
        changing language / tool-related prompts should be done before creating tools (use packs'
        chinese_registry(), or override first then build tools), otherwise the same tool's schema shows
        a half-mixed state of "overall intro in the old language, parameter descriptions in the new".
        (2) After the default isolation, a tool constructed separately (not passed the same prompts)
        has its wording determined by the tool's own prompts and does not change with this agent's
        update_prompts: to keep the tool and agent in sync, pass the same PromptRegistry to both, or
        use a global override.
        """
        self.prompts.override(updates)

    def __str__(self) -> str:
        """Let print(agent) show "Agent(name, provider)" for debugging.

        Returns:
            str: Of the form "Agent(name=assistant, provider=deepseek)".
        """
        return f"Agent(name={self.name}, provider={self.llm.provider})"


_BASE = BaseAgent          # Late binding for __init_subclass__ (see the in-class comment)
