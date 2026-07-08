"""agentmaker.runtime.harness: the cross-cutting service layer (harness).

The cross-cutting work every reasoning paradigm has to do (call the LLM, execute tools, emit traces) is
collected into this single class. Each paradigm only manages its own control flow and delegates all
cross-cutting concerns here. That way the cross-cutting logic is written once, every paradigm is on equal
footing, and a new paradigm gets it for free.

Fully async: the cross-cutting methods are all `a*` coroutines (`acall_llm`, `aexec_tool`, `aassemble`,
`areduce`, `astructured`, `atools_for`, etc.); the LLM client's `chat`/`stream` are async too (the whole
framework is async, with no synchronous surface). Cases that need a synchronous call are driven through
`core/aio` at the paradigm's `run`/`stream_run` synchronous facade layer; the harness layer no longer has a
synchronous facade.

Minimal by default (with no tracer/compactor/context_builder attached, behavior is identical to calling the
llm / registry directly):
    - acall_llm: call the LLM (optionally with tools) + emit a trace (with model/usage/latency when a tracer is attached)
    - aexec_tool: execute a tool (with permission gate / high-risk confirmation) + emit a trace (with params/result/latency)
    - aassemble: assemble history (compact when a compactor is attached) + inject a memory/RAG system block (when context_builder+sources are attached)
    - atools_for: this turn's tool schema (take the Tool-RAG subset when a tool_retriever is attached, otherwise the full set)
    - trace: emit structured events to the tracer (a no-op with zero overhead when none is attached)
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..core.exceptions import LLMResponseError, RunCancelled, RunLimitExceeded
from ..core.llm_clients import LLMClient
from ..core.text import TokenCounter, count_tokens
from ..prompts import DEFAULT_PROMPTS
from .hitl import ApprovalRequired, PendingAction
from .hooks import afire
from .execution.run_context import check_limits, correlation, enforce_token_limit_after_llm, record_llm, record_tool
from ..core.trace_events import (EVENT_CONTEXT_BLOCK, EVENT_CONTEXT_COMPACT, EVENT_CONTEXT_REDUCE,
                                 EVENT_LLM_CALL, EVENT_SUMMARIZE_FAILED, EVENT_TOOL_CALL)
from ..tools.registry import ToolRegistry
from ..tools.response import ToolResponse

if TYPE_CHECKING:                       # Type annotations only, not imported at runtime (keeps the harness from dragging in the context / retrieval / tool sub-stacks)
    from ..context.builder import ContextBuilder
    from ..context.history_compactor import HistoryCompactor
    from ..context.types import ContextSource, ReducerConfig
    from ..context.window_budget import WindowBudgetConfig
    from ..core.llm_response import LLMResponse
    from ..tools.base import ConfirmCallback
    from ..tools.permissions import ToolPermissions
    from ..tools.tool_retriever import ToolRetriever
    from .observability import Tracer

_logger = logging.getLogger(__name__)   # Warnings about truncated LLM output go here (the library installs no handler, the host takes over; see the NullHandler in agentmaker/__init__)

# Default number of retries after structured-output validation fails (used by astructured)
_DEFAULT_STRUCTURED_RETRIES = 1

# finish_reason / stop_reason values indicating output was truncated by length (normalized across protocols): OpenAI length; Anthropic max_tokens / context overflow
_TRUNCATION_REASONS = frozenset({"length", "max_tokens", "model_context_window_exceeded"})


@dataclass(frozen=True)
class HarnessConfig:
    """A set of harness-level cross-cutting knobs (shared across paradigms), assembled into a Harness by BaseAgent._make_harness.

    Deliberately excludes hooks / prompts (auto-injected by _make_harness as self._harness_hooks / self.prompts, eliminating a
    "remember to pass it" hidden contract) and tool_registry (given per paradigm as needed via _make_harness(tool_registry=...);
    Reflection's own harness intentionally carries none).
    """
    tracer: "Optional[Tracer]" = None
    confirm: "Optional[ConfirmCallback]" = None
    permissions: "Optional[ToolPermissions]" = None
    compactor: "Optional[HistoryCompactor]" = None
    tool_retriever: "Optional[ToolRetriever]" = None
    context_builder: "Optional[ContextBuilder]" = None
    sources: "Optional[list[ContextSource]]" = None
    reducer: "Optional[ReducerConfig]" = None
    window_budget: "Optional[WindowBudgetConfig]" = None
    token_counter: TokenCounter = count_tokens   # Pluggable token counter (defaults to count_tokens); used by _window_budget / areduce


def cli_confirm(tool, parameters: dict) -> bool:
    """Command-line high-risk confirmation battery: prints the action, asks y/n, returns whether it was approved.

    An explicit battery, not the default: CLI / teaching scenarios pass `confirm=cli_confirm` explicitly to use it. The
    framework no longer treats it as a fallback default: with no confirm passed, the safe choice is to deny (registry
    semantics), avoiding a headless server `input()` that hangs forever / raises EOFError.
    """
    print(f"\n⚠️  Tool '{tool.name}' is a high-risk operation, parameters: {parameters}")
    return input("Confirm execution? (y/N) ").strip().lower() == "y"


def _approved(tool, parameters: dict) -> bool:
    """The "already approved" confirmation callback for HITL mode: approval is handled at the _approval_gate layer, so the registry need not ask again (avoids double confirmation)."""
    return True


# Structured-output helpers.
# The structured instruction / retry note live in the registry (harness.schema_instruction / harness.retry_note); astructured renders them via self.prompts.


def _extract_json(text: str) -> str:
    """Extract JSON from model output: take from the first '{' to the last '}' (tolerates a ```json fence and stray surrounding text); empty string if none."""
    t = (text or "").strip()
    i, j = t.find("{"), t.rfind("}")
    return t[i:j + 1] if i != -1 and j > i else ""


def _validate_structured(schema_model, content, prompts):
    """Extract JSON + pydantic-validate it. Returns (instance, None) on success / (None, error text) on failure. The error text is taken from prompts (fed back to the model, swappable by language)."""
    from pydantic import ValidationError  # Used only on the structured path, imported lazily
    raw = _extract_json(content)
    if not raw:
        return None, prompts.text("harness.validate_empty")
    try:
        return schema_model.model_validate_json(raw), None   # Invalid JSON is also normalized by pydantic into a ValidationError
    except ValidationError as e:
        return None, prompts.render("harness.validate_failed", detail=e)


class Harness:
    """The Agent's cross-cutting services: acall_llm / aexec_tool / trace, shared across paradigms."""

    def __init__(self, llm: LLMClient, *, tool_registry: Optional[ToolRegistry] = None,
                 confirm: "Optional[ConfirmCallback]" = None, tracer=None, permissions=None, hooks=None,
                 compactor=None, tool_retriever=None, context_builder=None, sources=None, reducer=None,
                 window_budget=None, prompts=None, token_counter: TokenCounter = count_tokens):
        """
        Args:
            llm: The LLM client.
            tool_registry: Tool registry (for tool execution); pure-reasoning paradigms may omit it.
            confirm: High-risk tool confirmation callback, forwarded to ToolRegistry; if omitted, the safe choice is to deny (registry default, so a headless server does not hang). CLI scenarios may pass cli_confirm explicitly.
            tracer: Optional tracer (must have an emit(event) method); if omitted, trace is a no-op (zero overhead).
            permissions: Optional tool permissions (ToolPermissions, allow/deny lists); when attached, exec_tool runs the
                permission gate before HITL approval: a tool that is denied / not in allow is rejected outright, without even
                asking for approval (deny = hard ban). If not attached, there is no restriction.
            hooks: Optional lifecycle hook list (Hook, observe-only); when attached, call_llm fires before/after_model and
                exec_tool fires before/after_tool (run-level events are fired by the Agent). Zero overhead if not attached.
            compactor: Optional history compactor (HistoryCompactor); compacts overlong history during assemble to prevent token buildup.
            tool_retriever: Optional Tool-RAG (ToolRetriever); tools_for takes only the relevant tool subset by query.
            context_builder: Optional context assembler (ContextBuilder); once attached together with sources, assemble
                retrieves memory/RAG by query and stitches it into a single guardrailed system block for injection (history still stays as role-tagged messages).
            sources: Optional list of context sources (CallableSource wrapping memory.search / rag.retrieve, etc.); requires context_builder.
            reducer: Optional trajectory-reduction knobs (ReducerConfig); reduce() uses it to decide how many verbatim steps to keep at the tail of each paradigm.
                The trajectory's token budget comes from the window budget (see window_budget), not configured here. If omitted, ReducerConfig() defaults are used.
            window_budget: Optional window-allocation knobs (WindowBudgetConfig); context_block and reduce use it to divide the
                whole window across output reserve / fixed overhead (system + tool schema) / retrieval block / trajectory in one
                accounting, replacing the earlier "two half-windows each guessing on their own".
                If omitted, WindowBudgetConfig() defaults are used; when the window is unknown (llm.context_window is None), it falls back to "no cap / no reduction".

        Half-set assembly fails loud: context_builder and sources must come as a pair. Giving only one means the retrieval block
        silently has no effect (sources are consumed only by context_block), so raise instead of letting someone believe memory/RAG
        is wired up. The other dependencies are independent; missing one simply means that capability is off, not a half-set.
        """
        if context_builder is not None and not (sources or []):
            raise ValueError(
                "context_builder was attached but no sources were passed: the memory/RAG retrieval block will silently have no effect. "
                "To inject it, also pass sources=[...] (CallableSource wrapping memory.search / rag.retrieve); for pure history assembly, do not pass context_builder.")
        if sources and context_builder is None:
            raise ValueError(
                "sources were passed but no context_builder was attached: the sources will not be retrieval-assembled and silently have no effect. "
                "To inject them, also pass context_builder=ContextBuilder(...); to skip injection, do not pass sources.")
        # tool_registry duck-typing check: if non-None it must be able to produce an OpenAI schema (the framework's core dependency
        # on it). No isinstance enforcement: this allows third-party registry wrappers / proxies (same duck-typing philosophy as
        # agent.py applies to LLM capability slots); passing a str or other object without to_openai_schema fails loud at
        # construction, rather than blowing up with AttributeError deep inside _full_schema at runtime.
        if tool_registry is not None and not callable(getattr(tool_registry, "to_openai_schema", None)):
            raise TypeError(
                f"tool_registry must be a ToolRegistry (or an equivalent object with a to_openai_schema method), got {type(tool_registry).__name__}. "
                "Common cause: passing system_prompt as the 3rd positional arg to PlanAgent, whose signature is now (name, llm, system_prompt=None, *, tool_registry=None, ...).")
        self.llm = llm
        self.tool_registry = tool_registry
        self.confirm = confirm   # No fallback interactive confirmation: if not given it is None, passed through to the registry which safely denies (a headless server does not hang); CLI scenarios pass cli_confirm explicitly
        self.tracer = tracer
        self.permissions = permissions
        self.hooks = hooks or []
        self.compactor = compactor
        self.tool_retriever = tool_retriever
        self.context_builder = context_builder
        self.sources = sources or []
        self.reducer = reducer            # ReducerConfig (knobs for keeping recent step counts); when None, reduce() uses ReducerConfig() defaults
        self.window_budget = window_budget  # WindowBudgetConfig (window-accounting knobs); when None, WindowBudgetConfig() defaults are used
        self.prompts = prompts or DEFAULT_PROMPTS   # Prompt registry; context_guard / structured instruction / retry / the harness's own tool errors are taken from it
        self._count = token_counter       # Pluggable token counter (defaults to count_tokens); _window_budget estimates tool overhead, areduce passes it through to the reduction functions
        self._tool_tokens = None          # Token cache for the tool schema: the tool table does not change once built, so compute lazily once (avoids re-tokenizing on every reduce step)

    async def acall_llm(self, messages: "list[dict]", *, tools: "Optional[list[dict]]" = None, **kwargs) -> "LLMResponse":
        """Call the LLM (include tools only when given, to match a direct call) + emit a trace. Returns an LLMResponse.

        Timing and the assembled trace event (model/usage/latency/whether there are tool calls) happen only when a tracer is attached; otherwise there is zero extra overhead.
        When hooks are attached, before_model / after_model fire around the call (timing brackets only the LLM call, not the hook time).
        When a RunPolicy is attached, check_limits runs before the call and record_llm after (over-limit / cancelled raises and aborts this turn).
        """
        check_limits("llm")
        await afire(self.hooks, "before_model", messages)
        start = time.perf_counter() if self.tracer is not None else None
        extra = {"tools": tools} if tools else {}       # An empty list is synonymous with None: send no tools parameter (tools=[] triggers 400 with some providers, behavior varies)
        # The Anthropic protocol requires max_tokens: when the caller does not give one explicitly, use the window budget's output_reserve so the
        # desired_output_tokens knob takes effect for Claude and is no longer capped short by the adapter's hardcoded 4096 (option B: only
        # send it down under Anthropic; OpenAI/DeepSeek/Gemini keep the model's server-side default, zero regression).
        if "max_tokens" not in kwargs and getattr(self.llm, "protocol", None) == "anthropic":
            wb = self._window_budget()
            if wb is not None and wb.output_reserve > 0:
                kwargs = {**kwargs, "max_tokens": wb.output_reserve}
        resp = await self.llm.chat(messages, **extra, **kwargs)
        record_llm(getattr(resp, "usage", None))
        if start is not None:
            self.trace(self._llm_event(resp, start))
        self._warn_if_truncated(resp)                   # Make truncation observable: don't let "cut short" be mistaken for "done answering"
        await afire(self.hooks, "after_model", resp)
        enforce_token_limit_after_llm()
        return resp

    def _warn_if_truncated(self, resp) -> None:
        """Warn when finish_reason indicates output was truncated by length (the reply may be incomplete, don't treat it as finished); finish_reason also goes into the llm_call trace for inspection."""
        fr = getattr(resp, "finish_reason", None)
        if fr in _TRUNCATION_REASONS:
            model = getattr(resp, "model", None) or getattr(self.llm, "model", None)
            _logger.warning("LLM output was truncated (finish_reason=%s, model=%s): the reply may be incomplete; increase max_tokens / WindowBudgetConfig.desired_output_tokens",
                            fr, model)

    async def astream_llm(self, messages: "list[dict]", **kwargs):
        """Stream the LLM: yield text piece by piece; finish cleanly whether it drains normally / the consumer interrupts early /
        the stream raises mid-flow: emit one trace event (streaming usage is usually None, so only model/latency are recorded)
        and call record_llm.

        When hooks are attached, before_model fires before the stream starts. A plain text stream has no
        single response object, so after_model does not fire; when tools are passed the stream ends with a
        terminal LLMResponse (see LLMClient.stream) and after_model fires for it, matching acall_llm.
        When a RunPolicy is attached, check_limits runs before the start and record_llm after (streaming usage is usually None, only the call count is counted).
        Wrap-up goes in finally: a generator only runs to its next yield, so if the consumer breaks/closes early (GeneratorExit)
        or the stream raises mid-flow, statements after the loop are skipped. Only finally guarantees that an already-initiated
        streaming call is always accounted for and that before_model has a paired wrap-up event.
        """
        check_limits("llm")
        await afire(self.hooks, "before_model", messages)
        start = time.perf_counter() if self.tracer is not None else None
        stats_box = {}
        try:
            async for piece in self.llm.stream(messages, on_stats=lambda s: stats_box.update(s=s), **kwargs):
                if not isinstance(piece, str):
                    await afire(self.hooks, "after_model", piece)   # terminal LLMResponse of a tool-bearing stream: hook parity with acall_llm
                yield piece
        finally:
            stats = stats_box.get("s")                           # The stats returned for this call (may be None on an early break)
            usage = getattr(stats, "usage", None)                # On an early break usage is missing, only the call count is counted
            record_llm(usage)                                    # Streaming usage feeds the RunPolicy limit, no longer reading the concurrency-unreliable last_stream_stats
            if start is not None:
                self.trace(self._stream_event(start, usage, getattr(stats, "finish_reason", None)))
            if stats is not None:
                self._warn_if_truncated(stats)                   # Warn on streaming truncation too (consistent with non-streaming; stats carries finish_reason)
        enforce_token_limit_after_llm()                          # Check the limit only on a normal drain (on an early-break GeneratorExit this line is not reached after finally)

    async def astructured(self, messages, schema_model, *, retries=_DEFAULT_STRUCTURED_RETRIES, **kwargs):
        """Make the model output a structured result per a pydantic model: inject the schema, call the LLM, pydantic-validate,
        feed failures back and retry, return the instance. The cross-cutting implementation (single async copy).

        Args:
            messages: The unified message list.
            schema_model: A pydantic BaseModel subclass: it both fixes the output shape and performs validation.
            retries: How many times to retry after a validation failure (default 1).
            **kwargs: Passed through to the LLM (temperature, etc.).

        Returns:
            An instance of schema_model (already validated).

        Notes:
            The schema goes through each provider's native path via output_schema (json_schema / output_config / responseSchema)
            and is also injected into the system prompt: native providers get double insurance, json_object/none providers rely on
            the prompt. Everything is finally pydantic-validated; failures feed the error back and retry. After retries attempts it still raises LLMError if unresolved.
        """
        schema = schema_model.model_json_schema()
        msgs = [{"role": "system", "content": self.prompts.render("harness.schema_instruction", schema=json.dumps(schema, ensure_ascii=False))}] + list(messages)
        err = None
        for _ in range(retries + 1):
            resp = await self.acall_llm(msgs, output_schema=schema, **kwargs)
            obj, err = _validate_structured(schema_model, resp.content, self.prompts)
            if err is None:
                return obj
            msgs = msgs + [{"role": "assistant", "content": resp.content or ""},
                           {"role": "user", "content": self.prompts.render("harness.retry_note", err=err)}]   # Failed output + correction, keeping the user/assistant alternation
        raise LLMResponseError(f"Structured output still failed validation after {retries} retries: {err}")

    async def aexec_tool(self, name: str, parameters: dict, *,
                         call_id: Optional[str] = None, decisions: Optional[dict] = None) -> ToolResponse:
        """Execute a tool (with permission gate + high-risk confirmation, the confirmation callback held by this harness) + emit a trace. Returns a ToolResponse.
        The cross-cutting implementation (single async copy); the permission gate / HITL approval gate are pure-logic helpers _permission_gate / _approval_gate.

        Permissions: when permissions are attached, the permission gate runs first: a tool that is denied / not in allow returns a rejection result outright (before HITL).
        HITL: passing decisions (this turn's decision table {call_id: bool}) means HITL mode. For a high-risk tool, if that call_id
        has no decision it raises ApprovalRequired (a suspend signal captured by the policy loop); if already rejected it returns a
        rejection result. Not passing decisions (the default) goes through the synchronous confirm path. If no confirm is passed the
        safe choice is to deny (no stdin dependency); a blocking confirm like cli_confirm is dispatched to a thread pool (so it does
        not stall the event loop), but server scenarios should still use HITL (checkpoint_store + decisions) rather than interactive confirm.
        Timing and the assembled trace event (tool name/params/status/result/latency; secrets and sensitive paths redacted by the Tracer) happen only when a tracer is attached.
        """
        if self.tool_registry is None:
            return ToolResponse.error(self.prompts.render("tool.error.no_registry", name=name))
        check_limits("tool")
        denied = self._permission_gate(name)
        if denied is not None:
            self.trace_tool_gate(name, parameters, "denied", denied.text)   # A gate-blocked call is traced too (an audit must see "the AI tried to call X and was blocked")
            return denied
        gated = self._approval_gate(name, parameters, call_id, decisions)
        if gated is not None:
            self.trace_tool_gate(name, parameters, "rejected", gated.text)
            return gated
        # HITL mode (decisions is not None): by this point approval of any high-risk tool has been handled by _approval_gate (approved / not high-risk),
        # so call the registry with _approved and do not trigger the synchronous confirm again (avoids double confirmation); non-HITL mode uses self.confirm as usual.
        confirm = _approved if decisions is not None else self.confirm
        await afire(self.hooks, "before_tool", name, parameters)   # Past the permission / approval gate, fired only when execution is really about to happen
        start = time.perf_counter() if self.tracer is not None else None
        result, executed = await self.tool_registry.aexecute_tool_checked(name, parameters, confirm=confirm)
        if executed:
            record_tool()                                   # Counted only when tool.run is actually entered (calls blocked by param validation / confirmation are not)
        if start is not None:
            self.trace(self._tool_event(name, parameters, result, start))
        await afire(self.hooks, "after_tool", name, parameters, result)
        return result

    def _permission_gate(self, name: str) -> Optional[ToolResponse]:
        """Permission gate (a pure-logic helper for aexec_tool; no await):

        A tool that is denied / not in allow / whose origin is rejected returns a rejection ToolResponse; allowed or no permissions configured returns None (zero overhead).
        Runs before the HITL approval gate: a banned tool is blocked outright, without even asking for approval (deny = hard ban, taking precedence over review). Judged by the
        Tool object (including the origin dimension, same as the visible surface _visible_schema, so deny_origins is not missed at the execution point); returns a rejection result rather than raising, feeding it back so the LLM reroutes.
        """
        if self.permissions is None:
            return None
        tool = self.tool_registry.get(name) if self.tool_registry is not None else None
        reason = self.permissions.denial_reason(tool if tool is not None else name)
        return ToolResponse.error(self.prompts.render("tool.error.denied", reason=reason)) if reason is not None else None

    def _approval_gate(self, name: str, parameters: dict, call_id, decisions) -> Optional[ToolResponse]:
        """HITL approval gate (a pure-logic helper for aexec_tool; no await):

        - Non-HITL mode (decisions is None) or a non-high-risk tool: returns None (allowed, proceeds to normal execution).
        - High-risk and no decision for this call this turn: raises ApprovalRequired (suspend; the policy loop captures it, packages state, and converts to Interrupt).
        - High-risk and already rejected (decisions[call_id] is False): returns a rejection ToolResponse (fed back so the LLM reroutes, not an error-terminate).
        """
        if decisions is None:
            return None
        tool = self.tool_registry.get(name)
        if tool is None or not tool.needs_confirmation(parameters):
            return None
        if call_id is None:   # decisions is keyed by call_id: a None key would let multiple suspended actions in one turn overwrite each other and cross-contaminate approvals, so enforce it hard, no verbal contract
            raise ValueError(
                "In HITL mode a high-risk tool must carry a unique call_id (aexec_tool(call_id=...)), otherwise the decisions table gets key collisions")
        decision = decisions.get(call_id)
        if decision is True:
            return None                                       # Only an explicit approval is allowed through
        if decision is False:
            return ToolResponse.error(self.prompts.render("tool.error.user_rejected"))
        raise ApprovalRequired(PendingAction(name, parameters, call_id))   # No decision / not a True truthy value (defense in depth): re-suspend, never let through

    async def aassemble(self, history, query: str = "", scope=None) -> list[dict]:
        """Assemble the history messages sent to the LLM and, as needed, inject one memory/RAG system context block at the front. The cross-cutting implementation (single async copy).

        - When a compactor is attached: overlong history is summary-compacted first (to prevent token buildup; via acompact, with the
          summary going through _asummarize so it is governed). When compaction actually happens (acompact returns a new list, not the
          original history object), emit an EVENT_CONTEXT_COMPACT (symmetric with trajectory reduction's EVENT_CONTEXT_REDUCE: it lets
          the app see whether compaction actually fired and how many old turns were summarized away, no longer leaving saved tokens untraceable).
        - When context_builder + sources are attached and query is non-empty: retrieve memory/RAG by query and stitch it into a
          guardrailed system message inserted at the front (the conversation history still stays as role-tagged messages, not flattened). Retrieval is synchronous IO, dispatched to a thread pool.

        history is list[Message]; query is this turn's user input (for memory/RAG retrieval, no block injected if omitted);
        scope runs through from run and is passed to retrieval sources (each source decides which dimensions to use, see CallableSource). Returns list[dict].
        """
        if self.compactor is not None:
            msgs = await self.compactor.acompact(history, asummarize=self._asummarize)
            if msgs is not history:                      # Emit only on real compaction (when not triggered / the summary is empty, acompact returns the same history object): same criterion as areduce's out is not data
                self.trace({"type": EVENT_CONTEXT_COMPACT, "before": len(history), "after": len(msgs)})
        else:
            msgs = history
        block = await self.acontext_block(query, scope)   # Truly async: multiple sources run concurrently via abuild_block's gather (returns an empty string internally when builder/sources are not attached or query is empty)
        return self._assemble_out(msgs, block)

    @staticmethod
    def _assemble_out(msgs, block: str) -> list[dict]:
        """Convert messages to a list of dicts; if there is a memory/RAG block, insert it as a system message at the front (aassemble wrap-up)."""
        out = [m.to_dict() for m in msgs]
        if block:
            out.insert(0, {"role": "system", "content": block})
        return out

    def _window_budget(self):
        """Compute a window budget (WindowBudget) for this run; returns None when the window is unknown (llm.context_window is None).

        Fixed overhead takes the token count of the full tool description, counting the tool schema's footprint into the budget and
        filling the previously unmeasured gap. It is a heuristic estimate, not a strict upper bound: it estimates from the text tool
        description (get_tools_description), whereas native function-calling actually sends the JSON schema, so the two token counts
        differ slightly. But fixed is only used to divide the budget between "retrieval block / trajectory", so a small mis-estimate
        only affects the sizes of those two and never pushes the request past the window.
        The tool table does not change once built, so the first computed value is cached (self._tool_tokens) to avoid re-tokenizing on every reduce step.
        The system prompt is not counted separately: on the ReAct path it is already counted inside the reducer's protected head; for other paradigms it is relatively small and left as headroom.
        """
        from ..context.window_budget import WindowBudget, WindowBudgetConfig
        if self._tool_tokens is None:                    # Lazy compute + cache (the tool table does not change over the run)
            self._tool_tokens = self._count(self._full_description()) if self.tool_registry is not None else 0
        cfg = self.window_budget or WindowBudgetConfig()
        # No retrieval sources configured (sources empty; the harness already enforces builder/sources as a pair): the retrieval block is always an empty string, so set rag_ratio to 0 and don't waste trajectory budget on it
        rag_ratio = 0.0 if not self.sources else None
        return WindowBudget.for_run(llm=self.llm, cfg=cfg, tool_tokens=self._tool_tokens, rag_ratio=rag_ratio)

    def context_block(self, query: str, scope=None) -> str:
        """memory/RAG to a single guardrailed system context block of text; returns an empty string if builder/sources are not configured or query is empty.

        A public method: the single-loop Agent inserts it as a system message at the front automatically via aassemble; the Plan / Reflection recipes call it directly and stitch it into each stage's prompt.
        The retrieval block's token budget comes from the window budget (WindowBudget.rag_budget), same source as the trajectory budget, no longer each taking half the window.
        scope runs through from run and is passed to build_block to source.fetch (each source decides which dimensions to use).
        """
        if self.context_builder is None or not self.sources or not query:
            return ""
        start = time.perf_counter() if self.tracer is not None else None
        wb = self._window_budget()
        budget = wb.rag_budget if wb is not None else None   # Window unknown: None, and the builder falls back to its own criterion
        block = self.context_builder.build_block(query, sources=self.sources, scope=scope, budget=budget)
        return self._context_block_out(block, query, start)

    async def acontext_block(self, query: str, scope=None) -> str:
        """The async implementation of context_block (used by aassemble): budget / guardrail / trace match the sync version, but multiple sources run concurrently via abuild_block's gather."""
        if self.context_builder is None or not self.sources or not query:
            return ""
        start = time.perf_counter() if self.tracer is not None else None
        wb = self._window_budget()
        budget = wb.rag_budget if wb is not None else None
        block = await self.context_builder.abuild_block(query, sources=self.sources, scope=scope, budget=budget)
        return self._context_block_out(block, query, start)

    def _context_block_out(self, block: str, query: str, start) -> str:
        """Shared wrap-up for context_block / acontext_block: emit a trace (when a tracer is attached) + wrap it with the anti-injection guardrail prefix."""
        if start is not None:
            self.trace({"type": EVENT_CONTEXT_BLOCK, "query": query, "block_chars": len(block or ""),
                        "sources": len(self.sources), "latency_ms": int((time.perf_counter() - start) * 1000)})
        return self.prompts.text("harness.context_guard") + block if block else ""

    async def areduce(self, kind: str, data, **kw):
        """Loss-aware reduction of a paradigm's own trajectory (acts only when over budget; emits a trace if reduction happens). kind: agent / plan / reflection.
        The cross-cutting implementation (single async copy): the reduction functions (REDUCERS) are natively async, and the summary
        callback is _asummarize (completes within the same event loop, with no thread / loop hopping, so governance counting and
        limit pass-through naturally hold).

        The budget comes from the window budget WindowBudget.trajectory_budget (same-source accounting as the retrieval block; when the window is unknown, no reduction and returned as-is);
        each paradigm's preserve-policy is in context.reducer, keeping the lifeline (Agent's protected region + recent steps, Reflection's
        latest answer + critique points, Plan's key numbers). data's shape is determined by kind (agent=messages / plan=list[str] /
        reflection=list[{kind,text}]); returned as-is means no reduction happened this time.

        Args:
            **kw: The paradigm's private reduction parameters, passed through to the reduction function as-is (e.g. Agent's turn_start=this turn's start index).
        """
        from ..context.reducer import REDUCERS
        from ..context.types import ReducerConfig
        wb = self._window_budget()
        if wb is None:                                   # Window unknown: no reduction
            return data
        # Agent's trajectory is state.messages (the retrieval block is inside its protected region): rag_in_scope=True; Plan/Reflection reduce fragments: False
        budget = wb.trajectory_budget(rag_in_scope=(kind == "agent"))
        rc = self.reducer or ReducerConfig()             # Use defaults if not injected (each paradigm keeps recent step counts)
        # Each paradigm's keep_recent comes from config (reflection has no such parameter)
        extra = ({"keep_recent_steps": rc.agent_keep_recent_steps} if kind == "agent"
                 else {"keep_recent": rc.plan_keep_recent} if kind == "plan" else {})
        out = await REDUCERS[kind](data, summarize=self._asummarize, budget=budget, counter=self._count, **extra, **kw)
        if out is not data:                              # Reduction happened: emit a trace (don't silently drop the signal)
            self.trace({"type": EVENT_CONTEXT_REDUCE, "paradigm": kind, "before": len(data), "after": len(out)})
        return out

    async def _asummarize(self, text: str, instruction: str) -> str:
        """The summary used for compaction / reduction, via acall_llm, so it is included in hooks / tracer / RunPolicy counting (compaction cost does not bypass governance).

        On LLM / network failure it degrades to an empty string (each reducer decides its degradation copy, without raising to break the flow); but RunLimitExceeded / RunCancelled
        propagate: governance signals must not be swallowed (otherwise compaction's tokens are not counted against the limit and the limit is toothless).
        """
        try:
            return (await self.acall_llm([{"role": "system", "content": instruction},
                                          {"role": "user", "content": text}])).content.strip()
        except (RunLimitExceeded, RunCancelled):
            raise
        except Exception:  # noqa: BLE001
            self.trace({"type": EVENT_SUMMARIZE_FAILED, **correlation()})  # A silent degradation still leaves one observable event: distinguishes "compaction never triggered" from "the compaction LLM keeps failing"
            return ""

    async def atools_for(self, query) -> list[dict]:
        """The tool schema to expose to the LLM this turn: when a tool_retriever is attached, take the relevant subset by query (Tool-RAG), otherwise the full set;
        finally filter out banned tools by permissions (deny=hard ban, not even the schema is sent: saves tokens + does not leak the full registry to the model).
        The cross-cutting implementation (single async copy); retrieval is synchronous IO, dispatched to a thread pool to avoid blocking the event loop."""
        if self.tool_retriever is not None:
            schemas = await asyncio.to_thread(self.tool_retriever.schema_for, query)
        else:
            schemas = self._full_schema()
        return self._visible_schema(schemas)

    def _visible_schema(self, schemas: list[dict]) -> list[dict]:
        """Filter out the schema of rejected tools by permissions (consistent with "deny=hard ban": a banned tool's parameter schema is not sent to the model either).
        Judged by the Tool object (including the origin dimension); with no permissions configured / no tool table it is returned as-is. Permission filtering is collected here; the retrieval layer knows nothing about permissions."""
        if self.permissions is None or self.tool_registry is None:
            return schemas
        out = []
        for s in schemas:
            name = s.get("function", {}).get("name")
            tool = self.tool_registry.get(name)
            if self.permissions.denial_reason(tool if tool is not None else name) is None:
                out.append(s)
        return out

    def _full_schema(self) -> list[dict]:
        """The full tool schema (the fallback for atools_for when no tool_retriever is attached; empty if there is no tool table)."""
        return self.tool_registry.to_openai_schema() if self.tool_registry is not None else []

    def _full_description(self) -> str:
        """The full text tool description (used by _window_budget to estimate tool schema overhead; empty string if there is no tool table)."""
        return self.tool_registry.get_tools_description() if self.tool_registry is not None else ""

    def _llm_event(self, resp, start: float) -> dict:
        """Assemble one llm_call event (used by acall_llm).

        Merges `run_id` / `step_index` from `correlation()` (present only within a run context), so events can be grouped by run and ordered by step.
        """
        return {"type": EVENT_LLM_CALL,
                "model": getattr(resp, "model", None) or getattr(self.llm, "model", None),  # Prefer the response's actual model (may differ from the request name under a proxy/alias), fall back to the request name if missing
                "latency_ms": round((time.perf_counter() - start) * 1000),
                "usage": getattr(resp, "usage", None),
                "has_tool_calls": bool(getattr(resp, "tool_calls", None)),
                "finish_reason": getattr(resp, "finish_reason", None), **correlation()}

    def _stream_event(self, start: float, usage=None, finish_reason=None) -> dict:
        """Assemble one streaming llm_call event (used by astream_llm). usage / finish_reason come from the per-call StreamStats (may be missing on an early break)."""
        return {"type": EVENT_LLM_CALL, "model": getattr(self.llm, "model", None),
                "latency_ms": round((time.perf_counter() - start) * 1000), "streamed": True,
                "usage": usage, "finish_reason": finish_reason, **correlation()}

    def _tool_event(self, name: str, parameters: dict, result: ToolResponse, start: float) -> dict:
        """Assemble one tool_call event (used by aexec_tool): records status + result text."""
        return {"type": EVENT_TOOL_CALL, "tool": name, "params": parameters,
                "latency_ms": round((time.perf_counter() - start) * 1000),
                "status": result.status, "result": result.text, **correlation()}

    def trace_tool_gate(self, name: str, parameters: dict, status: str, detail: str = "") -> None:
        """Emit a "gate-blocked / invalid params" tool_call event (shared by aexec_tool's denied/rejected and the unified Agent's invalid_args).

        Calls denied by the permission gate, rejected by HITL, or failing parameter parsing previously did not get traced, so an audit could not see "the AI tried to call X but was blocked", exactly the thing that should be recorded.
        No latency (nothing was really executed); status is "denied" / "rejected" / "invalid_args". A no-op with zero overhead when no tracer is attached.
        """
        self.trace({"type": EVENT_TOOL_CALL, "tool": name, "params": parameters,
                    "status": status, "result": detail, **correlation()})

    def trace(self, event: dict) -> None:
        """Emit one structured trace event; a no-op with zero overhead when no tracer is attached."""
        if self.tracer is not None:
            self.tracer.emit(event)
