"""agentmaker.agents.agent: Agent, the framework's single "model calls tools in a loop" execution primitive.

Control flow stays with the model: each turn the messages are sent to the LLM (tools declared via the
native function-calling `tools` parameter), and the model either answers with text directly (loop ends) or
issues tool_calls. The framework executes the tools, feeds results back as role:"tool", and calls the next
turn, until the model answers or the turn budget is exhausted. With no tools the first turn is terminal,
which is plain question-answering. This one loop covers both the "chat" and "react" usages (the latter is
just a preset: tools required plus a system prompt asking the model to write its thinking before acting).

Inherits BaseAgent to get all the run boundaries for free (guardrails, history persistence, HITL
suspend/resume, checkpoints, hooks, sync facade). This class only implements `_arun` / `_adrive` (the loop
body) and streaming `astream_run`. The cross-cutting concerns (calling the model, executing tools, context
assembly, trajectory trimming) all go through the Harness.

Structured work inside the loop:
    - Trajectory trimming: before each model call, `areduce("agent", messages, turn_start=...)`. The
      protected region is the entire initial assembly (system + compacted history + RAG blocks + current
      user, delimited by the `meta["turn_start"]` index); the trimmable region is this turn's tool
      trajectory (an assistant(tool_calls) message and its tool results are an indivisible atom, see
      context/reducer.reduce_agent).
    - Tool discovery expansion: when a tool_search-style tool returns data.discovered, the new tools'
      schemas are merged into this turn's available set (so when Tool-RAG's static preselection misses a
      tool that only becomes relevant mid-run, the model can search for it and use it).
    - call_id collision guard: when a new turn's tool_calls have ids that collide with existing keys in the
      decision table (some servers auto-increment within a response and repeat across responses), rewrite
      them to unique ids so an old approval cannot be borrowed by a new action.
    - Empty-reply fallback: when the model returns empty text, nudge it once (agent.empty_reply); if still
      empty, return agent.invalid_reply. It never returns an empty string.
"""

import asyncio
import json
from typing import TYPE_CHECKING, Optional

from agentmaker.agents.base import BaseAgent
from agentmaker.core import LLMClient, Message, TokenCounter, count_tokens
from agentmaker.core.aio import iter_sync
from agentmaker.runtime.execution import ExecutionState
from agentmaker.runtime.harness import HarnessConfig
from agentmaker.runtime.hitl import ApprovalRequired, PendingAction
from agentmaker.runtime.hooks import afire
from agentmaker.tools import ToolRegistry

if TYPE_CHECKING:                       # Type hints only; not imported at runtime (keeps the base Agent from pulling in the context / retrieval stack)
    from agentmaker.context import ContextBuilder, ContextSource
    from agentmaker.context.history_compactor import HistoryCompactor
    from agentmaker.context.types import ReducerConfig
    from agentmaker.tools import ConfirmCallback
    from agentmaker.tools.tool_retriever import ToolRetriever


class Agent(BaseAgent):
    """Single-loop Agent: one input -> model-tool loop -> reply; with no tools it is plain question-answering. Automatically maintains multi-turn history per scope."""

    def __init__(self, name: str, llm: LLMClient, system_prompt: Optional[str] = None, *,
                 tools=None,
                 tool_registry: Optional[ToolRegistry] = None,
                 max_turns: int = 10,
                 compactor: "Optional[HistoryCompactor]" = None,
                 confirm: "Optional[ConfirmCallback]" = None,
                 permissions=None,
                 reducer: "Optional[ReducerConfig]" = None,
                 tool_retriever: "Optional[ToolRetriever]" = None,
                 context_builder: "Optional[ContextBuilder]" = None,
                 sources: "Optional[list[ContextSource]]" = None,
                 window_budget=None,
                 token_counter: TokenCounter = count_tokens,
                 tracer=None, hooks=None, run_policy=None,
                 session_store=None, scope=None, checkpoint_store=None,
                 input_guardrails=None, output_guardrails=None, prompts=None,
                 on_pending: str = "error", as_child: bool = False):
        """
        Args:
            name: Agent name.
            llm: LLM client.
            system_prompt: System prompt defining the persona; optional (if omitted, no system message is sent).
            tools: Convenience entry point: a list[Tool] (including @tool-decorated function objects) or a
                ToolRegistry, normalized into a registry internally via ToolRegistry.from_tools. Use it for a
                one-line start in simple cases; mutually exclusive with tool_registry (passing both raises ValueError).
            tool_registry: Tool registry (advanced case: reuse the same registry or customize prompts).
                Passing it enables the tool loop; omitting it means plain question-answering.
            max_turns: The maximum number of model turns in the loop, to prevent repeated calls from
                spinning into an infinite loop (must be a positive integer, default 10, since tool tasks
                often need multiple turns).
            compactor: Optional history compactor (HistoryCompactor); once attached, cross-turn session
                history is automatically summarized and compacted when it exceeds a threshold.
            confirm: Confirmation callback for high-risk tools (tool, params) -> bool; if omitted, it
                safely refuses (returns a readable error fed back to the model, without blocking on stdin).
                In CLI / teaching scenarios pass agentmaker.runtime.cli_confirm explicitly for command-line
                y/n; in server scenarios configure HITL (checkpoint_store).
            permissions: Optional tool permissions (ToolPermissions, allow/deny lists); denied tools are
                rejected directly at the execution gate.
            reducer: Optional trajectory-trimming knob (ReducerConfig); when this turn's tool trajectory
                overflows the window, keep recent units and summarize the earlier ones.
            tool_retriever: Optional Tool-RAG (ToolRetriever); selects only the relevant subset of tools for
                this turn's input (saves tokens when there are many tools).
            context_builder + sources: Optional memory/RAG injection; each turn retrieves by input and
                assembles a guardrailed system block for injection.
            window_budget: Optional window-budgeting knob (WindowBudgetConfig); unifies budgeting across
                retrieval blocks, trajectory, and output reservation.
            token_counter: Pluggable token counter (default count_tokens); the harness uses it to estimate
                tool overhead and to trim trajectories. For consistent accounting throughout, pass the same
                token_counter to your ContextBuilder / HistoryCompactor as well (the Agent does not silently
                override the counters they already carry).
            session_store / scope: Session persistence (see BaseAgent); once attached, history is resumed
                per scope and persisted automatically.
            checkpoint_store: Optional checkpoint store; once attached, enables HITL suspend/resume plus
                crash recovery (saved at every step).
            input_guardrails / output_guardrails / hooks / run_policy / prompts / on_pending: see BaseAgent.
            as_child: Set True when acting as an orchestration recipe's internal sub-agent (see BaseAgent:
                run-level hooks do not fire and checkpoint cleanup is left to the parent).
        """
        cfg = HarnessConfig(tracer=tracer, confirm=confirm, permissions=permissions, compactor=compactor,
                            tool_retriever=tool_retriever, context_builder=context_builder, sources=sources,
                            reducer=reducer, window_budget=window_budget, token_counter=token_counter)
        super().__init__(name, llm, system_prompt, session_store=session_store, scope=scope,
                         checkpoint_store=checkpoint_store, hooks=hooks, run_policy=run_policy,
                         harness_config=cfg, input_guardrails=input_guardrails, output_guardrails=output_guardrails,
                         prompts=prompts, on_pending=on_pending, as_child=as_child)
        if max_turns <= 0:
            raise ValueError(f"max_turns must be a positive integer, got {max_turns}")
        if tools is not None and tool_registry is not None:
            raise ValueError("tools and tool_registry are mutually exclusive, pick one (tools is the list[Tool]/registry convenience entry point, tool_registry is the advanced one)")
        # The internal registry built from the tools convenience entry point inherits this agent's prompts,
        # so registry-level errors (tool not found, argument validation) match the agent's language.
        resolved_registry = tool_registry if tool_registry is not None else ToolRegistry.from_tools(tools, prompts=self.prompts)
        # The framework only uses native function calling: configuring tools with a model that does not
        # support fc fails loud at construction time, rather than silently failing tools at runtime.
        # getattr defaults to True: duck-typed LLMs that do not declare this capability bit (test stubs,
        # etc.) are not blocked; only an explicit supports_function_calling=False is rejected.
        if resolved_registry is not None and getattr(llm, "supports_function_calling", True) is False:
            raise ValueError(
                "The model declares it does not support native function calling (supports_function_calling=False), "
                "but this Agent is configured with tools. Tool calling depends on native fc by default and "
                "cannot work. Pick one of three: (1) switch to a model that supports fc; (2) if you are sure "
                "this model supports it, construct LLMClient(..., supports_function_calling=True); (3) enable "
                "\"text emulation\" for the fc-less model by constructing LLMClient(..., emulate_tools=True) "
                "(writes the tool catalog into the prompt and parses calls from plain-text replies, see "
                "core/adapters/tool_emulation.py).")
        self.tool_registry = resolved_registry
        self.max_turns = max_turns
        # Cross-cutting funnel: calling the LLM, executing tools, history assembly, Tool-RAG, trajectory
        # trimming, and tracing all go through the harness.
        # Assembled via the base _make_harness, which auto-injects _harness_hooks (model/tool-level
        # observation unaffected by as_child) and self.prompts.
        self.harness = self._make_harness(tool_registry=resolved_registry)

    # Generation entry point (invoked by the base arun template; guardrails / history persistence funneled in the base).

    async def _arun(self, input_text: str, *, scope, output_schema=None, verbose: bool = False, **kwargs):
        """Single generation: passing output_schema goes structured (plain question-answering, no tools, returns the validated instance); otherwise enters the model-tool loop.

        Args:
            input_text: User input.
            scope: Session identifier (passed in by the base); determines which session history to load and
                where suspended state is stored.
            output_schema: Optional pydantic model; if given, structured output (whether to go structured is
                the caller's choice, non-structured by default).
            verbose: Whether to print each turn's thinking and tool results (🔧 prefix); default False
                (library-quiet).
            **kwargs: Passed through to the LLM (e.g. temperature, max_tokens).

        Returns:
            str (default) / an instance of output_schema / Interrupt (HITL suspend awaiting approval).
        """
        if output_schema is not None:
            return await self.harness.astructured(await self._initial_messages(input_text, scope),
                                                  output_schema, **kwargs)
        state = ExecutionState(messages=await self._initial_messages(input_text, scope),
                               input_text=input_text, remaining=self.max_turns)
        state.meta["turn_start"] = len(state.messages)   # Protected-region boundary: everything before this is system + history + RAG blocks + current user
        return await self._adrive(state, scope=scope, verbose=verbose, **kwargs)

    async def _initial_messages(self, input_text: str, scope) -> list[dict]:
        """Assemble the initial messages: system (custom persona or the default chat.persona) + assembled session history (compaction + memory/RAG block injection) + current user."""
        messages = [{"role": "system", "content": self.system_prompt or self.prompts.text("chat.persona")}]
        messages.extend(await self.harness.aassemble(await self._history_for(scope), input_text, scope))
        messages.append({"role": "user", "content": input_text})
        return messages

    # Loop body (shared by first run and resume; the base aresume calls it to continue).

    async def _adrive(self, state, *, scope, verbose: bool = False, **kwargs):
        """Drive the model-tool loop from an ExecutionState (can resume from mid-run).

        First consumes the calls remaining from the last suspend point (meta["pending_calls"]), then
        continues the main loop (state.remaining). In non-persistent mode (no checkpoint_store), decisions
        is None, so tools use synchronous confirm and never suspend. Suspend / per-step saves write
        checkpoints under the passed-in scope (base _suspend / _checkpoint). Returns str (done) or Interrupt
        (suspended).
        """
        tools = await self.harness.atools_for(state.input_text)
        decisions = state.decisions if self.checkpoint_store is not None else None
        susp = await self._run_calls(state.meta.pop("pending_calls", []), state, decisions, scope, verbose)
        if susp is not None:
            return susp
        await self._checkpoint(state, scope)               # Save after consuming the suspend leftover calls: avoids re-running already-approved tools if it crashes after resume
        tools = self._expand_tools(tools, state)     # If there were tool_search discoveries before resume (stored in meta), the continuation can use them too
        nudged = False                               # Empty replies are nudged only once (in-process is enough, not checkpointed: a best-effort fallback)
        while state.remaining > 0:
            state.remaining -= 1
            state.messages = await self.harness.areduce("agent", state.messages,
                                                        turn_start=state.meta.get("turn_start", 0))
            resp = await self.harness.acall_llm(state.messages, tools=tools, **kwargs)
            if not resp.tool_calls:
                content = (resp.content or "").strip()
                if content:
                    return content                   # No tool calls and a non-empty reply: this is the final answer
                if nudged:                           # Still empty after nudging: final fallback text (never returns an empty string)
                    return self.prompts.text("agent.invalid_reply")
                nudged = True                        # Occasional empty reply (content went into reasoning, the formal reply is empty): nudge once
                state.messages.append({"role": "user", "content": self.prompts.text("agent.empty_reply")})
                continue
            calls = self._unique_calls(resp.tool_calls, state)
            if verbose and (resp.content or "").strip():
                print(f"🔧[Thinking] {resp.content.strip()}")
            state.messages.append({"role": "assistant", "content": resp.content or "", "tool_calls": calls})
            susp = await self._run_calls(calls, state, decisions, scope, verbose)
            if susp is not None:
                return susp
            await self._checkpoint(state, scope)           # Save after a turn (LLM+tools) completes: crash recovery does not re-run already-completed tools
            tools = self._expand_tools(tools, state)   # Tools discovered mid-run are available the next turn
        return self.prompts.text("agent.exhausted")

    def _unique_calls(self, calls: list, state) -> list:
        """Prevent approval borrowing: if a new turn's tool_calls have ids that collide with existing keys
        in the decision table (some servers issue "call_0"-style ids that auto-increment within a response
        and repeat across responses), rewrite them to unique ids (appending a turn-and-index suffix).
        Otherwise `decisions[old_id] is True` would let a new action borrow an old approval and execute
        directly. The rewritten id is used consistently across all three places: the assistant message,
        tool execution, and feeding results back (this method funnels that consistency)."""
        if not state.decisions:
            return list(calls)
        turn = self.max_turns - state.remaining
        out = []
        for i, c in enumerate(calls):
            if c.get("id") in state.decisions:
                c = {**c, "id": f"{c['id']}#t{turn}-{i}"}
            out.append(c)
        return out

    async def _run_calls(self, calls, state, decisions, scope, verbose: bool = False):
        """Execute this turn's tool_calls: batch adjacent "parallel-safe read-only" calls concurrently
        (asyncio.gather, backfilled in original order), and run the rest strictly serially. On hitting one
        that needs approval, record the remaining calls into meta, persist under scope via the base
        _suspend, and return Interrupt; otherwise feed results back as role:"tool" and return None.

        Parallelism only touches a safe subset: a parallel sub-batch contains only consecutive calls that
        parsed successfully, support parallel, need no confirmation this time, and are not in the decision
        table. They can never raise ApprovalRequired and have no per-decision saves, so batching them does
        not disturb the two ordered semantics of "HITL suspend saves calls[idx:]" and "per-decision save"
        (the serial branch below is left verbatim). After executing each serial call with a decision (a
        high-risk action already approved/rejected at resume), it immediately `_checkpoint`s, shrinking the
        "side-effect already happened, board still suspended" double-execution window from the whole batch
        to a single tool (still at-least-once).
        """
        idx, n = 0, len(calls)
        while idx < n:
            run = self._parallel_run_len(calls, idx, decisions)   # Length of the "parallel-eligible consecutive sub-batch" starting at idx
            if run >= 2:                                          # Only worth concurrency at >=2 (a single concurrent call has no benefit, falls to the serial branch)
                await self._run_parallel_batch(calls[idx:idx + run], state, decisions, scope, verbose)
                idx += run
                continue
            call = calls[idx]
            name, params, perr = self._parse_call(call)
            if perr is not None:
                # Emit a tool_call trace even on argument-parse failure (status=invalid_args): otherwise an audit only sees the error stuffed into the message, not this "invalid call"
                self.harness.trace_tool_gate(name, {"raw_arguments": call["function"]["arguments"]}, "invalid_args", perr)
                state.messages.append({"role": "tool", "tool_call_id": call["id"], "content": perr})
                idx += 1
                continue
            try:
                result = await self.harness.aexec_tool(name, params, call_id=call["id"], decisions=decisions)
            except ApprovalRequired:
                # No more tools execute this turn from here on; surface all "needs approval, no decision" actions in the remaining calls at once (one suspend, batch approval),
                # pending_calls keeps calls[idx:] (including the read-only tools in between) so they replay in original call order on resume
                state.meta["pending_calls"] = list(calls[idx:])
                return await self._suspend(state, self._collect_pending(calls[idx:], decisions), scope)
            self._collect_discovered(result, state)
            if verbose:
                print(f"🔧  [{name}] {result.text}")
            state.messages.append({"role": "tool", "tool_call_id": call["id"],
                                   "content": self._tool_content(name, result)})
            if decisions and call["id"] in decisions:      # Just ran a high-risk call already approved/rejected: save immediately, shorten the double-execution window
                state.meta["pending_calls"] = list(calls[idx + 1:])   # Remaining unexecuted calls (resume continues from here after a crash)
                await self._checkpoint(state, scope)
            idx += 1
        state.meta.pop("pending_calls", None)              # Whole batch executed: clear the mid-batch "remaining calls" marker so leftovers are not replayed on resume
        return None

    def _collect_pending(self, calls, decisions) -> list:
        """Scan the calls for every "needs approval and has no decision yet" call, making each a
        PendingAction (for one suspend with batch approval).

        A pure read-only scan that executes no tools: mirrors the harness approval gate's decision
        (tool.needs_confirmation(params) and no decision for call_id). Skips parse failures, no-confirmation,
        and already-decided (including already-rejected) calls. This surfaces all of "one request's several
        high-risk actions" at once, rather than suspending and resuming one at a time. Always includes at
        least the call that triggered this suspend (so it is never empty).
        """
        out = []
        for call in calls:
            name, params, perr = self._parse_call(call)
            if perr is not None:
                continue
            tool = self.tool_registry.get(name) if self.tool_registry is not None else None
            if tool is None or not tool.needs_confirmation(params):
                continue
            if decisions and decisions.get(call["id"]) is not None:   # Already approved/rejected: do not suspend again
                continue
            out.append(PendingAction(name, params, call["id"]))
        return out

    def _parallel_run_len(self, calls, start, decisions) -> int:
        """Count consecutive "parallel-eligible" calls starting at start (the boundary for _run_calls to batch concurrency); stop at the first non-eligible one."""
        k = start
        while k < len(calls) and self._parallel_eligible(calls[k], decisions):
            k += 1
        return k - start

    def _parallel_eligible(self, call, decisions) -> bool:
        """Whether this call can join a parallel batch: parsed successfully + tool supports_parallel + needs no confirmation this time + not in the decision table.

        Excludes calls that need confirmation or are in the decision table (they may raise
        ApprovalRequired, requiring per-idx suspend or per-decision saves, and must never disturb the
        ordered semantics), as well as parse failures, no registry, or no supports_parallel flag (all
        serial). So a parallel batch inherently has no suspend, no per-decision save, and no side-effect
        ordering dependency.
        """
        if self.tool_registry is None:
            return False
        if decisions and call.get("id") in decisions:     # A high-risk call in the HITL decisions: never parallel (belt-and-suspenders; normally it would not be supports_parallel anyway)
            return False
        name, params, perr = self._parse_call(call)
        if perr is not None:
            return False
        tool = self.tool_registry.get(name)
        if tool is None or not getattr(tool, "supports_parallel", False):
            return False
        return not tool.needs_confirmation(params)         # A confirmation-required (high-risk) call is not parallelized even if flagged supports_parallel

    async def _run_parallel_batch(self, batch, state, decisions, scope, verbose: bool) -> None:
        """Execute a batch of "parallel-safe read-only" calls concurrently (asyncio.gather), then backfill results in the original call order (collect discoveries + feed back as role:"tool").

        Calls in the batch are all non-high-risk with no decision (see _parallel_eligible), so they do not
        raise ApprovalRequired and have no per-decision save. gather preserves order for backfilling, and
        the caller _adrive does a single _checkpoint at the end of the turn for this batch's results. Tool
        result messages are appended in original call order, so the feed-back order is identical to serial.
        """
        parsed = [self._parse_call(c) for c in batch]      # All confirmed to parse successfully (guaranteed by _parallel_eligible)
        results = await asyncio.gather(*(
            self.harness.aexec_tool(name, params, call_id=c["id"], decisions=decisions)
            for c, (name, params, _perr) in zip(batch, parsed)))
        for c, (name, _params, _perr), result in zip(batch, parsed, results):   # Backfill in original call order (gather preserves order)
            self._collect_discovered(result, state)
            if verbose:
                print(f"🔧  [{name}] {result.text}")
            state.messages.append({"role": "tool", "tool_call_id": c["id"],
                                   "content": self._tool_content(name, result)})

    def _tool_content(self, name: str, result) -> str:
        """The tool-result text fed back to the model: successful results from external_content tools
        (search / RAG / MCP) are wrapped in an anti-injection delimiter guardrail (OWASP LLM01: external
        text hiding "ignore previous instructions..." lures), the same as memory/RAG's context_guard;
        everything else is passed through verbatim. Only the copy the LLM sees is wrapped: the ToolResponse
        itself is unchanged (trace / data / _collect_discovered still read the original)."""
        tool = self.tool_registry.get(name) if self.tool_registry is not None else None
        if tool is not None and getattr(tool, "external_content", False) and result.status != "error":
            return self.prompts.render("tool.external_guard", content=result.text)
        return result.text

    @staticmethod
    def _parse_call(call: dict):
        """Parse a native tool_call into (tool_name, params dict, None) on success / (tool_name, None, error_text) on invalid args."""
        name = call["function"]["name"]
        try:
            return name, json.loads(call["function"]["arguments"] or "{}"), None
        except json.JSONDecodeError:
            return name, None, f"Error: arguments for tool '{name}' are not valid JSON"

    @staticmethod
    def _collect_discovered(result, state) -> None:
        """Collect discovered tool names from a tool result (ToolSearchTool's data.discovered) into meta (deduplicated), for _expand_tools to expand the available set."""
        data = getattr(result, "data", None)
        if isinstance(data, dict) and data.get("discovered"):
            known = state.meta.setdefault("discovered_tools", [])
            known.extend(n for n in data["discovered"] if n not in known)

    def _expand_tools(self, tools, state):
        """Merge tools discovered via tool_search (state.meta["discovered_tools"], collected by _run_calls) into this turn's available schema.

        Static preselection (Tool-RAG freezes a subset by the initial query) misses tools that only become
        relevant mid-run. After the model searches with ToolSearchTool, this appends the new names' schemas
        to tools so the next LLM call can invoke them directly (the execution end aexec_tool already runs
        the whole registry by name). The discovery list in meta is read-only, not consumed (an idempotent
        union): it persists with the checkpoint, so if any turn suspends or crashes after a discovery, the
        list can still expand tools back on resume. A consuming pop would let a "discovery" not survive the
        next save.
        """
        discovered = state.meta.get("discovered_tools")
        if not discovered or self.tool_registry is None:
            return tools
        have = {t.get("function", {}).get("name") for t in (tools or [])}
        perms = self.harness.permissions             # Expansion also passes the permission gate: denied tools do not enter the model-visible set (consistent with atools_for)
        new_names = [n for n in discovered
                     if n not in have and (t := self.tool_registry.get(n)) is not None
                     and (perms is None or perms.denial_reason(t) is None)]
        if not new_names:
            return tools
        return list(tools or []) + self.tool_registry.to_openai_schema(names=new_names)

    # Streaming (plain text, or a streaming tool loop when the agent has tools; no HITL suspend / checkpoints).

    async def astream_run(self, input_text: str, *, scope=None, buffer_output: bool = False, trace_carrier=None, **kwargs):
        """Streaming question-answering (the real async implementation): yields reply text piece by piece.

        Shares the base `_scaffold` (run context with scope + run-level hooks + exception routing). The
        output-side duties run inside the generator body only after the stream drains naturally: output
        guardrails -> one atomic history save -> on_run_end. If the consumer breaks early (break / aclose),
        none of the three happen (GeneratorExit is a BaseException and does not reach on_error), but the
        harness-level finally accounting always runs.

        Tool loop: when the agent has registered tools, this runs the same model-tool loop as arun but with
        every model turn streamed -- text deltas are yielded as they arrive (including text preceding a tool
        call, matching common chat UIs), tools execute between turns (before_tool/after_tool hooks fire as
        usual), and the final no-tool-call turn's text is the answer. Unlike arun, the streaming loop does
        NOT support HITL suspend/resume or checkpoints: a tool that requires confirmation uses its
        synchronous confirm (ApprovalRequired propagates as an error in persistent setups) -- use arun when
        you need suspend semantics. History save matches arun: one atomic user + final-assistant turn (tool
        trajectories are not persisted to session history).

        Output guardrail vs. streaming trade-off (buffer_output):
        - Default `buffer_output=False`: yield as it generates, check the guardrail on the final answer only
          after the stream drains. Content already emitted cannot be recalled (a tripped guardrail can only
          raise after the fact, it cannot stop text that already reached the consumer).
        - `buffer_output=True`: accumulate the full output first, run the output guardrail, and only after it
          passes release it piece by piece. With a tool loop this buffers ALL streamed text across turns and
          releases it only after the final answer passes the guardrail (tools still execute live in between).

        Args:
            input_text: User input (str, or a multimodal content-part list).
            scope: This session's identifier; defaults to self.scope.
            buffer_output: See above; default False (immediate streaming, guardrail checked after the fact).
            trace_carrier: Optional upstream W3C trace carrier (see the same-named arg on arun); only useful
                when an OTelExporter(carrier_provider=...) is attached.
            **kwargs: Passed through to the LLM streaming interface.
        """
        scope = scope if scope is not None else self.scope
        async with self._scope_lock(scope), self._scaffold(input_text, scope, trace_carrier):
            await self._check_guardrails(self.input_guardrails, input_text, "input")
            tools = await self.harness.atools_for(input_text)
            if tools:
                async for piece in self._astream_tool_loop(input_text, scope, tools,
                                                           buffer_output=buffer_output, **kwargs):
                    yield piece
                return
            messages = await self._initial_messages(input_text, scope)
            pieces = []
            if buffer_output:
                async for piece in self.harness.astream_llm(messages, **kwargs):
                    pieces.append(piece)                          # Accumulate everything first, do not yield
                output = "".join(pieces)
                await self._check_guardrails(self.output_guardrails, output, "output")   # Only release after passing; a trip raises here, nothing emitted
                for piece in pieces:
                    yield piece
            else:
                async for piece in self.harness.astream_llm(messages, **kwargs):
                    pieces.append(piece)
                    yield piece                                   # Emit as it generates: guardrail checked after the fact, cannot stop emitted content
                output = "".join(pieces)
                await self._check_guardrails(self.output_guardrails, output, "output")
            await self.add_messages([Message(input_text, "user"), Message(output, "assistant")], scope)  # One atomic turn save
            await afire(self.hooks, "on_run_end", output, scope=scope)

    async def _astream_tool_loop(self, input_text, scope, tools, *, buffer_output: bool, **kwargs):
        """The streaming model-tool loop body (called inside astream_run's scaffold; see its docstring).

        Mirrors _adrive's turn structure (reduce -> model -> tools -> repeat) but each model turn goes
        through harness.astream_llm with tools: text deltas stream out live, and the turn's terminal
        LLMResponse (see LLMClient.stream) carries the tool calls. Fallback texts that were never streamed
        (empty-reply fallback / turns exhausted) are yielded once at the end so the consumer always
        receives the final answer as stream content.
        """
        state = ExecutionState(messages=await self._initial_messages(input_text, scope),
                               input_text=input_text, remaining=self.max_turns)
        state.meta["turn_start"] = len(state.messages)
        tools = self._expand_tools(tools, state)
        buffered, nudged, output, unstreamed_tail = [], False, None, None
        while state.remaining > 0:
            state.remaining -= 1
            state.messages = await self.harness.areduce("agent", state.messages,
                                                        turn_start=state.meta.get("turn_start", 0))
            resp = None
            async for piece in self.harness.astream_llm(state.messages, tools=tools, **kwargs):
                if not isinstance(piece, str):
                    resp = piece                                  # terminal turn response (tool_calls live here)
                elif buffer_output:
                    buffered.append(piece)
                else:
                    yield piece
            if resp is None:                                      # defensive: adapter ended without a terminal response
                break
            if not resp.tool_calls:
                content = (resp.content or "").strip()
                if content:
                    output = resp.content                         # final answer: its text already streamed / buffered above
                    break
                if nudged:
                    output = unstreamed_tail = self.prompts.text("agent.invalid_reply")
                    break                                         # fallback text was never streamed: emitted after the guardrail below
                nudged = True                                     # occasional empty reply: nudge once, same as _adrive
                state.messages.append({"role": "user", "content": self.prompts.text("agent.empty_reply")})
                continue
            calls = self._unique_calls(resp.tool_calls, state)
            state.messages.append({"role": "assistant", "content": resp.content or "", "tool_calls": calls})
            await self._run_calls(calls, state, None, scope, False)   # decisions=None: synchronous confirm, never suspends
            tools = self._expand_tools(tools, state)
        if output is None:                                        # loop exhausted without a final answer
            output = unstreamed_tail = self.prompts.text("agent.exhausted")
        await self._check_guardrails(self.output_guardrails, output, "output")
        if buffer_output:
            for piece in buffered:
                yield piece                                       # release everything only after the final answer passed the guardrail
        if unstreamed_tail is not None:
            yield unstreamed_tail                                 # fallback texts reach the consumer as stream content too
        await self.add_messages([Message(input_text, "user"), Message(output, "assistant")], scope)  # One atomic turn save
        await afire(self.hooks, "on_run_end", output, scope=scope)

    def stream_run(self, input_text: str, *, scope=None, **kwargs):
        """Synchronous facade for astream_run (pumped via aio.iter_sync): returns a sync generator that yields piece by piece.
        Drain it explicitly or close it (e.g. contextlib.closing); in an async environment use astream_run instead."""
        return iter_sync(self.astream_run(input_text, scope=scope, **kwargs))


