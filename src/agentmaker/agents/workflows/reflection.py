"""agentmaker.agents.workflows.reflection: ReflectionAgent, the generate -> reflect -> refine orchestration recipe.

The model first produces a draft, then repeatedly polishes it via "self-critique (reflect) -> refine based on the critique" until the
critique decides no further change is needed or the iteration cap is reached. The stage order is fixed in code (draft -> (critique ->
refine) xN); the control flow is not in the model's hands, which is the fundamental difference from the single-loop Agent.

Optional tools (first-class support): if tool_registry is passed, the critique step can call tools to verify facts / arithmetic
(CRITIC-style self-correction, attacking the "factuality" dimension that LLM self-reflection is least reliable at). That step is delegated
to an internal critique-executor (a single-loop Agent with tools); high-risk tools suspend/resume via HITL (when checkpoint_store is
attached), and the Reflection main loop propagates the Interrupt upward via the base `_absorb_child`. Without tool_registry, critique
degrades to pure self-judgment. The framework provides the mechanism; the app decides which tools to register for critique.
"""

import re
from typing import TYPE_CHECKING, Optional

from agentmaker.agents.agent import Agent
from agentmaker.agents.base import BaseAgent
from agentmaker.core import LLMClient, TokenCounter, count_tokens
from agentmaker.runtime.execution import ExecutionState
from agentmaker.runtime.harness import HarnessConfig
from agentmaker.tools import ToolRegistry

if TYPE_CHECKING:
    from agentmaker.tools import ConfirmCallback


class ReflectionAgent(BaseAgent):
    """Reflection orchestration recipe: draft -> (reflect -> refine) xN, until the critique signals "best reached" or max_turns is hit. Optionally verifies with tools in the critique step."""

    def __init__(self, name: str, llm: LLMClient, system_prompt: Optional[str] = None, *,
                 max_turns: int = 3, tracer=None, hooks=None, run_policy=None, session_store=None, scope=None,
                 input_guardrails=None, output_guardrails=None,
                 tool_registry: Optional[ToolRegistry] = None, confirm: "Optional[ConfirmCallback]" = None, permissions=None,
                 checkpoint_store=None, tool_retriever=None, context_builder=None, sources=None, reducer=None,
                 window_budget=None, token_counter: TokenCounter = count_tokens, prompts=None):
        """
        Args:
            max_turns: Maximum number of "reflect-refine" rounds (must be a positive integer; the same name across Agent / PlanAgent / AgentSpec denotes the loop cap).
            tool_registry: Optional tool table; if passed, the critique step can call tools to verify facts / arithmetic. Without it, critique is pure self-reflection.
            confirm / permissions / checkpoint_store: Passed through to the critique-executor; with a checkpoint_store attached, critique's
                high-risk tools suspend/resume via HITL (the Reflection main loop propagates the Interrupt upward).
            tool_retriever: Passed through to the critique-executor (selects a relevant subset when critique has many tools).
            context_builder + sources: memory/RAG injection; splices the retrieval block into the prompt in the draft / refine steps (the generation step benefits most).
            reducer / window_budget: Window-governance knobs, given to Reflection's own Harness (when the reflection trajectory overflows the
                window it is trimmed against the window budget, keeping the latest answer plus the critique's key points) and also passed
                through to the critique-executor (whose verification trajectory is likewise bound).
            Other parameters are the same as BaseAgent.
        """
        # Cross-cutting knobs for Reflection's own harness (confirm/permissions/tool_retriever go only to the critic, not into its own harness).
        cfg = HarnessConfig(tracer=tracer, context_builder=context_builder, sources=sources,
                            reducer=reducer, window_budget=window_budget, token_counter=token_counter)
        super().__init__(name, llm, system_prompt, session_store=session_store, scope=scope, hooks=hooks,
                         run_policy=run_policy, checkpoint_store=checkpoint_store, harness_config=cfg,
                         input_guardrails=input_guardrails, output_guardrails=output_guardrails, prompts=prompts)
        if max_turns <= 0:
            raise ValueError(f"max_turns must be a positive integer, got {max_turns}")
        self.max_turns = max_turns
        # draft/refine use this harness (with context_builder/sources to inject the memory/RAG block; it deliberately has no tools:
        # the generation step does not call tools, tools are used only in critique for "verification", so _make_harness() is not passed a
        # tool_registry and the budget accounting does not count tool schemas either).
        # Assembled via the base _make_harness: _harness_hooks and self.prompts are injected automatically.
        self.harness = self._make_harness()
        # critique-executor: a single-loop Agent with tools runs the "critique + tool verification" step, reusing its fc + HITL loop.
        # Checkpoints are stored under a derived child scope (the agent dimension gets "::reflect_crit" appended) to keep them distinct from
        # Reflection's own; max_turns=2 bounds verification (at most two rounds), ensuring "critique only verifies, it does not drive open-ended
        # solving". Without a tool_registry it degrades to a single-shot LLM judgment.
        # as_child=True: run-level hooks only reach the critic's inner Harness; the run level is triggered once by Reflection, and the critic
        # does not clear its own checkpoints.
        # on_pending="discard": when the critic meets a leftover checkpoint (parent cleanup missed / a crash) it discards and restarts rather
        # than raising SessionError and wedging the scope (the parent meta["awaiting"] is the single source of truth for "awaiting", so when the
        # critic starts a new critique step any leftover child checkpoint must be stale and can be safely discarded).
        self._critic = Agent(
            f"{name}-critic", llm,
            system_prompt=self.prompts.text("reflection.critic_persona"),
            tool_registry=tool_registry, confirm=confirm, tracer=tracer, permissions=permissions,
            checkpoint_store=checkpoint_store, scope=self._derive_scope(self.scope, "reflect_crit"),
            tool_retriever=tool_retriever, max_turns=2, reducer=reducer, window_budget=window_budget,
            token_counter=token_counter, prompts=self.prompts, hooks=self._harness_hooks, as_child=True,
            on_pending="discard",
        )

    def _child_agents(self):
        """The internal critique-executor (checkpoints stored under the derived child scope "::reflect_crit"); used for cascaded cleanup by clear_checkpoint."""
        return [(self._critic, "reflect_crit")]

    async def _arun(self, input_text: str, *, scope, verbose: bool = False, **kwargs):
        """Build the initial ExecutionState and hand it to _adrive to drive "draft -> reflect -> refine". Guardrails and history persistence are handled in the base arun.

        Returns:
            str (the final answer) or an Interrupt (a critique high-risk tool suspended awaiting approval).
        """
        return await self._adrive(ExecutionState(messages=[], input_text=input_text), scope=scope, verbose=verbose, **kwargs)

    async def _adrive(self, state, *, scope, verbose: bool = False, **kwargs):
        """Drive "draft -> (reflect -> refine) xN" from an ExecutionState; can resume from an HITL suspension or a crash.

        The trajectory meta["trajectory"] is [{"kind","text"}]; the next step's shape is inferred from the last trajectory item (last item is
        draft/refine -> critique next; is critique and best not yet reached -> refine next). Critique is delegated to the critic-executor
        (can suspend when it has tools, propagated upward via the base _absorb_child).
        """
        meta = state.meta
        traj = meta.setdefault("trajectory", [])
        crit_scope = self._derive_scope(scope, "reflect_crit")
        block = await self.harness.acontext_block(state.input_text, scope)   # memory/RAG block (injected in draft/refine; truly async, multi-source concurrency).
        if meta.get("awaiting"):                                      # Resume: first resume the suspended critique.
            result = await self._critic.aresume(self._child_decision(state), scope=crit_scope, **kwargs)
            susp = await self._absorb_critique(result, state, crit_scope, verbose, scope)
            if susp is not None:
                return susp
        if not traj:                                                 # First time: draft.
            answer = await self._achat(self._initial_prompt(state.input_text, block), **kwargs)
            traj.append({"kind": "draft", "text": answer})
            self._show("Draft", answer, verbose)
            await self._checkpoint(state, scope)
        while not self._passed(traj) and meta.get("rounds", 0) < self.max_turns:
            if self._next_is_critique(traj):
                meta["trajectory"] = traj = await self.harness.areduce("reflection", traj)   # Overflow protection: keep the latest answer plus the critique's key points.
                self._critic.clear_history(scope=crit_scope)          # clear_history is the sync public surface.
                result = await self._critic.arun(self._reflect_prompt(state.input_text, traj), scope=crit_scope, **kwargs)
                susp = await self._absorb_critique(result, state, crit_scope, verbose, scope)
                if susp is not None:
                    return susp
            else:                                                    # Last step was a critique and best not yet reached -> refine.
                answer = await self._achat(self._refine_prompt(state.input_text, traj, block), **kwargs)
                traj.append({"kind": "refine", "text": answer})
                meta["rounds"] = meta.get("rounds", 0) + 1   # One round = critique + refine; counted after refine so every non-passing critique is applied (independent of the trajectory, which may be collapsed by trimming).
                self._show("Refined", answer, verbose)
                await self._checkpoint(state, scope)
        return self._latest_answer(traj)

    async def _absorb_critique(self, result, state, crit_scope, verbose, scope):
        """Absorb the critique-executor's result (delegates to the base _absorb_child; on_complete records it into the trajectory and prints when verbose)."""
        def on_complete(r):
            state.meta["trajectory"].append({"kind": "critique", "text": r})
            self._show("Reflection", r, verbose)

        return await self._absorb_child(result, state, scope, child=self._critic,
                                        child_scope=crit_scope, on_complete=on_complete)

    @staticmethod
    def _next_is_critique(traj) -> bool:
        """Last trajectory item is an answer (draft/refine) -> next step is critique; is critique -> next step is refine."""
        return traj[-1]["kind"] in ("draft", "refine")

    def _passed(self, traj) -> bool:
        """The most recent critique carries the pass signal -> best reached, can finish (the signal comes from the prompt registry key reflection.pass_signal).

        Uses a word-boundary regex rather than a substring `in`: otherwise the pass signal appearing as part of a longer word / phrase
        (embedded in a negation, or glued to another token) would be misjudged as a pass, prematurely wrapping up an answer that did not actually pass.
        """
        crits = [e for e in traj if e["kind"] == "critique"]
        if not crits:
            return False
        signal = self.prompts.text("reflection.pass_signal")
        return re.search(rf"\b{re.escape(signal)}\b", crits[-1]["text"]) is not None

    @staticmethod
    def _latest_answer(traj) -> str:
        """The latest answer (the last draft/refine)."""
        answers = [e["text"] for e in traj if e["kind"] in ("draft", "refine")]
        return answers[-1] if answers else "(no answer generated)"

    def _show(self, label: str, text: str, verbose: bool) -> None:
        """Print one trajectory entry when verbose."""
        if verbose:
            print(f"🔧[{label}]\n{text}\n")

    async def _achat(self, prompt: str, **kwargs) -> str:
        """Call the LLM once with a single user message (used by draft/refine, no tools)."""
        return (await self.harness.acall_llm([{"role": "user", "content": prompt}], **kwargs)).content

    def _initial_prompt(self, task: str, block: str = "") -> str:
        """Draft prompt (from the prompt registry key reflection.initial); prepended with the memory/RAG block when present."""
        base = self.system_prompt or self.prompts.text("reflection.assistant_persona")
        head = f"{block}\n\n" if block else ""
        return self.prompts.render("reflection.initial", head=head, base=base, task=task)

    def _reflect_prompt(self, task: str, traj: list) -> str:
        """Reflect prompt (from the prompt registry key reflection.reflect): critique the latest answer along evaluation dimensions, prompting tool verification for factual/numeric claims."""
        return self.prompts.render("reflection.reflect", task=task, trajectory=self._format_trajectory(traj),
                                   pass_signal=self.prompts.text("reflection.pass_signal"))

    def _refine_prompt(self, task: str, traj: list, block: str = "") -> str:
        """Refine prompt (from the prompt registry key reflection.refine); prepended with the memory/RAG block when present."""
        head = f"{block}\n\n" if block else ""
        return self.prompts.render("reflection.refine", head=head, task=task,
                                   trajectory=self._format_trajectory(traj))

    def _format_trajectory(self, traj: list) -> str:
        """Render the trajectory into coherent labeled text (labels and wrapping format from the prompt registry keys reflection.label.* / reflection.trajectory_item)."""
        return "\n\n".join(self.prompts.render("reflection.trajectory_item",
                                               label=self.prompts.text("reflection.label." + e["kind"]),
                                               text=e["text"]) for e in traj)


