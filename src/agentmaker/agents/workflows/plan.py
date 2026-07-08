"""agentmaker.agents.workflows.plan: PlanAgent, the Plan-and-Solve orchestration recipe (plan before acting).

Three stages, with the order fixed in code: first the model breaks a complex problem into an ordered plan of
sub-tasks (planning, via structured output), then each sub-task is executed step by step (delegated to an
internal single-loop Agent), and finally the per-step results are synthesized into a final answer. This is
the opposite of the single-loop Agent's "take one step and reassess": Plan works out the whole plan first and
then executes, which suits long-range planning and multi-step tasks that need strong goal consistency.

Relationship to tools: each step's execution is delegated to the internal `Agent`, so passing tool_registry
lets the execution stage call tools; without it, every step is pure reasoning. HITL: if a step's executor hits
a high-risk tool it suspends, and the Interrupt propagates upward via the base `_absorb_child`; resume
restores that step and continues.
"""

import ast
import re
from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field

from agentmaker.agents.agent import Agent
from agentmaker.agents.base import BaseAgent
from agentmaker.core import LLMClient, TokenCounter, count_tokens
from agentmaker.core.exceptions import LLMResponseError
from agentmaker.runtime.execution import ExecutionState
from agentmaker.runtime.harness import HarnessConfig
from agentmaker.tools import ToolRegistry

if TYPE_CHECKING:
    from agentmaker.tools import ConfirmCallback


class PlanSteps(BaseModel):
    """Structured output of the planning stage: an ordered list of sub-task steps."""
    steps: List[str] = Field(default_factory=list, description="Ordered sub-task steps that can be executed one by one")


class PlanAgent(BaseAgent):
    """Plan-and-Solve orchestration recipe: plan (structured step breakdown) -> delegate step-by-step execution to an internal Agent -> synthesize the answer."""

    def __init__(self, name: str, llm: LLMClient, system_prompt: Optional[str] = None, *,
                 tool_registry: Optional[ToolRegistry] = None,
                 max_turns: int = 3, confirm: "Optional[ConfirmCallback]" = None, tracer=None,
                 permissions=None, hooks=None, run_policy=None, session_store=None, scope=None, checkpoint_store=None,
                 input_guardrails=None, output_guardrails=None,
                 tool_retriever=None, context_builder=None, sources=None, reducer=None, window_budget=None,
                 token_counter: TokenCounter = count_tokens, prompts=None):
        """
        Args:
            system_prompt: Optional extra persona for the planning stage (third positional arg, aligned with Agent / ReflectionAgent).
            tool_registry: If passed, every execution step can call tools; without it, execution is pure reasoning (keyword-only).
            max_turns: Upper bound on the tool-loop turns for each sub-step executor (the internal single-loop Agent), default 3 (not the number of plan steps).
            tool_retriever / context_builder / sources: Context engineering, passed through to each step's executor; the Plan's own
                planning and synthesis calls also inject a memory/RAG block.
            confirm / permissions / checkpoint_store: Passed through to the executor; with a checkpoint_store attached, high-risk tools
                during a step suspend/resume via HITL (the Plan main loop catches and propagates the Interrupt upward).
            reducer / window_budget: Window-governance knobs, given to Plan's own Harness (when history overflows the window it is trimmed
                against the window budget) and also passed through to each step's executor (whose tool-trajectory trimming is likewise bound by these two knobs).
            Other parameters are the same as BaseAgent.
        """
        # Cross-cutting knobs for Plan's own harness (no tool_retriever / compactor, those belong to the executor).
        cfg = HarnessConfig(tracer=tracer, confirm=confirm, permissions=permissions,
                            context_builder=context_builder, sources=sources, reducer=reducer,
                            window_budget=window_budget, token_counter=token_counter)
        super().__init__(name, llm, system_prompt, session_store=session_store, scope=scope,
                         checkpoint_store=checkpoint_store, hooks=hooks, run_policy=run_policy,
                         harness_config=cfg, input_guardrails=input_guardrails, output_guardrails=output_guardrails,
                         prompts=prompts)
        # Passing tool_registry lets the window budget count tool schemas into the fixed cost. Plan's own plan/synthesize calls do not
        # actually emit tools (only the sub-executor does), so this is a deliberate conservative over-reservation (the trajectory/retrieval
        # block budget ends up slightly smaller, never over-issued). Do not mistake it for a bug and turn it into double counting (see window_budget.md).
        # Assembled via the base _make_harness: _harness_hooks and self.prompts are injected automatically.
        self.harness = self._make_harness(tool_registry=tool_registry)
        # Executor: a single-loop Agent runs each sub-step (native function-calling). Native fc plus an attached checkpoint_store is what
        # enables HITL suspension and per-step persistence; the executor's checkpoints are stored under a derived child scope
        # (the agent dimension gets "::plan_exec" appended) to keep them distinct from Plan's own.
        # as_child=True: run-level hooks only reach the executor's inner Harness (observing each model/tool-level step); the run level is
        # triggered once by Plan, not repeated, and the executor does not clear its own checkpoints (Plan clears them after committing
        # progress in _absorb_child). Tool-RAG / memory-RAG are passed through to the executor.
        # on_pending="discard": when the executor meets a leftover checkpoint (parent cleanup missed / a crash) it discards and restarts
        # rather than raising SessionError and wedging the scope. The parent state is the single source of truth for "awaiting" (Plan uses
        # meta["awaiting"] to decide whether to arun a new step or aresume a suspended one), so when arun-ing a new step any leftover child
        # checkpoint must be stale and can be safely discarded.
        self._executor = Agent(
            f"{name}-executor", llm,
            system_prompt=self.prompts.text("plan.executor_persona"),
            tool_registry=tool_registry, max_turns=max_turns, confirm=confirm, tracer=tracer, permissions=permissions,
            checkpoint_store=checkpoint_store, scope=self._derive_scope(self.scope, "plan_exec"),
            tool_retriever=tool_retriever, context_builder=context_builder, sources=sources,
            reducer=reducer, window_budget=window_budget, token_counter=token_counter, prompts=self.prompts,
            hooks=self._harness_hooks, as_child=True, on_pending="discard",
        )

    def _child_agents(self):
        """The internal executor (checkpoints stored under the derived child scope "::plan_exec"); used for cascaded cleanup by clear_checkpoint."""
        return [(self._executor, "plan_exec")]

    async def _arun(self, input_text: str, *, scope, verbose: bool = False, **kwargs):
        """Build the initial ExecutionState and hand it to _adrive to drive "plan -> step-by-step execution -> synthesize". Guardrails and history persistence are handled in the base arun.

        Returns:
            str (the synthesized final answer) or an Interrupt (a step's executor suspended on a high-risk action awaiting approval).
        """
        return await self._adrive(ExecutionState(messages=[], input_text=input_text), scope=scope, verbose=verbose, **kwargs)

    async def _adrive(self, state, *, scope, verbose: bool = False, **kwargs):
        """Drive Plan from an ExecutionState: first plan -> delegate step by step to the executor -> synthesize; can resume from mid-run.

        Plan state lives in state.meta (plan / history / cursor / awaiting). Nested suspension: when a step's executor suspends, the base
        `_absorb_child` records awaiting, persists Plan's progress, and propagates the parent scope's Interrupt upward; on resume (the base
        has already injected the decision into state.decisions) the executor first resumes that step, then the remaining steps continue.
        """
        meta = state.meta
        exec_scope = self._derive_scope(scope, "plan_exec")
        if "plan" not in meta:                                  # First time: plan.
            plan = await self._aplan(state.input_text, scope=scope, **kwargs)
            meta.update(plan=plan, history=[], cursor=0, awaiting=False)
            self._print_plan(plan, verbose)
            await self._checkpoint(state, scope)                  # Persist right after planning: crash recovery does not re-plan (the plan stays stable and consistent with already-executed steps).
        plan = meta["plan"]
        if meta.get("awaiting"):                                # Resume: the executor first resumes the suspended step.
            result = await self._executor.aresume(self._child_decision(state), scope=exec_scope, **kwargs)
            susp = await self._absorb_step(result, state, exec_scope, verbose, scope)
            if susp is not None:
                return susp
        while meta["cursor"] < len(plan):                       # Step-by-step execution.
            self._executor.clear_history(scope=exec_scope)      # Each step is independent; cross-step context is passed explicitly via history (clear_history is the sync public surface).
            meta["history"] = await self.harness.areduce("plan", meta["history"])   # Overflow protection: on window overflow, trim earlier step results.
            prompt = self._executor_prompt(plan[meta["cursor"]], state.input_text, plan, meta["history"])
            result = await self._executor.arun(prompt, scope=exec_scope, **kwargs)
            susp = await self._absorb_step(result, state, exec_scope, verbose, scope)
            if susp is not None:
                return susp
        return await self._asynthesize(state.input_text, meta["history"], scope=scope, **kwargs)

    async def _absorb_step(self, result, state, exec_scope, verbose, scope):
        """Absorb one executor step's result (delegates to the base _absorb_child; on_complete records history, advances cursor by 1, and prints when verbose)."""
        meta = state.meta

        def on_complete(r):
            cursor = meta["cursor"]
            meta["history"].append(f"Step {cursor + 1}: {meta['plan'][cursor]}\nResult: {r}")
            if verbose:
                print(f"🔧[Step {cursor + 1}/{len(meta['plan'])}] {meta['plan'][cursor]}\n🔧  Result: {r}\n")
            meta["cursor"] = cursor + 1

        return await self._absorb_child(result, state, scope, child=self._executor,
                                        child_scope=exec_scope, on_complete=on_complete)

    @staticmethod
    def _print_plan(plan, verbose):
        """Print the plan (when verbose)."""
        if verbose:
            print("🔧[Plan]")
            for i, step in enumerate(plan, 1):
                print(f"🔧  {i}. {step}")
            print()

    async def _aplan(self, question: str, *, scope=None, **kwargs) -> List[str]:
        """Planning stage: have the model break the question into an ordered list of sub-tasks (structured output PlanSteps).

        Three-level fallback: (1) structured output succeeds -> take steps; (2) structured output fails (LLMResponseError, i.e. a
        parse/validation failure) -> fall back to plain text plus line-by-line degraded parsing (preserving a multi-step plan the model
        listed in natural language / numbered form rather than collapsing to a single-step direct answer); (3) still empty -> [question]
        (treat the original question as a single step to execute). Only LLMResponseError is caught; LLMRequestError (rate limit / auth /
        network must propagate and should not be retried as a fallback) and RunLimitExceeded / RunCancelled (governance signals) are not swallowed.
        """
        block = await self.harness.acontext_block(question, scope)
        prompt = self._with_context(self._planner_prompt(question), block)
        try:
            steps = (await self.harness.astructured([{"role": "user", "content": prompt}], PlanSteps, **kwargs)).steps
            steps = [str(s).strip() for s in steps if str(s).strip()]
            if steps:
                return steps
        except LLMResponseError:
            resp = await self.harness.acall_llm([{"role": "user", "content": prompt}], **kwargs)   # Line-by-line degraded salvage.
            steps = self._parse_plan(resp.content)
            if steps:
                return steps
        return [question]

    @staticmethod
    def _with_context(prompt: str, block: str) -> str:
        """In the planning / synthesis stage, prepend a memory/RAG context block to the prompt (return it unchanged if block is empty).

        The block is computed by the caller ahead of time via harness.acontext_block (truly async, multi-source gather concurrency).
        """
        return f"{block}\n\n{prompt}" if block else prompt

    async def _asynthesize(self, question: str, history: List[str], *, scope=None, **kwargs) -> str:
        """Synthesis stage: assemble the final answer from the per-step results (first apply overflow trimming to history, then inject the memory/RAG block)."""
        history = await self.harness.areduce("plan", history)
        block = await self.harness.acontext_block(question, scope)
        prompt = self._with_context(self._synthesize_prompt(question, history), block)
        resp = await self.harness.acall_llm([{"role": "user", "content": prompt}], **kwargs)
        return resp.content

    @staticmethod
    def _parse_plan(response: str) -> List[str]:
        """Parse plan text into a step list (line-by-line degraded salvage when structured output fails): prefer parsing as a Python list (including a ``` code block), falling back to line splitting on failure.

        If literal_eval yields a list, it is trusted (including an empty list `[]`: return an empty list directly, do not fall through and
        treat the literal "[]" as a single step); an empty plan is left for `_aplan` to fall back on. The line-by-line path is only taken when
        literal_eval fails or its result is not a list.
        """
        text = response.strip()
        match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        try:
            steps = ast.literal_eval(text)
            if isinstance(steps, list):
                return [str(s).strip() for s in steps if str(s).strip()]
        except (ValueError, SyntaxError):
            pass
        # Fallback: literal_eval failed or its result is not a list; split line by line, stripping numbering / bullet prefixes.
        lines = [re.sub(r"^[\s\-*0-9.、)）]+", "", ln).strip() for ln in response.splitlines()]
        return [ln for ln in lines if ln]

    def _planner_prompt(self, question: str) -> str:
        """Planner prompt: ask the model to break the question into ordered sub-tasks (from the prompt registry key plan.planner)."""
        base = self.system_prompt or self.prompts.text("plan.planner_persona")
        return self.prompts.render("plan.planner", base=base, question=question)

    def _executor_prompt(self, step: str, question: str, plan: List[str], history: List[str]) -> str:
        """Executor prompt: have the executor focus solely on completing the current step (from the prompt registry key plan.executor)."""
        plan_text = chr(10).join(f"{i}. {s}" for i, s in enumerate(plan, 1))
        history_text = chr(10).join(history) or self.prompts.text("plan.history_empty")
        return self.prompts.render("plan.executor", question=question, plan_text=plan_text,
                                   history_text=history_text, step=step)

    def _synthesize_prompt(self, question: str, history: List[str]) -> str:
        """Synthesis prompt: give the final answer based on all step results (from the prompt registry key plan.synthesize)."""
        return self.prompts.render("plan.synthesize", question=question, history_text=chr(10).join(history))


