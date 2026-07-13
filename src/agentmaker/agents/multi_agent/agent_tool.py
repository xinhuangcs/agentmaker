"""agentmaker.agents.multi_agent.agent_tool: wrap an Agent as a Tool (the core of agents-as-tools / orchestrator-worker).

The industry-standard pattern for multi-agent collaboration is orchestrator-worker: a main (orchestrator)
agent delegates subtasks to specialist agents, collects their results, synthesizes an answer, and always
retains control of the conversation. The cleanest way to implement it is to adapt "an agent" into "a tool",
so the main agent delegates subtasks just like calling an ordinary tool. This mirrors OpenAI's `agent.as_tool`
and Anthropic spawning a subagent via the `Task` tool (both converge on "sub-agent = a tool").

It reuses everything already in place: the main agent is any existing strategy (single-loop Agent / PlanAgent...)
with an AgentTool registered; the sub-agent carries its own independent history / tools / memory (naturally
isolated, matching Anthropic's "each subagent has its own context"); delegation goes through the existing
Tool / Registry; native async is wired up, so the main agent can call multiple sub-agents concurrently.
"""

from typing import TYPE_CHECKING, Optional

from ..base import BaseAgent
from ...core.aio import run_sync
from ...runtime.execution.run_context import current_scope
from ...prompts import DEFAULT_PROMPTS
from ...tools.base import Tool, ToolParameter
from ...tools.response import ToolResponse

if TYPE_CHECKING:
    from ...retrieval.scope import Scope


class AgentTool(Tool):
    """Adapt an agent into a Tool: the main agent calls it to delegate one subtask and gets the result back (the main agent keeps control)."""

    def __init__(self, agent: BaseAgent, *, name: Optional[str] = None, description: Optional[str] = None,
                 scope: "Optional[Scope]" = None, prompts=None):
        """
        Args:
            agent: The wrapped sub-agent (any strategy).
            name: Tool name, defaults to agent.name; set it explicitly to disambiguate sub-agents that share a name.
            description: Tool description telling the orchestrating LLM what this sub-agent is good at and when to delegate to it; defaults to a value generated from agent.name.
            scope: The session ownership used when delegating to the sub-agent (advanced override: pin the sub-agent
                to a fixed scope). Defaults to None, in which case the parent agent's current run scope is picked up
                automatically (propagated through the run context, see _effective_scope), so that even when "one
                AgentTool instance serves multiple parent sessions" the sub-agent's history / memory stays isolated
                per parent session and never bleeds across them; if the parent has no run context (rare, calling the
                tool bare) it falls back to the sub-agent's own default scope. Pass this explicitly only when you
                genuinely need to pin the sub-agent to a specific scope.
        """
        self.prompts = prompts or DEFAULT_PROMPTS
        super().__init__(
            name=name or agent.name,
            description=description or self.prompts.render("tool.desc.agent", agent_name=agent.name),
        )
        self._agent = agent
        self._scope = scope

    def get_parameters(self) -> list[ToolParameter]:
        """Declare parameters: a single task string (the self-contained subtask handed to the sub-agent)."""
        return [ToolParameter("task", "string", self.prompts.text("tool.param.agent.task"))]

    def run(self, parameters: dict) -> ToolResponse:
        """Synchronous delegation: hand the task to the sub-agent, run it, and return its reply (normalized into a ToolResponse via _textualize)."""
        scope = self._effective_scope()
        result = self._agent.run(parameters.get("task", ""), scope=scope)
        if result.interrupted:
            run_sync(self._agent.clear_checkpoint(scope))   # See _textualize: after absorbing a suspend, clear the sub checkpoint so re-delegating on the same scope does not deadlock.
        return self._textualize(result)

    async def arun(self, parameters: dict) -> ToolResponse:
        """Native async delegation: await the sub-agent's arun (a truly async strategy runs async, otherwise its arun auto-dispatches to a thread pool)."""
        scope = self._effective_scope()
        result = await self._agent.arun(parameters.get("task", ""), scope=scope)
        if result.interrupted:
            await self._agent.clear_checkpoint(scope)
        return self._textualize(result)

    def _effective_scope(self):
        """The scope used when delegating to the sub-agent: an explicit scope passed at construction wins (advanced
        override, pinned); otherwise take the parent agent's current run scope (current_scope(), automatically isolated
        per parent session, preventing the sub-agent's history from bleeding when the instance is shared across
        sessions); if neither, the sub-agent's own default."""
        return self._scope if self._scope is not None else current_scope()

    def _textualize(self, result) -> ToolResponse:
        """Normalize the sub-agent's RunResult into a ToolResponse the orchestrating LLM can read (guaranteeing text is always a string).

        - Sub-agent HITL suspend (result.interrupted): in the agents-as-tools delegation context, the sub-agent's
          suspend cannot propagate back up through the Tool interface for the orchestrator to resume (unlike Plan,
          which has a nested-suspend mechanism), so it is turned into an error result telling the orchestrating LLM
          that "this subtask needs human approval and cannot be completed within delegation", letting it reroute:
          this neither silently swallows the approval nor lets a non-string end up in text and crash the flow. On
          suspend the sub-agent has already persisted via `_suspend`, and the caller (run / arun) first calls
          `clear_checkpoint` to clear it; otherwise re-delegating on the same scope would hit the sub-agent's
          `_guard_pending` and the sub-agent would be permanently unusable under that scope.
        - Completed state: `str(result)` (RunResult.__str__ on a completed result is equivalent to
          `_output_text(final_output)`, same semantics).
        """
        if result.interrupted:
            return ToolResponse.error(
                f"Sub-agent \"{self._agent.name}\" hit a high-risk action requiring human approval, which cannot be "
                "suspended and resumed in an agents-as-tools delegation scenario; please delegate a subtask without "
                "high-risk tools, or let that action be confirmed directly with the user in the main flow.")
        return ToolResponse.ok(str(result))


