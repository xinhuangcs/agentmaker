"""agentmaker.agents.multi_agent: the multi-agent orchestration layer.

Composes multiple single agents (Simple / ReAct / Reflection / PlanSolve) to collaborate, sitting on top of
the "single-agent reasoning layer". Currently provides:
    - AgentTool: wraps an agent as a Tool (agents-as-tools / orchestrator-worker): the main agent delegates
      subtasks, collects results, and keeps control. Mirrors OpenAI's agent.as_tool and Anthropic's Task tool.
"""

from .agent_tool import AgentTool

__all__ = ["AgentTool"]
