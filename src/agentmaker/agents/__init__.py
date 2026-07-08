"""agentmaker.agents: Agent base + execution primitive + the strategies.

`BaseAgent` (base.py): the unified run / resume template (guardrails + persistence + HITL resume) plus the
three sub-agent delegation methods.
Single-loop `Agent` (agent.py): the one "model calls tools in a loop" execution primitive (native
function-calling); the spec menu's "chat" maps to it and "react" is its preset.
Orchestration recipes (workflows subpackage): `PlanAgent` / `ReflectionAgent`, strategies whose control
flow is in code, driving an internal Agent for each step's work.
Multi-agent orchestration (multi_agent subpackage): `AgentTool`, wrapping one Agent as a Tool
(orchestrator-worker).

`Agent` is the single-loop class (agent.py); to write a new strategy, inherit `BaseAgent` (base.py).
"""

from .base import BaseAgent
from .agent import Agent
from .result import RunResult, RunStatus, RunUsage
from .workflows import PlanAgent, ReflectionAgent
from .spec import AgentSpec, build_agent
from .multi_agent import AgentTool

__all__ = ["Agent", "BaseAgent", "PlanAgent", "ReflectionAgent", "AgentSpec", "build_agent", "AgentTool",
           "RunResult", "RunStatus", "RunUsage"]
