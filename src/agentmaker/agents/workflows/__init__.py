"""agentmaker.agents.workflows: orchestration recipes (strategies above the agent, with control flow held in code).

In contrast to the single-loop `Agent` (control flow held by the model, an open tool-calling loop), the
strategies here have their stage order fixed in code: the model is only invoked once per stage to do one
piece of work. This maps to the workflow side of Anthropic's "Building effective agents" (plan-execute is
roughly orchestrator-workers, reflection is roughly evaluator-optimizer).

Both built-in recipes subclass `BaseAgent` and use its three sub-agent delegation methods (_derive_scope /
_child_decision / _absorb_child plus as_child) to attach an internal single-loop `Agent` that does each
step's work, so run/resume/HITL/checkpointing/guardrails/the sync facade all come for free.

To add a new recipe: subclass BaseAgent, implement `_arun`/`_adrive`, and attach a sub-Agent via the three
delegation methods (about 150 lines each; this package serves as the reference example).
"""

from .plan import PlanAgent
from .reflection import ReflectionAgent

__all__ = ["PlanAgent", "ReflectionAgent"]
