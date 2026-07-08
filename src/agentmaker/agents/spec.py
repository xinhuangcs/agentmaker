"""agentmaker.agents.spec: declarative agent configuration (framework-layer config) plus factory.

`AgentSpec` aggregates every configurable point of an agent into one dataclass; `build_agent`
constructs the matching strategy based on `strategy`. The declarative layer is a convenience on top
of imperative construction (`Agent(...)` and friends); both coexist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Union

from ..core import LLMClient, TokenCounter, count_tokens
from ..prompts import DEFAULT_PROMPTS
from ..tools import ToolRegistry
from .agent import Agent as UnifiedAgent
from .workflows import PlanAgent, ReflectionAgent

if TYPE_CHECKING:                       # type hints only (works with from __future__ annotations; not imported at runtime)
    from ..context import ContextBuilder, ContextSource, HistoryCompactor, ReducerConfig, WindowBudgetConfig
    from ..prompts import PromptRegistry
    from ..runtime.execution import CheckpointStore, RunPolicy
    from .base import BaseAgent
    from ..runtime.guardrails import Guardrail
    from ..runtime.hooks import Hook
    from ..runtime.observability import Tracer
    from ..retrieval import Scope
    from ..runtime.sessions import SessionStore
    from ..tools import ConfirmCallback
    from ..tools.permissions import ToolPermissions
    from ..tools.tool_retriever import ToolRetriever


@dataclass
class AgentSpec:
    """Declaratively aggregates every configurable point of an agent; `strategy` picks the paradigm, `build_agent` constructs from it.

    The fields are a superset across strategies: common fields (including the memory/RAG injection
    `context_builder`+`sources` and the Tool-RAG `tool_retriever`) apply to the relevant strategies,
    while strategy-specific fields (`compactor` is chat-only, the single-loop Agent) only take effect
    on strategies that support them. Setting one on an unsupported strategy is rejected by
    `build_agent` (fail loud).
    """
    name: str
    strategy: Literal["chat", "react", "reflection", "plan"] = "chat"   # picks the strategy; build_agent constructs from it (a typo turns mypy red on the spot)
    model: Optional[Union[str, LLMClient]] = None           # str -> "provider:model" (e.g. "deepseek:deepseek-v4-flash") or a bare provider name / LLMClient instance -> used directly / None -> default LLMClient()
    instructions: Optional[str] = None                      # -> system_prompt
    tools: Optional[Union[list, ToolRegistry]] = None       # list[Tool] -> build a registry / registry -> used as-is
    scope: Optional[Scope] = None
    session_store: Optional[SessionStore] = None
    checkpoint_store: Optional[CheckpointStore] = None
    tracer: Optional[Tracer] = None
    input_guardrails: Optional[list[Guardrail]] = None
    output_guardrails: Optional[list[Guardrail]] = None
    hooks: Optional[list[Hook]] = None                      # lifecycle hooks (observe-only; common to all strategies)
    run_policy: Optional[RunPolicy] = None                  # run governance (limits / cancellation; common to all strategies)
    confirm: Optional[ConfirmCallback] = None
    permissions: Optional[ToolPermissions] = None           # tool allow/deny list (strategies with tools: chat / react / plan / reflection)
    max_turns: Optional[int] = None                         # unified turn limit -> each strategy's turn field (None uses the strategy default)
    # -- chat (single-loop Agent) specific (other strategies do not support it; setting it fails loud) --
    compactor: Optional[HistoryCompactor] = None            # cross-session history compaction; other strategies trim their own trajectory automatically by window via the built-in reducer, so this parameter is not needed
    # -- context engineering (common to strategies with tools / retrieval) --
    tool_retriever: Optional[ToolRetriever] = None          # Tool-RAG: chat / react / plan / reflection (selects the relevant subset when tools are present)
    context_builder: Optional[ContextBuilder] = None        # memory/RAG injection: all strategies
    sources: Optional[list[ContextSource]] = None           # used together with context_builder
    reducer: Optional["ReducerConfig"] = None               # knob for how many recent trajectory steps to keep (all strategies via Harness; None uses the default)
    window_budget: Optional["WindowBudgetConfig"] = None     # knob for window-budget accounting (all strategies via Harness; None uses the WindowBudgetConfig() default)
    token_counter: "TokenCounter" = count_tokens            # pluggable token counter (all strategies via Harness for budget estimation / chunking); defaults to count_tokens
    prompts: Optional["PromptRegistry"] = None              # prompt registry (all strategies pass it through the base class to the harness / internal sub-agents); None uses the shared DEFAULT_PROMPTS


# Default values for chat-only fields (_reject uses these to decide "was it set?" -> setting it on another strategy fails loud).
# Contains only the single field _reject actually guards; the rest (context engineering / tools) are now unified across strategies and no longer rejected (see the branches in build_agent).
_FIELD_DEFAULTS = {"compactor": None}


def build_agent(spec: AgentSpec) -> BaseAgent:
    """Construct the matching strategy agent based on `spec.strategy`.

    model: str -> `"provider:model"` (e.g. `"deepseek:deepseek-v4-flash"`, split into
        `LLMClient(provider, model=right_half)`); a bare provider name (no colon, e.g. `"deepseek"`)
        is still treated as a provider (using the .env key + the provider's default model); an
        `LLMClient` instance -> used directly (lets you pin model / key / base_url); None ->
        `LLMClient()` default.
    tools: list[Tool] -> build a new `ToolRegistry` and register them / `ToolRegistry` -> used as-is
        / None -> no tools.
    A field unsupported by the strategy that is set to a non-default value raises `ValueError`
    (avoids the silent "set but had no effect" trap).

    Returns:
        BaseAgent: the constructed strategy instance (single-loop Agent (chat/react) / ReflectionAgent / PlanAgent).
    """
    registry = _as_registry(spec.tools, spec.prompts)
    if spec.strategy == "react" and (registry is None or not registry.list_tools()):  # react must have tools (both None and an empty list are rejected); fail loud before creating the LLMClient
        raise ValueError("strategy='react' requires at least one tool: ReAct's Acting step is calling a tool, and with no tools there is nothing to act on (raised at construction time rather than deferred to runtime)")
    llm = _resolve_llm(spec.model)
    common = dict(system_prompt=spec.instructions, tracer=spec.tracer, session_store=spec.session_store,
                  scope=spec.scope, input_guardrails=spec.input_guardrails, output_guardrails=spec.output_guardrails,
                  hooks=spec.hooks, run_policy=spec.run_policy, token_counter=spec.token_counter, prompts=spec.prompts)
    if spec.strategy == "chat":
        return UnifiedAgent(spec.name, llm, tool_registry=registry,
                            max_turns=_turns(spec.max_turns, 10), confirm=spec.confirm,
                            permissions=spec.permissions, checkpoint_store=spec.checkpoint_store,
                            compactor=spec.compactor, reducer=spec.reducer, tool_retriever=spec.tool_retriever,
                            context_builder=spec.context_builder, sources=spec.sources,
                            window_budget=spec.window_budget, **common)
    if spec.strategy == "react":
        # ReAct is a preset of the single-loop Agent: tools required (fail-loud above) + max_turns default 5 +
        # system injects react.persona/react.style (think before acting; when verbose, the reasoning is visible in content).
        common_react = {**common, "system_prompt": _react_system(spec)}
        return UnifiedAgent(spec.name, llm, tool_registry=registry, max_turns=_turns(spec.max_turns, 5),
                            confirm=spec.confirm, permissions=spec.permissions,
                            checkpoint_store=spec.checkpoint_store, compactor=spec.compactor, reducer=spec.reducer,
                            tool_retriever=spec.tool_retriever, context_builder=spec.context_builder,
                            sources=spec.sources, window_budget=spec.window_budget, **common_react)
    if spec.strategy == "reflection":
        _reject(spec, {"compactor"})   # Reflection trims its own trajectory automatically by window and does not accept a compactor
        return ReflectionAgent(spec.name, llm, max_turns=_turns(spec.max_turns, 3),
                               tool_registry=registry, confirm=spec.confirm, permissions=spec.permissions,
                               checkpoint_store=spec.checkpoint_store, tool_retriever=spec.tool_retriever,
                               context_builder=spec.context_builder, sources=spec.sources,
                               reducer=spec.reducer, window_budget=spec.window_budget, **common)
    if spec.strategy == "plan":
        _reject(spec, {"compactor"})   # Plan trims its own history automatically by window and does not accept a compactor
        return PlanAgent(spec.name, llm, tool_registry=registry, max_turns=_turns(spec.max_turns, 3), confirm=spec.confirm,
                         permissions=spec.permissions, checkpoint_store=spec.checkpoint_store,
                         tool_retriever=spec.tool_retriever, context_builder=spec.context_builder,
                         sources=spec.sources, reducer=spec.reducer,
                         window_budget=spec.window_budget, **common)
    raise ValueError(f"unknown strategy: {spec.strategy!r} (expected chat / react / reflection / plan)")


def _react_system(spec: AgentSpec) -> str:
    """Build the ReAct preset's system prompt: persona (spec.instructions or react.persona) + think-before-acting style (react.style).

    Extracted into a function so callers constructing the single-loop Agent imperatively can reuse
    this preset assembly (no need to copy the two prompts by hand).
    """
    prompts = spec.prompts or DEFAULT_PROMPTS
    base = spec.instructions or prompts.text("react.persona")
    return f"{base}\n\n{prompts.text('react.style')}"


def _as_registry(tools, prompts=None) -> Optional[ToolRegistry]:
    """Normalize tools into a ToolRegistry: a thin wrapper over ToolRegistry.from_tools (a single source of truth shared with Agent(tools=)).

    prompts is passed through to the new registry so its error messages (tool not found / parameter
    validation) match the language of the owning agent.
    """
    return ToolRegistry.from_tools(tools, prompts=prompts)


def _resolve_llm(model) -> LLMClient:
    """Resolve AgentSpec.model into an LLMClient: None -> default / LLMClient -> as-is / str -> split "provider:model" or a bare provider name.

    Follows Pydantic AI's "openai:gpt-4o" convention: with a colon, split into provider:model (an
    empty right half like "deepseek:" falls the model back to the provider default); without a colon,
    treat the string as a bare provider name (backward compatible with the older form). Other types
    fail loud.
    """
    if model is None:
        return LLMClient()
    if isinstance(model, LLMClient):
        return model
    if isinstance(model, str):
        provider, sep, name = model.partition(":")
        return LLMClient(provider, model=name or None) if sep else LLMClient(model)
    raise TypeError(f"AgentSpec.model must be a str ('provider:model' or a bare provider name) / LLMClient / None, got {type(model).__name__}")


def _reject(spec: AgentSpec, fields: set) -> None:
    """A field unsupported by the strategy that is set to a non-default value raises ValueError (fail loud, avoids silent no-op)."""
    bad = sorted(f for f in fields if getattr(spec, f) != _FIELD_DEFAULTS[f])
    if bad:
        raise ValueError(f"strategy={spec.strategy!r} does not support these fields, do not set them: {', '.join(bad)}")


def _turns(max_turns: Optional[int], default: int) -> int:
    """Normalize spec.max_turns into a strategy turn limit: None -> strategy default; otherwise it must be a positive integer.

    Explicitly distinguishes None from a numeric value (not `max_turns or default`: `or` would treat
    0 as unset and silently fall back to the default, while a negative value would slip through
    unchanged and blow the limit on the first turn). A value <= 0 raises immediately, ruling out the
    "set but had no effect / set and immediately over limit" silent trap.
    """
    if max_turns is None:
        return default
    if max_turns <= 0:
        raise ValueError(f"max_turns must be a positive integer (every strategy runs at least one turn), got {max_turns}")
    return max_turns
