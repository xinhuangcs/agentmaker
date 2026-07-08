"""agentmaker.runtime.hooks: lifecycle hooks (observe-only).

Gives the framework a single lifecycle extension point: an app subclasses `Hook`, overrides the events it
cares about, and inserts side-effect code (logging / metrics / auditing / cost / debugging) at the key points
of an agent run (before / after calling the model, before / after executing a tool, guardrail tripwire,
suspend, error, run start / end). Observe only, do not intervene: interception / modification is left to
dedicated layers like Guardrail / Permissions / HITL / compactor. Run-level events are fired by `Agent.run`,
model/tool-level events by `Harness`.
"""

import inspect
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:                       # Type annotations only, not imported at runtime (Hook is a hot-path base class, so no extra loading).
    from ..core.llm_response import LLMResponse
    from ..retrieval.scope import Scope
    from ..tools.response import ToolResponse
    from .hitl import PendingAction


class Hook:
    """Lifecycle hook base class (observe-only). Subclass it, override only the events you care about; the rest default to no-ops.

    All method return values are ignored (pure side effects); an exception raised inside a method propagates
    upward (fail loud). A hook is the app's own code, so wrap risky I/O yourself. For interception / modification
    use Guardrail / Permissions / HITL, not a hook.
    """

    def on_run_start(self, input_text: str, *, scope: "Optional[Scope]" = None):
        """Fires when a run begins (before the input guardrails). input_text is this turn's user input."""

    def before_model(self, messages: "list[dict]"):
        """Fires before each LLM call; messages is the message list about to be sent (also fires for streaming calls)."""

    def after_model(self, response: "LLMResponse"):
        """Fires after each (non-streaming) LLM call; response is an LLMResponse. Streaming has no single response object, so it does not fire."""

    def before_tool(self, name: str, parameters: dict):
        """Fires just before a tool actually executes (already past the permission gate / HITL approval gate; rejected or suspended tools do not fire); name / parameters are the tool name and its arguments."""

    def after_tool(self, name: str, parameters: dict, result: "ToolResponse"):
        """Fires after a tool executes; result is a ToolResponse."""

    def on_guardrail_trip(self, stage: str, message: str):
        """Fires on a guardrail tripwire; stage is "input" / "output", message is the human-readable block explanation."""

    def on_interrupt(self, pendings: "list[PendingAction]", *, scope: "Optional[Scope]" = None):
        """Fires on a HITL suspend; pendings is the list of PendingAction awaiting approval (more than one when a turn has several high-risk actions, or parallel sub-agents each suspend)."""

    def on_error(self, error: Exception):
        """Fires just before a non-guardrail exception propagates upward; error is that exception (guardrail tripwires go through on_guardrail_trip and do not also fire this event)."""

    def on_run_end(self, output, *, scope: "Optional[Scope]" = None):
        """Fires when a run produces its final result normally; output is the final output (suspend / error do not go through here)."""


def fire(hooks, event, *args, **kwargs):
    """Call each hook's event method in turn (observe-only).

    Returns immediately when hooks is empty (the default): the hot path (crossed on every LLM / tool call) keeps
    only a single empty-list check.

    Args:
        hooks: The list of Hooks.
        event: The event method name (such as "before_model").
        *args / **kwargs: Passed through to that event method.
    """
    if not hooks:
        return
    for hook in hooks:
        getattr(hook, event)(*args, **kwargs)


async def afire(hooks, event, *args, **kwargs):
    """The async version of fire (called by the framework's async execution layer): call each hook's event method, and if it returns an awaitable, await it.

    Lets an app write event methods as `async def` (such as asynchronously persisting an audit log); synchronous
    methods work as-is with no changes. Also short-circuits when hooks is empty.
    """
    if not hooks:
        return
    for hook in hooks:
        r = getattr(hook, event)(*args, **kwargs)
        if inspect.isawaitable(r):
            await r


