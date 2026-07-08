"""agentmaker.runtime.guardrails: guardrail interface and generic implementation.

A Guardrail checks a piece of text (an agent's input or output) and returns a GuardrailResult; a tripwire
(passed=False) is caught by the layer above (harness / recipe), which raises GuardrailTripwireError. The
concrete rules are app business logic: agentmaker only provides the interface plus a CallableGuardrail
that wraps any function into a guardrail (mirroring context's CallableSource), leaving the rules to the app.
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Union


@dataclass
class GuardrailResult:
    """The result of a guardrail check.

    Attributes:
        passed: True lets the text through; False signals a tripwire.
        message: A human-readable explanation of the block, shown to the user on a tripwire (may be empty when passed=True).
    """
    passed: bool
    message: str = ""


class Guardrail(ABC):
    """Guardrail interface: check a piece of text and return a GuardrailResult. Input guardrails check user
    input; output guardrails check the model's reply.

    Two tracks: synchronous implementations override check (pure-computation guardrails); the framework
    execution layer (the Agent.run template) calls acheck. By default acheck inlines a direct call to check
    (most guardrails are pure computation like length / regex checks, not worth dispatching to a thread pool).
    Override acheck if the guardrail does blocking I/O or wants to call an LLM to moderate.
    """

    @abstractmethod
    def check(self, text: str) -> GuardrailResult:
        """Check text and return a GuardrailResult (passed plus a human-readable message)."""

    async def acheck(self, text: str) -> GuardrailResult:
        """Async check (called by the framework execution layer).

        Defaults to a direct inline call to the synchronous check; guardrails with blocking I/O or LLM
        moderation override this method.
        """
        return self.check(text)


class CallableGuardrail(Guardrail):
    """Wrap any callable into a guardrail (mirroring CallableSource): fn(text) returns a bool or a GuardrailResult.

    When fn returns a bool, False is a tripwire and uses the message given at construction; fn may also return
    a GuardrailResult directly, carrying its own message.
    Example: CallableGuardrail(lambda t: len(t) < 4000, message="input too long").

    Both sync and async fn are accepted: a pure-computation fn returns a bool / GuardrailResult directly; a fn
    that returns an awaitable (an `async def`, a lambda wrapping an async call, or an object with an async
    `__call__`) is awaited via acheck (the framework execution layer goes through acheck). The synchronous
    check is a pure-sync path and fails loud on an awaitable (to avoid bool(coroutine) being always truthy and
    silently letting the text through).
    """

    def __init__(self, fn: Callable[[str], Union[bool, GuardrailResult]], *, message: str = "guardrail triggered"):
        """
        Args:
            fn: The check function; receives text and returns a bool (True lets through) or a GuardrailResult; may be sync or async.
            message: The block explanation used when fn returns a bool that is False.
        """
        self._fn = fn
        self._is_async = inspect.iscoroutinefunction(fn)
        self._message = message

    def _coerce(self, r) -> GuardrailResult:
        """Normalize fn's return value into a GuardrailResult: wrap a bool, pass a GuardrailResult through unchanged."""
        if isinstance(r, GuardrailResult):
            return r
        return GuardrailResult(passed=bool(r), message="" if r else self._message)

    def check(self, text: str) -> GuardrailResult:
        """Call fn synchronously: wrap a bool result into a GuardrailResult, pass a GuardrailResult through unchanged. An async fn fails loud on this path."""
        r = self._fn(text)
        if inspect.isawaitable(r):          # Sync signature but returned a coroutine / awaitable: reject it, so bool(awaitable) does not become always truthy and silently let the text through.
            if inspect.iscoroutine(r):
                r.close()                   # Close the un-awaited coroutine to suppress the "coroutine was never awaited" warning.
            raise TypeError("CallableGuardrail's synchronous check received an awaitable: run async fn through acheck (the framework execution layer calls it automatically)")
        return self._coerce(r)

    async def acheck(self, text: str) -> GuardrailResult:
        """Call fn asynchronously: await an async fn directly; call a sync-signature fn inline, and if it returns an awaitable, await that before normalizing.

        Why the second isawaitable fallback: `_is_async` (iscoroutinefunction) only recognizes functions defined
        with `async def`, not a lambda wrapping an async call or an object with an async __call__: their signature
        is synchronous yet the call returns a coroutine. Without this fallback the un-awaited coroutine would be
        judged truthy by `_coerce`'s bool(), silently letting through a guardrail that should have blocked (such as
        an async LLM moderation check).
        """
        r = await self._fn(text) if self._is_async else self._fn(text)
        if inspect.isawaitable(r):          # Sync signature that returned an awaitable (lambda wrapping async / async __call__, which _is_async does not recognize): await before normalizing.
            r = await r
        return self._coerce(r)
