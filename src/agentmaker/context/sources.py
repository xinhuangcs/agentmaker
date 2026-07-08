"""agentmaker.context.sources: adapt any "fetch candidates by query" callable into a ContextSource.

memory.search / rag.retrieve / etc. already return List[RetrievalResult]; they only lack a name and a uniform
method name. CallableSource fills that layer with a generic adapter, so each source need not write its own
ContextSource subclass, and so context need not depend in reverse on the concrete types of memory / rag (which
method to bind and how many items to take are given explicitly at construction).
"""

import asyncio
import inspect
from typing import Callable, List, Optional

from ..retrieval.types import RetrievalResult
from .types import ContextSource


class CallableSource(ContextSource):
    """Use a callable as a context source, with signature (query) or (query, scope) -> List[RetrievalResult].

    name decides which quota it consumes (corresponding to a key of ContextConfig.source_ratios, e.g. "memory"
    / "rag").

    scope threading (optional): whether / how the run's scope is passed to the callable is decided by
    pass_scope:
      - By default (pass_scope=None) it is auto-detected by positional-parameter count: if there are >= 2
        positional params (e.g. `lambda q, s: ...`), scope is passed as the second positional argument;
        a callable taking only (query) simply ignores scope.
      - Warning: auto-detection only counts positional params. If your fetch takes scope as keyword-only (e.g.
        `def f(query, *, scope=None)`, as memory.search / rag.retrieve do), it will NOT be auto-recognized and
        will not receive the run scope. This is intentional: bind memory.search / rag.retrieve directly and use
        their own scope. To make such a fetch receive the run scope, pass pass_scope=True (passed by keyword
        `scope=`), or rewrite it as `lambda q, s: f(q, scope=s)` (two positional params).

    Example:
        CallableSource("memory", memory.search)                                          # keyword-only scope, uses its own scope
        CallableSource("memory", lambda q, s: memory.search(q, scope=Scope(user=s.user)))  # positional, by the run's user dimension
        CallableSource("rag", rag.retrieve, pass_scope=True)                              # explicitly pass the run scope by keyword to a keyword-only scope
        CallableSource("rag", lambda q: rag.retrieve(q, top_k=8))                         # custom top_k, no scope
    """

    def __init__(self, name: str, fetch: Callable, *, pass_scope: Optional[bool] = None):
        """Build a CallableSource.

        Args:
            name: Source name (the quota key).
            fetch: A callable that fetches candidates by query, with signature (query) or (query, scope) ->
                List[RetrievalResult].
            pass_scope: Whether / how to pass the run's scope to fetch. None (default) = auto-detect by
                positional-parameter count (>= 2 passes scope as the second positional argument, otherwise not
                passed); True = force passing by keyword `fetch(query, scope=scope)` (for when scope is
                keyword-only and auto-detection cannot recognize it); False = force not passing.
        """
        self.name = name
        self._fetch = fetch
        self._is_async = inspect.iscoroutinefunction(fetch)   # async def fetch (e.g. memory.asearch) -> afetch awaits directly
        if pass_scope is True:
            self._mode = "keyword"          # explicit: pass by keyword scope= (for a fetch with keyword-only scope)
        elif pass_scope is False:
            self._mode = "none"             # explicit: do not pass
        else:                               # auto-detect: count positional params only; >= 2 passes as the second positional
            positional = [p for p in inspect.signature(fetch).parameters.values()
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
            self._mode = "positional" if len(positional) >= 2 else "none"

    def _invoke(self, query: str, scope):
        """Call the underlying callable per the mode fixed at construction (returns the raw result, which may be a list or a coroutine)."""
        if self._mode == "positional":
            return self._fetch(query, scope)
        if self._mode == "keyword":
            return self._fetch(query, scope=scope)
        return self._fetch(query)

    def fetch(self, query: str, scope=None) -> List[RetrievalResult]:
        """Call the underlying callable to fetch candidates (sync path). If the underlying is a coroutine (async fetch), fail loud: use afetch instead."""
        r = self._invoke(query, scope)
        if inspect.isawaitable(r):          # sync path received a coroutine: reject, to avoid silently treating a coroutine as the result
            if inspect.iscoroutine(r):
                r.close()
            raise TypeError(f"CallableSource({self.name!r}) fetch returned an awaitable: call an async fetch via afetch (ContextBuilder.abuild_block uses it)")
        return r

    async def afetch(self, query: str, scope=None) -> List[RetrievalResult]:
        """Fetch candidates asynchronously: an async fetch is awaited directly, a sync fetch goes through to_thread (each source occupies its own thread when concurrent via gather)."""
        if self._is_async:
            return await self._invoke(query, scope)
        return await asyncio.to_thread(lambda: self.fetch(query, scope))
