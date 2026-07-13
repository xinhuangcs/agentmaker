"""Async and synchronous extension seams under hermetic local execution."""

import asyncio
import time

import pytest

from agentmaker.runtime.guardrails import CallableGuardrail, Guardrail, GuardrailResult
from agentmaker.runtime.hooks import Hook, afire


# ---------- Guardrail: stop rejecting async fns + acheck dual track ----------

def test_callable_guardrail_async_fn_via_acheck():
    """acheck awaits an async callable while check rejects it."""
    async def afn(text):
        return len(text) < 3                                  # block long text

    g = CallableGuardrail(afn, message="过长")
    assert asyncio.run(g.acheck("ok")).passed is True
    r = asyncio.run(g.acheck("toolong"))
    assert r.passed is False and r.message == "过长"
    with pytest.raises(TypeError):                            # sync check must not run an async fn (avoids bool(coroutine) always being truthy)
        g.check("x")


def test_callable_guardrail_sync_fn_both_paths():
    """Sync fn: both check and the default acheck (which inlines check) are correct."""
    g = CallableGuardrail(lambda t: "bad" not in t, message="命中违禁")
    assert g.check("hello").passed is True
    assert asyncio.run(g.acheck("a bad word")).passed is False


def test_guardrail_default_acheck_inlines_check():
    """Guardrail's default acheck calls the subclass's sync check inline (no thread pool)."""
    class _G(Guardrail):
        def check(self, text):
            return GuardrailResult(passed=text != "stop", message="停")
    assert asyncio.run(_G().acheck("go")).passed is True
    assert asyncio.run(_G().acheck("stop")).passed is False


# ---------- Hook: afire allows async event methods ----------

def test_afire_awaits_async_hook_method():
    """Hook event methods may be async and afire awaits them correctly; sync methods work for free; an empty list short-circuits."""
    seen = []

    class _H(Hook):
        async def before_model(self, messages):
            await asyncio.sleep(0)
            seen.append(("async", messages))
        def after_model(self, resp):
            seen.append(("sync", resp))

    async def go():
        await afire([_H()], "before_model", ["m"])
        await afire([_H()], "after_model", "r")
        await afire([], "before_model", [])                   # empty list short-circuits, no raise

    asyncio.run(go())
    assert ("async", ["m"]) in seen and ("sync", "r") in seen


# ---------- SessionStore / CheckpointStore: default a* via to_thread ----------

def test_session_store_default_a_double_track():
    """SqliteSessionStore, unmodified, reads and writes consistently across threads via the default aappend_many/aload/aclear."""
    from agentmaker.core.message import Message
    from agentmaker.retrieval import Scope
    from agentmaker.runtime.sessions import SqliteSessionStore
    s = SqliteSessionStore()
    sc = Scope(user="u")

    async def go():
        await s.aappend_many([Message("hi", "user"), Message("yo", "assistant")], scope=sc)
        loaded = await s.aload(scope=sc)
        assert [m.content for m in loaded] == ["hi", "yo"]
        await s.aclear(scope=sc)
        assert await s.aload(scope=sc) == []
    asyncio.run(go())


def test_checkpoint_store_default_a_double_track():
    """SqliteCheckpointStore, unmodified, reads and writes consistently via the default asave/aload/aclear."""
    from agentmaker.retrieval import Scope
    from agentmaker.runtime.execution import SqliteCheckpointStore
    cp = SqliteCheckpointStore()
    sc = Scope(user="u")

    async def go():
        await cp.asave('{"x":1}', scope=sc)
        assert await cp.aload(scope=sc) == '{"x":1}'
        await cp.aclear(scope=sc)
        assert await cp.aload(scope=sc) is None
    asyncio.run(go())


# ---------- ContextSource: afetch + abuild_block gather concurrency ----------

def test_context_source_afetch_concurrent_and_failloud():
    """Two sync sources run concurrently under abuild_block's gather (wall clock < serial sum); CallableSource's sync fetch fails loud on a coroutine."""
    from agentmaker.context import CallableSource, ContextBuilder, ContextConfig
    from agentmaker.retrieval.types import RetrievalResult

    def slow_fetch(_q):
        time.sleep(0.15)
        return [RetrievalResult(content="c", score=1.0, source="vector", id="i")]

    builder = ContextBuilder(ContextConfig(max_tokens=2000, source_ratios={"a": 0.5, "b": 0.5}))
    srcs = [CallableSource("a", slow_fetch), CallableSource("b", slow_fetch)]
    t0 = time.perf_counter()
    asyncio.run(builder.abuild_block("q", sources=srcs))
    assert time.perf_counter() - t0 < 0.28        # concurrent: ~0.15s, not the ~0.30s of serial

    async def acoro(_q):
        return []
    bad = CallableSource("a", acoro)               # async fetch on the sync fetch path -> fail-loud
    with pytest.raises(TypeError):
        bad.fetch("q")


# ---------- Retrieval backend: SqliteHybridRetriever a* keeps cross-index writes atomic in one transaction ----------

def test_sqlite_hybrid_aadd_atomic_rollback():
    """If keyword_index write raises during aadd, everything rolls back with no residue in the vector store (cross-index single-transaction atomicity, wrapping the sync atomic version via to_thread)."""
    from agentmaker.retrieval import OpenAIEmbedder, Scope, build_sqlite_hybrid

    class _FakeEmbedder(OpenAIEmbedder):
        def __init__(self): self._dim = 4
        @property
        def dim(self): return 4
        def embed(self, texts): return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    r = build_sqlite_hybrid(_FakeEmbedder())
    sc = Scope(base="rag")
    # make the keyword index add raise (simulate a half-written failure)
    orig_add = r.keyword_index.add
    r.keyword_index.add = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kw boom"))

    async def go():
        with pytest.raises(RuntimeError):
            await r.aadd(["id1"], ["alpha text"], scope=sc)
    asyncio.run(go())
    r.keyword_index.add = orig_add                 # restore
    # no residue in the vector store (full rollback): not retrievable
    hits = r.search("alpha", top_k=5, scope=sc)
    assert all(h.id != "id1" for h in hits)


def test_sqlite_hybrid_asearch_works():
    """SqliteHybridRetriever's asearch override (whole thing via to_thread) returns normally."""
    from agentmaker.retrieval import OpenAIEmbedder, Scope, build_sqlite_hybrid

    class _FakeEmbedder(OpenAIEmbedder):
        def __init__(self): self._dim = 4
        @property
        def dim(self): return 4
        def embed(self, texts): return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    r = build_sqlite_hybrid(_FakeEmbedder())
    sc = Scope(base="rag")

    async def go():
        await r.aadd(["id1"], ["alpha beta"], scope=sc)
        return await r.asearch("alpha", top_k=5, scope=sc)
    hits = asyncio.run(go())
    assert any(h.id == "id1" for h in hits)
