"""Focused unit tests for the core/aio sync bridge: error on an already-running loop, loop reuse,
faithful streaming aclose, a shared Context visible across segments, and adapter async clients
cached per event loop. All hermetic (offline)."""

import asyncio
import contextvars
import gc
import threading
import weakref

import pytest

from agentmaker.core.aio import _ensure_loop, close_sync_loop, iter_sync, run_sync


# ---------- run_sync ----------

def test_run_sync_basic_and_loop_reuse():
    """run_sync drives a coroutine to completion; the same thread reuses one loop across calls (loop-bound objects stay valid across calls)."""
    async def f(x):
        return x + 1
    assert run_sync(f(1)) == 2
    loop1 = _ensure_loop()
    assert run_sync(f(2)) == 3
    assert _ensure_loop() is loop1


def test_close_sync_loop_closes_and_recreates_the_resident_loop():
    """Explicit bridge shutdown releases the current loop without preventing later sync calls."""
    loop = _ensure_loop()
    close_sync_loop()
    assert loop.is_closed()
    assert _ensure_loop() is not loop


def test_run_sync_rejects_inside_running_loop():
    """Inside an already-running loop (async context): raise a readable error pointing at the async entry point, leaving no "coroutine never awaited" warning."""
    async def outer():
        async def f():
            return 1
        with pytest.raises(RuntimeError, match="arun"):
            run_sync(f())
    asyncio.run(outer())


# ---------- iter_sync ----------

def test_iter_sync_rejects_inside_running_loop_eagerly():
    """iter_sync is a plain function: the check fires eagerly at call time, not deferred to the first next() with a cryptic error."""
    async def agen():
        yield 1
    async def outer():
        with pytest.raises(RuntimeError, match="arun"):
            iter_sync(agen())
    asyncio.run(outer())


def test_iter_sync_basic_and_exception_passthrough():
    """Pull segment by segment; an exception raised inside the async generator propagates as-is to the sync consumer."""
    async def agen():
        yield 1
        yield 2
        raise ValueError("boom")
    g = iter_sync(agen())
    assert next(g) == 1 and next(g) == 2
    with pytest.raises(ValueError, match="boom"):
        next(g)


def test_iter_sync_early_close_runs_aclose():
    """Consumer closes early -> the async stream's finally (which backs harness streaming accounting) still runs deterministically."""
    done = []

    async def agen():
        try:
            yield 1
            yield 2
        finally:
            done.append("closed")

    g = iter_sync(agen())
    assert next(g) == 1
    g.close()
    assert done == ["closed"]


def test_close_sync_loop_finalizes_active_stream_once():
    """Bridge shutdown closes an active stream before its loop, and later generator close is a no-op."""
    from agentmaker.core.aio import _register_loop_cleanup

    done = []

    async def register_client_cleanup():
        _register_loop_cleanup(object(), lambda _loop: done.append("client"))

    run_sync(register_client_cleanup())

    async def agen():
        try:
            yield 1
            yield 2
        finally:
            done.append("closed")

    g = iter_sync(agen())
    assert next(g) == 1
    loop = _ensure_loop()
    close_sync_loop()
    assert loop.is_closed()
    assert done == ["closed", "client"]
    g.close()
    g.close()
    assert done == ["closed", "client"]


def test_unstarted_sync_stream_close_unregisters_owner_cleanup():
    """Closing before the first next releases the owner registration immediately."""
    from agentmaker.core.aio import _local

    async def agen():
        yield 1

    _ensure_loop()
    baseline = len(_local.owner.cleanups)
    stream = iter_sync(agen())
    assert len(_local.owner.cleanups) == baseline + 1
    stream.close()
    assert len(_local.owner.cleanups) == baseline


def test_unstarted_sync_stream_gc_unregisters_owner_cleanup():
    """Discarding an unstarted stream does not leave its async generator retained by the loop owner."""
    from agentmaker.core.aio import _local

    async def agen():
        yield 1

    _ensure_loop()
    baseline = len(_local.owner.cleanups)
    stream = iter_sync(agen())
    stream_ref = weakref.ref(stream)
    assert len(_local.owner.cleanups) == baseline + 1
    del stream
    gc.collect()
    assert stream_ref() is None
    assert len(_local.owner.cleanups) == baseline


def test_iter_sync_contextvars_visible_across_segments():
    """A single shared Context drives the whole run: a contextvar set inside the async generator stays
    visible in the second segment and the teardown segment. A per-segment context-copy implementation would break here (the second segment would read the default)."""
    var = contextvars.ContextVar("v", default=None)
    seen = []

    async def agen():
        var.set("X")
        try:
            yield 1
            seen.append(var.get())
            yield 2
        finally:
            seen.append(var.get())

    g = iter_sync(agen())
    assert next(g) == 1 and next(g) == 2
    g.close()
    assert seen == ["X", "X"]


def test_iter_sync_start_run_inside_async_gen():
    """start_run/reset_run inside an async generator (the real streaming shape): reset raises no Token
    error and run_id stays readable across segments. A per-segment context-copy implementation would make reset_run raise ValueError('created in a different Context')."""
    from agentmaker.runtime.execution.run_context import current_run_id, record_llm, reset_run, start_run
    ids = []

    async def agen():
        tok = start_run("rid-1")
        try:
            yield 1
            ids.append(current_run_id())
            record_llm()
            yield 2
        finally:
            reset_run(tok)

    assert list(iter_sync(agen())) == [1, 2]
    assert ids == ["rid-1"]


def test_run_sync_interrupted_drains_pending_task():
    """run_until_complete is interrupted (Ctrl-C shape: the task is still pending) -> the task is
    cancelled and drained, leaving no ghost task in the persistent loop for a later run_sync to revive."""
    cancelled = []

    async def victim():
        try:
            asyncio.get_running_loop().stop()   # simulate interruption: loop stopped, run_until_complete raises but the task is unfinished
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)              # cancellation lands during drain
            raise

    with pytest.raises(RuntimeError, match="stopped"):
        run_sync(victim())
    assert cancelled == [True]                  # no ghost task: it was cancelled and drained

    async def f():
        return "ok"
    assert run_sync(f()) == "ok"                # persistent loop still usable, and no old task revived


def test_iter_sync_created_sync_consumed_in_async_reports_guidance():
    """An iter_sync generator created in sync code but consumed inside an async context: every pull re-checks and reports readable guidance instead of a cryptic asyncio error."""
    async def agen():
        yield 1

    g = iter_sync(agen())                       # created in sync code

    async def outer():
        with pytest.raises(RuntimeError, match="arun"):
            next(g)                             # consumed in an async context

    asyncio.run(outer())


def test_async_exec_tool_confirm_can_reenter_sync_facade():
    """The confirm callback runs on a worker thread: re-entering the sync facade from inside it
    (a legal pattern, e.g. using another LLM to judge approval) does not hit a "running loop" and the whole run does not crash."""
    from agentmaker.runtime.harness import Harness
    from agentmaker.tools import ToolRegistry

    reg = ToolRegistry()
    reg.register_function(lambda p: "已执行", name="danger", description="高风险桩", requires_confirmation=True)

    async def judge():
        return "yes"

    def confirm(tool, params):
        return run_sync(judge()) == "yes"       # re-enter the sync bridge inside the callback: the worker thread has no running loop, so it is legal

    h = Harness(_NoopLLM(), tool_registry=reg, confirm=confirm)
    out = asyncio.run(h.aexec_tool("danger", {}))
    assert out.status == "success" and "已执行" in out.text


class _NoopLLM:
    """Empty LLM stub (the aexec_tool path never calls the model; only a placeholder for Harness construction)."""
    model = "noop"


# ---------- Adapter async client per-loop caching ----------

def test_adapter_async_client_cached_per_loop():
    """Async SDK clients are cached per event loop: reused within a loop, one per loop across loops.
    The underlying connection pool binds to the loop of first use, so reusing it across loops would raise "attached to a different loop"."""
    pytest.importorskip("openai")
    from agentmaker.core.adapters import OpenAIAdapter
    a = OpenAIAdapter(model="m", api_key="k", base_url=None, timeout=5, default_temperature=0.0)

    async def grab():
        return a._ensure_client()

    c1 = run_sync(grab())
    c2 = run_sync(grab())          # same thread's persistent loop -> same instance
    assert c1 is c2
    c3 = asyncio.run(grab())       # asyncio.run's fresh loop -> new instance
    assert c3 is not c1


def test_adapter_async_clients_evicted_for_closed_loops():
    """Client entries for closed loops are evicted on access, so a long-running "one asyncio.run per task" pattern does not accumulate clients and connections (fd leak)."""
    pytest.importorskip("openai")
    from agentmaker.core.adapters import OpenAIAdapter
    a = OpenAIAdapter(model="m", api_key="k", base_url=None, timeout=5, default_temperature=0.0)

    async def grab():
        return a._ensure_client()

    for _ in range(5):
        asyncio.run(grab())                    # fresh loop and client each time; the loop closes when run ends
    asyncio.run(grab())                        # 6th access: entries for the 5 dead loops should be evicted
    assert len(a._async_clients) == 1


class _TrackedClient:
    def __init__(self, closed):
        self.closed = closed

    async def aclose(self):
        await asyncio.sleep(0)
        self.closed.set()


def test_temporary_sync_thread_closes_clients_tasks_and_loop():
    """Thread exit drains every resource owned by its resident sync bridge loop."""
    from agentmaker.core.adapters.base import BaseAdapter

    class Adapter(BaseAdapter):
        async def chat(self, messages, **kwargs):
            raise AssertionError

        async def stream(self, messages, **kwargs):
            if False:
                yield ""

    adapter = Adapter(model="m", api_key="k", base_url=None, timeout=1,
                      default_temperature=None)
    client_closed = threading.Event()
    task_cancelled = threading.Event()
    observed = {}

    def worker():
        async def setup():
            loop = asyncio.get_running_loop()
            client = adapter._async_client_for_loop(lambda: _TrackedClient(client_closed))

            async def linger():
                try:
                    await asyncio.Event().wait()
                finally:
                    task_cancelled.set()

            asyncio.create_task(linger())
            await asyncio.sleep(0)
            return loop, client

        observed["loop"], observed["client"] = run_sync(setup())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert client_closed.is_set()
    assert task_cancelled.is_set()
    assert observed["loop"].is_closed()
    assert adapter._async_clients == {}


def test_aclose_on_foreign_loop_keeps_resident_client_cleanup():
    """aclose on a non-resident loop closes only that loop's client; the resident loop's registered cleanup still runs at close_sync_loop."""
    from agentmaker.core.adapters.base import BaseAdapter

    class Adapter(BaseAdapter):
        async def chat(self, messages, **kwargs):
            raise AssertionError

        async def stream(self, messages, **kwargs):
            if False:
                yield ""

    adapter = Adapter(model="m", api_key="k", base_url=None, timeout=1,
                      default_temperature=None)
    resident_closed = threading.Event()

    async def create_resident_client():
        adapter._async_client_for_loop(lambda: _TrackedClient(resident_closed))

    run_sync(create_resident_client())

    async def close_on_fresh_loop():
        await adapter.aclose()                 # this loop has no cached client; must not drop the resident registration

    asyncio.run(close_on_fresh_loop())
    assert not resident_closed.is_set()

    close_sync_loop()                          # resident-loop teardown still closes the cached client
    assert resident_closed.is_set()


def test_gemini_cleanup_uses_aio_aclose():
    """Gemini closes the asynchronous transport rather than only its synchronous facade."""
    from agentmaker.core.adapters.gemini import GeminiAdapter

    closed = threading.Event()

    class Aio:
        async def aclose(self):
            closed.set()

    class Client:
        aio = Aio()

        def close(self):
            raise AssertionError("synchronous Gemini close must not be used")

    adapter = GeminiAdapter(model="m", api_key="k", base_url=None, timeout=1,
                            default_temperature=None)

    async def exercise():
        adapter._async_client_for_loop(Client)
        await adapter.aclose()

    run_sync(exercise())
    assert closed.is_set()
    assert adapter._async_clients == {}


def test_llm_client_sync_and_async_contexts_close_owned_client():
    """LLMClient contexts deterministically close the client created on their event loop."""
    from agentmaker.core.llm_clients import LLMClient

    class Adapter:
        def __init__(self):
            self.closed = 0

        async def aclose(self):
            self.closed += 1

    adapter = Adapter()
    client = object.__new__(LLMClient)
    client._adapter = adapter
    with client as entered:
        assert entered is client

    async def use_async_context():
        async with client as entered:
            assert entered is client

    asyncio.run(use_async_context())
    assert adapter.closed == 2
