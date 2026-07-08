"""agentmaker.core.aio: the single sync-to-async bridge (the implementation behind the synchronous facade).

Internally the framework has exactly one async implementation (the real async def bodies); every synchronous entry point drives that implementation to completion through this module and returns the result. This is the only place in the whole framework that touches the event loop (the event loop is asyncio's scheduling hub, and one thread can run only one at a time).

Three design rules (aligned with the run_sync shape of the OpenAI Agents SDK / Pydantic-AI):
    - One loop per thread, reused, never closed: objects bound to a loop (async SDK connection pools, asyncio.Lock, etc.) require the same loop every time; a use-and-discard approach (asyncio.run semantics) makes the second call hit "attached to a different loop".
    - Already inside a running loop -> fail loud: a second loop cannot be opened on the same thread, so raise a human-readable error steering the caller to the async entry points (inside an async function / Jupyter / FastAPI, please await; Jupyter supports top-level await, do not pull in nest_asyncio, which is unmaintained).
    - Streaming interruptions must be finalized: iter_sync drives the whole run with a single shared Context (contextvars set inside the async generator body remain visible in the second segment and the teardown segment), so even if the consumer breaks / closes early, aclose still runs (in-stream finally bookkeeping is not lost).

Interrupt safety: when run_until_complete is interrupted (typically Ctrl-C), the task may still be attached to the resident loop; if left alone, the next time that loop is driven the task would "resurrect" and continue (late hooks / trace / bookkeeping would leak into a new round). So all drive points finalize uniformly via "cancel + drain" (_drain) and never leave a ghost task on the loop.

Boundary: explicitly exhaust or close the synchronous generator returned by iter_sync (e.g. via contextlib.closing). If it is reclaimed by GC while "this thread is running a different loop" or "that loop is being driven", teardown is scheduled onto the loop and deferred until its next turn (best-effort, no re-entrant crash).
"""

import asyncio
import contextvars
import threading

_local = threading.local()   # per-thread resident event loop (lazily created, reused, never closed)


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Get this thread's resident event loop, creating and remembering a new one if absent (or if closed externally).

    Avoids asyncio.get_event_loop / set_event_loop: the former is a deprecated / restricted path as of 3.12, and
    the teardown of asyncio.run clears the thread's default loop pointer; a self-managed threading.local is the most
    robust.

    Returns:
        asyncio.AbstractEventLoop: The loop dedicated to this thread.
    """
    loop = getattr(_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _local.loop = loop
    return loop


def _reject_running_loop() -> None:
    """This thread already has a running loop -> raise a human-readable error (synchronous entry points are unusable in an async environment)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        "The current thread already has a running event loop (async function / Jupyter / FastAPI, etc.), so the synchronous entry points cannot be used: "
        "please use the async entry points instead (await agent.arun(...) / aresume / astream_run).")


def _drain(loop: asyncio.AbstractEventLoop, task: "asyncio.Task") -> None:
    """Finalization after run_until_complete is interrupted: cancel the task and drive the cancellation to completion, never leaving a ghost task to resurrect on the resident loop.

    Exceptions from the drain itself (CancelledError / the task's own error / a second interrupt) are all swallowed;
    the interrupter takes priority.
    """
    task.cancel()
    try:
        loop.run_until_complete(task)
    except BaseException:  # noqa: BLE001 -- see docstring: the original exception takes priority
        pass


def run_sync(coro):
    """Synchronously drive a coroutine to completion and return its result (the core of the synchronous facade).

    Args:
        coro: The coroutine object (the "ticket" produced by calling an async def).

    Returns:
        The coroutine's return value; any exception raised inside the coroutine propagates unchanged. On interruption
        (e.g. Ctrl-C) the task is cancelled and drained before the interrupt exception propagates.
    """
    try:
        _reject_running_loop()
    except RuntimeError:
        coro.close()   # this ticket will never run, so close it explicitly to avoid a "coroutine was never awaited" warning
        raise
    loop = _ensure_loop()
    task = loop.create_task(coro)
    try:
        return loop.run_until_complete(task)
    except BaseException:
        _drain(loop, task)
        raise


def iter_sync(agen):
    """Adapt an async generator into a synchronous generator (the core of the streaming facade).

    Written as a plain function rather than a generator function so the running-loop probe fires immediately at call
    time: generators execute lazily, so if the probe lived only in the generator body the error would be deferred to
    the first next; the probe also runs again on each pull (the generator may be created in a sync context and then
    carried into an async environment to be consumed, where it must also give human-readable guidance rather than
    asyncio gibberish).

    It does three things:
        - Drive segment by segment: each time the consumer calls next, spin the loop until the next segment is produced (interruptions likewise "cancel + drain").
        - Drive the whole run with a single shared Context: contextvars set inside the async generator body (e.g. the run context) remain visible in the second segment and the teardown segment; an implementation that copies a fresh context per segment would lose counts and raise a Token error on reset.
        - Always run aclose on teardown: even if the consumer breaks / closes early, the in-stream finally (bookkeeping / trace) executes; when it cannot be driven immediately (GC re-entry / this thread running a different loop), it is scheduled onto the loop and deferred, with no re-entrant crash.

    Args:
        agen: The async generator object.

    Returns:
        A synchronous generator yielding agen's elements one segment at a time; any exception raised inside agen
        propagates unchanged.
    """
    _reject_running_loop()
    loop = _ensure_loop()
    ctx = contextvars.copy_context()   # shared for the whole run (create_task(context=ctx) runs directly within it, so mutations are visible across segments)

    def _drive(coro):
        _reject_running_loop()
        task = loop.create_task(coro, context=ctx)
        try:
            return loop.run_until_complete(task)
        except BaseException:
            _drain(loop, task)
            raise

    def _close():
        """Finalize with aclose: normally drive it straight to completion; when this thread has a running loop
        (consumed inside an async environment / GC re-entry) or that loop is being driven by another thread, it
        cannot be driven here, so schedule aclose onto the loop (thread-safe) to run on its next turn."""
        try:
            asyncio.get_running_loop()
            busy = True
        except RuntimeError:
            busy = loop.is_running()   # cross-thread GC: the loop is being driven by its owning thread
        if busy:
            loop.call_soon_threadsafe(lambda: loop.create_task(agen.aclose(), context=ctx))
        else:
            _drive(agen.aclose())      # aclose is a no-op once exhausted / already errored

    def _pump():
        try:
            while True:
                try:
                    piece = _drive(anext(agen))
                except StopAsyncIteration:
                    break
                yield piece
        finally:
            _close()

    return _pump()
