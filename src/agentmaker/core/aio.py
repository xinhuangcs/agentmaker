"""agentmaker.core.aio: the single sync-to-async bridge (the implementation behind the synchronous facade).

Internally the framework has exactly one async implementation (the real async def bodies); every synchronous entry point drives that implementation to completion through this module and returns the result. This is the only place in the whole framework that touches the event loop (the event loop is asyncio's scheduling hub, and one thread can run only one at a time).

Three design rules (aligned with the run_sync shape of the OpenAI Agents SDK / Pydantic-AI):
    - One loop per thread, reused until the thread exits: objects bound to a loop (async SDK connection pools, asyncio.Lock, etc.) require the same loop every time; a use-and-discard approach (asyncio.run semantics) makes the second call hit "attached to a different loop".
    - Already inside a running loop -> fail loud: a second loop cannot be opened on the same thread, so raise a human-readable error steering the caller to the async entry points (inside an async function / Jupyter / FastAPI, please await; Jupyter supports top-level await, do not pull in nest_asyncio, which is unmaintained).
    - Streaming interruptions must be finalized: iter_sync drives the whole run with a single shared Context (contextvars set inside the async generator body remain visible in the second segment and the teardown segment), so even if the consumer breaks / closes early, aclose still runs (in-stream finally bookkeeping is not lost).

Interrupt safety: when run_until_complete is interrupted (typically Ctrl-C), the task may still be attached to the resident loop; if left alone, the next time that loop is driven the task would "resurrect" and continue (late hooks / trace / bookkeeping would leak into a new round). So all drive points finalize uniformly via "cancel + drain" (_drain) and never leave a ghost task on the loop.

Boundary: explicitly exhaust or close the synchronous generator returned by iter_sync (e.g. via contextlib.closing). If it is reclaimed by GC while "this thread is running a different loop" or "that loop is being driven", teardown is scheduled onto the loop and deferred until its next turn (best-effort, no re-entrant crash).
"""

import asyncio
import atexit
import contextvars
import inspect
import threading
import weakref
from collections.abc import AsyncGenerator, Coroutine, Iterator
from typing import Any, Awaitable, Callable, Optional, TypeVar


_T = TypeVar("_T")

_local = threading.local()
_owners: "weakref.WeakSet[_LoopOwner]" = weakref.WeakSet()
_owners_lock = threading.Lock()


class _SyncIterator(Iterator[_T]):
    """Proxy a synchronous stream so close works before its generator starts."""

    def __init__(self, iterator: Iterator[_T], cleanup: Callable[[], None]) -> None:
        self._iterator = iterator
        self._cleanup = cleanup
        self._closed = False

    def __iter__(self) -> "_SyncIterator[_T]":
        return self

    def __next__(self) -> _T:
        return next(self._iterator)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        iterator = self._iterator
        cleanup = self._cleanup
        self._iterator = iter(())
        self._cleanup = lambda: None
        try:
            closer = getattr(iterator, "close", None)
            if closer is not None:
                closer()
        finally:
            cleanup()

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:  # noqa: BLE001 -- destructors cannot report cleanup failures safely
            pass


class _LoopOwner:
    """Own one thread's resident loop and its loop-bound cleanup callbacks."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.cleanups: dict[
            object,
            tuple[int, Callable[[asyncio.AbstractEventLoop], Awaitable[None] | None]],
        ] = {}
        self.closed = False

    def close(self) -> None:
        """Close registered resources, pending tasks, and the resident loop."""
        if self.closed:
            return
        if self.loop.is_closed():
            self.cleanups.clear()
            self.closed = True
            return
        if self.loop.is_running():
            raise RuntimeError("Cannot close a resident event loop while it is running.")
        self.closed = True
        first_error: Optional[BaseException] = None
        callbacks = [callback for _, callback in sorted(self.cleanups.values(), key=lambda item: item[0])]
        self.cleanups.clear()
        for callback in callbacks:
            try:
                result = callback(self.loop)
                if inspect.isawaitable(result):
                    self.loop.run_until_complete(result)
            except BaseException as exc:  # noqa: BLE001 -- finish every cleanup before reporting the first failure
                first_error = first_error or exc
        pending = asyncio.all_tasks(self.loop)
        for task in pending:
            task.cancel()
        if pending:
            try:
                self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except BaseException as exc:  # noqa: BLE001
                first_error = first_error or exc
        try:
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
            self.loop.run_until_complete(self.loop.shutdown_default_executor())
        except BaseException as exc:  # noqa: BLE001
            first_error = first_error or exc
        finally:
            self.loop.close()
        if first_error is not None:
            raise first_error

    def __del__(self) -> None:
        try:
            self.close()
        except BaseException:  # noqa: BLE001 -- destructors cannot report cleanup failures safely
            pass


def _register_loop_cleanup(
    key: object,
    callback: Callable[[asyncio.AbstractEventLoop], Awaitable[None] | None],
) -> bool:
    """Register cleanup when the running loop belongs to the synchronous bridge."""
    owner = getattr(_local, "owner", None)
    if owner is None or owner.closed or owner.loop is not asyncio.get_running_loop():
        return False
    owner.cleanups[key] = (0, callback)
    return True


def _register_resident_cleanup(
    loop: asyncio.AbstractEventLoop,
    key: object,
    callback: Callable[[asyncio.AbstractEventLoop], Awaitable[None] | None],
    *,
    priority: int = 0,
) -> bool:
    """Register cleanup for this thread's resident loop outside an async call."""
    owner = getattr(_local, "owner", None)
    if owner is None or owner.closed or owner.loop is not loop:
        return False
    owner.cleanups[key] = (priority, callback)
    return True


def _unregister_loop_cleanup(key: object, *,
                             loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
    """Remove a cleanup callback from the current thread's resident loop.

    When ``loop`` is given, the callback is removed only if the resident loop is
    that loop: closing a resource on a non-resident loop must not drop the
    resident loop's still-pending cleanup for the same key.
    """
    owner = getattr(_local, "owner", None)
    if owner is None:
        return
    if loop is not None and owner.loop is not loop:
        return
    owner.cleanups.pop(key, None)


def close_sync_loop() -> None:
    """Close this thread's resident synchronous-bridge loop and loop-bound resources."""
    owner = getattr(_local, "owner", None)
    if owner is None:
        return
    _reject_running_loop()
    try:
        owner.close()
    finally:
        del _local.owner


def _close_resident_loops() -> None:
    with _owners_lock:
        owners = list(_owners)
    for owner in owners:
        try:
            owner.close()
        except BaseException:  # noqa: BLE001 -- interpreter shutdown is best-effort
            pass


atexit.register(_close_resident_loops)


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """Get this thread's resident event loop, creating and remembering a new one if absent (or if closed externally).

    Avoids asyncio.get_event_loop / set_event_loop: the former is a deprecated / restricted path as of 3.12, and
    the teardown of asyncio.run clears the thread's default loop pointer; a self-managed threading.local is the most
    robust.

    Returns:
        asyncio.AbstractEventLoop: The loop dedicated to this thread.
    """
    owner = getattr(_local, "owner", None)
    if owner is None or owner.closed or owner.loop.is_closed():
        owner = _LoopOwner()
        _local.owner = owner
        with _owners_lock:
            _owners.add(owner)
    return owner.loop


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


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
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


def iter_sync(agen: AsyncGenerator[_T, None]) -> Iterator[_T]:
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
    cleanup_key = object()
    close_task: Optional[asyncio.Task[None]] = None
    close_done = False

    def _drive(coro):
        _reject_running_loop()
        task = loop.create_task(coro, context=ctx)
        try:
            return loop.run_until_complete(task)
        except BaseException:
            _drain(loop, task)
            raise

    async def _finalize() -> None:
        nonlocal close_done
        try:
            await agen.aclose()
        finally:
            close_done = True
            _unregister_loop_cleanup(cleanup_key)

    def _close_task() -> Optional[asyncio.Task[None]]:
        nonlocal close_task, close_done
        if close_done:
            return None
        if close_task is not None:
            return close_task
        if loop.is_closed():
            close_done = True
            _unregister_loop_cleanup(cleanup_key)
            return None
        finalizer = _finalize()
        try:
            close_task = loop.create_task(finalizer, context=ctx)
        except BaseException:
            finalizer.close()
            raise
        return close_task

    def _owner_close(_loop: asyncio.AbstractEventLoop) -> Optional[asyncio.Task[None]]:
        return _close_task()

    _register_resident_cleanup(loop, cleanup_key, _owner_close, priority=-100)

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
            if not loop.is_closed():
                loop.call_soon_threadsafe(_close_task)
        else:
            task = _close_task()
            if task is not None:
                try:
                    loop.run_until_complete(task)
                except BaseException:
                    _drain(loop, task)
                    raise

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

    return _SyncIterator(_pump(), _close)
