"""agentmaker.core.adapters.base: adapter foundation: BaseAdapter (unified interface) + _BaseStreamState (streaming-parse base class) + _request_error.

Each wire protocol (OpenAI-compatible / Anthropic / Gemini) has one adapter subclass living in its own
file in this package. Each subclass imports its provider SDK lazily (only when used), so a missing SDK for
one provider does not affect the rest of the framework: this file must not import any provider SDK at the
top level.
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

from ..exceptions import LLMRequestError
from ..llm_response import LLMResponse, StreamStats


def _request_error(provider: str, model: str, e: Exception) -> LLMRequestError:
    """Normalize a runtime LLM-call exception into an LLMRequestError: duck-type the HTTP status code and infer retryable from it plus timeout-style exception names (408 / 429 / 5xx, or an exception name containing "timeout"). Shared by all three adapters.

    The status-code field differs across providers: openai / anthropic APIStatusError uses `.status_code`;
    google-genai `errors.APIError` uses `.code` (an int HTTP code, with no `.status_code`). Read
    `.status_code` first, then fall back to an int-typed `.code`.
    """
    status = getattr(e, "status_code", None)
    if status is None:
        code = getattr(e, "code", None)          # google-genai APIError.code = int HTTP code
        if isinstance(code, int):
            status = code
    retryable = bool(status and (status in (408, 429) or status >= 500)) or "timeout" in type(e).__name__.lower()
    return LLMRequestError(f"{provider} call failed (model={model}): {e}", provider=provider, model=model,
                           status_code=status, retryable=retryable)



class BaseAdapter(ABC):
    """Unified adapter interface. Each adapter creates its underlying SDK client lazily (imports / connects
    only when used), so a missing SDK for one provider does not affect the rest of the framework."""

    def __init__(self, *, model, api_key, base_url, timeout, default_temperature, max_tokens_field="max_tokens",
                 structured_output="none", provider=None):
        """
        Store the adapter's runtime parameters; the underlying SDK client is not created here but lazily on first call.

        Args:
            model: Model name.
            api_key: API key.
            base_url: Service endpoint (may be None for a native protocol, using the SDK's default endpoint).
            timeout: Timeout in seconds.
            default_temperature: Default sampling temperature.
            max_tokens_field: Name of the output-length limit field (from the provider profile; only the
                OpenAI protocol reads it, see OpenAIAdapter._params).
            structured_output: The provider's structured-output capability (only OpenAIAdapter branches on
                this: json_schema / json_object / none); the Anthropic / Gemini adapters always use their
                own native path and never read it. From the profile, see ProviderProfile in llm_clients.py.
            provider: The real provider identifier (e.g. deepseek / qwen); used to attribute errors: the
                OpenAI-compatible protocol serves many providers, so on failure mark the real provider rather
                than a generic "openai". When None, each adapter falls back to its own protocol name.
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.default_temperature = default_temperature
        self.max_tokens_field = max_tokens_field
        self.structured_output = structured_output
        self.provider = provider
        # Cache the SDK client per event loop (the framework is fully async: chat/stream are both async and
        # the underlying SDK client is async). An async client's underlying httpx connection pool is bound to
        # the loop that first uses it; reusing it across loops raises "attached to a different loop". Multiple
        # loops arise when the sync facade (aio.run_sync) keeps one resident loop per thread, when a user calls
        # asyncio.run with a fresh loop each time, and so on.
        # Entries for closed loops are evicted explicitly on access (weak references cannot clean them up: the
        # client strongly references its loop through the connection pool, and a value->key strong reference
        # keeps the weak key from ever being collected), see _async_client_for_loop.
        self._async_clients: dict = {}
        self._async_clients_lock = threading.Lock()
        self.last_stream_stats: Optional[StreamStats] = None  # Stats from the most recent stream() call.

    @abstractmethod
    async def chat(self, messages, *, temperature=None, max_tokens=None, **kwargs) -> LLMResponse:
        """
        One-shot (non-streaming) async call returning a unified LLMResponse. Implemented by each protocol subclass using its corresponding async SDK.

        Args:
            messages: Unified message list (in {"role", "content"} form).
            temperature: Sampling temperature; None means use the default.
            max_tokens: Maximum generated tokens, optional.
            **kwargs: Other parameters passed through to the underlying SDK.

        Returns:
            LLMResponse: The unified response object.
        """
        ...

    @abstractmethod
    async def stream(self, messages, *, temperature=None, max_tokens=None, on_stats=None, **kwargs):
        """
        Streaming async call (async generator) yielding text deltas piece by piece. Implemented by each protocol subclass; on completion it fills self.last_stream_stats.

        Args:
            messages: Unified message list.
            temperature: Sampling temperature; None means use the default.
            max_tokens: Maximum generated tokens, optional.
            **kwargs: Other parameters passed through to the underlying SDK.

        Returns:
            Text deltas yielded piece by piece (consumed with async for).
        """
        ...
        yield  # pragma: no cover: makes this method an async-generator signature; subclasses override it.

    def _async_client_for_loop(self, factory):
        """Get / create the async SDK client for the current event loop: reuse within a loop, one per loop across loops (see the caching notes in __init__).

        On access, also evict entries for closed loops: otherwise the "one asyncio.run per task" pattern would
        accumulate clients and keep-alive connections without bound (fd leak). Evicted clients' connections are
        left to GC (an async close cannot run on a dead loop). A lock guards against multiple threads racing on
        the first call and creating duplicate clients (one of which would be overwritten and never released).

        Args:
            factory: A zero-argument factory that builds a new async client instance.

        Returns:
            The async client for the current loop.
        """
        loop = asyncio.get_running_loop()   # Called from within chat/stream, so a running loop always exists.
        with self._async_clients_lock:
            for dead in [lp for lp in self._async_clients if lp.is_closed()]:
                self._close_quietly(self._async_clients.pop(dead))
            client = self._async_clients.get(loop)
            if client is None:
                client = factory()
                self._async_clients[loop] = client
        return client

    @staticmethod
    def _close_quietly(client) -> None:
        """Quietly close an evicted async client (its connections are bound to a dead loop, so a real close will most likely fail and the socket is left to GC).

        The goal is not a successful close but to mark the client as closed: otherwise some SDKs' __del__ (e.g.
        openai's AsyncAPIClient) schedule a doomed aclose task onto the current running loop at GC time, spamming
        "Task exception was never retrieved" noise. httpx's aclose sets the CLOSED state before awaiting, so here
        we just drive the close coroutine as a task and consume its exception.
        """
        closer = getattr(client, "aclose", None) or getattr(client, "close", None)
        if closer is None:
            return
        try:
            res = closer()
        except Exception:  # noqa: BLE001: best-effort, a failed close must not disrupt the main flow.
            return
        if asyncio.iscoroutine(res):
            try:
                task = asyncio.get_running_loop().create_task(res)
                task.add_done_callback(lambda t: t.cancelled() or t.exception())   # Consume the exception to avoid "never retrieved" noise.
            except Exception:  # noqa: BLE001
                res.close()

    @staticmethod
    def _elapsed_ms(start: float) -> int:
        """
        Compute the elapsed time from start until now, in milliseconds. Uses the monotonic clock perf_counter, unaffected by system-time adjustments.

        Args:
            start: The start instant recorded by time.perf_counter().

        Returns:
            int: Elapsed time in milliseconds.

        Example:
            t0 = time.perf_counter(); _elapsed_ms(t0) -> 1234
        """
        return int((time.perf_counter() - start) * 1000)

    def _temperature(self, temperature):
        """The temperature to send for this request: an explicit argument wins; otherwise the constructor's default_temperature (default None = do not send a temperature, deferring to each model's server-side default).

        Returning None means no temperature parameter is included (each adapter honors this by leaving it out of
        the request). Design choice (intentional): the framework does not decide for the developer whether a
        given model supports a temperature parameter. To use temperature, the developer passes `temperature=`
        explicitly (and confirms the model supports it); if a model does not support temperature and one is
        forced, the server error is returned as-is for the developer to act on. This avoids maintaining a
        "which models lock temperature" list and avoids brittle error-message matching.
        """
        return temperature if temperature is not None else self.default_temperature



class _BaseStreamState:
    """Shared base class for per-chunk streaming parse plus stats accumulation: __init__ stores model/finish_reason/usage, stats() produces the stats.

    Each protocol subclass only overrides feed() (how to parse each chunk). The stats() of OpenAI and Gemini
    were byte-for-byte identical, so they are unified here.
    """

    def __init__(self, model):
        self.model = model
        self.finish_reason = None
        self.usage = None

    def feed(self, chunk) -> str:
        """Feed in one streaming chunk: update stats and return this chunk's text delta (empty string if none). Implemented by each protocol subclass."""
        raise NotImplementedError

    def stats(self, latency_ms: int) -> StreamStats:
        """Produce the stats object for this stream."""
        return StreamStats(model=self.model, finish_reason=self.finish_reason,
                           usage=self.usage, latency_ms=latency_ms)
