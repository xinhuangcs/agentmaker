"""agentmaker.core.adapters.openai_compat: OpenAI-compatible protocol adapter (serves every OpenAI-compatible provider: OpenAI / DeepSeek / Qwen / Moonshot ...).

The file is named openai_compat rather than openai to (1) emphasize that it serves a whole class of
compatible providers, and (2) avoid colliding with the openai SDK package name.
"""

import time

from ..exceptions import LLMError, LLMResponseError
from ..llm_response import LLMResponse
from .base import BaseAdapter, _BaseStreamState, _request_error


def _to_openai_part(part: dict) -> dict:
    """Translate one neutral content part into the Chat Completions content-part format
    (verified against the official vision guide: images ride an image_url object, with
    base64 sources expressed as a data URL)."""
    kind = part.get("type")
    if kind == "text":
        return {"type": "text", "text": part.get("text", "")}
    if kind == "image":
        if part.get("url"):
            return {"type": "image_url", "image_url": {"url": part["url"]}}
        return {"type": "image_url",
                "image_url": {"url": f"data:{part['media_type']};base64,{part['data']}"}}
    raise LLMResponseError(f"Unknown content part type: {kind!r} (supported: text / image)")


def _to_openai_messages(messages) -> list:
    """Translate neutral multimodal part lists in message content into OpenAI content parts
    (plain str content and all other message fields pass through untouched)."""
    out = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            m = {**m, "content": [_to_openai_part(p) for p in content]}
        out.append(m)
    return out


class _StreamState(_BaseStreamState):
    """Per-chunk parse for OpenAI streaming (overrides feed; adds tool-call accumulation on top of the base class).

    Tool-call deltas are accumulated by `index` per the official streaming contract: the first delta of a
    call carries id / function.name, later deltas carry function.arguments as JSON fragments to concatenate.
    Text deltas are also accumulated so stream() can assemble a complete LLMResponse after the stream drains.
    """

    def __init__(self, model):
        super().__init__(model)
        self.text_parts: list = []      # every text delta, joined into LLMResponse.content at the end
        self._calls: dict = {}          # index -> accumulating {"id", "type", "function": {"name", "arguments"}}

    def feed(self, chunk) -> str:
        """Feed in one streaming chunk: update model / finish_reason / usage, accumulate tool-call deltas, and return this chunk's text delta (empty string if none)."""
        if getattr(chunk, "model", None):
            self.model = chunk.model
        if getattr(chunk, "usage", None):           # usage (requires stream_options) may land on a trailing empty chunk, or ride on the last chunk that still has choices (e.g. DeepSeek): catch both.
            self.usage = chunk.usage.model_dump()
        if not chunk.choices:
            return ""
        ch = chunk.choices[0]
        for tc in (getattr(ch.delta, "tool_calls", None) or []):
            # Accumulate by index (official streaming pattern): id / name arrive on the first delta only, arguments arrive as fragments.
            entry = self._calls.setdefault(tc.index, {"id": None, "type": "function",
                                                      "function": {"name": None, "arguments": ""}})
            if getattr(tc, "id", None):
                entry["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    entry["function"]["name"] = fn.name
                if getattr(fn, "arguments", None):
                    entry["function"]["arguments"] += fn.arguments
        if ch.finish_reason:
            self.finish_reason = ch.finish_reason
        piece = ch.delta.content or ""              # delta is incremental; the first chunk often carries only role.
        if piece:
            self.text_parts.append(piece)
        return piece

    def final_tool_calls(self):
        """Return accumulated tool calls in the unified feed-back format (same shape as the chat path), or None."""
        if not self._calls:
            return None
        return [self._calls[i] for i in sorted(self._calls)]



class OpenAIAdapter(BaseAdapter):
    """OpenAI's `POST /chat/completions` protocol; shared by all OpenAI-compatible providers."""

    def _ensure_client(self):
        """Lazily create the async openai client (AsyncOpenAI); cached per event loop (see the base class's _async_client_for_loop)."""
        def make():
            from openai import AsyncOpenAI
            return AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._async_client_for_loop(make)

    def _params(self, messages, temperature, max_tokens, *, stream, **kwargs):
        """
        Assemble the request parameters for /chat/completions; include optional fields only when they actually have a value, to avoid some compatible services erroring on empty fields.

        Args:
            messages: The message list.
            temperature: Sampling temperature; None uses the default.
            max_tokens: Maximum tokens, optional.
            stream: Whether to stream.
            **kwargs: Passed-through provider-specific parameters (extra_body, response_format,
                stream_options ...), the entry point for "native features".

        Returns:
            dict: A parameter dict ready to expand and pass to the SDK.
        """
        output_schema = kwargs.pop("output_schema", None)  # Intercept: cannot pass through as-is (the API does not recognize this key); translate it into response_format per capability.
        params = {
            "model": self.model,
            "messages": _to_openai_messages(messages),   # multimodal part lists -> OpenAI content parts; plain text passes through
            "stream": stream,
        }
        temp = self._temperature(temperature)
        if temp is not None:        # None = do not send a temperature by default (see BaseAdapter._temperature); send it only when passed explicitly or default_temperature is set.
            params["temperature"] = temp
        if stream:
            # By default the OpenAI protocol does not include usage on the trailing stream chunk; include_usage must be turned on explicitly (otherwise last_stream_stats.usage is always None and cost tracking breaks).
            params["stream_options"] = {"include_usage": True}
        if max_tokens is not None:
            # The field name is taken from the provider profile (self.max_tokens_field): openai/qwen/kimi use
            # max_completion_tokens (mandatory for OpenAI reasoning models, and it includes chain-of-thought
            # tokens), while deepseek/zhipu/local etc. use max_tokens. See _PROFILES in llm_clients.py.
            params[self.max_tokens_field] = max_tokens
        if output_schema is not None:
            rf = self._response_format(output_schema)
            if rf is not None:
                params["response_format"] = rf
        params.update(kwargs)       # Last: caller-supplied temperature / stream_options etc. can override the defaults above.
        return params

    def _response_format(self, schema):
        """Translate a JSON Schema into response_format per the provider's capability (self.structured_output); none -> None (send nothing, rely on the prompt fallback).

        json_schema: response_format json_schema (strict is not enabled: strict requires every field required
            plus additionalProperties:false, which easily hard-errors on optional fields; schema conformance is
            backstopped by harness.structured's validation plus retry).
        json_object: send only {"type":"json_object"} (guarantees valid JSON; the schema is injected via the
            prompt, see harness.structured).
        """
        if self.structured_output == "json_schema":
            return {"type": "json_schema",
                    "json_schema": {"name": schema.get("title") or "output", "schema": schema}}
        if self.structured_output == "json_object":
            return {"type": "json_object"}
        return None

    async def chat(self, messages, *, temperature=None, max_tokens=None, **kwargs) -> LLMResponse:
        """Call an OpenAI-compatible service (await the request via AsyncOpenAI) and translate the response into a unified LLMResponse.
        Network errors and response-structure anomalies are both normalized to LLMError. See BaseAdapter.chat for the parameters."""
        client = self._ensure_client()
        params = self._params(messages, temperature, max_tokens, stream=False, **kwargs)
        start = time.perf_counter()
        try:
            resp = await client.chat.completions.create(**params)
        except Exception as e:  # noqa: BLE001
            raise _request_error(self.provider or "openai", self.model, e) from e   # Mark the real provider (deepseek etc.), falling back to the protocol name.
        return self._parse_chat(resp, self._elapsed_ms(start))

    def _parse_chat(self, resp, latency: int) -> LLMResponse:
        """Translate a /chat/completions response into a unified LLMResponse.

        Defensive: a network success with an anomalous structure (content filtering / empty candidates) is also
        wrapped as LLMError rather than raising a bare IndexError.
        """
        try:
            if not getattr(resp, "choices", None):
                raise LLMResponseError("Response choices is empty (possibly content filtering or a service anomaly)")
            choice = resp.choices[0]
            msg = choice.message
            usage = resp.usage.model_dump() if getattr(resp, "usage", None) else None
            reasoning = getattr(msg, "reasoning_content", None)  # Present only on deepseek-reasoner / o1.
            # function-calling: convert the SDK's tool_calls into the standard OpenAI feed-back format (keep
            # only id/type/function, drop incremental fields like index; this feeds back more reliably across
            # providers); None when no tools were passed.
            raw_calls = getattr(msg, "tool_calls", None)
            tool_calls = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in raw_calls
            ] if raw_calls else None
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise LLMResponseError(f"Failed to parse OpenAI response (model={self.model}): {e}") from e
        return LLMResponse(
            content=msg.content or "", finish_reason=choice.finish_reason,
            model=getattr(resp, "model", self.model), usage=usage,
            reasoning_content=reasoning, tool_calls=tool_calls, latency_ms=latency, raw=resp,
        )

    async def stream(self, messages, *, temperature=None, max_tokens=None, on_stats=None, **kwargs):
        """Stream from an OpenAI-compatible service (AsyncOpenAI's async for), yielding text piece by piece; on completion write
        model / finish_reason / usage / elapsed into self.last_stream_stats. See BaseAdapter.stream for the parameters.

        With tools passed, the stream additionally yields exactly one final LLMResponse after the text drains
        (content = joined text, tool_calls = accumulated calls or None) so a streaming tool loop can consume
        text deltas live and still receive the complete turn. Without tools the stream stays str-only.
        """
        client = self._ensure_client()
        params = self._params(messages, temperature, max_tokens, stream=True, **kwargs)
        start = time.perf_counter()
        st = _StreamState(self.model)
        try:
            # async with: when the consumer breaks early or an error occurs mid-stream, __aexit__ closes the underlying HTTP stream (no leaked connection, no half-open socket).
            async with await client.chat.completions.create(**params) as resp_stream:
                async for chunk in resp_stream:
                    piece = st.feed(chunk)                  # Parse the chunk, update stats, return the text to yield.
                    if piece:
                        yield piece
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise _request_error(self.provider or "openai", self.model, e) from e
        finally:
            stats = st.stats(self._elapsed_ms(start))
            self.last_stream_stats = stats                  # Kept as a convenience attribute (unreliable under concurrency; use on_stats per call).
            if on_stats:
                on_stats(stats)                             # Hand back stats per call: concurrent streams on a shared client no longer overwrite each other.
        if params.get("tools"):
            # Terminal turn response for the streaming tool loop (after finally so stats are already recorded).
            yield LLMResponse(content="".join(st.text_parts), finish_reason=st.finish_reason,
                              model=st.model, usage=st.usage, tool_calls=st.final_tool_calls(),
                              latency_ms=self._elapsed_ms(start))
