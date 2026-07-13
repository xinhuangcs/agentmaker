"""agentmaker.core.adapters.anthropic: Anthropic (Claude) protocol adapter.

_close_objects is used only by Anthropic structured output (which requires every object
schema to set additionalProperties:false; OpenAI does not enable strict mode), so it lives here.
"""

import json
import time
from collections.abc import AsyncIterator
from typing import List, Optional

from ..exceptions import LLMConfigError, LLMError, LLMResponseError
from ..llm_response import LLMResponse, StreamStats
from ..multimodal import content_text
from .base import BaseAdapter, _request_error


def _to_anthropic_block(part: dict) -> dict:
    """Translate one neutral content part into an Anthropic content block."""
    kind = part.get("type")
    if kind == "text":
        return {"type": "text", "text": part.get("text", "")}
    if kind == "image":
        if part.get("url"):
            return {"type": "image", "source": {"type": "url", "url": part["url"]}}
        return {"type": "image", "source": {"type": "base64", "media_type": part["media_type"],
                                            "data": part["data"]}}
    raise LLMResponseError(f"Unknown content part type: {kind!r} (supported: text / image)")


def _block_dict(block) -> dict:
    """Return a JSON-safe copy of an Anthropic response block."""
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        try:
            raw = dump(mode="json", exclude_none=True)
        except TypeError:
            raw = dump(exclude_none=True)
        if not isinstance(raw, dict):
            raise LLMResponseError("Anthropic response block did not serialize to an object")
        return raw
    return {key: value for key, value in vars(block).items()
            if not key.startswith("_") and not callable(value)}


# JSON Schema keywords whose value is a mapping container of sub-schemas: the keys are user field
# names (which may happen to be "properties") and only the values are sub-schemas.
_SCHEMA_MAP_KEYWORDS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
# Keywords whose value is a list of sub-schemas.
_SCHEMA_LIST_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})


def _close_objects(schema):
    """Recursively set `additionalProperties=false` on every schema object in a JSON Schema, returning a new dict without mutating the original.

    Anthropic structured output strictly requires this on every object (per the official docs);
    OpenAI strict mode needs it too. Descent follows the schema position of each keyword: for
    mapping containers like properties / $defs the keys are user field names (which may happen to
    be "properties"), so we recurse only into their values (the sub-schemas) and never treat the
    container itself as a schema to add additionalProperties to. Otherwise a user field literally
    named "properties" would be mistaken for the properties container and pollute the schema.
    $ref (a string) is preserved as-is.
    """
    if isinstance(schema, list):
        return [_close_objects(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in _SCHEMA_MAP_KEYWORDS and isinstance(v, dict):
            out[k] = {name: _close_objects(sub) for name, sub in v.items()}   # values are sub-schemas; keys are field names (preserved as-is)
        elif k in _SCHEMA_LIST_KEYWORDS and isinstance(v, list):
            out[k] = [_close_objects(x) for x in v]
        else:
            out[k] = _close_objects(v)
    if out.get("type") == "object" or "properties" in out or "patternProperties" in out:
        out.setdefault("additionalProperties", False)
    return out



class AnthropicAdapter(BaseAdapter):
    """Anthropic (Claude) `POST /v1/messages` protocol. Requires `pip install anthropic`.

    Note: no key on this machine, not verified against the live API; verify with a real key on first use.
    """

    DEFAULT_MAX_TOKENS = 4096  # Anthropic requires max_tokens; fall back to this when the caller omits it

    def _ensure_client(self):
        """Lazily create the async anthropic client (AsyncAnthropic), cached per event loop (see BaseAdapter._async_client_for_loop).

        Raises an LLMError with install guidance when the SDK is not installed.
        """
        def make():
            try:
                import anthropic
            except ImportError as e:
                raise LLMConfigError("Anthropic requires installation first: pip install anthropic (or uv sync --extra anthropic)") from e
            kw = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kw["base_url"] = self.base_url
            return anthropic.AsyncAnthropic(**kw)
        return self._async_client_for_loop(make)

    @staticmethod
    def _tools_to_anthropic(tools):
        """Translate OpenAI tool schemas into Anthropic format (function.parameters -> input_schema).

        Args:
            tools: OpenAI-format tool list, each {"type":"function","function":{name,description,parameters}}.

        Returns:
            list: Anthropic-format tool list, each {name, description, input_schema}.
        """
        return [{"name": t["function"]["name"],
                 "description": t["function"].get("description", ""),
                 "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}})}
                for t in tools]

    @staticmethod
    def _to_anthropic(messages):
        """Translate unified messages (OpenAI style, including function-calling tool_calls / tool role) into Anthropic format.

        The system message is lifted into a top-level field; assistant tool_calls become tool_use
        blocks; OpenAI tool-role messages are merged into a user message's tool_result blocks
        (Anthropic requires tool results to be sent with the user role, with multiple results in the
        same turn merged together).

        Args:
            messages: Unified message list.

        Returns:
            tuple[Optional[str], list]: (system text or None, Anthropic messages list).

        Example:
            _to_anthropic([{"role": "system", "content": "x"}, {"role": "user", "content": "hi"}])
            -> ("x", [{"role": "user", "content": "hi"}])
        """
        system_parts: List[str] = []
        out: List[dict] = []
        pending_results: List[dict] = []  # accumulate consecutive tool results and merge into one user message

        def flush():
            if pending_results:
                out.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for m in messages:
            role = m.get("role")
            if role == "system":
                if m.get("content"):
                    system_parts.append(content_text(m["content"]))   # system stays text-only; flatten defensively
                continue
            if role == "tool":  # OpenAI tool result -> Anthropic tool_result block
                pending_results.append({"type": "tool_result", "tool_use_id": m.get("tool_call_id"),
                                        "content": m.get("content", "") or ""})
                continue
            flush()  # on a non-tool message, first land the accumulated tool results as one user message
            if role == "assistant" and m.get("tool_calls"):  # model-initiated tool call -> tool_use block
                saved = (m.get("_provider_payload") or {}).get("anthropic_content")
                if isinstance(saved, list):
                    calls = m["tool_calls"]
                    blocks = []
                    for saved_block in saved:
                        block = dict(saved_block)
                        index = block.pop("_tool_call_index", None)
                        if index is not None:
                            tc = calls[index]
                            block.update({"type": "tool_use", "id": tc["id"],
                                          "name": tc["function"]["name"],
                                          "input": json.loads(tc["function"]["arguments"] or "{}")})
                        blocks.append(block)
                    out.append({"role": "assistant", "content": blocks})
                    continue
                blocks: List[dict] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"],
                                   "input": json.loads(tc["function"]["arguments"] or "{}")})
                out.append({"role": "assistant", "content": blocks})
            else:
                content = m.get("content", "") or ""
                if isinstance(content, list):   # neutral multimodal parts -> Anthropic content blocks
                    content = [_to_anthropic_block(p) for p in content]
                out.append({"role": role if role in ("user", "assistant") else "user",
                            "content": content})
        flush()
        return ("\n\n".join(system_parts) or None), out

    def _params(self, messages, temperature, max_tokens, **kwargs):
        """Assemble the request parameters for Anthropic messages.create (including system extraction and max_tokens fallback).

        Args:
            messages: Unified message list.
            temperature: Sampling temperature; None uses the default.
            max_tokens: Max tokens; required by Anthropic, uses DEFAULT_MAX_TOKENS when None.
            **kwargs: Anthropic-specific parameters passed through.

        Returns:
            dict: Parameter dict ready to expand into the SDK call.
        """
        system, convo = self._to_anthropic(messages)
        params = {
            "model": self.model, "messages": convo,
            "max_tokens": max_tokens or self.DEFAULT_MAX_TOKENS,
        }
        temp = self._temperature(temperature)
        if temp is not None:        # None = do not send temperature by default (see BaseAdapter._temperature)
            params["temperature"] = temp
        if system:
            # Set system as a block with cache_control -> Anthropic caches the tools+system prefix
            # (cache order is tools -> system -> messages; marking system also covers the tools before it).
            # Below the model's minimum cache threshold it is silently ignored without error, so we keep it
            # on by default: it only saves cost/latency and does not change model output. Cache-hit price is
            # 0.1x and a 5-minute write is 1.25x (OpenAI/DeepSeek cache automatically; Gemini needs an explicit
            # cache object -> left to the app).
            params["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        params.update(kwargs)
        return params

    def _chat_params(self, messages, max_tokens, temperature, kwargs):
        """Assemble messages.create parameters (intercepting and translating tools, plus structured-output output_config); shared by chat and stream."""
        tools = kwargs.pop("tools", None)  # intercept tools: OpenAI schema must be translated to Anthropic format, not passed through as-is
        output_schema = kwargs.pop("output_schema", None)  # intercept structured-output schema
        # The whole request translation (messages + tools + output_schema) is wrapped in one try: bad JSON /
        # bad tool schema all normalize to LLMError, no bare exceptions leak through.
        try:
            params = self._params(messages, temperature, max_tokens, **kwargs)  # _to_anthropic's json.loads may raise
            if tools:
                params["tools"] = self._tools_to_anthropic(tools)  # a KeyError from a bad tool schema normalizes too
            if output_schema is not None:
                # Anthropic native structured output (GA, verified against anthropic 0.105.2's
                # OutputConfigParam/JSONOutputFormatParam): output_config.format. The docs require every
                # object schema to set additionalProperties:false (_close_objects).
                params["output_config"] = {"format": {"type": "json_schema", "schema": _close_objects(output_schema)}}
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise LLMResponseError(f"Failed to assemble Anthropic request (model={self.model}): {e}") from e
        return params

    async def chat(self, messages, *, temperature=None, max_tokens=None, **kwargs) -> LLMResponse:
        """Call the Claude Messages API (AsyncAnthropic await), translate the block-array response into a unified LLMResponse; exceptions normalize to LLMError."""
        client = self._ensure_client()
        params = self._chat_params(messages, max_tokens, temperature, kwargs)
        start = time.perf_counter()
        try:
            resp = await client.messages.create(**params)
        except Exception as e:  # noqa: BLE001
            raise _request_error("anthropic", self.model, e) from e
        return self._parse_message(resp, self._elapsed_ms(start))

    def _parse_message(self, resp, latency: int) -> LLMResponse:
        """Translate a Claude Messages response into a unified LLMResponse."""
        try:
            # Key difference: the response has no choices; content is a block array, with text in text
            # blocks and tool calls in tool_use blocks.
            text_parts = []
            tool_calls = []
            provider_blocks = []
            for block in (getattr(resp, "content", None) or []):
                btype = getattr(block, "type", None)
                saved_block = _block_dict(block)
                if btype == "text":
                    text_parts.append(block.text)
                elif btype == "tool_use":  # Anthropic tool_use (input is an object) -> OpenAI tool_calls (arguments is a JSON string)
                    saved_block["_tool_call_index"] = len(tool_calls)
                    tool_calls.append({"id": block.id, "type": "function",
                                       "function": {"name": block.name,
                                                    "arguments": json.dumps(block.input, ensure_ascii=False)}})
                provider_blocks.append(saved_block)
            usage = self._usage(getattr(resp, "usage", None))  # includes cached-token add-back, see _usage
        except Exception as e:  # noqa: BLE001
            raise LLMResponseError(f"Failed to parse Anthropic response (model={self.model}): {e}") from e
        return LLMResponse(
            content="".join(text_parts),
            finish_reason=getattr(resp, "stop_reason", None),  # Anthropic calls it stop_reason
            model=getattr(resp, "model", None) or self.model, usage=usage,
            tool_calls=tool_calls or None,
            assistant_message={"_provider_payload": {"anthropic_content": provider_blocks}},
            latency_ms=latency, raw=resp,
        )

    async def stream(self, messages, *, temperature=None, max_tokens=None, on_stats=None,
                     **kwargs) -> AsyncIterator[str | LLMResponse]:
        """Stream from Claude (AsyncAnthropic: async with + async for s.text_stream), yielding text piece by piece.

        After completion, get_final_message() supplies usage / finish reason written into last_stream_stats.
        With tools passed, the stream additionally yields exactly one final LLMResponse after the text drains
        (the SDK's stream accumulator already assembles tool_use blocks from input_json_delta events, so the
        final message parses through the same _parse_message as the chat path). Without tools a tool_use stop
        still fails loud (never silently dropped).
        """
        client = self._ensure_client()
        params = self._chat_params(messages, max_tokens, temperature, kwargs)  # same path as chat: intercept and translate tools / output_schema
        start = time.perf_counter()
        final = None
        try:
            async with client.messages.stream(**params) as s:
                async for piece in s.text_stream:
                    if piece:
                        yield piece
                final = await s.get_final_message()
            if not params.get("tools") and getattr(final, "stop_reason", None) == "tool_use":
                raise LLMError("Streaming produced a tool call but no tools were passed: pass tools to stream() or use chat().")
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise _request_error("anthropic", self.model, e) from e
        finally:
            stats = self._final_stats(final, self._elapsed_ms(start)) if final is not None else None
            self.last_stream_stats = stats
            if on_stats and stats is not None:
                on_stats(stats)
        if params.get("tools"):
            # Terminal turn response for the streaming tool loop (same unified parse as the chat path).
            yield self._parse_message(final, self._elapsed_ms(start))

    @staticmethod
    def _usage(u) -> Optional[dict]:
        """Translate Anthropic usage into a unified dict (shared by non-streaming _parse_message and streaming _final_stats).

        Anthropic's input_tokens does not include cached tokens; we add cache_read/creation back so
        prompt_tokens matches the OpenAI convention (which includes cache hits). When there is cache
        activity we attach cache_* details so trace can see the hit rate. Returns None when u is None.
        """
        if u is None:
            return None
        cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
        prompt = u.input_tokens + cache_read + cache_write
        usage = {"prompt_tokens": prompt, "completion_tokens": u.output_tokens,
                 "total_tokens": prompt + u.output_tokens}
        if cache_read or cache_write:
            usage["cache_read_input_tokens"] = cache_read
            usage["cache_creation_input_tokens"] = cache_write
        return usage

    def _final_stats(self, final, latency_ms: int) -> StreamStats:
        """Extract StreamStats from the streaming final message (used to wrap up stream)."""
        return StreamStats(model=getattr(final, "model", None) or self.model,
                           finish_reason=getattr(final, "stop_reason", None),
                           usage=self._usage(getattr(final, "usage", None)), latency_ms=latency_ms)
