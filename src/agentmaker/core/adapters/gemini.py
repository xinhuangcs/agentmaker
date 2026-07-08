"""agentmaker.core.adapters.gemini: Google Gemini protocol adapter."""

import base64
import json
import time
from typing import Any, List, Optional

from ..exceptions import LLMConfigError, LLMError, LLMResponseError
from ..llm_response import LLMResponse
from ..multimodal import content_text
from .base import BaseAdapter, _BaseStreamState, _request_error


def _to_gemini_parts(content: list, types) -> list:
    """Translate neutral multimodal content parts into Gemini Parts (verified against the
    official image-understanding guide: inline images go through Part.from_bytes).

    Raises:
        LLMConfigError: On a URL image part. The Gemini generateContent API takes inline
            bytes or Files-API references, not arbitrary remote URLs: download the image
            and pass image_part_from_bytes/file, or upload via the Files API yourself.
    """
    parts = []
    for p in content:
        kind = p.get("type") if isinstance(p, dict) else None
        if kind == "text":
            parts.append(types.Part(text=p.get("text", "")))
        elif kind == "image":
            if p.get("url"):
                raise LLMConfigError(
                    "The Gemini adapter does not support URL image parts (the API takes inline "
                    "bytes or Files-API references): download the image and pass "
                    "image_part_from_bytes/image_part_from_file instead.")
            parts.append(types.Part.from_bytes(data=base64.b64decode(p["data"]),
                                               mime_type=p["media_type"]))
        else:
            raise LLMResponseError(f"Unknown content part type: {kind!r} (supported: text / image)")
    return parts or [types.Part(text="")]

# Default retry count (attempts = retries + 1, including the first request): setting 3 = 2 retries,
# matching the openai / anthropic SDK default of max_retries=2. google-genai does not retry by default
# (zero retries unless retry_options is configured), so we supply an equivalent default here to erase
# the resilience gap of "switching to Gemini becomes more fragile".
_DEFAULT_RETRY_ATTEMPTS = 3


class GeminiAdapter(BaseAdapter):
    """Google Gemini generateContent protocol. Requires `pip install google-genai`.

    Note: no key on this machine, not verified against the live API; the google-genai SDK is still
    evolving, so verify with a real key on first use.
    """

    def _ensure_client(self):
        """Lazily create the genai client (async goes through its .aio sub-namespace), cached per event loop (client.aio's underlying connection pool binds to the first loop it is used on).

        Raises an LLMError with install guidance when the SDK is not installed. timeout / base_url are
        sent via HttpOptions (see _http_options_kwargs); an equivalent default retry (retry_options) is
        also configured: google-genai does not retry by default, and adding it aligns with the default
        backoff-retry of openai / anthropic.
        """
        def make():
            try:
                from google import genai
                from google.genai.types import HttpOptions, HttpRetryOptions
            except ImportError as e:
                raise LLMConfigError("Gemini requires installation first: pip install google-genai (or uv sync --extra gemini)") from e
            # The items beyond attempts (initial_delay 1s / exp_base 2 / jitter / http_status_codes 408, 429, 5xx)
            # use the SDK's documented defaults, i.e. exponential backoff that only retries retryable status codes,
            # matching the default behavior of the other two SDKs, so we pin only attempts to align the retry count.
            retry_options = HttpRetryOptions(attempts=_DEFAULT_RETRY_ATTEMPTS)
            http_options = HttpOptions(retry_options=retry_options,
                                       **self._http_options_kwargs(self.timeout, self.base_url))
            return genai.Client(api_key=self.api_key, http_options=http_options)
        return self._async_client_for_loop(make)

    @staticmethod
    def _http_options_kwargs(timeout, base_url) -> dict:
        """Build the arguments for genai HttpOptions: timeout is converted from seconds to milliseconds (genai's timeout unit is milliseconds), base_url is passed through, and items that are None are omitted (letting the SDK use its default). Kept as a pure function to ease hermetic assertions.

        Args:
            timeout: Timeout in seconds or None.
            base_url: Service address or None.

        Returns:
            dict: The dict passed to HttpOptions(**kwargs).
        """
        kwargs: dict = {}
        if timeout is not None:
            kwargs["timeout"] = int(timeout * 1000)   # seconds -> milliseconds
        if base_url:
            kwargs["base_url"] = base_url
        return kwargs

    @staticmethod
    def _tools_to_gemini(tools):
        """Translate OpenAI tool schemas into Gemini's Tool(function_declarations=...).

        Args:
            tools: OpenAI-format tool list.

        Returns:
            list: A list containing a single types.Tool whose function_declarations are the tool declarations.
        """
        from google.genai import types
        return [types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t["function"]["name"],
                                      description=t["function"].get("description", ""),
                                      parameters=t["function"].get("parameters"))
            for t in tools])]

    @staticmethod
    def _to_gemini(messages):
        """Translate unified messages (OpenAI style, including function-calling) into Gemini's (system_instruction, contents).

        system -> system_instruction; assistant -> "model", user -> "user"; assistant tool_calls ->
        function_call part; OpenAI tool-role messages -> function_response part.

        Parallel calls: multiple tool results from the same turn must be merged into the multiple
        function_response parts of one user message (required by the docs, in the same order as the
        function_call), so we accumulate them in pending_results and flush into one message on the
        next non-tool message.

        Matching results back to calls: in Gemini 3 every function_call carries a unique id, matched by
        id (function_call and function_response both carry the same id, the recommended approach); Gemini
        2.x has no id and is matched by order. Calls with an id go into an id->name map; calls without an
        id go into a FIFO queue (tool results are in the same order as calls, so names are taken one by
        one). Never use None as a dict key (multiple id-less parallel calls would overwrite each other and
        match the wrong name).

        Thought signature: Gemini 3 thinking models require the thought_signature on a function_call part
        to be sent back verbatim (mandatory for function-calling, otherwise the next turn 400s), so chat
        parsing stores it into tool_calls and we put it back when rebuilding the part here.

        Args:
            messages: Unified message list.

        Returns:
            tuple[Optional[str], list]: (system text or None, Gemini Content list).
        """
        from google.genai import types
        system_parts, contents = [], []
        id_to_name: dict = {}  # tool_call_id -> function name (calls with an id are matched by id; functionResponse needs the name, not the id)
        noid_names: List[str] = []  # FIFO queue of function names for id-less calls (Gemini 2.x parallel calls carry no id, matched by order, avoiding a None key collision)
        pending_results: List[Any] = []  # accumulate consecutive tool results into one user message (multiple results of a parallel call belong to one message)

        def flush():
            if pending_results:
                contents.append(types.Content(role="user", parts=list(pending_results)))
                pending_results.clear()

        for m in messages:
            role = m.get("role")
            content = m.get("content", "") or ""
            if role == "system":
                if content:
                    system_parts.append(content_text(content))   # system stays text-only; flatten defensively
                continue
            if role == "tool":  # OpenAI tool result -> Gemini function_response (accumulate, merge on flush)
                cid = m.get("tool_call_id")
                if cid:  # with an id (Gemini 3 / parallel), match the name by id and send the id back
                    name = id_to_name.get(cid, "")
                else:    # without an id (2.x), take the name from the queue by order (tool results are in the same order as calls, pop one by one)
                    name = noid_names.pop(0) if noid_names else ""
                fr_kwargs = {"name": name, "response": {"result": content}}
                if cid:
                    fr_kwargs["id"] = cid
                pending_results.append(types.Part(function_response=types.FunctionResponse(**fr_kwargs)))
                continue
            flush()  # on a non-tool message, first land the accumulated tool results as one user message
            if role == "assistant" and m.get("tool_calls"):  # model-initiated tool call -> function_call part
                parts = [types.Part(text=content)] if content else []
                for tc in m["tool_calls"]:
                    cid = tc["id"]
                    fname = tc["function"]["name"]
                    if cid:  # with an id go into the map; without an id go into the FIFO queue (do not use None as a key, multiple would overwrite each other)
                        id_to_name[cid] = fname
                    else:
                        noid_names.append(fname)
                    fc_kwargs = {"name": fname,
                                 "args": json.loads(tc["function"]["arguments"] or "{}")}
                    if cid:  # symmetric with function_response: with an id, both sides carry it, so the pairing stays consistent
                        fc_kwargs["id"] = cid
                    part_kwargs = {"function_call": types.FunctionCall(**fc_kwargs)}
                    sig = tc.get("thought_signature")
                    if sig is not None:  # Gemini 3: put the model's verbatim thought signature back (mandatory for function-calling);
                        # it was base64-encoded to a str when stored into tool_calls (see _parse_response), so decode it back to the original bytes here.
                        part_kwargs["thought_signature"] = base64.b64decode(sig) if isinstance(sig, str) else sig
                    parts.append(types.Part(**part_kwargs))
                contents.append(types.Content(role="model", parts=parts))
                continue
            g_role = "model" if role == "assistant" else "user"
            if isinstance(content, list):   # neutral multimodal parts -> Gemini Parts (text / inline image)
                contents.append(types.Content(role=g_role, parts=_to_gemini_parts(content, types)))
            else:
                contents.append(types.Content(role=g_role, parts=[types.Part(text=content)]))
        flush()  # when messages end with a tool result, land the final user message
        return ("\n\n".join(system_parts) or None), contents

    def _build_config(self, system, temperature, max_tokens, tools=None, *, output_schema=None):
        """Assemble the GenerateContentConfig for Gemini generate_content (system prompt, temperature, max tokens, optional tools, structured output).

        Args:
            system: System prompt text or None.
            temperature: Sampling temperature; None uses the default.
            max_tokens: Max output tokens, optional.
            tools: Tool list already translated into Gemini format, optional.
            output_schema: Optional JSON Schema (dict); when given, requires the model to output accordingly.

        Returns:
            types.GenerateContentConfig: The config object.
        """
        from google.genai import types
        cfg: dict = {}
        temp = self._temperature(temperature)
        if temp is not None:        # None = do not send temperature by default (see BaseAdapter._temperature)
            cfg["temperature"] = temp
        if system:
            cfg["system_instruction"] = system
        if max_tokens is not None:
            cfg["max_output_tokens"] = max_tokens
        if tools:
            cfg["tools"] = tools
        if output_schema is not None:
            # Gemini native structured output (verified against the real fields of installed google-genai
            # 2.7.0, not the docs' latest response_format): response_mime_type=application/json +
            # response_json_schema (which accepts a standard JSON Schema dict).
            cfg["response_mime_type"] = "application/json"
            cfg["response_json_schema"] = output_schema
        return types.GenerateContentConfig(**cfg)

    async def chat(self, messages, *, temperature=None, max_tokens=None, **kwargs) -> LLMResponse:
        """Call Gemini generateContent (client.aio await), translate the candidates/parts response into a unified LLMResponse; exceptions normalize to LLMError."""
        client = self._ensure_client()  # the genai client's .aio sub-namespace provides async (one per loop)
        contents, config = self._prep(messages, temperature, max_tokens, kwargs)
        start = time.perf_counter()
        try:
            resp = await client.aio.models.generate_content(model=self.model, contents=contents, config=config)
        except Exception as e:  # noqa: BLE001
            raise _request_error("gemini", self.model, e) from e
        return self._parse_response(resp, self._elapsed_ms(start))

    def _prep(self, messages, temperature, max_tokens, kwargs):
        """Assemble (contents, config), intercepting and translating tools plus the structured-output schema; shared by chat and stream."""
        tools = kwargs.pop("tools", None)  # intercept tools: OpenAI schema must be translated to Gemini format, not passed through as-is
        output_schema = kwargs.pop("output_schema", None)  # intercept structured-output schema
        # The whole request translation (messages + tools + output_schema) is wrapped in one try: bad JSON /
        # bad tool schema all normalize to LLMError, no bare exceptions leak through.
        try:
            system, contents = self._to_gemini(messages)  # _to_gemini's json.loads may raise
            config = self._build_config(system, temperature, max_tokens,
                                        self._tools_to_gemini(tools) if tools else None,  # a KeyError from a bad tool schema normalizes too
                                        output_schema=output_schema)
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise LLMResponseError(f"Failed to assemble Gemini request (model={self.model}): {e}") from e
        return contents, config

    def _parse_response(self, resp, latency: int) -> LLMResponse:
        """Translate a Gemini response into a unified LLMResponse."""
        try:
            # During content filtering, candidates may be empty or content may be None, so guard against it.
            if not getattr(resp, "candidates", None):
                raise LLMResponseError("Gemini returned no candidates (may have been blocked by the safety policy)")
            candidate = resp.candidates[0]
            parts = getattr(getattr(candidate, "content", None), "parts", None) or []
            text_parts = []
            tool_calls = []
            for part in parts:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                fc = getattr(part, "function_call", None)
                if fc:  # Gemini function_call (args is an object) -> OpenAI tool_calls (arguments is a JSON string)
                    # With an id (parallel call) use the id; without one (single call) use None: do not fabricate an id, pair by function name when sending back
                    call = {"id": getattr(fc, "id", None), "type": "function",
                            "function": {"name": fc.name,
                                         "arguments": json.dumps(dict(fc.args or {}), ensure_ascii=False)}}
                    sig = getattr(part, "thought_signature", None)
                    if sig is not None:  # Gemini 3: the thought signature must be sent back verbatim with the functionCall, otherwise the next turn 400s.
                        # It is bytes, but tool_calls must stay JSON-safe end to end (reducer token estimation / trace / checkpoint persistence all json.dumps)
                        # -> base64-encode to a str for storage; _to_gemini decodes it back to the original bytes when sending back (byte-for-byte verbatim, meeting the official requirement).
                        call["thought_signature"] = (base64.b64encode(sig).decode("ascii")
                                                     if isinstance(sig, (bytes, bytearray)) else sig)
                    tool_calls.append(call)
            um = getattr(resp, "usage_metadata", None)
            usage = _gemini_usage(um) if um else None
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise LLMResponseError(f"Failed to parse Gemini response (model={self.model}): {e}") from e
        return LLMResponse(
            content="".join(text_parts),
            finish_reason=_finish_reason(getattr(candidate, "finish_reason", None)),  # normalized to lowercase, consistent with the others, truncation observable
            model=self.model, usage=usage, tool_calls=tool_calls or None, latency_ms=latency, raw=resp,
        )

    async def stream(self, messages, *, temperature=None, max_tokens=None, on_stats=None, **kwargs):
        """Stream from Gemini (client.aio: async for), yielding text piece by piece.

        Takes usage_metadata / finish_reason from the final chunk and writes them into last_stream_stats.
        With tools passed, the stream additionally yields exactly one final LLMResponse after the text
        drains (content = joined text, tool_calls = collected function calls or None).
        """
        client = self._ensure_client()
        contents, config = self._prep(messages, temperature, max_tokens, kwargs)
        start = time.perf_counter()
        st = _GemStreamState(self.model)
        try:
            gen = await client.aio.models.generate_content_stream(
                model=self.model, contents=contents, config=config)
            try:
                async for chunk in gen:
                    piece = st.feed(chunk)
                    if piece:
                        yield piece
            finally:
                aclose = getattr(gen, "aclose", None)       # close the underlying stream whether the consumer breaks early or an exception occurs mid-stream (do not leak the connection)
                if aclose is not None:
                    await aclose()
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise _request_error("gemini", self.model, e) from e
        finally:
            stats = st.stats(self._elapsed_ms(start))
            self.last_stream_stats = stats
            if on_stats:
                on_stats(stats)
        if getattr(config, "tools", None):
            # Terminal turn response for the streaming tool loop.
            yield LLMResponse(content="".join(st.text_parts), finish_reason=st.finish_reason,
                              model=self.model, usage=st.usage, tool_calls=st.tool_calls or None,
                              latency_ms=self._elapsed_ms(start))



def _gemini_usage(um) -> dict:
    """Translate Gemini's usage_metadata into a unified {prompt/completion/total_tokens} (shared by non-streaming _parse_response and streaming feed, avoiding convention drift)."""
    return {"prompt_tokens": getattr(um, "prompt_token_count", 0),
            "completion_tokens": getattr(um, "candidates_token_count", 0),
            "total_tokens": getattr(um, "total_token_count", 0)}


def _finish_reason(fr) -> Optional[str]:
    """Normalize Gemini's finish_reason into a lowercase token, consistent with OpenAI (length) / Anthropic (max_tokens).

    Gemini's finish_reason is a `FinishReason` enum, and calling `str()` directly prints
    'FinishReason.MAX_TOKENS', which both fails to match the others and makes truncation observability
    (the harness's `_TRUNCATION_REASONS`) miss Gemini's length truncation. Take the enum member name and
    lowercase it -> 'max_tokens' (matches the set); for non-enums (e.g. already a string) fall back to
    lowercasing. None is returned as-is.
    """
    if fr is None:
        return None
    return getattr(fr, "name", str(fr)).lower() or None



class _GemStreamState(_BaseStreamState):
    """Gemini streaming per-chunk parsing (overrides feed; adds function-call collection on top of the base class).

    In the classic generateContent protocol a Part carries a complete functionCall object (the wire schema
    has no partial-arguments field, unlike OpenAI fragments / Anthropic input_json_delta), so streamed
    function calls are collected whole per chunk. Tool-call bearing streams have NOT been verified against
    the live API on this machine (no key) -- same caveat as the adapter itself.
    """

    def __init__(self, model):
        super().__init__(model)
        self.text_parts: list = []      # every text delta, joined into LLMResponse.content at the end
        self.tool_calls: list = []      # unified feed-back format, same conversion as _parse_response

    def feed(self, chunk) -> str:
        """Feed one streaming chunk: update usage / finish_reason, collect function-call parts, return this chunk's text delta (empty string if none)."""
        um = getattr(chunk, "usage_metadata", None)  # the final chunk usually carries usage
        if um:
            self.usage = _gemini_usage(um)
        cands = getattr(chunk, "candidates", None)
        if cands and getattr(cands[0], "finish_reason", None):
            self.finish_reason = _finish_reason(cands[0].finish_reason)
        content = getattr(cands[0], "content", None) if cands else None
        for part in (getattr(content, "parts", None) or []):
            fc = getattr(part, "function_call", None)
            if fc:  # same Gemini function_call -> OpenAI tool_calls conversion as _parse_response (incl. thought_signature base64 round-trip)
                call = {"id": getattr(fc, "id", None), "type": "function",
                        "function": {"name": fc.name,
                                     "arguments": json.dumps(dict(fc.args or {}), ensure_ascii=False)}}
                sig = getattr(part, "thought_signature", None)
                if sig is not None:
                    call["thought_signature"] = (base64.b64encode(sig).decode("ascii")
                                                 if isinstance(sig, (bytes, bytearray)) else sig)
                self.tool_calls.append(call)
        piece = getattr(chunk, "text", None) or ""
        if piece:
            self.text_parts.append(piece)
        return piece
