"""Structural invariants of the _PROFILES provider table + LLMClient context-window resolution (hermetic).

Deliberately does not pin volatile model names (those rely on periodic review, see llm_clients.md §2.3); it asserts
only the structural contract -- every protocol has an adapter, the structured-output enum is valid, cloud profiles
carry the window/default model they should -- so a fat-fingered table edit is caught instead of silently drifting.
"""

import json
from types import SimpleNamespace

import pytest

from agentmaker.core.llm_clients import LLMClient, ModelInfo, _KNOWN_MODELS, _PROFILES
from agentmaker.core.adapters import _ADAPTERS, AnthropicAdapter, OpenAIAdapter, _StreamState

_VALID_STRUCTURED_OUTPUT = {"json_schema", "json_object", "none", "native"}
# Profiles where the model is user-chosen and the context window is unknown (local / self-hosted / proxy / multi-model platforms)
_NO_DEFAULT_MODEL = {"ollama", "vllm", "sglang", "modelscope", "openai_compatible"}


def test_every_protocol_has_adapter():
    for name, p in _PROFILES.items():
        assert p.protocol in _ADAPTERS, f"{name} protocol={p.protocol!r} has no adapter"


def test_register_adapter_extends_and_rejects_non_subclass():
    """register_adapter merges a third-party protocol->adapter into _ADAPTERS (must be a BaseAdapter subclass, else TypeError)."""
    from agentmaker.core.adapters import BaseAdapter, register_adapter
    try:
        class _StubAdapter(BaseAdapter):
            def _ensure_client(self): ...
            def chat(self, *a, **k): ...
            def stream(self, *a, **k): ...
        register_adapter("myproto", _StubAdapter)
        assert _ADAPTERS["myproto"] is _StubAdapter
        with pytest.raises(TypeError):
            register_adapter("bad", object)                 # not a BaseAdapter subclass
    finally:
        _ADAPTERS.pop("myproto", None)                       # cleanup so other tests aren't polluted


def test_structured_output_enum_valid():
    for name, p in _PROFILES.items():
        assert p.structured_output in _VALID_STRUCTURED_OUTPUT, f"{name} structured_output={p.structured_output!r} is invalid"


def test_supports_fc_is_bool():
    """Every profile's supports_function_calling must be a bool (a provider-level default, never None)."""
    for name, p in _PROFILES.items():
        assert isinstance(p.supports_function_calling, bool), f"{name} supports_function_calling must be a bool"


def test_key_envs_is_tuple():
    for name, p in _PROFILES.items():
        assert isinstance(p.key_envs, tuple), f"{name} key_envs must be a tuple (immutable to match frozen)"


def test_cloud_profiles_have_default_model_and_window():
    """Profiles with a fixed vendor identity must set both default_model and context_window; user-chosen-model profiles are explicitly exempt."""
    for name, p in _PROFILES.items():
        if name in _NO_DEFAULT_MODEL:
            assert p.default_model is None and p.context_window is None, f"{name} must not preset default_model/context_window"
        else:
            assert p.default_model, f"{name} is missing default_model"
            assert p.context_window, f"{name} set default_model but is missing context_window"


# ---------- _KNOWN_MODELS catalog structural invariants (again, guards shape not specific values) ----------

def test_known_models_structural():
    """Each catalog entry is a ModelInfo; context_window a positive int; max_output_tokens a positive int or None (blank when the vendor hasn't published it)."""
    for mid, info in _KNOWN_MODELS.items():
        assert isinstance(info, ModelInfo), f"{mid} is not a ModelInfo"
        assert isinstance(info.context_window, int) and info.context_window > 0, f"{mid} context_window must be a positive int"
        assert info.max_output_tokens is None or (isinstance(info.max_output_tokens, int) and info.max_output_tokens > 0), \
            f"{mid} max_output_tokens must be a positive int or None"
        assert info.supports_function_calling is None or isinstance(info.supports_function_calling, bool), \
            f"{mid} supports_function_calling must be a bool or None (None = follows provider)"


def test_known_models_never_shadow_a_default():
    """The catalog holds only non-default models; a key must never collide with any provider's default_model (which would make resolution ambiguous)."""
    defaults = {p.default_model for p in _PROFILES.values() if p.default_model}
    clash = defaults & set(_KNOWN_MODELS)
    assert not clash, f"catalog key collides with a default_model: {clash} (the default model belongs in the profile, not the catalog)"


def test_non_openai_model_limits():
    """Provider limits match the current first-party model contracts."""
    assert (_PROFILES["deepseek"].context_window, _PROFILES["deepseek"].max_output_tokens) == (1_000_000, 393_216)
    assert (_PROFILES["gemini"].context_window, _PROFILES["gemini"].max_output_tokens) == (1_048_576, 65_536)
    assert (_PROFILES["zhipu"].context_window, _PROFILES["zhipu"].max_output_tokens) == (204_800, 131_072)
    expected = {
        "gemini-3.5-flash": (1_048_576, 65_536),
        "deepseek-v4-pro": (1_000_000, 393_216),
        "claude-sonnet-5": (1_000_000, 128_000),
        "glm-5.2": (1_000_000, 131_072),
        "qwen3.6-flash": (1_000_000, 32_768),
        "kimi-k2.7-code": (262_144, None),   # Moonshot output is window minus prompt, no independent cap
        "moonshot-v1-128k": (131_072, None),
    }
    assert {model: (_KNOWN_MODELS[model].context_window, _KNOWN_MODELS[model].max_output_tokens)
            for model in expected} == expected


# ---------- context_window resolves per-model (the profile value only holds for default_model) ----------

@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")


def test_default_model_uses_profile_window():
    assert LLMClient("deepseek").context_window == _PROFILES["deepseek"].context_window


def test_overridden_model_window_is_unknown():
    assert LLMClient("deepseek", model="some-128k-model").context_window is None   # model swapped -> unknown; the default window is not inherited


def test_explicit_window_wins():
    assert LLMClient("deepseek", model="some-128k-model", context_window=128_000).context_window == 128_000


# ---------- supports_function_calling resolves as explicit > model-level override > provider-level default ----------

def test_default_supports_fc_follows_provider():
    assert LLMClient("deepseek").supports_function_calling is True   # provider-level default is True


def test_explicit_supports_fc_wins():
    assert LLMClient("deepseek", supports_function_calling=False).supports_function_calling is False


def test_known_model_overrides_supports_fc(monkeypatch):
    """A _KNOWN_MODELS model-level override beats the provider default (a temporary entry exercises the mechanism without pinning a real model)."""
    from agentmaker.core import llm_clients as M
    monkeypatch.setitem(M._KNOWN_MODELS, "fake-no-fc-model",
                        ModelInfo(context_window=8000, supports_function_calling=False))
    assert LLMClient("deepseek", model="fake-no-fc-model").supports_function_calling is False


# ---------- OpenAI adapter request shaping: streaming usage / reasoning-model temperature (hermetic, no network) ----------

def _openai_adapter(model, default_temperature=None):
    return OpenAIAdapter(model=model, api_key="x", base_url="http://x/v1", timeout=1,
                         default_temperature=default_temperature, max_tokens_field="max_tokens", structured_output="none")


class _Usage:
    def __init__(self, d): self._d = d
    def model_dump(self): return self._d


class _Choice:
    def __init__(self, content=None, finish_reason=None):
        self.delta = type("D", (), {"content": content})()
        self.finish_reason = finish_reason


class _Chunk:
    def __init__(self, *, choices=None, usage=None, model="m"):
        self.choices = choices or []
        self.usage = usage
        self.model = model


def test_stream_state_captures_usage_on_choice_bearing_chunk():
    """Captures usage when it rides on the final choices-bearing chunk."""
    st = _StreamState("m")
    assert st.feed(_Chunk(choices=[_Choice("你好")])) == "你好"
    st.feed(_Chunk(choices=[_Choice("", finish_reason="stop")], usage=_Usage({"total_tokens": 46})))
    stats = st.stats(1)
    assert stats.usage == {"total_tokens": 46} and stats.finish_reason == "stop"


def test_openai_chat_preserves_reasoning_and_tool_extensions():
    """Compatible-provider continuation fields survive normalized tool-call parsing."""
    class _ToolCall:
        id = "c1"
        type = "function"
        function = SimpleNamespace(name="lookup", arguments='{"x":1}')

        def model_dump(self, **kwargs):
            return {"id": self.id, "type": self.type,
                    "function": {"name": "lookup", "arguments": '{"x":1}'},
                    "extra_content": {"google": {"thought_signature": "sig"}}}

    message = SimpleNamespace(content="", reasoning_content="private-reasoning",
                              tool_calls=[_ToolCall()])
    response = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                               usage=None, model="m")
    parsed = _openai_adapter("m")._parse_chat(response, 1)
    assert parsed.assistant_message == {"reasoning_content": "private-reasoning"}
    assert parsed.tool_calls[0]["extra_content"]["google"]["thought_signature"] == "sig"
    outbound = {**parsed.assistant_message, "role": "assistant", "content": "",
                "tool_calls": parsed.tool_calls}
    assert _openai_adapter("m")._params([outbound], None, None, stream=False)["messages"][0] == outbound


def test_params_injects_stream_options_only_when_streaming():
    """OpenAI-protocol streaming requests usage statistics by default."""
    msgs = [{"role": "user", "content": "hi"}]
    assert _openai_adapter("gpt-4.1-nano")._params(msgs, None, None, stream=True)["stream_options"] == {"include_usage": True}
    assert "stream_options" not in _openai_adapter("gpt-4.1-nano")._params(msgs, None, None, stream=False)


# ---------- Temperature: never sent by default; sent only when explicit or a default is configured (the framework doesn't guess whether a model supports it) ----------

def _anthropic_adapter(model, default_temperature=None):
    return AnthropicAdapter(model=model, api_key="x", base_url=None, timeout=1,
                            default_temperature=default_temperature, max_tokens_field="max_tokens", structured_output="native")


def test_temperature_resolver_uniform():
    """BaseAdapter._temperature: an explicit value wins, else default_temperature (None = don't send). Explicit 0 stays distinct from None."""
    ad = _openai_adapter("any")                       # default_temperature=None
    assert ad._temperature(None) is None              # default: don't send
    assert ad._temperature(0.0) == 0.0                # explicit 0 is honored (!= None)
    assert ad._temperature(0.9) == 0.9
    assert _openai_adapter("any", default_temperature=0.2)._temperature(None) == 0.2  # a configured default is used


def test_temperature_not_sent_by_default():
    """Temperature is never sent by default; no model (plain / o-series / new reasoning flagship) gets one, so temperature-locked models won't 400."""
    msgs = [{"role": "user", "content": "hi"}]
    for m in ("gpt-4.1-nano", "o3-mini", "gpt-5.5"):
        assert "temperature" not in _openai_adapter(m)._params(msgs, None, None, stream=False)
    assert "temperature" not in _anthropic_adapter("claude-opus-4-8")._params(msgs, None, None)


def test_temperature_sent_when_explicit_or_configured():
    """An explicit temperature= or a construction-time default_temperature is sent through (the framework doesn't judge whether the model supports it)."""
    msgs = [{"role": "user", "content": "hi"}]
    assert _openai_adapter("gpt-5.5")._params(msgs, 0.7, None, stream=False)["temperature"] == 0.7       # explicit
    assert _anthropic_adapter("claude-opus-4-8")._params(msgs, 0.5, None)["temperature"] == 0.5
    assert _openai_adapter("gpt-4.1-nano", default_temperature=0.2)._params(msgs, None, None, stream=False)["temperature"] == 0.2  # configured default


# ---------- Gemini thought_signature bytes must be JSON-safe ----------

def _gemini_fc_response(sig):
    """Build a mock Gemini response carrying a function_call + thought_signature (no SDK dependency)."""
    fc = type("FC", (), {"id": "c1", "name": "calc", "args": {"x": 1}})()
    part = type("P", (), {"text": None, "function_call": fc, "thought_signature": sig})()
    cand = type("Cand", (), {"content": type("C", (), {"parts": [part]})(), "finish_reason": "STOP"})()
    return type("R", (), {"candidates": [cand], "usage_metadata": None})()


def _gemini_adapter():
    from agentmaker.core.adapters import GeminiAdapter
    return GeminiAdapter(model="gemini-x", api_key="x", base_url=None, timeout=1,
                         default_temperature=None, max_tokens_field="max_tokens", structured_output="native")


def test_gemini_thought_signature_parsed_json_safe():
    """_parse_response converts Gemini 3's bytes thought_signature to a base64 str so tool_calls stay json.dumps-able
    for reducer token estimates and checkpoint persistence (hermetic, no SDK)."""
    import json as _json
    out = _gemini_adapter()._parse_response(_gemini_fc_response(b"\x00\x01\xff sig"), 1)
    assert isinstance(out.tool_calls[0]["thought_signature"], str)
    _json.dumps(out.tool_calls)                                      # key point: serializable (reducer / checkpoint won't crash)


def test_gemini_thought_signature_roundtrip_decodes_to_bytes():
    """_to_gemini decodes the base64-str thought_signature back to raw bytes (Gemini requires a byte-exact echo). Requires google-genai, else skipped."""
    pytest.importorskip("google.genai")
    ad = _gemini_adapter()
    raw_sig = b"\x00\x01\x02\xff binary signature"
    out = ad._parse_response(_gemini_fc_response(raw_sig), 1)
    msg = {"role": "assistant", "content": "", "tool_calls": out.tool_calls}
    _system, contents = ad._to_gemini([msg])
    assert contents[0].parts[0].thought_signature == raw_sig         # byte-exact round-trip


# ---------- Gemini adapter fixes: http_options / error mapping / id-less parallel pairing ----------

def test_request_error_maps_genai_code():
    """genai's errors.APIError exposes .code (int), not .status_code; _request_error should fall back to .code and derive retryable from it."""
    from agentmaker.core.adapters.base import _request_error

    class _GenaiErr(Exception):                        # mimics genai APIError: has .code / .status, no .status_code
        def __init__(self, code, status):
            self.code = code
            self.status = status
            super().__init__(status)

    e429 = _request_error("gemini", "m", _GenaiErr(429, "RESOURCE_EXHAUSTED"))
    assert e429.status_code == 429 and e429.retryable is True        # rate-limited -> retryable
    assert _request_error("gemini", "m", _GenaiErr(503, "UNAVAILABLE")).retryable is True   # 5xx -> retryable
    e404 = _request_error("gemini", "m", _GenaiErr(404, "NOT_FOUND"))
    assert e404.status_code == 404 and e404.retryable is False       # 4xx (other than 408/429) -> not retryable


def test_request_error_ignores_non_int_code():
    """openai's .code is a string error code (e.g. invalid_api_key), not an HTTP code, so it must not be used as a status code."""
    from agentmaker.core.adapters.base import _request_error

    class _OpenAIStrCode(Exception):
        code = "invalid_api_key"                        # string, not int
        status_code = None

    err = _request_error("openai", "m", _OpenAIStrCode())
    assert err.status_code is None and err.retryable is False        # the string code is rejected by isinstance(int)


def test_gemini_http_options_kwargs_pure():
    """_http_options_kwargs: timeout seconds->milliseconds, base_url passed through, None entries omitted (pure function, no SDK needed)."""
    from agentmaker.core.adapters import GeminiAdapter
    assert GeminiAdapter._http_options_kwargs(5, "http://proxy/v1") == {"timeout": 5000, "base_url": "http://proxy/v1"}
    assert GeminiAdapter._http_options_kwargs(2.5, None) == {"timeout": 2500}
    assert GeminiAdapter._http_options_kwargs(None, None) == {}


def test_gemini_client_applies_http_options(monkeypatch):
    """Constructing genai.Client actually applies timeout (->ms) / base_url via HttpOptions. Requires google-genai, else skipped."""
    import asyncio
    genai = pytest.importorskip("google.genai")
    from agentmaker.core.adapters import GeminiAdapter

    captured: dict = {}
    monkeypatch.setattr(genai, "Client", lambda **kw: captured.update(kw) or object())
    ad = GeminiAdapter(model="gemini-x", api_key="k", base_url="http://proxy/v1", timeout=5,
                       default_temperature=None, max_tokens_field="max_tokens", structured_output="native")

    async def _touch():
        return ad._ensure_client()                       # _async_client_for_loop needs a running loop
    asyncio.run(_touch())
    ho = captured["http_options"]
    assert ho.timeout == 5000 and ho.base_url == "http://proxy/v1"


def test_gemini_noid_parallel_results_pair_by_order():
    """Two id-less parallel tool calls: results pair to function names by order (not collapsed into one). Requires google-genai, else skipped."""
    pytest.importorskip("google.genai")
    ad = _gemini_adapter()
    messages = [
        {"role": "user", "content": "查两地天气"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": None, "type": "function", "function": {"name": "weather_bj", "arguments": "{}"}},
            {"id": None, "type": "function", "function": {"name": "weather_sh", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": None, "content": "北京晴"},
        {"role": "tool", "tool_call_id": None, "content": "上海雨"},
    ]
    _system, contents = ad._to_gemini(messages)
    fr_parts = [p for c in contents if c.role == "user" for p in c.parts if getattr(p, "function_response", None)]
    assert [p.function_response.name for p in fr_parts] == ["weather_bj", "weather_sh"]


# ---------- OpenAI-compat provider attribution / generic base_url fail-loud / _close_objects / stream close ----------

@pytest.fixture
def _no_generic_base_url(monkeypatch):
    """Clear the generic base_url env vars to isolate the test (a dev box / CI may have set them)."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)


def test_openai_provider_defaults_to_official_base_url(_no_generic_base_url):
    """The real openai profile with no base_url falls back to the official endpoint (its fixed address lives in profile.base_url)."""
    assert LLMClient("openai", api_key="x").base_url == "https://api.openai.com/v1"


def test_openai_compatible_without_base_url_fails_loud(_no_generic_base_url):
    """A generic openai_compatible profile missing base_url fails loud, never silently sending to OpenAI's official endpoint (the key would leak to the wrong host)."""
    from agentmaker.core.exceptions import LLMConfigError
    with pytest.raises(LLMConfigError):
        LLMClient("openai_compatible", api_key="x", model="m")


def test_openai_adapter_error_tagged_with_real_provider():
    """The OpenAI-compatible protocol fronts many vendors; a failed call is tagged with the real provider (deepseek), not a generic openai."""
    import asyncio
    from types import SimpleNamespace
    from agentmaker.core.exceptions import LLMRequestError

    ad = OpenAIAdapter(model="deepseek-v4", api_key="x", base_url="http://x/v1", timeout=1,
                       default_temperature=None, max_tokens_field="max_tokens",
                       structured_output="none", provider="deepseek")

    async def _create(**kw):
        raise RuntimeError("boom")
    ad._ensure_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))

    with pytest.raises(LLMRequestError) as ei:
        asyncio.run(ad.chat([{"role": "user", "content": "hi"}]))
    assert ei.value.provider == "deepseek"                       # attributed to the real vendor


def test_close_objects_preserves_user_field_named_properties():
    """_close_objects recursion: a user field literally named properties (not the properties container) must not be mistaken for a container and mutated."""
    from agentmaker.core.adapters.anthropic import _close_objects
    schema = {"type": "object", "properties": {
        "config": {"type": "object", "properties": {
            "properties": {"type": "string"}}}}}              # user field happens to be named properties; its value is a plain string sub-schema
    out = _close_objects(schema)
    assert out["additionalProperties"] is False               # top-level object closed
    assert out["properties"]["config"]["additionalProperties"] is False   # config object closed
    inner = out["properties"]["config"]["properties"]["properties"]
    assert inner == {"type": "string"}                        # the user properties field is untouched, no additionalProperties added


class _FakeOpenAIStream:
    """Fake openai AsyncStream: supports async with + async iteration + close(), and records whether it was closed (to verify an early break closes the stream)."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    def __aiter__(self):
        return self

    async def close(self):
        self.closed = True


def test_openai_stream_closed_on_early_break():
    """Breaking out of streaming early (async with __aexit__) closes the underlying HTTP stream; no leaked connection."""
    import asyncio
    from types import SimpleNamespace

    fake = _FakeOpenAIStream([_Chunk(choices=[_Choice("aaa")]), _Chunk(choices=[_Choice("bbb")])])
    ad = _openai_adapter("m")

    async def _create(**kw):
        return fake
    ad._ensure_client = lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))

    async def _drive():
        agen = ad.stream([{"role": "user", "content": "hi"}])
        async for _piece in agen:
            break                                             # take one piece, then stop
        await agen.aclose()                                   # closing the outer generator triggers async with __aexit__
    asyncio.run(_drive())
    assert fake.closed is True


def test_openai_tool_stream_emits_reasoning_continuation_state():
    """Streamed reasoning is accumulated into the terminal tool-turn response."""
    import asyncio

    def chunk(*, reasoning=None, call=None, finish=None):
        delta = SimpleNamespace(content=None, reasoning_content=reasoning,
                                tool_calls=[call] if call else None)
        return SimpleNamespace(model="m", usage=None,
                               choices=[SimpleNamespace(delta=delta, finish_reason=finish)])

    call = SimpleNamespace(index=0, id="c1", type="function",
                           function=SimpleNamespace(name="lookup", arguments="{}"))
    fake = _FakeOpenAIStream([
        chunk(reasoning="first "), chunk(reasoning="second", call=call),
        chunk(finish="tool_calls"),
    ])
    ad = _openai_adapter("m")

    async def _create(**kw):
        return fake
    ad._ensure_client = lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=_create)))

    async def _drive():
        return [item async for item in ad.stream(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}])]

    terminal = asyncio.run(_drive())[-1]
    assert terminal.reasoning_content == "first second"
    assert terminal.assistant_message == {"reasoning_content": "first second"}
    assert terminal.tool_calls[0]["id"] == "c1"


def test_anthropic_tool_turn_preserves_signed_blocks_and_order():
    """Anthropic thinking blocks round-trip unchanged while rewritten tool IDs stay aligned."""
    blocks = [
        SimpleNamespace(type="thinking", thinking="", signature="signed"),
        SimpleNamespace(type="redacted_thinking", data="opaque"),
        SimpleNamespace(type="tool_use", id="original", name="lookup", input={"x": 1}),
    ]
    response = SimpleNamespace(content=blocks, usage=None, stop_reason="tool_use", model="claude")
    parsed = _anthropic_adapter("claude")._parse_message(response, 1)
    json.dumps(parsed.assistant_message)
    message = {**parsed.assistant_message, "role": "assistant", "content": "",
               "tool_calls": [{**parsed.tool_calls[0], "id": "rewritten"}]}
    _system, conversation = _anthropic_adapter("claude")._to_anthropic([
        message, {"role": "tool", "tool_call_id": "rewritten", "content": "ok"},
    ])
    returned = conversation[0]["content"]
    assert [block["type"] for block in returned] == ["thinking", "redacted_thinking", "tool_use"]
    assert returned[0] == {"type": "thinking", "thinking": "", "signature": "signed"}
    assert returned[1] == {"type": "redacted_thinking", "data": "opaque"}
    assert returned[2] == {"type": "tool_use", "id": "rewritten", "name": "lookup", "input": {"x": 1}}


class _FakeGenStream:
    """Fake async generator as returned by genai generate_content_stream: async iteration + aclose(), records whether it was closed."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self):
        self.closed = True


def test_gemini_stream_closed_on_early_break():
    """Breaking out of Gemini streaming early -> the finally block aclose()s the underlying stream. Requires google-genai (_prep uses types), else skipped."""
    import asyncio
    from types import SimpleNamespace
    pytest.importorskip("google.genai")

    chunk = type("GC", (), {"text": "hello", "candidates": None, "usage_metadata": None})()
    fake = _FakeGenStream([chunk, chunk])
    ad = _gemini_adapter()

    async def _gcs(**kw):
        return fake
    ad._ensure_client = lambda: SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=_gcs)))

    async def _drive():
        agen = ad.stream([{"role": "user", "content": "hi"}])
        async for _piece in agen:
            break
        await agen.aclose()
    asyncio.run(_drive())
    assert fake.closed is True


# ---------- Gemini finish_reason normalization (keeps truncation observable and consistent across vendors) ----------

def test_gemini_finish_reason_normalized_lowercase():
    """Gemini's FinishReason enum str()s to 'FinishReason.MAX_TOKENS', which won't match other vendors; _finish_reason
    lowercases the member name to 'max_tokens' so the harness's truncation set recognizes it (else Gemini length-truncation is silently treated as complete)."""
    from agentmaker.core.adapters.gemini import _finish_reason
    from agentmaker.runtime.harness import _TRUNCATION_REASONS
    assert _finish_reason(None) is None
    assert _finish_reason("STOP") == "stop"                       # non-enum (string) input falls back to lowercase
    fake_enum = type("FR", (), {"name": "MAX_TOKENS"})()          # mimics the FinishReason enum (has .name)
    assert _finish_reason(fake_enum) == "max_tokens"
    assert "max_tokens" in _TRUNCATION_REASONS                    # after normalization it's recognized by truncation observability


def test_gemini_parse_response_normalizes_finish_reason():
    """_parse_response normalizes via _finish_reason: an enum candidate.finish_reason becomes a lowercase LLMResponse.finish_reason (hermetic)."""
    fc = type("FR", (), {"name": "MAX_TOKENS"})()
    cand = type("Cand", (), {"content": type("C", (), {"parts": []})(), "finish_reason": fc})()
    resp = type("R", (), {"candidates": [cand], "usage_metadata": None})()
    assert _gemini_adapter()._parse_response(resp, 1).finish_reason == "max_tokens"
