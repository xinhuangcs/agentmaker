"""agentmaker.core.llm_clients: multi-provider LLM client.

What it does:
    Uses `provider` to collect each vendor's configuration details (base URL, key env var names, which
    protocol it speaks) into a single table, and exposes only a unified `chat()` / `stream()`.

Call chain:
    provider -> look up _PROFILES to get the profile -> resolve model/key/base_url -> select the adapter
    by protocol -> chat()/stream().

Related modules:
    - Unified response types LLMResponse / StreamStats: llm_response.py
    - Protocol adapters (split by protocol): adapters/ subpackage
    - Unified exception LLMError: exceptions.py
"""

import os
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Optional, overload

from .exceptions import LLMConfigError
from .adapters import _ADAPTERS
from .llm_response import LLMResponse, StreamStats
from .multimodal import messages_have_images


# =============================================================================
# Provider config table: one row per vendor.
# Adding an OpenAI-compatible vendor is a single row (protocol defaults to "openai").
# Base URLs / env var names come from each vendor's public docs, may change; defer to the latest docs.
# =============================================================================
@dataclass(frozen=True)
class ProviderProfile:
    """Default configuration for a single provider.

    Fields (the provider name is the _PROFILES key, not repeated here):
        base_url: default base URL; openai / the generic fallback set this to None and read a generic env var instead.
        key_envs: the API key env var names this vendor conventionally uses (in priority order).
        default_key: placeholder key for local services that do not validate the key.
        default_model: fallback model name, set to the vendor's cheapest real model; used when model= is not passed.
            Models change as vendors update, so verify periodically and only fill in names that actually exist.
        protocol: decides which adapter to use (openai / anthropic / gemini).
        reads_generic_base_url: whether to accept the generic OPENAI_BASE_URL / LLM_BASE_URL.
        max_tokens_field: the field name that caps output length (openai protocol only). Defaults to
            max_tokens; OpenAI reasoning models require max_completion_tokens, and kimi (moonshot) has
            officially deprecated max_tokens in favor of max_completion_tokens, so those two override it.
            deepseek / qwen (dashscope, whose official compatibility docs use max_tokens) / zhipu (whose
            thinking goes through a separate thinking parameter) etc. keep the default max_tokens.
        context_window: the context window size (tokens) of default_model, for context engineering to
            estimate the budget against the real window. This is objective vendor data (like base_url):
            verify against official docs and re-check as models update; for local / self-hosted / proxy
            models the model is user-chosen, so set None when the window is unknown.
        max_output_tokens: the "max output tokens" (generation cap) for a single call of default_model,
            for the window budget to estimate the output reserve. It and context_window are two decoupled
            quantities: vendor windows can reach 1M, yet single-call output caps are commonly only 8K to
            128K and do not scale with the window, so each must be looked up from its own official value.
            The window budget uses this to clamp the "output reserve" to the range the model can actually
            emit (see WindowBudgetConfig.output_reserve), avoiding reserving a dead zone on a large window
            that the model can never fill. Set None when unknown for local / self-hosted / proxy.
        structured_output: this vendor's structured output capability (only the OpenAI protocol adapter
            branches on it): "json_schema" (response_format json_schema, schema carried at the API layer,
            e.g. real openai / gemini_openai), "json_object" (only guarantees valid JSON, schema filled in
            by prompt + validation, e.g. deepseek/qwen/kimi/glm), "none" (does not send response_format,
            pure prompt as backstop, e.g. local/proxy/unknown). The anthropic/gemini native protocols
            always go through their own native path (output_config / responseSchema); their value "native"
            is annotation only and the adapter does not read it.
        supports_function_calling: whether this provider supports native function calling (tool calls) by
            default. Native tool calls depend on this capability. The six commercial vendors and the
            mainstream local inference stacks (vLLM/Ollama/llama.cpp)
            all support it, so it defaults to True; the few models that do not (e.g. some pure reasoning
            models) are overridden at the model level in _KNOWN_MODELS, or by explicitly passing
            supports_function_calling=False when constructing LLMClient. An Agent with tools that hits False
            fails loud at construction time (see agents/agent.py) rather than failing silently at runtime.
            To make an fc-less model actually usable with tools, construct LLMClient(..., emulate_tools=True)
            to enable the text-emulation shim (adapters/tool_emulation.py, which flips this capability bit to True).
        supports_vision: whether this provider's chat models accept image input (multimodal content
            parts, see core/multimodal.py). True/False when the provider-wide answer is known from the
            official docs; None = unknown (the framework does not block, the server decides). Like other
            vendor facts, verify periodically; override per client via LLMClient(supports_vision=...).
    """
    base_url: Optional[str] = None
    key_envs: tuple[str, ...] = ()
    default_key: Optional[str] = None
    default_model: Optional[str] = None
    protocol: str = "openai"
    reads_generic_base_url: bool = False
    max_tokens_field: str = "max_tokens"
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    structured_output: str = "none"
    supports_function_calling: bool = True
    supports_vision: Optional[bool] = None


@dataclass(frozen=True)
class ModelInfo:
    """Model-specific context limits and optional function-calling override."""
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    supports_function_calling: Optional[bool] = None

_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

_PROFILES: Dict[str, ProviderProfile] = {
    # OpenAI-compatible providers
    "openai":            ProviderProfile(base_url=_DEFAULT_OPENAI_BASE_URL, key_envs=("OPENAI_API_KEY",), reads_generic_base_url=True, default_model="gpt-4.1-nano", max_tokens_field="max_completion_tokens", context_window=1_047_576, max_output_tokens=32_768, structured_output="json_schema", supports_vision=True),
    "deepseek":          ProviderProfile(base_url="https://api.deepseek.com/v1", key_envs=("DEEPSEEK_API_KEY",), default_model="deepseek-v4-flash", context_window=1_000_000, max_output_tokens=393_216, structured_output="json_object", supports_vision=False),  # official spec says "384K" output with binary K: 384*1024
    "dashscope":         ProviderProfile(base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", key_envs=("DASHSCOPE_API_KEY",), default_model="qwen-flash", context_window=1_000_000, max_output_tokens=32_768, structured_output="json_object"),
    "moonshot":          ProviderProfile(base_url="https://api.moonshot.cn/v1", key_envs=("MOONSHOT_API_KEY",), default_model="moonshot-v1-8k", max_tokens_field="max_completion_tokens", context_window=8_192, max_output_tokens=8_192, structured_output="json_object"),
    "zhipu":             ProviderProfile(base_url="https://open.bigmodel.cn/api/paas/v4", key_envs=("ZHIPUAI_API_KEY", "ZAI_API_KEY", "ZHIPU_API_KEY"), default_model="glm-4.7-flash", context_window=204_800, max_output_tokens=131_072, structured_output="json_object"),
    "modelscope":        ProviderProfile(base_url="https://api-inference.modelscope.cn/v1/", key_envs=("MODELSCOPE_API_KEY",)),
    "gemini_openai":     ProviderProfile(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", key_envs=("GEMINI_API_KEY", "GOOGLE_API_KEY"), default_model="gemini-3.1-flash-lite", context_window=1_048_576, max_output_tokens=65_536, structured_output="json_schema"),
    "ollama":            ProviderProfile(base_url="http://localhost:11434/v1", default_key="ollama"),
    "vllm":              ProviderProfile(base_url="http://localhost:8000/v1", default_key="vllm"),
    "sglang":            ProviderProfile(base_url="http://localhost:30000/v1", default_key="sglang"),
    "openai_compatible": ProviderProfile(key_envs=("LLM_API_KEY", "OPENAI_API_KEY"), reads_generic_base_url=True),

    # Native protocols
    "anthropic":         ProviderProfile(key_envs=("ANTHROPIC_API_KEY",), protocol="anthropic", default_model="claude-haiku-4-5-20251001", context_window=200_000, max_output_tokens=64_000, structured_output="native", supports_vision=True),
    "gemini":            ProviderProfile(key_envs=("GEMINI_API_KEY", "GOOGLE_API_KEY"), protocol="gemini", default_model="gemini-3.1-flash-lite", context_window=1_048_576, max_output_tokens=65_536, structured_output="native", supports_vision=True),
}


_KNOWN_MODELS: Dict[str, ModelInfo] = {
    # OpenAI
    "gpt-5.6":               ModelInfo(context_window=1_050_000, max_output_tokens=128_000),
    "gpt-5.6-sol":           ModelInfo(context_window=1_050_000, max_output_tokens=128_000),
    "gpt-5.6-terra":         ModelInfo(context_window=1_050_000, max_output_tokens=128_000),
    "gpt-5.6-luna":          ModelInfo(context_window=1_050_000, max_output_tokens=128_000),
    "gpt-5.5":               ModelInfo(context_window=1_050_000, max_output_tokens=128_000),
    "gpt-5.4":               ModelInfo(context_window=1_050_000, max_output_tokens=128_000),
    "gpt-5.4-mini":          ModelInfo(context_window=400_000, max_output_tokens=128_000),
    "gpt-5.4-nano":          ModelInfo(context_window=400_000, max_output_tokens=128_000),
    # Other providers
    "gemini-3.5-flash":      ModelInfo(context_window=1_048_576, max_output_tokens=65_536),
    "deepseek-v4-pro":       ModelInfo(context_window=1_000_000, max_output_tokens=393_216),
    "claude-fable-5":        ModelInfo(context_window=1_000_000, max_output_tokens=128_000),
    "claude-opus-4-8":       ModelInfo(context_window=1_000_000, max_output_tokens=128_000),
    "claude-sonnet-5":       ModelInfo(context_window=1_000_000, max_output_tokens=128_000),
    "claude-sonnet-4-6":     ModelInfo(context_window=1_000_000, max_output_tokens=128_000),
    "glm-5.2":               ModelInfo(context_window=1_000_000, max_output_tokens=131_072),
    "glm-5.1":               ModelInfo(context_window=204_800, max_output_tokens=131_072),
    "glm-5":                 ModelInfo(context_window=204_800, max_output_tokens=131_072),
    "glm-5-turbo":           ModelInfo(context_window=204_800, max_output_tokens=131_072),
    "glm-4.7":               ModelInfo(context_window=204_800, max_output_tokens=131_072),
    "qwen3.6-flash":         ModelInfo(context_window=1_000_000, max_output_tokens=32_768),
    "qwen3.5-flash":         ModelInfo(context_window=1_000_000, max_output_tokens=65_536),
    # Moonshot publishes no independent output cap; the caller's max_tokens applies.
    "kimi-k2.7-code":        ModelInfo(context_window=262_144, max_output_tokens=None),
    "kimi-k2.6":             ModelInfo(context_window=262_144, max_output_tokens=None),
    "kimi-k2.5":             ModelInfo(context_window=262_144, max_output_tokens=None),
    "moonshot-v1-128k":      ModelInfo(context_window=131_072, max_output_tokens=None),
}


# =============================================================================
# Front-end client
# =============================================================================
class LLMClient:
    """LLM front-end client: resolves configuration from an explicit provider and dispatches calls to the adapter for the corresponding protocol.

    Usage (provider defaults to deepseek; each cloud vendor's default_model is set to its cheapest model,
    so model may be omitted; local / self-hosted / proxy must pass model explicitly):
        LLMClient()                                             # equivalent to deepseek + deepseek-v4-flash
        LLMClient("openai")                                     # uses openai's default gpt-4.1-nano
        LLMClient("openai", model="<real model>")               # explicit model, highest priority
        LLMClient("openai_compatible", api_key="x", base_url="http://host/v1", model="m")  # self-hosted / proxy, must pass model
        LLMClient("anthropic")                                  # Claude native, default haiku
        LLMClient("gemini")                                     # Gemini native, default flash-lite

        # chat/stream are both async (the framework is fully async):
        resp = await llm.chat([{"role": "user", "content": "Hello"}]); print(resp.content)
        async for piece in llm.stream([{"role": "user", "content": "Tell a joke"}]): print(piece, end="")
        # For synchronous spots use agentmaker.core.aio: run_sync(llm.chat(...)) / iter_sync(llm.stream(...))
        print(llm.last_stream_stats)
    """

    def __init__(self, provider: str = "deepseek", model: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, *, timeout: float = 60.0,
                 default_temperature: Optional[float] = None, context_window: Optional[int] = None,
                 max_output_tokens: Optional[int] = None,
                 supports_function_calling: Optional[bool] = None, supports_vision: Optional[bool] = None,
                 emulate_tools: bool = False,
                 profile: Optional[ProviderProfile] = None):
        """Take the config profile by provider, resolve model/key/base_url, validate, and instantiate the adapter by protocol.

        Constructing the adapter sends no network request; the actual networking happens in chat()/stream().

        Args:
            provider: the vendor name in _PROFILES, defaults to "deepseek" (unknown raises and lists the options).
            model: the model name; if omitted, uses this provider's default_model (cloud vendors have the
                cheapest model filled in; local / self-hosted have no default and must pass it explicitly).
            api_key: the API key; if omitted, resolved via the _resolve_key fallback chain.
            base_url: the service URL; if omitted, resolved via the _resolve_base_url fallback chain (may be None for native protocols).
            timeout: timeout in seconds.
            default_temperature: framework-level default sampling temperature. Defaults to None = do not send
                a temperature parameter (leave it to each model server's own default): uniformly we do not decide
                temperature on the developer's behalf. For determinism / a custom temperature, pass
                `chat(..., temperature=...)` explicitly per call, or set a global default here; if set, it is
                sent on every call (developer's own risk: if a model does not support temperature and you set
                one, the server errors and returns it to you as usual for you to adjust).
            context_window: this model's context window (tokens), explicit value takes priority; if omitted,
                the profile value is trusted only when the model is this vendor's default_model, otherwise
                (switched model / local self-hosted) it is unknown (None): avoids the context budget being
                distorted by the wrong window.
            max_output_tokens: this model's single-call max output tokens, same resolution rule as
                context_window (explicit first, otherwise trust the profile only for default_model, None for
                a switched model / self-hosted); lets the window budget clamp the output reserve to the range the model can actually emit.
            supports_function_calling: whether native function calling is supported; None = resolve by
                model-level / provider-level default (see ProviderProfile.supports_function_calling), pass
                True/False explicitly to override. An Agent with tools that hits False fails loud at construction time.
            supports_vision: whether this model accepts image input (multimodal content parts, see
                core/multimodal.py); None = provider-level default from the profile (unknown providers
                stay None and are not blocked). False makes chat/stream fail loud before any network
                call when image parts are present.
            emulate_tools: run tools via text emulation for models that do not support native function calling
                (opt-in). Enabling it wraps the underlying adapter with ToolEmulationAdapter: it writes the tool
                catalog into system, flattens the tool trace into text, and parses tool calls out of the model's
                plain-text reply and exposes standard tool_calls to the agent loop,
                and supports_function_calling is exposed as True. Do not enable it if
                native fc is available: text emulation is inherently less reliable and costs extra tokens. Defaults to False.
            profile: an optional custom ProviderProfile: lets a developer add a custom provider without editing
                the framework source. If given, it is used and the built-in _PROFILES lookup is skipped, going
                through the exact same model/key/base_url/adapter resolution. provider here is just an
                identifier name (recommend passing provider="myvendor" alongside). Adding a model only needs
                model=; for an OpenAI-compatible service use "openai_compatible" + base_url.

        Example:
            LLMClient("deepseek").chat([{"role": "user", "content": "hi"}]).content
            LLMClient(provider="myvendor", profile=ProviderProfile(base_url=..., key_envs=("K",), default_model="m"), model="m")
        """
        if profile is None:                       # no profile given: look up the built-in table by name
            if provider not in _PROFILES:
                raise LLMConfigError(f"Unknown provider='{provider}'. For a custom one pass profile=ProviderProfile(...), "
                               f"or use 'openai_compatible' + base_url. Built-in options: {', '.join(sorted(_PROFILES))}. "
                               "If what you passed is actually a model name (e.g. 'gpt-5'), use the 'provider:model' format (AgentSpec.model) "
                               "or LLMClient(provider, model=...).")
            profile = _PROFILES[provider]
        self.provider = provider              # when a profile is passed, provider is just an identifier name
        self.protocol = profile.protocol

        self.api_key = self._resolve_key(api_key, profile)
        self.base_url = self._resolve_base_url(base_url, profile)
        self.model = self._resolve_model(model, profile)
        # Context window is computed per "this model": profile.context_window only holds for the vendor's default_model and is distorted for a switched model, so it must be resolved (see _resolve_context_window)
        self.context_window = self._resolve_context_window(context_window, self.model, profile)
        # Max output tokens is likewise computed per "this model" (same source and rule as the window, for the window budget to estimate the output reserve)
        self.max_output_tokens = self._resolve_max_output(max_output_tokens, self.model, profile)
        # Whether native function calling is supported (the framework's only tool mechanism): explicit > model-level override > provider-level default
        self.supports_function_calling = self._resolve_supports_fc(supports_function_calling, self.model, profile)
        # Whether image input (multimodal content parts) is accepted: explicit > provider-level default;
        # None = unknown (do not block, the server decides). False fails loud in chat/stream before any
        # network call, so an image sent to a text-only vendor errors clearly instead of confusingly.
        self.supports_vision = supports_vision if supports_vision is not None else profile.supports_vision

        if not self.api_key:
            raise LLMConfigError(f"No API key found (provider={provider}). Pass api_key, "
                           f"or set {' / '.join(profile.key_envs) or 'LLM_API_KEY'}.")
        if not self.model:
            raise LLMConfigError(f"No model specified (provider={provider}). Pass model (use the vendor's real model name), "
                           f"or configure default_model for this provider in _PROFILES.")
        if self.protocol == "openai" and not self.base_url:
            raise LLMConfigError(f"No base_url found (provider={provider}). Pass base_url or set OPENAI_BASE_URL / LLM_BASE_URL.")

        self._adapter = _ADAPTERS[self.protocol](
            model=self.model, api_key=self.api_key, base_url=self.base_url,
            timeout=timeout, default_temperature=default_temperature,
            max_tokens_field=profile.max_tokens_field,
            structured_output=profile.structured_output,
            provider=self.provider,               # label errors by the real vendor (the OpenAI-compatible protocol serves many)
        )
        if emulate_tools:
            from .adapters.tool_emulation import ToolEmulationAdapter
            self._adapter = ToolEmulationAdapter(self._adapter)
            self.supports_function_calling = True
        self.supports_streaming_tools = (
            getattr(self._adapter, "supports_streaming_tools", True) is not False
        )

    def __enter__(self) -> "LLMClient":
        """Enter a synchronous ownership context."""
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Close this thread's SDK client when leaving a synchronous context."""
        try:
            self.close()
        except Exception as close_error:   # KeyboardInterrupt/SystemExit during cleanup must still propagate
            if exc is None:
                raise
            exc.add_note(f"LLM client cleanup also failed: {close_error}")

    async def __aenter__(self) -> "LLMClient":
        """Enter an asynchronous ownership context."""
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Close this event loop's SDK client when leaving an asynchronous context."""
        try:
            await self.aclose()
        except Exception as close_error:   # KeyboardInterrupt/SystemExit during cleanup must still propagate
            if exc is None:
                raise
            exc.add_note(f"LLM client cleanup also failed: {close_error}")

    def close(self) -> None:
        """Close this thread's cached asynchronous SDK client."""
        from .aio import run_sync
        run_sync(self.aclose())

    async def aclose(self) -> None:
        """Close this event loop's cached SDK client."""
        await self._adapter.aclose()

    async def chat(self, messages: list[dict], *, temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None, output_schema: Optional[dict] = None,
                   **kwargs: Any) -> LLMResponse:
        """One-shot (non-streaming) async call, delegated to the underlying adapter, returning a unified LLMResponse (the framework is fully async: this is the only call entry point).

        Args:
            messages: unified message list.
            temperature: sampling temperature, None uses the default.
            max_tokens: max generated tokens, optional.
            output_schema: optional JSON Schema (dict); if given, requires the model to output per it: the
                adapter translates it into response_format json_schema / json_object / Anthropic output_config
                / Gemini responseSchema based on provider capability. Usually passed by the upper-layer
                harness.structured (which then does pydantic validation + retry on failure, see 2.3).
            **kwargs: other parameters passed through to the underlying SDK.

        Returns:
            LLMResponse: unified response object (awaiting it means "wait for this call to finish and get the result", deterministic order).

        Synchronous calls: driven via `agentmaker.core.aio.run_sync(client.chat(...))` (the framework's synchronous facade routes through it uniformly).
        """
        self._check_vision(messages)
        return await self._adapter.chat(messages, temperature=temperature, max_tokens=max_tokens,
                                        output_schema=output_schema, **kwargs)

    def _check_vision(self, messages) -> None:
        """Fail loud before any network call when image parts are sent to a provider known not to take them.

        Raises:
            LLMConfigError: supports_vision is False and the messages carry image content parts.
        """
        if self.supports_vision is False and messages_have_images(messages):
            raise LLMConfigError(
                f"Provider '{self.provider}' (model={self.model}) does not accept image input "
                "(supports_vision=False): switch to a vision-capable model, or pass "
                "supports_vision=True explicitly if you know this specific model does.")

    @overload
    def stream(self, messages: list[dict], *, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               on_stats: Optional[Callable[[StreamStats], None]] = None,
               tools: None = None, **kwargs: Any) -> AsyncIterator[str]: ...

    @overload
    def stream(self, messages: list[dict], *, temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               on_stats: Optional[Callable[[StreamStats], None]] = None,
               tools: list[dict], **kwargs: Any) -> AsyncIterator[str | LLMResponse]: ...

    async def stream(self, messages: list[dict], *, temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None,
                     on_stats: Optional[Callable[[StreamStats], None]] = None,
                     tools: Optional[list[dict]] = None,
                     **kwargs: Any) -> AsyncIterator[str | LLMResponse]:
        """Streaming async call (async generator), delegated to the underlying adapter to emit text piece by piece.

        Args:
            messages: unified message list.
            temperature: sampling temperature, None uses the default.
            max_tokens: max generated tokens, optional.
            on_stats: optional callback (StreamStats) -> None; hands back stats (usage / latency / finish
                reason) for this call when the stream is exhausted. More reliable than reading
                last_stream_stats: concurrent streams on a shared client overwrite last_stream_stats.
            tools: Optional native function-calling schemas. A non-empty list requires an adapter that
                emits a terminal LLMResponse for each streamed turn.
            **kwargs: other parameters passed through to the underlying SDK.

        Returns:
            The text deltas emitted piece by piece (consumed via async for; synchronous consumption via `aio.iter_sync`).

        Note: without tools the stream is str-only. With tools passed, the stream additionally yields
        exactly one final LLMResponse after the text drains -- content is the
        joined text, tool_calls carries the accumulated calls (or None). This terminal response is the
        channel the Agent streaming tool loop consumes; plain-text callers never see it.
        """
        self._check_vision(messages)   # generator body: raises on first iteration, still before any network call
        if tools and not self.supports_streaming_tools:
            raise LLMConfigError(
                "This adapter does not support streaming tool calls. Use chat() for text tool emulation, "
                "or switch to an adapter with native streaming function calling.")
        if tools is not None:
            kwargs["tools"] = tools
        async for piece in self._adapter.stream(messages, temperature=temperature,
                                                max_tokens=max_tokens, on_stats=on_stats, **kwargs):
            yield piece

    @property
    def last_stream_stats(self) -> Optional[StreamStats]:
        """Return the stats of the most recent stream() (usage / latency / finish reason); None if no streaming call has been made.

        Returns:
            Optional[StreamStats]: the streaming stats object or None.
        """
        return self._adapter.last_stream_stats

    @staticmethod
    def _resolve_key(api_key, profile):
        """Resolve the API key via the fallback chain: explicit > this vendor's dedicated env var > generic LLM_API_KEY > local-service placeholder key.

        Args:
            api_key: the explicitly passed key or None.
            profile: the provider config profile.

        Returns:
            Optional[str]: the resolved key or None.
        """
        if api_key:
            return api_key
        for env in profile.key_envs:
            if os.getenv(env):
                return os.getenv(env)
        return os.getenv("LLM_API_KEY") or profile.default_key

    @staticmethod
    def _resolve_model(model, profile):
        """Resolve the model name: an explicitly passed model takes priority, otherwise use the vendor's default_model (cheapest model).

        Args:
            model: the explicitly passed model name or None.
            profile: the provider config profile.

        Returns:
            Optional[str]: the resolved model name or None.
        """
        if model:
            return model
        return profile.default_model

    @staticmethod
    def _resolve_context_window(context_window, model, profile):
        """Resolve this model's context window: explicit value first; otherwise trust the profile value only when the model is this vendor's default_model, unknown for a switched model / local self-hosted (no default), returning None (downstream ContextConfig.for_window has a fallback).

        Args:
            context_window: the explicitly passed window or None.
            model: the already-resolved model name.
            profile: the provider config profile.

        Returns:
            Optional[int]: the context window token count or None (unknown).
        """
        if context_window is not None:
            return context_window
        if profile.default_model and model == profile.default_model:
            return profile.context_window
        info = _KNOWN_MODELS.get(model)              # non-default but known model -> look up the catalog (resolves the unknown-window problem for non-default models)
        if info is not None:
            return info.context_window
        return None

    @staticmethod
    def _resolve_max_output(max_output_tokens, model, profile):
        """Resolve this model's single-call max output tokens: the rule is identical to _resolve_context_window.

        Explicit value first; otherwise trust profile.max_output_tokens only when the model is this vendor's
        default_model, None (unknown) for a switched model / self-hosted.

        Args:
            max_output_tokens: the explicitly passed output cap or None.
            model: the already-resolved model name.
            profile: the provider config profile.

        Returns:
            Optional[int]: the single-call max output tokens or None (unknown).
        """
        if max_output_tokens is not None:
            return max_output_tokens
        if profile.default_model and model == profile.default_model:
            return profile.max_output_tokens
        info = _KNOWN_MODELS.get(model)              # non-default but known model -> look up the catalog
        if info is not None:
            return info.max_output_tokens
        return None

    @staticmethod
    def _resolve_supports_fc(supports_function_calling, model, profile):
        """Resolve whether this model supports native function calling: explicit value first -> the model-level override in _KNOWN_MODELS (non-None) -> the provider-level default (profile.supports_function_calling).

        Unlike window/output ("trust the profile only for default_model"): fc is a protocol / model capability,
        the provider-level default holds for the whole vendor, so a switched model still inherits the provider
        default (unless that model explicitly marks supports_function_calling in _KNOWN_MODELS).

        Args:
            supports_function_calling: the explicitly passed capability (bool) or None (not explicit, use default resolution).
            model: the already-resolved model name.
            profile: the provider config profile.

        Returns:
            bool: whether this model supports native fc (always a bool; the provider-level default is non-None).
        """
        if supports_function_calling is not None:
            return supports_function_calling
        info = _KNOWN_MODELS.get(model)
        if info is not None and info.supports_function_calling is not None:
            return info.supports_function_calling
        return profile.supports_function_calling

    @staticmethod
    def _resolve_base_url(base_url, profile):
        """Resolve the service URL via the fallback chain: explicit > [generic providers only] generic env var > the vendor's fixed URL.

        Fixed-URL vendors use only their own default value, avoiding accidental use of another vendor's proxy
        URL; anthropic / gemini are None (use the SDK default).

        It does NOT fall back to the OpenAI official URL: the real `openai` profile's fixed URL is the official
        URL (written in profile.base_url) and obtained via the normal fallback chain; whereas the generic
        `openai_compatible` (no fixed URL) resolves to None when base_url is missing, letting the constructor
        fail loud: it never silently sends the request (including the key) to OpenAI official (which could be
        another vendor's key sent to the wrong place).

        Args:
            base_url: the explicitly passed URL or None.
            profile: the provider config profile.

        Returns:
            Optional[str]: the resolved URL or None (generic profile missing config -> None -> upper layer raises).
        """
        if base_url:
            return base_url
        if profile.reads_generic_base_url:
            return os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or profile.base_url
        return profile.base_url
