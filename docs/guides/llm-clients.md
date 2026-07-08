# LLM clients & providers

`LLMClient` is the single entry point for talking to a model. You name a provider (and optionally a model), and the client resolves the API key, endpoint, and wire protocol for you, then exposes just two async methods: `chat()` for a one-shot reply and `stream()` for token-by-token output. Every provider, whether it speaks the OpenAI-compatible protocol, Anthropic's native protocol, or Gemini's native protocol, returns the same unified `LLMResponse`, so the rest of your code never branches on the vendor.

Reach for `LLMClient` directly when you want raw model access. When you build an [Agent](agents.md), you usually hand it an `LLMClient` (or let `AgentSpec` construct one from a `"provider:model"` string) and never call `chat()` yourself.

```python
from agentmaker import LLMClient

llm = LLMClient("deepseek")                       # provider's default model (deepseek-v4-flash)
resp = await llm.chat([{"role": "user", "content": "Hello"}])
print(resp.content)
```

!!! note
    `chat()` and `stream()` are coroutines: the framework is async to the core. Run them inside an event loop with `await`, or from synchronous code through the facade in `agentmaker.core.aio` (`run_sync(llm.chat(...))` / `iter_sync(llm.stream(...))`).

## Selecting a provider and model

The first positional argument is the provider name; it defaults to `"deepseek"`. If you omit `model`, the client uses that provider's built-in `default_model` (each cloud vendor's cheapest real model). Pass `model=` to switch, which always takes priority.

```python
LLMClient()                                        # deepseek + deepseek-v4-flash
LLMClient("openai")                                # openai's default (gpt-4.1-nano)
LLMClient("openai", model="gpt-5.4-nano")          # explicit model, highest priority
LLMClient("anthropic")                             # Claude native, default haiku
LLMClient("gemini")                                # Gemini native, default flash-lite
```

Local, self-hosted, and proxy providers have no default model, so you must pass `model=` explicitly:

```python
LLMClient("openai_compatible", api_key="x", base_url="http://host/v1", model="my-model")
```

An unknown provider raises `LLMConfigError` and lists the built-in options. If you accidentally pass a model name where a provider is expected (for example `LLMClient("gpt-5")`), the error points you to `LLMClient(provider, model=...)`.

### The `"provider:model"` string form

Declarative configuration ([`AgentSpec`](agents.md)) accepts the model as a single string using a colon convention. `build_agent` splits it into an `LLMClient`:

- `"deepseek:deepseek-v4-flash"` becomes `LLMClient("deepseek", model="deepseek-v4-flash")`.
- A bare provider name with no colon (`"deepseek"`) becomes `LLMClient("deepseek")`, using that provider's default model.
- An empty right half (`"deepseek:"`) falls the model back to the provider default.

You can also pass an `LLMClient` instance directly when you want to pin the key or base URL yourself. This colon syntax lives on `AgentSpec.model`; `LLMClient` itself always takes `provider` and `model` as separate arguments.

## Built-in providers

Providers are grouped by the wire protocol they speak. Adding an OpenAI-compatible vendor is a single configuration row, so most entries share one adapter. The `default_model` column shows the model used when you omit `model=`; a dash means the model is user-chosen and must be passed explicitly.

### OpenAI-compatible protocol

| Provider | Default model | API key env var(s) | Structured output |
| --- | --- | --- | --- |
| `openai` | `gpt-4.1-nano` | `OPENAI_API_KEY` | `json_schema` |
| `deepseek` | `deepseek-v4-flash` | `DEEPSEEK_API_KEY` | `json_object` |
| `dashscope` | `qwen-flash` | `DASHSCOPE_API_KEY` | `json_object` |
| `moonshot` | `moonshot-v1-8k` | `MOONSHOT_API_KEY` | `json_object` |
| `zhipu` | `glm-4.7-flash` | `ZHIPUAI_API_KEY`, `ZAI_API_KEY`, `ZHIPU_API_KEY` | `json_object` |
| `modelscope` | (pass `model=`) | `MODELSCOPE_API_KEY` | `none` |
| `gemini_openai` | `gemini-3.1-flash-lite` | `GEMINI_API_KEY`, `GOOGLE_API_KEY` | `json_schema` |
| `ollama` | (pass `model=`) | (local placeholder key) | `none` |
| `vllm` | (pass `model=`) | (local placeholder key) | `none` |
| `sglang` | (pass `model=`) | (local placeholder key) | `none` |
| `openai_compatible` | (pass `model=`) | `LLM_API_KEY`, `OPENAI_API_KEY` | `none` |

### Anthropic native protocol

| Provider | Default model | API key env var(s) | Structured output |
| --- | --- | --- | --- |
| `anthropic` | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` | native |

### Gemini native protocol

| Provider | Default model | API key env var(s) | Structured output |
| --- | --- | --- | --- |
| `gemini` | `gemini-3.1-flash-lite` | `GEMINI_API_KEY`, `GOOGLE_API_KEY` | native |

!!! note
    Model names and endpoints are vendor facts that drift as providers ship new models. The framework verifies them against official docs periodically; treat the table as the shipped defaults, not a permanent guarantee. Use `gemini` / `anthropic` (the native protocols) when you want each vendor's full native feature set; `gemini_openai` is the OpenAI-compatible shim for Gemini.

## Credentials and endpoints

You rarely pass keys in code. `LLMClient` resolves the API key through a fallback chain:

1. An explicit `api_key=` argument.
2. The provider's dedicated env var(s), tried in the order listed above.
3. The generic `LLM_API_KEY` env var.
4. A local placeholder key (for services like `ollama` that do not validate the key).

If nothing resolves, construction raises `LLMConfigError` naming the env vars to set. The base URL resolves similarly: an explicit `base_url=` wins; the generic providers (`openai`, `openai_compatible`) additionally read `OPENAI_BASE_URL` / `LLM_BASE_URL`; fixed-URL vendors use only their own endpoint. The native `anthropic` and `gemini` protocols leave `base_url` as `None` and use their SDK's default endpoint.

The safe default is to set the matching env var and construct with just the provider name:

```python
llm = LLMClient("openai")     # reads OPENAI_API_KEY from the environment
```

## Async chat

`chat()` sends the messages and returns one `LLMResponse`. Messages are a list of `{"role", "content"}` dicts (roles are `user`, `assistant`, `system`, `tool`).

```python
resp = await llm.chat(
    [{"role": "user", "content": "Summarize async I/O in one sentence."}],
    temperature=0.2,
    max_tokens=200,
)
print(resp.content)
print(resp.usage, resp.finish_reason)
```

Both `temperature` and `max_tokens` are optional. By default the client sends no temperature at all and defers to the model server's own default; pass `temperature=` per call (or set `default_temperature=` on the constructor) when you need determinism. Extra keyword arguments pass straight through to the underlying SDK.

## Streaming

`stream()` is an async generator that yields text deltas as the model produces them. Consume it with `async for`. The example below is fully hermetic (no API key, no network) using the `ScriptedLLM` test double, which mirrors the real client's `chat()` / `stream()` interface:

```python
import asyncio

from agentmaker import Agent
from agentmaker.testing import ScriptedLLM


async def main():
    # The async twin of agent.run(...).
    agent = Agent("assistant", ScriptedLLM(["Hello from an async run."]))
    result = await agent.arun("hi")
    print("arun:", result.final_output)

    # Token streaming is exposed on the LLM client as an async generator.
    llm = ScriptedLLM(["streamed piece by piece"])
    chunks = [chunk async for chunk in llm.stream([{"role": "user", "content": "hi"}])]
    print("stream chunks:", chunks)


asyncio.run(main())
```

With a real client the shape is identical:

```python
async for piece in llm.stream([{"role": "user", "content": "Tell a joke"}]):
    print(piece, end="")
```

### Stream statistics

A stream yields only text, so per-call metadata lives separately. After the stream drains, read `llm.last_stream_stats` (or `None` if you have not streamed yet). It exposes `model`, `finish_reason`, `usage`, and `latency_ms`. For token usage on OpenAI-family providers the request must opt in with `stream_options={"include_usage": True}`, so `usage` may be `None` otherwise.

```python
async for piece in llm.stream([{"role": "user", "content": "hi"}]):
    print(piece, end="")
stats = llm.last_stream_stats
print(stats.model, stats.latency_ms)
```

Under concurrent streams on a shared client, `last_stream_stats` can be overwritten. Pass an `on_stats` callback instead to receive this call's stats object reliably when its stream finishes:

```python
collected = []
async for piece in llm.stream(messages, on_stats=collected.append):
    ...
```

## The `LLMResponse`

Every non-streaming call returns an `LLMResponse` dataclass. The fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `content` | `str` | The reply text. Also what `str(resp)` / `print(resp)` shows. |
| `finish_reason` | `str \| None` | Why generation stopped. |
| `model` | `str` | The actual model name used. |
| `usage` | `dict \| None` | Token usage (may contain nested detail structures, not only ints). |
| `reasoning_content` | `str \| None` | Separate reasoning trace, when the model returns one. |
| `tool_calls` | `list \| None` | Function-calling tool calls in OpenAI format, ready to feed back into `messages`; `None` when absent. |
| `latency_ms` | `int` | Round-trip latency in milliseconds. |
| `raw` | `Any` | The provider's raw response object. |

```python
resp = await llm.chat([{"role": "user", "content": "hi"}])
print(resp.content)          # the text
print(resp)                  # same thing: __str__ returns content
```

## Structured output

Pass `output_schema=` (a JSON Schema dict) to `chat()` to ask the model to emit JSON that conforms to it. The adapter translates the schema according to the provider's capability (the `structured_output` column above):

- `json_schema`: the schema is carried at the API layer via `response_format` (for example `openai`, `gemini_openai`).
- `json_object`: the request only guarantees valid JSON; the schema is injected through the prompt and validated afterward (for example `deepseek`, `dashscope`, `moonshot`, `zhipu`).
- `none`: no `response_format` is sent; the prompt alone is the backstop (local, proxy, and unknown providers).
- native: the `anthropic` and `gemini` protocols always route through their own native structured path.

That is the low-level `chat()` view. For the agent-level `output_schema` on `run()`, with automatic retries and Pydantic validation, see [Structured output](structured-output.md).

For most work you do not call this directly. The [Agent](agents.md) layer accepts a Pydantic model as `output_schema`, drives this mechanism, then validates the JSON into an instance and retries on failure. That ergonomic path is hermetic to test:

```python
from pydantic import BaseModel

from agentmaker import Agent
from agentmaker.testing import ScriptedLLM


class Person(BaseModel):
    name: str
    age: int


llm = ScriptedLLM(['{"name": "Ada", "age": 36}'])
agent = Agent("extractor", llm)

person = agent.run("Extract the person from: Ada is 36.", output_schema=Person).final_output
print(f"{type(person).__name__}(name={person.name!r}, age={person.age})")
```

## Messages and multimodal content

A message's `content` is either a plain string (the common case) or a list of provider-neutral content parts. The `Message` dataclass models one message with a `role`, `content`, a `timestamp`, and a `metadata` dict; call `to_dict()` to get the `{"role", "content"}` shape that `chat()` and `stream()` consume.

```python
from agentmaker import Message

msg = Message(content="Hello", role="user")
await llm.chat([msg.to_dict()])
```

To send text and images in one message, build the content list with the part helpers (all importable from the top level):

- `text_part(text)` builds a text part.
- `image_part_from_bytes(data, media_type)` builds an inline image from raw bytes.
- `image_part_from_file(path, media_type=None)` reads a local file (the media type is inferred from the suffix when omitted).
- `image_part_from_url(url)` references a remote image the provider fetches.

```python
from agentmaker import LLMClient, text_part, image_part_from_file

llm = LLMClient("openai")
messages = [{
    "role": "user",
    "content": [
        text_part("What is in this image?"),
        image_part_from_file("photo.png"),
    ],
}]
resp = await llm.chat(messages)
```

The accepted image media types are `image/jpeg`, `image/png`, `image/gif`, and `image/webp`; an unsupported type raises `ValueError` at construction time rather than failing server-side. `image_part_from_url` is not supported by the Gemini adapter (use an inline part there).

### Vision gating

Each provider profile carries a `supports_vision` flag. When it is known to be `False` (for example `deepseek`), sending image parts raises `LLMConfigError` before any network call, so a text-only vendor fails with a clear message instead of a confusing server error. When it is unknown (`None`), the framework does not block and lets the server decide. Override per client with `LLMClient(..., supports_vision=True)` if you know a specific model accepts images.

## Custom providers and protocols

You do not have to edit the framework to add a vendor. Pass a `ProviderProfile` to reuse an existing protocol without touching the source:

```python
from agentmaker import LLMClient, ProviderProfile

llm = LLMClient(
    provider="myvendor",
    profile=ProviderProfile(base_url="https://api.myvendor.com/v1", key_envs=("MY_KEY",), default_model="m"),
    model="m",
)
```

For an entirely new wire protocol, register an adapter class (a `BaseAdapter` subclass) under a protocol name, then reference that name from a profile:

```python
from agentmaker.core.adapters import register_adapter

register_adapter("myproto", MyAdapter)   # MyAdapter is your BaseAdapter subclass
LLMClient("myvendor", profile=ProviderProfile(protocol="myproto", default_model="m", key_envs=("MYVENDOR_API_KEY",)))
```

For models that lack native function calling, `LLMClient(..., emulate_tools=True)` wraps the adapter with a text-emulation shim so tool-using agents still work. Enable it only when native function calling is unavailable, since emulation is less reliable and costs extra tokens. See [Tools](tools.md) for the tool system itself.

## Where to go next

- [Agents & workflows](agents.md): hand an `LLMClient` to an `Agent`, or configure one declaratively with `AgentSpec` and the `"provider:model"` string.
- [Tools](tools.md): give the model functions to call; `tool_calls` on `LLMResponse` carries the requests.
- [Context engineering](context-engineering.md): how `context_window` and `max_output_tokens` feed the window budget.
- [Observability](observability.md): trace and account for calls across a run.
