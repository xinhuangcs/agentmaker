# LLM 客户端与厂商

`LLMClient` 是与模型对话的唯一入口。你指定一个厂商（provider，即模型提供方，可选再指定具体模型），客户端便会替你解析出 API key、访问端点（endpoint）以及底层通信协议，然后只对外暴露两个异步方法：`chat()` 用于一次性获取完整回复，`stream()` 用于逐 token（token 即模型处理文本的最小单位）流式输出。无论某个厂商说的是 OpenAI 兼容协议、Anthropic 原生协议还是 Gemini 原生协议，返回的都是同一套统一的 `LLMResponse`，因此你其余的代码永远不必按厂商分支处理。

当你需要直接、原始地访问模型时，就用 `LLMClient`。而当你构建一个 [Agent](agents.md) 时，通常是把一个 `LLMClient` 交给它（或让 `AgentSpec` 从一个 `"provider:model"` 字符串来构造），你自己不会去调用 `chat()`。

```python
from agentmaker import LLMClient

llm = LLMClient("deepseek")                       # provider's default model (deepseek-v4-flash)
resp = await llm.chat([{"role": "user", "content": "Hello"}])
print(resp.content)
```

!!! note
    `chat()` 和 `stream()` 都是协程（coroutine）：框架从内核起就是异步的。请在事件循环里用 `await` 运行它们，或者通过 `agentmaker.core.aio` 里的门面（facade）从同步代码调用（`run_sync(llm.chat(...))` / `iter_sync(llm.stream(...))`）。

## 选择厂商与模型

第一个位置参数是厂商名，默认值为 `"deepseek"`。如果省略 `model`，客户端会使用该厂商内置的 `default_model`（各家云厂商最便宜的真实模型）。传入 `model=` 即可切换，它始终具有最高优先级。

```python
LLMClient()                                        # deepseek + deepseek-v4-flash
LLMClient("openai")                                # openai's default (gpt-4.1-nano)
LLMClient("openai", model="gpt-5.4-nano")          # explicit model, highest priority
LLMClient("anthropic")                             # Claude native, default haiku
LLMClient("gemini")                                # Gemini native, default flash-lite
```

本地部署、自托管以及代理类厂商没有默认模型，因此你必须显式传入 `model=`：

```python
LLMClient("openai_compatible", api_key="x", base_url="http://host/v1", model="my-model")
```

未知的厂商会抛出 `LLMConfigError` 并列出内置可选项。如果你不小心在应传厂商名的位置传了一个模型名（例如 `LLMClient("gpt-5")`），错误信息会提示你改用 `LLMClient(provider, model=...)`。

### `"provider:model"` 字符串形式

声明式配置（[`AgentSpec`](agents.md)）接受用冒号约定把模型写成单个字符串。`build_agent` 会把它拆分成一个 `LLMClient`：

- `"deepseek:deepseek-v4-flash"` 变成 `LLMClient("deepseek", model="deepseek-v4-flash")`。
- 不带冒号、仅有厂商名（`"deepseek"`）变成 `LLMClient("deepseek")`，使用该厂商的默认模型。
- 冒号右半为空（`"deepseek:"`）时，模型回退为该厂商的默认模型。

当你想自己固定 key 或 base URL 时，也可以直接传入一个 `LLMClient` 实例。这套冒号语法只存在于 `AgentSpec.model` 上；`LLMClient` 本身始终把 `provider` 和 `model` 作为两个独立参数接收。

## 内置厂商

厂商按其所说的通信协议分组。新增一个 OpenAI 兼容厂商只需一行配置，因此大多数条目共用同一个适配器（adapter）。`default_model` 列显示的是省略 `model=` 时所用的模型；短横线表示模型由用户选择、必须显式传入。

### OpenAI 兼容协议

| 厂商 | 默认模型 | API key 环境变量 | 结构化输出 |
| --- | --- | --- | --- |
| `openai` | `gpt-4.1-nano` | `OPENAI_API_KEY` | `json_schema` |
| `deepseek` | `deepseek-v4-flash` | `DEEPSEEK_API_KEY` | `json_object` |
| `dashscope` | `qwen-flash` | `DASHSCOPE_API_KEY` | `json_object` |
| `moonshot` | `moonshot-v1-8k` | `MOONSHOT_API_KEY` | `json_object` |
| `zhipu` | `glm-4.7-flash` | `ZHIPUAI_API_KEY`、`ZAI_API_KEY`、`ZHIPU_API_KEY` | `json_object` |
| `modelscope` | （传 `model=`） | `MODELSCOPE_API_KEY` | `none` |
| `gemini_openai` | `gemini-3.1-flash-lite` | `GEMINI_API_KEY`、`GOOGLE_API_KEY` | `json_schema` |
| `ollama` | （传 `model=`） | （本地占位 key） | `none` |
| `vllm` | （传 `model=`） | （本地占位 key） | `none` |
| `sglang` | （传 `model=`） | （本地占位 key） | `none` |
| `openai_compatible` | （传 `model=`） | `LLM_API_KEY`、`OPENAI_API_KEY` | `none` |

### Anthropic 原生协议

| 厂商 | 默认模型 | API key 环境变量 | 结构化输出 |
| --- | --- | --- | --- |
| `anthropic` | `claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` | native |

### Gemini 原生协议

| 厂商 | 默认模型 | API key 环境变量 | 结构化输出 |
| --- | --- | --- | --- |
| `gemini` | `gemini-3.1-flash-lite` | `GEMINI_API_KEY`、`GOOGLE_API_KEY` | native |

!!! note
    模型名与端点可能变化。请把这张表当作发布时的默认值，而非永久保证。当你想用上某家厂商完整的原生特性集时，请使用 `gemini` / `anthropic`（原生协议）；`gemini_openai` 则是 Gemini 的 OpenAI 兼容垫片（shim）。

### 已登记模型的限制

profile 中的限制只适用于该厂商的默认模型。显式传入另一个已登记的模型 ID 时，`LLMClient` 会解析该型号自己的上下文窗口与最大输出。相关的精确 token 限制如下：

| 模型 ID | 上下文窗口 | 最大输出 |
| --- | ---: | ---: |
| `gpt-4.1-nano` | 1,047,576 | 32,768 |
| `gpt-5.6`、`gpt-5.6-sol`、`gpt-5.6-terra`、`gpt-5.6-luna`、`gpt-5.5`、`gpt-5.4` | 1,050,000 | 128,000 |
| `gpt-5.4-mini`、`gpt-5.4-nano` | 400,000 | 128,000 |
| `deepseek-v4-flash`、`deepseek-v4-pro` | 1,000,000 | 393,216 |
| `gemini-3.1-flash-lite`、`gemini-3.5-flash` | 1,048,576 | 65,536 |
| `glm-5.2` | 1,000,000 | 131,072 |
| `glm-4.7-flash`、`glm-5.1`、`glm-5`、`glm-5-turbo`、`glm-4.7` | 204,800 | 131,072 |
| `claude-fable-5`、`claude-opus-4-8`、`claude-sonnet-5`、`claude-sonnet-4-6` | 1,000,000 | 128,000 |
| `qwen-flash`、`qwen3.6-flash` | 1,000,000 | 32,768 |
| `qwen3.5-flash` | 1,000,000 | 65,536 |
| `kimi-k2.7-code`、`kimi-k2.6`、`kimi-k2.5` | 262,144 | 未登记 |
| `moonshot-v1-8k` | 8,192 | 8,192 |
| `moonshot-v1-128k` | 131,072 | 未登记 |

Moonshot 未公布独立的输出上限（单次调用最多输出「窗口减去输入」），因此注册表对 Kimi K2.x 与 `moonshot-v1-128k` 不设输出值，你传入的 `max_tokens` 按原样生效。DeepSeek 官方标注的「384K」输出展开为 393,216（二进制 K）。DashScope 仍以成本更低的 `qwen-flash` 为默认模型；需要 `qwen3.6-flash` 时应显式指定。

## 凭据与端点

你很少需要在代码里传 key。`LLMClient` 会通过一条回退链来解析 API key：

1. 显式传入的 `api_key=` 参数。
2. 该厂商专属的环境变量，按上表所列顺序依次尝试。
3. 通用的 `LLM_API_KEY` 环境变量。
4. 本地占位 key（用于 `ollama` 这类不校验 key 的服务）。

如果都解析不到，构造时会抛出 `LLMConfigError` 并指明需要设置哪些环境变量。base URL 的解析类似：显式传入的 `base_url=` 优先；通用厂商（`openai`、`openai_compatible`）额外读取 `OPENAI_BASE_URL` / `LLM_BASE_URL`；端点固定的厂商只用自己的端点。原生的 `anthropic` 和 `gemini` 协议会让 `base_url` 保持为 `None`，转而使用各自 SDK 的默认端点。

最稳妥的做法是设置好对应的环境变量，然后只用厂商名来构造：

```python
llm = LLMClient("openai")     # reads OPENAI_API_KEY from the environment
```

## 异步 chat

`chat()` 发送消息并返回单个 `LLMResponse`。消息是一个由 `{"role", "content"}` 字典组成的列表（role 取值为 `user`、`assistant`、`system`、`tool`）。

```python
resp = await llm.chat(
    [{"role": "user", "content": "Summarize async I/O in one sentence."}],
    temperature=0.2,
    max_tokens=200,
)
print(resp.content)
print(resp.usage, resp.finish_reason)
```

`temperature` 和 `max_tokens` 都是可选的。默认情况下客户端根本不发送 temperature，而是沿用模型服务端自身的默认值；当你需要确定性输出时，可按次调用传入 `temperature=`（或在构造器上设置 `default_temperature=`）。多余的关键字参数会原样透传给底层 SDK。

### 客户端生命周期

底层 SDK client 按 event loop 缓存。应用持有客户端时应使用 `async with LLMClient(...)`，或调用 `await llm.aclose()`。同步代码可结合 `with LLMClient(...)` 与 `run_sync` 使用；`close()` 会释放该线程的 SDK client。临时同步工作线程退出时，同步桥还会自动关闭已登记的 client、残留任务和 resident loop。

## 流式输出

`stream()` 是一个异步生成器，会随着模型的产出逐块 yield 文本增量（delta）。用 `async for` 消费它。下面的示例完全 hermetic（自洽、无副作用：不需要 API key、不联网），用的是 `ScriptedLLM` 这个测试替身（test double），它复刻了真实客户端的 `chat()` / `stream()` 接口：

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

换成真实客户端，形态完全一致：

```python
async for piece in llm.stream([{"role": "user", "content": "Tell a joke"}]):
    print(piece, end="")
```

不传工具时，每个产出片段都是字符串。通过 `tools=` 传入原生工具 schema 时，支持流式函数调用的适配器会先产出文本增量，最后再产出恰好一个终态 `LLMResponse`；其中汇总的 `tool_calls` 供 Agent 的流式工具循环使用。文本工具模拟不会在流式路径上模拟调用：`ToolEmulationAdapter` 只流式输出纯文本，配置了 `emulate_tools=True` 的 Agent 也会拒绝流式工具循环。模拟工具请使用非流式的 `chat` / `run` / `arun`。

### 流式统计

每次调用的流式元数据（metadata）与产出的文本、可选的终态工具响应分开存放。在流耗尽之后，读取 `llm.last_stream_stats`（若你还没做过流式调用，则为 `None`）。它暴露 `model`、`finish_reason`、`usage` 和 `latency_ms`。对于 OpenAI 系列厂商，客户端会自动在流式请求中带上 `stream_options={"include_usage": True}`，所以 `usage` 通常有值；只有当后端本身不上报流式用量时才会是 `None`。

```python
async for piece in llm.stream([{"role": "user", "content": "hi"}]):
    print(piece, end="")
stats = llm.last_stream_stats
print(stats.model, stats.latency_ms)
```

在一个共享客户端上并发进行多路流式时，`last_stream_stats` 可能被覆写。改为传入 `on_stats` 回调，即可在本次调用的流结束时可靠地拿到它自己的 stats 对象：

```python
collected = []
async for piece in llm.stream(messages, on_stats=collected.append):
    ...
```

## `LLMResponse`

每次非流式调用都返回一个 `LLMResponse` dataclass；传入工具的流式调用也会在文本增量之后产出一个终态 `LLMResponse`。字段如下：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `content` | `str` | 回复文本。也是 `str(resp)` / `print(resp)` 所显示的内容。 |
| `finish_reason` | `str \| None` | 生成停止的原因。 |
| `model` | `str` | 实际使用的模型名。 |
| `usage` | `dict \| None` | token 用量（可能包含嵌套的明细结构，不只是整数）。 |
| `reasoning_content` | `str \| None` | 独立的推理轨迹，当模型返回它时才有。 |
| `tool_calls` | `list \| None` | OpenAI 格式的函数调用工具调用（tool call），可直接回喂到 `messages`；没有时为 `None`。 |
| `latency_ms` | `int` | 往返延迟，单位毫秒。 |
| `raw` | `Any` | 厂商的原始响应对象。 |
| `assistant_message` | `dict \| None` | 工具轮次所需、可 JSON 序列化的厂商续接状态；Agent 会自动回喂。 |

```python
resp = await llm.chat([{"role": "user", "content": "hi"}])
print(resp.content)          # the text
print(resp)                  # same thing: __str__ returns content
```

## 结构化输出

向 `chat()` 传入 `output_schema=`（一个 JSON Schema 字典），即可要求模型输出符合该 schema 的 JSON。适配器会按照厂商的能力（上文表格里的 `structured_output` 列）来转译该 schema：

- `json_schema`：schema 在 API 层通过 `response_format` 携带（例如 `openai`、`gemini_openai`）。
- `json_object`：请求只保证输出合法 JSON；schema 通过提示词（prompt）注入，事后再校验（例如 `deepseek`、`dashscope`、`moonshot`、`zhipu`）。
- `none`：不发送 `response_format`；仅靠提示词兜底（本地、代理以及未知厂商）。
- native：`anthropic` 和 `gemini` 协议始终走它们各自的原生结构化路径。

以上是底层 `chat()` 的视角。关于 agent 层面在 `run()` 上使用 `output_schema`（含自动重试与 Pydantic 校验），见 [结构化输出](structured-output.md)。

多数场景下你不会直接调用它。[Agent](agents.md) 层接受一个 Pydantic 模型作为 `output_schema`，驱动这套机制，然后把 JSON 校验成一个实例，失败时自动重试。这条更省心的路径可以 hermetic 测试：

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

## 消息与多模态内容

一条消息的 `content` 要么是纯字符串（常见情形），要么是一组厂商中立的内容片段（content part）列表。`Message` dataclass 用 `role`、`content`、`timestamp` 以及一个 `metadata` 字典来建模一条消息；调用 `to_dict()` 即可得到 `chat()` 和 `stream()` 所消费的 `{"role", "content"}` 形态。

```python
from agentmaker import Message

msg = Message(content="Hello", role="user")
await llm.chat([msg.to_dict()])
```

要在一条消息里同时发送文本和图像，用这些片段辅助函数（都可从顶层导入）来构建内容列表：

- `text_part(text)` 构建一个文本片段。
- `image_part_from_bytes(data, media_type)` 从原始字节构建一个内联图像。
- `image_part_from_file(path, media_type=None)` 读取一个本地文件（省略时会从后缀推断 media type）。
- `image_part_from_url(url)` 引用一张由厂商去抓取的远程图像。

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

可接受的图像 media type 有 `image/jpeg`、`image/png`、`image/gif` 和 `image/webp`；不受支持的类型会在构造时就抛出 `ValueError`，而不是等到服务端才失败。`image_part_from_url` 不被 Gemini 适配器支持（在那里请改用内联片段）。

### 视觉能力门控

每个厂商 profile 都带有一个 `supports_vision` 标志。当它明确为 `False`（例如 `deepseek`）时，发送图像片段会在任何网络调用之前就抛出 `LLMConfigError`，这样一个纯文本厂商会以清晰的报错失败，而不是给出一个令人困惑的服务端错误。当它未知（`None`）时，框架不会拦截，交由服务端决定。如果你确知某个具体模型能接受图像，可用 `LLMClient(..., supports_vision=True)` 逐客户端覆盖。

## 自定义厂商与协议

你不必修改框架就能新增一个厂商。传入一个 `ProviderProfile` 即可复用现有协议，无需改动源码：

```python
from agentmaker import LLMClient, ProviderProfile

llm = LLMClient(
    provider="myvendor",
    profile=ProviderProfile(base_url="https://api.myvendor.com/v1", key_envs=("MY_KEY",), default_model="m"),
    model="m",
)
```

若要接入一个全新的通信协议，请以某个协议名注册一个适配器类（`BaseAdapter` 的子类），然后在 profile 里引用该协议名：

```python
from agentmaker.core.adapters import register_adapter

register_adapter("myproto", MyAdapter)   # MyAdapter is your BaseAdapter subclass
LLMClient("myvendor", profile=ProviderProfile(protocol="myproto", default_model="m", key_envs=("MYVENDOR_API_KEY",)))
```

对于缺少原生函数调用能力的模型，`LLMClient(..., emulate_tools=True)` 会用一层文本模拟垫片包住适配器，让使用工具的 Agent 通过非流式 `chat` / `run` / `arun` 工作；它不支持流式工具循环。仅在原生函数调用不可用时才启用它，因为模拟方式可靠性较低、还会额外消耗 token。工具系统本身参见 [工具](tools.md)。

## 下一步去哪里

- [Agent 与工作流](agents.md)：把一个 `LLMClient` 交给 `Agent`，或用 `AgentSpec` 加 `"provider:model"` 字符串以声明式方式配置它。
- [工具](tools.md)：给模型可调用的函数；`LLMResponse` 上的 `tool_calls` 承载这些请求。
- [上下文工程](context-engineering.md)：`context_window` 与 `max_output_tokens` 如何参与窗口预算的计算。
- [可观测性](observability.md)：在一次运行中对调用进行追踪与统计。
