# 快速开始

本指南用十几行代码构建一个可运行的 Agent（智能体，能自主调用工具完成任务的程序）：把一个函数变成工具（tool），用一个脚本化的测试模型顶替真实的 LLM（大语言模型），因此无需 API key、无需联网即可运行，随后 Agent 以「模型在一个循环里调用工具」的方式跑完一轮，交回最终答案。如果你刚接触 agentmaker，请先读这一篇；其余每篇指南都默认你脑中已有这个基本形态。本文逐行讲解 [`examples/01_quickstart.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/01_quickstart.py)，再演示如何把测试模型换成真实的厂商。

## 完整程序

下面是示例的原文照录。它零配置：无需 API key，也无需联网。你可以这样运行：

```bash
uv run python examples/01_quickstart.py
```

```python
from agentmaker import Agent, tool
from agentmaker.testing import ScriptedLLM


@tool
def get_weather(city: str) -> str:
    """Return today's weather for a city.

    Args:
        city: The city name.
    """
    return f"{city}: sunny, 24C"


# With a real model the LLM decides when to call the tool. Here we script that decision:
# first it asks to call get_weather(city="Copenhagen"), then it writes the final answer.
llm = ScriptedLLM([
    ScriptedLLM.tool_call("get_weather", {"city": "Copenhagen"}),
    "It's sunny and 24C in Copenhagen today.",
])

agent = Agent("assistant", llm, tools=[get_weather])
result = agent.run("What's the weather in Copenhagen?")
print(result.final_output)
```

运行后打印：

```text
It's sunny and 24C in Copenhagen today.
```

本页余下部分逐块讲解每个部件。

## 用 `@tool` 定义一个工具

```python
@tool
def get_weather(city: str) -> str:
    """Return today's weather for a city.

    Args:
        city: The city name.
    """
    return f"{city}: sunny, 24C"
```

`@tool` 一行就能把一个带类型标注的函数变成 `Tool` 对象。被装饰之后，`get_weather` 不再是普通函数，而是一个可以交给 Agent 使用的 `Tool`。装饰器会读取函数本身来构建模型所需的 schema（function calling，即「函数调用」，是让 LLM 发出结构化的「用这些参数调用这个工具」指令的机制）：

- **参数、类型、默认值、是否必填**来自函数签名。这里 `city: str` 成为一个必填的字符串参数。
- **工具描述**来自 docstring 的第一行。
- **参数描述**来自 `Args:` 段落（如果你使用了 `Annotated[...]`，也可以来自其元数据）。

每个参数都必须带类型标注。缺少标注、出现 `*args` / `**kwargs` 参数，或使用了无法映射到 JSON 的类型（可映射的有 `str`、`int`、`float`、`bool`、`list`、`dict`，以及它们的 `Optional` / `Annotated` 包装），都会在注册时立即抛出 `ToolRegistrationError`，而不是等到后面才悄无声息地失败。

`@tool` 还接受用于特殊工具的关键字选项，例如给必须经过确认关卡的高风险动作加 `@tool(requires_confirmation=True)`，或给可与同一轮内其它工具并发执行的只读工具加 `@tool(supports_parallel=True)`。完整选项见 [工具](tools.md)。

## 用 `ScriptedLLM` 脚本化模型

```python
llm = ScriptedLLM([
    ScriptedLLM.tool_call("get_weather", {"city": "Copenhagen"}),
    "It's sunny and 24C in Copenhagen today.",
])
```

`ScriptedLLM` 是一个测试替身（test double）：它按调用顺序吐出预设好的响应，而不去联系真实厂商，因此 Agent 测试既不产生费用也不需要联网。它位于 `agentmaker.testing`，不属于顶层公开接口，所以要显式导入 `from agentmaker.testing import ScriptedLLM`。

脚本列表中的每一项，要么是：

- 一个普通的 `str`，会变成一条文本回复；要么是
- 一个 `LLMResponse`，让你精确控制工具调用及其它字段。

`ScriptedLLM.tool_call(name, arguments)` 是一个辅助函数，用来构建后一种：一个表示「模型请求调用 `name(arguments)`」的 `LLMResponse`，这样你就不必自己手工拼出工具调用的结构。

于是这段脚本表达的是：第一轮，请求调用 `get_weather(city="Copenhagen")`；第二轮，等工具结果回来后，用最后那句话作答。每次调用按顺序消耗下一项。如果 Agent 请求的响应比脚本提供的多出一次，`ScriptedLLM` 会抛出 `AssertionError`，告诉你还缺几项，这通常意味着循环多走了你没料到的一轮。

!!! note "为什么要脚本化这次工具调用？"
    在真实模型下，是 LLM 自己决定何时调用 `get_weather`。`ScriptedLLM` 只是让你把这个决定钉死，使测试具有确定性。无论哪种方式，Agent 的循环行为都完全一致，这正是这个测试有意义的原因。

## 构造 `Agent`

```python
agent = Agent("assistant", llm, tools=[get_weather])
```

`Agent` 是框架的核心执行原语：一个输入进来，模型跑一轮工具循环，一条回复出去。这里的三个参数是：

- `"assistant"`：Agent 的名字。
- `llm`：LLM 客户端（这里是 `ScriptedLLM` 替身；之后会换成真实的 `LLMClient`）。
- `tools=[get_weather]`：模型可调用的工具列表。这是一行式的便捷入口，接受一个 `Tool` 对象的列表（包括被 `@tool` 装饰的函数）。

没有单独的注册步骤：传入 `tools` 就够了。如果你省略 `tools`，Agent 就只做纯粹的问答，不走工具循环。有用的额外关键字参数（都可选）包括 `system_prompt=`（设定角色人设）和 `max_turns=`（限定循环最多可走多少轮模型调用，默认 `10`），后者用于防止工具循环永不终止。完整参数列表见 [Agent 与工作流](agents.md) 指南。

## 运行并读取 `final_output`

```python
result = agent.run("What's the weather in Copenhagen?")
print(result.final_output)
```

`agent.run(...)` 执行整个循环并返回一个 `RunResult`。在这一次调用背后，循环做了这些事：

1. 把用户消息发给模型。模型以脚本化的工具调用 `get_weather(city="Copenhagen")` 作答。
2. 框架执行该工具，再把它的结果作为一条工具消息喂回给模型。
3. 再发一次。这一次模型回复纯文本、不带工具调用，于是循环结束，那段文本就是答案。

`RunResult` 是对结果的统一封装，而不是一个裸字符串。它的主字段是 `final_output`，即本次运行完成后的答案（这里是字符串；如果你请求了 [结构化输出](structured-output.md)，则是一个结构化实例）。其它字段可用来审视这次运行：

- `result.status`：`"completed"` 或 `"interrupted"`。
- `result.interrupted`：一个便捷布尔值，当运行因等待人工审批而挂起时为 `True`（见 [护栏与人在回路](guardrails-and-hitl.md)）。
- `result.usage`：一个 `RunUsage` 快照，含 `llm_calls`、`tool_calls` 和 `total_tokens`。
- `result.new_messages`：本轮加入历史的用户消息与助手消息。
- `result.run_id`：本次运行的 trace 关联 id。

对大多数简单场景，你读一下 `final_output` 就可以继续了：

```python
result = agent.run("What's the weather in Copenhagen?")
print(result.final_output)                 # the answer text
print(result.usage.tool_calls)             # 1 (get_weather ran once)
```

!!! note "异步对应版本"
    `agent.run(...)` 是同步入口。框架以异步优先（async-first）为设计，因此 `await agent.arun(...)` 是异步版本，返回同样的 `RunResult`。在 `async def` 代码里用它；在普通脚本里用 `run`。

## 换成真实模型

唯一要改的只有 LLM 这一行。把 `ScriptedLLM(...)` 换成 `LLMClient`，现在就由模型自己决定何时调用 `get_weather`：

```python
from agentmaker import Agent, LLMClient, tool


@tool
def get_weather(city: str) -> str:
    """Return today's weather for a city.

    Args:
        city: The city name.
    """
    return f"{city}: sunny, 24C"


llm = LLMClient("deepseek")                 # reads DEEPSEEK_API_KEY from the environment
agent = Agent("assistant", llm, tools=[get_weather])
result = agent.run("What's the weather in Copenhagen?")
print(result.final_output)
```

`LLMClient(provider)` 会解析该厂商的配置，并从你的环境中读取它的 API key。provider 默认为 `"deepseek"`，且每家云厂商都预填了默认模型，因此 `LLMClient("deepseek")` 无需再写别的。为你选定的厂商设置对应的 key：

| 调用 | 读取的环境变量 |
| --- | --- |
| `LLMClient("deepseek")` | `DEEPSEEK_API_KEY` |
| `LLMClient("openai")` | `OPENAI_API_KEY` |
| `LLMClient("anthropic")` | `ANTHROPIC_API_KEY` |
| `LLMClient("gemini")` | `GEMINI_API_KEY`（或 `GOOGLE_API_KEY`） |

设置方式：在 shell 里执行 `export DEEPSEEK_API_KEY="sk-..."`（PowerShell：`$env:DEEPSEEK_API_KEY = "sk-..."`），或配合 python-dotenv 使用 `.env` 文件——见[安装页的「厂商 API 密钥」](../installation.md#厂商-api-密钥)。

传入 `model=` 可指定具体模型，例如 `LLMClient("openai", model="gpt-4.1-nano")`。完整的厂商列表、自托管与 OpenAI 兼容端点、以及逐次调用的选项，见 [LLM 客户端与厂商](llm-clients.md)。

其余一切保持不变：`@tool` 定义、`Agent` 构造、`run(...)` 和 `result.final_output` 的行为都完全一致。这正是 `ScriptedLLM` 的意义所在，你的测试代码和生产代码走的是同一个循环。

## 挂更多功能

上面这个 agent 是刻意做到最小的。其余每一样能力，都只是往同一个构造函数里多传几个参数，每个都可选。下面就是给那个 agent 挂上语义长期记忆、一套模型自选的技能库、检索到的上下文，以及一道输入护栏：

```python
from agentmaker import (Agent, LLMClient, Memory, MemoryStore, ContextBuilder,
                          CallableSource, SkillLoader, CallableGuardrail)
from agentmaker.retrieval import build_sqlite_hybrid, OpenAIEmbedder

llm = LLMClient("openai")
memory = Memory(build_sqlite_hybrid(OpenAIEmbedder()), MemoryStore())
skills = SkillLoader("./skills")

agent = Agent(
    "assistant", llm,
    tools=[get_weather],  # function calling
    sources=[CallableSource("memory", memory.search)],  # memory pulled into context each turn
    context_builder=ContextBuilder(),  # assemble context under a token budget
    system_prompt=f"You are a helpful assistant.\nSkills:\n{skills.catalog()}",  # model-invoked skills
    input_guardrails=[CallableGuardrail(lambda t: len(t) < 4000, message="input too long")],  # validate input
)
print(agent.run("Plan a day in Copenhagen, and remember I'm vegetarian.").final_output)
```

这些能力是一次加一样、不必一次全上，同样的写法也能触达框架其余部分：

- **更多 `sources=`**：对已摄入语料的 RAG 检索（分块、查询改写、名次融合、来源引用）与记忆并列。见 [检索与 RAG](retrieval-and-rag.md)。
- **更多 `tools=`**：通过 `MCPClient` 接入 MCP 服务器、通过 `AgentTool` 把子 agent 当工具（orchestrator-worker 多 agent 编排）、以及用 `ToolRetriever` 从庞大工具集中挑选。见 [工具](tools.md)。
- **更聪明的记忆**：`SmartWriter` 从对话里抽取事实、与已存内容做差分，再增 / 改 / 删，而不是照原文存。见 [记忆](memory.md)。
- **其它运行模式**：用 `run(..., output_schema=Model)` 得到[结构化输出](structured-output.md)、用 `async for` 遍历 `agent.astream_run(...)` 做流式、以及 [Agent 与工作流](agents.md) 里的 `PlanAgent` / `ReflectionAgent` 配方。
- **持久化与安全**：会话（session）、检查点（checkpoint，人在回路）、工具权限、历史压缩。见 [护栏与人在回路](guardrails-and-hitl.md) 与 [上下文工程](context-engineering.md)。

## 用 agent 调试

开发时，给它挂上基于 trace 的 agent 调试器。`Tracer` 记录一次运行的每一步，`DoctorHook` 会把一次失败的运行变成由 LLM 撰写的诊断（最早出错的步骤、根因、修复建议），直接打到你的终端：

```python
from agentmaker import Agent, Tracer
from agentmaker.devtools import DoctorHook

tracer = Tracer()
agent = Agent("assistant", llm, tools=[get_weather], tracer=tracer, hooks=[DoctorHook(tracer)])
print(agent.run("What's the weather in Copenhagen?").final_output)
```

`DoctorHook` 以及独立的 Trace 侦探（`python -m agentmaker.devtools`，一个架在录制运行之上的本地网页）本身就是 agentmaker 的 agent，所以框架会诊断它自己的运行。tracing、导出器（exporter）与 Trace 侦探界面详见 [可观测性](observability.md)。

## 下一步去哪里

- [LLM 客户端与厂商](llm-clients.md)：每一家厂商、模型选择与流式输出。
- [Agent 与工作流](agents.md)：完整的 `Agent` 接口，以及 plan-and-execute（先规划后执行）与 reflection（反思）配方。
- [工具](tools.md)：更丰富的工具、确认关卡、并行执行与工具注册表。
- [结构化输出](structured-output.md)：返回一个经过校验的对象而非文本。
- [护栏与人在回路](guardrails-and-hitl.md)：输入与输出护栏，以及对高风险动作的审批。
- [测试](testing.md)：`ScriptedLLM` 及其它测试替身，用于自洽（hermetic，不需要 API key、不联网）的 Agent 测试。
