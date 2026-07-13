# 测试

`agentmaker.testing` 为你提供确定性的测试替身（test double，即用来在测试里顶替真实依赖的假实现），去替换 Agent 中那些原本要花钱或要访问网络的部分：LLM、embedder（嵌入器，把文本转成向量的组件）、检查点存储，以及生命周期钩子。把它们换进去，你的 Agent 测试就能自洽（hermetic）运行：不需要 API key、不联网、也不会有偶发抖动，从而可以针对一组预先编排好的模型响应，精确断言你的 Agent 到底做了什么。只要你为基于本框架构建的 Agent、tool、human-in-the-loop（HITL，即「人在回路」，高风险动作需人工确认的流程）或检索接线编写单元测试，就该用到这个模块。

这些工具不会从顶层 `agentmaker` 命名空间重新导出，请直接从子模块导入：

```python
from agentmaker.testing import ScriptedLLM, FakeEmbedder, MemoryCheckpointStore, RecordingHook
```

本模块只定义了这四个公开的测试替身：

| 替身 | 替换对象 | 用来测试什么 |
| --- | --- | --- |
| `ScriptedLLM` | LLM 客户端 | 普通对话、工具循环、流式输出，所有决策全部脚本化 |
| `FakeEmbedder` | 一个 `Embedder` | 用确定性向量测试检索接线 |
| `MemoryCheckpointStore` | 一个 `CheckpointStore` | 在内存中测试 HITL 挂起 / 恢复以及崩溃恢复 |
| `RecordingHook` | 一个 `Hook` | 验证生命周期事件是否按预期顺序触发 |

## ScriptedLLM

`ScriptedLLM` 按调用顺序发出预设的响应。它以鸭子类型（duck typing，即「只要行为像就当它是」）实现 LLM 客户端契约（它暴露了 `provider`、`model`、`supports_function_calling`、`context_window`、`chat` 和 `stream`），但并不继承真实客户端，所以构造它既不会触发 API key 校验，也不会产生网络调用。由于行为完全由脚本决定，`chat` 会忽略传入的 `messages` 和 `tools`。`stream` 同样会忽略 `messages`，但当传入了 `tools` 时，它会在文本之后额外产出一个终结性的 `LLMResponse`，与真实适配器发出工具循环通道信号的方式一致。

向构造函数传入一个脚本条目列表。每个条目要么是一个普通的 `str`（文本回复），要么是一个现成的 `LLMResponse`（用于精确控制工具调用、用量或结束原因）：

```python
from agentmaker import Agent
from agentmaker.testing import ScriptedLLM

agent = Agent("assistant", ScriptedLLM(["Hello.", "Goodbye."]))
assert agent.run("hi").final_output == "Hello."
assert agent.run("bye").final_output == "Goodbye."
```

`agent.run(...)` 返回一个 `RunResult`。它的 `final_output` 字段保存已完成的回答（见下方 [对一次运行做断言](#对一次运行做断言)）。

构造函数签名为：

```python
ScriptedLLM(script=None, *, model="test", provider="test",
            supports_function_calling=True, context_window=None)
```

- `supports_function_calling` 是模型能力标志。启用了工具的 Agent 会在构造时对它做校验，所以传 `False` 可以测试「不支持函数调用」的分支（对这样的客户端构造一个带 tools 的 `Agent` 会抛出 `ValueError`）。
- `context_window` 默认保持为 `None`（未知），此时不会触发任何窗口预算削减。传入一个具体整数即可演练上下文窗口的预算控制。

### 编排一次工具调用

要让脚本化的模型「决定」调用某个 tool，请使用 `ScriptedLLM.tool_call` 辅助方法来构建工具调用响应，而不是手写底层的 tool-calls 结构：

```python
ScriptedLLM.tool_call(name, arguments=None, *, call_id="call_1", content="") -> LLMResponse
```

一个典型的工具循环会编排两个条目：先是模型请求调用 tool，然后写出最终答案。以下取自随包附带的 [`examples/01_quickstart.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/01_quickstart.py)：

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

当 tool 来自 [`ToolRegistry`](tools.md)（包括内置工具）时，同样的模式也适用。注意，你传给 `tool_call` 的名字是该 tool 注册时的名字（这里内置的计算器是 `"calculator"`），以下取自 [`examples/02_tools_and_registry.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/02_tools_and_registry.py)：

```python
from agentmaker import Agent, CalculatorTool, ToolRegistry, tool
from agentmaker.testing import ScriptedLLM


@tool
def to_upper(text: str) -> str:
    """Uppercase a string.

    Args:
        text: The input text.
    """
    return text.upper()


registry = ToolRegistry()
registry.register(CalculatorTool())   # built-in: safe arithmetic evaluation
registry.register(to_upper)           # your custom tool

# Script the model's decision to call the calculator, then its final answer.
llm = ScriptedLLM([
    ScriptedLLM.tool_call("calculator", {"expression": "(3 + 4) * 5"}),
    "The result is 35.",
])
agent = Agent("assistant", llm, tool_registry=registry)
print(agent.run("Compute (3 + 4) * 5").final_output)
```

### 统计调用次数

`ScriptedLLM` 会在其 `calls` 属性中记录已消耗了多少个脚本条目，这便于断言一个工具循环恰好用了你预期的模型轮数（一轮请求工具、一轮作答）：

```python
llm = ScriptedLLM([ScriptedLLM.tool_call("get_weather", {"city": "Oslo"}), "Sunny."])
agent = Agent("assistant", llm, tools=[get_weather])
agent.run("weather in Oslo?")
assert llm.calls == 2
```

### 脚本耗尽

如果 Agent 请求的响应比脚本提供的多出一条，`ScriptedLLM` 会抛出 `AssertionError`（并报告是第几次调用耗尽了脚本、脚本共有多少条目），而不是悄悄返回某个意料之外的东西。这会把「我的循环迭代次数超出预期」变成一次响亮、即时的测试失败：

```python
import pytest

agent = Agent("assistant", ScriptedLLM(["only one reply"]))
agent.run("a")
with pytest.raises(AssertionError, match="script exhausted"):
    agent.run("b")
```

### 流式输出

`ScriptedLLM.stream` 会把下一条响应的内容切成小片段逐一产出，与真实适配器的契约保持一致。通过 Agent 的同步流式外观（facade）来驱动它，并把这些片段重新拼接起来：

```python
agent = Agent("assistant", ScriptedLLM(["A streamed reply."]))
pieces = list(agent.stream_run("hi"))
assert "".join(pieces) == "A streamed reply."
```

内容为空时不会产出任何 chunk（空字符串产生的是空的流，而不是单个 `""`）。运行结束时，测试替身会把该响应的用量和结束原因记录到 `last_stream_stats` 上，与真实适配器上报流式统计的方式一致。若要对流式的用量或结束原因做断言，请编排一个完整的 `LLMResponse` 而不是一个裸字符串：

```python
from agentmaker.core.llm_response import LLMResponse

llm = ScriptedLLM([LLMResponse(content="hello world", model="test",
                               finish_reason="stop", usage={"total_tokens": 5})])
```

!!! note
    `LLMResponse` 位于 `agentmaker.core.llm_response`。只有当你想固定用量、`finish_reason` 或某个精确的 `tool_calls` 载荷时才需要它；对于普通的文本回复，一个纯字符串脚本条目就足够了。

## 对一次运行做断言

每一次脚本化运行返回的都是一个 `RunResult`，所以你的断言读取的字段与你在生产环境中读取的完全一样。最常断言的几个：

- `final_output`：已完成的回答（一个 `str`，或者当 Agent 被指定了输出 schema 时是一个结构化实例）；当运行因等待审批而挂起时为 `None`。
- `status`：`"completed"` 或 `"interrupted"`。
- `usage`：一个 `RunUsage` 快照，包含 `llm_calls`、`tool_calls` 和 `total_tokens`。

完整字段列表（`interrupt`、`new_messages`、`run_id` 等）见 [Agent 与工作流](agents.md)。

```python
result = agent.run("What's the weather in Copenhagen?")
assert not result.interrupted
assert result.final_output == "It's sunny and 24C in Copenhagen today."
assert result.usage.tool_calls == 1
```

## FakeEmbedder

`FakeEmbedder` 是一个确定性、离线的 `Embedder`：相同的文本总是映射到相同的向量，不同的文本映射到不同的向量（每个向量都由一个 SHA-256 哈希推导得出并做 L2 归一化，因此余弦相似度依然有意义，检索也能真正区分不同的条目）。用它来测试检索和 RAG（Retrieval-Augmented Generation，检索增强生成，即先检索资料再让模型据此作答）的接线，无需调用真实的 embedding API。

```python
from agentmaker.testing import FakeEmbedder

emb = FakeEmbedder(dim=8)
assert emb.dim == 8
assert emb.model_id == "fake-embedder-8"

# Same text yields the same vector; different text yields a different one.
assert emb.embed(["cat"]) == emb.embed(["cat"])
assert emb.embed(["dog"]) != emb.embed(["cat"])
```

构造函数接收向量宽度，`FakeEmbedder(dim=8)`。`embed(texts)` 为每一段输入文本返回一个向量，并且该替身暴露了检索栈所需的 `dim` 和 `model_id` 属性，因此你可以把它直接塞进 [检索与 RAG](retrieval-and-rag.md) 中描述的各个组件。

## MemoryCheckpointStore

`MemoryCheckpointStore` 是一个进程内的 `CheckpointStore`，它把每次运行的执行状态保存在一个普通字典里，而不是写到磁盘上。它的存在是为了让你无需触碰文件系统就能测试 human-in-the-loop 的挂起 / 恢复以及崩溃恢复。用无参构造它，并作为 `checkpoint_store` 传给 Agent。

当 Agent 尝试运行一个需要确认的 tool 时，运行会挂起并返回一个 `interrupted == True` 的 `RunResult`。通过用挂起时携带的作用域调用 `resume(True, scope=...)` 来批准它：

```python
from agentmaker import Agent, Tool, ToolParameter, ToolResponse
from agentmaker.testing import MemoryCheckpointStore, ScriptedLLM


class DeleteTool(Tool):
    requires_confirmation = True   # high-risk: routes through the HITL confirmation gate

    def __init__(self):
        super().__init__("delete", "Delete a path")

    def get_parameters(self):
        return [ToolParameter("path", "string", "Target path")]

    def run(self, parameters):
        return ToolResponse.ok(f"deleted {parameters.get('path')}")


llm = ScriptedLLM([ScriptedLLM.tool_call("delete", {"path": "/tmp/a"}), "Done."])
agent = Agent("assistant", llm, tools=[DeleteTool()],
              checkpoint_store=MemoryCheckpointStore())

result = agent.run("delete /tmp/a")
assert result.interrupted
assert result.interrupt.pending.tool_name == "delete"

resumed = agent.resume(True, scope=result.interrupt.scope)
assert resumed.final_output == "Done."
```

完整的审批模型（包括拒绝动作以及一次性批准多个待处理调用）见 [护栏与人在回路](guardrails-and-hitl.md)。

## RecordingHook

`RecordingHook` 是一个 `Hook`，它会把收到的每一个生命周期事件以 `(event_name, key_param)` 元组的形式追加到自己的 `events` 列表中。用它来断言钩子的分发是否按你预期的顺序发生。通过 `hooks=[...]` 把它传给 Agent：

```python
from agentmaker import Agent
from agentmaker.testing import RecordingHook, ScriptedLLM

hook = RecordingHook()
Agent("assistant", ScriptedLLM(["Answer."]), hooks=[hook]).run("Question?")

names = [name for name, _ in hook.events]
assert names[0] == "on_run_start"
assert "before_model" in names
assert "after_model" in names
assert names[-1] == "on_run_end"
assert hook.events[-1][1] == "Answer."   # on_run_end carries the final output
```

`RecordingHook` 记录运行级事件（`on_run_start`、`on_run_end`、`on_interrupt`、`on_error`）、模型事件（`before_model`、`after_model`）、工具事件（`before_tool`、`after_tool`）以及 `on_guardrail_trip`。每个元组的第二个元素是该事件的一个小的关键参数（例如 `on_run_start` 的输入文本、`before_tool` 的 tool 名字，或 `on_run_end` 的最终输出），它让你无需搭建一个完整的 mock 就能断言事件携带了什么。关于这些事件在生产环境中观测的内容，见 [可观测性](observability.md)。

## 运行你的测试

由于这四个替身都是隔离的，基于它们构建的测试套件不需要 API key、也不需要网络访问，因此在笔记本上和在 CI 中运行结果一致。如果你还没构建过 Agent，请从 [快速上手](quickstart.md) 开始，然后用上面这些替身把你关心的每一个行为都包进一条断言里。
