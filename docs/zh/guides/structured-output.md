# 结构化输出

结构化输出让 agent 返回一个带类型、经过校验的对象，而不是一段自由文本字符串。你用一个 Pydantic 模型来描述想要的数据结构（Pydantic 是一个 Python 库，会按照声明好的 schema 校验数据），把它传给 `run()`，就能拿回一个 `RunResult`，其 `.final_output` 便是该模型的一个实例，且已经解析并校验完毕。只要回复是给代码用而不是给人看的，就该用它：从文本里抽取字段、把内容归类到固定的标签集合，或生成另一套系统要消费的数据负载。

## 基础用法

把一个 Pydantic 模型作为 `output_schema` 传给 `run()`。完成后的 `RunResult` 会在 `.final_output` 里携带这个校验通过的实例：

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

这段是 [`examples/03_structured_output.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/03_structured_output.py) 的逐字拷贝。它无需 API key、无需联网即可运行：`ScriptedLLM` 是框架的测试替身（一个替代品，回放事先准备好的回复，而不去调用真实模型），这里它直接返回那个 JSON 对象，框架把它解析并校验成一个 `Person`。`person` 是一个真正的 `Person` 实例，所以 `person.name` 和 `person.age` 是带类型的属性，而不是字典取值。

想对接真实模型，只要把测试替身换成真实客户端，其余一切保持不变：

```python
from agentmaker import LLMClient

agent = Agent("extractor", LLMClient("deepseek"))
person = agent.run("Extract the person from: Ada is 36.", output_schema=Person).final_output
```

在环境变量里配好对应的 API key（见 [LLM 客户端](llm-clients.md)）。现在由模型决定输出什么，而框架会强制它符合你的 schema。

!!! note
    结构化输出以纯粹的问答方式运行：没有工具循环，也不流式。即便你在构建 agent 时挂了工具，在 `run(..., output_schema=...)` 调用里也不会提供这些工具。当你想拿回单个校验过的对象时用 schema，想让模型采取行动时则用工具循环（见 [工具](tools.md)）。

## 底层发生了什么

当你传入 `output_schema` 时，agent 会把这次调用路由到它的 harness（包裹每一次模型调用的内部层），后者会做以下事情：

1. **推导出一份 JSON Schema**，通过 `model_json_schema()` 从你的模型生成。JSON Schema 是一种标准、语言无关的方式，用来描述一个 JSON 对象必须具备哪些字段和类型。
2. **在前面加上一条系统指令**，要求模型只返回一个符合该 schema 的 JSON 对象，不带任何解释性文字，也不带 markdown 代码围栏。
3. **调用模型**，并附上 schema，然后从回复中抽取 JSON。抽取是宽容的：它取从第一个 `{` 到最后一个 `}` 之间的全部内容，因此一段游离的 ```json 围栏或前后的散文都不会破坏解析。
4. **校验**，用 `model_validate_json()` 把抽取出的 JSON 校验到你的模型。成功即得到实例。
5. **失败时重试。** 如果校验失败，会把无效输出连同一段简短的纠正提示反馈给模型，并重复这次调用。默认重试一次；如果把重试次数用尽后仍然失败，框架会抛出 `LLMResponseError`。

由于校验用的是 Pydantic 自身的能力，你的模型声明的每一项保证（必填字段、类型、约束、嵌套模型）都会被强制执行。一个是合法 JSON 但结构不对的回复会被当作失败处理并重试，而不是悄悄返回。

### 各厂商如何强制 schema

只要厂商提供了原生的结构化输出通路，schema 就会通过该通路发给模型，此外也会写进系统提示词。对于 OpenAI 兼容协议，这取决于厂商声明的能力：

- `json_schema`：schema 作为类型为 `json_schema` 的 `response_format` 发送，在 API 层面约束模型。
- `json_object`：请求要求返回合法 JSON，而 schema 本身由提示词承载。
- `none`：API 层面什么都不发送，提示词里的指令是唯一的引导。

Anthropic 和 Gemini 协议使用各自的原生结构化通路。无论走哪条通路，结果最终都由 Pydantic 校验，因此仅靠提示词的情形（`json_object` / `none`）和原生约束的情形会汇聚到同一个校验过的实例。厂商能力如何配置见 [LLM 客户端](llm-clients.md)。

## 使用结果

`run()` 返回一个 `RunResult`。在一次完成的运行中，`.final_output` 持有校验通过的模型实例。它就是你 schema 定义的那个对象，可以直接使用：

```python
result = agent.run("Extract the person from: Ada is 36.", output_schema=Person)
person = result.final_output
send_to_database(person.name, person.age)
```

护栏（guardrail）与对话历史在结构化通路上依然生效。当框架需要输出的文本形式时（用来检查输出护栏，或把这一轮持久化进历史），它会用 `model_dump_json()` 序列化模型，所以落进历史的是你对象的 JSON 形式。输出护栏见 [护栏与人在回路](guardrails-and-hitl.md)。

## 调整重试

传给 `run()` 的额外关键字参数会沿着结构化通路向下转发，因此当模型需要更多空间来自我纠正时，你可以提高重试预算：

```python
person = agent.run(
    "Extract the person from: Ada is 36.",
    output_schema=Person,
    retries=3,
).final_output
```

`retries` 统计的是首次尝试之后的纠正次数，所以 `retries=3` 总共最多允许四次模型调用。

## 失败时

如果模型在重试预算内无法产出一个有效对象，框架会抛出带有最后一次校验错误的 `LLMResponseError`。像处理任何其他模型调用失败那样处理它即可：

```python
from agentmaker import LLMResponseError

try:
    person = agent.run("...", output_schema=Person).final_output
except LLMResponseError as exc:
    ...  # log, fall back, or surface the error
```

抛出错误意味着不会返回任何猜测的或部分填充的对象。你要么得到一个完全校验通过的实例，要么得到一个明确的失败，绝不会拿到半成品结果。
