# Agent 与工作流

一个 Agent（智能体：接收输入、按需调用工具、返回回复的执行单元）接收一份输入，按需在循环中调用工具（tool），最后返回一段回复。agentmaker 为此提供了一个统一的执行原语（`Agent`）、两种工作流范式（`PlanAgent`、`ReflectionAgent`），以及一种声明式描述它们中任意一个的方式（`AgentSpec` + `build_agent`）。每一种 Agent 策略都返回同一种结果封装 `RunResult`。另有一个独立的适配器 `AgentTool`，把一个 Agent 当作工具交给另一个 Agent 使用；正因为它本身是一个工具，它返回给编排方（orchestrator，即发起委派的上层 Agent）的是 `ToolResponse`，而不是 `RunResult`。

## 统一循环

`Agent` 是「模型在循环中调用工具」这一模式的核心原语（primitive：最基础、不可再拆的构件）。每一轮，消息连同工具的 schema（结构描述：告诉模型每个工具叫什么、接收哪些参数）一起发给模型，模型要么用文本作答（循环结束），要么请求调用工具。框架执行这些工具、把结果回喂给模型、再次调用模型，如此往复，直到模型给出答案或轮次预算耗尽。若未注册任何工具，第一轮即为终止轮，这就是纯粹的问答。

这一个循环同时覆盖了「chat」和「react」两种用法。ReAct（reason then act，即「先推理再行动」：模型在每次调用工具前先写出自己的推理）不过是同一个循环的一个预设，后文在[声明式构建](#声明式构建)一节展开。

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

`ScriptedLLM` 是 `agentmaker.testing` 提供的测试替身（test double：测试时用来顶替真实依赖的假对象）：它按预先给定的固定回复列表逐条回放，因此运行 Agent 代码时无需 API key、也无需联网。把它换成 `LLMClient("deepseek")`（或 `"openai"` / `"anthropic"` / `"gemini"`）即可接入真实模型，届时由模型自行决定何时调用工具。参见 [LLM 客户端](llm-clients.md)。

### 构造一个 Agent

除 `name` 和 `llm` 之外，最常用的两个参数：

| 参数 | 作用 |
| --- | --- |
| `system_prompt` | 人设。可选；省略时不发送任何 system 消息。 |
| `tools` | 便捷入口：一个 `list[Tool]`（含被 `@tool` 装饰的函数）或一个 `ToolRegistry`。传入它即启用工具循环。 |
| `tool_registry` | `tools` 的进阶形式（复用或定制一个注册表）。与 `tools` 互斥；两者同时传入会抛出 `ValueError`。 |
| `max_turns` | 循环中模型的最大轮次（默认 10，必须为正整数）。用于给失控的工具调用设上限。 |

`Agent` 会自动保存多轮历史，并按 `scope`（作用域）相互隔离，因此单个实例可以服务于多个会话。挂上一个 `session_store` 即可让历史在重启后依然保留。关于 scope 如何为会话建立键值区分，参见[作用域与异步](scope-and-async.md)；关于如何构建工具与注册表，参见[工具](tools.md)。

还有更多可选参数（`confirm`、`permissions`、`checkpoint_store`、`context_builder`、`tool_retriever`、护栏（guardrail）、钩子（hook）等等）用于接入各类横切能力，它们各有专门的指南：[护栏与人在回路](guardrails-and-hitl.md)、[上下文工程](context-engineering.md)、[记忆](memory.md)、[可观测性](observability.md)。

### 结构化输出

向 `run` / `arun` 传入 `output_schema`（一个 Pydantic 模型）后，返回的 `RunResult.final_output` 保存经过校验的实例，而非纯文本。这一过程不使用工具。完整用法参见[结构化输出](structured-output.md)。

### 流式输出

`Agent.stream_run(input_text, ...)` 会逐片产出回复（其异步对应版本是 `astream_run`，用 `async for` 迭代）。当 Agent 配有工具且模型适配器原生支持流式工具调用时，流式循环会走完全相同的轮次结构，并在每一轮的文本到达时随之流式输出。文本工具模拟不支持流式工具循环：使用 `LLMClient(..., emulate_tools=True)` 的 Agent 在这种组合下会抛出 `LLMConfigError`，请改用 `run` / `arun`。

默认情况下，文本片段会立即产出，输出护栏在流结束后检查。设 `buffer_output=True` 可先缓冲完整输出，只有通过护栏后才向调用方产出。

!!! note
    流式循环不支持 human-in-the-loop（HITL，人在回路：让人在关键处介入审批）的挂起/恢复，也不支持检查点（checkpoint）。需要确认的工具会退回到它的同步确认回调。当你需要挂起语义时，请改用 `run` / `arun`（参见[返回类型](#返回类型)与[护栏与人在回路](guardrails-and-hitl.md)）。

## 工作流范式

`Agent` 让模型自行决定下一步做什么，而工作流范式则在代码里把各阶段的先后顺序固定下来。两者底层都构建在同一个单循环 `Agent` 之上，因此接收相同的 LLM 和相同的工具。

下面的示例是自洽的（hermetic：不依赖网络或外部服务、可独立运行，`ScriptedLLM` 顶替了真实模型本会生成的内容），开箱即可运行：

```python
from agentmaker import PlanAgent, ReflectionAgent
from agentmaker.testing import ScriptedLLM

# Reflection: draft -> critique -> refine, looping until the critic replies "GOOD ENOUGH"
# (the default English pass signal; the Chinese pack uses a Chinese one).
reflection = ReflectionAgent("writer", ScriptedLLM([
    "The Earth orbits the Sun.",                              # draft
    "Add that one orbit takes about 365 days.",              # critique
    "The Earth orbits the Sun once every ~365 days.",        # refine
    "GOOD ENOUGH",                                            # critique -> pass, stop
]), max_turns=3)
print("Reflection:", reflection.run("Explain Earth's orbit in one sentence.").final_output)

# Plan-and-Solve: break the task into an ordered plan, execute each step, then synthesize.
plan = PlanAgent("solver", ScriptedLLM([
    '{"steps": ["Name the capital of Denmark", "State its approximate population"]}',  # plan (structured)
    "The capital of Denmark is Copenhagen.",                  # step 1 execution
    "Copenhagen has roughly 660,000 residents.",             # step 2 execution
    "Copenhagen is Denmark's capital, home to about 660,000 people.",  # synthesis
]))
print("Plan:", plan.run("Tell me about Denmark's capital.").final_output)
```

### PlanAgent

Plan-and-Solve（先规划再求解）会先把整个计划想清楚，再逐步执行，适合那些跨度长、多步骤、且需要很强目标一致性的任务。它按固定顺序运行三个阶段：模型先把问题拆解为一份有序的子任务步骤列表（通过结构化输出），随后逐个执行每个步骤（委派给内部的单循环 `Agent`），最后把各步骤的结果综合成最终答案。

```python
PlanAgent(name, llm, system_prompt=None, *, tool_registry=None, max_turns=3, ...)
```

传入 `tool_registry`，每个执行步骤都能调用工具；不传，则每一步都是纯推理。注意这里的 `max_turns`（默认 3）是每个子步骤执行器的工具循环上限，而非计划步骤的数量。

### ReflectionAgent

Reflection（反思）会反复打磨一个答案：模型先写出草稿，然后循环执行 `reflect -> refine`（先自我批评，再依据批评修订），直到批评环节表示无需再改，或达到轮数上限。

```python
ReflectionAgent(name, llm, system_prompt=None, *, max_turns=3, tool_registry=None, ...)
```

这里的 `max_turns`（默认 3）是 reflect-refine（反思-修订）的最大轮数。默认的通过信号是 `GOOD ENOUGH`（当答案已足够好、可以停下时，批评环节会写出它）。若传入 `tool_registry`，批评环节便能调用工具来核实事实或算术；不传，批评就纯粹是自我判断。

!!! note
    与 `Agent` 不同，`PlanAgent` 和 `ReflectionAgent` 只接受 `tool_registry`（仅限关键字参数），不提供 `tools` 列表这一便捷形式。请先构建一个注册表（参见[工具](tools.md)）再传入。

当某个步骤或批评环节调用了高风险工具、且挂有 `checkpoint_store` 时，该内部运行会挂起以等待审批，中断会沿着范式向上传播；恢复时从该处继续。参见[护栏与人在回路](guardrails-and-hitl.md)。

## 声明式构建

除了直接调用构造函数，你还可以用 `AgentSpec`（一个纯配置 dataclass）来描述一个 Agent，再用 `build_agent` 把它构建出来。两种形式并存；声明式只是叠在命令式构造函数之上的一层便捷封装。

```python
from agentmaker import AgentSpec, tool


@tool
def get_time() -> str:
    """Return the current time."""
    return "12:00"


# strategy is one of: "chat" / "react" / "plan" / "reflection".
spec = AgentSpec(name="helper", strategy="react", model="deepseek", tools=[get_time])
print(f"spec: name={spec.name!r} strategy={spec.strategy!r} "
      f"model={spec.model!r} tools={[t.name for t in spec.tools]}")

# To build and run it (needs the provider's API key in your environment):
#     from agentmaker import build_agent
#     agent = build_agent(spec)              # resolves model="deepseek" to a real LLMClient
#     print(agent.run("what time is it?").final_output)
print("build with: agent = build_agent(spec)  # needs DEEPSEEK_API_KEY to run")
```

`strategy` 选定使用哪种范式，`build_agent` 则返回对应的实例：

- `"chat"` 构建单循环 `Agent`（默认项，`max_turns` 默认 10）。
- `"react"` 是同一个 `Agent` 的 ReAct 预设：它要求至少配备一个工具（不带任何工具构建会抛出 `ValueError`），`max_turns` 默认为 5，并额外加入一段「先思考再行动」的 system prompt。
- `"reflection"` 构建一个 `ReflectionAgent`（`max_turns` 默认 3）。
- `"plan"` 构建一个 `PlanAgent`（`max_turns` 默认 3）。

关键字段：

- `model`：一个 `"provider:model"` 字符串（例如 `"deepseek:deepseek-v4-flash"`）、一个裸的 provider 名称（如 `"deepseek"`，使用该 provider 的默认模型）、一个 `LLMClient` 实例、一个暴露 chat/stream 的鸭子类型客户端（如 `ScriptedLLM`，因而声明式构造出的 Agent 也能自洽测试），或 `None`（使用默认客户端）。
- `instructions`：会成为该 Agent 的 `system_prompt`。
- `tools`：一个 `list[Tool]` 或一个 `ToolRegistry`。
- `max_turns`：一个统一的轮次上限，会被映射到各策略各自的上限；`None` 则采用该策略的默认值。

`AgentSpec` 的字段是各策略字段的超集。如果某个策略不支持的字段被设成了非默认值，`build_agent` 会抛出 `ValueError`，而不是悄悄忽略它（例如 `compactor` 会被 `plan` 和 `reflection` 策略拒绝）。

## 多 Agent 编排

orchestrator-worker（协调者-工作者）模式让一个主 Agent 把子任务委派给各领域的专家 Agent，同时始终掌控整段对话。`AgentTool` 通过把一个 Agent 适配成 `Tool` 来实现这一模式，于是主 Agent 委派子任务的方式与调用任何其他工具别无二致。子 Agent 携带自己独立的历史和工具，因此其上下文保持隔离。

```python
from agentmaker import Agent, AgentTool
from agentmaker.testing import ScriptedLLM

# The worker: a specialist sub-agent.
translator = Agent("translator", ScriptedLLM(["Bonjour le monde"]))

# The coordinator calls the worker through AgentTool, then composes the final answer.
coordinator = Agent("coordinator", ScriptedLLM([
    ScriptedLLM.tool_call("translate", {"task": "translate 'hello world' to French"}),
    "In French, 'hello world' is: Bonjour le monde.",
]), tools=[AgentTool(translator, name="translate", description="Translate text to French")])

print(coordinator.run("How do you say 'hello world' in French?").final_output)
```

`AgentTool(agent, *, name=None, description=None, scope=None, prompts=None)` 可包装任意 Agent。`name` 默认取 `agent.name`；`description` 告诉负责协调的模型这个子 Agent 擅长什么、何时该把任务委派给它。该工具只暴露一个 `task` 字符串参数，即交给子 Agent 的那份自足完整的子任务。

默认 `scope=None` 时，委派会继承父 run 的当前 scope，因此一个 `AgentTool` 实例可以服务多个父会话，而不会混淆子 Agent 的历史。只有需要把子 Agent 固定到某个归属时才显式传 `scope`。并行分支仍应使用不同的子 Agent 实例，因为一个 Agent 对象只有一套可变的执行与历史表面。

!!! note
    通过 `AgentTool` 调用的子 Agent，无法在委派中途挂起以等待人工审批。如果它触及某个高风险动作，本次委派会返回一个错误结果，告知协调者：该子任务需要人工审批、无法以这种方式完成，从而让协调者改走别的路径。请把高风险动作留在主流程里，或改用 `PlanAgent`，它确实会向上传播嵌套的挂起。参见[护栏与人在回路](guardrails-and-hitl.md)。

## 返回类型

每个 Agent 上的 `run`、`arun`、`resume`、`aresume` 都返回一个 `RunResult`。一次运行有两种终止状态，由 `RunStatus = Literal["completed", "interrupted"]` 刻画：要么产出了最终输出，要么有一个高风险动作被挂起、等待人工审批（human-in-the-loop，HITL，人在回路）。这个「中断」状态是显式的，而不是一个你可能误当成答案的裸值。

```python
r = agent.run("...")
if r.interrupted:
    handle(r.interrupt)            # HITL: take the suspended state to resume
else:
    use(r.final_output)            # completed: take the final output (str or structured instance)
```

`RunResult` 是一个冻结（frozen）dataclass，包含以下字段：

- `final_output`：完成时的输出（一个 `str`，或在传入 `output_schema` 时为一个结构化实例）；挂起时为 `None`。
- `status`：`"completed"` 或 `"interrupted"`。
- `interrupt`：挂起状态对应的 `Interrupt`（待处理的动作，加上恢复所需的作用域）；完成时为 `None`。
- `usage`：本次运行的 `RunUsage` 快照。
- `new_messages`：本轮新增到历史中的消息（用户 + 助手）；挂起时为空。
- `run_id`：本次运行的 trace 关联 id（用于把同一次运行的各条追踪记录关联起来）。

`RunResult.interrupted` 是一个便捷属性（等价于 `status == "interrupted"`）。调用 `str(result)` 会直接得到最终输出文本（挂起的结果会显示一段可读的提示，而非一个裸的 `None`）。

`RunUsage` 是一份冻结快照，用于成本核算与额度可观测，含三个字段：`llm_calls`、`tool_calls`、`total_tokens`（均为整次运行的累计值）。

关于 `Interrupt` 对象，以及用于继续一次挂起运行的 `resume(decision)` 流程，参见[护栏与人在回路](guardrails-and-hitl.md)。
