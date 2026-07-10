# 可观测性

每次 agent 运行都可以产出一份结构化的 trace（运行轨迹）：每次 LLM 调用、工具调用和上下文操作各记一条，附带耗时与 token 用量。当你想调试某次运行、审计成本，或把事件推送到 SQLite、OpenTelemetry 这类后端时，就挂上一个 `Tracer`。默认什么都不挂，因此没有 tracer 的 agent 零额外开销。当你确实想观测时，就注入一个 `Tracer`，而它的事件落到哪里由可插拔的导出器（exporter）决定。本页后面的 [Trace 侦探](#trace-侦探-devtools) 会把一份录好的 trace 变成由 LLM 撰写的问题诊断，说明哪里出了错。

## 挂上 tracer

构造一个 `Tracer` 并传给 agent。tracer 会收集 agent 在运行期间产出的事件；运行结束后你再从导出器里读回它们。下面这个示例是自洽的（无需 API key、无需联网），摘自 [`examples/13_observability.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/13_observability.py)：

```python
from agentmaker import Agent, MemoryExporter, Tracer, tool
from agentmaker.testing import ScriptedLLM


@tool
def double(x: int) -> int:
    """Double a number.

    Args:
        x: The number to double.
    """
    return x * 2


exporter = MemoryExporter()
tracer = Tracer(exporters=[exporter])

agent = Agent("assistant", ScriptedLLM([
    ScriptedLLM.tool_call("double", {"x": 21}),
    "The answer is 42.",
]), tools=[double], tracer=tracer)
agent.run("double 21")

print("captured trace events:")
for event in exporter.events:
    print("  -", event.get("type"))
```

这次运行会捕获一次 `llm_call`（模型决定调用工具）、一次 `tool_call`（工具执行），以及最后一次 `llm_call`（模型写出答案）。

`Agent(..., tracer=None)` 是默认值，所以在生产路径中若不想开启 tracing，把这个参数省略即可。

## 会记录哪些内容

框架为每次操作产出一个事件字典。每个事件都带有一个 `type` 以及该类型特有的字段，并且每个事件都会被盖上关联字段，让你能按运行分组、按步骤排序。多数事件来自 agent 的 harness（执行调度层）；记忆和 RAG 子系统也以同样的方式各自产出自己的 `memory_search` 和 `rag_retrieve` 事件：

| `type` | 关键字段 |
| --- | --- |
| `llm_call` | `model`、`latency_ms`、`usage` |
| `tool_call` | `tool`、`params`、`status`、`latency_ms`、`result` |
| `context_block` | `query`、`block_chars` |
| `memory_search` | `query`、`hits` |
| `rag_retrieve` | `query`、`hits`、`latency_ms` |

每个事件还带有 `run_id` 和 `step_index`（由框架的关联步骤添加），因此同一次运行产出的事件共享同一个 id，并按顺序递增。

!!! note "密钥永不落到任何 sink"
    在事件扇出之前，tracer 会对其做脱敏：键名看起来像密钥的值（`api_key`、`token`、`password` 及类似者）会被掩码成 `***`，看起来像密钥的字符串（`sk-` 开头的 key、`Bearer` token、一长串连续 token）会就地掩码，路径里的家目录用户名（`/Users/<name>/`）也会被掩码。即便关闭脱敏，过长的字符串值也始终会被截断到 `max_value_len`（默认 200 个字符）。框架不认识任何业务概念，所以应用自有的敏感字段要你自己声明：在 `Tracer` 构造函数上用 `extra_secret_keys=[...]`（键名子串）或 `extra_secret_patterns=[...]`（值的正则表达式）。`run_id` 和 `step_index` 字段是豁免的，因此掩码永远不会破坏关联。

## 导出器

导出器决定事件去往何处。四个导出器都继承自 `TraceExporter`（接口为 `export(event)`，外加一个释放资源的 `close()`），且单个 `Tracer` 可以同时驱动多个导出器。脱敏在扇出之前只做一次，所以每个导出器收到的都是已经清洗过的事件。

| 导出器 | 签名 | 事件去向 |
| --- | --- | --- |
| `MemoryExporter` | `MemoryExporter(max_events=2048)` | 一个内存列表（环形缓冲区，超过上限后丢弃最旧的）。默认 sink；重启即丢失。 |
| `JsonlExporter` | `JsonlExporter(path)` | 每个事件追加一行 JSON（JSON Lines 格式），立即 flush。 |
| `SqliteExporter` | `SqliteExporter(db_path=":memory:")` | 每个事件在 `traces` 表里占一行（`type`、`run_id`、`event`、`created_at`），并在 `run_id` 上建索引。 |
| `OTelExporter` | `OTelExporter(tracer_name="agentmaker", *, carrier_provider=None)` | 每个事件生成一个 OpenTelemetry（OTel，厂商中立的分布式追踪标准）span，供 Jaeger / Grafana / Datadog 使用。 |

如果你不传 `exporters`，tracer 默认使用 `[MemoryExporter()]`。若想在持久化的同时仍能在进程内读取事件，就在持久化导出器旁边一并放上一个 `MemoryExporter()`：

```python
from agentmaker import JsonlExporter, MemoryExporter, Tracer

tracer = Tracer(exporters=[MemoryExporter(), JsonlExporter("run.jsonl")])
```

在进程退出前调用 `tracer.close()`，以 flush 并释放文件 / 数据库句柄。

### OpenTelemetry

`OTelExporter` 把每个事件映射为一个 span。它用事件的 `latency_ms` 让 span 在瀑布图里拥有真实的宽度（而不是一个零宽度的点），并且总是把 `run_id` 挂为 span 属性，好让后端能按运行过滤。它会惰性导入 `opentelemetry`，所以要安装 `otel` extra：

```bash
pip install "agentmaker[otel]"
```

若想让 agent 的 span 并入上游的请求 trace，传入 `carrier_provider=current_trace_carrier`。carrier 如何提供，见下文 [运行级上下文](#运行级上下文)。

## 读回 trace

`Tracer` 在其内存事件（导出器列表中的第一个 `MemoryExporter`）之上提供了一组便捷读取方法：

- `tracer.events` 返回收集到的事件列表。
- `tracer.summary()` 返回一个字典，含 `events`（事件总数）、`by_type`（每种类型的计数）、`total_tokens`、`total_latency_ms`，外加 `dropped`（因导出器失败而丢失的事件数，按导出器计）和 `dropped_uncleanable`（因清洗本身抛异常而被丢弃的事件数）。
- `str(tracer)` 渲染出一份可读的、每个事件一行的时间线。
- `tracer.clear()` 清空内存事件（文件 / 数据库 sink 不受影响）。

抛异常的导出器默认会被吞掉，这样一个旁路故障（磁盘满、数据库锁、collector 不可达）永远不会拖垮本次运行。用 `strict=True` 构造 tracer，则导出器与清洗的失败会改为重新抛出，这在测试里很有用。

## 运行级上下文

框架通过 `contextvars` 传播一次运行的身份与治理状态，因此异步任务与线程池之间彼此隔离。这些访问器让应用、工具或 hook 能读取当前运行的上下文。它们全部可从顶层导入：

```python
from agentmaker import (
    current_run_id, current_scope, current_step, current_trace_carrier,
)
```

- `current_run_id()` 返回本次运行的 `run_id`（在运行之外则返回 `None`），让你能把自己的日志与 trace 关联起来。
- `current_step()` 返回本次运行已到达的步骤编号。
- `current_scope()` 返回本次运行的会话作用域（例如，被委派的工具会用它，按父会话来隔离子 agent 的历史）。
- `current_trace_carrier()` 返回本次运行上游的 W3C trace carrier（一个形如 `{"traceparent": ...}` 的字典），若未提供则为 `None`。

carrier 由你在启动运行时提供。`agent.run(...)` 和 `agent.arun(...)` 都接受 `trace_carrier`，因此一个 web handler 可以把入站请求的 `traceparent` 头传进来：

```python
result = agent.run(user_text, trace_carrier={"traceparent": request_header})
```

挂上 `OTelExporter(carrier_provider=current_trace_carrier)` 之后，本次运行的每个 span 就会成为应用跨服务 trace 的子节点，而不是一个新的根节点。

### governed_chat

大多数 LLM 与工具调用都经由 harness，它免费施加运行限额与 tracing。少数框架路径会直接调用模型、绕过 harness。如果你手写一个直接调用 LLM 的 recipe，又希望它遵守同一套运行治理，就把调用经由 `governed_chat`（异步）路由：

```python
from agentmaker import governed_chat

response = await governed_chat(llm, messages, tracer=tracer, origin="my.recipe")
```

它会检查本次运行的限额，await `llm.chat(messages, ...)`，记录该次调用的计数与 token 用量，可选地产出一个带 `origin` 标签的 trace 事件，然后强制执行硬性 token 上限。在运行上下文之外，治理部分（限额检查与用量记账）是零开销的空操作——LLM 调用本身、以及传入 tracer 时的 trace 事件，仍会照常发生。`tracer` 参数是可选的；额外的关键字参数会透传给 `llm.chat`。

## Trace 侦探（devtools） { #trace-侦探-devtools }

Trace 侦探是一个可选的开发者工具，它消费一份录好的 trace 并返回由 LLM 撰写的诊断：最早出错的步骤、根因，以及最小的修复方案。它位于 `agentmaker.devtools` 子包中，框架核心从不导入该子包，所以上文描述的原生 tracing 无论有没有它都能工作。它随 `devtools` extra 一同发布：

```bash
pip install "agentmaker[devtools]"
```

因为它不在顶层命名空间里，所以按需导入：

```python
from agentmaker.devtools import diagnose_trace, DoctorHook
```

### 以库的方式诊断

把一次运行录到一个 JSONL 文件（按上文所示挂一个 `JsonlExporter`），然后把该文件交给 `diagnose_trace`。它会解析整份 trace，选出一次运行（按 `run_id`，或最近的一次），并用任意 LLM 客户端诊断它。它返回解析出的运行与判定结果：

```python
from agentmaker import LLMClient
from agentmaker.devtools import diagnose_trace

run, verdict = diagnose_trace(open("run.jsonl").read(), LLMClient("deepseek"))
```

判定结果是一个 `TraceDiagnosis`，含这些字段：`healthy`（bool）、`first_bad_step`（最早出错的步骤编号，或 `None`）、`what_went_wrong`、`root_cause`、`suggested_fix`，以及 `confidence`（`"low"` / `"medium"` / `"high"`）。诊断经由一个普通的 agentmaker agent 以结构化输出运行，因此框架支持的任意 LLM 客户端在这里都能原样使用。

### 在 web UI 里诊断

启动本地 web 服务器：

```bash
python -m agentmaker.devtools
```

它默认绑定 `127.0.0.1:8765`（一个本地调试工具，不应对外暴露）。粘贴或加载一份 trace，即可看到确定性的时间线及各项发现，然后请求一次 LLM 诊断。服务器会用环境里的 API key 构建其诊断客户端；若没有可用的 key，它仍会以「仅解析」模式启动，让时间线保持可用。常用参数：`--host`、`--port`、`--provider`（默认 `deepseek`）、`--model`，以及 `--no-llm`（仅解析、跳过 LLM）。

### DoctorHook：就地诊断

若想在开发时走零摩擦路径，挂上一个 `DoctorHook`，每次有问题的运行都会在终端里自我诊断，无需导出文件、也无需打开 web UI。把同一个 `Tracer` 同时传给 agent 和这个 hook（hook 会从 tracer 的 `MemoryExporter` 里读回本次运行的事件）：

```python
tracer = Tracer()
agent = Agent("bot", llm, tools=[...], tracer=tracer, hooks=[DoctorHook(tracer)])
agent.run("...")   # a failed tool / truncation / exception now prints a three-part diagnosis
```

抛异常的运行总会触发诊断；正常结束的运行，仅当其 trace 携带的发现达到或超过 hook 的 `severity` 阈值（默认 `"error"`，覆盖工具失败与截断；`"warn"` 会放宽到包含空检索及其它降级情形）时才触发。诊断用的 LLM 从环境 key 惰性构建，你也可以用 `llm=` 直接交给它一个现成的客户端（或用 `provider=` / `model=` 选择付费厂商）。hook 内部的每个失败都会被捕获并作为一行控制台信息报告，因此坏掉的诊断永远不会影响运行本身的结果。

!!! note
    `DoctorHook` 是一个生命周期 `Hook`，也就是 [护栏与人在回路](guardrails-and-hitl.md) 里介绍的那个扩展点。它在一个全新的上下文下、于一个工作线程里运行诊断，因此永远不会消耗宿主运行的限额：即便一次因运行限额错误而死掉的运行，仍然可以被诊断。
