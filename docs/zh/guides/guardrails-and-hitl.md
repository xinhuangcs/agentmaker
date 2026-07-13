# 护栏与人在回路

本指南介绍 Agent 运行外围的安全与控制层：一是**护栏**（guardrails），负责审查输入和输出，一旦违反规则就中断运行；二是 HITL（即 human-in-the-loop，指某个动作执行前先由人来批准），在遇到高风险工具调用时挂起运行，等待人做决定。同时也介绍这两者所依赖的配套机制：用于观测运行的生命周期**钩子**（hooks）、用于保存对话历史的**会话**（session）持久化、用于挂起/恢复与崩溃恢复的**检查点**（checkpoints），以及为单次运行设定全局上限的**运行策略**（run policies）。当你需要拦截某些输入、在危险动作前要求批准、保留审计记录，或限制单次运行能做什么时，就查阅本页。

本页所有内容都基于 [`examples/07_guardrails_and_hitl.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/07_guardrails_and_hitl.py)，它是自洽（hermetic）的，借助 `ScriptedLLM`（LLM 测试替身）实现无需 API key、无需联网即可运行：

```bash
uv run python examples/07_guardrails_and_hitl.py
```

## 护栏

护栏检查一段文本（Agent 的输入或它的最终输出）并返回一个裁决结果。裁决失败即为一次*触发线*（tripwire）：运行停止，并抛出 `GuardrailTripwireError`。你通过 `input_guardrails=[...]`（在模型运行前对用户输入进行检查）和 `output_guardrails=[...]`（在最终输出返回前对其进行检查）把护栏挂到 `Agent` 上。

最快的做法是使用 `CallableGuardrail`，它把任意函数 `fn(text)` 包装成一个护栏。该函数返回一个布尔值（True 让文本通过，False 即拉响触发线），而你在构造时传入的 `message=` 就成为拦截说明：

```python
from agentmaker import Agent, CallableGuardrail, GuardrailTripwireError, tool
from agentmaker.testing import MemoryCheckpointStore, ScriptedLLM

# 1) Guardrail: reject any input that mentions a password.
no_secrets = CallableGuardrail(lambda text: "password" not in text.lower(),
                               message="input mentions a password")
guarded = Agent("assistant", ScriptedLLM(["ok"]), input_guardrails=[no_secrets])
try:
    guarded.run("my password is 1234")
except GuardrailTripwireError as e:
    print("Guardrail blocked the input:", e)
```

`GuardrailTripwireError` 的 `str` 就是展示给用户的可读拦截说明，因此上面的 `print(..., e)` 会打印出你配置的 `message`。

### 护栏接口

`CallableGuardrail`、`Guardrail` 和 `GuardrailResult` 都属于公开 API。

`GuardrailResult` 是护栏返回的裁决结果，它有两个字段：

- `passed: bool`（True 让文本通过，False 表示触发线）。
- `message: str`（拦截原因的可读说明；默认为 `""`，当 `passed=True` 时可以为空）。

`CallableGuardrail` 包装一个既可以返回布尔值、也可以返回 `GuardrailResult` 的函数。返回布尔值时，`False` 会触发护栏，并使用你在构造时给定的 message；改为返回一个 `GuardrailResult`，则可让该函数携带自己的 message。例如：

```python
from agentmaker import CallableGuardrail

# Bool form: False trips, using the message given at construction.
length_limit = CallableGuardrail(lambda t: len(t) < 4000, message="input too long")
```

对于超出一行代码的逻辑，请继承 `Guardrail` 并实现 `check`：

```python
from agentmaker import Guardrail, GuardrailResult


class BlocklistGuardrail(Guardrail):
    def __init__(self, words):
        self._words = [w.lower() for w in words]

    def check(self, text: str) -> GuardrailResult:
        hit = next((w for w in self._words if w in text.lower()), None)
        if hit is not None:
            return GuardrailResult(passed=False, message=f"input contains a blocked word: {hit}")
        return GuardrailResult(passed=True)
```

`Guardrail` 是一个抽象基类，只有一个抽象方法 `check(self, text) -> GuardrailResult`。此外还有一个异步版本 `acheck(self, text) -> GuardrailResult`，框架的执行层实际调用的正是它；默认情况下它内联为对 `check` 的直接调用。大多数护栏都是纯计算（长度、正则、屏蔽词检查），所以默认实现即可。只有当护栏涉及阻塞式 I/O，或想调用 LLM 来对文本做内容审核时，才需要重写 `acheck`。`CallableGuardrail` 也接受异步函数（一个 `async def`，或包装了异步调用的 lambda），并通过 `acheck` 对其进行 await。

!!! note
    agentmaker 提供的是护栏*接口*以及 `CallableGuardrail`；具体规则是你自己的业务逻辑。框架不内置任何可配置的内容策略。请编写你的应用所需的检查，并把它们挂为 `input_guardrails` / `output_guardrails`。

## 人在回路（HITL）

当一个工具被标记为高风险动作时，Agent 运行不会悄无声息地执行它，而是在该调用处*挂起*运行，把待执行动作的描述交给你，并一直等到你批准或拒绝。启用这一机制需要两样东西：

1. 用 `@tool(requires_confirmation=True)` 标记该工具。
2. 给 Agent 配一个**检查点存储**（checkpoint store），以便保存并恢复挂起状态。真实部署用 `SqliteCheckpointStore`；测试中用 `MemoryCheckpointStore`。

```python
# 2) HITL: a high-risk tool suspends the run until a human approves.
@tool(requires_confirmation=True)
def delete_file(path: str) -> str:
    """Delete a file (high-risk, requires confirmation).

    Args:
        path: File path to delete.
    """
    return f"deleted {path}"


ops = Agent("ops", ScriptedLLM([
    ScriptedLLM.tool_call("delete_file", {"path": "/tmp/old.log"}),
    "Done, the file was deleted.",
]), tools=[delete_file], checkpoint_store=MemoryCheckpointStore())

result = ops.run("please delete /tmp/old.log")
if result.interrupt:                                    # run paused, awaiting approval
    pending = result.interrupt.pendings[0]
    print(f"Approval needed for: {pending.tool_name}({pending.arguments})")
    approved = ops.resume(True, scope=result.interrupt.scope)   # resume(False) would reject
    print("After approval:", approved.final_output)
```

### 读取 interrupt

`run`（以及 `resume`）总是返回一个 `RunResult`。当高风险工具挂起了运行时，`RunResult.status` 为 `"interrupted"`，便捷属性 `RunResult.interrupted` 为 `True`，而 `RunResult.interrupt` 持有一个 `Interrupt`。当运行改为正常结束时，`status` 为 `"completed"`，`interrupt` 为 `None`，`final_output` 持有答案。完整的 `RunResult` 结构见 [Agent 与工作流](agents.md)。

一个 `Interrupt` 描述了正在等待什么：

- `pendings`：等待批准的 `PendingAction` 列表。一次挂起可以包含多个动作（同一轮请求了多个高风险工具，或多个并行子 Agent 各自挂起）。
- `pending`：一个便捷属性，返回第一个待处理动作，若没有则返回 `None`。
- `scope`：恢复凭据。把它传回给 `resume`（跨会话重新加载挂起状态时这是必需的）。

每个 `PendingAction` 都携带 `tool_name`、`arguments`（调用参数）和 `call_id`（本次工具调用的唯一 id；`resume` 用它来匹配决定）。

### 恢复运行

调用 `resume(decision, *, scope=...)` 来继续一个挂起的运行。`decision` 可以是：

- 一个布尔值：`resume(True, scope=...)` 批准并执行被挂起的动作；`resume(False, scope=...)` 拒绝它（拒绝会被反馈给模型以便它改换路线，而不会被当作错误）。单个布尔值会对本轮所有待处理动作做出一致的决定。
- 一个字典 `{call_id: bool}`：按动作逐个决定，以每个 `PendingAction.call_id` 为键。当一次挂起持有多个待处理动作时用这种形式。
- 省略（`None`）：这是一次崩溃恢复式的恢复。不注入任何决定；运行只是从最后一个检查点继续，而仍待处理的高风险动作会重新挂起并返回一个全新的 `Interrupt`，不会被误判为拒绝。

`resume` 与 `run` 一样返回一个 `RunResult`：`completed` 并带有 `final_output`，或者在另一个高风险动作正在等待时再次 `interrupted`。请传入 `scope=result.interrupt.scope`，让 `resume` 加载正确的挂起状态；如果省略，它会默认使用 Agent 自身的 `scope`。

!!! note
    `ApprovalRequired` 是一个公开名字，但它是内部控制流信号，不是你需要捕获的错误。当 harness 遇到一个本轮尚无决定的高风险工具时会抛出它；运行循环捕获它，打包状态，并把它转换成你收到的那个 `Interrupt`。你只与 `Interrupt` 和 `resume` 打交道，永远不直接接触 `ApprovalRequired`。

### 按调用逐次确认

装饰器上的 `requires_confirmation=True` 把整个工具标记为高风险。真正读取的门控是 `Tool.needs_confirmation(parameters)`，它默认返回 `requires_confirmation`。继承了 `Tool` 的工具可以重写 `needs_confirmation`，从而按每次调用来决定（例如只在删除安全目录之外的文件时才确认）。

### 用 `cli_confirm` 做同步确认

对于服务器，HITL 的挂起/恢复才是正确模型：请求返回一个中断态 `RunResult`，其中 `interrupt` 描述待处理动作；人在带外做出决定，之后的一次请求再恢复运行。而在命令行或教学场景中，你可能想要一个内联的、阻塞式的 y/n 提示。`cli_confirm` 就是这样一个开箱即用的组件：把它作为 `confirm=cli_confirm` 传入，高风险工具就会打印出它的动作并在标准输入上发问。

```python
from agentmaker import Agent, cli_confirm

agent = Agent("ops", llm, tools=[delete_file], confirm=cli_confirm)
```

`cli_confirm(tool, parameters) -> bool` 打印工具名和参数，并返回用户是否键入了 `y`。它不是默认行为：如果你既不传 `confirm` 也不传 `checkpoint_store`，安全的选择是拒绝（这样无人值守的服务器就不会卡在 `input()` 上）。交互式 CLI 用 `cli_confirm`，服务器用 `checkpoint_store` + `resume`。

## 钩子

`Hook` 是一个只观测的生命周期回调。继承 `Hook`，只重写你关心的事件（其余都是空操作），并用 `hooks=[...]` 挂上一个列表。钩子用于日志、指标、审计、成本跟踪等副作用。它们无法拦截或修改运行；拦截是护栏、权限和 HITL 的职责。

```python
from agentmaker import Agent, Hook


class AuditHook(Hook):
    def before_tool(self, name: str, parameters: dict):
        print(f"about to run tool {name} with {parameters}")

    def on_guardrail_trip(self, stage: str, message: str):
        print(f"guardrail tripped at {stage}: {message}")


agent = Agent("assistant", llm, tools=[delete_file], hooks=[AuditHook()])
```

全部事件（默认都是空操作）：

| 方法 | 触发时机 |
| --- | --- |
| `on_run_start(input_text, *, scope=None)` | 一次运行开始时，在输入护栏之前。 |
| `before_model(messages)` | 每次 LLM 调用之前（流式也会触发）。 |
| `after_model(response)` | 每次非流式 LLM 调用之后，以及传入工具的流式调用产出终态 `LLMResponse` 之后。纯文本流没有单一响应对象，因此不触发此事件。 |
| `before_tool(name, parameters)` | 工具执行前的最后一刻（此时已通过权限门和批准门）。 |
| `after_tool(name, parameters, result)` | 工具执行后；`result` 是一个 `ToolResponse`。 |
| `on_guardrail_trip(stage, message)` | 护栏触发线时；`stage` 为 `"input"` 或 `"output"`。 |
| `on_interrupt(pendings, *, scope=None)` | HITL 挂起时；`pendings` 是 `PendingAction` 列表。 |
| `on_error(error)` | 非护栏异常向外传播前的最后一刻。 |
| `on_run_end(output, *, scope=None)` | 一次运行正常产出最终结果时。 |

所有返回值都会被忽略（钩子是纯副作用），钩子内部抛出的异常会向上传播（fail loud，即出错就大声报错），所以有风险的 I/O 请自行包裹处理。运行级事件（`on_run_start`、`on_interrupt`、`on_guardrail_trip`、`on_error`、`on_run_end`）由 Agent 触发；模型与工具事件由底层 harness 触发。当框架在其异步路径上运行某个事件方法时，该方法可以写成 `async def`。

## 会话

默认情况下，Agent 把对话历史保存在进程内，因此一次重启就会丢失历史。挂上一个 `SessionStore` 即可持久化历史并跨重启存续。`SqliteSessionStore` 是内置后端；生产环境请给它一个文件路径（默认的 `":memory:"` 仅供测试）。历史按 `Scope` 隔离，这与检索和记忆中使用的隔离标签是同一个（见 [检索与 RAG](retrieval-and-rag.md)）。

```python
from agentmaker import Agent, Scope, SqliteSessionStore

store = SqliteSessionStore("daemon.db")
agent = Agent("assistant", llm, session_store=store, scope=Scope(user="alice", session="chat-1"))
```

`SessionStore` 是仅追加（append-only）的：每条消息是一行，只追加、从不改写。其接口为 `append` / `append_many` / `load` / `clear`，每个都接受一个关键字参数 `scope`。`load` 和 `clear` 默认精确匹配所有 scope 维度（空 scope 只读取默认桶，绝不跨入另一个会话）；若要有意做跨会话操作，传入 `all_scopes=True`。`SqliteSessionStore` 还额外提供 `prune(...)` 来截断旧历史（`keep_last=N` 或 `before=time`），以及 `list_scopes(along="session")` 来枚举存在哪些会话（每个返回的 `ScopeSummary` 都带有 `message_count` 和首/末时间戳，便于构建会话列表）。`append` / `append_many` / `load` / `clear` / `list_scopes` 各有 `a*` 异步版本；`prune` 仅同步。

### 检索过往对话

`ConversationSearch` 包裹任意 `SessionStore`，使过往对话可按语义检索（情景式回忆，即「我们之前聊过什么」）。它本身也是一个 `SessionStore`，所以你可以用它替换普通存储直接挂上；在常规方法之上，它新增了 `search(query, *, top_k=5, scope=None)`，返回一个 `RetrievalResult` 列表。它需要一个共享的检索骨干（一个 `HybridRetriever`）来建立索引：

```python
from agentmaker import ConversationSearch, SqliteSessionStore

searchable = ConversationSearch(SqliteSessionStore("daemon.db"), retriever)
agent = Agent("assistant", llm, session_store=searchable, scope=scope)
```

若要让*模型*自己去检索过往轮次，用 `ConversationSearchTool(searchable, scope=scope)` 把它包装成一个工具，并交给 Agent 的 tools。写入会先落到作为事实来源（source-of-truth）的存储中，再以尽力而为（best-effort）的方式喂入索引，因此索引出岔子也绝不会丢失一条消息。

`clear` 在相反方向上采用失败即停的语义：它先严格删除匹配的派生索引项，只有物理删除成功后才清权威会话记录。单个 scope 使用精确 ownership footprint；`all_scopes=True` 使用显式的 conversation 范围。按旧版（无精确删除）接口编写的自定义 `IndexSync` 或检索后端仍可用：精确删除会退化为按范围删除，没有 `strict` 参数的实现则保持其尽力而为的删除契约。使用结束后调用 `close()`；包装器负责关闭传入的会话存储及其索引同步接缝。

## 执行状态、检查点与恢复

在底层，一次运行的轨迹保存在一个 `ExecutionState` 里：消息列表、待处理的 HITL 动作、决定表、剩余的迭代预算，以及各范式各自的恢复元数据。`CheckpointStore` 按 scope 持久化序列化后的 `ExecutionState`，使一次运行可以被暂停和恢复。它支撑三种用途：

- **HITL**：在挂起点保存，然后用 `resume(decision)` 继续。
- **崩溃恢复**：未完成状态每步都会保存，因此进程重启后 `resume()`（不带决定）会从最后一个可恢复检查点继续。
- **长任务恢复**：与崩溃恢复是同一套机制。

与会话存储不同，检查点是唯一的当前可恢复状态：`save` 会覆盖每个 scope 的单个点。带检查点的 Agent 在收尾时先把执行标为完成，再写入对话历史；历史写入成功后才清除检查点。如果进程在这个窗口中停止，或历史持久化失败，完成标记会保留下来。之后调用 `resume()` 会清除它并抛出 `SessionError`，而不会重放工具或重复追加这一轮；应用应核对会话历史，因为最终轮次可能已经存在，也可能尚未写入。这是框架重放层面的 at-most-once（至多一次）取舍，并不为分布式副作用提供 exactly-once（恰好一次）保证。

`CheckpointStore` 是抽象接口（按 scope 的 `save` / `load` / `clear`，外加 `a*` 异步形式）；`SqliteCheckpointStore` 是内置后端，可以与会话和记忆共用一个数据库文件。

```python
from agentmaker import Agent, SqliteCheckpointStore

agent = Agent("ops", llm, tools=[delete_file],
              checkpoint_store=SqliteCheckpointStore("daemon.db"))
```

你通常不需要自己构造 `ExecutionState`；只需传入一个检查点存储，交给 Agent 去管理。`ExecutionState` 之所以公开，是为了让你能够检视它，或实现一个自定义的 `CheckpointStore` 后端。

## 运行策略与上限

`RunPolicy` 为单次运行设定全局上限，并可选地设一个取消钩子。用 `run_policy=...` 挂上它。当超出某个上限时，运行以 `RunLimitExceeded` 中止；当取消钩子返回 `True` 时，运行以 `RunCancelled` 中止。两者都是框架异常（`AgentmakerError` 的子类）。

```python
from agentmaker import Agent, RunPolicy, RunLimitExceeded

policy = RunPolicy(max_llm_calls=8, max_tool_calls=20, deadline_seconds=30)
agent = Agent("assistant", llm, tools=[delete_file], run_policy=policy)
try:
    result = agent.run("do a long multi-step task")
except RunLimitExceeded as e:
    print("run hit a limit:", e)
```

各字段（每个取 `None` 表示不限）：

- `max_llm_calls`：本次运行中 LLM 调用的最大次数（含流式和嵌套的子执行器）；必须 `>= 1`。
- `max_tool_calls`：*实际执行*的工具最大次数（被权限或确认拦下的调用不计入）；必须 `>= 0`。设为 `0` 会在本次运行中禁用工具：LLM 仍可被调用，但模型一旦试图执行工具，运行立即中止（一种硬性的「只读/安全模式」）。
- `max_tokens`：累计 token 上限（各 LLM 响应 `usage.total_tokens` 之和）；必须 `>= 1`。
- `deadline_seconds`：从运行开始计的墙钟时间上限；必须 `> 0`。它在模型/工具调用前、模型调用后以及最终历史提交前的框架边界上协作式执行，不会强行中断正在进行的 SDK 或工具调用；但运行一旦超过期限，就不会作为成功的最终结果提交。若拒绝发生在历史提交这一步，会清掉本回合的检查点让 scope 可复用；若死线在更早处触发，则留下一个可供 `resume()` 继续的检查点。对非缓冲流式，死线是尽力而为：已送达消费者的文本不会回滚，但流式途中越过死线仍会中断本次运行。
- `cancel`：一个快速、非阻塞的回调 `() -> bool`，在每次 LLM 和工具调用前检查；返回 `True` 则中止。当一个 Agent 服务多个会话时，该回调可以调用 `current_run_id()` 来判断它当前看的是哪一次运行。

只要配置了任一数值型调用或 token 上限，本次运行就会禁用工具批并发。串行准入可确保直接工具计数以及工具内部嵌套的 LLM/token 记账都不越过精确上限。

上限在构造时就会校验，因此无意义的取值（负数计数、`max_llm_calls=0`）会立即抛出 `ValueError`，而不是在运行途中才暴露。

!!! note
    上限是针对最外层运行进行全局计数的。嵌套子 Agent 自己的 `RunPolicy` 在父运行内部不生效（它会给出警告）。一次恢复是一次新的运行，所以对它而言上限会重置。要限制子任务，请把上限设在父 Agent 的 `run_policy` 上。

## 相关指南

- [工具](tools.md)，了解 `@tool`、`requires_confirmation` 和工具权限。
- [Agent 与工作流](agents.md)，了解 `RunResult`、`run` 和 `resume`。
- [可观测性](observability.md)，了解如何追踪运行（钩子负责观测；tracer 记录结构化事件）。
- [检索与 RAG](retrieval-and-rag.md)，了解 `Scope` 以及 `ConversationSearch` 所索引进的 `HybridRetriever`。
