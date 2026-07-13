# Scope 与异步

两项贯穿整个框架的能力，决定了你在生产环境中如何运行 agentmaker。**Scope**（作用域）是附加在每一次读写上的标签，用于在共享同一后端的多个用户、多个 Agent、多个应用之间隔离检索、记忆和会话。**异步优先**意味着 agentmaker 从内核起就是异步的：协程（coroutine，可被暂停和恢复的异步函数）才是每项能力的真正实现，每个对外能力都有一个 `a*` 对应形式，流式输出是一个 `async for`，而那些普通的同步方法只是包在异步主体外的一层薄壳。只要有多于一个用户（或多于一个 Agent）共享同一个存储，你就会用到 Scope；只要运行在 Web 服务器或其它事件循环里，你就会用到异步 API。

## Scope

一个 `Scope` 是带有五个可选维度的归属标签。每个存储和索引列都携带这些维度，因此带 Scope 标记的读写只会触及与之匹配的行。`Scope` 直接从顶层导入：

```python
from agentmaker import Scope
```

它是一个 frozen（不可变、可哈希）的 dataclass。每个字段都是可选的，默认值为 `None`：

| 字段 | 含义 |
| --- | --- |
| `base` | 子系统区分（memory / rag 等）；留空表示不作限制。 |
| `user` | 用户标识（多用户隔离的关键，也是最小安全边界）。 |
| `agent` | Agent 标识（在多 Agent 系统中，每个 Agent 各自保存自己的记录）。 |
| `session` | 会话标识（等于 `run_id`，即单次对话的临时上下文）。 |
| `app` | 应用 / 组织标识（共享上下文）。 |

### 隔离记忆与检索

这正是 Scope 的全部意义所在：多个租户可以共享同一个后端、同一个索引，却各自只检索到自己的数据。下面这段代码无需 API key、无需联网即可运行，与 [`examples/10_scope_isolation.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/10_scope_isolation.py) 中随附的完全一致：

```python
from agentmaker import Memory, MemoryStore, Scope
from agentmaker.retrieval import build_sqlite_hybrid
from agentmaker.testing import FakeEmbedder

# One shared store + index; the only thing separating the two users is their Scope.
store = MemoryStore()
index = build_sqlite_hybrid(FakeEmbedder())
alice = Memory(retriever=index, store=store, scope=Scope(base="memory", user="alice"))
bob = Memory(retriever=index, store=store, scope=Scope(base="memory", user="bob"))

alice.add("Alice loves tea")
bob.add("Bob loves coffee")

print("alice sees:", [h.content for h in alice.search("favorite drink", top_k=5)])
print("bob sees:  ", [h.content for h in bob.search("favorite drink", top_k=5)])
```

`alice` 和 `bob` 写入同一个 `MemoryStore`、查询同一个混合索引，但由于每个 `Memory` 在构造时都带了不同的 `Scope`，`alice.search(...)` 永远不会返回 Bob 的笔记，反之亦然。这里并不涉及任何独立的数据库。关于 `Memory` API 本身，参见 [记忆](memory.md) 指南。

### 过滤语义

Scope 只在非空维度上做过滤。一次读操作会为你赋了值的每个维度加上一条约束，而对未设置的维度完全不作限制。因此 `Scope(user="alice")` 会返回 Alice 的全部记录，无论它们由哪个 Agent 或哪个会话产生；而某个维度为空，就意味着「这一维匹配任何值」。包括 `base` 在内的每个维度都遵循同样的规则：空即不限制。

每个维度只能是字符串或 `None`。空字符串会归一化为 `None`；数字和布尔值会抛出 `TypeError`，避免它们被误当成不受限维度。应用使用数字 ID 时应显式转换，例如 `Scope(user=str(user_id))`。

`Memory`、`RagRetriever` 和 `IngestionPipeline` 始终强制各自的规范 `base`（`memory` 或 `rag`）。按次调用的 scope 可以替换其它维度，但缺失的 base 会自动补齐，冲突的 base 会直接报错，因此 `Scope(user="alice")` 不会意外跨越子系统索引。

`base` 用于区分诸如 memory 和 RAG（retrieval-augmented generation，检索增强生成，即把检索到的文档喂进 prompt）这类子系统。按照约定，每个上层都会显式传入它，这也是上面示例使用 `Scope(base="memory", user="alice")` 的原因：`Memory` 在 `base="memory"` 下工作，因此它的数据不会与共用同一文件的 RAG 存储发生碰撞。

!!! note "全作用域防护栏"
    完全为空的 `Scope()` 不作任何限制，因此会匹配整个数据库。对于破坏性或全局操作，检索层会拒绝一个裸的 `Scope()`，除非调用方显式选择放行。如果你要自建检索后端，辅助函数 `scope_is_empty` 和 `require_explicit_scope`（两者都可从 `agentmaker.retrieval` 导入）实现了这项检查：`require_explicit_scope(scope, all_scopes, action)` 会抛出异常，除非对一个不限制任何维度的 scope 传入了 `all_scopes=True`。框架内置的 memory 和 RAG 始终携带非空的 `base`，因此永远不会被拦截。

### Agent 中的 Scope（会话）

同一个标签也用来隔离对话历史。`Agent` 在构造时接受一个默认的 `scope=`，而 `run` / `arun` / `resume` / `stream_run` 各自都接受一个按次调用的 `scope=` 来覆盖它。历史按 scope 加载和保存，因此单个 `Agent` 实例可以服务许多互相独立的会话：

```python
from agentmaker import Agent, Scope
from agentmaker.testing import ScriptedLLM

agent = Agent("assistant", ScriptedLLM(["Hi Alice.", "Hi Bob."]), scope=Scope(user="alice"))

agent.run("hello")                            # recorded under Scope(user="alice")
agent.run("hello", scope=Scope(user="bob"))   # a separate session on the same instance
```

当为 `Agent` 提供了 `session_store` 时，历史会按 scope 持久化，并在每次运行时按 scope 重新加载，因此长时间运行的守护进程在重启后不会丢失对话。用于 HITL（human-in-the-loop，人在回路，即暂停一次运行以等待人工批准）和崩溃恢复的检查点，同样以 scope 为键。那些委托给内部子 Agent 的编排配方，会在派生出的子 scope 下运行每个子 Agent，因此子 Agent 的历史和检查点永远不会与其父级碰撞。挂起与恢复的流程参见 [护栏与人在回路](guardrails-and-hitl.md)。

同一个 Agent 实例会按精确 scope 串行化完整的运行、恢复与流式生命周期。同一 event loop 中的并发调用会在该 scope 上排队；来自另一个 event loop 的竞争调用（通常是两个线程同时调用同步薄壳）不会等待一个属于错误 loop 的锁，而是抛出 `SessionError`。scope 空闲后，另一个线程发起的不重叠调用可以正常运行。这道门只覆盖一个 Agent 实例和一个进程；多个进程共享检查点或会话后端时，应用仍需自行协调。

有状态工具不会暗中继承 Agent 的运行 scope。`MemoryTool` 和 `RAGTool` 默认使用 `scope_policy="fixed"`。一个 Agent 服务多个租户时，可显式选择 `scope_policy="merge_run"`：它只填充固定 scope 中为空的 `user`、`agent`、`app`，并在固定值与运行值冲突时直接报错。`session` 默认不继承，以免长期记忆或知识库意外退化成单次会话数据；确有需要时通过 `inherit_dimensions=...` 加入。应用也可以传 callable 实现自己的归属规则。

## 异步

每个对外能力都为其同步方法暴露一个 `a*` 孪生形式：Agent 暴露 `arun`，记忆暴露 `asearch` / `aadd` / `aupdate`，RAG 暴露 `aingest_text` / `aingest_file`，以此类推。Token 流式输出位于下一层，即 LLM 客户端上的一个异步生成器，用 `async for` 来消费。那些同步方法（`run`、`resume`、`stream_run`）只是一行薄壳，把异步主体驱动到完成，从而让脚本和 notebook 保持简单。

### 异步运行一个 Agent

下面这段代码无需 API key、无需联网即可运行，与 [`examples/09_async.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/09_async.py) 中随附的完全一致：

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

`await agent.arun("hi")` 返回的 `RunResult` 与同步的 `agent.run("hi")` 返回的完全相同；`.final_output` 保存着回复文本。流式调用会一段一段地产出文本增量，这也是列表推导式使用 `async for` 的原因。

### `a*` 对照表

| 能力 | 异步形式 | 同步薄壳 |
| --- | --- | --- |
| 运行一个 Agent | `agent.arun(...)` | `agent.run(...)` |
| HITL / 崩溃后恢复 | `agent.aresume(...)` | `agent.resume(...)` |
| 流式返回 Agent 的回复 | `agent.astream_run(...)` | `agent.stream_run(...)` |
| 追加到会话历史 | `agent.add_messages(...)` | （仅异步） |
| 读取会话历史 | （异步主体为内部实现） | `agent.get_history(...)` |

[LLM 客户端](llm-clients.md) 是原生异步的：`chat` 是一个异步调用，`stream` 是一个异步生成器，客户端本身没有单独的同步方法。

```python
resp = await llm.chat([{"role": "user", "content": "Hello"}])
print(resp.content)
async for piece in llm.stream([{"role": "user", "content": "Tell a joke"}]):
    print(piece, end="")
```

`llm.stream(...)` 是一个产出文本增量的异步生成器。不带工具时它只产出字符串；传入工具时，原生支持流式函数调用的适配器会在文本流耗尽后额外产出一个最终的 `LLMResponse`（也就是 Agent 的流式工具循环所消费的那条通道）。文本工具模拟不支持流式工具循环，因此 `emulate_tools=True` 时请使用 `run` / `arun`，不要使用 `stream_run` / `astream_run`。一次流式结束后，你可以读取 `llm.last_stream_stats` 来获取该次调用的用量、延迟和结束原因。

记忆和 RAG 遵循同样的形态：[Scope 示例](#隔离记忆与检索) 中展示的同步 `search` / `add` 都有异步孪生形式 `asearch` / `aadd` / `aupdate`，RAG 的入库则有 `aingest_text` / `aingest_file`。在事件循环内部使用异步形式，在脚本中使用同步形式。参见 [记忆](memory.md) 和 [检索与 RAG](retrieval-and-rag.md)。

!!! note "在正在运行的事件循环内，请 await 异步形式"
    同步薄壳（`run`、`resume`、`stream_run`、`get_history`）会把异步主体驱动到完成，而这在一个已经运行的事件循环内是无法做到的。在异步应用、Jupyter 或 FastAPI 处理函数中，请直接调用 `a*` 方法（`await agent.arun(...)`、`async for piece in agent.astream_run(...)`），而不是同步薄壳。
