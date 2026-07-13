# 记忆（Memory）

记忆（memory）让 agent 拥有可跨会话长期保留的事实：用户对什么过敏、住在哪里、喜欢怎样的咖啡。agentmaker 内置两种互补的记忆类型。`Memory` 是语义记忆（semantic memory），你写入自由形式的事实，再按语义把它们召回。`KVMemory` 是键值记忆（key-value memory），你把结构化的事实写在一个确切的键（key）下，之后原样读回。当一个事实比较模糊、你希望取回最相关的若干条时（比如「我该避开什么食物」），用 `Memory`；当一个事实是确定且单值的（`location = Beijing`）时，用 `KVMemory`。

## 快速上手

`Memory` 把一个作为唯一事实来源的存储（`MemoryStore`）与一个由 [检索与 RAG](retrieval-and-rag.md) 构建的检索索引配对起来。嵌入器（embedder）把文本转成向量，让意思相近的内容在向量空间里彼此靠近；下面的代码片段用的是 `FakeEmbedder`，它是一个确定性的离线替身，因此无需 API key、无需联网即可运行。以下内容与 [`examples/04_memory.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/04_memory.py) 完全一致：

```python
from agentmaker import Memory, MemoryStore
from agentmaker.retrieval import build_sqlite_hybrid
from agentmaker.testing import FakeEmbedder

memory = Memory(retriever=build_sqlite_hybrid(FakeEmbedder()), store=MemoryStore())

memory.add("I am allergic to peanuts")
memory.add("I like oat milk in the evening")
memory.add("I work as a backend engineer")

# Note: FakeEmbedder is a deterministic hash-based stand-in, so ranking is stable but NOT
# semantic. With a real embedder (OpenAIEmbedder), the allergy fact would rank on top here.
print("Top matches for 'what food should I avoid':")
for hit in memory.search("what food should I avoid", top_k=2):
    print("  -", hit.content)
```

在生产中，把 `FakeEmbedder()` 换成 `OpenAIEmbedder()`（它需要 `OPENAI_API_KEY`），排序就会变成真正基于语义的。

`search` 返回一个 `RetrievalResult` 列表，按最优先的顺序排列。每条结果都带有 `content`、一个综合 `score`、来源 `id`，以及一个 `metadata` 字典：

```python
for hit in memory.search("what food should I avoid", top_k=2):
    print(hit.content, hit.score, hit.metadata["final"])
```

## search 如何排序

`search` 并不只按相关性排序。受 Generative Agents 检索模型的启发，最终得分是三个分量的加权和，每个分量都归一化到 0..1 区间：

- **relevance（相关性）**：记忆与查询的匹配程度，来自混合检索（向量相似度加关键词检索，二者融合）。
- **recency（新近度）**：记忆有多新，按一个半衰期（half-life）衰减（越新的记忆越接近 1，越旧的越接近 0）。
- **importance（重要性）**：记忆自身的 `importance` 值（0..1），在写入时设定。

这三个权重、半衰期，以及默认返回条数都放在 `MemoryConfig` 里，并有一组合理的基线默认值（所有权重均为 `1.0`，`recency_halflife_hours=72.0`，`search_top_k=5`）。在构造时传入一个 config 可做全局调优，或者在每次调用时以关键字参数的形式传给 `search` 做单次覆盖：

```python
from agentmaker import Memory, MemoryStore, MemoryConfig
from agentmaker.retrieval import build_sqlite_hybrid
from agentmaker.testing import FakeEmbedder

memory = Memory(
    retriever=build_sqlite_hybrid(FakeEmbedder()),
    store=MemoryStore(),
    config=MemoryConfig(recency_halflife_hours=24, importance_weight=2.0),
)

# per-call override wins over the config for this one search
hits = memory.search("coffee", top_k=3, recency_weight=0.0)
```

把三个权重都设为 `0`，就退化为纯相关性排序。每条返回结果都在 `hit.metadata` 下暴露它的分量得分，分别是 `relevance`、`recency`、`importance` 和 `final`，这在调权重时很有用。

!!! note "硬过滤 vs 软排序"
    这三个权重是一种软排序：它们只重排，不排除。若要在排序之前对候选做硬过滤（例如按某个 metadata 字段），给 `filters=` 传一个 `MetadataFilter` 列表。过滤契约以及后端必须声明哪些列，见 [检索与 RAG](retrieval-and-rag.md)。

## 单条记忆：MemoryItem

`add` 返回它所存储的 `MemoryItem`，`search` 的结果也各自映射回一条。它的字段：

| 字段 | 含义 |
| --- | --- |
| `content` | 记忆的正文文本。 |
| `id` | 在一份完整归属 scope 内的标识符；除非你自行设置，否则自动生成 uuid。 |
| `type` | 一个自由形式的标签（默认为 `"semantic"`）；不做强制约束，纯粹供你自己分组。 |
| `importance` | 0..1 之间的重要性（默认为 `0.5`）；参与 importance 得分并影响 `forget`。 |
| `created_at` | 记忆被记录的时间。 |
| `updated_at` | 最近一次正文编辑的时间；一旦编辑过，新近度就从这里开始衰减。 |
| `last_accessed_at` | 最近一次被检索命中的时间（一个可选的新近度锚点）。 |
| `invalid_at` | 软失效时间；`None` 表示有效。 |
| `superseded_by` | 取代了这条记忆的那条更新记忆的 id。 |
| `metadata` | 一个附带的字典，默认为空。 |

同一个显式 id 可以存在于不同的 sibling scope 中。`MemoryStore.get` 和 `MemoryStore.replace` 是点访问操作：如果 `Scope(base="memory", user="alice")` 这类较粗的 scope 在不同 agent 或 session 维度下命中多条同 id 记录，它们会抛出 `RetrievalError`，而不是任意挑选一个 sibling。请传入收窄到单一归属 footprint 的 scope。search 和 `all` 等集合操作仍采用常规的「只过滤非空维度」语义。

你可以在写入时控制 `type`、`importance` 和 `metadata`：

```python
memory.add("Ships to production on Fridays", type="procedural", importance=0.9,
           metadata={"team": "backend"})
```

## 聪明地写入：SmartWriter

对每一条进来的消息都调用 `add`，很快就会让记忆里塞满重复条目和过时的矛盾内容。`SmartWriter` 是一个 Mem0 风格的智能写入层，用来保持记忆干净。对于每一段输入，它会：

1. 用 LLM 从文本中**提取（extract）**原子事实，
2. 为每个事实在已有记忆里**检索（search）**，
3. 让 LLM **决定（decide）** `ADD`、`UPDATE`、`DELETE`、`NOOP` 之一，然后
4. **执行（execute）**该决定。

`write` 是异步的，且需要一个 LLM（像 DeepSeek 这样便宜的模型就很合适）。它对每个事实返回一条记录，因此你能确切看到发生了什么：

```python
import asyncio
from agentmaker import Memory, MemoryStore, SmartWriter, LLMClient
from agentmaker.retrieval import build_sqlite_hybrid, OpenAIEmbedder

memory = Memory(retriever=build_sqlite_hybrid(OpenAIEmbedder()), store=MemoryStore())
writer = SmartWriter(memory, LLMClient("deepseek"))

records = asyncio.run(writer.write("I moved from Shanghai to Beijing last month"))
for r in records:
    print(r["op"], r["fact"])   # each record has: fact, op, id, content
```

`UPDATE` 和 `DELETE` 不会物理抹除旧的事实。它们对其做**软失效**：旧行仍留在存储里（仍可供审计查看），并带上一个 `invalid_at` 时间戳，而 `UPDATE` 会通过 `superseded_by` 把它链接到它的后继。于是「从上海搬到北京」在取代旧位置的同时，并不抹掉历史。

`SmartWriter` 被特意设计为故障安全（fail-safe）的。如果事实提取无法被解析，`fail_open=True`（默认）会退化为把整段输入当作一条事实来存储，这样什么都不会丢失；对于闲聊或敏感文本，如果你宁愿丢弃也不愿整段存下，就设 `fail_open=False`。如果协调（reconcile）步骤返回了任何无效内容，它会回退到 `ADD`，因此一个犯迷糊的模型永远不会触发错误的删除。要更改提取所用的语言或类别，传入你自己的 `extract_prompt` / `reconcile_prompt`。

## 更新与遗忘

除了 `add` 和 `search`，`Memory` 还暴露了完整的生命周期：

- `update(id, content)` 在单个原子事务里替换一条记忆的正文，并把它的新近度重新计到编辑时间。
- `invalidate(id, superseded_by=...)` 对一条记忆做软失效：正本记录会标为失效并请求清理索引；即使残留陈旧索引行，召回也会排除它。记录本身保留下来供审计。这正是 `SmartWriter` 所用的方式。
- `delete(id)` / `delete_many(ids)` 从权威存储中物理移除记忆，并请求清理派生索引。索引清理是 best-effort，可能还需执行 `rebuild_index()` / 对账，因此仅调用这个 API 不保证立即从所有后端完成物理擦除。
- `forget(strategy=...)` 批量修剪并返回被删除的 id。策略有：`"importance"`（丢弃低于 `threshold` 的条目）、`"age"`（丢弃早于 `max_age_days` 的条目），以及 `"capacity"`（只保留最重要且最新的前 N 条）。
- `stats()` 返回 `{"total": ..., "by_type": {...}}`，是纯计数，不调用 LLM。

有两个生命周期操作会用到 LLM，因此是异步协程（需要在构造时传入 `llm=`）：

- `summary(query=None)` 把匹配到的记忆折叠成一段连贯的文字。
- `consolidate()` 把所有记忆交给 LLM，合并重复项、在任何矛盾中保留最新的一个，然后重写存储。它返回 `{"before": ..., "after": ...}`。和 `SmartWriter` 一样，它对旧条目做软失效，而不是删除。

```python
paragraph = await memory.summary()
result = await memory.consolidate()   # {"before": 12, "after": 8}
```

!!! note "异步接口"
    读写基础操作（`add`、`search`、`update`、`delete`、`forget` 等）各自都有一个 `a*` 对应版本（`aadd`、`asearch` 等），把阻塞的数据库和嵌入工作挪出事件循环执行。`summary` 和 `consolidate` 天生就是异步的，因为它们要调用模型。

## 键值记忆

对于确定且单值的事实，用语义召回既大材小用又不够精确。`KVMemory` 每个键只存一个值，并原样读回，不做任何猜测。`KVStore` 是底层的 SQLite 表（值为字符串）；`KVMemory` 是它之上的一层门面（facade），负责 JSON 编码和解码，因此值可以是字符串、数字、列表或字典。它带有一个固定的 [scope](retrieval-and-rag.md)（作用域）用于归属：

```python
from agentmaker import KVStore, KVMemory, Scope

kv = KVMemory(KVStore(), scope=Scope(base="kv", user="alice"))

kv.set("location", "Beijing")
kv.set("allergies", ["peanuts"])

print(kv.get("location"))        # "Beijing"
print(kv.get("theme", "light"))  # default when the key is missing
print(kv.as_dict())              # {"location": "Beijing", "allergies": ["peanuts"]}
```

`set` 就地覆写，`get(key, default=None)` 在键不存在时返回默认值，`delete(key)` 移除它，`as_dict()` 返回整套解码后的内容。

## 把记忆赋予 agent

`MemoryTool` 把一个 `Memory`（可选地带上一个 `SmartWriter`）包装成一个 [工具](tools.md)，这样 agent 就能在对话中途自行决定去记住和召回。像注册其它任何工具一样注册它：

```python
from agentmaker import Agent, Memory, MemoryStore, MemoryTool, LLMClient
from agentmaker.retrieval import build_sqlite_hybrid, OpenAIEmbedder

memory = Memory(retriever=build_sqlite_hybrid(OpenAIEmbedder()), store=MemoryStore())
agent = Agent("assistant", LLMClient("deepseek"), tools=[MemoryTool(memory)])
```

这个工具接收一个 `action` 加上一个 `content` 或 `query`，并分派到：`remember`、`recall`、`summary`、`stats`、`forget` 和 `consolidate`。传入一个 `writer=` 可让 `remember` 走 `SmartWriter`，从而自动去重和重写，而不是走一次普通的 `add`。

由于某些动作会修改或删除已存的数据，`MemoryTool` 会把它们挡在人工确认之后：`forget` 和 `consolidate` 始终需要确认，`remember` 在挂了 writer 时也需要（因为 `SmartWriter` 可能会更新或删除已有记忆）。读取动作（`recall`、`summary`、`stats`）以及一次普通的 add 会直接放行。这道由 writer 触发的确认默认开启；传入 `MemoryTool(memory, writer, confirm_writer_edits=False)` 即可关闭它。确认关卡是如何接线的，见 [护栏与人在回路](guardrails-and-hitl.md)（人在回路）。

工具默认使用 Memory 的固定 scope。单个 Agent 实例服务多个租户时，使用 `MemoryTool(memory, scope_policy="merge_run")`：它从当前运行中填充固定 scope 为空的 `user`、`agent`、`app`，并拒绝冲突；是否继承 session 需通过 `inherit_dimensions` 显式选择。`recall` 和 `summary` 的结果在返回模型前会标记为外部内容；这种定界式提示注入防护只能降低风险，并不是安全沙箱。

## 持久化与隔离

上面那两个构造参数默认使用内存中的 SQLite，进程退出时即被清空。要持久化，就给存储和检索后端同一个文件路径：

```python
memory = Memory(
    retriever=build_sqlite_hybrid(FakeEmbedder(), db_path="memory.db"),
    store=MemoryStore(db_path="memory.db"),
)
```

`MemoryStore` 是权威的唯一事实来源：它保存完整的 `MemoryItem` 记录，而检索索引只是一个可重建的派生物。如果索引丢失，或者你更换了后端，`rebuild_index()` 会把每一条已存记忆重新嵌入回索引。

`Memory.from_config()` 组装的资源由该 `Memory` 持有，`close()` 或上下文管理器会释放它们。直接构造时传入的对象仍由调用方持有，因此 retriever 可以安全地与 RAG 或其它管理器共享。

`summary(top_k=N)` 会让有查询和无查询的摘要都最多使用 `N` 条记忆，避免存储增长后生成无界 prompt。

要做多用户隔离，就给每个用户各自的 scope。在构造时传入 `scope=`，让每一次写入和读取都留在那个所有者的数据范围内，并且每个用户保持一个 `Memory`（和一个 `SmartWriter`）：

```python
from agentmaker import Scope

alice = Scope(base="memory", user="alice")
memory = Memory(retriever=..., store=..., scope=alice)
```

scope 在检索、记忆和 RAG 子系统之间共享（RAG 即 retrieval-augmented generation，检索增强生成，从你自己的文档中作答）。完整的 scope 模型见 [检索与 RAG](retrieval-and-rag.md)。
