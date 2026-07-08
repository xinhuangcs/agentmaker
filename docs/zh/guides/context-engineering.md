# 上下文工程

上下文工程（context engineering）是决定「究竟哪些内容真正进入模型窗口」的最终组装与质量控制环节。检索（retrieval）与记忆（memory）是供货方：它们返回按相关性排序的候选片段。本子系统拿到这些候选后，挑出最值得占位的那些、给它们排序、把一切控制在明确的 token（模型处理文本的最小计量单位）预算之内，并在溢出时进行压缩。每当你要拼装 RAG（retrieval-augmented generation，检索增强生成，即把检索到的文本喂给模型）、长对话历史，或多步 agent（智能体，能自主规划并调用工具完成任务的程序）轨迹，并且希望提示词（prompt，发给模型的输入文本）保持在已知预算内、而不是一路膨胀直到撑爆窗口时，就该用到它。

贯穿始终的设计原则是「显式分配」而非「被动堆积」：每一条争夺窗口的数据流都从同一本总账里领取自己的配额，因此提示词的任何一半都无法悄悄吞掉另一半。

本页的所有符号都可以从顶层导入（`from agentmaker import ContextBuilder`），只有轨迹缩减器（trajectory reducers）例外，它们位于 `agentmaker.context`。

## 流水线一览

[`ContextBuilder`](#contextbuilder) 运行一条固定的四阶段流水线：

```
Gather      collect candidates from each source
   -> MMR   per-source de-duplication + diversity selection
   -> Budget   three-region budget with two-round quota borrowing
   -> Structure   fixed layout: system -> memory -> rag -> history -> tool -> question
```

它不会再做第二次相关性重排（rerank）。基线排序来自底层检索（见 [检索与 RAG](retrieval-and-rag.md)）；这一层只做「单靠组装」就能做的事：去重、在预算内挑选、把预算分配到各来源、以及把各区段排好版。

## 估算 token：`count_tokens`

每一个预算决策都从一次 token 估算开始。`count_tokens` 是一个零依赖的估算器，面向中日韩（CJK）与西文混排的文本（每个中文、日文或韩文字符算作一个 token；其余内容大致按每四个字符一个 token 计）。

```python
from agentmaker import count_tokens

count_tokens("hello world")   # 3
```

!!! note "只是估算，不用于计费"
    `count_tokens` 是发送前的预算估算；它绝不参与成本或配额核算（那些一律使用模型返回的真实 token 用量）。它有意不按空白切分，因此不含空格的长串（base64、长 URL）仍按每四个字符一个 token 的规则计量，而不是当成一个 token。

这个估算器是一个可插拔的接缝。`ContextBuilder` 与 `HistoryCompactor` 接受一个类型为 `TokenCounter` 的 `token_counter` 参数，而 `TokenCounter` 其实就是 `Callable[[str], int]`；缩减器则以 `counter` 之名接受同样的可调用对象。如果生产环境需要更精确的核算，可以注入一个更精确的计数器（例如基于 `tiktoken` 的那种）。

## 挑选不冗余的候选：`mmr_select`

`mmr_select` 实现了 MMR（maximal marginal relevance，最大边际相关性，一种「既相关又互不重复」的选取算法）：它挑出一个既相关、彼此又各不相同的子集。把每个候选都塞进窗口会浪费 token、稀释信号（即 context rot，上下文腐化）；MMR 一次只选一项，每一步都在「候选有多相关」与「它和已选内容有多相似」之间权衡。

```python
from agentmaker import mmr_select, RetrievalResult

candidates = [
    RetrievalResult(content="Cats are great pets.", score=0.9, source="rag", embedding=[1.0, 0.0]),
    RetrievalResult(content="Cats make wonderful pets.", score=0.8, source="rag", embedding=[0.99, 0.01]),
    RetrievalResult(content="The Eiffel Tower is in Paris.", score=0.6, source="rag", embedding=[0.0, 1.0]),
]

selected = mmr_select(candidates, top_k=2, lambda_=0.7)
for r in selected:
    print(r.content)
# Cats are great pets.
# The Eiffel Tower is in Paris.
```

近乎重复的第二句被丢弃，取而代之的是一个主题上截然不同的条目。其签名为：

```python
mmr_select(candidates, *, top_k=None, lambda_=0.7, dedup_threshold=0.95)
```

- `top_k`：最多选取的数量；`None` 表示不限数量（仅依靠去重来剔除近似重复项）。
- `lambda_`：相关性与多样性之间的权衡，取值范围 `[0, 1]`。`1.0` 为纯相关性（不施加去重惩罚）；越低越强调多样性。默认 `0.7` 体现了「检索结果本已排过序，因此适度去重即可」。
- `dedup_threshold`：某候选与任何已选条目的余弦相似度（cosine similarity，两个向量方向接近程度的度量，1 表示完全同向）达到或超过此值时，即被视为近似重复并直接丢弃。`0.95` 意味着两项必须几乎完全相同才算重复；大于 `1` 的取值实际上会关闭近似重复剔除。

MMR 复用检索已经随每个 `RetrievalResult` 带回的 `embedding` 向量（embedding，把文本映射成一串数字向量、便于比较语义的表示），因此不会重新计算任何东西。没有 embedding 的候选（例如仅靠关键词命中的项）被当作相似度 `0`：既然无法判断冗余，就不惩罚其多样性。逐字节完全相同的内容会先被合并，只保留得分最高的那一份。

## 来源：`ContextSource` 与 `CallableSource`

构建器通过一套统一的供货接口来消费候选。`ContextSource` 是抽象基类：每个来源都有一个 `name`（决定它从哪份配额支取）和一个返回 `RetrievalResult` 列表的 `fetch(query, scope=None)` 方法，外加一个异步对应版本 `afetch`。

多数情况下你不用自己写子类。`CallableSource` 会把任意 `(query)` 或 `(query, scope)` 形式的可调用对象适配成一个来源，因此 `memory.search`、`rag.retrieve` 或你自己的函数都能直接接入：

```python
from agentmaker import CallableSource, RetrievalResult

def search_docs(query: str) -> list[RetrievalResult]:
    return [
        RetrievalResult(content="Meals are capped at 80 per day, no receipt needed.", score=0.9, source="rag"),
        RetrievalResult(content="Hotels are capped at 500 per night, receipt required.", score=0.7, source="rag"),
    ]

source = CallableSource("rag", search_docs)
```

`name`（这里是 `"rag"`）决定该来源消耗哪份预算配额；它必须是配置中 `source_ratios` 的一个键（见下文）。

### 贯穿传递 scope

`scope` 是贯穿一次运行始终的会话标识（见 [作用域隔离](retrieval-and-rag.md)）。它如何传到你的可调用对象，由 `pass_scope` 控制：

```python
CallableSource("memory", memory.search)                                             # keyword-only scope, uses its own
CallableSource("memory", lambda q, s: memory.search(q, scope=Scope(user=s.user)))   # positional, by the run's user
CallableSource("rag", rag.retrieve, pass_scope=True)                                # force pass by keyword scope=
CallableSource("rag", lambda q: rag.retrieve(q, top_k=8))                           # custom top_k, no scope
```

默认情况下（`pass_scope=None`），模式会根据位置参数的个数自动判定：拥有两个或更多位置参数的可调用对象，会把 `scope` 作为第二个位置参数收到，否则收不到它。

!!! warning "自动判定只数位置参数"
    以关键字方式接收 scope 的可调用对象（`def f(query, *, scope=None)`，`memory.search` 和 `rag.retrieve` 正是如此）不会被自动识别，也收不到本次运行的 scope。这是有意为之：直接绑定这些方法，让它们使用各自的 scope。若要把运行 scope 强行塞进一个仅限关键字的参数，传 `pass_scope=True`；若要强制关闭，传 `pass_scope=False`。

## 组装提示词：`ContextBuilder` { #contextbuilder }

`ContextBuilder` 运行完整的流水线并返回组装好的文本。它有两个入口。

`build` 产出一个扁平字符串，`system -> sections -> current question`，适用于单轮或 RAG 式的调用：

```python
from agentmaker import CallableSource, ContextBuilder, ContextConfig, RetrievalResult

def search_docs(query: str) -> list[RetrievalResult]:
    return [
        RetrievalResult(content="Meals are capped at 80 per day, no receipt needed.", score=0.9, source="rag"),
        RetrievalResult(content="Hotels are capped at 500 per night, receipt required.", score=0.7, source="rag"),
    ]

builder = ContextBuilder(ContextConfig.for_window(None, fallback_window=8000))
context = builder.build(
    "how much can I spend on meals?",
    sources=[CallableSource("rag", search_docs)],
    system_prompt="You are a finance assistant.",
)
print(context)
```

```text
You are a finance assistant.

[Knowledge]
- Meals are capped at 80 per day, no receipt needed.
- Hotels are capped at 500 per night, receipt required.

[Current question]
how much can I spend on meals?
```

各区段的标题（`[Knowledge]`、`[Current question]` 等）来自提示词注册表；如果某个自定义来源名没有注册对应标题，则回退为 `[name]`。

`build_block` 只组装动态来源块（memory / RAG / ...），不含系统提示词、也不含当前问题。它用于多轮对话：把这个块作为一条系统消息注入，并把对话历史作为带角色的消息单独传入，这样 user / assistant 角色就不会被压平抹掉。当没有任何候选时，它返回空字符串。

```python
build(query, *, sources, system_prompt="", scope=None) -> str
build_block(query, *, sources, scope=None, budget=None) -> str
abuild_block(query, *, sources, scope=None, budget=None) -> str   # async; fans out over sources concurrently
```

异步的 `abuild_block` 沿用同样的预算约定。

### 预算旋钮：`ContextConfig`

`ContextConfig` 是一份冻结、不可变的预算配置。它用比例而非绝对数字来表达预算，因此换用窗口更大的模型只是等比缩放的事，不需要重新调参。

| 字段 | 默认值 | 控制什么 |
| --- | --- | --- |
| `max_tokens` | `None` | 总的上下文预算。没有硬编码默认值：请根据模型的真实窗口来设置。 |
| `output_reserve_ratio` | `0.2` | 为输出加当前问题预留的比例（不参与候选的竞争）。 |
| `source_ratios` | `{"history": 0.35, "rag": 0.30, "memory": 0.20, "tool": 0.15}` | 每个来源在动态区中所占的份额。键为来源名。 |
| `mmr_lambda` | `0.7` | 作为 `lambda_` 传给 `mmr_select`。 |
| `dedup_threshold` | `0.95` | 作为 `dedup_threshold` 传给 `mmr_select`。 |
| `allow_borrow` | `True` | 某来源闲置的配额是否会在第二轮中重新分配给仍有候选待放置的来源。 |
| `min_chunk_tokens` | `64` | 一份配额必须至少能容纳的单个候选大小，用于合理性检查。 |

用 `for_window` 从模型窗口设定 `max_tokens`：

```python
ContextConfig.for_window(context_window, *, use_ratio=0.5, fallback_window=None, **kwargs)
```

```python
ContextConfig.for_window(LLMClient("deepseek").context_window)   # 1M window -> max_tokens = 500,000
ContextConfig.for_window(None, fallback_window=8000)             # unknown local model, explicit fallback
```

`use_ratio` 默认为 `0.5`：上下文只占用窗口的一半，为输出和安全余量留出充裕空间。`fallback_window` 没有默认值，窗口未知时必须显式提供，从而逼你钉死一个保守数值，而不是悄悄替你挑一个。

!!! note "大声报错的校验"
    「来源名不在 `source_ratios` 中」和「两个来源同名」这两种情况，都会在任何抓取开始前就被拒绝：前者会悄无声息地拿到零配额、永远不出现，后者会在组装时覆盖候选。配置在构造时还会校验每个来源的配额至少能容纳一个完整的候选块；小到连最相关的那一项都放不下的配额会立即报错。

两轮分配先按比例给每个来源发放各自的配额，然后（当 `allow_borrow` 开启时）把任何闲置配额分给仍有候选待放置的来源，按各自还想要多少来分摊余量，而不是按输入顺序。

## 窗口预算：`WindowBudgetConfig` 与 `WindowBudget`

当多条数据流争夺同一个窗口时（系统提示词、工具 schema、检索块、agent 轨迹以及输出预留），若各自独立决定自身大小，就可能把总量推过窗口上限。窗口预算把整个分配收拢进同一本总账：

```
whole window = output reserve + fixed overhead (system + tool schemas) + retrieval block + trajectory
```

`WindowBudgetConfig` 持有可序列化的旋钮：

| 字段 | 默认值 | 控制什么 |
| --- | --- | --- |
| `desired_output_tokens` | `4096` | 本次运行你最多希望模型生成多少 token（输出预留的主旋钮）。 |
| `max_output_fraction` | `0.5` | 小窗口护栏：输出预留最多占窗口的这个比例。 |
| `rag_ratio` | `0.35` | 检索块在可分配余额中所占的份额；其余归轨迹。 |

输出预留取以下几项中的最小值：`desired_output_tokens`、`window * max_output_fraction`，以及（当模型的单次调用输出上限已知时）该上限。每一道钳制各自防守一种失效模式：预留不超过你所要求的、大窗口下不留一块模型永远填不满的死区、小窗口下不吞掉输入。只配置了一个比例（`rag_ratio`）；轨迹取剩余部分，这从结构上杜绝了「两个比例相加超过窗口」的可能。

`WindowBudget` 是一个值对象，每次运行时根据真实窗口加上实测的固定开销计算一次。用 `for_run` 来构建它；当窗口未知时它返回 `None`（此时调用方回退为不设上限）：

```python
from agentmaker import WindowBudget, WindowBudgetConfig
from agentmaker.testing import ScriptedLLM

llm = ScriptedLLM(context_window=200_000)
budget = WindowBudget.for_run(llm=llm, cfg=WindowBudgetConfig(), system_tokens=800, tool_tokens=1200)

budget.rag_budget                              # retrieval block cap
budget.trajectory_budget(rag_in_scope=True)    # trajectory trimming budget
```

```python
WindowBudget.for_run(*, llm, cfg, system_tokens=0, tool_tokens=0, rag_ratio=None) -> Optional[WindowBudget]
```

它的只读账目：

- `fixed`：固定开销总和，`system_tokens + tool_tokens`。
- `spendable`：扣除输出预留与固定开销后，可在检索块与轨迹之间分配的余额（永不为负）。
- `rag_budget`：检索块的上限，`spendable * rag_ratio`。这正是作为 `build_block(..., budget=...)` 传入的值，好让检索块从共享总账里支取，而不是自己再预留一次输出。
- `trajectory_budget(*, rag_in_scope)`：范式轨迹的裁剪预算，会根据被裁剪的数据是否已包含检索块而分岔处理。

工具 schema 是搭在请求的 `tools=` 载荷里、而非 `messages` 里，因此轨迹裁剪看不到它们；总账会单独扣除它们，这样不断增长的轨迹就无法把工具 schema 挤出窗口。

## 压缩对话历史：`HistoryCompactor`

对话会无限增长：几十轮之后终将超出预算并稀释信号。`HistoryCompactor` 用一个 LLM 把较早的若干轮总结成一段回顾（recap），并把最近的若干轮原样保留。最近的对话必须保持精确（模型要接着它继续作答）；久远的历史则只需要一份摘要。

```python
from agentmaker import HistoryCompactor, Message
from agentmaker.testing import ScriptedLLM

llm = ScriptedLLM(["The user asked how to get a refund and was told to check Settings > Billing."])
compactor = HistoryCompactor(llm, keep_recent=2, trigger_tokens=10)

history = [
    Message(content="How do I get a refund?", role="user"),
    Message(content="Open Settings then Billing.", role="assistant"),
    Message(content="I did that but I am still stuck.", role="user"),
    Message(content="Let me escalate this for you.", role="assistant"),
]

compacted = compactor.compact(history)
for m in compacted:
    print(m.role, "::", m.content)
```

```text
system :: [Recap] The user asked how to get a refund and was told to check Settings > Billing.
user :: I did that but I am still stuck.
assistant :: Let me escalate this for you.
```

构造函数：

```python
HistoryCompactor(llm, *, keep_recent=4, trigger_tokens=2000, max_summary_tokens=1000,
                 summary_prompt=None, prompts=None, token_counter=count_tokens)
```

- `keep_recent`（默认 `4`，必须 `>= 1`）：原样保留多少个最近的轮次。
- `trigger_tokens`（默认 `2000`，必须 `>= 0`）：仅当历史总量超过此值才压缩，否则原样返回历史、不花费任何 LLM 调用。
- `max_summary_tokens`（默认 `1000`，必须 `>= 1`）：回顾的硬上限，超出即截断。这可以避免增量合并的摘要在数百轮之间无限膨胀，因为缓存的摘要会在下一轮作为输入回喂。
- `summary_prompt`：覆盖默认的摘要指令（例如用来切换语言）；若省略则使用框架默认值。

`compact(history, *, summarize=None)` 返回压缩后的 `Message` 列表；`acompact(history, *, asummarize=None)` 是其异步对应版本。当历史不超过阈值、或至多有 `keep_recent` 个轮次时，原始历史原封不动地返回。若总结失败，压缩器会保留原始历史，而不是把它弄丢。

`CompactionConfig` 是喂给 `HistoryCompactor.from_config(llm, config)` 的可序列化切片：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `keep_recent` | `4` | 原样保留的最近轮次。 |
| `trigger_tokens` | `2000` | 触发压缩的历史 token 数。 |

!!! note "压缩不等于组装期压缩"
    历史压缩用一次 LLM 调用来总结一个大对象（整段对话）。它有别于在组装期裁剪零散的检索候选，而后者构建器并不做；候选的大小转而在上游通过分块（chunking）来控制。

## 裁剪范式轨迹：缩减器

与历史压缩互补，缩减器层在 agent 自身的工作轨迹逼近窗口预算时对其裁剪，先丢掉最不重要的信号，同时保住每种范式的命脉。笼统的摘要会剥掉反思循环过往的批评要点、或计划里精确的步骤编号，从而破坏该范式，因此每种范式都有各自「意识到损失」的策略。它们位于 `agentmaker.context`：

```python
from agentmaker.context import reduce_agent, reduce_plan, reduce_reflection, tokens_of, REDUCERS
```

- `reduce_agent` 裁剪统一循环的工具调用轨迹：最近的原子单元（一条 assistant 消息加上它的工具结果）原样保留，更早的则被总结进一条 system 条目。
- `reduce_plan` 裁剪计划的步骤结果，原样保留最近的步骤，并保住关键的数字、日期与结论。
- `reduce_reflection` 裁剪反思轨迹，保留最新的答案外加一份去重后的过往批评要点清单，并丢弃被取代的草稿。

三者都是异步的，接受一个由调用方提供的 `summarize(text, instruction) -> str` 异步回调，外加来自 `WindowBudget.trajectory_budget` 的 token `budget`。`REDUCERS` 把 `"agent"` / `"plan"` / `"reflection"` 映射到这些函数，而 `tokens_of(*texts, counter=count_tokens)` 估算若干文本的总 token 数。如果必须保留的部分已经超出预算，缩减器会抛出 `ContextWindowExceeded`，而不是悄悄截断。

`ReducerConfig` 持有「保留多少最近文本不压缩」的可序列化旋钮：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `agent_keep_recent_steps` | `3` | 原样保留的工具轨迹尾部单元数。 |
| `plan_keep_recent` | `3` | 原样保留的计划步骤结果尾部数。 |

轨迹自身的 token 预算不在这份配置里；它来自窗口总账，因此那两个比例相加永远不会超过窗口。

## 把一切配置到一起

`AgentmakerConfig` 把这些子配置（`context`、`reducer`、`compaction`、`window_budget` 等）聚合进一个容器，你在装配根处设置一次、再层层传下去。`to_dict` / `from_dict` 对它做序列化，`for_window(context_window)` 会派生出一个实例，其 `context.max_tokens` 由模型窗口设定。当构建器或压缩器被接入某个 agent 时，检索块与轨迹的预算由共享的 `WindowBudget` 提供，因此 `ContextConfig` 上的 `max_tokens` 可以不设；只有独立的 `build` / `build_block(budget=None)` 调用才需要它。

候选从何而来见 [检索与 RAG](retrieval-and-rag.md)，而你最常接入的记忆来源见 [记忆](memory.md)。
