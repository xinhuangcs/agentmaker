# 检索与 RAG

agentmaker 为「找到相关文本并让 LLM 基于它作答」提供了两层能力。较底层是**检索地基**：一个混合检索器，同时跑稠密（向量）检索和稀疏（关键词）检索，把两路排名融合，并可选地做重排。较上层是 **RAG**（Retrieval-Augmented Generation，检索增强生成：先检索出支撑性的文段，再让模型只用这些文段作答）：它把文件读成文档、切成分块、灌入索引，并给出带引用的问答。同一套地基被 [记忆](memory.md) 共享，因此两个子系统拿到的是同一套检索能力和同一套数据隔离。

想直接对自己的文本做原始检索时，用检索地基；想要一个能给出有据可查、带来源答案的文档知识库时，用 RAG。

## 两层能力速览

| 层 | 包 | 你能得到 |
| --- | --- | --- |
| 检索地基 | `agentmaker.retrieval` | `HybridRetriever`、抽象端口（`Embedder`、`VectorStore`、`KeywordIndex`、`Reranker`、`FusionStrategy`）及其电池实现、`reciprocal_rank_fusion`、`Scope`、`MetadataFilter`、`RetrievalResult`、`IndexSync` |
| RAG | `agentmaker.rag` | `Document` / `Chunk`、`split_document`、`IngestionPipeline`、`RagRetriever`、`Contextualizer`、`MultiQueryExpander` / `HyDETransformer`、`NeighborWindowExpander`、`RAGTool`、`AskResult` / `SourceRef` |

## 快速上手：灌入与检索

下面是 [`examples/05_rag.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/05_rag.py) 的原样内容。它无需 API key、无需联网即可运行：`FakeEmbedder` 顶替真实的嵌入模型，本地 SQLite 后端存放索引。

```python
from agentmaker import IngestionPipeline, RagRetriever, SourceStore
from agentmaker.retrieval import build_sqlite_hybrid
from agentmaker.testing import FakeEmbedder, ScriptedLLM

retriever = build_sqlite_hybrid(FakeEmbedder())
source_store = SourceStore()

pipeline = IngestionPipeline(retriever=retriever, source_store=source_store)
report = pipeline.ingest_text(
    "# Expense Policy\n"
    "## Meals\nThe daily meal allowance is 80, no receipt needed.\n\n"
    "## Lodging\nHotels are capped at 500 per night, receipt required.",
    source="policy.md", fmt="md",
)
print(f"Ingested {report.chunks} chunks.\n")

rag = RagRetriever(retriever, source_store, ScriptedLLM([]))
print("Relevant chunks for 'how much can I spend on meals':")
for chunk in rag.retrieve("how much can I spend on meals", top_k=2):
    print("  -", chunk.content)
```

在生产环境中，把 `FakeEmbedder()` 换成 `OpenAIEmbedder()`，把 `ScriptedLLM([])` 换成真实的 `LLMClient`（见 [LLM 客户端](llm-clients.md)）。`build_sqlite_hybrid` 构建出整个混合检索器（向量存储加关键词索引，共用同一条 SQLite 连接）；`SourceStore` 是 RAG 的事实源存储，保存分块的完整文本。

---

## 检索地基

### 端口与电池

地基采用「端口与适配器」（ports-and-adapters）设计：五个抽象基类定义接口，可替换的电池实现来填充它们。替换某个后端只需写一个子类；`HybridRetriever` 及其之上的一切都无需改动。

| 端口（抽象基类） | 职责 | 默认电池实现 |
| --- | --- | --- |
| `Embedder` | 把文本变成向量 | `OpenAIEmbedder` |
| `VectorStore` | 存储向量，跑最近邻（稠密）检索 | `SqliteVecStore` |
| `KeywordIndex` | 关键词检索，按 BM25（一种经典的词频相关性打分）排序 | `Fts5KeywordIndex` |
| `Reranker` | 用 cross-encoder（交叉编码器，一种把查询和文段一起打分的模型，比向量更精确）重新排序候选 | `CohereReranker` |
| `FusionStrategy` | 把多个排好序的列表合并成一个 | `RRFFusion` |

五个端口和五个电池实现全部从包顶层导出，例如 `from agentmaker import HybridRetriever, OpenAIEmbedder, SqliteVecStore`。

### HybridRetriever

`HybridRetriever` 是与存储无关的编排器。用一个嵌入器、一个向量存储、一个关键词索引来构造它，重排器和融合策略可选：

```python
from agentmaker import HybridRetriever
```

```python
def __init__(self, embedder, vector_store, keyword_index,
             reranker=None, *, config=None, fusion=None)
```

`search` 是读取路径。每一路（向量和关键词）各取 `candidate_pool` 个条目，两个列表被融合（默认用 RRF），如果配了重排器，融合后的候选池会被精炼到 `top_k` 个：

```python
def search(self, query, *, top_k=None, candidate_pool=None,
           scope=None, all_scopes=False, filters=None) -> List[RetrievalResult]
```

省略 `top_k` 或 `candidate_pool` 时，使用 `config` 里的取值。`search_many` 一次为多个查询检索，把它们放在一个批次里统一嵌入，以节省往返开销。每个方法都有异步孪生版本（`asearch`、`asearch_many`、`aadd` 等等）。朴素的 `HybridRetriever` 会并发地跑向量和关键词两路；而由 `build_sqlite_hybrid` 得到的默认 SQLite 后端共用一条连接、并置于同一把锁之下，因此它的异步孪生版本会在单个线程里执行检索（把两路拆开并行只会在那把锁上互相争用）。

写入走 `add`（一个 upsert：用同一个 id 再写一次会覆盖它）、`replace`（用新分块替换一篇文档的旧分块）和 `delete`。多数情况下你不会直接调用它们；RAG 的 `IngestionPipeline` 和记忆会替你驱动。

!!! note "本地使用优先选 `build_sqlite_hybrid`"
    朴素的 `HybridRetriever` 把它的两个索引当成各自独立的连接，因此 `add` 在部分失败时只能尽力补偿。`build_sqlite_hybrid` 让向量存储和关键词索引共用同一条 SQLite 连接，使跨两个索引的写入在单个事务里保持原子性。它还在打开时对嵌入模型做指纹校验，这样换入一个不匹配的模型会当场大声报错，而不是悄悄把不可比较的向量混在一起。

### 倒数排名融合（RRF）

RRF 是两路检索结果合并的默认方式。它只看每个条目的名次，不看原始分数，从而绕开「向量距离和 BM25 分数处在不可比较的量纲上」这个问题，而且无需调参。某条目在一个列表中排名为 `r`（从 1 起算），贡献 `1 / (k + r)` 分；同一个 id 在各列表中的分数相加，因此被两路都命中的条目自然升到顶部。平滑常数 `k` 默认为 60。

你可以对任意一组排好序的列表直接调用这个函数：

```python
from agentmaker import RetrievalResult, reciprocal_rank_fusion

dense = [RetrievalResult(content="A", score=0.91, source="rag", id="1"),
         RetrievalResult(content="B", score=0.83, source="rag", id="2")]
keyword = [RetrievalResult(content="B", score=7.4, source="rag", id="2"),
           RetrievalResult(content="C", score=6.1, source="rag", id="3")]

fused = reciprocal_rank_fusion([dense, keyword], top_k=3)
# "B" appears in both lists, so it fuses to the top; each result's
# metadata carries an "rrf_score".
```

融合按 `id` 对齐（id 为空时退化为按内容对齐）。若想给两路加权而不用朴素 RRF，实现 `FusionStrategy` 并通过 `HybridRetriever(fusion=...)` 注入；`RRFFusion` 只是包装 `reciprocal_rank_fusion` 的默认电池实现。

### RetrievalResult

地基里的每次查询（以及它之上的 RAG 和记忆）都返回一个 `RetrievalResult` 列表，因此不论命中来自哪里，上层都面对同一种形状：

| 字段 | 含义 |
| --- | --- |
| `content` | 命中的文本 |
| `score` | 相关性，按约定越高越相关；只用于排序，不可跨后端比较 |
| `source` | 来源标签，例如 `"rag"` 或文档名 |
| `id` | 该来源内的唯一 id（不存在时为空字符串） |
| `embedding` | 该条目的向量，可用时一并带回，便于上下文工程的 MMR（Maximal Marginal Relevance，最大边际相关，一种考虑冗余的挑选步骤）复用 |
| `metadata` | 附带字段（原始距离、标题路径等等） |

### Scope：隔离与过滤

`Scope` 是一个冻结的 dataclass，从五个维度给每份数据标注归属：`base`、`user`、`agent`、`session` 和 `app`。同一套共享地基可以容纳多个租户的数据而不互相污染。

```python
from agentmaker import Scope

Scope(base="rag", user="alice")
```

过滤永远只做收窄：检索只为你设置了的维度加 `WHERE`，其余维度不加限制。`Scope(user="alice")` 会返回 alice 的全部数据，不论 agent 或 session。按约定每个子系统都设置 `base`（RAG 用 `Scope(base="rag")`，记忆用 `Scope(base="memory")`），从而在共享后端上保持各自数据隔离。

完全为空的 `Scope()` 不限制任何东西，因此一个不带条件的删除或检索会命中整个存储。`require_explicit_scope` 守卫会拒绝这种操作，除非你传入 `all_scopes=True`，从而把一次意外的全局操作变成一个前置的报错。

### 元数据过滤器

`MetadataFilter` 在计算相似度之前先按结构化字段收窄候选（前置过滤）。多个条件之间取 AND，且只有 `eq`（相等）和 `in`（多选一）两个运算符：

```python
from agentmaker import MetadataFilter

MetadataFilter("doc_id", "abc123")                    # doc_id = 'abc123'
MetadataFilter("tag", ["faq", "policy"], op="in")     # tag in ('faq', 'policy')
```

可过滤的字段必须在构建索引时声明，通过 `build_sqlite_hybrid(..., metadata_columns=("doc_id", "tag"))`。过滤一个从未声明的字段会当场大声报错，而不是悄悄返回零命中。

### RetrievalConfig

`RetrievalConfig` 保存那些可调的旋钮，在构造时校验：

```python
from agentmaker import RetrievalConfig

RetrievalConfig(top_k=5, candidate_pool=20, rrf_k=60)   # the defaults
```

`candidate_pool`（每一路取多少条目进入融合和重排）必须至少等于 `top_k`。把 config 传给 `HybridRetriever(config=...)`，或传给 `build_sqlite_hybrid(config=...)`。

### 让索引保持同步

RAG 和记忆都维护一个事实源存储，外加一个必须向它收敛的派生检索索引。这个横切关注点被收口成一个可插拔的接缝 `IndexSync`，于是每个子系统的写入路径只管调用它。默认的 `SyncIndexSync` 同步地写穿并记录簿记（一个用于幂等跳过的内容指纹，外加一个用于自愈的待处理集合）。簿记的存储本身也可替换：`InMemoryBookkeeping` 是零依赖的默认实现（重启后可从事实源重建），`SqliteBookkeeping` 则跨进程持久化它。若要一套完全异步或分布式的方案，自行实现你自己的 `IndexSync`，无需触碰任何子系统的写入路径。

你很少直接构造这些；流水线的 `from_config` 会替你装好 `SqliteBookkeeping`。

---

## RAG

### 文档与分块

一个 `Document` 是一篇源文档（加载器的输出）。它被切成若干 `Chunk` 对象，共享该文档的 `doc_id`，正是这个 id 让重新灌入时可以按文档删掉旧分块、upsert 新分块。

`Document` 关键字段：`content`、`doc_id`（省略时自动生成）、`title`、`source`、`format`（`md` / `json` / `pdf` 等，用于挑选切分器）和 `metadata`。`Chunk` 关键字段：`content`、`chunk_id`、`doc_id`、`heading_path`（用于 Markdown）、`index`（在文档内的位置）和 `metadata`。

### 分块切分

`split_document` 根据 `doc.format` 挑选切分器并返回一个分块列表。Markdown 按标题层级切分（保留标题路径），结构化数据（`json` / `jsonl` / `csv`）按记录切分，其余一切按 token 数带重叠切分。一篇能装进单个分块的短文档永远不会被切分。

```python
from agentmaker import Document, split_document

doc = Document(content="# Title\n\nSome body text.", format="md")
chunks = split_document(doc, chunk_tokens=512, overlap_tokens=64)
```

`ChunkingConfig(chunk_tokens=512, overlap_tokens=64)` 保存这些默认值；`overlap_tokens` 必须满足 `0 <= overlap < chunk_tokens`。

### 灌入流水线

`IngestionPipeline` 把加载器和切分器接到检索地基与事实源存储上。灌入一篇文档做两件事：把完整分块存进 `SourceStore`，并把它们推入检索索引使之可被检索到。两侧都以同一个 `chunk_id` 为键。

```python
report = pipeline.ingest_file("handbook.md")                 # read a file, chunk, ingest
report = pipeline.ingest_text("some text", source="note.txt")  # ingest raw text
```

两者都返回一个 `IngestReport(doc_id, chunks, skipped)`。灌入按 `doc_id` 去重：当同一篇文档原封不动地被再次灌入时，内容指纹会短路整趟运行（`skipped=True`，不切分、不嵌入）；而一篇有改动的文档会原子地替换其旧版本，因此失败时不会丢失任何东西。`ingest_file` 从文件的绝对路径推导出一个稳定的 `doc_id`，因此重新灌入同一个文件会命中这个短路。流水线还提供 `delete_document`、`rebuild_index`（从事实源完全重新推送索引，用于后端或模型迁移）、`verify`（报告两侧的分歧但不修复）和 `stats`。异步孪生版本（`aingest_file`、`aingest_text`、`adelete_document`）也可用。

`RagRetriever` 与 `IngestionPipeline` 还都提供一个 `from_config` 构造入口，它接受一个 `RagConfig`（聚合的 RAG 配置）以及 embedder 和 LLM。要组装一条流水线和一个检索器并让它们共享同一个后端，先构建检索器再复用它：

```python
rag = RagRetriever.from_config(config, embedder=emb, llm=llm)
pipeline = IngestionPipeline.from_config(
    config, retriever=rag.retriever, source_store=rag.source_store)
```

### 上下文化检索

一个分块被切出来后可能丢失它周围的上下文（一个「Lodging」分块的正文只写着「capped at 500 per night」，而「lodging」这个词被孤零零地留在标题里）。`Contextualizer` 在建索引前把那份上下文补回去。关键细节：增强后的文本仅用于检索。事实源仍然存放干净的原始分块，这也正是命中最终返回的东西。

两个电池实现：

- `HeadingContextualizer` 在分块前面拼上它的标题路径。零 LLM 开销，修复「丢失标题词」这种情况。
- `LLMContextualizer` 让 LLM 为每个分块写一句上下文（更强，但每个分块要一次 LLM 调用）。

```python
from agentmaker import IngestionPipeline, HeadingContextualizer

pipeline = IngestionPipeline(
    retriever=retriever, source_store=source_store,
    contextualizer=HeadingContextualizer())
```

### 检索与提问

`RagRetriever` 把分块读回来并生成有据可查的答案。

`retrieve` 返回最相关的分块（`RetrievalResult` 对象），命中后回到事实源存储去补全 `heading_path`、`doc_id` 这些完整字段：

```python
for hit in rag.retrieve("how much for meals?", top_k=2, filters=None):
    print(hit.content, hit.metadata["heading_path"])
```

`ask` 是非流式、异步的问答入口。它先检索，把分块拼成一段带编号的上下文，再让 LLM 只用这些材料作答、标注来源，并在答案缺失时明说自己不知道。它返回一个 `AskResult`：

```python
import asyncio
from agentmaker import RagRetriever
from agentmaker.testing import ScriptedLLM

rag = RagRetriever(retriever, source_store, ScriptedLLM(["The meal allowance is 80."]))

async def main():
    result = await rag.ask("how much can I spend on meals?", top_k=2)
    print(result.answer)                       # the grounded answer text
    for src in result.sources:                 # list[SourceRef]
        print(src.n, src.heading_path, src.doc_id)

asyncio.run(main())
```

`AskResult` 带有 `answer`（答案文本）和 `sources`（一个 `SourceRef` 列表，每个含 `n`、`content`、`heading_path` 和 `doc_id`；`n` 与模型看到的 `[n]` 引用标记相对应）。没有命中时 `sources` 为 `[]`，你可以据此分支处理。`ask_stream` 逐段产出答案文本，供 `async for` 消费。防幻觉的系统提示词是框架默认；向 `RagRetriever` 传入 `system_prompt=` 可覆盖其措辞。

### 查询变换

用户的措辞往往和文档的措辞对不上。`QueryTransformer` 在检索前改写或扩展查询，对每个变体各自检索，再用 RRF 融合结果。两种变换默认都关闭（每种都会给每次检索加一次 LLM 调用），在构造时传入 `query_transformer=` 即可启用：

- `MultiQueryExpander`（MQE，多查询扩展）让 LLM 把问题改写成几种不同的表述。
- `HyDETransformer`（HyDE，hypothetical document embeddings，假想文档嵌入）让 LLM 起草一个假想的答案并用它来检索，因为形似答案的文本比问题本身更精确地匹配答案分块。

```python
from agentmaker import RagRetriever, MultiQueryExpander

rag = RagRetriever(retriever, source_store, llm,
                   query_transformer=MultiQueryExpander(llm, n=3))
```

当 LLM 失败时，变换会回退到原始查询而不是抛出异常。

### 邻窗扩展

小分块检索起来精确，但可能太单薄，不足以让模型答好。「small-to-big」（先小后大）模式用小分块检索，然后把每个命中扩展成更完整的上下文，再交给 LLM。`NeighborWindowExpander` 把每个命中与紧邻它前后的分块（同一文档，按 `index` 排序）合并，并做去重，让重叠的窗口不被重复计入。它默认关闭；用 `expander=` 启用：

```python
from agentmaker import RagRetriever, NeighborWindowExpander

rag = RagRetriever(retriever, source_store, llm,
                   expander=NeighborWindowExpander(window=1))
```

`window=1` 意为「命中分块加上其两侧各一个分块」。想要不同的策略，就提供你自己的 `ChunkExpander` 子类（例如父分块合并）。

### RAGTool：agentic RAG

`RAGTool` 把灌入和检索包装成一个 [工具](tools.md)，让 Agent 能管理一个知识库并从中作答。它暴露 `add_text`、`add_document`、`search`、`ask` 和 `stats` 这几个动作。

```python
from agentmaker import RAGTool

rag_tool = RAGTool(pipeline, rag, top_k=5)
```

内置了两处安全细节。`add_document` 会从磁盘读文件，属于高风险动作，因此只有它的 `RAGTool.needs_confirmation` 返回 `True`；随后统一的确认关卡会在它运行前要求人工批准（见 [护栏与 HITL](guardrails-and-hitl.md)，其中 HITL 即 human-in-the-loop，人在回路）。而且因为 `search` / `ask` 返回的是来自外部知识库的文本，`RAGTool` 设置了 `external_content = True`，于是执行环境会在把这段文本喂回模型前用防注入定界符把它包起来。传入 `filter_fields=("doc_id", "tag")` 会把这些元数据字段变成模型可以填写的可选工具参数（自然语言自查询）；对应的列必须在构建索引时已通过 `metadata_columns=` 声明。

---

## 下一步去哪

- [记忆](memory.md) 建立在同一套检索地基上，并共享它的 `Scope` 隔离。
- [上下文工程](context-engineering.md) 讲述检索到的分块如何被预算进 prompt，以及 MMR 在哪里复用每个 `RetrievalResult` 上携带的向量。
- [工具](tools.md) 和 [护栏与 HITL](guardrails-and-hitl.md) 解释了 `RAGTool` 所依赖的确认关卡和外部内容处理。
- 在 [API 参考](../reference.md) 里查阅精确的签名。
