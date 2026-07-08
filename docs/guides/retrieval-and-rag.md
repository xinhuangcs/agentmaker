# Retrieval & RAG

agentmaker ships two layers for finding relevant text and grounding an LLM in it. The lower layer is a **retrieval foundation**: a hybrid retriever that runs dense (vector) search and sparse (keyword) search side by side, fuses the two rankings, and optionally reranks the result. The upper layer is **RAG** (Retrieval-Augmented Generation: retrieve supporting passages, then let the model answer using only those passages): it reads files into documents, splits them into chunks, ingests them, and answers questions with citations. The same foundation is shared by [Memory](memory.md), so both subsystems get the same search and the same data isolation.

Reach for the retrieval foundation when you want raw search over your own text; reach for RAG when you want a document knowledge base with grounded, sourced answers.

## The two layers at a glance

| Layer | Package | You get |
| --- | --- | --- |
| Retrieval foundation | `agentmaker.retrieval` | `HybridRetriever`, the abstract ports (`Embedder`, `VectorStore`, `KeywordIndex`, `Reranker`, `FusionStrategy`) and their batteries, `reciprocal_rank_fusion`, `Scope`, `MetadataFilter`, `RetrievalResult`, `IndexSync` |
| RAG | `agentmaker.rag` | `Document` / `Chunk`, `split_document`, `IngestionPipeline`, `RagRetriever`, `Contextualizer`, `MultiQueryExpander` / `HyDETransformer`, `NeighborWindowExpander`, `RAGTool`, `AskResult` / `SourceRef` |

## Quickstart: ingest and retrieve

This is [`examples/05_rag.py`](https://github.com/xinhuangcs/agentmaker/blob/main/examples/05_rag.py), verbatim. It runs with no API key and no network: `FakeEmbedder` stands in for a real embedding model and the local SQLite backend holds the index.

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

In production, swap `FakeEmbedder()` for `OpenAIEmbedder()` and `ScriptedLLM([])` for a real `LLMClient` (see [LLM clients](llm-clients.md)). `build_sqlite_hybrid` builds the whole hybrid retriever (vector store plus keyword index sharing one SQLite connection); `SourceStore` is the RAG source-of-truth store that keeps the full chunk text.

---

## The retrieval foundation

### Ports and batteries

The foundation follows a ports-and-adapters design: five abstract base classes define the interfaces, and swappable batteries fill them in. Replacing a backend means writing one subclass; `HybridRetriever` and everything above it stay untouched.

| Port (abstract base) | Job | Default battery |
| --- | --- | --- |
| `Embedder` | Turn text into vectors | `OpenAIEmbedder` |
| `VectorStore` | Store vectors, run nearest-neighbor (dense) search | `SqliteVecStore` |
| `KeywordIndex` | Keyword search, ranked by BM25 (a classic term-frequency relevance score) | `Fts5KeywordIndex` |
| `Reranker` | Re-order candidates with a cross-encoder (a model that scores a query and passage together, more precise than vectors) | `CohereReranker` |
| `FusionStrategy` | Merge several ranked lists into one | `RRFFusion` |

All five ports and all five batteries are exported from the package top level, for example `from agentmaker import HybridRetriever, OpenAIEmbedder, SqliteVecStore`.

### HybridRetriever

`HybridRetriever` is the storage-agnostic orchestrator. Construct it from an embedder, a vector store, and a keyword index, with an optional reranker and fusion strategy:

```python
from agentmaker import HybridRetriever
```

```python
def __init__(self, embedder, vector_store, keyword_index,
             reranker=None, *, config=None, fusion=None)
```

`search` is the read path. Each path (vector and keyword) fetches `candidate_pool` items, the two lists are fused (RRF by default), and if a reranker is present the fused pool is refined down to `top_k`:

```python
def search(self, query, *, top_k=None, candidate_pool=None,
           scope=None, all_scopes=False, filters=None) -> List[RetrievalResult]
```

When `top_k` or `candidate_pool` is omitted, the values from `config` are used. `search_many` retrieves for several queries at once, embedding them all in a single batch to save round trips. Every method has an async twin (`asearch`, `asearch_many`, `aadd`, and so on). The base `HybridRetriever` runs the vector and keyword paths concurrently; the default SQLite backend from `build_sqlite_hybrid` shares one connection under a single lock, so its async twins run the search in one thread (splitting the two paths would only contend on that lock).

Writes go through `add` (an upsert: writing the same id again overwrites it), `replace` (swap a document's old chunks for new ones), and `delete`. In most cases you will not call these directly; the RAG `IngestionPipeline` and Memory drive them for you.

!!! note "Prefer `build_sqlite_hybrid` for local use"
    The plain `HybridRetriever` treats its two indexes as separate connections, so `add` can only best-effort compensate on partial failure. `build_sqlite_hybrid` gives the vector store and keyword index one shared SQLite connection, making writes across both indexes atomic in a single transaction. It also fingerprint-checks the embedding model on open, so swapping in a mismatched model fails loudly instead of silently mixing incomparable vectors.

### Reciprocal Rank Fusion (RRF)

RRF is the default way the two retrieval paths are combined. It looks only at each item's rank, not its raw score, which sidesteps the problem that vector distances and BM25 scores live on incomparable scales, and it needs no tuning. An item ranked `r` (starting at 1) in a list contributes `1 / (k + r)` points; scores for the same id across lists are summed, so an item hit by both paths naturally rises to the top. The smoothing constant `k` defaults to 60.

You can call the function directly on any set of ranked lists:

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

Fusion aligns by `id` (an empty id degrades to aligning by content). To weight the two paths instead of using plain RRF, implement `FusionStrategy` and inject it via `HybridRetriever(fusion=...)`; `RRFFusion` is just the default battery wrapping `reciprocal_rank_fusion`.

### RetrievalResult

Every query in the foundation (and in RAG and Memory above it) returns a list of `RetrievalResult`, so the upper layers face a single shape regardless of where the hit came from:

| Field | Meaning |
| --- | --- |
| `content` | The hit text |
| `score` | Relevance, by convention higher is more relevant; only meaningful for ordering, not comparable across backends |
| `source` | Origin label, such as `"rag"` or a document name |
| `id` | Unique id within the source (empty string if absent) |
| `embedding` | The item's vector, carried back when available so context engineering's MMR (Maximal Marginal Relevance, a redundancy-aware selection step) can reuse it |
| `metadata` | Attached fields (raw distance, heading path, and so on) |

### Scope: isolation and filtering

`Scope` is a frozen dataclass that labels every piece of data by ownership across five dimensions: `base`, `user`, `agent`, `session`, and `app`. One shared foundation can hold many tenants' data without cross-contamination.

```python
from agentmaker import Scope

Scope(base="rag", user="alice")
```

Filtering only ever narrows: search adds a `WHERE` only for the dimensions you set, and leaves the rest unrestricted. `Scope(user="alice")` returns all of alice's data regardless of agent or session. By convention each subsystem sets `base` (RAG uses `Scope(base="rag")`, Memory uses `Scope(base="memory")`), which keeps their data isolated on the shared backend.

A fully empty `Scope()` restricts nothing, so a bare delete or search would hit the whole store. The `require_explicit_scope` guard rejects that unless you pass `all_scopes=True`, which turns an accidental global operation into an up-front error.

### Metadata filters

`MetadataFilter` narrows candidates by structured fields before similarity is computed (pre-filtering). Conditions are AND'd, and the only operators are `eq` (equality) and `in` (one of several):

```python
from agentmaker import MetadataFilter

MetadataFilter("doc_id", "abc123")                    # doc_id = 'abc123'
MetadataFilter("tag", ["faq", "policy"], op="in")     # tag in ('faq', 'policy')
```

Filterable fields must be declared when you build the index, via `build_sqlite_hybrid(..., metadata_columns=("doc_id", "tag"))`. Filtering a field that was never declared fails loudly rather than silently returning zero hits.

### RetrievalConfig

`RetrievalConfig` holds the tunable knobs, validated at construction:

```python
from agentmaker import RetrievalConfig

RetrievalConfig(top_k=5, candidate_pool=20, rrf_k=60)   # the defaults
```

`candidate_pool` (how many items each path fetches into fusion and rerank) must be at least `top_k`. Pass the config to `HybridRetriever(config=...)`, or to `build_sqlite_hybrid(config=...)`.

### Keeping the index in sync

Both RAG and Memory keep a source-of-truth store plus a derived retrieval index that must converge to it. That cross-cutting concern is collapsed into one pluggable seam, `IndexSync`, so each subsystem's write path just calls it. The default `SyncIndexSync` writes through synchronously and records bookkeeping (a content fingerprint for idempotent skipping, plus a pending set for self-healing). Bookkeeping storage is itself swappable: `InMemoryBookkeeping` is the zero-dependency default (rebuildable from the source of truth after a restart), and `SqliteBookkeeping` persists it across processes. For a fully async or distributed setup, implement your own `IndexSync` without touching any subsystem write path.

You rarely construct these directly; the pipeline's `from_config` installs `SqliteBookkeeping` for you.

---

## RAG

### Documents and chunks

A `Document` is one source document (the output of a loader). It is split into `Chunk` objects, all sharing the document's `doc_id`, which is what lets a re-ingest delete the old chunks and upsert the new ones by document.

Key `Document` fields: `content`, `doc_id` (auto-generated if omitted), `title`, `source`, `format` (`md` / `json` / `pdf` and so on, used to pick the splitter), and `metadata`. Key `Chunk` fields: `content`, `chunk_id`, `doc_id`, `heading_path` (for Markdown), `index` (position within the document), and `metadata`.

### Chunking

`split_document` picks a splitter from `doc.format` and returns a list of chunks. Markdown is split by heading level (preserving the heading path), structured data (`json` / `jsonl` / `csv`) is split by record, and everything else is split by token count with overlap. A short document that fits within one chunk is never split.

```python
from agentmaker import Document, split_document

doc = Document(content="# Title\n\nSome body text.", format="md")
chunks = split_document(doc, chunk_tokens=512, overlap_tokens=64)
```

`ChunkingConfig(chunk_tokens=512, overlap_tokens=64)` holds these defaults; `overlap_tokens` must satisfy `0 <= overlap < chunk_tokens`.

### The ingestion pipeline

`IngestionPipeline` wires the loader and splitter to the retrieval foundation and the source-of-truth store. Ingesting a document does two things: store the full chunks in `SourceStore`, and push them into the retrieval index so they become searchable. Both sides are keyed by the same `chunk_id`.

```python
report = pipeline.ingest_file("handbook.md")                 # read a file, chunk, ingest
report = pipeline.ingest_text("some text", source="note.txt")  # ingest raw text
```

Both return an `IngestReport(doc_id, chunks, skipped)`. Ingestion deduplicates by `doc_id`: when the same document is re-ingested unchanged, a content fingerprint short-circuits the whole run (`skipped=True`, no chunking, no embedding) and a changed document atomically replaces its old version, so nothing is lost on failure. `ingest_file` derives a stable `doc_id` from the file's absolute path, so re-ingesting the same file hits that short-circuit. The pipeline also offers `delete_document`, `rebuild_index` (fully re-push the index from the source of truth, for backend or model migrations), `verify` (report divergence between the two without repairing), and `stats`. Async twins (`aingest_file`, `aingest_text`, `adelete_document`) are available.

Both `RagRetriever` and `IngestionPipeline` also offer a `from_config` constructor that takes a `RagConfig` (the aggregated RAG configuration) alongside the embedder and LLM. To assemble a pipeline and retriever that share one backend, build the retriever first and reuse it:

```python
rag = RagRetriever.from_config(config, embedder=emb, llm=llm)
pipeline = IngestionPipeline.from_config(
    config, retriever=rag.retriever, source_store=rag.source_store)
```

### Contextual Retrieval

When a chunk is split out it can lose the context around it (a "Lodging" chunk whose body reads only "capped at 500 per night", with the word "lodging" stranded in a heading). A `Contextualizer` adds that context back before indexing. The important detail: the enhanced text is used for retrieval only. The source of truth still stores the clean original chunk, which is what a hit ultimately returns.

Two batteries:

- `HeadingContextualizer` prepends the chunk's heading path. Zero LLM cost, fixes the "lost heading word" case.
- `LLMContextualizer` asks an LLM to write one sentence of context per chunk (stronger, but one LLM call per chunk).

```python
from agentmaker import IngestionPipeline, HeadingContextualizer

pipeline = IngestionPipeline(
    retriever=retriever, source_store=source_store,
    contextualizer=HeadingContextualizer())
```

### Retrieving and asking

`RagRetriever` reads chunks back and generates grounded answers.

`retrieve` returns the most relevant chunks as `RetrievalResult` objects, going back to the source-of-truth store after a hit to fill in complete fields like `heading_path` and `doc_id`:

```python
for hit in rag.retrieve("how much for meals?", top_k=2, filters=None):
    print(hit.content, hit.metadata["heading_path"])
```

`ask` is the non-streaming, async question-answering entry point. It retrieves, assembles the chunks into a numbered context, and asks the LLM to answer using only that material, cite its sources, and say it does not know when the answer is absent. It returns an `AskResult`:

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

`AskResult` carries `answer` (the text) and `sources` (a list of `SourceRef`, each with `n`, `content`, `heading_path`, and `doc_id`; the `n` matches the `[n]` citation markers the model sees). `sources` is `[]` when there were no hits, which you can branch on. `ask_stream` yields the answer text piece by piece for `async for` consumption. The anti-hallucination system prompt is a framework default; pass `system_prompt=` to `RagRetriever` to override the wording.

### Query transforms

The user's wording often does not match the document's wording. A `QueryTransformer` rewrites or expands the query before retrieval, retrieves each variant, and fuses the results with RRF. Both transforms are off by default (each adds one LLM call per retrieval) and are enabled by passing `query_transformer=` at construction:

- `MultiQueryExpander` (MQE, multi-query expansion) has the LLM rewrite the question into several phrasings.
- `HyDETransformer` (HyDE, hypothetical document embeddings) has the LLM draft a hypothetical answer and searches with that, since answer-shaped text matches answer chunks more precisely than the question does.

```python
from agentmaker import RagRetriever, MultiQueryExpander

rag = RagRetriever(retriever, source_store, llm,
                   query_transformer=MultiQueryExpander(llm, n=3))
```

On LLM failure a transform falls back to the original query rather than raising.

### Neighbor-window expansion

Small chunks retrieve precisely but can be too thin for the model to answer well. The "small-to-big" pattern retrieves with small chunks, then expands each hit into a fuller context before handing it to the LLM. `NeighborWindowExpander` merges each hit with the chunks immediately before and after it (same document, ordered by `index`), deduplicating so overlapping windows are not counted twice. It is off by default; enable it with `expander=`:

```python
from agentmaker import RagRetriever, NeighborWindowExpander

rag = RagRetriever(retriever, source_store, llm,
                   expander=NeighborWindowExpander(window=1))
```

`window=1` means "the hit chunk plus one chunk on each side". Provide your own `ChunkExpander` subclass for a different strategy (for example parent-chunk merging).

### RAGTool: agentic RAG

`RAGTool` wraps ingestion and retrieval as a [Tool](tools.md) so an Agent can manage a knowledge base and answer from it. It exposes the actions `add_text`, `add_document`, `search`, `ask`, and `stats`.

```python
from agentmaker import RAGTool

rag_tool = RAGTool(pipeline, rag, top_k=5)
```

Two safety details are built in. `add_document` reads a file from disk, a high-risk action, so `RAGTool.needs_confirmation` returns `True` only for it; the unified confirmation gate then requires a human approval before it runs (see [Guardrails & HITL](guardrails-and-hitl.md), where HITL is human-in-the-loop). And because `search` / `ask` return text from an external knowledge base, `RAGTool` sets `external_content = True`, so the harness wraps that text in anti-injection delimiters before feeding it back to the model. Passing `filter_fields=("doc_id", "tag")` turns those metadata fields into optional tool parameters the model can fill (natural-language self-query); the matching columns must have been declared via `metadata_columns=` when the index was built.

---

## Where to go next

- [Memory](memory.md) is built on the same retrieval foundation and shares its `Scope` isolation.
- [Context engineering](context-engineering.md) covers how retrieved chunks are budgeted into the prompt and where MMR reuses the vectors carried on each `RetrievalResult`.
- [Tools](tools.md) and [Guardrails & HITL](guardrails-and-hitl.md) explain the confirmation gate and external-content handling that `RAGTool` relies on.
- Look up exact signatures in the [API Reference](../reference/core.md).
