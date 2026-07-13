"""agentmaker.rag.retriever: RAG retrieval and question answering.

retrieve: fetch the most relevant chunks from the retrieval backend (after a hit,
go back to the source-of-truth store to fill in complete information such as heading_path).
ask / ask_stream: assemble chunks into a numbered context plus the question and hand them
to the LLM, generating an answer that only relies on the material, cites its sources, and says
"I don't know" when the answer is absent (anti-hallucination, traceable). ask returns a
non-streaming AskResult(answer, sources); ask_stream streams the answer text piece by piece.
Optional query expansion (disabled by default): pass query_transformer=
(MultiQueryExpander / HyDETransformer) at construction time to rewrite / expand the query into
several queries before retrieval, retrieve each one, and fuse with RRF, addressing the mismatch
between question wording and document wording.
Optional post-retrieval expansion (disabled by default): pass expander=
(ChunkExpander / NeighborWindowExpander, small-to-big: small chunks retrieve precisely, and
after a hit the context is expanded into a neighbor window to give the LLM fuller context).
"""

import asyncio
from abc import ABC, abstractmethod
from time import perf_counter
from typing import TYPE_CHECKING, List, Optional

from ..core.exceptions import RetrievalError, RunCancelled, RunLimitExceeded
from ..core.llm_clients import LLMClient
from ..prompts import DEFAULT_PROMPTS
from ..core.aio import run_sync
from ..runtime.execution.run_context import (check_deadline, check_limits, correlation,
                                             enforce_token_limit_after_llm, governed_chat,
                                             record_llm)
from ..retrieval.hybrid import HybridRetriever, reciprocal_rank_fusion
from ..retrieval.index_sync import IndexSync, SyncIndexSync
from ..retrieval.scope import Scope, canonical_scope
from ..retrieval.types import MetadataFilter, RetrievalConfig, RetrievalResult
from .source_store import SourceStore
from .types import AskResult, RagConfig, SourceRef
from ..core.trace_events import (EVENT_LLM_CALL, EVENT_RAG_QUERY_TRANSFORM_FAILED,
                                 EVENT_RAG_RETRIEVE)

if TYPE_CHECKING:
    from ..config import AgentmakerConfig
    from ..prompts import PromptRegistry
    from ..retrieval.base import Embedder, Reranker
    from ..runtime.observability import Tracer


# Default anti-hallucination system prompt for ask; the framework only provides the mechanism,
# while wording / tone / extra rules belong to the app: pass system_prompt= when constructing
# RagRetriever to override the whole thing.
DEFAULT_ASK_PROMPT = DEFAULT_PROMPTS.text("rag.ask")

class QueryTransformer(ABC):
    """Pre-retrieval query expansion (disabled by default): transform the user query into "queries that search better".

    transform returns one or more retrieval queries; RagRetriever retrieves each one and fuses
    them by rank with RRF. See MultiQueryExpander (MQE) / HyDETransformer (HyDE) for
    implementations; both call the LLM and add one extra LLM cost per retrieval, hence opt-in.
    """

    @abstractmethod
    def transform(self, query: str) -> List[str]:
        """Transform the original query into one or more retrieval queries (at least one).

        On LLM failure it should fall back to [query] itself and not raise; however governance
        exceptions (RunLimitExceeded / RunCancelled) must propagate and must not be swallowed,
        otherwise RunPolicy limits / cancellation would be bypassed on the retrieval side path.
        """


DEFAULT_MQE_PROMPT = DEFAULT_PROMPTS.text("rag.mqe")


class MultiQueryExpander(QueryTransformer):
    """MQE (multi-query expansion): have the LLM rewrite one question into several phrasings, search each, and fuse with RRF. Addresses the mismatch between the user's wording and the document's wording."""

    def __init__(self, llm: LLMClient, *, n: int = 3, include_original: bool = True,
                 expand_prompt: Optional[str] = None, prompts=None, tracer=None):
        """
        Args:
            llm: The LLM that generates the rewrites (a cheap model is recommended).
            n: How many rewrites to expect (>= 1; take whatever the LLM gives, up to n).
            include_original: Whether to also retrieve the original query (default True: the
                original query preserves the keyword-based recall path).
            expand_prompt: System prompt for multi-query rewriting; if omitted the framework
                default DEFAULT_MQE_PROMPT is used. Pass your own to switch languages.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.llm = llm
        self.tracer = tracer   # Optional tracer: routes rewrite calls into trace and RunPolicy governance (governed_chat).
        self.n = n
        self.include_original = include_original
        base = prompts or DEFAULT_PROMPTS                          # prompts: optional prompt registry; expand_prompt is a local shortcut override for rag.mqe.
        self.prompts = base.with_overrides({"rag.mqe": expand_prompt}) if expand_prompt else base
        self.expand_prompt = self.prompts.text("rag.mqe")

    def transform(self, query: str) -> List[str]:
        """Generate several rewritten queries (plus the optional original); fall back to [query] on LLM failure.

        An LLM sub-step during retrieval: driven via aio.run_sync over async governed_chat (keeps
        transform's synchronous interface, so retrieve's whole chain does not become async:
        retrieval is a synchronous IO path, consistent with the synchronous embedder).
        """
        user = self.prompts.render("rag.mqe_user", n=self.n, query=query)
        try:
            text = run_sync(governed_chat(self.llm, [{"role": "system", "content": self.expand_prompt},
                                                     {"role": "user", "content": user}],
                                          tracer=self.tracer, origin="rag.mqe")).content
            # Strip leading list markers / whitespace before filtering out empties (so a line like "- " does not become an empty query),
            # and cap at n items (to keep a chatty LLM from blowing up the fan-out).
            variants = [s for ln in text.splitlines() if (s := ln.strip(" -·•*–—\t"))][:self.n]
        except (RunLimitExceeded, RunCancelled):
            raise                                       # Governance control-flow exceptions pass through; do not swallow them as "the LLM died".
        except Exception:  # noqa: BLE001  An LLM failure is not fatal: fall back to the original query.
            if self.tracer is not None:
                self.tracer.emit({"type": EVENT_RAG_QUERY_TRANSFORM_FAILED, "origin": "rag.mqe", **correlation()})
            variants = []
        queries = ([query] if self.include_original else []) + variants
        return queries or [query]


DEFAULT_HYDE_PROMPT = DEFAULT_PROMPTS.text("rag.hyde")


class HyDETransformer(QueryTransformer):
    """HyDE (hypothetical document embeddings): have the LLM first write a "hypothetical answer" and search with it. Matching an answer chunk with "answer wording" is more precise than matching with the question."""

    def __init__(self, llm: LLMClient, *, include_original: bool = True, hyde_prompt: Optional[str] = None,
                 prompts=None, tracer=None):
        """
        Args:
            llm: The LLM that generates the hypothetical document.
            include_original: Whether to also retrieve with the original query (default True; in
                hybrid retrieval the original query preserves the keyword-based recall path).
            hyde_prompt: System prompt for generating the hypothetical document; if omitted the
                framework default DEFAULT_HYDE_PROMPT is used. Pass your own to switch languages.
        """
        self.llm = llm
        self.tracer = tracer   # Optional tracer: routes hypothetical-document generation into trace and RunPolicy governance (governed_chat).
        self.include_original = include_original
        base = prompts or DEFAULT_PROMPTS                          # prompts: optional prompt registry; hyde_prompt is a local shortcut override for rag.hyde.
        self.prompts = base.with_overrides({"rag.hyde": hyde_prompt}) if hyde_prompt else base
        self.hyde_prompt = self.prompts.text("rag.hyde")

    def transform(self, query: str) -> List[str]:
        """Generate a hypothetical answer text as the retrieval query (plus the optional original); fall back to [query] on LLM failure.

        The LLM sub-step during retrieval is driven via aio.run_sync over async governed_chat
        (same as MQE, keeping the synchronous interface).
        """
        try:
            hypo = run_sync(governed_chat(self.llm, [{"role": "system", "content": self.hyde_prompt},
                                                     {"role": "user", "content": query}],
                                          tracer=self.tracer, origin="rag.hyde")).content.strip()
        except (RunLimitExceeded, RunCancelled):
            raise                                       # Governance control-flow exceptions pass through.
        except Exception:  # noqa: BLE001
            if self.tracer is not None:
                self.tracer.emit({"type": EVENT_RAG_QUERY_TRANSFORM_FAILED, "origin": "rag.hyde", **correlation()})
            hypo = ""
        queries = ([query] if self.include_original else []) + ([hypo] if hypo else [])
        return queries or [query]


# Post-retrieval chunk expansion (small-to-big): small chunks retrieve precisely, and after a
# hit they are expanded into a larger context.
# The inherent tension of chunking: small chunks retrieve precisely (semantically focused),
# large chunks give full context (better for the model to answer). The industry-standard fix is
# small-to-big (same idea as LangChain ParentDocumentRetriever / LlamaIndex AutoMergingRetriever):
# retrieve with small chunks, then after a hit expand the context into a neighbor window or the
# parent chunk before handing it to the LLM. Inject via RagRetriever(expander=...) to enable;
# disabled by default (zero behavior change). The parent-chunk / auto-merging variant is left as
# another ChunkExpander implementation, without introducing dual-index complexity.

class ChunkExpander(ABC):
    """Abstract base for post-retrieval expansion: expand a hit's small chunk into a fuller context."""

    @abstractmethod
    def expand(self, results: List[RetrievalResult], *, source_store: SourceStore,
               scope: Optional[Scope] = None) -> List[RetrievalResult]:
        """Expand a batch of hits and return the expanded results (preserving the relevance-ordered input order).

        Args:
            results: Retrieval hits from RagRetriever (metadata contains doc_id / index).
            source_store: RAG source-of-truth store (SourceStore, fetches neighbor chunks by
                doc_id + idx).
            scope: Retrieval scope (same scope as the hit, to avoid fetching chunks from a
                sibling scope).
        """


class NeighborWindowExpander(ChunkExpander):
    """Default battery: neighbor-window expansion. A hit chunk together with the window chunks before/after it (same document, ordered by idx) is merged into one result.

    Deduplication rule: each (doc_id, idx) is used at most once (assigned by relevance from
    highest to lowest); if a hit's window is already fully covered by an earlier, higher-ranked
    hit, that hit is skipped (to avoid duplicate content eating the budget). Hits without doc_id /
    index (e.g. non-RAG sources) are kept as-is.
    """

    def __init__(self, window: int = 1):
        """window: how many chunks to take before / after (default 1, i.e. "hit chunk +/- 1")."""
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window

    def expand(self, results: List[RetrievalResult], *, source_store: SourceStore,
               scope: Optional[Scope] = None) -> List[RetrievalResult]:
        """For each hit, take neighbor chunks [idx-window, idx+window] and merge by idx; deduplicate across hits by (doc_id, idx)."""
        used = set()                                    # (doc_id, idx) pairs already assigned.
        out = []
        for r in results:
            doc_id = r.metadata.get("doc_id")
            idx = r.metadata.get("index")
            if not doc_id or not isinstance(idx, int):   # Non-RAG chunk (no locating info): keep as-is.
                out.append(r)
                continue
            lo, hi = idx - self.window, idx + self.window
            neighbors = source_store.get_doc_chunks(doc_id, index_range=(lo, hi), scope=scope)
            fresh = [c for c in neighbors if (doc_id, c.index) not in used]
            if not fresh:                                # Window already fully covered by a higher-scoring hit: skip to avoid duplicate content.
                continue
            used.update((doc_id, c.index) for c in fresh)
            merged = "\n".join(c.content for c in fresh)   # fresh comes from an ordered query, so it is naturally ascending by idx.
            out.append(RetrievalResult(
                content=merged, score=r.score, source=r.source, id=r.id, embedding=r.embedding,
                metadata={**r.metadata, "expanded": f"{fresh[0].index}-{fresh[-1].index}"}))
        return out


class RagRetriever:
    """RAG retriever: retrieve document chunks and generate answers grounded in those chunks."""

    def __init__(self, retriever: HybridRetriever, source_store: SourceStore,
                 llm: Optional[LLMClient], *, scope: Optional[Scope] = None, system_prompt: Optional[str] = None,
                 query_transformer: Optional[QueryTransformer] = None,
                 config: Optional[RetrievalConfig] = None, rag_config: Optional[RagConfig] = None,
                 prompts: "Optional[PromptRegistry]" = None,
                 expander: Optional[ChunkExpander] = None, tracer: "Optional[Tracer]" = None,
                 index_sync: Optional[IndexSync] = None):
        """
        Args:
            retriever: Retrieval backend (vector + keyword + RRF + optional rerank).
            source_store: RAG source-of-truth store; fetch complete chunks from it after a hit.
            llm: The LLM used to generate answers (a cheap model such as deepseek is recommended).
            scope: Retrieval scope, defaults to Scope(base="rag"), isolated from memory.
            system_prompt: Anti-hallucination system prompt for ask / ask_stream; if omitted the
                framework default (DEFAULT_ASK_PROMPT) is used. The framework only provides the
                mechanism; the specific wording / tone / business rules belong to the app: pass
                your own to customize.
            query_transformer: Optional query expander (MQE / HyDE); None by default (disabled),
                retrieving with the original query directly. When provided, the query is rewritten
                / expanded into several queries before retrieval, each is retrieved, and results
                are fused with RRF. Adds one extra LLM cost per retrieval, so enable on demand.
            expander: Optional post-retrieval chunk expander (ChunkExpander, see above); None by
                default (disabled). Pass NeighborWindowExpander(window=N) for small-to-big: small
                chunks retrieve precisely, and after a hit the context is expanded into a neighbor
                window to give the LLM fuller context.
            tracer: Optional tracer (duck-typed emit); once attached, retrieve emits a rag_retrieve
                event, and the LLM calls of ask and query expansion enter trace and RunPolicy
                governance. Zero overhead when not attached.
            index_sync: Index synchronization seam used for orphan cleanup. Managers sharing a
                retriever also share the default in-process bookkeeping.
        """
        self.retriever = retriever
        self.source_store = source_store
        self.llm = llm
        self.scope = canonical_scope(scope, "rag", "RagRetriever construction")
        self.expander = expander
        self.tracer = tracer
        self._sync = index_sync if index_sync is not None else SyncIndexSync(retriever, tracer=tracer)
        self._owns_source_store = False
        self._owns_retriever = False
        self._owns_sync = index_sync is None
        self._closed = False
        base = prompts or DEFAULT_PROMPTS                          # prompts: optional prompt registry; system_prompt is a local shortcut override for rag.ask.
        self.prompts = base.with_overrides({"rag.ask": system_prompt}) if system_prompt else base
        self.system_prompt = self.prompts.text("rag.ask")
        self.query_transformer = query_transformer
        self.cfg = config or RetrievalConfig()        # top_k / candidate_pool / rrf_k
        self.rag_cfg = rag_config or RagConfig()      # mq_pool_factor / mq_max_queries

    def _scope(self, scope: Optional[Scope], action: str) -> Scope:
        """Resolve a per-call scope while preserving the RAG subsystem boundary."""
        return self.scope if scope is None else canonical_scope(scope, "rag", action)

    def _require_llm(self, action: str) -> LLMClient:
        """Return the configured LLM or fail before an answer-generation action."""
        if self.llm is None:
            raise RetrievalError(f"{action} requires an llm passed at RagRetriever construction")
        return self.llm

    @classmethod
    def from_config(cls, config: "AgentmakerConfig", *, embedder: "Optional[Embedder]" = None,
                    retriever: Optional[HybridRetriever] = None,
                    source_store: Optional[SourceStore] = None, llm: Optional[LLMClient] = None,
                    db_path: str = ":memory:", reranker: "Optional[Reranker]" = None,
                    query_transformer: Optional[QueryTransformer] = None,
                    prompts: "Optional[PromptRegistry]" = None,
                    scope: Optional[Scope] = None,
                    index_sync: Optional[IndexSync] = None) -> "RagRetriever":
        """Assemble a RagRetriever (read-only) from an AgentmakerConfig in one line: uses the sqlite backend by default; pass retriever / source_store to inject a custom backend.

        The backend is pluggable, same as Memory.from_config (the assembly root is in the app).
        Ingestion uses a separate IngestionPipeline, and the two must share the same backend.
        Typical usage: build rag first, then IngestionPipeline.from_config(config,
        retriever=rag.retriever, source_store=rag.source_store).

        Args:
            config: AgentmakerConfig (reads config.retrieval / config.rag).
            embedder: Required when using the default sqlite backend; not needed if retriever is
                injected.
            retriever: Inject a custom retrieval backend; if omitted a default sqlite one is built.
            source_store: Inject a custom source-of-truth store; if omitted a default sqlite one is built.
            llm: Used to generate answers; can be omitted if you only retrieve without asking.
            scope: Fixed ownership scope; per-call scopes may replace its non-base dimensions.

        Example:
            rag = RagRetriever.from_config(AgentmakerConfig(retrieval=RetrievalConfig(top_k=8)), embedder=emb, llm=llm)
        """
        config.retrieval.validate()    # Validate the slices we use before dispatch: fail early on illegal values instead of crashing at retrieval time.
        config.rag.validate()
        owns_retriever = retriever is None
        owns_source_store = source_store is None
        owns_sync = index_sync is None
        unattached_bookkeeping = None
        try:
            if retriever is None:
                if embedder is None:
                    raise ValueError("RagRetriever.from_config needs an embedder to use the default sqlite backend; or pass retriever= to inject a custom backend")
                from ..retrieval.backends import build_sqlite_hybrid   # Lazy import: default sqlite, keeps this module's top level decoupled.
                retriever = build_sqlite_hybrid(embedder, db_path=db_path, reranker=reranker, config=config.retrieval)
            if index_sync is None:
                from ..retrieval.index_sync import SqliteBookkeeping
                unattached_bookkeeping = SqliteBookkeeping(db_path)
                index_sync = SyncIndexSync(retriever, bookkeeping=unattached_bookkeeping)
                unattached_bookkeeping = None
            if source_store is None:
                source_store = SourceStore(db_path)
            rag = cls(
                retriever,
                source_store,
                llm,
                scope=scope,
                query_transformer=query_transformer,
                config=config.retrieval,
                rag_config=config.rag,
                prompts=prompts,
                index_sync=index_sync,
            )
        except BaseException as construction_error:
            seen = set()
            resources = [
                (owns_sync, index_sync),
                (owns_retriever, retriever),
                (owns_source_store, source_store),
            ]
            if unattached_bookkeeping is not None:
                resources.insert(0, (True, unattached_bookkeeping))
            for owned, resource in resources:
                if not owned or resource is None or id(resource) in seen:
                    continue
                seen.add(id(resource))
                try:
                    resource.close()
                except BaseException as cleanup_error:
                    construction_error.add_note(f"RagRetriever construction cleanup also failed: {cleanup_error}")
            raise
        rag._owns_source_store = owns_source_store
        rag._owns_retriever = owns_retriever
        rag._owns_sync = owns_sync
        return rag

    def retrieve(self, query: str, *, top_k: Optional[int] = None, scope: Optional[Scope] = None,
                 filters: Optional[List[MetadataFilter]] = None) -> List[RetrievalResult]:
        """Retrieve the most relevant chunks; after a hit, go back to the source-of-truth store to fill in complete information such as heading_path / doc_id.

        Args:
            query: The query text.
            top_k: How many chunks to return; if omitted, uses self.cfg.top_k (None uses the
                default, and does not swallow 0).
            scope: Retrieval scope; defaults to self.scope (fixed at construction). If provided,
                retrieval follows it (so context engineering can thread the run scope through).
            filters: Optional hard metadata filter (a list of MetadataFilter, see
                agentmaker.retrieval.types), passed to the backend for pre-filtering (e.g.
                search only a given doc_id / tag); the corresponding fields must have been declared
                via metadata_columns= when building the index.

        Returns:
            List[RetrievalResult]: The hit chunks; metadata contains heading_path / doc_id.
        """
        top_k = self.cfg.top_k if top_k is None else top_k
        eff_scope = self._scope(scope, "RagRetriever.retrieve")
        import time as _time
        t0 = _time.perf_counter() if self.tracer is not None else None   # Only time when a tracer is attached.
        hits = self._search(query, top_k=top_k, scope=eff_scope, filters=filters)
        results, orphans = [], []
        for hit in hits:
            chunk = self.source_store.get(hit.id, scope=eff_scope)
            if chunk is None:
                # Present in the index but missing from the source-of-truth store: a stale entry
                # (most likely an orphan left behind when the best-effort index deletion during a
                # document deletion failed). Skip it and do not return it: never leak leftover
                # content from the index as a result; self-heal by clearing it after the loop.
                orphans.append(hit.id)
                continue
            results.append(RetrievalResult(
                content=chunk.content, score=hit.score, source="rag", id=chunk.chunk_id,
                embedding=hit.embedding,  # Pass the vector through for reuse by context-engineering MMR.
                metadata={"heading_path": chunk.heading_path, "doc_id": chunk.doc_id,
                          "index": chunk.index,   # Position within the document: the expander locates neighbor / parent chunks by it.
                          "source": chunk.metadata.get("source", "")}))
        if orphans:
            self._sync.drop(orphans, scope=eff_scope)
        if self.expander is not None and results:
            results = self.expander.expand(results, source_store=self.source_store, scope=eff_scope)
        if self.tracer is not None and t0 is not None:
            self.tracer.emit({"type": EVENT_RAG_RETRIEVE, "query": query, "hits": len(results),
                              "latency_ms": int((_time.perf_counter() - t0) * 1000), **correlation()})
        return results

    def _search(self, query: str, *, top_k: int, scope: Scope,
                filters: Optional[List[MetadataFilter]] = None) -> List[RetrievalResult]:
        """Backend retrieval; if a query_transformer is attached, expand the query into several, embed them in a single batch, search each, then fuse back down to top_k."""
        pool = max(self.cfg.candidate_pool, top_k)
        if self.query_transformer is None:
            return (self.retriever.search(query, top_k=top_k, candidate_pool=pool, scope=scope,
                                          filters=filters) if filters else
                    self.retriever.search(query, top_k=top_k, candidate_pool=pool, scope=scope))
        # Clean the transformer output: first drop non-strings / whitespace (also avoiding a crash when dict.fromkeys hits an unhashable value), then deduplicate and cap; if all empty, fall back to the original query.
        clean = [q for q in (self.query_transformer.transform(query) or [query]) if isinstance(q, str) and q.strip()]
        queries = list(dict.fromkeys(clean))[:self.rag_cfg.mq_max_queries] or [query]
        if len(queries) == 1:
            return (self.retriever.search(queries[0], top_k=top_k, candidate_pool=pool, scope=scope,
                                          filters=filters) if filters else
                    self.retriever.search(queries[0], top_k=top_k, candidate_pool=pool, scope=scope))
        per_query = top_k * self.rag_cfg.mq_pool_factor   # Take a few more candidates per query, so recall is more stable after cross-query fusion.
        result_lists = (self.retriever.search_many(
            queries, top_k=per_query, candidate_pool=max(self.cfg.candidate_pool, per_query), scope=scope,
            filters=filters) if filters else self.retriever.search_many(
                queries, top_k=per_query, candidate_pool=max(self.cfg.candidate_pool, per_query), scope=scope))
        fusion = getattr(self.retriever, "fusion", None)
        if fusion is not None:
            return fusion.fuse(result_lists, top_k=top_k)
        return reciprocal_rank_fusion(result_lists, k=self.cfg.rrf_k, top_k=top_k)

    async def aretrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Async version of retrieve (to_thread; the embedding network call does not block the event loop). Same parameters as retrieve."""
        return await asyncio.to_thread(lambda: self.retrieve(query, **kwargs))

    async def ask(self, query: str, *, top_k: Optional[int] = None, scope: Optional[Scope] = None,
                  filters: Optional[List[MetadataFilter]] = None) -> "AskResult":
        """RAG question answering (non-streaming, async): retrieve -> assemble context -> LLM generation. Returns AskResult(answer, sources).

        Retrieval goes through aretrieve (does not block the event loop), then awaits chat
        generation. Synchronous callers use agentmaker.core.aio.run_sync.

        Args:
            query: The user's question.
            top_k: How many chunks to use as grounding.
            scope: Retrieval scope; defaults to self.scope, retrieval follows it when provided
                (specify as needed in multi-user / multi-app scenarios).
            filters: Optional hard metadata filter, passed through to retrieve (e.g. answer only
                within a given doc_id / tag).

        Returns:
            AskResult: answer text plus sources (list[SourceRef]; [] when there are no hits, which
                can be used to branch programmatically).
        """
        chunks = await self.aretrieve(query, top_k=top_k, scope=scope, filters=filters)
        if not chunks:
            return AskResult(answer=self.prompts.text("rag.no_hits"), sources=[])
        answer = (await governed_chat(self._require_llm("ask"), self._build_messages(query, chunks),
                                      tracer=self.tracer, origin="rag.ask")).content
        return AskResult(answer=answer, sources=self._sources(chunks))

    async def ask_stream(self, query: str, *, top_k: Optional[int] = None, scope: Optional[Scope] = None,
                         filters: Optional[List[MetadataFilter]] = None):
        """RAG question answering (streaming, async): yield the answer text piece by piece (consume with async for). Sources can be obtained separately via retrieve before / after.

        Retrieval goes through aretrieve, then yields piece by piece with async for (using
        llm.stream). Synchronous consumption uses agentmaker.core.aio.iter_sync.

        Args:
            query: The user's question.
            top_k: How many chunks to use as grounding.
            scope: Retrieval scope; defaults to self.scope.
            filters: Optional hard metadata filter, passed through to retrieve.
        """
        chunks = await self.aretrieve(query, top_k=top_k, scope=scope, filters=filters)
        if not chunks:
            yield self.prompts.text("rag.no_hits")
            return
        async for piece in self._stream_answer(self._build_messages(query, chunks)):
            yield piece

    async def _stream_answer(self, messages: List[dict]):
        """Stream one governed answer-generation call."""
        llm = self._require_llm("ask_stream")
        check_limits("llm")
        stats_box = {}
        start = perf_counter() if self.tracer is not None else 0.0
        stream_error: Optional[BaseException] = None
        try:
            async for piece in llm.stream(
                    messages, on_stats=lambda stats: stats_box.update(stats=stats)):
                yield piece
        except BaseException as error:
            stream_error = error
            raise
        finally:
            try:
                stats = stats_box.get("stats")
                usage = getattr(stats, "usage", None)
                record_llm(usage)
                if self.tracer is not None:
                    self.tracer.emit({
                        "type": EVENT_LLM_CALL,
                        "origin": "rag.ask_stream",
                        "model": getattr(llm, "model", None),
                        "usage": usage,
                        "latency_ms": int((perf_counter() - start) * 1000),
                        "streamed": True,
                        "finish_reason": getattr(stats, "finish_reason", None),
                        **correlation(),
                    })
            except Exception as cleanup_error:
                if stream_error is None:
                    raise
                stream_error.add_note(f"Streaming accounting or trace cleanup also failed: {cleanup_error}")
        enforce_token_limit_after_llm()
        check_deadline()

    def close(self) -> None:
        """Close resources created by this retriever wrapper."""
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        closed_ids: set[int] = set()
        resources = (
            (self._owns_sync, self._sync),
            (self._owns_retriever, self.retriever),
            (self._owns_source_store, self.source_store),
        )
        for owned, resource in resources:
            if not owned or id(resource) in closed_ids:
                continue
            closed_ids.add(id(resource))
            try:
                resource.close()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, exc, _tb):
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"RagRetriever cleanup also failed: {cleanup_error}")
        return False

    def _build_messages(self, query: str, chunks: List[RetrievalResult]) -> List[dict]:
        """Assemble chunks into a numbered [material] block plus the question, paired with the system prompt, forming the messages sent to the LLM."""
        context = "\n\n".join(
            f"[Source {i}: untrusted data]\n[{i}] {c.content}\n[End Source {i}]"
            for i, c in enumerate(chunks, start=1))
        user = self.prompts.render("rag.ask_user", context=context, query=query)
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]

    @staticmethod
    def _sources(chunks: List[RetrievalResult]) -> List[SourceRef]:
        """Assemble chunks into a source list (numbering matches the [n] markers in _build_messages)."""
        return [SourceRef(n=i, content=c.content,
                          heading_path=c.metadata.get("heading_path", ""),
                          doc_id=c.metadata.get("doc_id", ""))
                for i, c in enumerate(chunks, start=1)]
