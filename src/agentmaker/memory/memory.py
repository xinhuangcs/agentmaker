"""agentmaker.memory.memory: the memory manager.

Combines the source-of-truth store (MemoryStore, the authoritative record) with the
retrieval backend (HybridRetriever, the fast index), exposing: add / search / update /
delete (basics) plus forget / stats / summary / consolidate (lifecycle). summary and
consolidate require an LLM (pass ``llm`` at construction, ideally a cheap model such as
deepseek); everything else is pure data operations.

The source-of-truth store and the index are kept in sync through the replaceable IndexSync
seam (see index_sync.py): write paths (add/update/delete/rebuild) propagate changes into the
index and reconcile through it, while the read path (search) talks directly to the retrieval
backend. The store is authoritative and the index is eventually consistent: an index write
failure does not roll back the store, it only marks the entry pending reindex, which converges
via rebuild_index or read-time self-heal.
"""

import asyncio
from collections import defaultdict
import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, List, Optional, Sequence

from ..core.clock import now_utc
from ..core.exceptions import RetrievalError
from ..core.llm_clients import LLMClient
from ..runtime.execution.run_context import correlation, governed_chat
from ..retrieval.hybrid import HybridRetriever
from ..retrieval._coordination import shared_coordinator
from ..retrieval.scope import Scope, canonical_scope
from ..retrieval.index_sync import IndexSync, SyncIndexSync
from ..retrieval.types import MetadataFilter, RetrievalResult
from .store import MemoryStore

if TYPE_CHECKING:
    from ..config import AgentmakerConfig
    from ..retrieval.base import Embedder, Reranker
    from ..retrieval.types import RetrievalConfig
    from ..runtime.observability import Tracer
from .types import MemoryConfig, MemoryItem
from ..core.trace_events import EVENT_MEMORY_SEARCH


def _minmax_normalize(values: Sequence[float]) -> List[float]:
    """Linearly normalize a set of scores to 0..1 (max -> 1, min -> 0); when all equal or a single value, return 1.0 for each."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class Memory:
    """Memory manager: coordinates the source-of-truth store (MemoryStore) with the retrieval index (HybridRetriever)."""

    def __init__(self, retriever: HybridRetriever, store: MemoryStore, *,
                 llm: Optional[LLMClient] = None, scope: Optional[Scope] = None,
                 config: Optional[MemoryConfig] = None, index_sync: Optional[IndexSync] = None,
                 tracer: "Optional[Tracer]" = None):
        """Initialize the memory manager.

        Args:
            retriever: Retrieval backend handling the read path (vector + keyword + RRF + optional rerank).
            store: The memory source-of-truth store, holding complete MemoryItem records.
            llm: Optional; used by summary / consolidate. Calling those two without it raises.
            scope: Default ownership label that isolates memories from rag documents and from
                different users / agents; defaults to Scope(base="memory").
            config: Optional memory knobs (scoring weights / halflife / search_top_k / summary_top_k /
                batch_size / default_importance / forget_threshold); defaults to MemoryConfig().
                Per-call kwargs on each method override these (three-level resolution).
            index_sync: Optional derived-index sync seam (IndexSync, see index_sync.py); write paths
                (add/update/delete/rebuild) propagate changes to the retrieval index and reconcile
                through it. Defaults to SyncIndexSync(retriever) (synchronous write-through + in-process
                tracking); for async / distributed delivery (outbox + worker), implement one and inject
                it, and the Memory write paths stay unchanged.
            tracer: Optional tracer (duck-typed emit, same shape as Harness); once attached, search emits
                memory_search events and the summary / consolidate LLM calls flow into trace and RunPolicy
                governance. Zero overhead when not attached.
        """
        self.retriever = retriever          # read path (search) goes direct; writes / sync go through the seam below
        self.store = store
        self.llm = llm
        self.scope = canonical_scope(scope, "memory", "Memory construction")
        self.cfg = config or MemoryConfig()
        self.tracer = tracer
        self._sync = index_sync if index_sync is not None else SyncIndexSync(retriever, tracer=tracer)
        self._mutations = shared_coordinator(store)
        self._owns_store = False
        self._owns_retriever = False
        self._owns_sync = index_sync is None
        self._closed = False

    @classmethod
    def from_config(cls, config: "AgentmakerConfig", *, embedder: "Optional[Embedder]" = None,
                    retriever: Optional[HybridRetriever] = None,
                    store: Optional[MemoryStore] = None, llm: Optional[LLMClient] = None,
                    db_path: str = ":memory:", reranker: "Optional[Reranker]" = None,
                    scope: Optional[Scope] = None,
                    retrieval: "Optional[RetrievalConfig]" = None, index_sync: Optional[IndexSync] = None,
                    tracer: "Optional[Tracer]" = None) -> "Memory":
        """Assemble a Memory in one line from an AgentmakerConfig: defaults to the sqlite backend; pass retriever / store to inject a custom backend (without touching framework source).

        Pluggable backends follow the "assembly root lives in the app" principle (the library does not
        hardwire the wiring, it only ships default batteries; wiring is the app's job): to swap in
        pgvector or similar, implement the retrieval VectorStore / KeywordIndex interfaces and pass the
        retriever (plus store if needed); otherwise build_sqlite_hybrid is used.

        Args:
            config: AgentmakerConfig (reads config.memory; the default backend's retrieval knobs come
                from retrieval or config.retrieval).
            embedder: Text-to-vector encoder; required when using the default sqlite backend, unneeded
                when a retriever is injected.
            retriever: Inject a custom retrieval backend (HybridRetriever); defaults to building the
                sqlite backend.
            store: Inject a custom memory metadata store; defaults to MemoryStore(db_path).
            retrieval: Optional RetrievalConfig override (for the default backend): pass it when memory
                and rag should use different retrieval knobs. Note: Memory's return count / candidate pool
                is set by MemoryConfig.search_top_k plus an internal strategy (x4), which overrides the
                backend RetrievalConfig's top_k / candidate_pool; for memory the backend config mainly
                just supplies rrf_k (the fusion constant).

        Example:
            mem = Memory.from_config(AgentmakerConfig(memory=MemoryConfig(recency_halflife_hours=24)), embedder=emb, scope=ALICE)
        """
        config.memory.validate()                              # validate the slice we use before dispatch: fail early on bad values instead of crashing at search time
        owns_retriever = retriever is None
        owns_store = store is None
        owns_sync = index_sync is None
        unattached_bookkeeping = None
        try:
            if retriever is None:
                if embedder is None:
                    raise ValueError("Memory.from_config with the default sqlite backend requires an embedder; or pass retriever= to inject a custom backend")
                eff_retrieval = retrieval or config.retrieval
                eff_retrieval.validate()
                from ..retrieval.backends import build_sqlite_hybrid   # lazy import: default to sqlite without coupling the backend into this module's top level
                retriever = build_sqlite_hybrid(embedder, db_path=db_path, reranker=reranker, config=eff_retrieval)
            if index_sync is None:
                from ..retrieval.index_sync import SqliteBookkeeping
                unattached_bookkeeping = SqliteBookkeeping(db_path)
                index_sync = SyncIndexSync(
                    retriever, bookkeeping=unattached_bookkeeping, tracer=tracer)
                unattached_bookkeeping = None
            if store is None:
                store = MemoryStore(db_path)
            memory = cls(
                retriever,
                store,
                llm=llm,
                scope=scope,
                config=config.memory,
                index_sync=index_sync,
                tracer=tracer,
            )
        except BaseException as construction_error:
            seen = set()
            resources = [
                (owns_sync, index_sync),
                (owns_retriever, retriever),
                (owns_store, store),
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
                    construction_error.add_note(f"Memory construction cleanup also failed: {cleanup_error}")
            raise
        memory._owns_store = owns_store
        memory._owns_retriever = owns_retriever
        memory._owns_sync = owns_sync
        return memory

    # ---- basic read/write ----

    def _scope(self, scope: Optional[Scope], action: str) -> Scope:
        """Resolve a per-call scope while preserving the memory subsystem boundary."""
        return self.scope if scope is None else canonical_scope(scope, "memory", action)

    def _require_llm(self, action: str) -> LLMClient:
        """Return the configured LLM or fail before an LLM-dependent action."""
        if self.llm is None:
            raise RetrievalError(f"{action} requires an llm passed at Memory construction")
        return self.llm

    def add(self, content: str, *, type: str = "semantic", importance: Optional[float] = None,
            metadata: Optional[dict] = None, scope: Optional[Scope] = None) -> MemoryItem:
        """Record a memory: build a MemoryItem, persist to the source-of-truth store (authoritative), sync into the retrieval index through the seam, and return the item.

        An index write failure does not roll back the store; the entry is marked pending reindex
        (the store is authoritative and eventually consistent, see index_sync / pending_reindex).
        (This stores directly; for de-duplicating / updating "smart writes" of the same fact see SmartWriter.)

        Args:
            content: The memory body text.
            type: Memory type (free-form label, defaults to semantic).
            importance: Importance 0..1; when None, uses self.cfg.default_importance (an explicit 0 is not swallowed).
            metadata: Attached information.
            scope: Ownership label for this item; defaults to the Memory's default scope.

        Returns:
            MemoryItem: The newly created and persisted memory.
        """
        importance = self.cfg.default_importance if importance is None else importance
        if not 0.0 <= importance <= 1.0:
            raise RetrievalError(f"importance must be in 0..1, got {importance}")
        sc = self._scope(scope, "Memory.add")
        item = MemoryItem(content=content, type=type, importance=importance, metadata=metadata or {})
        with self._mutations.hold([item.id]):
            self.store.save(item, scope=sc)
            self._sync.index([item.id], [item.content], scope=sc)
        return item

    def update(self, id: str, content: str, *, scope: Optional[Scope] = None) -> Optional[MemoryItem]:
        """Update a memory's body text: the store deletes the old row and inserts the new one in a single transaction (authoritative, no lost rows), and the index converges via an upsert through the seam. Returns None if the item is not in this scope.

        Scope defaults to this Memory's scope (by default it only touches its own, preventing cross-scope
        edits); it mirrors add's scope= to allow explicit targeting when multiple scopes share one Memory
        instance (e.g. a coarse-ownership instance editing a fine-ownership memory). Deleting the old and
        writing the new go through store.replace in one transaction, so a mid-way crash does not lose this
        memory. An index write failure does not roll back the store; the entry is marked pending reindex
        and converges via rebuild_index or read-time self-heal (store authoritative, eventually consistent).
        """
        sc = self._scope(scope, "Memory.update")
        with self._mutations.hold([id]):
            resolved = self.store.get_with_scope(id, scope=sc)
            if resolved is None:
                return None
            item, exact_scope = resolved
            item.content = content
            item.updated_at = now_utc()
            self.store.replace(id, item, scope=exact_scope)
            self._sync.index([id], [content], scope=exact_scope)
            return item

    def invalidate(self, id: str, *, superseded_by: Optional[str] = None,
                   scope: Optional[Scope] = None) -> Optional[MemoryItem]:
        """Exclude a memory from retrieval while retaining its audit record.

        Args:
            id: The memory id to invalidate.
            superseded_by: Optional id of the replacement memory.
            scope: Ownership; defaults to this Memory's scope.

        Returns:
            MemoryItem: The item marked invalid; None if not in this scope.
        """
        sc = self._scope(scope, "Memory.invalidate")
        with self._mutations.hold([id]):
            resolved = self.store.get_with_scope(id, scope=sc)
            if resolved is None:
                return None
            item, exact_scope = resolved
            item.invalid_at = now_utc()
            item.superseded_by = superseded_by
            self.store.replace(id, item, scope=exact_scope)
            drop_exact = getattr(self._sync, "drop_exact", None)
            if drop_exact is None:
                self._sync.drop([id], scope=exact_scope)
            else:
                drop_exact([id], scope=exact_scope)
            return item

    def delete(self, id: str, *, scope: Optional[Scope] = None) -> None:
        """Delete a memory from the authoritative store and request index cleanup."""
        self.delete_many([id], scope=scope)

    def delete_many(self, ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Delete memories from the authoritative store and request best-effort index cleanup."""
        if not ids:
            return
        sc = self._scope(scope, "Memory.delete_many")
        with self._mutations.hold(ids):
            grouped = defaultdict(list)
            for id_, exact_scope in self.store.scopes_for_ids(ids, scope=sc):
                grouped[exact_scope].append(id_)
            try:
                requested = set(ids)
                for exact_scope in self._sync.exact_scopes(scope=sc):
                    indexed = self._sync.tracked_ids(scope=exact_scope)
                    pending = self._sync.pending(scope=exact_scope)
                    grouped[exact_scope].extend(sorted(requested & (indexed | pending)))
            except (AttributeError, NotImplementedError):
                pass
            self.store.delete_many(ids, scope=sc)
            covered: set = set()
            drop_range = getattr(self._sync, "drop_range", None)
            if grouped and drop_range is not None:
                drop_range(grouped, scope=sc)
                covered = {id_ for group_ids in grouped.values() for id_ in group_ids}
            leftover = sorted(set(ids) - covered)   # ids unknown to store and bookkeeping may still have stale index rows
            if leftover:
                self._sync.drop(leftover, scope=sc)

    def close(self) -> None:
        """Close resources created by this Memory instance."""
        if self._closed:
            return
        self._closed = True
        resources = (
            (self._owns_sync, self._sync),
            (self._owns_retriever, self.retriever),
            (self._owns_store, self.store),
        )
        errors: list[BaseException] = []
        closed_ids: set[int] = set()
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
        """Support closing owned resources on context-manager exit."""
        return self

    def __exit__(self, *exc):
        try:
            self.close()
        except BaseException as cleanup_error:
            original = exc[1] if len(exc) > 1 else None
            if original is None:
                raise
            original.add_note(f"Memory cleanup also failed: {cleanup_error}")
        return False

    def rebuild_index(self, *, scope: Optional[Scope] = None, batch_size: Optional[int] = None) -> int:
        """Rebuild the retrieval index from the store: iterate all memories in this scope and re-write them into the retrieval backend (re-embed + reindex).

        The store is the authoritative record and the index is its disposable derivative; this method
        delivers on "the derivative can be rebuilt from the record." Typical uses: after swapping the
        retrieval backend (e.g. SQLite vector store -> pgvector), import the data into the new index; or
        reload the whole thing when the index is corrupt or inconsistent with the store.

        Notes:
            - retriever.add is an upsert (same id overwrites): a fresh build for an empty index, an
              in-place refresh for an existing one.
            - Bookkeeping-known index orphans are removed per exact ownership footprint. Physical entries
              absent from bookkeeping remain covered by read-time self-heal.
            - Rebuilds by this Memory's (or the given) scope; with one Memory instance per user, call
              rebuild_index for each.

        Args:
            scope: Which ownership range to rebuild; defaults to this Memory's scope.
            batch_size: Rows written per batch; batching avoids embedding too much at once on large stores.

        Returns:
            int: The number of memories reloaded.
        """
        batch_size = self.cfg.rebuild_batch_size if batch_size is None else batch_size
        sc = self._scope(scope, "Memory.rebuild_index")
        with self._mutations.hold_all():
            rows = self.store.all_with_scopes(scope=sc)
            grouped = defaultdict(list)
            for item, exact_scope in rows:
                grouped[exact_scope].append(item)
            try:
                footprints = set(self._sync.exact_scopes(scope=sc)) | set(grouped)
            except (AttributeError, NotImplementedError):
                footprints = set(grouped)
            if not footprints:
                return self._sync.reconcile([], scope=sc, batch_size=batch_size)
            return sum(
                self._sync.reconcile(grouped.get(exact_scope, []), scope=exact_scope,
                                     batch_size=batch_size)
                for exact_scope in footprints
            )

    def pending_reindex(self, *, scope: Optional[Scope] = None) -> set:
        """Return the set of ids pending reindex (recent index writes that failed and have not yet been converged by rebuild_index / reconcile).

        Store authoritative, index eventually consistent (see index_sync): an index write failure does not
        roll back the store, it only marks the entry pending. An app can use this to monitor drift and
        periodically trigger rebuild_index to converge. The default backend's (SyncIndexSync) pending set
        is in-process state (cleared on restart, rebuildable via rebuild).

        Args:
            scope: Which ownership range to query; defaults to this Memory's scope.
        """
        return self._sync.pending(scope=self._scope(scope, "Memory.pending_reindex"))

    def search(self, query: str, *, top_k: Optional[int] = None, scope: Optional[Scope] = None,
               relevance_weight: Optional[float] = None, recency_weight: Optional[float] = None,
               importance_weight: Optional[float] = None,
               recency_halflife_hours: Optional[float] = None,
               filters: Optional[List[MetadataFilter]] = None) -> List[RetrievalResult]:
        """Recall the most relevant memories for a query, ranked by a combined relevance x recency x importance score.

        Inspired by Generative Agents: the final score = each component normalized to 0..1 then weighted
        and summed. With all three weights at 0 it degrades to pure relevance ranking (backward
        compatible). Recency decays via a halflife: more recent approaches 1, older approaches 0. The knobs
        below default to self.cfg (MemoryConfig) when None (an explicit 0 is not swallowed).

        Args:
            query: The query text.
            top_k: Number of results to return (defaults to cfg.search_top_k).
            relevance_weight: Relevance weight (defaults to cfg.relevance_weight).
            recency_weight: Recency weight (defaults to cfg.recency_weight).
            importance_weight: Importance weight (defaults to cfg.importance_weight).
            recency_halflife_hours: The recency halflife in hours (defaults to cfg.recency_halflife_hours);
                smaller favors newer memories more strongly.
            filters: Optional metadata hard filter (list of MetadataFilter, see agentmaker.retrieval.types),
                passed through to the retrieval backend for pre-filtering; the backend must have declared
                the corresponding filterable columns. Complements the three-way scoring (soft ranking):
                filters are hard constraints.

        Returns:
            List[RetrievalResult]: Ordered by combined score, highest first; metadata carries the
                relevance / recency / importance / final component scores.
        """
        top_k = self.cfg.search_top_k if top_k is None else top_k
        relevance_weight = self.cfg.relevance_weight if relevance_weight is None else relevance_weight
        recency_weight = self.cfg.recency_weight if recency_weight is None else recency_weight
        importance_weight = self.cfg.importance_weight if importance_weight is None else importance_weight
        recency_halflife_hours = self.cfg.recency_halflife_hours if recency_halflife_hours is None else recency_halflife_hours
        if top_k < 1:
            raise RetrievalError(f"top_k must be >= 1, got {top_k}")
        if recency_halflife_hours <= 0:
            raise RetrievalError(f"recency_halflife_hours must be > 0, got {recency_halflife_hours}")
        if min(relevance_weight, recency_weight, importance_weight) < 0:
            raise RetrievalError("scoring weights must be non-negative (relevance / recency / importance)")
        sc = self._scope(scope, "Memory.search")
        t0 = now_utc() if self.tracer is not None else None   # only time when a tracer is attached (zero-overhead principle)
        # over-fetch candidates then re-rank on the combined score: otherwise we only rank the top few of the relevance list and miss items that are "so-so relevant but very new / very important"
        pool = max(top_k * 4, top_k)
        # candidate_pool must scale up to >= pool: the backend defaults to candidate_pool=20, and when pool>20 (i.e. top_k>=6) its validation raises RetrievalError
        hits = (self.retriever.search(query, top_k=pool, candidate_pool=max(pool, 20), scope=sc,
                                      filters=filters) if filters else
                self.retriever.search(query, top_k=pool, candidate_pool=max(pool, 20), scope=sc))
        raw = [(h, self.store.get(h.id, scope=sc)) for h in hits]
        # in the index but not in the store (stale), or already soft-invalidated (leftover from a failed invalidate drop): both are invisible, self-heal at read time
        orphans = [h.id for h, m in raw if m is None or m.invalid_at is not None]
        if orphans:
            self._sync.drop(orphans, scope=sc)         # read-time self-heal: drop orphans from the index through the seam (drop is best-effort internally and does not raise)
        items = [(h, m) for h, m in raw if m is not None and m.invalid_at is None]
        if not items:
            if self.tracer is not None and t0 is not None:
                self.tracer.emit({"type": EVENT_MEMORY_SEARCH, "query": query, "hits": 0,
                                  "latency_ms": int((now_utc() - t0).total_seconds() * 1000), **correlation()})
            return []

        now = now_utc()
        rel_raw = _minmax_normalize([h.score for h, _ in items])      # relevance scores are on different scales across implementations, so normalize first
        scored = []
        for (hit, item), rel in zip(items, rel_raw):
            anchor = self._recency_anchor_time(item)
            rec = 0.5 ** ((now - anchor).total_seconds() / 3600.0 / recency_halflife_hours)
            imp = item.importance
            final = relevance_weight * rel + recency_weight * rec + importance_weight * imp
            scored.append((final, rel, rec, imp, item, hit.embedding))  # carry the vector for MMR reuse
        scored.sort(key=lambda t: t[0], reverse=True)

        results = []
        for final, rel, rec, imp, item, embedding in scored[:top_k]:
            results.append(RetrievalResult(
                content=item.content, score=final, source="memory", id=item.id, embedding=embedding,
                metadata={"type": item.type, "created_at": item.created_at.isoformat(),
                          "relevance": round(rel, 4), "recency": round(rec, 4),
                          "importance": imp, "final": round(final, 4), **item.metadata}))
        if results:
            try:
                self.store.touch([r.id for r in results], scope=sc)   # a hit means "was used": write back last_accessed_at
            except Exception as e:  # noqa: BLE001  usage feedback is a side channel (best-effort); a failure does not affect this read, just log at debug
                logging.getLogger(__name__).debug("failed to write back last_accessed (does not affect this read): %r", e)
        if self.tracer is not None and t0 is not None:
            self.tracer.emit({"type": EVENT_MEMORY_SEARCH, "query": query, "hits": len(results),
                              "latency_ms": int((now_utc() - t0).total_seconds() * 1000), **correlation()})
        return results

    def _recency_anchor_time(self, item: MemoryItem) -> datetime:
        """Time anchor for recency scoring: with anchor="last_accessed" and a recorded hit, use last_accessed_at (the Generative Agents original "used memories stay fresh"); otherwise use updated_at (edited content dates from the edit) falling back to created_at."""
        if self.cfg.recency_anchor == "last_accessed" and item.last_accessed_at is not None:
            return item.last_accessed_at
        return item.updated_at or item.created_at

    # ---- a* async variants (writes / retrieval are synchronous SQLite + embedding network calls, all wrapped in to_thread so they can be awaited without blocking the event loop) ----

    async def aadd(self, content: str, **kwargs) -> MemoryItem:
        """Async variant of add (to_thread)."""
        return await asyncio.to_thread(lambda: self.add(content, **kwargs))

    async def aupdate(self, id: str, content: str, **kwargs) -> Optional[MemoryItem]:
        """Async variant of update (to_thread)."""
        return await asyncio.to_thread(lambda: self.update(id, content, **kwargs))

    async def adelete(self, id: str, **kwargs) -> None:
        """Async variant of delete (to_thread)."""
        await asyncio.to_thread(lambda: self.delete(id, **kwargs))

    async def ainvalidate(self, id: str, **kwargs) -> Optional[MemoryItem]:
        """Async variant of invalidate (to_thread)."""
        return await asyncio.to_thread(lambda: self.invalidate(id, **kwargs))

    async def adelete_many(self, ids: List[str], **kwargs) -> None:
        """Async variant of delete_many (to_thread)."""
        await asyncio.to_thread(lambda: self.delete_many(ids, **kwargs))

    async def asearch(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Async variant of search (to_thread)."""
        return await asyncio.to_thread(lambda: self.search(query, **kwargs))

    async def aforget(self, **kwargs) -> List[str]:
        """Async variant of forget (to_thread)."""
        return await asyncio.to_thread(lambda: self.forget(**kwargs))

    async def arebuild_index(self, **kwargs) -> int:
        """Async variant of rebuild_index (to_thread; full re-embedding is a long network operation, so never run it synchronously on the event loop)."""
        return await asyncio.to_thread(lambda: self.rebuild_index(**kwargs))

    # ---- lifecycle ----

    def forget(self, *, strategy: str = "importance", threshold: Optional[float] = None,
               max_age_days: Optional[float] = None, capacity: Optional[int] = None,
               scope: Optional[Scope] = None) -> List[str]:
        """Forget by strategy, returning the list of deleted ids.

        Strategies:
            importance: delete items with importance < threshold (threshold defaults to self.cfg.forget_threshold);
            age:        delete items older than max_age_days days (requires max_age_days);
            capacity:   keep only the "most important + newest" capacity items, delete the rest (requires capacity).
        """
        threshold = self.cfg.forget_threshold if threshold is None else threshold
        sc = self._scope(scope, "Memory.forget")
        items = self.store.all(scope=sc)
        if strategy == "importance":
            if not 0.0 <= threshold <= 1.0:
                raise RetrievalError(f"forget(strategy='importance') threshold must be in 0..1, got {threshold}")
            victims = [m for m in items if m.importance < threshold]
        elif strategy == "age":
            if max_age_days is None:
                raise RetrievalError("forget(strategy='age') requires max_age_days")
            if max_age_days <= 0:
                raise RetrievalError(f"forget(strategy='age') max_age_days must be > 0, got {max_age_days}")
            cutoff = now_utc() - timedelta(days=max_age_days)
            victims = [m for m in items if m.created_at < cutoff]
        elif strategy == "capacity":
            if capacity is None:
                raise RetrievalError("forget(strategy='capacity') requires capacity")
            if capacity < 0:
                raise RetrievalError(f"forget(strategy='capacity') capacity must be >= 0, got {capacity}")
            ranked = sorted(items, key=lambda m: (m.importance, m.created_at), reverse=True)
            victims = ranked[capacity:]
        else:
            raise RetrievalError(f"unknown forget strategy: {strategy} (available: importance / age / capacity)")
        self.delete_many([m.id for m in victims], scope=sc)
        return [m.id for m in victims]

    def stats(self, *, scope: Optional[Scope] = None) -> dict:
        """Stats: {total, by_type} (total count + distribution by type). Pure data, no LLM call."""
        items = self.store.all(scope=self._scope(scope, "Memory.stats"))
        by_type: dict = {}
        for m in items:
            by_type[m.type] = by_type.get(m.type, 0) + 1
        return {"total": len(items), "by_type": by_type}

    _SUMMARY_SYS = "In concise English, summarize the following memory entries about the user into one coherent paragraph; base it only on the given content, do not fabricate."
    _CONSOLIDATE_SYS = ("Tidy up the user's memories: merge the semantically duplicate ones, keep the latest of any contradictions, and phrase them concisely. "
                        "Output one final fact per line, with no numbering and no extra text.")

    async def summary(self, query: Optional[str] = None, *, top_k: Optional[int] = None,
                      scope: Optional[Scope] = None) -> str:
        """Use the LLM to summarize the selected memories into one paragraph (async: the LLM call awaits chat; the synchronous DB fetch runs in a thread pool).

        Requires an llm passed at construction. top_k defaults to self.cfg.summary_top_k when None.
        Sync callers go through agentmaker.core.aio.run_sync.
        """
        top_k = self.cfg.summary_top_k if top_k is None else top_k
        if top_k < 1:
            raise RetrievalError(f"summary top_k must be >= 1, got {top_k}")
        sc = self._scope(scope, "Memory.summary")
        llm = self._require_llm("summary")
        items = await asyncio.to_thread(self._summary_items, query, top_k, sc)
        if not items:
            return "(no relevant memories)"
        msgs = self._summary_messages(items)
        return (await governed_chat(llm, msgs, tracer=self.tracer, origin="memory.summary")).content

    def _summary_items(self, query, top_k, scope=None):
        """Fetch the memory entries to summarize (shared by sync/async)."""
        if self.llm is None:
            raise RetrievalError("summary requires an llm passed at Memory construction")
        sc = scope or self.scope
        if query:
            return [m for m in (self.store.get(h.id, scope=sc)
                                for h in self.search(query, top_k=top_k, scope=sc)) if m]
        return self.store.all(scope=sc)[:top_k]

    def _summary_messages(self, items):
        """Assemble the message list for the summary."""
        bullets = "\n".join(f"- {m.content}" for m in items)
        return [{"role": "system", "content": self._SUMMARY_SYS}, {"role": "user", "content": bullets}]

    async def consolidate(self, *, scope: Optional[Scope] = None) -> dict:
        """Consolidate (async): hand all memories to the LLM to de-duplicate / merge / keep-latest-on-conflict, then replace them in the store; returns {before, after}. Requires an llm.

        The LLM call awaits chat; the synchronous fetch / persist DB operations run in a thread pool.
        Sync callers go through agentmaker.core.aio.run_sync.
        """
        sc = self._scope(scope, "Memory.consolidate")
        llm = self._require_llm("consolidate")
        items = await asyncio.to_thread(self._consolidate_items, sc)
        if not items:
            return {"before": 0, "after": 0}
        resp = await governed_chat(llm, self._consolidate_messages(items),
                                   tracer=self.tracer, origin="memory.consolidate")
        return await asyncio.to_thread(self._apply_consolidate, items,
                                       self._parse_consolidate(resp.content), sc)

    def _consolidate_items(self, scope=None):
        """Fetch all memories (shared by sync/async)."""
        if self.llm is None:
            raise RetrievalError("consolidate requires an llm passed at Memory construction")
        return self.store.all(scope=scope or self.scope)

    def _consolidate_messages(self, items):
        """Assemble the message list for consolidation."""
        bullets = "\n".join(f"- {m.content}" for m in items)
        return [{"role": "system", "content": self._CONSOLIDATE_SYS}, {"role": "user", "content": bullets}]

    @staticmethod
    def _parse_consolidate(content: str) -> List[str]:
        """Parse the consolidation output line by line into a list of final facts."""
        merged = [line.strip("-•* \t").strip() for line in content.splitlines() if line.strip()]
        return [m for m in merged if m]

    def _apply_consolidate(self, items, merged: List[str], scope=None) -> dict:
        """Persist: add the merged results first, then soft-invalidate the old entries; do nothing if the parse is empty. Shared by sync/async.

        Attribute inheritance (lossy compression, many-to-many with no per-item correspondence): importance
        takes the mean of the source entries (not a global max, which would make repeated consolidate pull
        every memory's importance toward the maximum and permanently break the importance dimension), type
        takes the mode. Old entries go through soft invalidate rather than physical deletion (consistent
        with SmartWriter's evolution-chain philosophy, so the audit trail does not evaporate; physical
        deletion is reserved for forget), and superseded_by is left empty (no single successor to point to).
        Invalidated entries are excluded from store.all's default result, so a second consolidate does not
        re-feed them.
        """
        if not merged:
            return {"before": len(items), "after": len(items)}  # empty parse: do nothing, to avoid accidentally wiping everything
        with self._mutations.hold_all():
            for snapshot in items:
                current = self.store.get(snapshot.id, scope=scope or self.scope)
                if current is None or (current.content, current.updated_at, current.invalid_at) != (
                        snapshot.content, snapshot.updated_at, snapshot.invalid_at):
                    raise RetrievalError(
                        "memories changed while consolidation was running; retry with a fresh snapshot")
            kept_importance = sum(m.importance for m in items) / len(items)
            kept_type = max(set(m.type for m in items), key=lambda t: sum(
                x.type == t for x in items))
            sc = scope or self.scope
            for fact in merged:
                self.add(fact, type=kept_type, importance=kept_importance, scope=sc)
            for item in items:
                self.invalidate(item.id, scope=sc)
        return {"before": len(items), "after": len(merged)}
