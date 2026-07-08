"""agentmaker.retrieval.hybrid: hybrid retrieval orchestration (vector + keyword, fusion (RRF by default), optional rerank).

Vector retrieval understands semantics while keyword (BM25) retrieval is strong at exact terms; the two
complement each other. HybridRetriever runs them in parallel, then combines them through a pluggable
FusionStrategy into a single ranking. The default is RRF (Reciprocal Rank Fusion: it only looks at ranks
rather than scores, sidestepping the "two incomparable score scales" problem, and needs no tuning). When
you have an evaluation set and want to weight the two paths, inject your own fusion strategy (interface in
base.py). If a Reranker is injected, the fused candidates are further refined by a cross-encoder. This is
the unified retrieval entry point shared by memory and rag.

This class is storage-agnostic: it only knows the abstract interfaces in `base.py` and touches no concrete
backend. SQLite's shared connection and single-transaction atomic writes across both indexes are handled by
the `SqliteHybridRetriever` subclass in `sqlite.py` (constructed via `build_sqlite_hybrid`).
Every public capability has an a* async version (defaults to wrapping the sync implementation in to_thread;
native async backends may override).
"""

import asyncio
import logging
from typing import Dict, List, Optional

from ..core.exceptions import RetrievalError
from .base import Embedder, FusionStrategy, KeywordIndex, Reranker, VectorStore
from .scope import Scope, require_explicit_scope
from .types import MetadataFilter, RetrievalConfig, RetrievalResult

# RRF smoothing constant; 60 is the default proposed by Cormack et al. (2009) and widely adopted since,
# used to dampen the excessive dominance of top ranks.
# It is also the fallback default for direct callers of reciprocal_rank_fusion; HybridRetriever instead
# reads config.rrf_k (which also defaults to this value).
_RRF_K = 60


def require_valid_top_k(top_k: int, *, candidate_pool: Optional[int] = None) -> None:
    """Validate the number of results to return: top_k must be at least 1, and if candidate_pool is given it must be >= top_k (otherwise the candidate pool is smaller than the final requirement).

    This unifies how backends handle an invalid top_k; otherwise FTS5's LIMIT -1 would "return everything"
    and vec0's k=-1 would raise outright, producing inconsistent behavior.
    """
    if top_k < 1:
        raise RetrievalError(f"top_k must be >= 1, got {top_k}.")
    if candidate_pool is not None and candidate_pool < top_k:
        raise RetrievalError(f"candidate_pool ({candidate_pool}) must be >= top_k ({top_k}).")


def reciprocal_rank_fusion(result_lists: List[List[RetrievalResult]], *,
                           k: int = _RRF_K, top_k: int = 10) -> List[RetrievalResult]:
    """Fuse multiple "each already sorted" result lists into one ranking via RRF.

    A result ranked r-th (starting at 1) in a given list contributes 1/(k+r) points; the scores for the same
    id across multiple lists are summed, and finally the top_k are taken by descending total score. Entries
    hit by multiple paths at once naturally score higher.

    Fusion aligns by id (in HybridRetriever, two paths sharing an id are the same entry, and merging by id to
    add scores is exactly the point of RRF); when the id is empty it degrades to aggregating by content, to
    avoid wrongly merging distinct empty-id results. Note: if you use this function to fuse results from
    different corpora that happen to share ids, the caller must first make the ids globally unique.

    Args:
        result_lists: Multiple result lists, each already sorted internally from most to least relevant.
        k: Smoothing constant, default 60.
        top_k: Number of results to return after fusion.

    Returns:
        List[RetrievalResult]: The top_k results after fusion re-ranking, with score being the RRF total.
    """
    scores: Dict[str, float] = {}
    keep: Dict[str, RetrievalResult] = {}
    merged_meta: Dict[str, dict] = {}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            fuse_key = r.id if r.id else "\x00" + r.content   # use id when present; degrade to content-based aggregation for empty id
            scores[fuse_key] = scores.get(fuse_key, 0.0) + 1.0 / (k + rank)
            kept_r = keep.setdefault(fuse_key, r)  # keep the first one seen, used to carry content / source / id
            if kept_r.embedding is None and r.embedding is not None:
                keep[fuse_key] = r                # first seen without a vector, later seen with one, so upgrade the carrier (same fuse_key must be the same entry): avoids multi-query RRF dropping the vector and weakening downstream MMR dedup
            merged_meta.setdefault(fuse_key, {}).update(r.metadata)  # merge each path's metadata (e.g. keep both distance and bm25)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out = []
    for key, score in ranked:
        r = keep[key]
        out.append(RetrievalResult(content=r.content, score=score, source=r.source, id=r.id,  # output still uses the original id (may be empty)
                                   embedding=r.embedding,  # pass the vector through for the upper layer's MMR to reuse
                                   metadata={**merged_meta[key], "rrf_score": score}))
    return out


class RRFFusion(FusionStrategy):
    """The default battery for the fusion strategy: wraps reciprocal_rank_fusion (a tuning-free baseline)."""

    def __init__(self, k: int = _RRF_K):
        """k: the RRF smoothing constant (default 60)."""
        self.k = k

    def fuse(self, result_lists, *, top_k):
        """Fuse via RRF (see reciprocal_rank_fusion)."""
        return reciprocal_rank_fusion(result_lists, k=self.k, top_k=top_k)


def _maybe_filters(filters) -> dict:
    """Pass filters downstream as a kwarg only when given (absent from the call when not set): this keeps the
    default path zero-overhead and does not force duck-typed injected implementations or test stubs to grow a
    filters parameter."""
    return {"filters": filters} if filters else {}


class HybridRetriever:
    """Hybrid retriever (storage-agnostic): coordinates Embedder + VectorStore + KeywordIndex (+ optional Reranker / FusionStrategy),
    exposing add() / search().

    This is the unified entry point the retrieval foundation gives the upper layers; both memory and rag use
    it, isolating their data by scope. Each dependency is injected, so the vector store, keyword, rerank, and
    fusion backends can all be replaced independently. `add` is best-effort compensating (the generic fallback
    when the two indexes use separate connections): if the keyword write fails, the vector just written is
    rolled back. For "single-transaction atomic writes across both indexes" (shared connection, old values not
    lost on update failure), use `SqliteHybridRetriever` in `sqlite.py` (constructed via `build_sqlite_hybrid`).
    """

    def __init__(self, embedder: Embedder, vector_store: VectorStore, keyword_index: KeywordIndex,
                 reranker: Optional[Reranker] = None, *, config: Optional[RetrievalConfig] = None,
                 fusion: Optional[FusionStrategy] = None):
        """
        Args:
            embedder: Turns text into vectors.
            vector_store: Vector storage and retrieval (dense).
            keyword_index: Keyword retrieval (sparse / BM25).
            reranker: Optional reranker; if omitted, only "vector + keyword + fusion" runs, and if given it
                refines after fusion.
            config: Optional retrieval knobs (top_k / candidate_pool / rrf_k); defaults to RetrievalConfig() if
                omitted. Parsed once at construction into self.config; search's per-call kwargs may override it
                (three-level resolution).
            fusion: Optional fusion strategy (FusionStrategy, see base.py); defaults to RRFFusion(config.rrf_k)
                if omitted. Inject your own (alpha weighting / RSF etc.) when you have an evaluation set and
                want to tune the relative weights of the two paths.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.reranker = reranker
        self.config = config or RetrievalConfig()
        self.config.validate()   # reject invalid knobs (negative rrf_k / candidate_pool < top_k) at construction, rather than crashing at retrieval time
        self.fusion = fusion if fusion is not None else RRFFusion(k=self.config.rrf_k)

    def add(self, ids: List[str], contents: List[str], *, scope: Optional[Scope] = None,
            metadatas: Optional[List[dict]] = None) -> None:
        """Write in one call (upsert), keeping the vector store and keyword index in sync (embedding is done internally).

        Best-effort compensation: write the vector first, then the keyword; if the latter fails, delete the
        vector just written back out (to avoid a "vector present, keyword missing" half-write), and the
        original exception is re-raised as usual. Compensation rolls back cleanly for a failed insert; for a
        failed update-overwrite it only deletes without restoring the old value. To avoid this entirely, use
        `SqliteHybridRetriever` (shared connection, single transaction).

        Args:
            ids: A unique identifier per entry.
            contents: The original text per entry.
            scope: Ownership label; defaults to Scope().
            metadatas: Optional, a metadata dict per entry (same length as ids); only fields the index has
                declared are stored as filterable columns (see base.py).
        """
        if len(ids) != len(contents):
            raise RetrievalError("add(): ids and contents must have the same length.")
        md = {"metadatas": metadatas} if metadatas else {}
        vectors = self.embedder.embed(contents)
        self.vector_store.add(ids, vectors, contents, scope=scope, **md)
        try:
            self.keyword_index.add(ids, contents, scope=scope, **md)
        except Exception:
            try:
                self.vector_store.delete_exact(ids, scope=scope)  # undo the vector just written: exact delete (== the write footprint), so it does not wrongly delete same-id rows from other scopes
            except Exception as ce:  # noqa: BLE001  a failed compensation must not mask the original exception; but consistency is harmed (dirty chunks left in the vector store), so leave a signal via error
                logging.getLogger(__name__).error("failed to undo the vector after a keyword-index write failure; the vector store may contain dirty chunks ids=%s: %r", ids, ce)
            raise

    def replace(self, old_ids: List[str], new_ids: List[str], contents: List[str], *,
                scope: Optional[Scope] = None, metadatas: Optional[List[dict]] = None) -> None:
        """Replace a document's old chunks with new-version chunks: first ingest the new chunks (add), then delete the old chunks outside of new. Used for RAG document re-ingestion dedup.

        The base class is compensating "add first, then delete" (the two indexes use separate connections and
        cannot share a single transaction): old chunks are deleted only after all new chunks are ingested
        successfully; on failure it does not delete the old and does not lose data. For "atomic replace" (no
        window of concurrent double hits, no residue even if deleting the old fails), use
        `SqliteHybridRetriever` (shared connection, single transaction, see its replace override). Old and new
        chunk ids differ, so deleting the old does not touch the new.
        """
        # Reject empty scope early: this guard originally fired only in the delete after add, at which point the new chunks were already ingested but the error left old and new coexisting. Move it ahead of the write.
        require_explicit_scope(scope or Scope(), False, "replace")
        self.add(new_ids, contents, scope=scope, metadatas=metadatas)
        stale = [i for i in old_ids if i not in set(new_ids)]
        if stale:
            self.delete(stale, scope=scope)

    def delete(self, ids: List[str], *, scope: Optional[Scope] = None, all_scopes: bool = False) -> None:
        """Delete by id from the vector store and keyword index in sync (within the scope range).

        Both deletes are idempotent (deleting a nonexistent id is a no-op), so if one side fails you can simply
        retry the whole delete to reach consistency. When the scope is entirely empty (a delete across the
        whole store), it is rejected unless all_scopes=True is passed explicitly (guard, see scope.py).
        """
        require_explicit_scope(scope or Scope(), all_scopes, "delete")
        self.vector_store.delete(ids, scope=scope)
        self.keyword_index.delete(ids, scope=scope)

    def close(self) -> None:
        """Close the underlying vector store and keyword index resources."""
        self.vector_store.close()
        self.keyword_index.close()

    def search(self, query: str, *, top_k: Optional[int] = None, candidate_pool: Optional[int] = None,
               scope: Optional[Scope] = None, all_scopes: bool = False,
               filters: Optional[List[MetadataFilter]] = None) -> List[RetrievalResult]:
        """Take candidate_pool entries each from vector and keyword -> fuse (RRF by default) -> (if a reranker) refine -> return top_k.

        Args:
            query: The query text.
            top_k: Final number of results; defaults to config.top_k if omitted (None uses the default, without
                swallowing 0).
            candidate_pool: How many entries each path (vector / keyword) takes into fusion / rerank; defaults
                to config.candidate_pool if omitted.
            scope: The ownership range to restrict retrieval to (B semantics: only filters non-empty
                dimensions); defaults to Scope().
            all_scopes: Guard switch; when the scope is entirely empty (searching the whole store) this must be
                set to True explicitly, otherwise it is rejected (see scope.py).
            filters: Optional metadata filter (AND semantics, see MetadataFilter in types.py); both retrieval
                paths narrow candidates by it first (pre-filtering).

        Returns:
            List[RetrievalResult]: The final top_k entries.
        """
        top_k = self.config.top_k if top_k is None else top_k
        candidate_pool = self.config.candidate_pool if candidate_pool is None else candidate_pool
        require_valid_top_k(top_k, candidate_pool=candidate_pool)
        require_explicit_scope(scope or Scope(), all_scopes, "search")  # fail before embed, saving a pointless embedding call
        query_vector = self.embedder.embed([query])[0]
        return self._fuse_one(query, query_vector, top_k=top_k, candidate_pool=candidate_pool,
                              scope=scope, filters=filters)

    def search_many(self, queries: List[str], *, top_k: Optional[int] = None, candidate_pool: Optional[int] = None,
                    scope: Optional[Scope] = None, all_scopes: bool = False,
                    filters: Optional[List[MetadataFilter]] = None) -> List[List[RetrievalResult]]:
        """Batch-retrieve for multiple queries: embed all queries at once, then run vector + keyword retrieval (+ optional rerank) for each.

        Returns a list the same length as queries (one top_k ranking per query). Compared with searching one
        by one, this saves N-1 embedding network round-trips; rag's query expansion (MQE / HyDE) uses it for
        multiple queries, batching N embeddings into one. Parameters are the same as search (queries is the
        list of queries).
        """
        top_k = self.config.top_k if top_k is None else top_k
        candidate_pool = self.config.candidate_pool if candidate_pool is None else candidate_pool
        require_valid_top_k(top_k, candidate_pool=candidate_pool)
        require_explicit_scope(scope or Scope(), all_scopes, "search")
        if not queries:
            return []
        query_vectors = self.embedder.embed(list(queries))   # one batched embedding, saving N-1 network round-trips
        return [self._fuse_one(q, v, top_k=top_k, candidate_pool=candidate_pool, scope=scope, filters=filters)
                for q, v in zip(queries, query_vectors)]

    def _fuse_one(self, query: str, query_vector: List[float], *, top_k: int,
                  candidate_pool: int, scope: Optional[Scope],
                  filters: Optional[List[MetadataFilter]] = None) -> List[RetrievalResult]:
        """The single-query "vector + keyword -> fuse (self.fusion) -> (optional) refine -> top_k"; shared by search / search_many."""
        fkw = _maybe_filters(filters)
        vec_hits = self.vector_store.search(query_vector, top_k=candidate_pool, scope=scope, **fkw)
        kw_hits = self.keyword_index.search(query, top_k=candidate_pool, scope=scope, **fkw)
        if self.reranker is None:
            return self.fusion.fuse([vec_hits, kw_hits], top_k=top_k)
        # With rerank: first fuse a larger candidate pool (dedup + coarse ranking), then hand it to the cross-encoder to refine into top_k
        fused = self.fusion.fuse([vec_hits, kw_hits], top_k=candidate_pool)
        return self.reranker.rerank(query, fused, top_k=top_k)

    # -- a* async versions --
    #    The read path (asearch / asearch_many, hot path, no atomicity concerns) uses the port's a* true async:
    #    aembed gets the vector, and the vector path + keyword path run concurrently via asyncio.gather. Native
    #    async backends (httpx embedder / remote vector store) thus gain real concurrency, while sync backends
    #    each occupy a thread via the port's default a* (to_thread). The write path (aadd / areplace / adelete,
    #    ingestion, non-hot, sensitive to cross-index single-transaction atomicity) keeps wrapping its own sync
    #    atomic version in to_thread (the atomic semantics of shared connection + commit are not split down to
    #    the port level, to avoid accidentally dropping the commit).
    #    Backends with a shared connection + lock (SqliteHybridRetriever) override the read a* to be a whole
    #    to_thread (gather gains nothing under a single lock).

    async def asearch(self, query: str, *, top_k: Optional[int] = None, candidate_pool: Optional[int] = None,
                      scope: Optional[Scope] = None, all_scopes: bool = False,
                      filters: Optional[List[MetadataFilter]] = None) -> List[RetrievalResult]:
        """Async version of search: aembed gets the query vector, then _afuse_one runs the vector path / keyword path concurrently via gather."""
        top_k = self.config.top_k if top_k is None else top_k
        candidate_pool = self.config.candidate_pool if candidate_pool is None else candidate_pool
        require_valid_top_k(top_k, candidate_pool=candidate_pool)
        require_explicit_scope(scope or Scope(), all_scopes, "search")
        query_vector = (await self.embedder.aembed([query]))[0]
        return await self._afuse_one(query, query_vector, top_k=top_k, candidate_pool=candidate_pool,
                                     scope=scope, filters=filters)

    async def asearch_many(self, queries: List[str], *, top_k: Optional[int] = None, candidate_pool: Optional[int] = None,
                           scope: Optional[Scope] = None, all_scopes: bool = False,
                           filters: Optional[List[MetadataFilter]] = None) -> List[List[RetrievalResult]]:
        """Async version of search_many: aembed all queries at once, then each query runs the two paths concurrently via _afuse_one."""
        top_k = self.config.top_k if top_k is None else top_k
        candidate_pool = self.config.candidate_pool if candidate_pool is None else candidate_pool
        require_valid_top_k(top_k, candidate_pool=candidate_pool)
        require_explicit_scope(scope or Scope(), all_scopes, "search")
        if not queries:
            return []
        query_vectors = await self.embedder.aembed(list(queries))
        return [await self._afuse_one(q, v, top_k=top_k, candidate_pool=candidate_pool, scope=scope, filters=filters)
                for q, v in zip(queries, query_vectors)]

    async def _afuse_one(self, query: str, query_vector: List[float], *, top_k: int,
                         candidate_pool: int, scope: Optional[Scope],
                         filters: Optional[List[MetadataFilter]] = None) -> List[RetrievalResult]:
        """Async version of _fuse_one: the vector path + keyword path run concurrently via the port's asearch (gather); fusion / refine are sync (CPU / possibly-async rerank)."""
        fkw = _maybe_filters(filters)
        vec_hits, kw_hits = await asyncio.gather(
            self.vector_store.asearch(query_vector, top_k=candidate_pool, scope=scope, **fkw),
            self.keyword_index.asearch(query, top_k=candidate_pool, scope=scope, **fkw))
        if self.reranker is None:
            return self.fusion.fuse([vec_hits, kw_hits], top_k=top_k)
        fused = self.fusion.fuse([vec_hits, kw_hits], top_k=candidate_pool)
        return await self.reranker.arerank(query, fused, top_k=top_k)

    async def aadd(self, ids: List[str], contents: List[str], **kwargs) -> None:
        """Async version of add (wraps its own sync atomic version in to_thread; the write path is not hot and preserves cross-index single-transaction semantics)."""
        await asyncio.to_thread(lambda: self.add(ids, contents, **kwargs))

    async def areplace(self, old_ids: List[str], new_ids: List[str], contents: List[str], **kwargs) -> None:
        """Async version of replace (wraps the sync atomic version in to_thread)."""
        await asyncio.to_thread(lambda: self.replace(old_ids, new_ids, contents, **kwargs))

    async def adelete(self, ids: List[str], **kwargs) -> None:
        """Async version of delete (to_thread)."""
        await asyncio.to_thread(lambda: self.delete(ids, **kwargs))
