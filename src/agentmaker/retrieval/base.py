"""agentmaker.retrieval.base: the five abstract ports of the retrieval foundation.

Defines one abstract base class each for "text to vector / vector storage / keyword index / rerank / multi-way fusion";
concrete implementations fill in the interfaces. Swapping a backend means writing a new subclass, leaving the
orchestration layer (HybridRetriever) and everything above it untouched. This module holds interfaces only, no
implementations: the local SQLite backend lives in `sqlite.py`, the OpenAI / Cohere vendor implementations in
`embedder.py` / `reranker.py`, and the default fusion battery RRFFusion in `hybrid.py`.

metadata filtering (pre-filtering) runs through both storage ports: add can carry per-item metadata (whatever
filterable fields were declared when building the index get stored), and search can carry a list of `MetadataFilter`
to narrow candidates. See the contract in `types.py`.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

from .scope import Scope
from .types import MetadataFilter, RetrievalResult

# Each of the four I/O ports (Embedder / VectorStore / KeywordIndex / Reranker) gets a default a* method: the base
# class wraps the sync version with to_thread (embedding / DB / rerank are all network or disk IO, so paying one
# thread hop is reasonable), so sync backends get async for free; natively-async backends such as httpx can override
# with a real async implementation. FusionStrategy.fuse is pure CPU with no IO, so it has no a* pair. HybridRetriever's
# a* orchestration goes through these port a* methods for true async.


class Embedder(ABC):
    """Abstract base class for turning text into vectors. Subclasses implement embed() and dim."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Turn a batch of texts into a batch of vectors (batched to save round trips).

        Args:
            texts: List of texts.

        Returns:
            List[List[float]]: Vectors, same length and order as the input.
        """

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed (defaults to to_thread)."""
        return await asyncio.to_thread(self.embed, texts)

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimension. Used to declare the column width float[dim] when building the vector store."""

    @property
    def model_id(self) -> Optional[str]:
        """Model identifier (e.g. "text-embedding-3-small"), used for the index<->embedder fingerprint check. Vectors
        from different models live in incomparable spaces; swapping to "a different model of the same dimension"
        without a fingerprint would silently mix vectors and quietly degrade retrieval. Defaults to None (unknown, in
        which case the check compares dimension only); subclasses should override.
        """
        return None


class VectorStore(ABC):
    """Abstract base class for vector storage + nearest-neighbor retrieval.

    scope isolates different data (memory / rag, different user / agent, etc.): one shared foundation, mutually
    non-interfering. metadata filtering: the implementation decides which fields are filterable via its own declaration
    mechanism (the SQLite battery uses the constructor parameter `metadata_columns=`); add stores the declared fields,
    search narrows by filters. Filtering an undeclared field should fail loud.
    """

    @abstractmethod
    def add(self, ids: List[str], vectors: List[List[float]], contents: List[str],
            *, scope: Optional[Scope] = None, metadatas: Optional[List[dict]] = None) -> None:
        """Batch write: each item = id + vector + original text (+ optional metadata). All lists are equal length and aligned.

        Args:
            ids: Unique identifier per item.
            vectors: Vector per item; dimension must match the one used to build the store.
            contents: Original text per item.
            scope: Ownership label; defaults to Scope() (all dimensions empty, i.e. unrestricted, B semantics). Writing
                the same (id, scope) again is an upsert (overwrite).
            metadatas: Optional, one metadata dict per item; only fields declared when building the index are stored as
                filterable columns, the rest are ignored (the full metadata still lives in each subsystem's
                source-of-truth store). Not passing it = all declared columns stored empty.
        """

    @abstractmethod
    def search(self, query_vector: List[float], *, top_k: int = 5, scope: Optional[Scope] = None,
               filters: Optional[List[MetadataFilter]] = None) -> List[RetrievalResult]:
        """Given a query vector, return the top_k nearest items within the range limited by scope (+ optional filters).

        Args:
            query_vector: The query vector.
            top_k: Number of items to return.
            scope: Ownership filter; defaults to Scope() (B semantics: only filters non-empty dimensions).
            filters: Optional metadata filter conditions (AND semantics, see MetadataFilter in types.py); filtering an
                undeclared field raises RetrievalError.

        Returns:
            List[RetrievalResult]: Sorted by relevance, highest first.
        """

    @abstractmethod
    def delete(self, ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Batch delete by id (within the range limited by scope, B semantics)."""

    def delete_exact(self, ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Delete by exact write footprint (all-dimension match, deleting only rows whose footprint matches exactly,
        leaving sibling rows with the same id but different scope untouched). Defaults to falling back to delete (B
        semantics, backward compatible, does not break existing backends); backends that are exact across all
        dimensions (such as SqliteVecStore) should override. Used by HybridRetriever.add's compensating path: when
        rolling back a just-written vector, the delete range must equal the write footprint, otherwise it would
        wrongly delete same-id rows under other scopes."""
        self.delete(ids, scope=scope)

    async def aadd(self, ids: List[str], vectors: List[List[float]], contents: List[str],
                   *, scope: Optional[Scope] = None, metadatas: Optional[List[dict]] = None) -> None:
        """Async version of add (defaults to to_thread)."""
        await asyncio.to_thread(lambda: self.add(ids, vectors, contents, scope=scope, metadatas=metadatas))

    async def asearch(self, query_vector: List[float], *, top_k: int = 5, scope: Optional[Scope] = None,
                      filters: Optional[List[MetadataFilter]] = None) -> List[RetrievalResult]:
        """Async version of search (defaults to to_thread)."""
        return await asyncio.to_thread(lambda: self.search(query_vector, top_k=top_k, scope=scope, filters=filters))

    async def adelete(self, ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Async version of delete (defaults to to_thread)."""
        await asyncio.to_thread(lambda: self.delete(ids, scope=scope))

    async def adelete_exact(self, ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Async version of delete_exact (defaults to to_thread)."""
        await asyncio.to_thread(lambda: self.delete_exact(ids, scope=scope))

    def close(self) -> None:
        """Release underlying resources (such as database connections). No-op by default; subclasses override as needed."""


class KeywordIndex(ABC):
    """Abstract base class for keyword retrieval. Subclasses implement add() and search(). metadata filtering follows the same convention as VectorStore."""

    @abstractmethod
    def add(self, ids: List[str], contents: List[str], *, scope: Optional[Scope] = None,
            metadatas: Optional[List[dict]] = None) -> None:
        """Batch write: each item = id + original text (+ optional metadata, same convention as VectorStore.add). Lists are equal length and aligned."""

    @abstractmethod
    def search(self, query: str, *, top_k: int = 5, scope: Optional[Scope] = None,
               filters: Optional[List[MetadataFilter]] = None) -> List[RetrievalResult]:
        """Keyword retrieval, returning the top_k most BM25-relevant items (highest first) within the range limited by scope (+ optional filters)."""

    @abstractmethod
    def delete(self, ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Batch delete by id (within the range limited by scope, B semantics)."""

    async def aadd(self, ids: List[str], contents: List[str], *, scope: Optional[Scope] = None,
                   metadatas: Optional[List[dict]] = None) -> None:
        """Async version of add (defaults to to_thread)."""
        await asyncio.to_thread(lambda: self.add(ids, contents, scope=scope, metadatas=metadatas))

    async def asearch(self, query: str, *, top_k: int = 5, scope: Optional[Scope] = None,
                      filters: Optional[List[MetadataFilter]] = None) -> List[RetrievalResult]:
        """Async version of search (defaults to to_thread)."""
        return await asyncio.to_thread(lambda: self.search(query, top_k=top_k, scope=scope, filters=filters))

    async def adelete(self, ids: List[str], *, scope: Optional[Scope] = None) -> None:
        """Async version of delete (defaults to to_thread)."""
        await asyncio.to_thread(lambda: self.delete(ids, scope=scope))

    def close(self) -> None:
        """Release underlying resources (such as database connections). No-op by default; subclasses override as needed."""


class Reranker(ABC):
    """Abstract base class for reranking. Subclasses implement rerank()."""

    @abstractmethod
    def rerank(self, query: str, results: List[RetrievalResult], *, top_k: int = 5) -> List[RetrievalResult]:
        """Precisely re-order the candidate results by relevance to query, returning the top_k most relevant (score is the rerank score).

        Args:
            query: The query text.
            results: The candidates to rerank (typically from hybrid retrieval).
            top_k: Number of items to return.

        Returns:
            List[RetrievalResult]: The top_k items after reranking.
        """

    async def arerank(self, query: str, results: List[RetrievalResult], *, top_k: int = 5) -> List[RetrievalResult]:
        """Async version of rerank (defaults to to_thread)."""
        return await asyncio.to_thread(lambda: self.rerank(query, results, top_k=top_k))


class FusionStrategy(ABC):
    """Abstract base class for the strategy that fuses multi-way retrieval results (the fifth port).

    Fuses several result lists (each already sorted by relevance) into a single ranking. The default battery RRFFusion
    (reciprocal-rank scoring, tuning-free) lives in hybrid.py; when you have an eval set and need to tune the relative
    weights of the two paths, implement your own (e.g. alpha-weighted / RSF normalized weighting) and inject it via
    HybridRetriever(fusion=).
    """

    @abstractmethod
    def fuse(self, result_lists: List[List[RetrievalResult]], *, top_k: int) -> List[RetrievalResult]:
        """Fuse multiple result lists, returning the top_k items.

        Convention: HybridRetriever passes [dense path, keyword path]; rag query expansion (MQE/HyDE) passes one list
        per query. Fusion aligns by id (same id counts as the same item).

        Args:
            result_lists: Multiple result lists, each already sorted by relevance highest first.
            top_k: Number of items to return after fusion.

        Returns:
            List[RetrievalResult]: The top_k items after fusion and re-ranking.
        """
