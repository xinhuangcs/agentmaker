"""agentmaker.retrieval: shared retrieval foundation for memory and rag (hybrid retrieval + rerank, isolated by scope).

Layering (ports & adapters): the pure framework (interfaces + orchestration + contracts) is kept separate from
the swappable batteries (backends/):
    - types / scope   : RetrievalResult unified result + MetadataFilter filter contract; Scope multi-dimensional
                        ownership concept + guardrails (scope_sql holds the Scope->SQL mapping, shared across agentmaker)
    - base            : the abstract ports Embedder / VectorStore / KeywordIndex / Reranker
    - hybrid          : storage-agnostic orchestrator HybridRetriever + default fusion battery RRFFusion / reciprocal_rank_fusion
    - index_sync      : source-of-truth -> derived index sync seam IndexSync / SyncIndexSync (shared by memory and rag)
    - backends/       : swappable adapters (batteries): openai_embedder / cohere_reranker / sqlite; changing backend
                        or model touches only this layer
"""

from ..core.exceptions import RetrievalError
from .backends import (CohereReranker, Fts5KeywordIndex, OpenAIEmbedder, SqliteHybridRetriever,
                       SqliteVecStore, build_sqlite_hybrid)
from .base import Embedder, FusionStrategy, KeywordIndex, Reranker, VectorStore
from .hybrid import HybridRetriever, RRFFusion, reciprocal_rank_fusion, require_valid_top_k
from .index_sync import IndexSync, InMemoryBookkeeping, SqliteBookkeeping, SyncBookkeeping, SyncIndexSync
from .scope import Scope, require_explicit_scope, scope_is_empty
from .scope_sql import (scope_column_names, scope_exact_where, scope_exact_where_clause,
                        scope_store_values, scope_where, scope_where_clause)
from .types import MetadataFilter, RetrievalConfig, RetrievalResult

__all__ = ["RetrievalResult", "RetrievalConfig", "Scope", "Embedder", "OpenAIEmbedder", "VectorStore", "SqliteVecStore",
           "KeywordIndex", "Fts5KeywordIndex", "Reranker", "CohereReranker", "MetadataFilter",
           "FusionStrategy", "RRFFusion",
           "HybridRetriever", "reciprocal_rank_fusion", "build_sqlite_hybrid", "IndexSync", "SyncIndexSync",
           "SyncBookkeeping", "InMemoryBookkeeping", "SqliteBookkeeping",
           # Extension symbols for authors of custom retrieval backends (sub-package API, not promoted to the agentmaker top level)
           "SqliteHybridRetriever", "require_valid_top_k", "require_explicit_scope", "scope_is_empty", "RetrievalError",
           "scope_column_names", "scope_store_values", "scope_where", "scope_where_clause",
           "scope_exact_where", "scope_exact_where_clause"]
