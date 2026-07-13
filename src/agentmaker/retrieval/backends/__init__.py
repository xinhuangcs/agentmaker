"""agentmaker.retrieval.backends: swappable "batteries", the default concrete implementations of the four retrieval interfaces.

`base.py` defines the interfaces and `hybrid.py` orchestrates them (both backend-agnostic); this package holds the
out-of-the-box default adapters. To swap the database / model, write another file in this directory implementing the
interfaces in `base.py`; the framework core (base / hybrid / types / scope) stays untouched:
    - openai_embedder: OpenAIEmbedder, text -> vector (OpenAI-compatible, can point at DeepSeek etc. via base_url)
    - cohere_reranker: CohereReranker, cross-encoder rerank (optional)
    - sqlite:          full local SQLite backend (vector vec0 + keyword FTS5 + shared-connection atomic orchestration)
"""

from .cohere_reranker import CohereReranker
from .openai_embedder import OpenAIEmbedder
from .sqlite import Fts5KeywordIndex, SqliteHybridRetriever, SqliteVecStore, build_sqlite_hybrid

__all__ = ["OpenAIEmbedder", "CohereReranker", "SqliteVecStore", "Fts5KeywordIndex",
           "SqliteHybridRetriever", "build_sqlite_hybrid"]
