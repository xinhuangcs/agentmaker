"""agentmaker.retrieval.backends.cohere_reranker: Cohere implementation of cross-encoder reranking.

The candidates from hybrid retrieval are only coarsely ordered (RRF scores bunch up together). Reranking uses a
cross-encoder model that "reads the query and document together" to score each candidate precisely and reorder them,
keeping only the top few most relevant. This directly determines the quality of the top items fed to the LLM.
The abstract `Reranker` interface lives one level up in `../base.py`; this is the Cohere implementation (multilingual,
including Chinese), which can later be swapped for a local bge-reranker on a server.
"""

import os
from typing import Optional

from ...core.exceptions import RetrievalError
from ..base import Reranker
from ..hybrid import require_valid_top_k
from ..types import RetrievalResult


class CohereReranker(Reranker):
    """Cross-encoder reranking backed by the Cohere Rerank API (multilingual, including Chinese)."""

    def __init__(self, model: str = "rerank-v4.0-fast", api_key: Optional[str] = None, *, timeout: float = 30.0):
        """
        Args:
            model: Cohere rerank model name, defaults to rerank-v4.0-fast (multilingual / cost-effective); use
                rerank-v4.0-pro for highest quality.
            api_key: API key; if omitted, read from the COHERE_API_KEY environment variable (note the Cohere SDK
                defaults to looking for CO_API_KEY, so we read it explicitly here).
            timeout: Timeout in seconds.
        """
        self.model = model
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise RetrievalError("COHERE_API_KEY not found; pass api_key or configure it in .env.")
        self._client = None

    def _ensure_client(self):
        """Lazily create the cohere client (built only on first call)."""
        if self._client is None:
            try:
                import cohere
            except ImportError as e:
                raise RetrievalError("Cohere reranking requires installing: uv add cohere") from e
            self._client = cohere.ClientV2(api_key=self.api_key, timeout=self.timeout)
        return self._client

    def rerank(self, query, results, *, top_k=5):
        """Hand the candidate texts to Cohere for precise reranking; map the returned index back to the original RetrievalResult and swap in the [0,1] relevance_score."""
        require_valid_top_k(top_k)  # guard top_k<1 even on direct calls to this class (calls via HybridRetriever are pre-validated)
        if not results:
            return []
        client = self._ensure_client()
        documents = [r.content for r in results]
        try:
            resp = client.rerank(model=self.model, query=query, documents=documents, top_n=top_k)
        except Exception as e:  # noqa: BLE001
            raise RetrievalError(f"Cohere rerank failed (model={self.model}): {e}") from e
        out = []
        for item in resp.results:
            base = results[item.index]  # index points back into the input documents (i.e. results)
            out.append(RetrievalResult(content=base.content, score=item.relevance_score,
                                       source=base.source, id=base.id, embedding=base.embedding,
                                       metadata={**base.metadata, "rerank_score": item.relevance_score}))
        return out


