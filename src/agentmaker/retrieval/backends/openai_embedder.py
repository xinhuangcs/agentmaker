"""agentmaker.retrieval.backends.openai_embedder: OpenAI implementation of text embedding.

Turning text into vectors is the first step of "semantic retrieval": similar-meaning texts -> nearby vectors.
The abstract `Embedder` interface lives one level up in `../base.py`; this is an OpenAI-compatible implementation.
To switch models (e.g. a local BGE-M3 on a server later), just write another Embedder subclass.
"""

import os
from typing import List, Optional

from ...core.exceptions import RetrievalError
from ..base import Embedder

# Default dimensions for each OpenAI embedding model (verified against the 2026-05 official docs; changes as vendors
# update, so recheck periodically).
_OPENAI_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIEmbedder(Embedder):
    """OpenAI-compatible embedding implementation (defaults to text-embedding-3-small).

    Consistent with the LLM clients in agentmaker.core: reuses the openai package, builds the client lazily,
    and reads the OPENAI_* environment variables by default.
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, *, dimensions: Optional[int] = None, timeout: float = 30.0,
                 max_batch: int = 256):
        """
        Resolve key / base URL / dimensions and validate them; no network request here (the client is built on the
        first embed call).

        Args:
            model: Model name, defaults to text-embedding-3-small.
            api_key: API key; if omitted, read from the OPENAI_API_KEY environment variable.
            base_url: Service base URL; if omitted, read OPENAI_BASE_URL, then fall back to the official endpoint.
            dimensions: Optional, shortens the output dimension (supported by the OpenAI 3 series); if omitted,
                use the model's default dimension.
            timeout: Timeout in seconds.
            max_batch: Maximum number of texts embedded per API request; anything beyond is split into serial batches
                automatically (default 256; OpenAI has dual per-request limits, roughly 2048 texts + ~300k total
                tokens, so lower it when chunks are especially large).
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or _DEFAULT_OPENAI_BASE_URL
        self.timeout = timeout
        self._dimensions = dimensions
        self._dim = dimensions or _OPENAI_DIMS.get(model)
        if not self.api_key:
            raise RetrievalError("OPENAI_API_KEY not found; pass api_key or configure it in .env.")
        if not self._dim:
            raise RetrievalError(f"Unknown default dimension for model '{model}'; pass dimensions= explicitly.")
        if max_batch < 1:
            raise RetrievalError(f"max_batch must be >= 1, got {max_batch}")
        self.max_batch = max_batch
        self._client = None

    def _ensure_client(self):
        """Lazily create the openai client (built only on first call; construction makes no network request)."""
        if self._client is None:
            from openai import OpenAI  # core dependency of this project
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts; anything over max_batch is split into serial calls, then concatenated **in input order**, with exceptions normalized to RetrievalError.

        OpenAI embeddings have dual per-request limits (roughly 2048 texts + ~300k total tokens); sending several
        thousand chunks from a large document at once would fail the whole batch with a 400. So we split by max_batch
        (default 256), validating count / dimension independently per batch.

        Args:
            texts: List of texts (an empty list returns empty directly).

        Returns:
            List[List[float]]: Vectors strictly aligned to the input order.
        """
        if not texts:
            return []
        texts = list(texts)
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self.max_batch):   # slicing happens before the call; any sub-batch failure still raises RetrievalError (embed runs before writing to the store, so a failure leaves no half-write)
            vectors.extend(self._embed_batch(texts[i:i + self.max_batch]))
        return vectors

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """Embed a single sub-batch (<= max_batch texts): call the API, align by index, validate count / dimension; normalize exceptions to RetrievalError."""
        client = self._ensure_client()
        kwargs = {"model": self.model, "input": batch, "encoding_format": "float"}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        try:
            resp = client.embeddings.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            raise RetrievalError(f"embedding call failed (model={self.model}): {e}") from e
        # Sort by index to guarantee strict alignment with the input order (the API is usually already ordered; this is an extra safeguard).
        items = sorted(resp.data, key=lambda d: d.index)
        # Validate count and dimension: a count mismatch would let the upstream zip(ids, vectors) silently truncate and misalign id<->vector; a dimension mismatch would fail index creation / corrupt retrieval.
        if len(items) != len(batch):
            raise RetrievalError(f"embedding returned {len(items)} items, inconsistent with input {len(batch)} (model={self.model}).")
        vectors = [list(d.embedding) for d in items]
        for vec in vectors:
            if len(vec) != self._dim:
                raise RetrievalError(
                    f"embedding dimension {len(vec)} inconsistent with expected {self._dim} (model={self.model}).")
        return vectors

    @property
    def dim(self) -> int:
        """Vector dimension."""
        return self._dim

    @property
    def model_id(self) -> str:
        """Model identifier (goes into the index fingerprint, see base.Embedder.model_id): the model name; when the dimension is shortened explicitly, append the dimension suffix to distinguish it."""
        return f"{self.model}@{self._dimensions}" if self._dimensions is not None else self.model
