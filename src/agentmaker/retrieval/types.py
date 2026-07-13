"""agentmaker.retrieval.types: the unified contract types for retrieval.

Every "query" operation in memory and rag ultimately returns a batch of RetrievalResult; the upper layers (context
engineering / Agent) face only this one shape and do not care whether the results came from memory or documents. This
is the unified contract that wires the retrieval capability into the whole framework.

Also contains the metadata filtering contract MetadataFilter (pre-filtering: narrow candidates by metadata first, then
compute similarity, which industry vector stores pgvector / Qdrant / Pinecone all support natively): the upper layers
only assemble it, and translating it into a SQL WHERE / payload filter is each backend's job (backends/), the same
division of labor as Scope->scope_sql (concept on top, dialect underneath). The only operators today are eq / in, and
multiple conditions are always AND'd; values are compared as text (the SQLite battery also stores them as text on write).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.exceptions import RetrievalError


@dataclass(frozen=True)
class RetrievalConfig:
    """Tunable knobs for the hybrid retrieval foundation (shared by memory / rag; inject a separate one per subsystem if you want them heterogeneous). All frozen; scalars are inherently immutable.

    Fields:
        top_k: Final number of items to return.
        candidate_pool: How many items each path (dense / keyword) fetches into fusion / rerank (must be >= top_k).
        rrf_k: The smoothing constant for RRF fusion.
    """
    top_k: int = 5
    candidate_pool: int = 20
    rrf_k: int = 60

    def validate(self) -> None:
        """Range check: top_k >= 1, candidate_pool >= top_k, rrf_k >= 1."""
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.candidate_pool < self.top_k:
            raise ValueError(f"candidate_pool ({self.candidate_pool}) must be >= top_k ({self.top_k})")
        if self.rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {self.rrf_k}")


@dataclass
class RetrievalResult:
    """One retrieval result: the hit content + relevance score + source + identifier + metadata.

    Fields:
        content: The hit text body.
        score: Relevance score, by convention "higher is more relevant"; the actual value is produced by each retrieval
            implementation, used only for ordering, and not guaranteed comparable across implementations.
        source: Source identifier, such as "memory" / "rag" / a document name, to help the upper layers distinguish and trace.
        id: The item's unique identifier within its source, usable for going back to the original text / dedup; empty string if absent.
        embedding: The item's content vector; vector retrieval can carry it back for free (vec0's vec_to_json), for
            reuse by context engineering's MMR to avoid recomputation.
        metadata: Attached information (time, type, raw distance, etc.), defaults to an empty dict.
    """
    content: str
    score: float
    source: str
    id: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Make print(result) show "[score] body (truncated if too long)", for easy debugging.

        Returns:
            str: Something like "[0.812] I can't eat nuts".
        """
        text = self.content if len(self.content) <= 40 else self.content[:40] + "…"
        return f"[{self.score:.3f}] {text}"


# Supported operators; when adding one (e.g. gt / lt / contains), each backend's compiler must implement it in sync.
_OPS = ("eq", "in")


@dataclass(frozen=True)
class MetadataFilter:
    """One metadata filter condition: field key, comparison value, and operator op (eq for equality / in for one-of-many).

    Multiple conditions are AND'd (a hit must satisfy all of them). Filterable fields must be declared when building the
    index (the SQLite battery uses `metadata_columns=`); filtering an undeclared field fails loud, since a silent empty
    result is harder to diagnose than an error.

    Example:
        MetadataFilter("doc_id", "abc123")                       -> doc_id = 'abc123'
        MetadataFilter("tag", ["faq", "policy"], op="in")        -> tag IN ('faq', 'policy')
    """
    key: str
    value: Any
    op: str = "eq"

    def __post_init__(self):
        if not self.key or not isinstance(self.key, str):
            raise RetrievalError(f"MetadataFilter.key must be a non-empty string, got {self.key!r}.")
        if self.op not in _OPS:
            raise RetrievalError(f"MetadataFilter.op only supports {_OPS}, got {self.op!r}.")
        if self.op == "eq" and self.value is None:
            # In SQL, `col = NULL` is always false and silently returns zero hits (rather than matching empty values):
            # fail loud, so a failed match is not mistaken for missing data.
            raise RetrievalError("value for op='eq' cannot be None (in SQL, = NULL is always false and silently returns zero hits).")
        if self.op == "in":
            if not isinstance(self.value, (list, tuple, set)) or not self.value:
                raise RetrievalError(f"value for op='in' must be a non-empty list / tuple / set, got {self.value!r}.")
