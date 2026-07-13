"""agentmaker.context.mmr: MMR (maximal marginal relevance) selection.

Picks a subset of retrieval candidates that is both relevant and non-redundant, the
core of de-duplication in context engineering. Stuffing everything into the window wastes
tokens and dilutes the signal (context rot); MMR selects one item at a time, and at each
step weighs both "how relevant it is" and "how similar it is to what is already selected",
automatically avoiding duplicates while preserving topical diversity.

Formula (each round takes the remaining candidate with the highest MMR score):
    MMR(c) = lambda * rel(c) - (1 - lambda) * max_{s in selected} sim(c, s)
             more relevant adds score      more similar to selected subtracts score
- rel(c): the candidate's own score (given by retrieval, not recomputed), normalized to
  0~1 before selection by dividing by the max (not min-max; see _normalize for the reason).
- sim(c, s): cosine similarity of two embeddings (reuses the vectors returned by retrieval,
  not recomputed).
- lambda: a 0~1 knob. 1 = pure relevance (no de-dup); lower emphasizes diversity more.
  Default 0.7 (retrieval is already ranked, so moderate de-dup is enough).
- Missing candidate embedding (keyword-only hit, vector not recalled) => similarity treated
  as 0: if redundancy cannot be judged, do not penalize diversity.
"""

import math
from typing import List, Optional, Sequence

from ..retrieval.types import RetrievalResult


def _cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    """Cosine similarity of two vectors (may fall outside 0~1, but lands in [-1,1] for normalized vectors); returns 0 if either is missing.

    Cosine = dot product / (norm a * norm b). Returns 0 when either vector is None, a zero
    vector, or the two vectors have mismatched dimensions (semantics: cannot judge
    similarity => do not treat as duplicate).
    """
    # Use `is None` / `len()==0` to test emptiness (not `not a`): numpy arrays raise an
    # "ambiguous" truth-value error, and this also covers lists.
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return 0.0
    if len(a) != len(b):
        return 0.0  # Mismatched dimensions (e.g. mixing different embedding models) => cannot judge similarity; do not silently truncate and compute a fake score
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _normalize(values: List[float]) -> List[float]:
    """Normalize a set of relevance scores to 0~1 by dividing by the max (highest => 1, rest proportional).

    For the common non-negative case, divide by the max (not min-max: min-max forces the
    lowest score down to 0, which erases the "least relevant but fully non-redundant"
    candidate and weakens MMR de-dup); dividing by the max preserves relative proportion,
    never zeroes out, and keeps the same scale as cosine.
    But when scores include negatives (e.g. some rerankers' logits), dividing by the max
    would flip / flatten the ordering (negative / negative), so switch to shifted min-max
    to preserve relative ranking.
    Returns all 1s (degenerating to diversity-only) when all values are equal (including
    all 0) or the input is empty.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)               # All equal (including all 0): no relevance distinction, diversity only
    if lo < 0:
        return [(v - lo) / (hi - lo) for v in values]   # Has negatives: shift into [0,1], preserve relative order
    return [v / hi for v in values]              # Common non-negative case: divide by max, do not crush the lowest score to 0


def _dedup_exact(candidates: List[RetrievalResult]) -> List[RetrievalResult]:
    """De-duplicate exactly by content: keep only the highest-scoring copy of identical text, output stably in first-appearance order.

    This covers the gap where, when embeddings are missing (similarity treated as 0,
    near-duplicates undetectable), byte-for-byte duplicates would otherwise both be kept;
    for duplicates that do have embeddings it is a harmless prefilter (saving one cosine).
    Keeping the highest score rather than the first occurrence: mmr_select is a public
    function and the MMR body does not require sorted input, so a caller may pass candidates
    not ordered by relevance; keeping the high-scoring copy avoids dropping the more relevant one.
    """
    best: dict = {}
    order: List[str] = []
    for c in candidates:
        if c.content not in best:
            best[c.content] = c
            order.append(c.content)
        elif c.score > best[c.content].score:
            best[c.content] = c
    return [best[k] for k in order]


def mmr_select(candidates: List[RetrievalResult], *, top_k: Optional[int] = None,
               lambda_: float = 0.7, dedup_threshold: float = 0.95) -> List[RetrievalResult]:
    """Select a subset from candidates by MMR: both relevant and non-redundant, dropping near-duplicates.

    Two layers of de-duplication:
    - dedup_threshold: a candidate whose similarity to any already-selected item is >= this
      value is treated as a "near-duplicate" and dropped outright (actually reducing count).
      0.95 means two items must be nearly byte-for-byte identical to count as duplicates;
      this is a "near-duplicate test", not a "relevance threshold", and is not sensitive to
      the exact value.
    - lambda_: among the non-duplicate candidates, orders by MMR balancing "relevant" against
      "diverse" (earlier-selected items rank higher).

    Args:
        candidates: Retrieval candidates (each with its own score and optional embedding;
            mixing candidates from different sources together enables cross-source de-dup).
        top_k: Maximum number to select; None = no cap on count (relies only on dedup to
            remove near-duplicates).
        lambda_: Trade-off between relevance and diversity, 0~1; 1 = purely by relevance.
        dedup_threshold: Near-duplicate removal threshold (cosine), default 0.95.

    Returns:
        List[RetrievalResult]: The selected subset, in selection order (earlier-selected first).
    """
    if top_k is not None and top_k < 0:
        raise ValueError(f"top_k must not be negative, got {top_k} (None = no cap on count).")
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError(f"lambda_ must be within [0, 1], got {lambda_} (1 = pure relevance, 0 = pure diversity).")
    if dedup_threshold < 0:
        raise ValueError(f"dedup_threshold must not be negative, got {dedup_threshold} (>1 disables near-duplicate removal).")
    if not candidates:
        return []
    candidates = _dedup_exact(candidates)  # Exact de-dup by content first, then embedding-based MMR
    limit = len(candidates) if top_k is None else min(top_k, len(candidates))
    rel = _normalize([c.score for c in candidates])  # Normalized relevance scores (divide by max), same scale as cosine
    selected_idx: List[int] = []
    remaining = list(range(len(candidates)))

    while remaining and len(selected_idx) < limit:
        kept_remaining = []
        for i in remaining:
            max_sim = max((_cosine(candidates[i].embedding, candidates[s].embedding)
                           for s in selected_idx), default=0.0)
            if max_sim < dedup_threshold:
                kept_remaining.append((i, max_sim))
        remaining = [i for i, _ in kept_remaining]
        if not remaining:
            break
        # Among the non-duplicate candidates, pick the one with the highest MMR score.
        best_i, best_score = remaining[0], -math.inf
        for i, max_sim in kept_remaining:
            mmr = lambda_ * rel[i] - (1.0 - lambda_) * max_sim
            if mmr > best_score:
                best_score, best_i = mmr, i
        selected_idx.append(best_i)
        remaining.remove(best_i)

    return [candidates[i] for i in selected_idx]
