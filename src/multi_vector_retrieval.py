# src/multi_vector_retrieval.py
"""Multi-vector retrieval: search with multiple query vectors and merge results.

Motivation: a single user embedding is a compressed representation that may
lose fine-grained preference signals.  Instead we build separate query vectors
for different user facets (taste, nutritional need, meal context) and merge
the result lists.  This is Experiment 6 in Notebook 3.

Merge strategies:
- ``"rrf"``      — Reciprocal Rank Fusion (robust, parameter-free)
- ``"union"``    — Union with max-score deduplication
- ``"weighted"`` — Weighted score combination (requires query weights)
"""
from __future__ import annotations

import numpy as np


def build_retrieval_queries(
    query_embeddings: dict[str, np.ndarray],
    strategy: str = "multi",
) -> list[np.ndarray]:
    """Return query vectors to use for retrieval.

    Args:
        query_embeddings: Named embeddings, e.g.
            ``{"taste": emb_taste, "nutrition": emb_nutrition}``.
        strategy: ``"single"`` uses only the first query (baseline),
            ``"multi"`` uses all queries.

    Returns:
        List of query vectors to search with.
    """
    vectors = list(query_embeddings.values())
    if strategy == "single" or len(vectors) == 0:
        return [vectors[0]] if vectors else []
    return vectors


def multi_vector_search(
    queries: list[np.ndarray],
    index,  # faiss.Index
    top_k: int = 50,
    strategy: str = "multi",
    merge: str = "rrf",
    query_weights: list[float] | None = None,
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    """Search a FAISS index with one or more query vectors, then merge results.

    Args:
        queries: List of L2-normalized query vectors (shape ``(dim,)`` each).
        index: A ``faiss.IndexFlatIP`` (or similar) index.
        top_k: Number of final results to return.
        strategy: ``"single"`` — use only ``queries[0]`` (baseline for ablation);
            ``"multi"`` — use all queries.
        merge: How to merge multiple result lists:
            - ``"rrf"``      — Reciprocal Rank Fusion
            - ``"union"``    — Max-score deduplication
            - ``"weighted"`` — Weighted score sum (needs ``query_weights``)
        query_weights: Weights per query for ``"weighted"`` merge.
            Must sum to 1. Defaults to uniform.
        rrf_k: Smoothing constant for RRF (default 60 is standard).

    Returns:
        List of ``(item_id, score)`` sorted by score descending, length ≤ ``top_k``.

    Example:
        >>> import faiss, numpy as np
        >>> dim = 768
        >>> index = faiss.IndexFlatIP(dim)
        >>> vecs = np.random.randn(100, dim).astype(np.float32)
        >>> faiss.normalize_L2(vecs)
        >>> index.add(vecs)
        >>> q = np.random.randn(1, dim).astype(np.float32)
        >>> faiss.normalize_L2(q)
        >>> results = multi_vector_search([q[0]], index, top_k=5, strategy="single")
        >>> len(results)
        5
    """
    if strategy == "single" or len(queries) == 1:
        q = queries[0].reshape(1, -1).astype(np.float32)
        scores, indices = index.search(q, top_k)
        return [(int(idx), float(sc)) for idx, sc in zip(indices[0], scores[0]) if idx >= 0]

    # Multi-query: gather per-query result lists
    per_query_results: list[list[tuple[int, float]]] = []
    search_k = top_k * 3  # retrieve more per query to allow for overlap
    for query in queries:
        q = query.reshape(1, -1).astype(np.float32)
        scores, indices = index.search(q, search_k)
        results = [(int(idx), float(sc)) for idx, sc in zip(indices[0], scores[0]) if idx >= 0]
        per_query_results.append(results)

    if merge == "rrf":
        merged = _rrf_merge(per_query_results, rrf_k=rrf_k)
    elif merge == "weighted":
        weights = query_weights or [1.0 / len(queries)] * len(queries)
        merged = _weighted_merge(per_query_results, weights)
    else:  # union
        merged = _union_merge(per_query_results)

    return merged[:top_k]


def _rrf_merge(
    result_lists: list[list[tuple[int, float]]],
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion.

    Each item's score = sum over lists of 1 / (rank + rrf_k).
    Robust to score scale differences between queries.
    """
    scores: dict[int, float] = {}
    for results in result_lists:
        for rank, (item_id, _) in enumerate(results):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rank + rrf_k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _union_merge(
    result_lists: list[list[tuple[int, float]]],
) -> list[tuple[int, float]]:
    """Union with max-score deduplication."""
    best: dict[int, float] = {}
    for results in result_lists:
        for item_id, score in results:
            if score > best.get(item_id, -1.0):
                best[item_id] = score
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def _weighted_merge(
    result_lists: list[list[tuple[int, float]]],
    weights: list[float],
) -> list[tuple[int, float]]:
    """Weighted score combination."""
    combined: dict[int, float] = {}
    for results, w in zip(result_lists, weights):
        for item_id, score in results:
            combined[item_id] = combined.get(item_id, 0.0) + w * score
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
