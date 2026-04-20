"""Hybrid lexical (BM25) + dense (FAISS) retrieval merged via Reciprocal Rank Fusion (RRF).

This is intentionally lightweight and dependency-driven:
- BM25 is provided by ``rank_bm25`` (install: ``pip install rank-bm25``).
- Dense retrieval assumes a FAISS ``IndexFlatIP`` with **L2-normalized** vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from multi_vector_retrieval import _rrf_merge


def _try_import_bm25():
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing dependency for BM25. Install with: pip install rank-bm25"
        ) from e
    return BM25Okapi


def tokenize(s: str) -> list[str]:
    """Very simple tokenizer: good enough for ingredient-ish tokens on English-ish text."""
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if t]


@dataclass(frozen=True)
class HybridRRFConfig:
    """Hyperparameters for hybrid retrieval."""

    # How many candidates to pull from each channel before fusion
    dense_k: int = 200
    bm25_k: int = 200
    # RRF smoothing constant (common default: 60)
    rrf_k: int = 60


def build_bm25_index(documents: Sequence[str]):
    """Return a BM25Okapi index over tokenized documents."""
    BM25Okapi = _try_import_bm25()
    tokenized_corpus = [tokenize(d) for d in documents]
    return BM25Okapi(tokenized_corpus)


def bm25_topk(bm25, query_text: str, top_k: int) -> list[tuple[int, float]]:
    """Return (doc_index, bm25_score) sorted by score desc."""
    scores = bm25.get_scores(tokenize(query_text))
    if len(scores) == 0:
        return []
    top_k = min(int(top_k), len(scores))
    idx = np.argpartition(-scores, top_k - 1)[:top_k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


def dense_topk_faiss(index, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Return (faiss_row_index, inner_product) sorted by score desc."""
    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    scores, indices = index.search(q, int(top_k))
    out: list[tuple[int, float]] = []
    for idx, sc in zip(indices[0], scores[0]):
        if int(idx) < 0:
            continue
        out.append((int(idx), float(sc)))
    return out


def hybrid_rrf_fuse_lists(
    lists: Iterable[list[tuple[int, float]]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse multiple ranked lists with RRF; returns top_k by fused score."""
    merged = _rrf_merge(list(lists), rrf_k=int(rrf_k))
    return merged[: int(top_k)]


def hybrid_search_dish_ids(
    *,
    query_text: str,
    query_vec: np.ndarray,
    bm25_index,
    faiss_index,
    cfg: HybridRRFConfig | None = None,
    final_top_k: int = 20,
) -> list[tuple[int, float]]:
    """Hybrid search returning FAISS row indices (same IDs as dense-only path)."""
    cfg = cfg or HybridRRFConfig()

    dense_list = dense_topk_faiss(faiss_index, query_vec, cfg.dense_k)
    bm25_list = bm25_topk(bm25_index, query_text, cfg.bm25_k)

    # `_rrf_merge` expects arbitrary float scores; ranks are what matter.
    fused = hybrid_rrf_fuse_lists([dense_list, bm25_list], top_k=final_top_k, rrf_k=cfg.rrf_k)
    return fused


def dish_text_for_bm25(row_dict: dict, tags) -> str:
    """Build a BM25-friendly document without touching ``text_builders.py`` flags contract.

    We intentionally bias BM25 toward lexical overlap on:
    - dish name
    - ingredient list (structured or extracted elsewhere)
    - short description
    """
    from text_builders import extract_ingredients

    name = str(row_dict.get("name") or "").strip()
    desc = str(row_dict.get("description") or row_dict.get("short_description") or "").strip()
    recipe = str(row_dict.get("recipe_text") or row_dict.get("instructions") or "").strip()

    ingredients = row_dict.get("ingredients")
    ing_parts: list[str] = []
    if ingredients is not None:
        seq = ingredients.tolist() if hasattr(ingredients, "tolist") else list(ingredients)
        ing_parts = [str(x).strip() for x in seq if str(x).strip()]

    if not ing_parts and recipe:
        ing_parts = extract_ingredients(recipe)

    tag_list = []
    if tags is not None:
        seq = tags.tolist() if hasattr(tags, "tolist") else list(tags)
        tag_list = [str(t) for t in seq if str(t).strip()]

    parts = [p for p in (name, desc, ", ".join(ing_parts), " ".join(tag_list)) if p]
    return "\n".join(parts)
