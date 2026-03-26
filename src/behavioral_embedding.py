# src/behavioral_embedding.py
"""Behavioral embedding via weighted aggregation of precomputed dish embeddings.

Instead of encoding a text description of user behavior through a
sentence-transformer (e.g. "STRONGLY PREFERS Grilled Chicken"), we aggregate
the actual dish embeddings directly.  The resulting vector already lives in
the dish embedding space, making cosine-similarity retrieval more accurate.

This eliminates the *semantic gap*: the text "STRONGLY PREFERS X" lands in a
different region of embedding space than the embedding of dish X itself.

Reference: see Notebook 3, Experiment 5 (Behavioral Embedding from Dish Vectors).
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np


# Category weights — how much each interaction type contributes.
# Orders carry the strongest signal (explicit purchase intent).
_CATEGORY_WEIGHTS: dict[str, float] = {
    "orders": 0.55,
    "high_ratings": 0.20,
    "favorites": 0.15,
    "views": 0.10,
}

# Datetime field name per category
_DATE_FIELDS: dict[str, str] = {
    "orders": "ordered_at",
    "high_ratings": "rated_at",
    "favorites": "added_at",
    "views": "viewed_at",
}


def build_behavioral_embedding_from_dishes(
    dish_embeddings: dict[int, np.ndarray],
    recent_orders: list[dict] | None = None,
    high_ratings: list[dict] | None = None,
    favorites: list[dict] | None = None,
    recent_views: list[dict] | None = None,
    order_counts: dict[int, int] | None = None,
    decay_rate: float = 0.03,
) -> np.ndarray | None:
    """Build a behavioral embedding as a weighted sum of dish embeddings.

    Weight scheme per category (mirrors ``_CATEGORY_WEIGHTS``):
      - orders: 0.55
      - high_ratings: 0.20
      - favorites: 0.15
      - views: 0.10

    Within each category every event's contribution is scaled by temporal decay:
      ``exp(-decay_rate * days_since_event)``

    For orders an additional frequency boost is applied:
      ``min(order_count, 5) / 5``

    Args:
        dish_embeddings: ``{dish_id: np.ndarray}`` of shape ``(dim,)``.
            In notebooks, built from ``embeddings[recipe_id_to_idx[dish_id]]``.
        recent_orders: ``[{"dish_id": int, "ordered_at": datetime}, ...]``
        high_ratings: ``[{"dish_id": int, "rated_at": datetime, "rating": int}, ...]``
        favorites: ``[{"dish_id": int, "added_at": datetime}, ...]``
        recent_views: ``[{"dish_id": int, "viewed_at": datetime}, ...]``
        order_counts: ``{dish_id: int}`` — total order count per dish.
        decay_rate: Exponential decay constant. ``0.03`` ≈ half-weight at 23 days.

    Returns:
        L2-normalized ``np.ndarray`` of shape ``(dim,)`` or ``None`` if no
        matching dish embeddings were found.

    Example:
        >>> import numpy as np
        >>> embs = {42: np.random.randn(768).astype(np.float32)}
        >>> embs[42] /= np.linalg.norm(embs[42])
        >>> beh = build_behavioral_embedding_from_dishes(
        ...     embs,
        ...     recent_orders=[{"dish_id": 42, "ordered_at": datetime.now(timezone.utc)}],
        ... )
        >>> beh.shape
        (768,)
    """
    categories: dict[str, list[dict] | None] = {
        "orders": recent_orders,
        "high_ratings": high_ratings,
        "favorites": favorites,
        "views": recent_views,
    }

    now = datetime.now(timezone.utc)
    category_vectors: dict[str, np.ndarray] = {}

    for cat_name, events in categories.items():
        if not events:
            continue

        date_field = _DATE_FIELDS[cat_name]
        weighted_sum: np.ndarray | None = None
        total_weight = 0.0

        for event in events:
            dish_id = event.get("dish_id") or event.get("recipe_id")
            if dish_id is None or dish_id not in dish_embeddings:
                continue

            emb = dish_embeddings[dish_id]

            # Temporal decay
            dt = event.get(date_field)
            if dt is not None:
                if isinstance(dt, str):
                    dt = datetime.fromisoformat(dt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                days_since = max((now - dt).total_seconds() / 86400.0, 0.0)
            else:
                days_since = 0.0
            w = math.exp(-decay_rate * days_since)

            # Frequency boost for orders
            if cat_name == "orders" and order_counts:
                count = order_counts.get(dish_id, 1)
                w *= min(count, 5) / 5.0

            weighted_sum = w * emb if weighted_sum is None else weighted_sum + w * emb
            total_weight += w

        if weighted_sum is not None and total_weight > 0:
            category_vectors[cat_name] = weighted_sum / total_weight

    if not category_vectors:
        return None

    dim = next(iter(category_vectors.values())).shape[0]
    combined = np.zeros(dim, dtype=np.float32)
    weight_sum = 0.0

    for cat_name, vec in category_vectors.items():
        w = _CATEGORY_WEIGHTS[cat_name]
        combined += w * vec
        weight_sum += w

    if weight_sum > 0:
        combined /= weight_sum

    norm = np.linalg.norm(combined)
    return combined / norm if norm > 0 else combined
