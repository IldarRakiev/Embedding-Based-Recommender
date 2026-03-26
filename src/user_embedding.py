# src/user_embedding.py
"""Build weighted user embeddings from profile components.

Extracted from the production recommendation system. All database
dependencies removed — data is passed as plain Python dicts.

The key idea: instead of concatenating all user information into one long
text (which lets the model weight everything uniformly), we encode each
component separately and combine the embeddings with explicit weights.
This gives us direct control over the influence of each signal.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .embedding_model import EmbeddingModel


# Default weights — sum to 1.0
DEFAULT_WEIGHTS = {
    "static": 0.25,      # goal, allergies, medical, budget
    "dynamic": 0.15,     # remaining macros, meal time
    "behavioral": 0.25,  # order/view/rating history
    "query": 0.35,       # explicit user query (highest priority when present)
}

DEFAULT_WEIGHTS_NO_QUERY = {
    "static": 0.40,
    "dynamic": 0.25,
    "behavioral": 0.35,
}


def build_user_embedding(
    model: "EmbeddingModel",
    static_text: str,
    dynamic_text: str,
    behavioral_text: str,
    query: str | None = None,
    weights: dict[str, float] | None = None,
    behavioral_embedding: np.ndarray | None = None,
) -> np.ndarray:
    """Build a weighted user embedding from profile components.

    Each component is encoded separately and combined with explicit weights.
    Optionally, a pre-built behavioral embedding from dish vectors can be
    provided (see ``behavioral_embedding.py``) — this eliminates the semantic
    gap between the text-encoded behavioral description and dish embeddings.

    Args:
        model: EmbeddingModel instance.
        static_text: Static profile text (goal, allergies, budget).
        dynamic_text: Dynamic state text (remaining macros, meal time).
        behavioral_text: Behavioral profile text (orders, views, favorites).
        query: Optional explicit user query.
        weights: Custom weight dict. Defaults to ``DEFAULT_WEIGHTS`` or
            ``DEFAULT_WEIGHTS_NO_QUERY`` depending on whether query is provided.
        behavioral_embedding: Pre-built behavioral vector from dish embeddings.
            When provided, ``behavioral_text`` is NOT encoded — this vector
            is used directly. Pass ``None`` to fall back to text encoding.

    Returns:
        L2-normalized ``np.ndarray`` of shape ``(dim,)``.
    """
    w = (weights or (DEFAULT_WEIGHTS if query else DEFAULT_WEIGHTS_NO_QUERY)).copy()

    has_behavioral_vec = behavioral_embedding is not None
    has_behavioral_text = bool(
        behavioral_text and behavioral_text.strip() and behavioral_text != "USER BEHAVIOR:"
    )
    has_behavioral = has_behavioral_vec or has_behavioral_text

    # Cold start: no behavioral signal — redistribute weight
    if not has_behavioral:
        w["behavioral"] = 0
        if query and query.strip():
            w.update({"static": 0.50, "dynamic": 0.30, "query": 0.20})
        else:
            w.update({"static": 0.60, "dynamic": 0.40})

    texts_to_encode: list[str] = []
    text_weights: list[float] = []

    if static_text and static_text.strip():
        texts_to_encode.append(static_text)
        text_weights.append(w.get("static", 0.4))

    if dynamic_text and dynamic_text.strip():
        texts_to_encode.append(dynamic_text)
        text_weights.append(w.get("dynamic", 0.25))

    if has_behavioral_text and not has_behavioral_vec:
        texts_to_encode.append(behavioral_text)
        text_weights.append(w.get("behavioral", 0.25))

    if query and query.strip():
        texts_to_encode.append(query)
        text_weights.append(w.get("query", 0.35))

    if not texts_to_encode and not has_behavioral_vec:
        raise ValueError("At least one non-empty text component is required.")

    # Normalize weights
    behavioral_weight = w.get("behavioral", 0.25) if has_behavioral_vec else 0.0
    total = sum(text_weights) + behavioral_weight
    if total > 0:
        text_weights = [wt / total for wt in text_weights]
        behavioral_weight /= total

    # Encode in one batch
    if texts_to_encode:
        encoded = model.encode(texts_to_encode)
        dim = encoded.shape[1]
    else:
        dim = behavioral_embedding.shape[0]  # type: ignore[union-attr]

    user_emb = np.zeros(dim, dtype=np.float32)
    if texts_to_encode:
        for i, wt in enumerate(text_weights):
            user_emb += wt * encoded[i]
    if has_behavioral_vec:
        user_emb += behavioral_weight * behavioral_embedding  # type: ignore[arg-type]

    norm = np.linalg.norm(user_emb)
    return user_emb / norm if norm > 0 else user_emb
