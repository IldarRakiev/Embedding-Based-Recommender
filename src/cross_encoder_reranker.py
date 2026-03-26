# src/cross_encoder_reranker.py
"""Cross-encoder reranker for candidate dishes.

Bi-encoder (sentence-transformer) retrieval is fast but compresses both
query and document into fixed-size vectors independently.  A cross-encoder
sees the (query, document) pair jointly, allowing richer token-level
interactions — at the cost of higher latency.

Typical pipeline:
    1. Bi-encoder retrieves top-50 candidates  (fast)
    2. Cross-encoder reranks top-50 → top-10   (accurate)

This is Experiment 7 in Notebook 3.

Graceful degradation: if the cross-encoder model is unavailable (network
error or Colab CPU-only session), the class falls back to the original
bi-encoder scores transparently.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Rerank retrieval candidates using a cross-encoder model.

    Args:
        model_name: HuggingFace cross-encoder model identifier.
            Default is a small, fast model suitable for free Colab CPU.
        max_length: Maximum token length for the cross-encoder.

    Example:
        >>> reranker = CrossEncoderReranker()
        >>> candidates = [(1, "Grilled chicken breast"), (2, "Chocolate cake")]
        >>> results = reranker.rerank("high protein dinner", candidates, top_k=1)
        >>> results[0][0]  # dish_id with highest cross-encoder score
        1
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 256,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model = None  # lazy loading

    def _load(self) -> bool:
        """Lazy-load the cross-encoder model. Returns True on success."""
        if self._model is not None:
            return True
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            self._model = CrossEncoder(self.model_name, max_length=self.max_length)
            logger.info(f"CrossEncoder loaded: {self.model_name}")
            return True
        except Exception as e:
            logger.warning(f"CrossEncoder unavailable ({e}). Falling back to bi-encoder scores.")
            return False

    def rerank(
        self,
        query: str,
        candidates: list[tuple[int, str]],
        top_k: int = 10,
        fallback_scores: list[float] | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank candidates by cross-encoder score.

        Args:
            query: The user query or user text used for retrieval.
            candidates: List of ``(item_id, item_text)`` pairs from bi-encoder.
            top_k: How many results to return.
            fallback_scores: Original bi-encoder scores (same order as candidates).
                Used when cross-encoder is unavailable.

        Returns:
            List of ``(item_id, score)`` sorted by score descending, length ≤ ``top_k``.
        """
        if not candidates:
            return []

        if not self._load():
            # Graceful degradation: return original order
            scores = fallback_scores or [1.0 / (i + 1) for i in range(len(candidates))]
            ranked = sorted(
                zip([c[0] for c in candidates], scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return list(ranked[:top_k])

        # Build (query, document) pairs
        pairs = [(query, text) for _, text in candidates]
        ce_scores: list[float] = self._model.predict(pairs).tolist()

        ranked = sorted(
            zip([c[0] for c in candidates], ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return list(ranked[:top_k])

    @property
    def available(self) -> bool:
        """True if the cross-encoder model loaded successfully."""
        return self._load()
