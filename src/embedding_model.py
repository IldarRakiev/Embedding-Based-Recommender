# src/embedding_model.py
"""Wrapper around sentence-transformers for encoding texts."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


class EmbeddingModel:
    """Loads and caches a sentence-transformer model for encoding.

    Args:
        model_name: HuggingFace model identifier.
            Default: ``paraphrase-multilingual-mpnet-base-v2`` (768-dim,
            multilingual, strong zero-shot food retrieval performance).

    Example:
        >>> model = EmbeddingModel()
        >>> embs = model.encode(["Grilled chicken", "Caesar salad"])
        >>> embs.shape
        (2, 768)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim: int = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode texts into L2-normalized embeddings.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for encoding (tune based on GPU memory).

        Returns:
            ``np.ndarray`` of shape ``(len(texts), dim)`` with L2-normalized rows.
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 500,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
