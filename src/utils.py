# src/utils.py
"""Evaluation metrics and visualization helpers for food recommendation experiments.

Metrics follow the standard information retrieval definitions.
All functions operate on integer item IDs.

Evaluation protocol used in this case study (see Notebook 2 for details):
    Co-preference item-to-item: for each user with ≥5 positive ratings,
    take one positive recipe as query, use remaining positives as relevant set,
    measure how many top-K retrieved items are relevant.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# ============================================================
# METRICS
# ============================================================

def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of top-K recommended items that are relevant.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cutoff.

    Returns:
        P@K in [0, 1].

    Example:
        >>> precision_at_k([1, 2, 3, 4, 5], {2, 4, 6}, k=5)
        0.4
    """
    if k == 0 or not recommended:
        return 0.0
    top_k = recommended[:k]
    return sum(1 for item in top_k if item in relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Fraction of relevant items found in the top-K recommendations.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of ground-truth relevant item IDs.
        k: Cutoff.

    Returns:
        R@K in [0, 1].

    Example:
        >>> recall_at_k([1, 2, 3, 4, 5], {2, 4, 6}, k=5)
        0.6666...
    """
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    return len(top_k & relevant) / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K.

    Uses binary relevance (1 if relevant, 0 otherwise).

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of relevant item IDs.
        k: Cutoff.

    Returns:
        NDCG@K in [0, 1].
    """
    if not relevant or k == 0:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant
    )

    ideal_k = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_k))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(recommended: list[int], relevant: set[int]) -> float:
    """Mean Reciprocal Rank: reciprocal rank of the first relevant item.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of relevant item IDs.

    Returns:
        MRR in [0, 1]. Returns 0 if no relevant item found.

    Example:
        >>> mrr([5, 2, 3], {2, 4})
        0.5
    """
    for rank, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (rank + 1)
    return 0.0


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Whether at least one relevant item appears in top-K.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of relevant item IDs.
        k: Cutoff.

    Returns:
        1.0 if any relevant item is in top-K, else 0.0.
    """
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def evaluate_all(
    recommended: list[int],
    relevant: set[int],
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Compute all metrics at multiple cutoffs.

    Args:
        recommended: Ordered list of recommended item IDs.
        relevant: Set of relevant item IDs.
        ks: Cutoff values. Default: ``[5, 10, 20]``.

    Returns:
        Dict with keys like ``"P@5"``, ``"R@10"``, ``"NDCG@10"``, ``"MRR"``, ``"HR@5"``.

    Example:
        >>> evaluate_all([1, 2, 3], {1, 5}, ks=[3])
        {'P@3': 0.333..., 'R@3': 0.5, 'NDCG@3': ..., 'MRR': 1.0, 'HR@3': 1.0}
    """
    if ks is None:
        ks = [5, 10, 20]

    result: dict[str, float] = {}
    for k in ks:
        result[f"P@{k}"] = precision_at_k(recommended, relevant, k)
        result[f"R@{k}"] = recall_at_k(recommended, relevant, k)
        result[f"NDCG@{k}"] = ndcg_at_k(recommended, relevant, k)
        result[f"HR@{k}"] = hit_rate_at_k(recommended, relevant, k)
    result["MRR"] = mrr(recommended, relevant)

    return result


# ============================================================
# VISUALIZATION
# ============================================================

def plot_embeddings_umap(
    embeddings: np.ndarray,
    labels: list[Any],
    title: str,
    save_path: str | None = None,
    random_state: int = 42,
) -> None:
    """2D UMAP visualization of embeddings colored by labels.

    Args:
        embeddings: ``np.ndarray`` of shape ``(N, dim)``.
        labels: List of length N used for color coding (categorical or numeric).
        title: Plot title.
        save_path: If given, save figure to this path instead of showing.
        random_state: UMAP random seed for reproducibility.
    """
    import matplotlib.pyplot as plt
    import umap

    reducer = umap.UMAP(n_components=2, random_state=random_state, metric="cosine")
    coords = reducer.fit_transform(embeddings)

    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 7))
    for lab in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == lab]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=8, alpha=0.6,
            color=cmap(label_to_idx[lab]),
            label=str(lab),
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if len(unique_labels) <= 15:
        ax.legend(markerscale=2, fontsize=8, loc="best")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_metric_comparison(
    results_dict: dict[str, dict[str, float]],
    metric_name: str,
    title: str,
    save_path: str | None = None,
) -> None:
    """Bar chart comparing a metric across experiment variants.

    Args:
        results_dict: ``{variant_name: metrics_dict}``.
        metric_name: Key to plot (e.g. ``"P@10"``).
        title: Plot title.
        save_path: Optional path to save figure.

    Example:
        >>> plot_metric_comparison(
        ...     {"baseline": {"P@10": 0.12}, "no_recipe": {"P@10": 0.15}},
        ...     metric_name="P@10", title="P@10 by configuration"
        ... )
    """
    import matplotlib.pyplot as plt

    names = list(results_dict.keys())
    values = [results_dict[n].get(metric_name, 0.0) for n in names]
    baseline_val = values[0] if values else 0.0

    colors = []
    for v in values:
        if v > baseline_val * 1.02:
            colors.append("#2ecc71")   # green — improvement
        elif v < baseline_val * 0.98:
            colors.append("#e74c3c")   # red — regression
        else:
            colors.append("#3498db")   # blue — neutral

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_title(title, fontsize=13)
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, max(values) * 1.15 if values else 1.0)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_cumulative_improvements(
    results_list: list[dict[str, float]],
    metric_name: str,
    labels: list[str],
    title: str = "Cumulative Improvement",
    save_path: str | None = None,
) -> None:
    """Line chart showing cumulative metric improvement across experiment steps.

    Args:
        results_list: List of metrics dicts in order of experiments.
        metric_name: Metric to plot (e.g. ``"P@10"``).
        labels: Label for each step (same length as results_list).
        title: Plot title.
        save_path: Optional save path.
    """
    import matplotlib.pyplot as plt

    values = [r.get(metric_name, 0.0) for r in results_list]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 4))
    ax.plot(labels, values, marker="o", linewidth=2, markersize=7, color="#2980b9")

    for i, (lab, val) in enumerate(zip(labels, values)):
        ax.annotate(
            f"{val:.3f}",
            xy=(i, val),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center", fontsize=9,
        )

    ax.set_title(title, fontsize=13)
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, max(values) * 1.2 if values else 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=30, ha="right", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()


def plot_nearest_neighbors(
    query_name: str,
    neighbor_names: list[str],
    scores: list[float],
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Horizontal bar chart of nearest neighbors with similarity scores.

    Args:
        query_name: Name of the query item.
        neighbor_names: Names of nearest neighbors (in ranked order).
        scores: Similarity scores (same order).
        title: Plot title. Defaults to ``"Nearest neighbors for: <query_name>"``.
        save_path: Optional save path.
    """
    import matplotlib.pyplot as plt

    title = title or f"Nearest neighbors for: {query_name}"
    names = list(reversed(neighbor_names))
    vals = list(reversed(scores))

    fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.5)))
    bars = ax.barh(names, vals, color="#3498db", edgecolor="white")

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", fontsize=9,
        )

    ax.set_xlim(0, max(vals) * 1.15 if vals else 1.0)
    ax.set_xlabel("Cosine similarity")
    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close()
