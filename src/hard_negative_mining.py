"""
Hard-negative mining utilities for contrastive fine-tuning.

Motivation:
- In this case study, MultipleNegativesRankingLoss (MNRL) with in-batch negatives
  underperforms because random negatives are too easy.
- The synthetic data generator contains a strong "cuisine_cluster" signal.
  A good hard negative for a dish is a *similar-cluster* dish that the user
  did NOT order/favorite.

This module builds explicit (anchor, positive, hard_negative) triplets that can
be trained with sentence-transformers TripletLoss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class HardNegativeMiningConfig:
    """Configuration for hard-negative sampling."""

    # How many nearest neighbors to consider within the cluster
    candidate_pool: int = 50
    # Fallback negative sampling attempts per triplet
    max_fallback_tries: int = 30
    # Ensure negative is not among user's positives (orders/favorites)
    exclude_user_positives: bool = True
    # Prefer mining within the same cuisine cluster if possible
    prefer_same_cluster: bool = True


def _as_int(x) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def build_dish_cluster_index(dishes_df) -> dict[int, np.ndarray]:
    """Map cuisine_cluster -> array of dish_ids in that cluster."""
    if "id" not in dishes_df.columns:
        raise ValueError("dishes_df must contain an 'id' column")
    if "cuisine_cluster" not in dishes_df.columns:
        raise ValueError("dishes_df must contain a 'cuisine_cluster' column")

    cluster_to_ids: dict[int, list[int]] = {}
    for _, row in dishes_df.iterrows():
        did = _as_int(row.get("id"))
        cc = _as_int(row.get("cuisine_cluster"))
        if did is None or cc is None:
            continue
        cluster_to_ids.setdefault(cc, []).append(did)

    return {cc: np.array(ids, dtype=np.int64) for cc, ids in cluster_to_ids.items()}


def mine_hard_negative(
    anchor_id: int,
    *,
    dish_id_to_idx: dict[int, int],
    idx_to_dish_id: dict[int, int],
    dish_embeddings: np.ndarray,
    cluster_to_ids: dict[int, np.ndarray],
    dish_id_to_cluster: dict[int, int],
    user_positive_ids: set[int] | None = None,
    rng: np.random.Generator,
    cfg: HardNegativeMiningConfig | None = None,
    # Optional: mine negatives in the cluster of another dish (typically the positive),
    # while still ranking candidates by similarity to `anchor_id`'s embedding.
    cluster_source_id: int | None = None,
) -> int | None:
    """Pick a hard negative dish id for an anchor.

    Strategy:
    - Prefer the same cuisine_cluster as the anchor.
    - Within that cluster, take top-N nearest neighbors by cosine similarity.
    - Return the best one that is NOT the anchor and (optionally) not in user positives.
    - If the cluster is missing / empty, fallback to random sampling from the global pool.
    """
    cfg = cfg or HardNegativeMiningConfig()
    user_positive_ids = user_positive_ids or set()

    q_idx = dish_id_to_idx.get(anchor_id)
    if q_idx is None:
        return None

    # --- 1) Try same-cluster nearest neighbors (hardest) ---
    cluster_key = cluster_source_id if cluster_source_id is not None else anchor_id
    anchor_cluster = dish_id_to_cluster.get(int(cluster_key))
    if cfg.prefer_same_cluster and anchor_cluster is not None:
        ids_in_cluster = cluster_to_ids.get(int(anchor_cluster))
        if ids_in_cluster is not None and len(ids_in_cluster) > 2:
            # Convert candidate ids -> indices
            cand_ids = ids_in_cluster
            cand_indices = np.array(
                [dish_id_to_idx.get(int(d), -1) for d in cand_ids],
                dtype=np.int64,
            )
            mask_valid = cand_indices >= 0
            cand_ids = cand_ids[mask_valid]
            cand_indices = cand_indices[mask_valid]

            if len(cand_indices) > 2:
                q = dish_embeddings[q_idx]
                sims = dish_embeddings[cand_indices] @ q  # cosine if normalized
                # Take top-K most similar candidates
                k = min(cfg.candidate_pool, len(sims))
                topk = np.argpartition(-sims, kth=k - 1)[:k]
                # Sort those topk by similarity desc
                topk = topk[np.argsort(-sims[topk])]
                for j in topk:
                    neg_id = int(cand_ids[j])
                    if neg_id == anchor_id:
                        continue
                    if cluster_source_id is not None and neg_id == int(cluster_source_id):
                        continue
                    if cfg.exclude_user_positives and neg_id in user_positive_ids:
                        continue
                    return neg_id

    # --- 2) Fallback: random sampling (still ensures not positive/anchor) ---
    # Sample random indices from the whole embedding matrix
    n = dish_embeddings.shape[0]
    for _ in range(cfg.max_fallback_tries):
        neg_idx = int(rng.integers(0, n))
        neg_id = idx_to_dish_id.get(neg_idx)
        if neg_id is None:
            continue
        if int(neg_id) == int(anchor_id):
            continue
        if cfg.exclude_user_positives and int(neg_id) in user_positive_ids:
            continue
        return int(neg_id)

    return None


def build_triplets_same_user_with_hard_negatives(
    *,
    interactions_train_df,
    id_to_text: dict[int, str],
    dishes_df,
    dish_embeddings: np.ndarray,
    positives_interaction_types: Iterable[str] = ("order", "favorite"),
    max_triplets: int = 80_000,
    triplets_per_user: int = 25,
    seed: int = 42,
    cfg: HardNegativeMiningConfig | None = None,
) -> list[tuple[str, str, str]]:
    """Create (anchor_text, positive_text, hard_negative_text) triplets.

    Positives:
    - Two distinct dishes that the same user ordered/favorited in train.

    Hard negatives:
    - Prefer dishes from the **same cuisine_cluster as the positive**, mined by cosine
      similarity to the **anchor** embedding, excluding user's positives.

    Returns:
      A list of text triplets suitable for sentence-transformers InputExample(texts=[a,p,n]).
    """
    cfg = cfg or HardNegativeMiningConfig()
    rng = np.random.default_rng(seed)

    # User -> list of positive dish ids
    pos_df = interactions_train_df[
        interactions_train_df["interaction_type"].isin(list(positives_interaction_types))
    ]
    user_to_pos = (
        pos_df.groupby("user_id")["dish_id"]
        .apply(list)
        .to_dict()
    )

    # Dish id mappings (embedding row order must match dishes_df order used to build embeddings)
    dish_ids_in_order = [int(x) for x in dishes_df["id"].tolist()]
    dish_id_to_idx = {did: i for i, did in enumerate(dish_ids_in_order)}
    idx_to_dish_id = {i: did for i, did in enumerate(dish_ids_in_order)}

    dish_id_to_cluster = {}
    for _, row in dishes_df.iterrows():
        did = _as_int(row.get("id"))
        cc = _as_int(row.get("cuisine_cluster"))
        if did is not None and cc is not None:
            dish_id_to_cluster[did] = cc

    cluster_to_ids = build_dish_cluster_index(dishes_df)

    triplets: list[tuple[str, str, str]] = []

    user_ids = list(user_to_pos.keys())
    rng.shuffle(user_ids)

    for uid in user_ids:
        pos_set = {int(d) for d in user_to_pos[uid] if int(d) in id_to_text}
        if len(pos_set) < 5:
            continue

        pos_list = np.array(list(pos_set), dtype=np.int64)
        # Similar to notebook: draw a bit more than triplets_per_user on larger sets
        n_draws = min(triplets_per_user, max(1, len(pos_list) * 3))

        for _ in range(n_draws):
            a_id, p_id = rng.choice(pos_list, size=2, replace=False)

            neg_id = mine_hard_negative(
                int(a_id),
                dish_id_to_idx=dish_id_to_idx,
                idx_to_dish_id=idx_to_dish_id,
                dish_embeddings=dish_embeddings,
                cluster_to_ids=cluster_to_ids,
                dish_id_to_cluster=dish_id_to_cluster,
                user_positive_ids=pos_set,
                rng=rng,
                cfg=cfg,
                cluster_source_id=int(p_id),
            )
            if neg_id is None or neg_id not in id_to_text:
                continue

            triplets.append((id_to_text[int(a_id)], id_to_text[int(p_id)], id_to_text[int(neg_id)]))
            if len(triplets) >= max_triplets:
                return triplets

    return triplets

