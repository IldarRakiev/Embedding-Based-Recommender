#!/usr/bin/env python3
"""
Fine-tune a sentence-transformer with explicit hard negatives (TripletLoss).

This script is meant as an "upgrade path" from the notebook fine-tuning baseline:
instead of MultipleNegativesRankingLoss with random in-batch negatives, we mine
hard negatives from the same cuisine cluster using cosine neighbors.

Example:
  python scripts/finetune_triplet_hard_negatives.py ^
    --data-dir data/synthetic ^
    --output-dir models/food-recsys-triplet-hardneg ^
    --max-triplets 80000
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from src.hard_negative_mining import (
    HardNegativeMiningConfig,
    build_triplets_same_user_with_hard_negatives,
)
from src.text_builders import dish_to_rich_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=os.path.join("data", "synthetic"))
    p.add_argument("--model-name", default="paraphrase-multilingual-mpnet-base-v2")
    p.add_argument("--output-dir", default=os.path.join("models", "food-recsys-triplet-hardneg"))
    p.add_argument("--max-triplets", type=int, default=80_000)
    p.add_argument("--triplets-per-user", type=int, default=25)
    p.add_argument("--candidate-pool", type=int, default=50)
    # TripletLoss runs 3 forwards per step (anchor/positive/negative) -> keep this smaller than MNRL.
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Avoid accidental DataParallel paths on multi-GPU VMs; this case study targets a single T4.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    data_dir = os.path.abspath(args.data_dir)
    dishes_path = os.path.join(data_dir, "dishes.parquet")
    train_path = os.path.join(data_dir, "interactions_train.parquet")

    dishes = pd.read_parquet(dishes_path)
    train = pd.read_parquet(train_path)

    # Best text config from the case study notebooks (works slightly better than baseline).
    id_to_text = {
        int(row["id"]): dish_to_rich_text(
            row.to_dict(),
            tags=row.get("tag_list", []),
            include_recipe=False,
            include_macro_tokens=False,
            include_ratios=True,
            include_ingredients=True,
        )
        for _, row in dishes.iterrows()
    }

    # Mine hard negatives using the *current* model's embedding space.
    model = SentenceTransformer(args.model_name)
    ordered_ids = [int(x) for x in dishes["id"].tolist()]
    texts_in_order = [id_to_text[i] for i in ordered_ids]
    embs = model.encode(
        texts_in_order,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    cfg = HardNegativeMiningConfig(candidate_pool=int(args.candidate_pool))
    text_triplets = build_triplets_same_user_with_hard_negatives(
        interactions_train_df=train,
        id_to_text=id_to_text,
        dishes_df=dishes,
        dish_embeddings=embs,
        max_triplets=int(args.max_triplets),
        triplets_per_user=int(args.triplets_per_user),
        seed=int(args.seed),
        cfg=cfg,
    )
    if not text_triplets:
        raise RuntimeError("No triplets mined. Check data and parameters.")

    train_examples = [InputExample(texts=[a, p, n]) for (a, p, n) in text_triplets]
    print(f"Mined triplets: {len(train_examples):,}")

    dl = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=int(args.batch_size),
        drop_last=True,
    )
    train_loss = losses.TripletLoss(model=model)

    os.makedirs(os.path.abspath(args.output_dir), exist_ok=True)
    model.fit(
        train_objectives=[(dl, train_loss)],
        epochs=int(args.epochs),
        warmup_steps=min(500, max(1, len(train_examples) // int(args.batch_size))),
        output_path=os.path.abspath(args.output_dir),
        use_amp=True,
        show_progress_bar=True,
    )

    # Small sanity output: cosine sim stats for a random mini-batch
    batch = rng.choice(len(embs), size=min(128, len(embs)), replace=False)
    sims = embs[batch] @ embs[batch].T
    print(f"Embedding cosine sim matrix sample: mean={sims.mean():.4f}, max={sims.max():.4f}")
    print(f"Saved model to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()

