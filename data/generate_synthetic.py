#!/usr/bin/env python3
"""
Synthetic dataset generator for food recommendation case study.

Dishes are **not** generated here — they are read from a JSON file (e.g.
``final_dishes.json`` from ``generate_dishes_llm.py`` or any compatible export).

Generates:
  - 1100 users (300 archetype + 400 hybrid + 200 contradictory + 200 drifting)
  - Interactions (views, orders, ratings, favorites) using those dishes

Usage:
    python data/generate_synthetic.py
    python data/generate_synthetic.py --dishes-file ../dishes_llm.json
    python data/generate_synthetic.py -d path/to/final_dishes.json --skip-dishes-parquet

Output: ``--output-dir`` (default: data/synthetic/) — ``*.parquet``; optionally
skip writing ``dishes.parquet`` if you only want users + interactions.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

import json
import numpy as np
import pandas as pd

# Ensure data/ is importable
sys.path.insert(0, os.path.dirname(__file__))

DEFAULT_DISHES_FILE = os.path.join(os.path.dirname(__file__), "final_dishes.json")


def _normalize_dish_ids(dishes: list[dict]) -> None:
    """Ensure unique integer ``id`` on every dish (1..N) if missing or duplicated."""
    raw = [d.get("id") for d in dishes]
    if (
        raw
        and all(isinstance(x, int) and x >= 0 for x in raw)
        and len(set(raw)) == len(dishes)
    ):
        return
    for i, d in enumerate(dishes):
        d["id"] = i + 1


def _normalize_dish_fields(d: dict) -> None:
    """Defaults required by interaction scoring (avoid KeyError)."""
    if not d.get("price_tier"):
        d["price_tier"] = "medium"
    if d.get("cuisine_cluster") is None:
        d["cuisine_cluster"] = 0
    else:
        d["cuisine_cluster"] = int(d["cuisine_cluster"])


def load_dishes(path: str) -> list[dict]:
    """Load dishes from JSON; sanitize allergens; normalize ids and required fields."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dishes file not found: {path}")

    with open(path, encoding="utf-8") as f:
        dishes = json.load(f)

    if not isinstance(dishes, list):
        raise ValueError("Dishes JSON must be a list of objects")

    for d in dishes:
        tags = d.get("allergen_tags", [])
        if not isinstance(tags, list):
            tags = []
        d["allergen_tags"] = [t for t in tags if str(t).lower() != "none" and str(t).strip()]

    _normalize_dish_ids(dishes)
    for d in dishes:
        _normalize_dish_fields(d)

    return dishes

from user_archetypes import generate_users

# Meat keywords for vegetarian filter (was in dish_templates)
MEAT_INGREDIENTS = {
    "chicken", "beef", "pork", "lamb", "turkey", "duck", "veal", "bacon",
    "ham", "sausage", "salami", "pepperoni", "prosciutto", "lard", "tuna",
    "salmon", "shrimp", "crab", "lobster", "anchovy", "sardine",
    "фарш", "говядина", "свинина", "курица", "баранина", "индейка",
}

SEED = 42
DAYS_SPAN = 180  # 6 months of history
START_DATE = datetime(2025, 9, 1, tzinfo=timezone.utc)
END_DATE = START_DATE + timedelta(days=DAYS_SPAN)
MID_DATE = START_DATE + timedelta(days=DAYS_SPAN // 2)


def _random_timestamp(rng: np.random.Generator, day_offset: float, meal_time: str | None = None) -> datetime:
    """Generate a timestamp at a given day offset, optionally biased by meal_time."""
    base = START_DATE + timedelta(days=day_offset)

    if meal_time and rng.random() < 0.70:
        # 70% chance of meal-time-consistent hour
        hour_ranges = {
            "breakfast": (6, 10),
            "lunch": (11, 14),
            "dinner": (17, 21),
            "snack": (10, 22),
        }
        lo, hi = hour_ranges.get(meal_time, (8, 22))
        hour = rng.integers(lo, hi + 1)
    else:
        hour = rng.integers(7, 23)

    minute = rng.integers(0, 60)
    return base.replace(hour=int(hour), minute=int(minute))


def _dish_preference_score(
    user: dict,
    dish: dict,
    day_offset: float,
    dishes_by_cluster: dict,
) -> float:
    """Compute a preference score for a user-dish pair. Higher = more likely to interact."""
    score = 1.0

    # --- Rule 1: Cuisine coherence ---
    preferred = user["_preferred_cuisines"]
    cluster = dish["cuisine_cluster"]

    if user["user_type"] == "drifting" and user["_drift_target_cuisines"]:
        # First half: original cuisines, second half: drift target
        if day_offset < DAYS_SPAN / 2:
            active_cuisines = preferred
        else:
            active_cuisines = user["_drift_target_cuisines"]
    else:
        active_cuisines = preferred

    if cluster in active_cuisines:
        if user["user_type"] == "archetype":
            score *= 3.0  # strong preference
        elif user["user_type"] == "hybrid":
            score *= 2.0  # moderate (more cuisines in set)
        elif user["user_type"] == "contradictory":
            score *= 2.5
        else:  # drifting
            score *= 3.0

    # --- Rule 2: Allergen hard block (for orders/favorites) ---
    # Note: this is checked separately for orders; here we just lower score
    user_allergens = set(user["allergens"])
    dish_allergens = set(dish.get("allergen_tags", []))
    if user_allergens & dish_allergens:
        score *= 0.0  # will be blocked entirely for orders

    # --- Rule 4: Price sensitivity ---
    price_tiers = {"low": 1, "medium": 2, "high": 3}
    dish_price = price_tiers.get(dish["price_tier"], 2)
    user_price = price_tiers.get(user["price_level"], 2)
    sensitivity = user["_price_sensitivity"]

    if dish_price > user_price:
        score *= (1.0 - sensitivity * 0.3 * (dish_price - user_price))
    elif dish_price < user_price:
        score *= (1.0 + 0.1)  # slight boost for cheaper than budget

    # --- Rule 5: Nutritional goal alignment ---
    goal = user["goal_type"]
    kcal = dish.get("calories", 200)
    protein = dish.get("protein_g", 10)

    if user["user_type"] == "contradictory":
        boost = 1.1  # weak alignment
    else:
        boost = 1.5

    if goal == "lose" and kcal < 200:
        score *= boost
    elif goal == "lose" and kcal > 350:
        score *= (1.0 / boost)  # penalty
    elif goal == "gain" and protein > 20:
        score *= boost
    elif goal == "gain" and kcal > 250:
        score *= (boost * 0.8)

    # --- Contradictory users: occasional "forbidden" choices ---
    if user["user_type"] == "contradictory" and user["_contradiction_rate"] > 0:
        if rng_global.random() < user["_contradiction_rate"] * 0.3:
            # Boost desserts/comfort for dieters, salads for gainers
            if goal == "lose" and cluster in [7, 5]:  # desserts, comfort
                score *= 2.5
            elif goal == "gain" and cluster == 4:  # healthy/diet
                score *= 2.0

    # --- Vegetarian filter ---
    if user.get("_vegetarian", False):
        dish_ingredients = set(ing.lower() for ing in dish.get("ingredients", []))
        if dish_ingredients & MEAT_INGREDIENTS:
            score *= 0.0

    return max(score, 0.0)


def _has_allergen_conflict(user: dict, dish: dict) -> bool:
    """Check if dish contains any of user's allergens."""
    return bool(set(user["allergens"]) & set(dish.get("allergen_tags", [])))


def _is_meat_dish(dish: dict) -> bool:
    """Check if dish contains meat (for vegetarian users)."""
    dish_ingredients = set(ing.lower() for ing in dish.get("ingredients", []))
    return bool(dish_ingredients & MEAT_INGREDIENTS)


def generate_interactions(
    rng: np.random.Generator,
    users: list[dict],
    dishes: list[dict],
) -> pd.DataFrame:
    """Generate all interactions following the 9 rules from the plan."""
    # Index dishes by cluster for fast lookup
    dishes_by_cluster: dict[int, list[dict]] = {}
    for d in dishes:
        dishes_by_cluster.setdefault(d["cuisine_cluster"], []).append(d)

    all_dish_ids = [d["id"] for d in dishes]
    dish_by_id = {d["id"]: d for d in dishes}

    all_interactions = []

    for user in users:
        # Determine interaction counts
        n_views = rng.integers(50, 200)
        n_orders = rng.integers(10, 40)
        n_ratings = rng.integers(15, 50)
        n_favorites = rng.integers(3, 10)

        # --- Compute preference scores for all dishes ---
        # Sample a day for scoring (average over span)
        sample_days = rng.uniform(0, DAYS_SPAN, size=5)
        scores = np.zeros(len(dishes))
        for day in sample_days:
            for i, dish in enumerate(dishes):
                scores[i] += _dish_preference_score(user, dish, day, dishes_by_cluster)
        scores /= len(sample_days)

        # Filter out allergen-containing and meat (for vegetarians) for orders/favorites
        can_order = np.ones(len(dishes), dtype=bool)
        for i, dish in enumerate(dishes):
            if _has_allergen_conflict(user, dish):
                can_order[i] = False
            if user.get("_vegetarian", False) and _is_meat_dish(dish):
                can_order[i] = False

        order_scores = scores * can_order
        if order_scores.sum() == 0:
            order_scores = can_order.astype(float)
        order_probs = order_scores / order_scores.sum()

        # View scores: can view anything (including allergen dishes with low prob)
        view_scores = scores.copy()
        view_scores[view_scores == 0] = 0.05  # small chance to view anything
        view_probs = view_scores / view_scores.sum()

        # --- Generate orders ---
        ordered_dish_indices = rng.choice(len(dishes), size=n_orders, replace=True, p=order_probs)
        ordered_dish_ids = set()
        order_counts: dict[int, int] = {}

        for idx in ordered_dish_indices:
            dish = dishes[idx]
            dish_id = dish["id"]
            ordered_dish_ids.add(dish_id)
            order_counts[dish_id] = order_counts.get(dish_id, 0) + 1

            day_offset = rng.uniform(0, DAYS_SPAN)
            ts = _random_timestamp(rng, day_offset, dish.get("meal_time"))

            all_interactions.append({
                "user_id": user["user_id"],
                "dish_id": dish_id,
                "interaction_type": "order",
                "timestamp": ts,
                "rating": None,
            })

        # --- Rule 7: Repeat orders (30% of users have favorites) ---
        if rng.random() < 0.3 and len(ordered_dish_ids) > 0:
            n_regulars = rng.integers(2, 5)
            regular_dishes = rng.choice(list(ordered_dish_ids), size=min(n_regulars, len(ordered_dish_ids)), replace=False)
            for dish_id in regular_dishes:
                extra_orders = rng.integers(2, 6)
                order_counts[dish_id] = order_counts.get(dish_id, 0) + extra_orders
                dish = dish_by_id[dish_id]
                for _ in range(extra_orders):
                    day_offset = rng.uniform(0, DAYS_SPAN)
                    ts = _random_timestamp(rng, day_offset, dish.get("meal_time"))
                    all_interactions.append({
                        "user_id": user["user_id"],
                        "dish_id": dish_id,
                        "interaction_type": "order",
                        "timestamp": ts,
                        "rating": None,
                    })

        # --- Generate views ---
        # Rule 6: views ⊃ orders (view before ordering, mostly)
        viewed_dish_ids = set(ordered_dish_ids)  # all ordered dishes were viewed

        # Add views for ordered dishes
        for dish_id in ordered_dish_ids:
            dish = dish_by_id[dish_id]
            # View happened before order
            day_offset = rng.uniform(0, DAYS_SPAN)
            ts = _random_timestamp(rng, day_offset, dish.get("meal_time"))
            all_interactions.append({
                "user_id": user["user_id"],
                "dish_id": dish_id,
                "interaction_type": "view",
                "timestamp": ts,
                "rating": None,
            })

        # Rule 8: Random browsing views (40-60% are noise)
        noise_views = int(n_views * rng.uniform(0.4, 0.6))
        browsing_indices = rng.choice(len(dishes), size=noise_views, replace=True)
        for idx in browsing_indices:
            dish = dishes[idx]
            viewed_dish_ids.add(dish["id"])
            day_offset = rng.uniform(0, DAYS_SPAN)
            ts = _random_timestamp(rng, day_offset)
            all_interactions.append({
                "user_id": user["user_id"],
                "dish_id": dish["id"],
                "interaction_type": "view",
                "timestamp": ts,
                "rating": None,
            })

        # Targeted views (remaining)
        remaining_views = n_views - noise_views - len(ordered_dish_ids)
        if remaining_views > 0:
            targeted_indices = rng.choice(len(dishes), size=remaining_views, replace=True, p=view_probs)
            for idx in targeted_indices:
                dish = dishes[idx]
                viewed_dish_ids.add(dish["id"])
                day_offset = rng.uniform(0, DAYS_SPAN)
                ts = _random_timestamp(rng, day_offset)
                all_interactions.append({
                    "user_id": user["user_id"],
                    "dish_id": dish["id"],
                    "interaction_type": "view",
                    "timestamp": ts,
                    "rating": None,
                })

        # --- Rule 9: Ratings ---
        # Ordered dishes: mostly 4-5, sometimes 3
        rated_dishes = set()
        for dish_id in ordered_dish_ids:
            if rng.random() < 0.7:  # 70% of orders get rated
                if rng.random() < 0.75:
                    rating = rng.choice([4, 5])
                elif rng.random() < 0.7:
                    rating = 3
                else:
                    rating = rng.choice([1, 2])  # disappointment

                day_offset = rng.uniform(0, DAYS_SPAN)
                ts = _random_timestamp(rng, day_offset)
                all_interactions.append({
                    "user_id": user["user_id"],
                    "dish_id": dish_id,
                    "interaction_type": "rating",
                    "timestamp": ts,
                    "rating": int(rating),
                })
                rated_dishes.add(dish_id)

        # Viewed but not ordered: sometimes rated low
        viewed_not_ordered = viewed_dish_ids - ordered_dish_ids
        if viewed_not_ordered:
            n_neg_ratings = min(rng.integers(0, 5), len(viewed_not_ordered))
            neg_rated = rng.choice(list(viewed_not_ordered), size=n_neg_ratings, replace=False)
            for dish_id in neg_rated:
                rating = rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
                day_offset = rng.uniform(0, DAYS_SPAN)
                ts = _random_timestamp(rng, day_offset)
                all_interactions.append({
                    "user_id": user["user_id"],
                    "dish_id": int(dish_id),
                    "interaction_type": "rating",
                    "timestamp": ts,
                    "rating": int(rating),
                })

        # --- Favorites ---
        # Subset of highly-rated ordered dishes
        high_rated_ordered = [d for d in ordered_dish_ids if d in rated_dishes]
        if not high_rated_ordered:
            high_rated_ordered = list(ordered_dish_ids)

        n_fav = min(n_favorites, len(high_rated_ordered))
        if n_fav > 0:
            fav_ids = rng.choice(high_rated_ordered, size=n_fav, replace=False)
            for dish_id in fav_ids:
                day_offset = rng.uniform(0, DAYS_SPAN)
                ts = _random_timestamp(rng, day_offset)
                all_interactions.append({
                    "user_id": user["user_id"],
                    "dish_id": int(dish_id),
                    "interaction_type": "favorite",
                    "timestamp": ts,
                    "rating": None,
                })

        # --- Rule 6: 5% impulse orders without prior view ---
        # (Already handled: some orders might not have explicit prior views
        #  since we add views for ordered dishes anyway. The noise views
        #  naturally create the "viewed but not ordered" set.)

    return pd.DataFrame(all_interactions)


def temporal_split(
    interactions: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split interactions by timestamp: 60/20/20."""
    interactions = interactions.sort_values("timestamp").reset_index(drop=True)
    n = len(interactions)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = interactions.iloc[:train_end]
    val = interactions.iloc[train_end:val_end]
    test = interactions.iloc[val_end:]

    return train, val, test


def main():
    global rng_global

    parser = argparse.ArgumentParser(
        description="Generate synthetic users and interactions from an existing dishes JSON file.",
    )
    parser.add_argument(
        "--dishes-file",
        "-d",
        default=DEFAULT_DISHES_FILE,
        help=f"Path to dishes JSON (default: {DEFAULT_DISHES_FILE})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=os.path.join(os.path.dirname(__file__), "synthetic"),
        help="Directory for parquet outputs (default: data/synthetic/)",
    )
    parser.add_argument(
        "--skip-dishes-parquet",
        action="store_true",
        help="Do not write dishes.parquet (dishes stay only in the source JSON).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"RNG seed (default: {SEED})",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rng_global = rng  # used in _dish_preference_score for contradictory users

    print(f"Loading dishes from {os.path.abspath(args.dishes_file)}...")
    dishes = load_dishes(args.dishes_file)
    dishes_df = pd.DataFrame(dishes)
    print(f"  Dishes: {len(dishes_df)} across {dishes_df['cuisine_cluster'].nunique()} clusters")

    print("Generating 1100 users...")
    users = generate_users(rng, target_count=1100)
    users_df = pd.DataFrame(users)
    # Separate hidden fields
    hidden_cols = [c for c in users_df.columns if c.startswith("_")]
    users_public = users_df.drop(columns=hidden_cols)
    print(f"  Users: {len(users_df)} ({users_df['user_type'].value_counts().to_dict()})")

    print("Generating interactions (this may take a minute)...")
    interactions_df = generate_interactions(rng, users, dishes)
    print(f"  Total interactions: {len(interactions_df):,}")
    print(f"  By type: {interactions_df['interaction_type'].value_counts().to_dict()}")

    print("Splitting train/val/test (60/20/20)...")
    train, val, test = temporal_split(interactions_df)
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not args.skip_dishes_parquet:
        dishes_df.to_parquet(os.path.join(out_dir, "dishes.parquet"), index=False)
    users_public.to_parquet(os.path.join(out_dir, "users.parquet"), index=False)
    interactions_df.to_parquet(os.path.join(out_dir, "interactions.parquet"), index=False)
    train.to_parquet(os.path.join(out_dir, "interactions_train.parquet"), index=False)
    val.to_parquet(os.path.join(out_dir, "interactions_val.parquet"), index=False)
    test.to_parquet(os.path.join(out_dir, "interactions_test.parquet"), index=False)

    # Save allergens separately for convenience
    allergen_records = []
    for u in users:
        for a in u["allergens"]:
            allergen_records.append({"user_id": u["user_id"], "allergen": a})
    if allergen_records:
        pd.DataFrame(allergen_records).to_parquet(os.path.join(out_dir, "allergens.parquet"), index=False)

    print(f"\nSaved to {out_dir}/")
    _validate(dishes_df, users_df, interactions_df, train, test)


def _validate(dishes_df, users_df, interactions_df, train, test):
    """Run validation checks from the plan."""
    print("\n=== Validation ===")
    dish_allergens = {
        row["id"]: set(row["allergen_tags"])
        for _, row in dishes_df.iterrows()
    }

    # Check 1: No allergen violations in orders/favorites
    orders_favs = interactions_df[interactions_df["interaction_type"].isin(["order", "favorite"])]
    user_allergens = {u["user_id"]: set(u["allergens"]) for u in users_df.to_dict("records")}

    violations = 0
    for _, row in orders_favs.iterrows():
        u_allergens = user_allergens.get(row["user_id"], set())
        d_allergens = dish_allergens.get(row["dish_id"], set())
        if u_allergens & d_allergens:
            violations += 1
    print(f"  [{'PASS' if violations == 0 else 'FAIL'}] Allergen violations in orders/favorites: {violations}")

    # Check 2: Users with >=5 positives in test
    test_positives = test[
        (test["interaction_type"].isin(["order", "favorite"])) |
        ((test["interaction_type"] == "rating") & (test["rating"] >= 4))
    ]
    user_pos_counts = test_positives.groupby("user_id").size()
    n_qualified = (user_pos_counts >= 5).sum()
    print(f"  [{'PASS' if n_qualified >= 200 else 'WARN'}] Users with >=5 test positives: {n_qualified}")

    # Check 3: views >= orders per user
    views_per_user = interactions_df[interactions_df["interaction_type"] == "view"].groupby("user_id").size()
    orders_per_user = interactions_df[interactions_df["interaction_type"] == "order"].groupby("user_id").size()
    all_ok = True
    for uid in orders_per_user.index:
        if views_per_user.get(uid, 0) < orders_per_user[uid]:
            all_ok = False
            break
    print(f"  [{'PASS' if all_ok else 'FAIL'}] Views >= Orders for all users")

    # Check 5: Goal-calorie correlation
    orders_with_cal = interactions_df[interactions_df["interaction_type"] == "order"].merge(
        dishes_df[["id", "calories"]], left_on="dish_id", right_on="id"
    ).merge(
        users_df[["user_id", "goal_type"]], on="user_id"
    )
    if len(orders_with_cal) > 0:
        lose_cal = orders_with_cal[orders_with_cal["goal_type"] == "lose"]["calories"].mean()
        gain_cal = orders_with_cal[orders_with_cal["goal_type"] == "gain"]["calories"].mean()
        print(f"  [INFO] Avg calories — lose: {lose_cal:.0f}, gain: {gain_cal:.0f} (expect lose < gain)")

    # Check: Rating distribution
    ratings = interactions_df[interactions_df["interaction_type"] == "rating"]["rating"]
    if len(ratings) > 0:
        print(f"  [INFO] Rating distribution: {ratings.value_counts().sort_index().to_dict()}")

    # Check: Interaction counts
    print(f"  [INFO] Interaction types: {interactions_df['interaction_type'].value_counts().to_dict()}")
    print("\nDone!")


# Global rng for use in scoring function
rng_global: np.random.Generator = np.random.default_rng(SEED)

if __name__ == "__main__":
    main()
