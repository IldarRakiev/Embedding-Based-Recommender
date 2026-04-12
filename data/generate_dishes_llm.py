#!/usr/bin/env python3
"""
Generate rich dish descriptions using GPT-4o-mini via OpenRouter.

Reads dish_templates.py for structure (clusters, nutrition, allergens),
then asks the LLM to produce natural names, descriptions, recipe texts,
and ingredient lists.

Output: data/dishes_llm.json — used by generate_synthetic.py instead of
template-based text.

Usage:
    python data/generate_dishes_llm.py

Cost estimate: ~$0.85 for 5000 dishes with gpt-4o-mini via OpenRouter.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────
BATCH_SIZE = 10          # dishes per LLM request
MODEL = "openai/gpt-4o-mini"
OUTPUT_FILE = Path(__file__).parent / "dishes_llm.json"
PROGRESS_FILE = Path(__file__).parent / "dishes_llm_progress.json"
FINAL_FILE = Path(__file__).parent / "final_dishes.json"

# Load API config from backend/.env
ENV_PATH = Path(__file__).resolve().parents[2] / "backend" / ".env"


def _load_env(path: Path) -> dict[str, str]:
    """Parse a .env file into a dict."""
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


env = _load_env(ENV_PATH)
API_KEY = env.get("ORACLE_API_KEY", "")
API_BASE = env.get("ORACLE_API_BASE", "https://openrouter.ai/api/v1")

if not API_KEY:
    print(f"ERROR: ORACLE_API_KEY not found in {ENV_PATH}")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

# ── Cluster definitions ─────────────────────────────────────────────────
CLUSTERS = {
    0: {
        "name": "Russian/Slavic",
        "examples": "borscht, pelmeni, bliny, solyanka, kasha, shchi, beef stroganoff, pirozhki, olivier salad, syrniki",
        "meal_times": ["breakfast", "lunch", "dinner"],
    },
    1: {
        "name": "Mediterranean",
        "examples": "Greek salad, pasta carbonara, hummus, grilled fish, moussaka, falafel, risotto, bruschetta, tabbouleh",
        "meal_times": ["lunch", "dinner"],
    },
    2: {
        "name": "Asian",
        "examples": "ramen, green curry, fried rice, sushi, pad thai, teriyaki bowl, tom yum, dim sum, bibimbap, pho",
        "meal_times": ["lunch", "dinner"],
    },
    3: {
        "name": "American",
        "examples": "burger, BBQ ribs, mac and cheese, steak, chicken wings, pancakes, clam chowder, hot dog, BLT sandwich",
        "meal_times": ["breakfast", "lunch", "dinner"],
    },
    4: {
        "name": "Healthy/Diet",
        "examples": "quinoa bowl, green smoothie, grilled chicken breast, acai bowl, steamed fish, kale salad, protein shake",
        "meal_times": ["breakfast", "lunch", "dinner", "snack"],
    },
    5: {
        "name": "Comfort/Hearty",
        "examples": "beef stew, lasagna, shepherd's pie, chicken pot pie, chili con carne, meatloaf, pot roast",
        "meal_times": ["dinner"],
    },
    6: {
        "name": "Breakfast",
        "examples": "oatmeal, omelet, granola, avocado toast, french toast, eggs benedict, yogurt parfait, smoothie bowl",
        "meal_times": ["breakfast"],
    },
    7: {
        "name": "Desserts/Snacks",
        "examples": "chocolate cake, fruit salad, brownies, banana bread, energy bar, cheesecake, muffin, tiramisu",
        "meal_times": ["snack", "breakfast"],
    },
}

ALLERGENS_ALL = ["dairy", "gluten", "nuts", "eggs", "soy", "fish"]

SYSTEM_PROMPT = """You are a food database generator. You produce realistic dish entries for a food recommendation dataset.

For each dish, generate:
1. "name" — natural, appetizing dish name (English). Varied, not repetitive.
2. "description" — 1-2 sentences describing the dish, its taste and character.
3. "recipe_text" — 3-5 step cooking instructions. Include SPECIFIC ingredient amounts. Be detailed with cooking procedures (temperature, time, technique). This should read like a real recipe.
4. "ingredients" — flat list of main ingredients (just names, no amounts).
5. "calories" — kcal per 100g (realistic value).
6. "protein_g" — grams per 100g.
7. "fat_g" — grams per 100g.
8. "carbs_g" — grams per 100g.
9. "fiber_g" — grams per 100g.
10. "allergen_tags" — list from: dairy, gluten, nuts, eggs, soy, fish. Only include if the dish ACTUALLY contains these.
11. "meal_time" — one of: breakfast, lunch, dinner, snack.
12. "price_tier" — one of: low, medium, high.
13. "tag_list" — 3-6 descriptive tags (e.g., "spicy", "comfort", "high-protein", "vegan").

IMPORTANT:
- Each dish must be UNIQUE — different name, different recipe.
- Nutrition values must be REALISTIC per 100g.
- Vary the dishes within the cuisine: different proteins, cooking methods, sides.
- Include both common and less common dishes from this cuisine.
- Output valid JSON array only, no markdown."""


def generate_batch(cluster_id: int, batch_num: int, count: int) -> list[dict]:
    """Generate a batch of dishes for a given cluster."""
    cluster = CLUSTERS[cluster_id]

    user_prompt = f"""Generate {count} unique dishes from the "{cluster['name']}" cuisine.

Examples of this cuisine: {cluster['examples']}
Suitable meal times: {', '.join(cluster['meal_times'])}

This is batch {batch_num + 1} — make dishes DIFFERENT from common/obvious ones if batch > 1.
{"Focus on lesser-known, regional, or creative variations." if batch_num > 2 else ""}
{"Include some fusion or modern interpretations." if batch_num > 4 else ""}

Return a JSON array of {count} dish objects."""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.9,
                max_tokens=4096,
            )
            text = response.choices[0].message.content.strip()

            # Clean markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text.rsplit("```", 1)[0]
                text = text.strip()

            dishes = json.loads(text)
            if not isinstance(dishes, list):
                dishes = [dishes]

            # Sanitize and add cluster metadata
            for d in dishes:
                d["cuisine_cluster"] = cluster_id
                # Fix LLM sometimes returning ["none"] instead of []
                if "allergen_tags" in d:
                    d["allergen_tags"] = [
                        a for a in d["allergen_tags"]
                        if a.lower() not in ("none", "n/a", "")
                    ]

            return dishes

        except (json.JSONDecodeError, KeyError) as e:
            print(f"    Parse error (attempt {attempt + 1}): {e}")
            time.sleep(2)
        except Exception as e:
            print(f"    API error (attempt {attempt + 1}): {e}")
            time.sleep(5)

    return []


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--count", type=int, default=5000,
                        help="Total dishes to generate (default: 5000)")
    args = parser.parse_args()

    total_target = args.count
    target_per_cluster = (total_target + len(CLUSTERS) - 1) // len(CLUSTERS)
    batches_per_cluster = (target_per_cluster + BATCH_SIZE - 1) // BATCH_SIZE

    # Load progress if resuming
    all_dishes: list[dict] = []
    completed: dict[int, int] = {}  # cluster_id -> batches completed

    if PROGRESS_FILE.exists():
        progress = json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        all_dishes = progress.get("dishes", [])
        completed = {int(k): v for k, v in progress.get("completed", {}).items()}
        print(f"Resuming: {len(all_dishes)} dishes already generated")

    total_batches = batches_per_cluster * len(CLUSTERS)
    done_batches = sum(completed.values())

    for cluster_id in sorted(CLUSTERS.keys()):
        cluster_name = CLUSTERS[cluster_id]["name"]
        start_batch = completed.get(cluster_id, 0)

        if start_batch >= batches_per_cluster:
            continue

        print(f"\n=== Cluster {cluster_id}: {cluster_name} ===")

        for batch_num in range(start_batch, batches_per_cluster):
            remaining = target_per_cluster - (batch_num * BATCH_SIZE)
            count = min(BATCH_SIZE, remaining)
            if count <= 0:
                break

            done_batches += 1
            pct = done_batches / total_batches * 100
            print(f"  Batch {batch_num + 1}/{batches_per_cluster} ({count} dishes) [{pct:.0f}%]...", end=" ", flush=True)

            dishes = generate_batch(cluster_id, batch_num, count)
            if dishes:
                all_dishes.extend(dishes)
                print(f"OK ({len(dishes)} dishes)")
            else:
                print("FAILED — skipping")

            # Save progress every batch
            completed[cluster_id] = batch_num + 1
            PROGRESS_FILE.write_text(
                json.dumps({"dishes": all_dishes, "completed": completed}, ensure_ascii=False),
                encoding="utf-8",
            )

            # Rate limiting: small delay between requests
            time.sleep(0.5)

    # Load existing final dishes and append new ones
    existing: list[dict] = []
    if FINAL_FILE.exists():
        existing = json.loads(FINAL_FILE.read_text(encoding="utf-8"))
        print(f"\nAppending to existing {len(existing)} dishes in {FINAL_FILE.name}")

    id_offset = len(existing)
    for i, dish in enumerate(all_dishes):
        dish["id"] = id_offset + i + 1

    combined = existing + all_dishes
    FINAL_FILE.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Also save just the new batch to OUTPUT_FILE for reference
    OUTPUT_FILE.write_text(
        json.dumps(all_dishes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    print(f"\n=== Done! ===")
    print(f"New dishes generated: {len(all_dishes)}")
    print(f"Total in {FINAL_FILE.name}: {len(combined)}")
    print(f"Saved to: {FINAL_FILE}")

    # Stats
    from collections import Counter
    clusters = Counter(d.get("cuisine_cluster") for d in all_dishes)
    for cid, count in sorted(clusters.items()):
        print(f"  Cluster {cid} ({CLUSTERS[cid]['name']}): {count}")


if __name__ == "__main__":
    main()
