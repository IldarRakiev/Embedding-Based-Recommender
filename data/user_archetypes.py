"""
User archetype definitions for synthetic dataset generation.

1100 users total:
  - 300 pure archetypes (10 types x 30 each) — ~27%
  - 400 hybrids (2 archetypes mixed) — ~36%
  - 200 contradictory (goal vs behavior mismatch) — ~18%
  - 200 drifting (preference shift over time) — ~18%

Each user has fields compatible with user_static_to_text():
  goal_type, activity_level, gender, height_cm, weight_kg,
  has_diabetes, price_level, allergens, kcal_target, etc.
"""
from __future__ import annotations
import numpy as np

# ============================================================
# 10 BASE ARCHETYPES
# ============================================================

ARCHETYPES = [
    {
        "name": "fitness",
        "goal_type": "lose",
        "activity_level": "high",
        "budget": "medium",
        "allergens": [],
        "preferred_cuisines": [4, 2],  # Healthy + Asian
        "price_sensitivity": 0.4,
        "weight_range": (70, 95),
        "height_range": (165, 185),
        "kcal_modifier": 0.85,  # slight deficit
        "protein_modifier": 1.3,  # high protein
        "vegetarian": False,
        "diabetes_prob": 0.02,
    },
    {
        "name": "student",
        "goal_type": "keep",
        "activity_level": "low",
        "budget": "low",
        "allergens": [],
        "preferred_cuisines": [5, 3],  # Comfort + American
        "price_sensitivity": 0.85,
        "weight_range": (55, 85),
        "height_range": (160, 185),
        "kcal_modifier": 1.0,
        "protein_modifier": 1.0,
        "vegetarian": False,
        "diabetes_prob": 0.01,
    },
    {
        "name": "health_mom",
        "goal_type": "keep",
        "activity_level": "medium",
        "budget": "medium",
        "allergens": ["dairy"],
        "preferred_cuisines": [1, 4],  # Mediterranean + Healthy
        "price_sensitivity": 0.5,
        "weight_range": (55, 75),
        "height_range": (155, 172),
        "kcal_modifier": 0.95,
        "protein_modifier": 1.0,
        "vegetarian": False,
        "diabetes_prob": 0.03,
    },
    {
        "name": "muscle_builder",
        "goal_type": "gain",
        "activity_level": "high",
        "budget": "high",
        "allergens": [],
        "preferred_cuisines": [3, 0],  # American + Russian
        "price_sensitivity": 0.2,
        "weight_range": (80, 110),
        "height_range": (172, 195),
        "kcal_modifier": 1.2,
        "protein_modifier": 1.6,
        "vegetarian": False,
        "diabetes_prob": 0.01,
    },
    {
        "name": "vegetarian",
        "goal_type": "keep",
        "activity_level": "medium",
        "budget": "medium",
        "allergens": [],
        "preferred_cuisines": [1, 2],  # Mediterranean + Asian
        "price_sensitivity": 0.45,
        "weight_range": (50, 75),
        "height_range": (155, 180),
        "kcal_modifier": 1.0,
        "protein_modifier": 0.9,
        "vegetarian": True,
        "diabetes_prob": 0.02,
    },
    {
        "name": "gluten_free",
        "goal_type": "lose",
        "activity_level": "medium",
        "budget": "high",
        "allergens": ["gluten"],
        "preferred_cuisines": [4, 1],  # Healthy + Mediterranean
        "price_sensitivity": 0.25,
        "weight_range": (60, 85),
        "height_range": (158, 178),
        "kcal_modifier": 0.9,
        "protein_modifier": 1.1,
        "vegetarian": False,
        "diabetes_prob": 0.05,
    },
    {
        "name": "traditional",
        "goal_type": "keep",
        "activity_level": "low",
        "budget": "low",
        "allergens": [],
        "preferred_cuisines": [0],  # Russian only
        "price_sensitivity": 0.8,
        "weight_range": (65, 100),
        "height_range": (160, 185),
        "kcal_modifier": 1.05,
        "protein_modifier": 1.0,
        "vegetarian": False,
        "diabetes_prob": 0.08,
    },
    {
        "name": "allergy_prone",
        "goal_type": "keep",
        "activity_level": "medium",
        "budget": "medium",
        "allergens": ["dairy", "nuts", "gluten"],
        "preferred_cuisines": [4],  # Healthy
        "price_sensitivity": 0.4,
        "weight_range": (55, 80),
        "height_range": (158, 180),
        "kcal_modifier": 1.0,
        "protein_modifier": 1.0,
        "vegetarian": False,
        "diabetes_prob": 0.04,
    },
    {
        "name": "strict_dieter",
        "goal_type": "lose",
        "activity_level": "high",
        "budget": "medium",
        "allergens": [],
        "preferred_cuisines": [4, 2],  # Healthy + Asian
        "price_sensitivity": 0.5,
        "weight_range": (75, 110),
        "height_range": (162, 185),
        "kcal_modifier": 0.75,
        "protein_modifier": 1.2,
        "vegetarian": False,
        "diabetes_prob": 0.06,
    },
    {
        "name": "foodie_explorer",
        "goal_type": "keep",
        "activity_level": "medium",
        "budget": "high",
        "allergens": [],
        "preferred_cuisines": [0, 1, 2, 3, 4, 5, 6, 7],  # All
        "price_sensitivity": 0.15,
        "weight_range": (60, 90),
        "height_range": (160, 185),
        "kcal_modifier": 1.0,
        "protein_modifier": 1.0,
        "vegetarian": False,
        "diabetes_prob": 0.03,
    },
]

# Meat-containing tags for vegetarian filtering
MEAT_TAGS = {"chicken", "beef", "pork", "lamb", "meat", "poultry", "BBQ", "seafood"}
MEAT_INGREDIENTS = {
    "beef", "pork", "chicken", "lamb", "bacon", "ham", "sausage",
    "ground beef", "ground meat", "mixed meat", "beef tenderloin",
    "beef chunks", "pork ribs", "chicken breast", "chicken wings",
    "chashu pork", "smoked sausage", "pancetta", "guanciale",
    "clams", "shrimp", "salmon", "tuna", "fish", "fish fillet",
    "white fish fillet", "sea bass", "sea bream",
}


def _mifflin_st_jeor(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """Basal metabolic rate (kcal/day) via Mifflin-St Jeor equation."""
    if gender == "male":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161


def _activity_multiplier(level: str) -> float:
    return {"low": 1.2, "medium": 1.55, "high": 1.725}.get(level, 1.4)


def generate_users(rng: np.random.Generator, target_count: int = 1100) -> list[dict]:
    """Generate user profiles: 300 archetype + 400 hybrid + 200 contradictory + 200 drifting."""
    users: list[dict] = []
    user_id = 1

    n_archetype = 300
    n_hybrid = 400
    n_contradictory = 200
    n_drifting = target_count - n_archetype - n_hybrid - n_contradictory

    # --- Pure archetypes ---
    per_arch = n_archetype // len(ARCHETYPES)
    for arch in ARCHETYPES:
        for _ in range(per_arch):
            user = _create_user_from_archetype(rng, user_id, arch, "archetype")
            users.append(user)
            user_id += 1

    # --- Hybrids: mix 2 archetypes ---
    for _ in range(n_hybrid):
        a1, a2 = rng.choice(len(ARCHETYPES), size=2, replace=False)
        arch1, arch2 = ARCHETYPES[a1], ARCHETYPES[a2]

        # Merge: take profile from one, cuisines from both
        primary = arch1 if rng.random() < 0.5 else arch2
        merged = dict(primary)
        merged["preferred_cuisines"] = list(set(arch1["preferred_cuisines"] + arch2["preferred_cuisines"]))
        merged["price_sensitivity"] = (arch1["price_sensitivity"] + arch2["price_sensitivity"]) / 2
        # Sometimes take goal from other archetype
        if rng.random() < 0.3:
            merged["goal_type"] = arch2["goal_type"]
        # Sometimes mix allergens
        if arch2["allergens"] and rng.random() < 0.4:
            merged["allergens"] = list(set(arch1["allergens"] + arch2["allergens"]))

        user = _create_user_from_archetype(rng, user_id, merged, "hybrid")
        user["_source_archetypes"] = [arch1["name"], arch2["name"]]
        users.append(user)
        user_id += 1

    # --- Contradictory: goal vs behavior mismatch ---
    for _ in range(n_contradictory):
        arch = ARCHETYPES[rng.integers(0, len(ARCHETYPES))]
        user = _create_user_from_archetype(rng, user_id, arch, "contradictory")
        # Flip behavior tendencies
        user["_contradiction_rate"] = rng.uniform(0.2, 0.4)  # 20-40% contradictory actions
        users.append(user)
        user_id += 1

    # --- Drifting: preference changes over time ---
    for _ in range(n_drifting):
        a1, a2 = rng.choice(len(ARCHETYPES), size=2, replace=False)
        arch1, arch2 = ARCHETYPES[a1], ARCHETYPES[a2]
        user = _create_user_from_archetype(rng, user_id, arch1, "drifting")
        user["_drift_target_cuisines"] = list(arch2["preferred_cuisines"])
        user["_drift_target_goal"] = arch2["goal_type"]
        user["_source_archetypes"] = [arch1["name"], arch2["name"]]
        users.append(user)
        user_id += 1

    return users[:target_count]


def _create_user_from_archetype(
    rng: np.random.Generator,
    user_id: int,
    arch: dict,
    user_type: str,
) -> dict:
    """Create a single user from an archetype with random variation."""
    gender = rng.choice(["male", "female"])
    weight = rng.uniform(*arch["weight_range"])
    height = rng.uniform(*arch["height_range"])
    age = int(rng.uniform(20, 55))

    bmr = _mifflin_st_jeor(weight, height, age, gender)
    tdee = bmr * _activity_multiplier(arch["activity_level"])
    kcal_target = tdee * arch.get("kcal_modifier", 1.0)

    protein_per_kg = 1.6 * arch.get("protein_modifier", 1.0)
    protein_target = weight * protein_per_kg
    fat_target = kcal_target * 0.28 / 9  # ~28% fat
    carbs_target = (kcal_target - protein_target * 4 - fat_target * 9) / 4

    has_diabetes = rng.random() < arch.get("diabetes_prob", 0.03)

    return {
        "user_id": user_id,
        "user_type": user_type,
        "archetype_name": arch.get("name", "hybrid"),
        "goal_type": arch["goal_type"],
        "activity_level": arch["activity_level"],
        "gender": gender,
        "age": age,
        "height_cm": round(height, 1),
        "weight_kg": round(weight, 1),
        "has_diabetes": has_diabetes,
        "price_level": arch["budget"],
        "allergens": list(arch["allergens"]),
        "kcal_target": round(kcal_target),
        "protein_target_g": round(protein_target),
        "fat_target_g": round(fat_target),
        "carbs_target_g": round(max(0, carbs_target)),
        # Hidden fields for interaction generation
        "_preferred_cuisines": list(arch["preferred_cuisines"]),
        "_price_sensitivity": arch["price_sensitivity"],
        "_vegetarian": arch.get("vegetarian", False),
        "_contradiction_rate": 0.0,
        "_drift_target_cuisines": None,
        "_drift_target_goal": None,
        "_source_archetypes": None,
    }
