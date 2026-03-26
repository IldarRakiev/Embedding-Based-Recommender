# src/text_builders.py
"""
Build rich text representations of dishes and users for embedding.

This module is extracted from a production food recommendation system.
All database dependencies have been removed — data is passed as plain dicts.

Key design principle (see case_study_spec section 0.3):
    dish_to_rich_text() accepts experiment flags with defaults matching the
    *baseline* configuration (include_recipe=True, include_macro_tokens=True).
    Improvement notebooks simply pass different flag values — the src/ code
    itself never changes between notebooks.
"""
from __future__ import annotations

import re
from typing import Any


# ============================================
# MACRO TOKENS — discrete nutrient categories
# ============================================

def macro_tokens(kcal: float, p: float, f: float, c: float) -> list[str]:
    """Generate discrete macro-nutrient category tokens.

    Used in the *baseline* text representation (include_macro_tokens=True).
    Discrete thresholds create artificial cluster boundaries in embedding
    space — this is one of the things we ablate in Notebook 3.

    Args:
        kcal: Calories per 100 g.
        p: Protein grams per 100 g.
        f: Fat grams per 100 g.
        c: Carbs grams per 100 g.

    Returns:
        List of tokens e.g. ``["HIGH_PROTEIN", "LOW_FAT", "CARB_DOMINANT", "LOW_CALORIE"]``.
    """
    tokens = []

    if p >= 40:
        tokens.append("VERY_HIGH_PROTEIN")
    elif p >= 25:
        tokens.append("HIGH_PROTEIN")
    elif p >= 15:
        tokens.append("MEDIUM_PROTEIN")
    else:
        tokens.append("LOW_PROTEIN")

    if f >= 30:
        tokens.append("HIGH_FAT")
    elif f >= 15:
        tokens.append("MEDIUM_FAT")
    else:
        tokens.append("LOW_FAT")

    if c >= 50:
        tokens.append("HIGH_CARB")
    elif c >= 25:
        tokens.append("MEDIUM_CARB")
    else:
        tokens.append("LOW_CARB")

    total = max(p + f + c, 1e-6)
    pr, cr, fr = p / total, c / total, f / total
    if pr >= cr and pr >= fr:
        tokens.append("PROTEIN_DOMINANT")
    elif cr >= pr and cr >= fr:
        tokens.append("CARB_DOMINANT")
    else:
        tokens.append("FAT_DOMINANT")

    if kcal < 300:
        tokens.append("LOW_CALORIE")
    elif kcal < 600:
        tokens.append("MEDIUM_CALORIE")
    else:
        tokens.append("HIGH_CALORIE")

    return tokens


# ============================================
# INGREDIENT EXTRACTION
# ============================================

_COOKING_VERBS_RE = re.compile(
    r"\b(?:add|mix|combine|cook|bake|fry|boil|stir|whisk|chop|slice|dice|"
    r"season|marinate|simmer|roast|grill|blend|mince|peel|cut|fold)\b",
    re.IGNORECASE,
)


def extract_ingredients(recipe_text: str) -> list[str]:
    """Extract ingredient names from a recipe text using heuristic parsing.

    Supports:
    - Bullet lists (lines starting with ``-``, ``*``, or ``•``)
    - Numbered lists (``1. ..``, ``1) ..``)
    - Comma-separated items preceding a cooking verb (fallback)

    Args:
        recipe_text: Raw recipe text.

    Returns:
        Deduplicated list of ingredient strings.

    Example:
        >>> extract_ingredients("- 2 chicken breasts\\n- 1 cup rice\\nCook until done.")
        ['2 chicken breasts', '1 cup rice']
    """
    if not recipe_text or not recipe_text.strip():
        return []

    ingredients: list[str] = []
    seen: set[str] = set()

    for line in recipe_text.splitlines():
        stripped = line.strip()
        match = re.match(r"^(?:[-*•]|\d+[.)]\s*)\s*(.+)", stripped)
        if match:
            item = match.group(1).strip().rstrip(".,;")
            if item and not _COOKING_VERBS_RE.search(item):
                key = item.lower()
                if key not in seen:
                    seen.add(key)
                    ingredients.append(item)

    if ingredients:
        return ingredients

    # Fallback: comma-separated before a cooking verb
    for line in recipe_text.splitlines():
        verb_match = _COOKING_VERBS_RE.search(line)
        if verb_match:
            before = line[: verb_match.start()]
            parts = [p.strip().rstrip(".,;") for p in before.split(",")]
            for p in parts:
                if p and len(p) > 1:
                    key = p.lower()
                    if key not in seen:
                        seen.add(key)
                        ingredients.append(p)

    return ingredients


# ============================================
# DISH TEXT BUILDER  (parametrized for experiments)
# ============================================

_CONTEXT_DESCRIPTIONS: dict[str, str] = {
    "breakfast": "CONTEXT: breakfast meal, energizing morning food",
    "lunch": "CONTEXT: lunch, main meal of the day, substantial",
    "dinner": "CONTEXT: dinner, evening meal, lighter option",
    "snack": "CONTEXT: snack, quick light bite between meals",
}


def dish_to_rich_text(
    dish: dict,
    tags: list[str] | None = None,
    # --- Experiment flags (defaults = baseline behavior) ---
    include_recipe: bool = True,
    include_macro_tokens: bool = True,
    include_ratios: bool = False,
    include_ingredients: bool = False,
    context: str | None = None,
) -> str:
    """Build a rich text representation of a dish for embedding.

    The flags control which components are included, enabling A/B comparison
    of text representations **without modifying this file**.

    Defaults reflect the *baseline* configuration used in Notebook 2.
    Notebook 3 passes different flag combinations to measure each component's
    impact on retrieval quality.

    Args:
        dish: Dict with keys: ``name``, ``description`` (or ``short_description``),
              ``recipe_text``, ``calories`` (or ``kcal_per_100g``),
              ``protein_g`` (or ``protein_g_per_100g``), ``fat_g``, ``carbs_g``,
              ``fiber_g``.
        tags: List of tag strings.
        include_recipe: Include recipe instructions (baseline: True).
            Hypothesis: recipe text adds noise — sentences describe cooking
            procedure, not dish identity. Removing it should improve retrieval.
        include_macro_tokens: Include discrete HIGH_PROTEIN/LOW_FAT tokens
            (baseline: True).  Hypothesis: discrete thresholds create hard
            boundaries in embedding space; continuous ratios are smoother.
        include_ratios: Include continuous protein_ratio/fat_ratio/carb_ratio
            (baseline: False).  Improvement over discrete tokens.
        include_ingredients: Extract and include ingredient list from recipe
            (baseline: False).  Ingredients signal dish identity without
            procedural noise.
        context: Meal context string prepended to the text.  Sentence-
            transformers weight early tokens more, so context placed first
            biases retrieval toward the right meal slot.

    Returns:
        Rich text string ready for sentence-transformer encoding.

    Examples:
        >>> # Baseline (all defaults)
        >>> text = dish_to_rich_text({"name": "Grilled Chicken", "calories": 165,
        ...                           "protein_g": 31, "fat_g": 4, "carbs_g": 0})
        >>> # Improvement experiment: no recipe, no macro tokens, with ratios
        >>> text = dish_to_rich_text(dish, tags,
        ...     include_recipe=False, include_macro_tokens=False, include_ratios=True)
    """
    name = dish.get("name") or ""
    desc = dish.get("description") or dish.get("short_description") or ""
    recipe = dish.get("recipe_text") or dish.get("instructions") or ""

    kcal = float(dish.get("calories") or dish.get("kcal_per_100g") or 0)
    p = float(dish.get("protein_g") or dish.get("protein_g_per_100g") or 0)
    f = float(dish.get("fat_g") or dish.get("fat_g_per_100g") or 0)
    c = float(dish.get("carbs_g") or dish.get("carbs_g_per_100g") or 0)
    fiber = float(dish.get("fiber_g") or dish.get("fiber_g_per_100g") or 0)

    parts: list[str] = []

    # Context goes FIRST — higher weight for early tokens
    if context:
        parts.append(_CONTEXT_DESCRIPTIONS.get(context, f"CONTEXT: {context}"))

    # Core fields (always present)
    parts.append(f"DISH_NAME: {name}")
    if desc:
        parts.append(f"DESCRIPTION: {desc}")

    # Nutrition line
    nutrition = (
        f"NUTRITION: {kcal:.0f} kcal, {p:.0f}g protein, "
        f"{f:.0f}g fat, {c:.0f}g carbs, {fiber:.0f}g fiber"
    )
    if include_ratios and kcal > 0:
        pr = (p * 4) / kcal
        fr = (f * 9) / kcal
        cr = (c * 4) / kcal
        nutrition += f" | protein_ratio: {pr:.2f}, fat_ratio: {fr:.2f}, carb_ratio: {cr:.2f}"
    parts.append(nutrition)

    # Discrete macro tokens (baseline only)
    if include_macro_tokens:
        tokens = macro_tokens(kcal, p, f, c)
        parts.append("MACRO_TAGS: " + " ".join(tokens))

    # Tags
    if tags:
        parts.append("TAGS: " + " ".join(tags))

    # Ingredients extracted from recipe
    if include_ingredients and recipe:
        ingredients = extract_ingredients(recipe)
        if ingredients:
            parts.append("INGREDIENTS: " + ", ".join(ingredients))

    # Recipe text (baseline: included, improvement: removed)
    if include_recipe and recipe:
        parts.append("RECIPE_BRIEF: " + recipe[:400])

    return "\n".join(parts)


# ============================================
# USER TEXT BUILDERS  (unchanged from production)
# ============================================

def user_static_to_text(
    profile: dict | None,
    allergens: list[dict] | None,
    habits: dict | None,
) -> str:
    """Build static user profile text (goal, allergies, medical, budget)."""
    parts = ["USER STATIC PROFILE:"]

    if profile:
        goal = profile.get("goal_type")
        if goal:
            goal_text = {
                "lose": "GOAL: WEIGHT LOSS",
                "keep": "GOAL: MAINTAIN WEIGHT",
                "gain": "GOAL: WEIGHT GAIN",
            }.get(goal, f"GOAL: {goal.upper()}")
            parts.append(goal_text)

        activity = profile.get("activity_level")
        if activity:
            parts.append(f"ACTIVITY: {activity.upper()}")

        conditions = []
        if profile.get("has_diabetes"):
            conditions.append("DIABETES")
        if profile.get("other_conditions"):
            conditions.append(profile["other_conditions"].upper())
        if conditions:
            parts.append("MEDICAL: " + ", ".join(conditions))

        price_level = profile.get("price_level")
        if price_level:
            parts.append(f"BUDGET: {price_level.upper()}")

    if allergens:
        names = [a.get("allergen_name", "").upper() for a in allergens if a.get("allergen_name")]
        if names:
            parts.append("ALLERGIES: " + ", ".join(f"NO {a}" for a in names))

    if habits:
        habit_parts = []
        if habits.get("smoking"):
            habit_parts.append("SMOKER")
        if habits.get("alcohol") and habits.get("alcohol") != "none":
            habit_parts.append(f"ALCOHOL_{habits['alcohol'].upper()}")
        if habit_parts:
            parts.append("HABITS: " + ", ".join(habit_parts))

    return "\n".join(parts)


def user_dynamic_to_text(
    daily_targets: dict | None,
    day_stats: dict | None,
    meal_time: str | None = None,
) -> str:
    """Build dynamic user state text (remaining macros, meal time)."""
    parts = ["TODAY STATUS:"]

    if day_stats:
        kcal_c = int(day_stats.get("kcal_consumed") or 0)
        p_c = float(day_stats.get("protein_g") or 0)
        f_c = float(day_stats.get("fat_g") or 0)
        c_c = float(day_stats.get("carbs_g") or 0)
        parts.append(f"CONSUMED TODAY: {kcal_c} kcal, {p_c:.0f}g protein, {f_c:.0f}g fat, {c_c:.0f}g carbs")

    if daily_targets:
        t_kcal = int(daily_targets.get("kcal_target") or 2000)
        t_p = float(daily_targets.get("protein_target_g") or 100)
        t_f = float(daily_targets.get("fat_target_g") or 65)
        t_c = float(daily_targets.get("carbs_target_g") or 250)

        c_kcal = int(day_stats.get("kcal_consumed") or 0) if day_stats else 0
        c_p = float(day_stats.get("protein_g") or 0) if day_stats else 0
        c_f = float(day_stats.get("fat_g") or 0) if day_stats else 0
        c_c2 = float(day_stats.get("carbs_g") or 0) if day_stats else 0

        r_kcal = max(0, t_kcal - c_kcal)
        r_p = max(0, t_p - c_p)
        r_f = max(0, t_f - c_f)
        r_c = max(0, t_c - c_c2)
        parts.append(f"REMAINING: {r_kcal} kcal, {r_p:.0f}g protein, {r_f:.0f}g fat, {r_c:.0f}g carbs")

        needs = []
        if r_p / max(t_p, 1) > 0.5:
            needs.append("NEEDS_PROTEIN")
        if r_c / max(t_c, 1) > 0.5:
            needs.append("NEEDS_CARBS")
        if r_kcal < 300:
            needs.append("LOW_CALORIES_LEFT")
        if needs:
            parts.append("NEEDS: " + " ".join(needs))

    if meal_time:
        meal_text = {
            "breakfast": "MEAL_TIME: BREAKFAST - looking for energizing start",
            "lunch": "MEAL_TIME: LUNCH - main meal of the day",
            "dinner": "MEAL_TIME: DINNER - lighter evening meal",
            "snack": "MEAL_TIME: SNACK - quick light bite",
        }.get(meal_time.lower(), f"MEAL_TIME: {meal_time.upper()}")
        parts.append(meal_text)

    return "\n".join(parts)
