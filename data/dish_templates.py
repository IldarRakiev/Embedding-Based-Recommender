"""
Dish template definitions for synthetic dataset generation.

5000 dishes across 8 cuisine clusters, generated via combinatorial
expansion of base families × cooking methods × garnishes × variations.

Each dish has fields compatible with dish_to_rich_text():
  name, description, recipe_text, calories, protein_g, fat_g, carbs_g,
  fiber_g, tag_list, allergen_tags, meal_time, price_tier, cuisine_cluster,
  ingredients
"""
from __future__ import annotations
import itertools
import numpy as np

# ============================================================
# CUISINE CLUSTERS
# ============================================================
CUISINE_NAMES = {
    0: "Russian/Slavic",
    1: "Mediterranean",
    2: "Asian",
    3: "American",
    4: "Healthy/Diet",
    5: "Comfort/Hearty",
    6: "Breakfast",
    7: "Desserts/Snacks",
}

# ============================================================
# ALLERGEN TYPES
# ============================================================
ALLERGENS = ["dairy", "gluten", "nuts", "eggs", "soy", "fish"]

# ============================================================
# COOKING INSTRUCTIONS — procedural noise for recipe_text
# ============================================================
COOKING_STEPS = [
    "Preheat the oven to {temp}°C. Line a baking sheet with parchment paper.",
    "Heat {oil} in a large skillet over medium-high heat until shimmering.",
    "Bring a large pot of salted water to a rolling boil.",
    "In a mixing bowl, combine the dry ingredients and whisk until smooth.",
    "Marinate for at least {time} minutes in the refrigerator.",
    "Sauté the onions and garlic for 3-4 minutes until translucent and fragrant.",
    "Reduce the heat to low and let it simmer for {time} minutes, stirring occasionally.",
    "Transfer to a preheated {temp}°C oven and bake for {time} minutes.",
    "Stir constantly for 5 minutes to prevent sticking and ensure even cooking.",
    "Let rest for 10 minutes before slicing. This allows the juices to redistribute.",
    "Season generously with salt and freshly ground black pepper to taste.",
    "Drain on paper towels and serve immediately while still hot and crispy.",
    "Fold gently to avoid deflating the mixture. Use a rubber spatula.",
    "Cover with aluminum foil and let it steam for 15 minutes undisturbed.",
    "Garnish with fresh herbs and a squeeze of lemon juice before serving.",
]

TEMPS = [160, 170, 180, 190, 200, 210, 220]
TIMES = [10, 15, 20, 25, 30, 40, 45, 60]
OILS = ["olive oil", "vegetable oil", "butter", "coconut oil", "sesame oil"]


def _random_cooking_steps(rng: np.random.Generator, n: int = 3) -> str:
    """Generate n random cooking instruction steps (procedural noise)."""
    steps = rng.choice(COOKING_STEPS, size=min(n, len(COOKING_STEPS)), replace=False)
    result = []
    for s in steps:
        s = s.replace("{temp}", str(rng.choice(TEMPS)))
        s = s.replace("{time}", str(rng.choice(TIMES)))
        s = s.replace("{oil}", str(rng.choice(OILS)))
        result.append(s)
    return " ".join(result)


# ============================================================
# DISH FAMILIES PER CLUSTER
# Each family: base_name, description_template, base_nutrition,
#   ingredient_pool, allergen_set, meal_times, price_tiers
# ============================================================

# Nutrition: (kcal, protein, fat, carbs, fiber) per 100g
_FAMILIES = {
    # --- Cluster 0: Russian/Slavic ---
    0: [
        {
            "names": ["Borscht", "Beetroot Soup", "Ukrainian Borscht"],
            "proteins": ["beef", "pork", "chicken"],
            "garnishes": ["with Sour Cream", "with Dill", "with Garlic Bread", "Classic"],
            "desc": "Traditional Slavic beet soup with rich broth and vegetables",
            "nutrition": (45, 3, 1.5, 6, 1.5),
            "ingredients": ["beetroot", "cabbage", "potato", "carrot", "onion", "tomato paste", "beef broth"],
            "allergens": [],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["low", "medium"],
            "tags": ["soup", "slavic", "traditional"],
        },
        {
            "names": ["Pelmeni", "Russian Dumplings", "Siberian Pelmeni"],
            "proteins": ["beef", "pork", "mixed meat", "chicken"],
            "garnishes": ["with Sour Cream", "with Butter", "with Vinegar", "Fried"],
            "desc": "Hearty Russian dumplings filled with seasoned meat",
            "nutrition": (220, 12, 10, 22, 1),
            "ingredients": ["flour", "egg", "ground meat", "onion", "salt", "pepper"],
            "allergens": ["gluten", "eggs"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["low", "medium"],
            "tags": ["dumplings", "slavic", "traditional", "hearty"],
        },
        {
            "names": ["Bliny", "Russian Crepes", "Blini"],
            "proteins": [],
            "garnishes": ["with Caviar", "with Sour Cream", "with Jam", "with Honey", "with Smoked Salmon"],
            "desc": "Thin Russian pancakes, a classic breakfast or celebration dish",
            "nutrition": (230, 7, 9, 30, 1),
            "ingredients": ["flour", "milk", "eggs", "sugar", "butter"],
            "allergens": ["gluten", "dairy", "eggs"],
            "meal_times": ["breakfast", "snack"],
            "price_tiers": ["low", "medium"],
            "tags": ["pancakes", "slavic", "traditional", "breakfast"],
        },
        {
            "names": ["Solyanka", "Russian Sour Soup"],
            "proteins": ["smoked sausage", "beef", "mixed meats"],
            "garnishes": ["with Lemon", "with Olives", "Classic", "Spicy"],
            "desc": "Thick, spicy and sour Russian soup with mixed meats",
            "nutrition": (65, 5, 3, 4, 1),
            "ingredients": ["smoked sausage", "pickles", "olives", "tomato paste", "onion", "lemon"],
            "allergens": [],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium"],
            "tags": ["soup", "slavic", "spicy"],
        },
        {
            "names": ["Kasha", "Buckwheat Porridge", "Buckwheat"],
            "proteins": [],
            "garnishes": ["with Mushrooms", "with Butter", "with Milk", "Plain", "with Fried Onions"],
            "desc": "Nutritious buckwheat porridge, a Russian staple",
            "nutrition": (130, 5, 2, 25, 3),
            "ingredients": ["buckwheat", "water", "salt", "butter"],
            "allergens": ["dairy"],
            "meal_times": ["breakfast", "lunch", "dinner"],
            "price_tiers": ["low"],
            "tags": ["porridge", "slavic", "healthy", "grain"],
        },
        {
            "names": ["Shchi", "Cabbage Soup", "Russian Cabbage Soup"],
            "proteins": ["beef", "pork"],
            "garnishes": ["with Sour Cream", "Classic", "with Dill"],
            "desc": "Classic Russian cabbage soup, comfort food for cold days",
            "nutrition": (35, 2, 1, 5, 1.5),
            "ingredients": ["cabbage", "potato", "carrot", "onion", "beef broth", "bay leaf"],
            "allergens": [],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["low"],
            "tags": ["soup", "slavic", "traditional", "low-calorie"],
        },
        {
            "names": ["Beef Stroganoff", "Stroganoff"],
            "proteins": ["beef", "mushroom"],
            "garnishes": ["with Rice", "with Pasta", "with Mashed Potatoes", "Classic"],
            "desc": "Tender beef strips in a rich creamy mushroom sauce",
            "nutrition": (180, 15, 11, 6, 0.5),
            "ingredients": ["beef tenderloin", "mushrooms", "onion", "sour cream", "flour", "butter"],
            "allergens": ["dairy", "gluten"],
            "meal_times": ["dinner"],
            "price_tiers": ["medium", "high"],
            "tags": ["slavic", "creamy", "classic", "dinner"],
        },
        {
            "names": ["Pirozhki", "Russian Pies", "Baked Pirozhki"],
            "proteins": [],
            "garnishes": ["with Meat", "with Cabbage", "with Potato", "with Egg and Onion", "with Mushrooms"],
            "desc": "Small Russian baked or fried pies with various fillings",
            "nutrition": (260, 8, 12, 30, 1),
            "ingredients": ["flour", "yeast", "milk", "egg", "butter", "filling"],
            "allergens": ["gluten", "dairy", "eggs"],
            "meal_times": ["lunch", "snack"],
            "price_tiers": ["low", "medium"],
            "tags": ["pastry", "slavic", "traditional", "baked"],
        },
    ],
    # --- Cluster 1: Mediterranean ---
    1: [
        {
            "names": ["Greek Salad", "Horiatiki Salad"],
            "proteins": [],
            "garnishes": ["Classic", "with Extra Feta", "with Olives", "with Avocado"],
            "desc": "Fresh Mediterranean salad with feta cheese and olives",
            "nutrition": (90, 4, 7, 4, 1.5),
            "ingredients": ["tomato", "cucumber", "red onion", "feta cheese", "olives", "olive oil", "oregano"],
            "allergens": ["dairy"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["low", "medium"],
            "tags": ["salad", "mediterranean", "fresh", "vegetarian"],
        },
        {
            "names": ["Pasta Carbonara", "Spaghetti Carbonara"],
            "proteins": ["pancetta", "bacon", "guanciale"],
            "garnishes": ["Classic", "with Extra Parmesan", "with Black Pepper"],
            "desc": "Creamy Italian pasta with egg, cheese and cured pork",
            "nutrition": (310, 14, 15, 32, 1.5),
            "ingredients": ["spaghetti", "eggs", "parmesan", "pancetta", "black pepper"],
            "allergens": ["gluten", "dairy", "eggs"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium"],
            "tags": ["pasta", "italian", "creamy", "classic"],
        },
        {
            "names": ["Hummus", "Chickpea Hummus"],
            "proteins": [],
            "garnishes": ["Classic", "with Paprika", "with Pine Nuts", "with Roasted Garlic", "Spicy"],
            "desc": "Smooth chickpea dip with tahini and lemon",
            "nutrition": (170, 8, 10, 14, 4),
            "ingredients": ["chickpeas", "tahini", "lemon juice", "garlic", "olive oil", "cumin"],
            "allergens": ["nuts"],
            "meal_times": ["lunch", "snack"],
            "price_tiers": ["low"],
            "tags": ["dip", "mediterranean", "vegan", "healthy"],
        },
        {
            "names": ["Grilled Sea Bass", "Mediterranean Fish", "Baked Sea Bream"],
            "proteins": ["sea bass", "sea bream", "salmon"],
            "garnishes": ["with Lemon", "with Herbs", "with Capers", "with Tomato Sauce"],
            "desc": "Fresh fish grilled Mediterranean style with herbs and lemon",
            "nutrition": (140, 22, 5, 1, 0),
            "ingredients": ["fish fillet", "olive oil", "lemon", "garlic", "rosemary", "thyme"],
            "allergens": ["fish"],
            "meal_times": ["dinner"],
            "price_tiers": ["medium", "high"],
            "tags": ["fish", "mediterranean", "grilled", "healthy"],
        },
        {
            "names": ["Moussaka", "Greek Moussaka"],
            "proteins": ["lamb", "beef"],
            "garnishes": ["Classic", "with Bechamel", "Light Version"],
            "desc": "Layered Greek casserole with eggplant, meat sauce and bechamel",
            "nutrition": (160, 10, 9, 12, 2),
            "ingredients": ["eggplant", "ground meat", "tomato sauce", "onion", "bechamel", "cinnamon"],
            "allergens": ["dairy", "gluten", "eggs"],
            "meal_times": ["dinner"],
            "price_tiers": ["medium"],
            "tags": ["casserole", "greek", "traditional", "baked"],
        },
        {
            "names": ["Falafel", "Chickpea Falafel"],
            "proteins": [],
            "garnishes": ["with Tahini", "in Pita", "with Hummus", "Salad Bowl", "Wrap"],
            "desc": "Crispy fried chickpea patties with Middle Eastern spices",
            "nutrition": (330, 13, 18, 32, 5),
            "ingredients": ["chickpeas", "parsley", "cilantro", "onion", "garlic", "cumin", "coriander"],
            "allergens": [],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["low", "medium"],
            "tags": ["mediterranean", "vegan", "fried", "street-food"],
        },
        {
            "names": ["Risotto", "Italian Risotto"],
            "proteins": ["mushroom", "seafood", "chicken"],
            "garnishes": ["Milanese", "with Parmesan", "with Truffle Oil", "with Peas"],
            "desc": "Creamy Italian rice dish slowly cooked in broth",
            "nutrition": (170, 5, 6, 24, 0.5),
            "ingredients": ["arborio rice", "broth", "onion", "white wine", "parmesan", "butter"],
            "allergens": ["dairy"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium", "high"],
            "tags": ["rice", "italian", "creamy", "elegant"],
        },
    ],
    # --- Cluster 2: Asian ---
    2: [
        {
            "names": ["Chicken Ramen", "Tonkotsu Ramen", "Miso Ramen"],
            "proteins": ["chicken", "pork", "tofu"],
            "garnishes": ["with Soft Egg", "with Nori", "Spicy", "with Corn", "Classic"],
            "desc": "Rich Japanese noodle soup with savory broth and toppings",
            "nutrition": (190, 12, 7, 22, 1),
            "ingredients": ["ramen noodles", "broth", "chashu pork", "soft-boiled egg", "nori", "green onion"],
            "allergens": ["gluten", "eggs", "soy"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium"],
            "tags": ["soup", "japanese", "noodles", "umami"],
        },
        {
            "names": ["Thai Green Curry", "Green Curry", "Thai Curry"],
            "proteins": ["chicken", "shrimp", "tofu", "beef"],
            "garnishes": ["with Jasmine Rice", "with Naan", "Spicy", "Mild"],
            "desc": "Aromatic Thai curry with coconut milk and fresh herbs",
            "nutrition": (145, 10, 9, 8, 1.5),
            "ingredients": ["coconut milk", "green curry paste", "bamboo shoots", "basil", "fish sauce", "lime"],
            "allergens": ["fish"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium"],
            "tags": ["curry", "thai", "spicy", "coconut"],
        },
        {
            "names": ["Fried Rice", "Egg Fried Rice", "Yangzhou Fried Rice"],
            "proteins": ["shrimp", "chicken", "pork", "vegetable"],
            "garnishes": ["Classic", "with Extra Vegetables", "Spicy", "with Pineapple"],
            "desc": "Wok-fried rice with vegetables, eggs and protein",
            "nutrition": (170, 7, 5, 26, 1),
            "ingredients": ["jasmine rice", "eggs", "soy sauce", "vegetables", "sesame oil", "green onion"],
            "allergens": ["eggs", "soy"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["low", "medium"],
            "tags": ["rice", "chinese", "wok", "quick"],
        },
        {
            "names": ["Sushi Roll", "Maki Roll", "California Roll"],
            "proteins": ["salmon", "tuna", "shrimp", "avocado"],
            "garnishes": ["with Wasabi", "with Ginger", "Spicy Mayo", "Tempura Style"],
            "desc": "Japanese rice and fish rolls wrapped in nori seaweed",
            "nutrition": (150, 8, 3, 25, 1),
            "ingredients": ["sushi rice", "nori", "fish", "avocado", "cucumber", "rice vinegar"],
            "allergens": ["fish", "soy"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium", "high"],
            "tags": ["sushi", "japanese", "raw", "elegant"],
        },
        {
            "names": ["Pad Thai", "Thai Stir-Fried Noodles"],
            "proteins": ["shrimp", "chicken", "tofu"],
            "garnishes": ["Classic", "with Extra Peanuts", "Spicy", "with Lime"],
            "desc": "Sweet and tangy Thai stir-fried rice noodles with peanuts",
            "nutrition": (210, 10, 8, 28, 1.5),
            "ingredients": ["rice noodles", "eggs", "bean sprouts", "peanuts", "tamarind", "fish sauce", "lime"],
            "allergens": ["eggs", "nuts", "fish"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium"],
            "tags": ["noodles", "thai", "stir-fry", "sweet"],
        },
        {
            "names": ["Teriyaki Bowl", "Teriyaki Chicken", "Teriyaki Salmon"],
            "proteins": ["chicken", "salmon", "tofu", "beef"],
            "garnishes": ["with Rice", "with Vegetables", "with Edamame", "Glazed"],
            "desc": "Japanese teriyaki glazed protein served over steamed rice",
            "nutrition": (185, 18, 5, 20, 1),
            "ingredients": ["protein", "teriyaki sauce", "rice", "broccoli", "sesame seeds", "ginger"],
            "allergens": ["soy"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium"],
            "tags": ["japanese", "bowl", "glazed", "rice"],
        },
        {
            "names": ["Tom Yum Soup", "Thai Sour Soup"],
            "proteins": ["shrimp", "chicken", "mushroom"],
            "garnishes": ["Spicy", "Mild", "with Coconut Milk", "Clear"],
            "desc": "Hot and sour Thai soup with lemongrass and galangal",
            "nutrition": (50, 5, 1.5, 4, 0.5),
            "ingredients": ["lemongrass", "galangal", "lime leaves", "chili", "mushrooms", "lime juice"],
            "allergens": ["fish"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium"],
            "tags": ["soup", "thai", "spicy", "sour"],
        },
    ],
    # --- Cluster 3: American ---
    3: [
        {
            "names": ["Classic Burger", "Cheeseburger", "Gourmet Burger"],
            "proteins": ["beef", "turkey", "chicken"],
            "garnishes": ["with Fries", "with Coleslaw", "Double Patty", "with Bacon", "with Avocado"],
            "desc": "Juicy grilled burger with fresh toppings on a toasted bun",
            "nutrition": (290, 17, 16, 24, 1),
            "ingredients": ["ground beef", "burger bun", "lettuce", "tomato", "onion", "pickles", "cheese"],
            "allergens": ["gluten", "dairy"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium"],
            "tags": ["burger", "american", "grilled", "fast-food"],
        },
        {
            "names": ["BBQ Ribs", "Smoked Ribs", "Baby Back Ribs"],
            "proteins": ["pork", "beef"],
            "garnishes": ["with Cornbread", "with Coleslaw", "Memphis Style", "Kansas City Style"],
            "desc": "Slow-smoked tender ribs glazed with tangy BBQ sauce",
            "nutrition": (280, 20, 18, 10, 0),
            "ingredients": ["pork ribs", "BBQ sauce", "dry rub", "brown sugar", "apple cider vinegar"],
            "allergens": [],
            "meal_times": ["dinner"],
            "price_tiers": ["medium", "high"],
            "tags": ["BBQ", "american", "smoked", "meat"],
        },
        {
            "names": ["Mac and Cheese", "Baked Mac and Cheese"],
            "proteins": [],
            "garnishes": ["Classic", "with Bacon", "with Truffle", "with Breadcrumbs", "Spicy"],
            "desc": "Creamy baked macaroni with a rich cheese sauce",
            "nutrition": (350, 14, 18, 34, 1),
            "ingredients": ["macaroni", "cheddar cheese", "milk", "butter", "flour", "breadcrumbs"],
            "allergens": ["gluten", "dairy"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["low", "medium"],
            "tags": ["pasta", "american", "cheesy", "comfort"],
        },
        {
            "names": ["Grilled Steak", "Ribeye Steak", "New York Strip"],
            "proteins": ["ribeye", "sirloin", "filet mignon"],
            "garnishes": ["with Mashed Potatoes", "with Asparagus", "with Mushroom Sauce", "Rare", "Medium"],
            "desc": "Premium cut steak grilled to perfection",
            "nutrition": (250, 26, 16, 0, 0),
            "ingredients": ["beef steak", "salt", "pepper", "garlic butter", "rosemary", "thyme"],
            "allergens": ["dairy"],
            "meal_times": ["dinner"],
            "price_tiers": ["high"],
            "tags": ["steak", "american", "grilled", "premium"],
        },
        {
            "names": ["Chicken Wings", "Buffalo Wings", "BBQ Wings"],
            "proteins": [],
            "garnishes": ["Buffalo", "BBQ", "Teriyaki", "Garlic Parmesan", "Honey Mustard"],
            "desc": "Crispy fried chicken wings tossed in flavorful sauce",
            "nutrition": (230, 18, 15, 6, 0),
            "ingredients": ["chicken wings", "flour", "sauce", "butter", "celery", "blue cheese dressing"],
            "allergens": ["gluten", "dairy"],
            "meal_times": ["dinner", "snack"],
            "price_tiers": ["medium"],
            "tags": ["chicken", "american", "fried", "spicy"],
        },
        {
            "names": ["Pancakes", "American Pancakes", "Buttermilk Pancakes"],
            "proteins": [],
            "garnishes": ["with Maple Syrup", "with Blueberries", "with Banana", "with Whipped Cream", "with Chocolate"],
            "desc": "Fluffy American-style pancakes stacked high",
            "nutrition": (225, 6, 8, 33, 1),
            "ingredients": ["flour", "buttermilk", "eggs", "butter", "baking powder", "sugar", "vanilla"],
            "allergens": ["gluten", "dairy", "eggs"],
            "meal_times": ["breakfast"],
            "price_tiers": ["low"],
            "tags": ["pancakes", "american", "breakfast", "sweet"],
        },
        {
            "names": ["Clam Chowder", "New England Chowder"],
            "proteins": [],
            "garnishes": ["in Bread Bowl", "Classic", "with Crackers"],
            "desc": "Rich and creamy clam soup with potatoes and herbs",
            "nutrition": (130, 7, 6, 12, 1),
            "ingredients": ["clams", "potatoes", "celery", "onion", "cream", "bacon", "thyme"],
            "allergens": ["dairy", "fish"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium"],
            "tags": ["soup", "american", "creamy", "seafood"],
        },
    ],
    # --- Cluster 4: Healthy/Diet ---
    4: [
        {
            "names": ["Quinoa Bowl", "Quinoa Salad", "Buddha Bowl"],
            "proteins": ["grilled chicken", "tofu", "salmon", "chickpeas"],
            "garnishes": ["with Avocado", "with Tahini Dressing", "Rainbow", "Mediterranean Style"],
            "desc": "Nutrient-dense grain bowl with fresh vegetables and protein",
            "nutrition": (160, 10, 6, 20, 4),
            "ingredients": ["quinoa", "mixed greens", "avocado", "cherry tomatoes", "cucumber", "lemon dressing"],
            "allergens": [],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium"],
            "tags": ["bowl", "healthy", "grain", "high-fiber"],
        },
        {
            "names": ["Green Smoothie", "Detox Smoothie", "Superfood Smoothie"],
            "proteins": [],
            "garnishes": ["with Protein Powder", "with Chia Seeds", "with Spirulina", "with Ginger"],
            "desc": "Blended green vegetables and fruits for a nutrient boost",
            "nutrition": (80, 3, 1, 16, 3),
            "ingredients": ["spinach", "banana", "apple", "ginger", "lemon", "water"],
            "allergens": [],
            "meal_times": ["breakfast", "snack"],
            "price_tiers": ["medium"],
            "tags": ["smoothie", "healthy", "vegan", "low-calorie", "detox"],
        },
        {
            "names": ["Grilled Chicken Breast", "Herb Chicken", "Lean Chicken"],
            "proteins": [],
            "garnishes": ["with Steamed Vegetables", "with Brown Rice", "with Quinoa", "with Sweet Potato"],
            "desc": "Lean grilled chicken breast seasoned with herbs",
            "nutrition": (165, 31, 3.5, 0, 0),
            "ingredients": ["chicken breast", "olive oil", "lemon", "garlic", "herbs", "salt", "pepper"],
            "allergens": [],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium"],
            "tags": ["chicken", "healthy", "high-protein", "grilled", "lean"],
        },
        {
            "names": ["Acai Bowl", "Berry Acai Bowl"],
            "proteins": [],
            "garnishes": ["with Granola", "with Coconut", "with Honey", "with Nuts", "Tropical"],
            "desc": "Thick smoothie bowl topped with fruits, granola and seeds",
            "nutrition": (210, 4, 6, 38, 5),
            "ingredients": ["acai puree", "banana", "berries", "granola", "coconut flakes", "honey"],
            "allergens": ["nuts"],
            "meal_times": ["breakfast", "snack"],
            "price_tiers": ["medium", "high"],
            "tags": ["bowl", "healthy", "superfood", "breakfast"],
        },
        {
            "names": ["Steamed Fish", "Steamed White Fish", "Light Fish"],
            "proteins": ["cod", "tilapia", "sole", "halibut"],
            "garnishes": ["with Lemon", "with Ginger Soy", "with Herbs", "with Vegetables"],
            "desc": "Delicately steamed fish fillet, light and healthy",
            "nutrition": (100, 21, 1.5, 0, 0),
            "ingredients": ["white fish fillet", "lemon", "dill", "salt", "steamed vegetables"],
            "allergens": ["fish"],
            "meal_times": ["lunch", "dinner"],
            "price_tiers": ["medium", "high"],
            "tags": ["fish", "healthy", "steamed", "low-calorie", "lean"],
        },
        {
            "names": ["Kale Salad", "Superfood Salad", "Power Salad"],
            "proteins": ["grilled chicken", "salmon", "tofu"],
            "garnishes": ["with Seeds", "with Cranberries", "with Avocado", "with Lemon Dressing"],
            "desc": "Hearty kale-based salad packed with superfoods",
            "nutrition": (120, 8, 5, 12, 4),
            "ingredients": ["kale", "sunflower seeds", "dried cranberries", "avocado", "lemon", "olive oil"],
            "allergens": [],
            "meal_times": ["lunch"],
            "price_tiers": ["medium"],
            "tags": ["salad", "healthy", "superfood", "high-fiber"],
        },
        {
            "names": ["Protein Shake", "Whey Shake", "Post-Workout Shake"],
            "proteins": [],
            "garnishes": ["Chocolate", "Vanilla", "Strawberry", "with Banana", "with Peanut Butter"],
            "desc": "High-protein shake for muscle recovery and energy",
            "nutrition": (130, 25, 2, 6, 1),
            "ingredients": ["whey protein", "milk", "banana", "ice"],
            "allergens": ["dairy"],
            "meal_times": ["breakfast", "snack"],
            "price_tiers": ["medium"],
            "tags": ["shake", "healthy", "high-protein", "fitness"],
        },
    ],
    # --- Cluster 5: Comfort/Hearty ---
    5: [
        {
            "names": ["Beef Stew", "Hearty Stew", "Slow-Cooked Stew"],
            "proteins": ["beef", "lamb"],
            "garnishes": ["with Bread", "with Dumplings", "Classic", "Irish Style"],
            "desc": "Rich slow-cooked stew with tender meat and root vegetables",
            "nutrition": (130, 12, 5, 10, 2),
            "ingredients": ["beef chunks", "potatoes", "carrots", "onion", "celery", "beef broth", "tomato paste"],
            "allergens": [],
            "meal_times": ["dinner"],
            "price_tiers": ["medium"],
            "tags": ["stew", "hearty", "slow-cooked", "comfort"],
        },
        {
            "names": ["Lasagna", "Meat Lasagna", "Baked Lasagna"],
            "proteins": ["beef", "pork", "vegetable"],
            "garnishes": ["Classic", "with Extra Cheese", "with Bechamel", "Bolognese"],
            "desc": "Layered pasta bake with meat sauce, ricotta and mozzarella",
            "nutrition": (190, 12, 9, 18, 1),
            "ingredients": ["lasagna sheets", "ground beef", "ricotta", "mozzarella", "tomato sauce", "parmesan"],
            "allergens": ["gluten", "dairy", "eggs"],
            "meal_times": ["dinner"],
            "price_tiers": ["medium"],
            "tags": ["pasta", "italian", "baked", "comfort", "cheesy"],
        },
        {
            "names": ["Shepherd's Pie", "Cottage Pie"],
            "proteins": ["lamb", "beef"],
            "garnishes": ["Classic", "with Cheddar Top", "with Sweet Potato", "with Peas"],
            "desc": "Savory meat filling topped with creamy mashed potatoes",
            "nutrition": (150, 10, 7, 14, 2),
            "ingredients": ["ground meat", "potatoes", "carrots", "peas", "onion", "gravy", "butter"],
            "allergens": ["dairy"],
            "meal_times": ["dinner"],
            "price_tiers": ["medium"],
            "tags": ["pie", "british", "comfort", "baked"],
        },
        {
            "names": ["Chicken Pot Pie", "Pot Pie"],
            "proteins": ["chicken", "turkey"],
            "garnishes": ["Classic", "with Puff Pastry", "with Biscuit Top", "Creamy"],
            "desc": "Warm chicken and vegetable filling under flaky pastry",
            "nutrition": (220, 12, 12, 18, 1.5),
            "ingredients": ["chicken", "peas", "carrots", "celery", "cream", "puff pastry", "onion"],
            "allergens": ["gluten", "dairy"],
            "meal_times": ["dinner"],
            "price_tiers": ["medium"],
            "tags": ["pie", "american", "comfort", "baked", "creamy"],
        },
        {
            "names": ["Chili Con Carne", "Texas Chili", "Bean Chili"],
            "proteins": ["beef", "turkey", "mixed beans"],
            "garnishes": ["with Rice", "with Sour Cream", "with Cornbread", "Spicy", "Mild"],
            "desc": "Spicy slow-cooked meat and bean stew, Tex-Mex classic",
            "nutrition": (140, 12, 5, 14, 4),
            "ingredients": ["ground beef", "kidney beans", "tomatoes", "chili peppers", "onion", "garlic", "cumin"],
            "allergens": [],
            "meal_times": ["dinner"],
            "price_tiers": ["low", "medium"],
            "tags": ["stew", "tex-mex", "spicy", "hearty", "beans"],
        },
        {
            "names": ["Meatloaf", "Classic Meatloaf"],
            "proteins": ["beef", "turkey", "mixed meat"],
            "garnishes": ["with Gravy", "with Ketchup Glaze", "with Mashed Potatoes", "with Green Beans"],
            "desc": "Seasoned ground meat loaf baked until golden",
            "nutrition": (200, 16, 12, 8, 0.5),
            "ingredients": ["ground beef", "breadcrumbs", "egg", "onion", "ketchup", "Worcestershire sauce"],
            "allergens": ["gluten", "eggs"],
            "meal_times": ["dinner"],
            "price_tiers": ["low", "medium"],
            "tags": ["american", "comfort", "baked", "classic"],
        },
    ],
    # --- Cluster 6: Breakfast ---
    6: [
        {
            "names": ["Oatmeal", "Porridge", "Steel-Cut Oats"],
            "proteins": [],
            "garnishes": ["with Berries", "with Banana", "with Honey", "with Nuts", "with Cinnamon", "with Apple"],
            "desc": "Warm hearty oatmeal, a nutritious start to the day",
            "nutrition": (70, 2.5, 1.5, 12, 2),
            "ingredients": ["oats", "milk", "water", "salt"],
            "allergens": ["gluten", "dairy"],
            "meal_times": ["breakfast"],
            "price_tiers": ["low"],
            "tags": ["porridge", "breakfast", "healthy", "grain", "warm"],
        },
        {
            "names": ["Omelet", "French Omelet", "Western Omelet"],
            "proteins": [],
            "garnishes": ["with Cheese", "with Mushrooms", "with Ham", "with Spinach", "with Peppers", "Garden"],
            "desc": "Fluffy egg omelet with fresh fillings",
            "nutrition": (155, 11, 12, 1, 0),
            "ingredients": ["eggs", "butter", "salt", "pepper", "filling"],
            "allergens": ["eggs", "dairy"],
            "meal_times": ["breakfast"],
            "price_tiers": ["low", "medium"],
            "tags": ["eggs", "breakfast", "quick", "protein"],
        },
        {
            "names": ["Granola", "Homemade Granola", "Crunchy Granola"],
            "proteins": [],
            "garnishes": ["with Yogurt", "with Milk", "with Berries", "with Honey", "with Banana"],
            "desc": "Crunchy baked oat clusters with nuts and dried fruit",
            "nutrition": (470, 10, 20, 64, 6),
            "ingredients": ["oats", "honey", "almonds", "coconut oil", "dried cranberries", "pumpkin seeds"],
            "allergens": ["gluten", "nuts"],
            "meal_times": ["breakfast", "snack"],
            "price_tiers": ["medium"],
            "tags": ["granola", "breakfast", "crunchy", "healthy"],
        },
        {
            "names": ["Avocado Toast", "Smashed Avocado"],
            "proteins": [],
            "garnishes": ["with Poached Egg", "with Tomato", "with Salmon", "with Feta", "Classic"],
            "desc": "Creamy mashed avocado on toasted artisan bread",
            "nutrition": (190, 5, 12, 17, 4),
            "ingredients": ["avocado", "sourdough bread", "lemon", "salt", "red pepper flakes"],
            "allergens": ["gluten"],
            "meal_times": ["breakfast", "lunch"],
            "price_tiers": ["medium"],
            "tags": ["toast", "breakfast", "trendy", "healthy"],
        },
        {
            "names": ["French Toast", "Brioche French Toast"],
            "proteins": [],
            "garnishes": ["with Maple Syrup", "with Berries", "with Powdered Sugar", "with Cream", "Cinnamon"],
            "desc": "Bread soaked in egg mixture and pan-fried until golden",
            "nutrition": (250, 8, 10, 32, 1),
            "ingredients": ["bread", "eggs", "milk", "vanilla", "cinnamon", "butter", "sugar"],
            "allergens": ["gluten", "dairy", "eggs"],
            "meal_times": ["breakfast"],
            "price_tiers": ["low", "medium"],
            "tags": ["french-toast", "breakfast", "sweet", "classic"],
        },
        {
            "names": ["Eggs Benedict", "Florentine Benedict"],
            "proteins": [],
            "garnishes": ["Classic", "with Salmon", "with Avocado", "with Bacon", "with Spinach"],
            "desc": "Poached eggs on English muffin with rich hollandaise sauce",
            "nutrition": (290, 14, 20, 14, 0.5),
            "ingredients": ["eggs", "English muffin", "hollandaise sauce", "ham", "butter", "lemon"],
            "allergens": ["gluten", "dairy", "eggs"],
            "meal_times": ["breakfast"],
            "price_tiers": ["medium", "high"],
            "tags": ["eggs", "breakfast", "elegant", "brunch"],
        },
        {
            "names": ["Yogurt Parfait", "Greek Yogurt Bowl"],
            "proteins": [],
            "garnishes": ["with Granola", "with Berries", "with Honey", "with Nuts", "Tropical"],
            "desc": "Layered Greek yogurt with fruits, granola and honey",
            "nutrition": (140, 10, 4, 18, 2),
            "ingredients": ["Greek yogurt", "berries", "granola", "honey", "chia seeds"],
            "allergens": ["dairy", "gluten", "nuts"],
            "meal_times": ["breakfast", "snack"],
            "price_tiers": ["medium"],
            "tags": ["yogurt", "breakfast", "healthy", "layered"],
        },
    ],
    # --- Cluster 7: Desserts/Snacks ---
    7: [
        {
            "names": ["Chocolate Cake", "Dark Chocolate Cake", "Devil's Food Cake"],
            "proteins": [],
            "garnishes": ["with Ganache", "with Frosting", "with Berries", "with Ice Cream", "Classic"],
            "desc": "Rich and moist chocolate cake with decadent frosting",
            "nutrition": (370, 5, 18, 50, 2),
            "ingredients": ["flour", "cocoa powder", "sugar", "eggs", "butter", "milk", "vanilla"],
            "allergens": ["gluten", "dairy", "eggs"],
            "meal_times": ["snack"],
            "price_tiers": ["medium"],
            "tags": ["cake", "dessert", "chocolate", "sweet", "baked"],
        },
        {
            "names": ["Fruit Salad", "Mixed Berry Bowl", "Fresh Fruit Bowl"],
            "proteins": [],
            "garnishes": ["with Honey", "with Mint", "with Yogurt", "with Lime", "Tropical"],
            "desc": "Colorful mix of fresh seasonal fruits",
            "nutrition": (50, 0.5, 0.3, 12, 2),
            "ingredients": ["strawberries", "blueberries", "kiwi", "mango", "banana", "mint"],
            "allergens": [],
            "meal_times": ["breakfast", "snack"],
            "price_tiers": ["low", "medium"],
            "tags": ["fruit", "dessert", "healthy", "fresh", "vegan"],
        },
        {
            "names": ["Brownies", "Fudge Brownies", "Chocolate Brownies"],
            "proteins": [],
            "garnishes": ["with Walnuts", "with Ice Cream", "Classic", "with Caramel", "with Sea Salt"],
            "desc": "Dense, fudgy chocolate brownies with a crackly top",
            "nutrition": (400, 5, 20, 52, 2),
            "ingredients": ["chocolate", "butter", "sugar", "eggs", "flour", "vanilla", "cocoa powder"],
            "allergens": ["gluten", "dairy", "eggs"],
            "meal_times": ["snack"],
            "price_tiers": ["low", "medium"],
            "tags": ["brownies", "dessert", "chocolate", "baked"],
        },
        {
            "names": ["Banana Bread", "Walnut Banana Bread"],
            "proteins": [],
            "garnishes": ["Classic", "with Chocolate Chips", "with Walnuts", "with Blueberries", "Glazed"],
            "desc": "Moist banana bread with natural sweetness",
            "nutrition": (290, 4, 10, 46, 2),
            "ingredients": ["bananas", "flour", "sugar", "eggs", "butter", "baking soda", "vanilla"],
            "allergens": ["gluten", "dairy", "eggs"],
            "meal_times": ["breakfast", "snack"],
            "price_tiers": ["low"],
            "tags": ["bread", "dessert", "banana", "baked", "sweet"],
        },
        {
            "names": ["Energy Bar", "Protein Bar", "Granola Bar"],
            "proteins": [],
            "garnishes": ["Chocolate", "Peanut Butter", "Berry", "Coconut", "Oat and Honey"],
            "desc": "Nutritious grab-and-go snack bar packed with energy",
            "nutrition": (250, 10, 8, 36, 3),
            "ingredients": ["oats", "honey", "peanut butter", "protein powder", "dried fruit", "seeds"],
            "allergens": ["gluten", "nuts"],
            "meal_times": ["snack"],
            "price_tiers": ["low", "medium"],
            "tags": ["bar", "snack", "energy", "portable", "fitness"],
        },
        {
            "names": ["Cheesecake", "New York Cheesecake", "Baked Cheesecake"],
            "proteins": [],
            "garnishes": ["Classic", "with Strawberry Topping", "with Blueberry", "with Caramel", "Chocolate"],
            "desc": "Creamy baked cheesecake with a buttery graham cracker crust",
            "nutrition": (320, 6, 22, 26, 0.5),
            "ingredients": ["cream cheese", "sugar", "eggs", "vanilla", "sour cream", "graham crackers", "butter"],
            "allergens": ["dairy", "gluten", "eggs"],
            "meal_times": ["snack"],
            "price_tiers": ["medium", "high"],
            "tags": ["cheesecake", "dessert", "creamy", "baked", "sweet"],
        },
        {
            "names": ["Muffin", "Blueberry Muffin", "Chocolate Chip Muffin"],
            "proteins": [],
            "garnishes": ["Blueberry", "Chocolate Chip", "Banana Walnut", "Lemon Poppy", "Bran"],
            "desc": "Soft and fluffy bakery-style muffin",
            "nutrition": (340, 5, 14, 48, 1.5),
            "ingredients": ["flour", "sugar", "eggs", "butter", "milk", "baking powder", "vanilla"],
            "allergens": ["gluten", "dairy", "eggs"],
            "meal_times": ["breakfast", "snack"],
            "price_tiers": ["low"],
            "tags": ["muffin", "dessert", "baked", "sweet", "breakfast"],
        },
    ],
}


def generate_dishes(rng: np.random.Generator, target_count: int = 5000) -> list[dict]:
    """Generate dish records by expanding families with combinatorial variations.

    Returns a list of dicts with all fields needed by dish_to_rich_text().
    """
    dishes: list[dict] = []
    dish_id = 1

    # Calculate how many dishes we need per cluster
    per_cluster = target_count // len(_FAMILIES)
    remainder = target_count % len(_FAMILIES)

    for cluster_id, families in _FAMILIES.items():
        cluster_target = per_cluster + (1 if cluster_id < remainder else 0)
        cluster_dishes: list[dict] = []

        # Generate all combinations for each family
        for family in families:
            names = family["names"]
            proteins = family["proteins"] if family["proteins"] else [""]
            garnishes = family["garnishes"]

            combos = list(itertools.product(names, proteins, garnishes))
            rng.shuffle(combos)

            for base_name, protein, garnish in combos:
                if len(cluster_dishes) >= cluster_target:
                    break

                # Build name
                if protein and garnish:
                    name = f"{base_name} with {protein.title()} {garnish}"
                elif protein:
                    name = f"{protein.title()} {base_name}"
                elif garnish:
                    name = f"{base_name} {garnish}"
                else:
                    name = base_name

                # Deduplicate names
                name = name.replace("  ", " ").strip()

                # Nutrition with gaussian noise ±5%
                base_nutr = family["nutrition"]
                noise = 1.0 + rng.normal(0, 0.05, size=5)
                noise = np.clip(noise, 0.85, 1.15)
                kcal = max(10, base_nutr[0] * noise[0])
                prot = max(0, base_nutr[1] * noise[1])
                fat = max(0, base_nutr[2] * noise[2])
                carbs = max(0, base_nutr[3] * noise[3])
                fiber = max(0, base_nutr[4] * noise[4])

                # Build recipe_text (procedural noise + ingredient bullets)
                ingredient_list = list(family["ingredients"])
                if protein:
                    ingredient_list = [protein] + ingredient_list
                ingredient_bullets = "\n".join(f"- {ing}" for ing in ingredient_list)
                cooking = _random_cooking_steps(rng, n=rng.integers(2, 5))
                recipe_text = f"Ingredients:\n{ingredient_bullets}\n\nInstructions:\n{cooking}"

                # Price tier
                price_tier = rng.choice(family["price_tiers"])

                # Meal time
                meal_time = rng.choice(family["meal_times"])

                # Tags
                tag_list = list(family["tags"])
                if kcal < 150:
                    tag_list.append("low-calorie")
                if prot >= 25:
                    tag_list.append("high-protein")
                if fat < 5:
                    tag_list.append("low-fat")
                tag_list.append(meal_time)
                tag_list = list(set(tag_list))

                dish = {
                    "id": dish_id,
                    "name": name,
                    "description": family["desc"],
                    "recipe_text": recipe_text,
                    "calories": round(kcal, 1),
                    "protein_g": round(prot, 1),
                    "fat_g": round(fat, 1),
                    "carbs_g": round(carbs, 1),
                    "fiber_g": round(fiber, 1),
                    "tag_list": tag_list,
                    "allergen_tags": list(family["allergens"]),
                    "meal_time": meal_time,
                    "price_tier": price_tier,
                    "cuisine_cluster": cluster_id,
                    "ingredients": ingredient_list,
                }
                cluster_dishes.append(dish)
                dish_id += 1

            if len(cluster_dishes) >= cluster_target:
                break

        # If we haven't reached target, duplicate with variations
        while len(cluster_dishes) < cluster_target:
            source = cluster_dishes[rng.integers(0, len(cluster_dishes))].copy()
            source["id"] = dish_id
            # Add slight variation to name and nutrition
            source["name"] = source["name"] + f" #{dish_id}"
            noise = 1.0 + rng.normal(0, 0.08, size=5)
            noise = np.clip(noise, 0.85, 1.15)
            source["calories"] = round(source["calories"] * noise[0], 1)
            source["protein_g"] = round(source["protein_g"] * noise[1], 1)
            source["fat_g"] = round(source["fat_g"] * noise[2], 1)
            source["carbs_g"] = round(source["carbs_g"] * noise[3], 1)
            source["fiber_g"] = round(source["fiber_g"] * noise[4], 1)
            cluster_dishes.append(source)
            dish_id += 1

        dishes.extend(cluster_dishes)

    return dishes[:target_count]
