# Data

Data files are **not committed** to this repository. They are downloaded automatically in the notebooks.

## Food.com Recipes and Interactions (primary dataset)

**Source:** https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

**Option A — Kaggle API (recommended):**

1. Create a Kaggle account and go to *Account → API → Create New Token*
2. Upload `kaggle.json` to Colab via the Files panel
3. The first cell of Notebook 01 will run:
   ```bash
   kaggle datasets download -d shuyangli94/food-com-recipes-and-user-interactions -p data/
   unzip data/food-com-recipes-and-user-interactions.zip -d data/
   ```

**Option B — Manual download:**

1. Download from the Kaggle link above
2. Upload `RAW_recipes.csv` and `RAW_interactions.csv` to `data/`

**Files used:**
- `RAW_recipes.csv` — 230K recipes with nutrition, tags, instructions
- `RAW_interactions.csv` — 700K user ratings (1–5 stars)

## MealRec+ (supplementary dataset)

**Source:** https://github.com/WUT-IDEA/MealRec

Used in supplementary experiments for meal-level recommendation (not required for main notebooks).

```bash
git clone https://github.com/WUT-IDEA/MealRec data/MealRec
```

## Processed Data

After running Notebook 01, `data/processed/` will contain:

| File | Description |
|---|---|
| `recipes.parquet` | Cleaned recipes with parsed nutrition columns |
| `interactions_train.parquet` | 60% temporal split |
| `interactions_val.parquet` | 20% temporal split |
| `interactions_test.parquet` | 20% temporal split |
| `results.json` | Experiment metrics (updated by each notebook) |
| `embeddings_baseline.npy` | Saved embeddings from NB02 |
| `embeddings_finetuned.npy` | Saved embeddings from NB04 |
| `final_results.csv` | Grand comparison table from NB05 |
