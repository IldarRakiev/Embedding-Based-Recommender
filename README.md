# Recommendation Systems using Text Embeddings
## Case Study: Food Recommendation with Semantic Embeddings

**NLP 2026 · [Author Name] · [University]**

---

### Overview

This case study explores how text embeddings can power a food recommendation system. We investigate the effectiveness of pretrained sentence-transformers for dish retrieval, propose several embedding architecture improvements, and evaluate their impact on recommendation quality using the Food.com dataset.

The work is motivated by a real production system for health-aware food recommendations, where text embeddings serve as the foundation for candidate retrieval via FAISS vector search.

### Key Research Questions

1. How well do pretrained text embeddings capture food preferences compared to keyword-based (TF-IDF) retrieval?
2. Which components of a dish text representation (recipe instructions, discrete macro tokens, ingredient lists) contribute most to embedding quality?
3. Does representing user behavior via dish-vector aggregation outperform text-based behavioral encoding?
4. Can multi-vector retrieval and cross-encoder reranking improve accuracy?
5. Does domain-specific fine-tuning on food interaction data improve recommendation quality?

### Datasets

- **Food.com Recipes and Interactions** (Kaggle): 230K recipes, 700K user interactions — primary evaluation dataset
- **MealRec+** (GitHub): meal-level recommendation data — supplementary experiment

See [`data/README.md`](data/README.md) for download instructions.

### Project Structure

```
Embedding-Based-Recommender/
├── src/                          # ML modules (extracted from production)
│   ├── text_builders.py          # Parametrized dish/user text representation
│   ├── embedding_model.py        # Sentence-transformer wrapper
│   ├── user_embedding.py         # Weighted user embedding
│   ├── behavioral_embedding.py   # Behavioral embedding via dish vectors
│   ├── multi_vector_retrieval.py # Multi-query search with RRF merge
│   ├── cross_encoder_reranker.py # Cross-encoder reranking
│   └── utils.py                  # Metrics (P@K, NDCG, MRR) + visualization
├── notebooks/                    # Experiments (each self-contained for Colab)
├── data/                         # Downloaded by notebooks (not in git)
└── requirements.txt
```

**Key design principle:** `src/` code does not change between notebooks. All experiment variants are realized through function parameters — `dish_to_rich_text(include_recipe=False, include_ratios=True, ...)`. The reviewer sees baseline and improvement in the same notebook without switching commits.

### How to Run

Each notebook is self-contained and runs in Google Colab. The first cell clones the repo, installs dependencies, and sets up imports.

| Notebook | Description | Colab |
|----------|-------------|-------|
| [01 Data Exploration](notebooks/01_data_exploration.ipynb) | EDA, nutrition parsing, dataset limitations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IldarRakiev/Embedding-Based-Recommender/blob/main/notebooks/01_data_exploration.ipynb) |
| [02 Baseline Embeddings](notebooks/02_baseline_embeddings.ipynb) | Pretrained embeddings vs TF-IDF | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IldarRakiev/Embedding-Based-Recommender/blob/main/notebooks/02_baseline_embeddings.ipynb) |
| [03 Embedding Improvements](notebooks/03_embedding_improvements.ipynb) | 7 experiments with text representation flags | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IldarRakiev/Embedding-Based-Recommender/blob/main/notebooks/03_embedding_improvements.ipynb) |
| [04 Fine-Tuning](notebooks/04_fine_tuning.ipynb) | Domain-specific fine-tuning (**requires GPU**) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IldarRakiev/Embedding-Based-Recommender/blob/main/notebooks/04_fine_tuning.ipynb) |
| [05 Evaluation](notebooks/05_evaluation.ipynb) | Grand comparison table, ablation study, conclusions | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IldarRakiev/Embedding-Based-Recommender/blob/main/notebooks/05_evaluation.ipynb) |

### Results Summary

*[Fill in after running the experiments — copy the grand comparison table from Notebook 05]*

### Evaluation Protocol

**Item-to-item co-preference:** For each user with ≥5 positive ratings (≥4 stars) in the test set, one positive recipe is used as the query and the remaining positives form the relevant set. We measure how many of the top-K retrieved recipes are in the relevant set (P@K, NDCG@K, MRR).

This tests whether recipes liked by the same person are nearby in embedding space — the core assumption of content-based retrieval.
