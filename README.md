# Recommendation Systems using Text Embeddings

**Case Study · NLP 2026 · Innopolis University**
**Ildar Rakiev · Group DS-01**

Applied research for **Reil.App** - a mobile service that generates daily meal recommendations aligned with user preferences and macro-nutrient (KBJU) targets, auto-assembles the cart, and places orders via delivery partners.

---

## TL;DR

Systematic ablation of text-embedding strategies for content-based food recommendation.

**Central finding (~4× MRR lift).** Replacing text-encoded user profiles with **embedding aggregation over positively-interacted items** — the approach prescribed in the case-study brief — is the single lever with a multiplicative impact on retrieval quality. Text-representation flags (recipe / macro tokens / ratios / ingredients), MNRL fine-tuning on same-user pairs, and hybrid BM25+dense retrieval all move metrics by fractions of a basis point.

This finding directly motivated the weighting of the behavioral-vector feature in Reil.App's production ranker.

---

## Research Questions

1. How well do pretrained text embeddings capture food preferences vs. TF-IDF?
2. Which components of the dish-text representation (recipe / macros / ingredients / tags) matter?
3. Does aggregating **dish vectors** for a user-profile embedding beat text-based profile encoding?
4. Does domain fine-tuning on user co-preference pairs improve retrieval?
5. Can hybrid (BM25+dense) / cross-encoder / LLM rerankers add value on item-to-item?

---

## Dataset

Synthetic, designed to mirror Reil.App production signals. Generated offline by `data/generate_synthetic.py` + `data/generate_dishes_llm.py`.

| | |
|---|---|
| Dishes | **4,939** (LLM-generated · 8 cuisine clusters) |
| Users | **1,100** across **10** archetypes (fitness · student · diabetic · vegetarian · gluten-free · allergy-prone · family-mom · ...) |
| Per-user fields | **KBJU targets** · allergens · goal-type (lose/keep/gain) · price tier |
| Interactions | **195,749** (view / order / rating / favorite) · 60/20/20 split |
| Signal validation | calorie-goal coherence ✓ · 0 allergen violations in orders/favorites ✓ |

See [`data/README.md`](data/README.md) for reproduction commands.

---

## Evaluation Protocol — Item-to-Item Co-Preference

For each test user with ≥5 positive interactions (orders/favorites):

1. Take one positive dish as the **query**.
2. The remaining positives form the **relevant set** $R_u$.
3. Retrieve top-K from the full 4,939-dish catalog.
4. Report $P@K$, $NDCG@K$, $HR@K$, $MRR$ — averaged over users.

Absolute $P@K$ is structurally small ($K{=}10$ vs. few positives per user); **$MRR$ and $NDCG$ are the more informative signals**.

---

## Production Mapping (Reil.App)

Findings from this study shaped concrete architectural decisions in Reil.App's recommender (three-stage pipeline with a cold-start maturity router). See [`poster/poster-a2.pdf`](poster/poster-a2.pdf) for the full diagram.

**Kept — motivated by the study**
- Behavioral-vector similarity carries the **dominant weight** in the per-slot scorer.
- BM25 + dense RRF as an additive lexical channel whenever the user types a free-text query.
- Onboarding → segment key (goal × archetype, **3 × 11 = 33 segments**).
- Thompson-sampling bandit $\text{Beta}(\alpha, \beta)$ per (segment, dish) for exploration during ICE → WARM; disabled in HOT.
- Hard filters last (allergens · goal · diabetes) — non-negotiable safety layer.

**Dropped or deferred**
- Elaborate text-representation flags — no gain.
- MNRL fine-tuning on same-user pairs — signal too weak; replaced by a planned two-tower on real logs.
- Cross-encoder & LLM profile-reranker — offline only (latency / $ too high for a daily-plan service).
- **No heavy LTR ranker yet** (LightGBM / LambdaMART) — the weighted sum is a deliberate stand-in until pre-seed interaction data accumulates.

### Cold-start maturity router (production)

| Phase | Condition | Candidate strategy |
|---|---|---|
| **ICE** | 0 meaningful events | segment top-K + Thompson bandit $\text{Beta}(\alpha,\beta)$ |
| **WARM** | 1–14 meaningful events | 60% segment · 40% behavioral vector |
| **HOT** | ≥15 meaningful events | behavioral vector only *(study's hero lever)* |

Meaningful events: `click · add_to_meal · add_to_cart · add_to_favorites · rate · skip`.

---

## Repository Layout

```
Embedding-Based-Recommender/
├── src/                              # reusable modules (frozen across notebooks)
│   ├── text_builders.py              # parametrized dish-text builder (flag-driven)
│   ├── embedding_model.py            # sentence-transformer wrapper
│   ├── user_embedding.py             # text-based user-profile encoder (baseline)
│   ├── behavioral_embedding.py       # HERO: aggregates dish vectors for a user
│   ├── multi_vector_retrieval.py     # per-field multi-query search + RRF
│   ├── hybrid_rrf.py                 # BM25 + dense RRF fusion
│   ├── cross_encoder_reranker.py     # pairwise cross-encoder rerank
│   ├── hard_negative_mining.py       # cluster-aware hard negatives (for triplet fine-tune)
│   ├── llm_profile_reranker.py       # LLM rerank with USER_PROFILE context
│   └── utils.py                      # metrics (P@K, NDCG, MRR) + plotting + results IO
├── notebooks/                        # experiments (each self-contained, Kaggle/Colab ready)
│   ├── data_exploration.ipynb              # EDA + signal validation
│   ├── baseline_embeddings.ipynb           # pretrained mpnet vs TF-IDF
│   ├── embedding_improvements.ipynb        # 7 ablations (hero behavioral experiment)
│   ├── fine_tuning.ipynb                   # MNRL + triplet with hard negatives (GPU)
│   ├── hybrid_bm25_dense_rrf.ipynb         # lexical fusion
│   ├── item-to-item-llm-retrieval-aug.ipynb  # LLM retrieval hints
│   └── llm-rerank-profile.ipynb            # LLM profile-reranker (fair-compare)
├── scripts/
│   ├── augment_dishes_retrieval_llm.py     # generates dish_retrieval_aug.json
│   └── finetune_triplet_hard_negatives.py  # triplet fine-tune driver (CLI)
├── data/
│   ├── synthetic/                    # parquet dataset (generated, not in git)
│   ├── generate_synthetic.py         # user · interaction generator (archetype-driven)
│   ├── generate_dishes_llm.py        # LLM-based dish generator
│   ├── dish_templates.py             # per-cuisine templates
│   └── user_archetypes.py            # 10 archetype definitions
├── poster/
│   ├── poster.html                   # editable source (MathJax formulas)
│   └── poster-a2.pdf                 # exported PDF (317×900 mm · single page)
├── requirements.txt
└── README.md
```

**Design principle.** `src/` stays frozen across notebooks. All experiment variants are realized through function arguments — e.g. `dish_to_rich_text(include_recipe=False, include_ratios=True, ...)`. The reviewer sees baseline and improvement side-by-side in the same notebook, without switching commits.

---

## How to Run

Each notebook is self-contained and provisions itself in the first cell (repo clone, pip install, data-path resolution for Kaggle / Colab / local).

Recommended order:
1. `data_exploration` — sanity-check the synthetic dataset
2. `baseline_embeddings` — mpnet vs TF-IDF
3. `embedding_improvements` — the hero behavioral experiment lives here
4. `fine_tuning` (GPU) — optional
5. any of `hybrid_bm25_dense_rrf` / `item-to-item-llm-retrieval-aug` / `llm-rerank-profile` in any order

### Local

```bash
pip install -r requirements.txt

# One-time: generate the synthetic dataset
#   OPENAI_API_KEY + OPENAI_BASE_URL required for LLM dish generation
python data/generate_dishes_llm.py
python data/generate_synthetic.py

jupyter notebook notebooks/
```

### Kaggle / Colab

Every notebook's first cell clones the repo, installs dependencies, and locates `data/synthetic/*.parquet`. On Kaggle, add the generated parquet files (or `dish_retrieval_aug.json`) as a Dataset input — the notebooks auto-discover them under `/kaggle/input/`.

---

## Acknowledgements

- **NLP 2026 · Innopolis University** — course assignment that motivated this systematic study.
- **Reil.App** — applied context; this repo is the foundational research for its content-based recommender (pre-seed).
- **Open-source stack**: `sentence-transformers` (`paraphrase-multilingual-mpnet-base-v2`), cross-encoder `ms-marco-MiniLM-L-6-v2`, `gpt-4o-mini` via OpenRouter, FAISS for ANN, `rank_bm25` for lexical retrieval.
