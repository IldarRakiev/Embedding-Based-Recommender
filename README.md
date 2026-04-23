# Recommendation Systems using Text Embeddings

**Case Study · NLP 2026 · Innopolis University**
**Ildar Rakiev · Group DS-01**

Applied research for **Reil.App** — a mobile service that generates daily meal recommendations aligned with user preferences and macro-nutrient (KBJU) targets, auto-assembles the cart, and places orders via delivery partners.

---

## TL;DR

Systematic ablation of text-embedding strategies for content-based food recommendation.

**Central finding (~4× MRR lift).** Replacing text-encoded user profiles with **embedding aggregation over positively-interacted items** — the approach prescribed in the case-study brief — is the single lever with a multiplicative impact on retrieval quality. Text-representation flags (recipe / macro tokens / ratios / ingredients), MNRL fine-tuning on same-user pairs, and hybrid BM25+dense retrieval all move metrics by fractions of a basis point.

This finding directly motivated the weighting of the behavioral-vector feature in Reil.App's production ranker.

---

## Case-Study Brief (professor's formulation)

> Build a content-based recommendation system using text embeddings to suggest items to users based on the semantic content of their profiles or past interactions. **Create user profiles by aggregating the embeddings of items a user has interacted with positively.** Evaluate the system's ability to make relevant and diverse recommendations, comparing against a simple keyword-based recommender.

Coverage map:

| Brief requirement | Status | Where |
|---|---|---|
| Content-based recsys on text embeddings | ✅ | entire project |
| Items with textual descriptions | ✅ | dishes: name · description · recipe · ingredients · tags |
| User profile via embedding aggregation of positives | ✅ **hero experiment** | [`src/behavioral_embedding.py`](src/behavioral_embedding.py), Exp 5 in [`notebooks/embedding_improvements.ipynb`](notebooks/embedding_improvements.ipynb) |
| Recommend via similarity to profile vector | ✅ | FAISS `IndexFlatIP` over aggregated user vector |
| Evaluate relevance | ✅ | P@K, NDCG@K, HR@K, MRR |
| Evaluate diversity | ⚠️ *scope-extension* | handled in production via MMR (`backend/app/recommendations/segments.py:select_diverse_top_dishes`), not measured offline here |
| Compare to keyword-based | ✅ | TF-IDF + TruncatedSVD(256) in [`notebooks/baseline_embeddings.ipynb`](notebooks/baseline_embeddings.ipynb) |

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

## Results

### Grand comparison

| Configuration | P@5 | P@10 | NDCG@10 | HR@10 | MRR |
|---|---:|---:|---:|---:|---:|
| TF-IDF + SVD(256) *(lexical baseline)* | 0.0016 | 0.0020 | 0.0024 | 0.0198 | 0.0048 |
| mpnet — baseline text | 0.0013 | 0.0016 | 0.0015 | 0.0158 | 0.0032 |
| − recipe | 0.0013 | 0.0015 | 0.0014 | – | 0.0033 |
| − macro tokens | 0.0011 | 0.0016 | 0.0017 | – | 0.0037 |
| + macro ratios | 0.0011 | 0.0015 | 0.0013 | – | 0.0031 |
| + ingredients (full improved text) | 0.0013 | 0.0015 | 0.0014 | – | 0.0032 |
| **+ behavioral embedding (dish vectors)** | **0.0085** | **0.0070** | **0.0109** | – | **0.0299** |
| + multi-vector retrieval (RRF) | 0.0018 | 0.0021 | 0.0023 | – | 0.0058 |
| Hybrid BM25 + dense (RRF, $k{=}60$) | 0.0013 | 0.0013 | 0.0015 | – | 0.0041 |
| Fine-tuned mpnet (MNRL, 4 epochs) | 0.0013 | 0.0015 | 0.0016 | – | 0.0053 |
| LLM retrieval-hint augmentation *(50/4939 coverage)* | 0.0013 | 0.0015 | 0.0014 | – | 0.0032 |
| LLM profile-rerank (inside pool $N{=}50$) | 0.0000 | 0.0000 | 0.0000 | – | 0.0020 |

Behavioral embedding is the single config with a multiplicative uplift — **~4× MRR** and **~7× NDCG@10** over the text-profile baseline. All other ablations / additions move metrics within noise.

### Why this matters

Direct empirical validation of the brief's prescribed approach (*"create user profiles by aggregating the embeddings of items a user has interacted with positively"*). The experiment quantifies how much better the behavioral approach is than encoding the profile as text.

### Known issues in the notebook runs

- **Cross-encoder rerank** returns $P@K{=}0$ — pairing bug, not a genuine null result (row excluded from conclusions).
- **Fine-tune comparison row** shows NaN for "Best Pretrained" — lookup key mismatch in [`notebooks/fine_tuning.ipynb`](notebooks/fine_tuning.ipynb).
- **LLM retrieval-hint experiment** ran with only 50 / 4,939 hints generated — insufficient coverage to evaluate.

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

### Regenerate the poster PDF

```bash
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu \
  --no-pdf-header-footer --print-to-pdf-no-header --no-margins \
  --virtual-time-budget=20000 \
  --print-to-pdf="poster/poster-a2.pdf" \
  "file:///$(pwd)/poster/poster.html"
```

`--virtual-time-budget=20000` gives MathJax 20 s (virtual) to render formulas before the snapshot.

---

## Artifacts

- **Poster**: [`poster/poster-a2.pdf`](poster/poster-a2.pdf) — single-page 317×900 mm summary of motivation, methods, results, production mapping. Editable source in [`poster/poster.html`](poster/poster.html) (MathJax via CDN).
- **Results JSON**: each notebook writes metrics to `data/results.json` / `processed/results.json` (see `utils.load_run_results_json` for cross-notebook loading).

---

## Scope Note on Diversity

The offline notebook suite evaluates **relevance only**. Recommendation **diversity** — the second axis asked for in the case-study brief — is handled in Reil.App's production via a greedy MMR selector with a tag-overlap penalty ([`backend/app/recommendations/segments.py:select_diverse_top_dishes`](https://github.com/IldarRakiev/Kitchen-OS)). Adding intra-list diversity (ILD) and category-coverage metrics to the notebook suite is a natural next step and deliberately out-of-scope for this case study.

---

## Acknowledgements

- **NLP 2026 · Innopolis University** — course assignment that motivated this systematic study.
- **Reil.App** — applied context; this repo is the foundational research for its content-based recommender (pre-seed).
- **Open-source stack**: `sentence-transformers` (`paraphrase-multilingual-mpnet-base-v2`), cross-encoder `ms-marco-MiniLM-L-6-v2`, `gpt-4o-mini` via OpenRouter, FAISS for ANN, `rank_bm25` for lexical retrieval.
