# Model & System Evaluation

All numbers on this page were computed by re-running the actual pipeline and models in this repo (`scripts/run_customer_segmentation.py`, `scripts/run_purchase_intent.py`, `src/rag/`), not estimated. Reproduce any of them with the commands noted in each section.

---

## 1. Customer Segmentation (KMeans, k=5)

**Features** (13, StandardScaler-normalized): `total_transactions`, `unique_products`, `total_spend`, `avg_price`, `max_price`, `days_since_last_purchase`, `customer_lifetime_days`, `purchase_frequency`, `low_price_purchase_ratio`, `high_price_purchase_ratio`, `fashion_news_binary`, `is_active`, `age`.

| Metric | Value |
|---|---|
| Silhouette score (20,000-customer sample) | **0.251** |
| Inertia (full dataset, 317,897 customers) | 2,003,553 |

A silhouette score of ~0.25 is modest, not excellent — expected for RFM-style behavioral data, where segments blend into each other rather than forming tight, well-separated clusters. See [`customer_pca_plot.png`](figures/customer_pca_plot.png) for a 2D PCA projection showing this overlap directly, and [`kmeans_elbow_method.png`](figures/kmeans_elbow_method.png) for the elbow-method justification of k=5.

**Cluster profile** (from `data/processed/hm/customer_segment_summary.csv`):

| Segment (cluster) | Customers | Avg transactions | Avg spend | Avg days since last purchase | Active rate |
|---|---|---|---|---|---|
| High-Value One-Time Buyers (0) | 72,231 | 1.08 | 0.052 | 379 | 36.9% |
| Inactive Budget Shoppers (1) | 98,429 | 1.08 | 0.020 | 374 | 0.0% |
| Engaged Budget Shoppers (2) | 65,464 | 1.12 | 0.020 | 365 | 99.3% |
| Regular Shoppers (3) | 67,830 | 2.48 | 0.065 | 217 | 42.4% |
| Loyal High-Value Customers (4) | 13,943 | 5.27 | 0.180 | 148 | 49.5% |

(Spend/price columns are scaled, not raw currency — see `build_customer_features`.) The clusters do separate on the dimension that matters most for targeting — recency and transaction frequency — which is why the segment names and strategies in [`segment_strategy.md`](segment_strategy.md) are usable even with a moderate silhouette score.

**Known limitation** (see main [README](../README.md#limitations--future-improvements)): cluster-to-segment-name mapping is a hardcoded `{0: "High-Value One-Time Buyers", ...}` index lookup, not sorted by any cluster property. Rerunning KMeans on different data could silently relabel segments. Fixing this means mapping by cluster characteristics (e.g., sort by avg spend) instead of raw index — flagged but not yet fixed.

**Reproduce:** `PYTHONPATH=. python scripts/run_customer_segmentation.py`

---

## 2. Purchase-Intent Model (XGBoost)

Binary classifier predicting purchase probability for a (customer, product) pair. Trained on a synthetically balanced dataset: 498,294 positive examples (observed H&M transactions) and 498,273 negative examples (sampled non-purchases), 80/20 train/test split, `random_state=42`.

| Metric | Value |
|---|---|
| ROC-AUC | **0.926** |
| Accuracy | 0.85 |
| Precision (purchased=1) | 0.80 |
| Recall (purchased=1) | 0.93 |
| F1 (purchased=1) | 0.86 |

Confusion matrix (test set, n=199,314):

| | Predicted: no purchase | Predicted: purchase |
|---|---|---|
| **Actual: no purchase** | 77,157 | 22,498 |
| **Actual: purchase** | 7,282 | 92,377 |

The model is tuned toward recall on the positive class (93%) at some cost to precision (80%) — reasonable for a recommendation use case, where missing a real purchase signal is costlier than over-recommending. ROC-AUC of 0.93 indicates strong separation, but this number should be read with the caveat below.

**Top feature importances** (from `data/processed/hm/purchase_model_feature_importance.csv`):

| Feature | Importance |
|---|---|
| `target_audience_Baby/Children` | 0.129 |
| `avg_selling_price` | 0.080 |
| `index_group_Baby/Children` | 0.065 |
| `price_vs_customer_avg` | 0.061 |
| `index_group_Menswear` | 0.037 |

Price-relative-to-customer-history and product category dominate — sensible, and consistent with how the negative-sampling strategy was constructed (see next point).

**Important caveat:** negative examples are *sampled* non-purchases, not observed "customer saw this and declined it" signals — there's no real browse/cart-abandon data in the H&M dataset. This means the 0.93 ROC-AUC reflects how well the model separates "products a customer actually bought" from "products drawn from the sampling distribution," which is a real and useful signal for ranking, but isn't the same claim as "predicts real-world purchase intent with 93% discriminative power." Treat the ranking as directionally correct, not as a calibrated probability.

**Reproduce:** `PYTHONPATH=. python scripts/run_purchase_intent.py`

---

## 3. Campaign Ranking Quality

`campaign_score = 0.50 × purchase_probability + 0.25 × inventory_score + 0.25 × segment_match_score`, then min-max normalized.

**Curated top-100 recommendations** (20 per segment × 5 segments, what the dashboard shows by default):

| | Min | Mean | Max |
|---|---|---|---|
| Campaign score (normalized) | 0.000 | 0.695 | 1.000 |
| Purchase probability | 0.944 | 0.974 | 0.991 |

Per-segment average campaign score:

| Segment | Avg campaign score |
|---|---|
| Inactive Budget Shoppers | 0.968 |
| Engaged Budget Shoppers | 0.917 |
| Loyal High-Value Customers | 0.850 |
| High-Value One-Time Buyers | 0.678 |
| Regular Shoppers | **0.062** |

**Notable finding:** the normalization (`min-max` in `normalize_campaign_scores`) is computed *globally* across all 100 curated rows, not per-segment. That's why Regular Shoppers' own best picks still average 0.062 — that segment's raw purchase-probability scores are systematically lower than other segments' (worth checking `purchase_features.py` if this segment's targeting matters for a real use case), and the global normalization makes that visible instead of masking it with a flattering per-segment rescale. This is a byproduct of the current design, not a bug — but if the dashboard needs to make every segment look equally "campaign-ready," per-segment normalization would be the fix, at the cost of hiding this real signal.

**Catalog-wide distribution** (330,013 scored candidate segment-product pairs, before top-N selection — see `data/processed/hm/campaign_candidate_summary.csv`):

| Strategy | Count | Share |
|---|---|---|
| Awareness campaign | 125,539 | 38.0% |
| Discount campaign | 78,544 | 23.8% |
| Deprioritize | 66,901 | 20.3% |
| Personalized recommendation | 38,802 | 11.8% |
| Promote aggressively | 10,134 | 3.1% |
| Premium positioning | 10,093 | 3.1% |

Only 3.1% of all scored candidates are "Promote aggressively" — but that 3.1% is exactly what dominates the curated top-100 (see the bugfix chapter of this project's history: the ranking formula's positive weight on inventory naturally selects for high-inventory, high-intent products at the top). The Overview and Analytics dashboards deliberately chart *this* catalog-wide distribution rather than the curated top-N, specifically so a viewer sees the real spread instead of a sample biased toward one strategy.

---

## 4. RAG Retrieval Quality

Retrieval: TF-IDF (max 20,000 features) → TruncatedSVD (128 dims) → FAISS `IndexFlatIP` over the full 105,542-product catalog.

**Quantitative check:** sampled 300 random products, retrieved their top-5 nearest neighbors (1,500 neighbor pairs total), and measured how often the neighbor shares the query's own catalog metadata:

| Match type | Rate |
|---|---|
| Same `product_type` (e.g. "T-shirt") | 94.1% |
| Same `product_group` (e.g. "Garment Upper body") | 99.1% |
| Same inferred `style` tag | 96.4% |

This is strong for a purely lexical retrieval method — but the reason it works this well is that H&M's product copy is templated/structured (name, type, group, color, description all follow consistent patterns), so TF-IDF overlap is a good proxy for category similarity here specifically. This wouldn't necessarily hold up on free-form, unstructured product text.

**Example (real, from a live `/generate-copy` call):**

> Query: *Tilly (1)* — an elegant T-shirt, black, for work/outing wear.
> Retrieved neighbors: *Tilly*, *Tilly (1)* ×3 (color/size variants), *Bob V-neck*.
> Generated copy: *"Effortless everyday style in soft, lightweight jersey—the Tilly tee is cut just right with a flattering longer back and rounded hem. Get the perfect basics staple that works hard and feels great, now available in classic black."*

The generated copy correctly stays grounded in what the retrieved context actually supports (fabric, fit, color) and doesn't invent details like price or material composition that weren't in the product record.

**Caveat:** TF-IDF + SVD is a classical/lexical embedding, not a neural one — a deliberate choice for this project (see [README limitations](../README.md#limitations--future-improvements): no torch wheel supports both Intel macOS and the numpy 2.x this stack needs). A neural embedding model would likely generalize better to queries phrased differently from the catalog's own vocabulary, at the cost of the dependency conflict this project specifically avoided.

**Reproduce:** `PYTHONPATH=. python scripts/build_product_index.py`, then query via `src/rag/product_index.retrieve_similar_products`.

---

## Summary

| Component | Headline metric | Read with this caveat |
|---|---|---|
| Segmentation | Silhouette 0.251 | Real customer behavior clusters loosely, not cleanly — expected |
| Purchase intent | ROC-AUC 0.926 | Negative class is sampled, not observed non-purchases |
| Campaign ranking | 3.1% of catalog scores "Promote aggressively" | Curated top-N is a biased sample of that 3.1%, by design |
| RAG retrieval | 94–99% category-match rate | High because H&M product text is templated; may not generalize |

Every number above is reproducible from a clean clone by following the [README setup steps](../README.md#setup) through `scripts/run_full_pipeline.py` and `scripts/build_product_index.py`.
