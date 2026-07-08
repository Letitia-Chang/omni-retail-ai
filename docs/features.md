# Key Features

**Customer segmentation** — KMeans clustering over RFM-style behavioral features (recency, frequency, spend, price sensitivity), producing five named segments — e.g. *Loyal High-Value Customers*, *Inactive Budget Shoppers* — each with a documented targeting strategy. See [Customer Segment Strategy](../reports/segment_strategy.md).

**Product enrichment** — derives style and occasion tags, plus marketing attributes, per product from catalog metadata (product name, type, description).

**Purchase-intent prediction** — an XGBoost classifier trained on simulated customer-product interactions, predicting a purchase probability for each customer-product pair. Evaluated at ROC-AUC 0.926 — see [Model & System Evaluation](../reports/evaluation.md).

**Inventory-aware campaign ranking** — combines purchase probability × inventory signal × segment-fit score into a single campaign score, with a human-readable ranking explanation attached to every recommendation.

**RAG-grounded ad-copy generation** — a FAISS index over the ~105K-product catalog (TF-IDF + SVD embeddings) retrieves similar products at request time to ground a live Claude Haiku call, so generated ad copy is tied to real catalog context instead of a fixed template. Rate-limited (10/minute per IP, 200/day global cap) since it's a public endpoint calling a paid API.

**FastAPI backend** — serves segments, ranked campaigns, summary stats, catalog-wide analytics, and live RAG-grounded copy generation as JSON.

**React dashboard** — a TanStack Start app with five views: overview, segment explorer, campaign list, analytics, and a campaign generator. See the [Demo Walkthrough](demo.md).

**Real, reproducible evaluation** — silhouette score, ROC-AUC, ranking-quality distributions, and a quantitative RAG grounding check, all computed from this repo's own pipeline, not estimated. See [Model & System Evaluation](../reports/evaluation.md).
