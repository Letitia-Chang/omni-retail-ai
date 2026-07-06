# OmniRetail AI

**An end-to-end retail marketing intelligence pipeline — from raw transaction data to a segmented, ranked, ready-to-ship campaign dashboard.**

Portfolio project built to demonstrate applied data science and ML engineering: customer segmentation, a supervised purchase-intent model, and an inventory-aware campaign ranking system, served through a FastAPI backend and a React dashboard.

> Built for **AI/ML Engineer**, **Data Scientist**, and **Data Analyst** roles.

---

## Problem Statement

Retail marketing teams manage thousands of products across multiple customer segments, but typically lack the tooling to plan campaigns at that scale:

- Campaign planning is manual and doesn't scale with catalog size or customer diversity.
- Promotions are rarely tied to actual purchase likelihood or current inventory, wasting spend on products customers won't buy or that are already out of stock.
- Customer segments exist as a concept ("loyal", "budget-conscious") but aren't operationalized into an actual targeting workflow.

**OmniRetail AI** turns raw customer/transaction/product data into a ranked list of "which product, to which segment, with what strategy" — the concrete output a marketing team would act on.

## Key Features

- **Customer segmentation** — KMeans clustering over RFM-style behavioral features, producing five named segments (e.g. *Loyal High-Value Customers*, *Inactive Budget Shoppers*) with a documented strategy per segment ([reports/segment_strategy.md](reports/segment_strategy.md))
- **Product enrichment** — derives style/occasion tags and marketing attributes per product from catalog metadata
- **Purchase-intent prediction** — an XGBoost classifier trained on simulated customer-product interactions, predicting purchase probability per customer-product pair
- **Inventory-aware campaign ranking** — combines purchase probability × inventory signal × segment-fit score into a single campaign score, with a human-readable ranking explanation per recommendation
- **FastAPI backend** — serves segments, ranked campaigns, and summary stats as JSON
- **React dashboard** — a TanStack Start app with 5 views: overview, segment explorer, campaign list, analytics, and a campaign generator

## Tech Stack

| Layer | Tools |
|---|---|
| Data / ML | Python, pandas, NumPy, scikit-learn, XGBoost, joblib |
| Data source | [Kaggle H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) dataset, via `kagglehub` |
| Backend | FastAPI, Uvicorn |
| Frontend | React 19, TanStack Start/Router/Query, Tailwind CSS, Radix UI |
| Deploy target | Cloudflare (Vite plugin + `wrangler` config included) |

## Architecture

```mermaid
flowchart LR
    subgraph Data
        A[Kaggle H&M dataset] -->|scripts/download_data.py| B[data/raw/hm]
        B -->|notebooks/01_data_schema_and_cleaning| C[data/processed/hm]
    end

    subgraph Pipeline["src/ pipeline, run via scripts/run_full_pipeline.py"]
        C --> D[Customer segmentation\nsrc/models/segmentation.py]
        C --> E[Product enrichment\nsrc/features/product_enrichment.py]
        D --> F[Purchase intent model\nsrc/models/purchase_intent.py]
        E --> F
        F --> G[Campaign ranking\nsrc/campaigns/campaign_ranker.py]
        G --> H[Campaign generation\nsrc/campaigns/campaign_generator.py]
    end

    D --> I[(saved_models/segmentation)]
    F --> J[(saved_models/purchase_model)]
    H --> K[(data/processed/hm/*.csv)]

    K --> L[FastAPI backend]
    I -.-> L
    J -.-> L
    L --> M[React dashboard]
```

## Setup

### 1. Clone and install Python dependencies

```bash
git clone git@github.com:Letitia-Chang/omni-retail-ai.git
cd omni-retail-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get the data

Requires a [Kaggle](https://www.kaggle.com/) account with `kaggle.json` credentials in `~/.kaggle/` (and having accepted the competition rules).

```bash
python scripts/download_data.py
```

Then run [notebooks/01_data_schema_and_cleaning.ipynb](notebooks/01_data_schema_and_cleaning.ipynb) to produce the cleaned `data/processed/hm/` tables the pipeline expects (`customers.csv`, `articles.csv`, `transactions.csv`).

### 3. Run the ML pipeline

```bash
python scripts/run_full_pipeline.py
```

This runs segmentation → product enrichment → purchase-intent modeling → campaign ranking → campaign generation in sequence, writing outputs to `data/processed/hm/` and trained models to `saved_models/`.

### 4. Run the backend

```bash
uvicorn backend.main:app --reload
```

Serves at `http://127.0.0.1:8000` — see `/` for available endpoints.

### 5. Run the frontend

```bash
cd frontend
cp .env.example .env   # points the dashboard at the local API
npm install
npm run dev
```

Dashboard runs at `http://localhost:3000` (TanStack Start dev server).

## Demo

With the backend and frontend both running:

1. **Overview** (`/`) — total campaigns, segment count, average campaign score
2. **Segments** (`/segments`) — explore each customer segment and its top recommended products
3. **Campaigns** (`/campaigns`) — search/filter the full ranked campaign list
4. **Analytics** (`/analytics`) — score distributions and inventory-level breakdowns
5. **Generator** (`/generator`) — walks through a single campaign recommendation with its ranking explanation

## Screenshots

**Overview** — live campaign stats, top recommendations, and strategy mix

![Dashboard overview](reports/figures/dashboard_overview.png)

**Segments** — pick a segment, browse its AI-ranked product picks with images

![Segment explorer](reports/figures/dashboard_segments.png)

**Campaigns** — full ranked list with search and filters

![Campaign list](reports/figures/dashboard_campaigns.png)

**AI Generator** — single-campaign recommendation with its ranking explanation

![AI campaign generator](reports/figures/dashboard_generator.png)

**Analytics** — score and inventory distributions across segments

![Analytics](reports/figures/dashboard_analytics.png)

**Modeling notebooks** — customer segmentation (PCA projection and elbow method)

![Customer segments (PCA projection)](reports/figures/customer_pca_plot.png)
![KMeans elbow method](reports/figures/kmeans_elbow_method.png)

## Limitations & Future Improvements

- **Product/ad copy generation is currently rule-based, not a live LLM call.** `product_enrichment.py` and `campaign_generator.py` use keyword-matching and string templates today — this is the most important gap to close next.
- **Inventory levels are synthetic** (randomly assigned), not sourced from a real inventory system.
- **Purchase-intent training pairs are simulated**, not from real clickstream/purchase logs beyond the H&M transaction history used for labels.
- No automated tests or CI yet.
- Product images require a separate Kaggle download and aren't bundled in the repo.

**Roadmap:**
1. ~~Repo cleanup~~ (this pass)
2. Add retrieval-augmented generation: FAISS index over the product catalog to ground ad-copy generation in real product context, replacing `mock_llm_enrich`
3. Wire the RAG-grounded generator into the FastAPI backend as a real endpoint
4. Polish the frontend and deploy (Cloudflare, per the existing `wrangler.jsonc`)
5. Add a proper evaluation write-up (model metrics, ranking quality, RAG grounding quality)

## License

[MIT](LICENSE)

## Contact

Ting Ya Chang — [LinkedIn](https://www.linkedin.com/in/tingya-chang/)
