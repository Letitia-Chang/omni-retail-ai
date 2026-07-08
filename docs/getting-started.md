# Getting Started

## 1. Clone and install Python dependencies

```bash
git clone git@github.com:Letitia-Chang/omni-retail-ai.git
cd omni-retail-ai
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## 2. Get the data

Requires a [Kaggle](https://www.kaggle.com/) account with `kaggle.json` credentials in `~/.kaggle/` (and having accepted the competition rules).

```bash
python scripts/download_data.py
```

Then run [`notebooks/01_data_schema_and_cleaning.ipynb`](../notebooks/01_data_schema_and_cleaning.ipynb) to produce the cleaned `data/processed/hm/` tables the pipeline expects (`customers.csv`, `articles.csv`, `transactions.csv`).

## 3. Run the ML pipeline

```bash
python scripts/run_full_pipeline.py
```

This runs segmentation → product enrichment → purchase-intent modeling → campaign ranking → campaign generation in sequence, writing outputs to `data/processed/hm/` and trained models to `saved_models/`.

## 4. Build the RAG product index

```bash
python scripts/build_product_index.py
```

Embeds the enriched product catalog (TF-IDF + SVD) and builds a FAISS index at `saved_models/rag/` — this grounds the live ad-copy generation endpoint. Takes well under a minute (no GPU or large model download required).

Then copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` (get one at [console.anthropic.com](https://console.anthropic.com/)) — required for `/generate-copy`. Without it, the dashboard still runs fine and falls back to the pre-computed ad-copy template.

## 5. Run the backend

```bash
uvicorn backend.main:app --reload
```

Serves at `http://127.0.0.1:8000` — see `/` for available endpoints.

## 6. Run the frontend

```bash
cd frontend
cp .env.example .env   # points the dashboard at the local API
npm install
npm run dev
```

Dashboard runs at `http://localhost:3000` (TanStack Start dev server).

Next: see the [Demo Walkthrough](demo.md) for a tour of the five views, or [Deployment](deployment.md) if you want to ship it.
