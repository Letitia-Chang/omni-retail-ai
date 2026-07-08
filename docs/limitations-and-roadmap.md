# Limitations & Roadmap

## Known limitations

- **Bulk product enrichment** (`product_enrichment.py`) is still rule-based (keyword-matching for style/occasion tags) — only the on-demand ad-copy call in the Generator view (`/generate-copy`) uses a live, RAG-grounded LLM call. Extending the LLM call to bulk enrichment would mean ~105K API calls, a batch-processing / cost tradeoff worth its own decision rather than doing by default.
- **Catalog embeddings are TF-IDF + SVD, not neural embeddings** — a deliberate platform-driven choice; see [Tech Stack](tech-stack.md) for why.
- **Inventory levels are synthetic** (randomly assigned per product), not sourced from a real inventory system.
- **Purchase-intent training pairs are simulated**, not from real clickstream/purchase logs beyond the H&M transaction history used for labels. See [Model & System Evaluation](../reports/evaluation.md) for what this means for the reported ROC-AUC.
- **Cluster-to-segment-name mapping is fragile** — it's a hardcoded `{0: "High-Value One-Time Buyers", ...}` index lookup, not sorted by any cluster property. Rerunning KMeans on different data could silently relabel segments.
- **No automated tests or CI.**
- **Product images require a separate Kaggle download** and aren't bundled in the repo (~30GB) — the dashboard degrades gracefully (placeholder icons) when they're missing, including on the live deployment.
- **CORS is wide open** (`allow_origins=["*"]`) on the backend — acceptable for a public demo API, not for a real production system with sensitive data.

## Completed roadmap

1. ~~Repo cleanup~~
2. ~~Add retrieval-augmented generation~~ — FAISS index over the product catalog, grounding ad-copy generation in real product context
3. ~~Wire the RAG-grounded generator into the FastAPI backend~~ — `POST /generate-copy`, called live from the Generator view's "Regenerate" button
4. ~~Polish the frontend and deploy~~ — see [Deployment](deployment.md) (Cloudflare Workers + Render)
5. ~~Add a proper evaluation write-up~~ — see [Model & System Evaluation](../reports/evaluation.md)
6. ~~Rate-limit the public LLM endpoint~~ — 10/minute per IP + 200/day global cap on `/generate-copy`

## Next up

- **Tests + CI** — currently zero automated tests. Even a small `pytest` suite (campaign scoring math, the ranking pipeline, an API smoke test) plus a GitHub Actions workflow on push would be the highest-leverage next addition.
- **Fix the cluster-name mapping** — map segment names by cluster characteristics (e.g., sort by avg spend) instead of raw cluster index.
- **Custom domains** — currently on the default `workers.dev` and `onrender.com` subdomains.
- **Re-pin/retrain saved models** — the segmentation and purchase-intent models were saved under an older scikit-learn version than what's currently installed; harmless today (a version-mismatch warning on load) but worth cleaning up.
