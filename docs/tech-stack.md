# Tech Stack

| Layer | Tools |
|---|---|
| Data / ML | Python, pandas, NumPy, scikit-learn, XGBoost, joblib |
| RAG | FAISS, scikit-learn (TF-IDF + TruncatedSVD embeddings), Anthropic Claude (`claude-haiku-4-5`) |
| Data source | [Kaggle H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) dataset, via `kagglehub` |
| Backend | FastAPI, Uvicorn, slowapi (rate limiting) |
| Frontend | React 19, TanStack Start/Router/Query, Tailwind CSS, Radix UI |
| Deploy target | Frontend on Cloudflare Workers (Vite plugin + `wrangler` config included); backend on Render (`render.yaml` blueprint included) |

## Notable choices worth explaining

**Why TF-IDF + SVD instead of a neural embedding model for RAG?** This was a deliberate, platform-driven decision: no current `torch` wheel supports both Intel macOS and the `numpy>=2` the rest of this stack (`pandas`, `scikit-learn`) requires. Rather than downgrade the whole pipeline to satisfy one dependency, the retrieval layer uses classical TF-IDF + TruncatedSVD embeddings feeding a FAISS `IndexFlatIP` index — FAISS is still the real retrieval technology in the stack, just paired with lexical rather than neural embeddings. See [Model & System Evaluation](../reports/evaluation.md) for a quantitative check on how well this actually performs (94–99% category-match rate on sampled queries).

**Why Claude Haiku for generation?** The ad-copy task — a couple of grounded sentences per product, called live and interactively — is short-form and latency-sensitive, which is exactly Haiku's profile. There was no need for a frontier-tier model here.

**Why two separate deploy targets?** Cloudflare Workers doesn't run a Python/pandas/FAISS stack, so the backend (FastAPI + the full ML/RAG pipeline) deploys to Render instead, while the frontend deploys to Cloudflare Workers. See [Deployment](deployment.md).
