from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from fastapi.staticfiles import StaticFiles

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from src.utils.paths import PROCESSED_DATA_DIR, RAG_MODEL_DIR
from src.rag.product_index import load_product_index, retrieve_similar_products
from src.rag.copy_generator import generate_grounded_copy
from src.features.product_enrichment import create_product_context

load_dotenv()

app = FastAPI(
    title="OmniRetail AI API",
    description="Backend API for campaign recommendations and retail marketing insights.",
    version="0.1.0",
)

IMAGES_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "hm"
    / "images"
)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/images",
    StaticFiles(directory=IMAGES_DIR),
    name="images",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_campaigns():
    path = PROCESSED_DATA_DIR / "campaign_recommendations_final.csv"
    return pd.read_csv(path)


def load_summary():
    path = PROCESSED_DATA_DIR / "campaign_summary.csv"
    return pd.read_csv(path)


def load_candidate_summary():
    path = PROCESSED_DATA_DIR / "campaign_candidate_summary.csv"
    return pd.read_csv(path)


_product_index_cache = None
_articles_cache = None


def get_product_index():
    """Load the FAISS product index lazily and cache it for the process
    lifetime — it's ~50MB and rebuilding/reloading it per request would make
    every /generate-copy call noticeably slower for no benefit."""
    global _product_index_cache

    if _product_index_cache is None:
        if not (RAG_MODEL_DIR / "product_catalog.faiss").exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Product index not found. Run "
                    "`python scripts/build_product_index.py` first."
                ),
            )
        _product_index_cache = load_product_index(RAG_MODEL_DIR)

    return _product_index_cache


def get_articles():
    global _articles_cache

    if _articles_cache is None:
        _articles_cache = pd.read_csv(
            PROCESSED_DATA_DIR / "articles_enriched.csv"
        )

    return _articles_cache


class GenerateCopyRequest(BaseModel):
    article_id: int
    customer_segment: str
    promotion_strategy: str = "Promote aggressively"


@app.get("/")
def root():
    return {
        "message": "OmniRetail AI API is running.",
        "available_endpoints": [
            "/segments",
            "/campaigns",
            "/campaigns/{segment}",
            "/summary",
            "/analytics/candidate-summary",
            "/generate-copy",
        ],
    }


@app.get("/segments")
def get_segments():
    campaigns = load_campaigns()
    segments = sorted(campaigns["customer_segment"].dropna().unique().tolist())

    return {"segments": segments}


@app.get("/campaigns")
def get_campaigns():
    campaigns = load_campaigns()
    return campaigns.to_dict(orient="records")


@app.get("/campaigns/{segment}")
def get_campaigns_by_segment(segment: str):
    campaigns = load_campaigns()

    filtered = campaigns[
        campaigns["customer_segment"].str.lower() == segment.lower()
    ]

    if filtered.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No campaigns found for segment: {segment}",
        )

    return filtered.to_dict(orient="records")


@app.get("/summary")
def get_summary():
    summary = load_summary()
    return summary.to_dict(orient="records")


@app.get("/analytics/candidate-summary")
def get_candidate_summary():
    """Strategy/inventory distributions across every scored candidate
    product, not just the top-N recommendations `/campaigns` returns —
    used by the Analytics dashboard to show catalog-wide distributions
    rather than a biased sample of only the curated top picks."""
    summary = load_candidate_summary()
    # avg_purchase_probability is NaN for strategy_mix/inventory_distribution
    # rows (they don't carry that metric) — swap to None, which is valid JSON.
    # (Must cast to object first: assigning None into a float64 column just
    # coerces it back to NaN, which the JSON encoder then rejects.)
    summary = summary.astype(object).where(pd.notnull(summary), None)
    return summary.to_dict(orient="records")


@app.post("/generate-copy")
def generate_copy(request: GenerateCopyRequest):
    """RAG-grounded ad copy for a single product, generated live.

    Retrieves similar catalog products via the FAISS index (TF-IDF + SVD
    embeddings) to ground the prompt, then calls Claude Haiku to write the
    copy — replacing the rule-based template in campaign_generator.py with
    an actual LLM call for the interactive "Regenerate" flow."""
    index, article_ids, vectorizer, svd = get_product_index()
    articles = get_articles()

    matches = articles[articles["article_id"] == request.article_id]
    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No product found for article_id: {request.article_id}",
        )
    product_row = matches.iloc[0]

    query_text = create_product_context(product_row)
    similar_products = retrieve_similar_products(
        query_text, index, article_ids, articles, vectorizer, svd, top_k=5
    )

    try:
        copy = generate_grounded_copy(
            product_row=product_row,
            similar_products=similar_products,
            customer_segment=request.customer_segment,
            promotion_strategy=request.promotion_strategy,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ad copy generation failed: {exc}",
        )

    return {
        "article_id": request.article_id,
        "copy": copy,
        "grounded_on": similar_products["product_name"].tolist(),
    }