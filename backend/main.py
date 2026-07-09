import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from fastapi.staticfiles import StaticFiles

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

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

def get_client_ip(request: Request) -> str:
    """Real client IP for rate-limiting, not the proxy's own connection.

    Behind Railway's (or any) reverse proxy, `request.client.host` is the
    proxy's internal connection to this container, not the visitor's IP —
    using it as the rate-limit key means every request can look like it's
    from a different, uncounted "client". slowapi's own `get_remote_address`
    only reads that raw peer address, and its `get_ipaddr` helper has a bug
    (it checks for a header literally named "X_FORWARDED_FOR" with an
    underscore, which real HTTP headers never use — they use hyphens,
    "X-Forwarded-For" — so it never actually matches). Read the real header
    name explicitly instead.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # First entry in the chain is the original client.
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


# This is a public portfolio demo on usage-billed hosting (Railway), so every
# route gets a default rate limit — not just /generate-copy — otherwise a
# scripted loop against e.g. /campaigns runs up real compute/bandwidth cost
# even though that endpoint doesn't call any paid API. /generate-copy layers
# a much stricter limit (and a daily cap) on top, since it also costs money
# per call via the Anthropic API.
limiter = Limiter(key_func=get_client_ip, default_limits=["60/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# default_limits above only takes effect on routes without their own
# @limiter.limit(...) decorator via this middleware — without it, undecorated
# routes (everything except /generate-copy) would be completely unlimited.
app.add_middleware(SlowAPIMiddleware)

IMAGES_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "processed"
    / "hm"
    / "images"
)
# Only the ~95 product images actually referenced by the curated campaign
# recommendations are tracked here (~24MB) — the full 29GB Kaggle image set
# lives at data/raw/hm/images/ locally but is gitignored, so it was never
# present on deploy and every request 404'd to a placeholder icon.

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


_campaigns_cache = None
_summary_cache = None
_candidate_summary_cache = None


def load_campaigns():
    # These CSVs are pipeline output, static for the life of the running
    # process (they only change on the next deploy) — re-reading and
    # re-parsing them from disk on every request is wasted compute, which
    # matters on usage-billed hosting even for endpoints with no abuse risk.
    global _campaigns_cache
    if _campaigns_cache is None:
        path = PROCESSED_DATA_DIR / "campaign_recommendations_final.csv"
        _campaigns_cache = pd.read_csv(path)
    return _campaigns_cache


def load_summary():
    global _summary_cache
    if _summary_cache is None:
        path = PROCESSED_DATA_DIR / "campaign_summary.csv"
        _summary_cache = pd.read_csv(path)
    return _summary_cache


def load_candidate_summary():
    global _candidate_summary_cache
    if _candidate_summary_cache is None:
        path = PROCESSED_DATA_DIR / "campaign_candidate_summary.csv"
        _candidate_summary_cache = pd.read_csv(path)
    return _candidate_summary_cache


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


MAX_DAILY_GENERATE_CALLS = 200
_daily_call_tracker = {"date": None, "count": 0}


def check_daily_generate_cap():
    today = datetime.date.today()
    if _daily_call_tracker["date"] != today:
        _daily_call_tracker["date"] = today
        _daily_call_tracker["count"] = 0

    if _daily_call_tracker["count"] >= MAX_DAILY_GENERATE_CALLS:
        raise HTTPException(
            status_code=429,
            detail="Daily ad-copy generation limit reached. Try again tomorrow.",
        )

    _daily_call_tracker["count"] += 1


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
@limiter.limit("10/minute")
def generate_copy(request: Request, body: GenerateCopyRequest):
    """RAG-grounded ad copy for a single product, generated live.

    Retrieves similar catalog products via the FAISS index (TF-IDF + SVD
    embeddings) to ground the prompt, then calls Claude Haiku to write the
    copy — replacing the rule-based template in campaign_generator.py with
    an actual LLM call for the interactive "Regenerate" flow."""
    check_daily_generate_cap()

    index, article_ids, vectorizer, svd = get_product_index()
    articles = get_articles()

    matches = articles[articles["article_id"] == body.article_id]
    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No product found for article_id: {body.article_id}",
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
            customer_segment=body.customer_segment,
            promotion_strategy=body.promotion_strategy,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ad copy generation failed: {exc}",
        )

    return {
        "article_id": body.article_id,
        "copy": copy,
        "grounded_on": similar_products["product_name"].tolist(),
    }