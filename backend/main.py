from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
from fastapi.staticfiles import StaticFiles

import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR

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


@app.get("/")
def root():
    return {
        "message": "OmniRetail AI API is running.",
        "available_endpoints": [
            "/segments",
            "/campaigns",
            "/campaigns/{segment}",
            "/summary",
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