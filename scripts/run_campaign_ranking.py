import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR
from src.campaigns.campaign_ranker import (
    build_campaign_candidate_summary,
    build_ranked_campaign_recommendations,
)


def main():
    print("Loading data...")

    predictions = pd.read_csv(
        PROCESSED_DATA_DIR / "purchase_intent_predictions.csv"
    )

    articles = pd.read_csv(
        PROCESSED_DATA_DIR / "articles_enriched.csv"
    )

    customers = pd.read_csv(
        PROCESSED_DATA_DIR / "customer_segments.csv"
    )

    segment_strategy = pd.read_csv(
        PROCESSED_DATA_DIR / "segment_strategy.csv"
    )

    print("Building ranked campaign recommendations...")

    ranked_campaigns = build_ranked_campaign_recommendations(
        predictions=predictions,
        articles=articles,
        customers=customers,
        segment_strategy=segment_strategy,
        top_n=20,
    )

    output_path = (
        PROCESSED_DATA_DIR
        / "ranked_campaign_recommendations.csv"
    )

    ranked_campaigns.to_csv(output_path, index=False)

    print(f"Saved ranked campaigns to: {output_path}")
    print(ranked_campaigns.head())

    print("Building catalog-wide candidate summary...")

    candidate_summary = build_campaign_candidate_summary(
        predictions=predictions,
        articles=articles,
        customers=customers,
        segment_strategy=segment_strategy,
    )

    summary_output_path = (
        PROCESSED_DATA_DIR
        / "campaign_candidate_summary.csv"
    )

    candidate_summary.to_csv(summary_output_path, index=False)

    print(f"Saved candidate summary to: {summary_output_path}")
    print(candidate_summary)


if __name__ == "__main__":
    main()