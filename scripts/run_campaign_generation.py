import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR
from src.campaigns.campaign_generator import (
    generate_campaign_table,
    create_campaign_summary,
)


def main():
    print("Loading ranked campaign recommendations...")

    ranked_campaigns = pd.read_csv(
        PROCESSED_DATA_DIR / "ranked_campaign_recommendations.csv"
    )

    print("Generating campaign messages...")

    campaign_table = generate_campaign_table(ranked_campaigns)
    campaign_summary = create_campaign_summary(campaign_table)

    campaign_output_path = PROCESSED_DATA_DIR / "campaign_recommendations_final.csv"
    summary_output_path = PROCESSED_DATA_DIR / "campaign_summary.csv"

    campaign_table.to_csv(campaign_output_path, index=False)
    campaign_summary.to_csv(summary_output_path, index=False)

    print(f"Saved campaign recommendations to: {campaign_output_path}")
    print(f"Saved campaign summary to: {summary_output_path}")

    print(campaign_table.head())


if __name__ == "__main__":
    main()