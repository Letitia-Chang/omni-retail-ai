import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR
from src.features.product_enrichment import enrich_product_catalog


def main():
    print("Loading product catalog...")

    articles = pd.read_csv(PROCESSED_DATA_DIR / "articles.csv")

    print("Enriching product catalog...")

    articles_enriched = enrich_product_catalog(articles)

    output_path = PROCESSED_DATA_DIR / "articles_enriched.csv"
    articles_enriched.to_csv(output_path, index=False)

    print(f"Saved enriched product catalog to: {output_path}")
    print(articles_enriched.head())


if __name__ == "__main__":
    main()