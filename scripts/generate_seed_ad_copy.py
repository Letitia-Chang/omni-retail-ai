"""One-time script: generates real, RAG-grounded ad copy for a curated
sample of products and commits the result as a static CSV.

This is *not* re-run at deploy time (unlike build_product_index.py) — it
costs real Claude API calls, so it's run once locally and the output is
checked into the repo, the same way the bundled product images are. The
Ad Copies page reads this file for its permanent "sample library"; live
"Regenerate" clicks from the Generator page are a separate, session-only
path that never touches this file (see frontend sessionStorage handling).

Usage: PYTHONPATH=. python scripts/generate_seed_ad_copy.py
"""

import datetime
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.utils.paths import PROCESSED_DATA_DIR, RAG_MODEL_DIR
from src.rag.product_index import load_product_index, retrieve_similar_products
from src.rag.copy_generator import generate_grounded_copy
from src.features.product_enrichment import create_product_context

ROWS_PER_SEGMENT = 4


def pick_seed_products(campaigns: pd.DataFrame) -> pd.DataFrame:
    """Up to ROWS_PER_SEGMENT products per segment, spread across whatever
    distinct strategies that segment's curated picks span (falls back to
    highest-score picks once strategies run out), so the seeded library
    shows the same range the diversified curation actually produces."""
    picks = []
    for segment, group in campaigns.groupby("customer_segment"):
        group = group.sort_values("campaign_score", ascending=False)
        seen_strategies = set()
        rows = []
        for _, row in group.iterrows():
            if len(rows) >= ROWS_PER_SEGMENT:
                break
            if row["promotion_strategy"] in seen_strategies:
                continue
            seen_strategies.add(row["promotion_strategy"])
            rows.append(row)
        # Backfill with next-best if the segment has fewer distinct strategies.
        if len(rows) < ROWS_PER_SEGMENT:
            picked_ids = {r["article_id"] for r in rows}
            for _, row in group.iterrows():
                if len(rows) >= ROWS_PER_SEGMENT:
                    break
                if row["article_id"] in picked_ids:
                    continue
                rows.append(row)
                picked_ids.add(row["article_id"])
        picks.extend(rows)
    return pd.DataFrame(picks)


def main():
    print("Loading data...")
    campaigns = pd.read_csv(PROCESSED_DATA_DIR / "campaign_recommendations_final.csv")
    articles = pd.read_csv(PROCESSED_DATA_DIR / "articles_enriched.csv")
    index, article_ids, vectorizer, svd = load_product_index(RAG_MODEL_DIR)

    seed_products = pick_seed_products(campaigns)
    print(f"Selected {len(seed_products)} seed products across "
          f"{seed_products['customer_segment'].nunique()} segments.")

    rows = []
    for i, campaign_row in seed_products.reset_index(drop=True).iterrows():
        article_id = campaign_row["article_id"]
        matches = articles[articles["article_id"] == article_id]
        if matches.empty:
            print(f"  skip {article_id}: not found in articles_enriched.csv")
            continue
        product_row = matches.iloc[0]

        query_text = create_product_context(product_row)
        similar_products = retrieve_similar_products(
            query_text, index, article_ids, articles, vectorizer, svd, top_k=5
        )

        print(f"  [{i + 1}/{len(seed_products)}] generating for "
              f"{product_row['product_name']} ({campaign_row['customer_segment']})...")

        try:
            copy = generate_grounded_copy(
                product_row=product_row,
                similar_products=similar_products,
                customer_segment=campaign_row["customer_segment"],
                promotion_strategy=campaign_row["promotion_strategy"],
            )
        except Exception as exc:
            print(f"    failed: {exc}")
            continue

        rows.append({
            "article_id": int(article_id),
            "product_name": product_row["product_name"],
            "customer_segment": campaign_row["customer_segment"],
            "promotion_strategy": campaign_row["promotion_strategy"],
            "campaign_score": campaign_row["campaign_score"],
            "purchase_probability": campaign_row["purchase_probability"],
            "generated_copy": copy,
            "grounded_on": "; ".join(similar_products["product_name"].tolist()),
            "status": "approved",
            "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        })

        time.sleep(0.3)  # light pacing, not required at this volume

    output = pd.DataFrame(rows)
    output_path = PROCESSED_DATA_DIR / "pregenerated_ad_copy.csv"
    output.to_csv(output_path, index=False)
    print(f"\nSaved {len(output)} rows to {output_path}")
    if not output.empty:
        print(output[["product_name", "customer_segment", "promotion_strategy"]])


if __name__ == "__main__":
    main()
