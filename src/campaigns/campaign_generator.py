import pandas as pd


def generate_campaign_message(row: pd.Series) -> str:
    segment = row.get("customer_segment", "customers")
    product_name = row.get("product_name", "this product")
    product_type = row.get("product_type", "item")
    style = row.get("style", "versatile")
    occasion = row.get("occasion", "everyday")
    promotion_strategy = row.get("promotion_strategy", "recommended campaign")

    if "budget" in str(segment).lower() or "discount" in str(promotion_strategy).lower():
        return (
            f"Refresh your wardrobe with {product_name}, a {style} {product_type} "
            f"made for {occasion}. Great style, smart value."
        )

    if "premium" in str(promotion_strategy).lower() or "high-value" in str(segment).lower():
        return (
            f"Elevate your look with {product_name}, a carefully selected "
            f"{product_type} designed for {occasion} and confident everyday style."
        )

    if "loyal" in str(segment).lower() or "regular" in str(segment).lower():
        return (
            f"Picked for your style: {product_name}. A {style} {product_type} "
            f"that fits naturally into your {occasion} wardrobe."
        )

    return (
        f"Discover {product_name}, a {style} {product_type} perfect for "
        f"{occasion}. A simple way to refresh your style."
    )


def generate_campaign_table(ranked_campaigns: pd.DataFrame) -> pd.DataFrame:
    campaigns = ranked_campaigns.copy()

    campaigns["campaign_message"] = campaigns.apply(
        generate_campaign_message,
        axis=1,
    )

    return campaigns


def create_campaign_summary(campaigns: pd.DataFrame) -> pd.DataFrame:
    summary = (
        campaigns.groupby("customer_segment")
        .agg(
            total_recommended_products=("article_id", "count"),
            avg_campaign_score=("avg_campaign_score", "mean"),
            avg_purchase_probability=("avg_purchase_probability", "mean"),
            total_reachable_customers=("customer_count", "sum"),
        )
        .reset_index()
    )

    return summary