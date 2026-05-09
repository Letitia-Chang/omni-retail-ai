import pandas as pd


def create_product_context(row: pd.Series) -> str:
    return (
        f"Product name: {row.get('product_name', '')}\n"
        f"Type: {row.get('product_type', '')}\n"
        f"Group: {row.get('product_group', '')}\n"
        f"Color: {row.get('color_group', '')}\n"
        f"Index group: {row.get('index_group', '')}\n"
        f"Garment group: {row.get('garment_group', '')}\n"
        f"Description: {row.get('description', '')}"
    )


def infer_style(row: pd.Series) -> str:
    text = " ".join(
        [
            str(row.get("product_name", "")),
            str(row.get("product_type", "")),
            str(row.get("garment_group", "")),
            str(row.get("description", "")),
        ]
    ).lower()

    if any(word in text for word in ["sport", "training", "running", "gym"]):
        return "sporty"
    if any(word in text for word in ["dress", "blouse", "shirt", "tailored"]):
        return "elegant"
    if any(word in text for word in ["hoodie", "sweatshirt", "tee", "t-shirt"]):
        return "casual"
    if any(word in text for word in ["denim", "jacket", "cargo"]):
        return "streetwear"

    return "everyday"


def infer_occasion(row: pd.Series) -> str:
    text = " ".join(
        [
            str(row.get("product_name", "")),
            str(row.get("product_type", "")),
            str(row.get("garment_group", "")),
            str(row.get("description", "")),
        ]
    ).lower()

    if any(word in text for word in ["sport", "training", "running", "gym"]):
        return "workout"
    if any(word in text for word in ["dress", "blouse", "shirt"]):
        return "work_or_outing"
    if any(word in text for word in ["swim", "bikini"]):
        return "vacation"

    return "daily_wear"


def mock_llm_enrich(row: pd.Series) -> dict:
    style = row.get("style", "everyday")
    occasion = row.get("occasion", "daily_wear")
    product_type = row.get("product_type", "product")
    color = row.get("color_group", "neutral")
    target_audience = row.get("index_group", "general customers")

    return {
        "style": style,
        "occasion": occasion,
        "material_hint": "unknown",
        "target_audience": target_audience,
        "selling_points": [
            f"Easy to style {product_type}",
            f"Versatile for {occasion}",
        ],
        "marketing_keywords": [
            str(style),
            str(occasion),
            str(color),
        ],
        "copy_angle": f"{style} and {occasion} focused",
    }


def enrich_product_catalog(articles: pd.DataFrame) -> pd.DataFrame:
    products = articles.copy()

    products["product_context"] = products.apply(create_product_context, axis=1)
    products["style"] = products.apply(infer_style, axis=1)
    products["occasion"] = products.apply(infer_occasion, axis=1)

    enrichment_records = []

    for _, row in products.iterrows():
        enrichment = mock_llm_enrich(row)
        enrichment_records.append(
            {
                "article_id": row["article_id"],
                **enrichment,
            }
        )

    enrichment_df = pd.DataFrame(enrichment_records)

    enriched_articles = articles.merge(
        enrichment_df,
        on="article_id",
        how="left",
    )

    return enriched_articles