import numpy as np
import pandas as pd


def prepare_campaign_dataset(
    predictions: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    segment_strategy: pd.DataFrame,
) -> pd.DataFrame:
    campaign_df = predictions.merge(
        customers[["customer_id", "customer_segment"]],
        on="customer_id",
        how="left",
    )

    product_cols = [
        "article_id",
        "product_name",
        "product_type",
        "product_group",
        "color_group",
        "index_group",
        "garment_group",
        "avg_selling_price",
        "style",
        "occasion",
        "material_hint",
        "target_audience",
        "selling_points",
        "marketing_keywords",
        "copy_angle",
    ]

    available_product_cols = [col for col in product_cols if col in articles.columns]

    campaign_df = campaign_df.merge(
        articles[available_product_cols],
        on="article_id",
        how="left",
    )

    campaign_df = campaign_df.merge(
        segment_strategy,
        on="customer_segment",
        how="left",
    )

    campaign_df = campaign_df.rename(
        columns={
            "copy_angle_x": "product_copy_angle",
            "copy_angle_y": "segment_copy_angle",
        }
    )

    return campaign_df


def add_inventory_signal(
    campaign_df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    df = campaign_df.copy()
    np.random.seed(random_state)

    df["inventory_level"] = np.random.choice(
        ["low", "medium", "high"],
        size=len(df),
        p=[0.2, 0.5, 0.3],
    )

    inventory_score_map = {
        "low": 0.3,
        "medium": 0.7,
        "high": 1.0,
    }

    df["inventory_score"] = df["inventory_level"].map(inventory_score_map)

    return df


def add_product_score(campaign_df: pd.DataFrame) -> pd.DataFrame:
    df = campaign_df.copy()
    df["product_score"] = df["purchase_probability"] * df["inventory_score"]
    return df


def calculate_segment_match_scores(campaign_df: pd.DataFrame) -> pd.DataFrame:
    df = campaign_df.copy()

    low_price_threshold = df["avg_selling_price"].quantile(0.35)
    mid_price_threshold = df["avg_selling_price"].quantile(0.50)
    high_price_threshold = df["avg_selling_price"].quantile(0.65)

    def calculate_segment_match(row):
        segment = str(row.get("customer_segment", "")).lower()
        style = str(row.get("style", "")).lower()
        occasion = str(row.get("occasion", "")).lower()
        product_group = str(row.get("product_group", "")).lower()

        copy_angle_text = " ".join(
            [
                str(row.get("product_copy_angle", "")),
                str(row.get("segment_copy_angle", "")),
            ]
        ).lower()

        avg_price = row.get("avg_selling_price", 0)

        score = 0.5

        if "budget" in segment:
            if avg_price <= low_price_threshold:
                score += 0.25
            if any(word in copy_angle_text for word in ["value", "affordable", "practical"]):
                score += 0.15

        if "premium" in segment or "high-value" in segment:
            if avg_price >= high_price_threshold:
                score += 0.25
            if any(word in copy_angle_text for word in ["premium", "quality", "elevated", "exclusive"]):
                score += 0.15

        if "fashion" in segment or "engaged" in segment:
            if any(word in style for word in ["streetwear", "elegant", "sporty", "trendy"]):
                score += 0.20
            if any(word in occasion for word in ["outing", "work", "vacation"]):
                score += 0.10

        if "loyal" in segment or "regular" in segment:
            if any(word in product_group for word in ["garment", "accessories", "shoes"]):
                score += 0.15
            score += 0.10

        if "inactive" in segment or "occasional" in segment:
            if avg_price <= mid_price_threshold:
                score += 0.20
            if any(word in copy_angle_text for word in ["easy", "everyday", "low-commitment"]):
                score += 0.10

        return min(score, 1.0)

    df["segment_match_score"] = df.apply(calculate_segment_match, axis=1)

    return df


def add_campaign_score(
    campaign_df: pd.DataFrame,
    purchase_weight: float = 0.50,
    inventory_weight: float = 0.25,
    segment_match_weight: float = 0.25,
) -> pd.DataFrame:
    df = campaign_df.copy()

    df["campaign_score"] = (
        purchase_weight * df["purchase_probability"]
        + inventory_weight * df["inventory_score"]
        + segment_match_weight * df["segment_match_score"]
    )

    return df


def assign_promotion_strategy(campaign_df: pd.DataFrame) -> pd.DataFrame:
    df = campaign_df.copy()

    def strategy(row):
        intent = row["purchase_probability"]
        inventory = row["inventory_level"]

        if inventory == "high" and intent >= 0.90:
            return "Promote aggressively"

        if inventory == "low" and intent >= 0.85:
            return "Premium positioning"

        if inventory == "medium" and intent >= 0.80:
            return "Personalized recommendation"

        if inventory == "high" and intent < 0.80:
            return "Discount campaign"

        if inventory == "medium" and intent < 0.80:
            return "Awareness campaign"

        return "Deprioritize"

    df["promotion_strategy"] = df.apply(strategy, axis=1)

    return df


def normalize_campaign_scores(ranked_campaigns: pd.DataFrame) -> pd.DataFrame:
    df = ranked_campaigns.copy()

    df["raw_campaign_score"] = df["avg_campaign_score"]

    min_score = df["avg_campaign_score"].min()
    max_score = df["avg_campaign_score"].max()

    if max_score > min_score:
        df["campaign_score"] = (
            (df["avg_campaign_score"] - min_score)
            / (max_score - min_score)
        )
    else:
        df["campaign_score"] = df["avg_campaign_score"]

    df["purchase_probability"] = df["avg_purchase_probability"]

    return df


def rank_campaigns_by_segment(
    campaign_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    group_cols = [
        "customer_segment",
        "article_id",
        "product_name",
        "product_type",
        "product_group",
        "color_group",
        "style",
        "occasion",
        "target_audience",
        "promotion_strategy",
        "recommended_strategy",
        "product_copy_angle",
        "segment_copy_angle",
    ]

    group_cols = [
        col for col in group_cols
        if col in campaign_df.columns
    ]

    segment_product_scores = (
        campaign_df.groupby(group_cols, as_index=False)
        .agg(
            avg_purchase_probability=("purchase_probability", "mean"),
            avg_inventory_score=("inventory_score", "mean"),
            avg_segment_match_score=("segment_match_score", "mean"),
            avg_campaign_score=("campaign_score", "mean"),
            customer_count=("customer_id", "nunique"),
        )
    )

    # Deduplicate visually/marketing-equivalent products
    dedupe_cols = [
        "customer_segment",
        "product_name",
        "product_type",
        "product_group",
        "color_group",
        "style",
        "occasion",
        "target_audience",
    ]

    dedupe_cols = [
        col for col in dedupe_cols
        if col in segment_product_scores.columns
    ]

    segment_product_scores = (
        segment_product_scores
        .sort_values("avg_campaign_score", ascending=False)
        .drop_duplicates(subset=dedupe_cols)
    )

    ranked_campaigns = (
        segment_product_scores
        .sort_values(
            ["customer_segment", "avg_campaign_score"],
            ascending=[True, False],
        )
        .groupby("customer_segment")
        .head(top_n)
        .reset_index(drop=True)
    )

    ranked_campaigns = normalize_campaign_scores(
        ranked_campaigns
    )

    return ranked_campaigns

def create_ranking_explanations(ranked_campaigns: pd.DataFrame) -> pd.DataFrame:
    df = ranked_campaigns.copy()

    df["ranking_explanation"] = df.apply(
        lambda row: (
            f"Recommended for {row['customer_segment']} because it has "
            f"an average purchase probability of "
            f"{row['avg_purchase_probability']:.2f}, "
            f"an inventory score of {row['avg_inventory_score']:.2f}, "
            f"a segment match score of "
            f"{row['avg_segment_match_score']:.2f}, "
            f"and a raw campaign score of "
            f"{row['raw_campaign_score']:.2f}."
        ),
        axis=1,
    )

    return df


def build_ranked_campaign_recommendations(
    predictions: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    segment_strategy: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    campaign_df = prepare_campaign_dataset(
        predictions,
        articles,
        customers,
        segment_strategy,
    )

    campaign_df = add_inventory_signal(campaign_df)
    campaign_df = add_product_score(campaign_df)
    campaign_df = calculate_segment_match_scores(campaign_df)
    campaign_df = add_campaign_score(campaign_df)
    campaign_df = assign_promotion_strategy(campaign_df)

    ranked_campaigns = rank_campaigns_by_segment(campaign_df, top_n=top_n)
    ranked_campaigns = create_ranking_explanations(ranked_campaigns)

    return ranked_campaigns