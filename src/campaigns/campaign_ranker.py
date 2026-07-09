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
    # Inventory is a product-level attribute: assign one level per article_id
    # (not per customer-article row), otherwise the same product can randomly
    # land in different inventory buckets across customers, which then lets
    # downstream ranking cherry-pick whichever draw happened to be "high".
    df = campaign_df.copy()
    rng = np.random.RandomState(random_state)

    unique_articles = df[["article_id"]].drop_duplicates().reset_index(drop=True)
    unique_articles["inventory_level"] = rng.choice(
        ["low", "medium", "high"],
        size=len(unique_articles),
        p=[0.2, 0.5, 0.3],
    )

    inventory_score_map = {
        "low": 0.3,
        "medium": 0.7,
        "high": 1.0,
    }
    unique_articles["inventory_score"] = unique_articles["inventory_level"].map(
        inventory_score_map
    )

    df = df.merge(unique_articles, on="article_id", how="left")

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


def normalize_campaign_scores(
    ranked_campaigns: pd.DataFrame,
    min_score=None,
    max_score=None,
) -> pd.DataFrame:
    """Min-max normalize avg_campaign_score into a 0-1 "match %" score.

    min_score/max_score default to the bounds of `ranked_campaigns` itself,
    but callers should pass the bounds of the *full* candidate pool instead
    when `ranked_campaigns` is already a curated top-N slice. The top-N's
    raw scores cluster tightly (e.g. 0.91-0.995 in practice) since they're
    all "good" picks by construction — normalizing against just that narrow
    band stretches a real few-percent quality gap between segments into a
    misleading 6%-97% spread. Normalizing against the full population's
    much wider range keeps the same 0-1 scale meaningful.
    """
    df = ranked_campaigns.copy()

    df["raw_campaign_score"] = df["avg_campaign_score"]

    if min_score is None:
        min_score = df["avg_campaign_score"].min()
    if max_score is None:
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


def build_candidate_pool(campaign_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (segment, product) candidate, scored and deduplicated,
    across the *entire* catalog — before any top-N-per-segment truncation.

    This is the full population `rank_campaigns_by_segment` samples its
    top picks from; it's also what the Analytics dashboard should draw
    catalog-wide distributions from, since the top-N alone is a biased
    sample (it over-represents high-inventory items, since inventory_score
    is a direct positive term in campaign_score).
    """
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
        "inventory_level",
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

    return segment_product_scores


def rank_campaigns_by_segment(
    campaign_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    segment_product_scores = build_candidate_pool(campaign_df)

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
        ranked_campaigns,
        min_score=segment_product_scores["avg_campaign_score"].min(),
        max_score=segment_product_scores["avg_campaign_score"].max(),
    )

    return ranked_campaigns


def summarize_candidate_pool(candidate_pool: pd.DataFrame) -> pd.DataFrame:
    """Catalog-wide distributions (not just the curated top-N) for the
    Analytics dashboard: how promotion strategy and inventory are actually
    distributed across every scored (segment, product) candidate.

    Returns a long-format table: columns [metric, key, count, avg_purchase_probability].
    """
    rows = []

    strategy_counts = candidate_pool["promotion_strategy"].value_counts()
    for strategy, count in strategy_counts.items():
        rows.append(
            {
                "metric": "strategy_mix",
                "key": strategy,
                "count": int(count),
                "avg_purchase_probability": None,
            }
        )

    inventory_groups = candidate_pool.groupby("inventory_level")
    for level, group in inventory_groups:
        rows.append(
            {
                "metric": "inventory_distribution",
                "key": level,
                "count": int(len(group)),
                "avg_purchase_probability": None,
            }
        )

    for level, group in inventory_groups:
        rows.append(
            {
                "metric": "inventory_risk",
                "key": level,
                "count": int(len(group)),
                "avg_purchase_probability": float(
                    group["avg_purchase_probability"].mean()
                ),
            }
        )

    # Per-segment purchase probability across the *entire* candidate pool —
    # the curated top-N recommendations are selected for near-highest
    # purchase probability by construction (it's a direct term in
    # campaign_score), so averaging over just those always lands near 100%
    # regardless of segment. The candidate pool gives the real spread.
    segment_groups = candidate_pool.groupby("customer_segment")
    for segment, group in segment_groups:
        rows.append(
            {
                "metric": "segment_purchase_probability",
                "key": segment,
                "count": int(len(group)),
                "avg_purchase_probability": float(
                    group["avg_purchase_probability"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)

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


def score_campaign_candidates(
    predictions: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    segment_strategy: pd.DataFrame,
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

    return campaign_df


def build_ranked_campaign_recommendations(
    predictions: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    segment_strategy: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    campaign_df = score_campaign_candidates(
        predictions,
        articles,
        customers,
        segment_strategy,
    )

    ranked_campaigns = rank_campaigns_by_segment(campaign_df, top_n=top_n)
    ranked_campaigns = create_ranking_explanations(ranked_campaigns)

    return ranked_campaigns


def build_campaign_candidate_summary(
    predictions: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    segment_strategy: pd.DataFrame,
) -> pd.DataFrame:
    """Catalog-wide strategy/inventory distributions, computed over every
    scored candidate rather than just the top-N recommendations shown in
    the dashboard. See `summarize_candidate_pool` for the shape."""
    campaign_df = score_campaign_candidates(
        predictions,
        articles,
        customers,
        segment_strategy,
    )

    candidate_pool = build_candidate_pool(campaign_df)

    return summarize_candidate_pool(candidate_pool)