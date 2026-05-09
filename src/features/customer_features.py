import pandas as pd


def build_customer_behavior_features(transactions: pd.DataFrame) -> pd.DataFrame:
    transactions = transactions.copy()
    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])

    snapshot_date = transactions["transaction_date"].max() + pd.Timedelta(days=1)

    customer_behavior = (
        transactions.groupby("customer_id")
        .agg(
            total_transactions=("article_id", "count"),
            unique_products=("article_id", "nunique"),
            total_spend=("price", "sum"),
            avg_price=("price", "mean"),
            max_price=("price", "max"),
            first_purchase_date=("transaction_date", "min"),
            last_purchase_date=("transaction_date", "max"),
        )
        .reset_index()
    )

    customer_behavior["days_since_last_purchase"] = (
        snapshot_date - customer_behavior["last_purchase_date"]
    ).dt.days

    customer_behavior["customer_lifetime_days"] = (
        customer_behavior["last_purchase_date"]
        - customer_behavior["first_purchase_date"]
    ).dt.days + 1

    customer_behavior["purchase_frequency"] = (
        customer_behavior["total_transactions"]
        / customer_behavior["customer_lifetime_days"]
    )

    return customer_behavior


def get_top_value(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    output_col: str,
) -> pd.DataFrame:
    return (
        df.groupby([group_col, value_col])
        .size()
        .reset_index(name="count")
        .sort_values([group_col, "count"], ascending=[True, False])
        .drop_duplicates(group_col)
        [[group_col, value_col]]
        .rename(columns={value_col: output_col})
    )


def build_customer_preference_features(
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    product_cols = [
        "article_id",
        "product_group",
        "index_group",
        "garment_group",
        "color_group",
    ]

    txn = transactions.merge(
        articles[product_cols],
        on="article_id",
        how="left",
    )

    preference_features = None

    mappings = [
        ("product_group", "favorite_product_group"),
        ("index_group", "favorite_index_group"),
        ("garment_group", "favorite_garment_group"),
        ("color_group", "favorite_color"),
    ]

    for value_col, output_col in mappings:
        top_df = get_top_value(txn, "customer_id", value_col, output_col)

        if preference_features is None:
            preference_features = top_df
        else:
            preference_features = preference_features.merge(
                top_df,
                on="customer_id",
                how="left",
            )

    return preference_features


def build_price_sensitivity_features(transactions: pd.DataFrame) -> pd.DataFrame:
    transactions = transactions.copy()

    low_price_threshold = transactions["price"].quantile(0.25)
    high_price_threshold = transactions["price"].quantile(0.75)

    transactions["is_low_price"] = transactions["price"] <= low_price_threshold
    transactions["is_high_price"] = transactions["price"] >= high_price_threshold

    price_features = (
        transactions.groupby("customer_id")
        .agg(
            low_price_purchase_ratio=("is_low_price", "mean"),
            high_price_purchase_ratio=("is_high_price", "mean"),
        )
        .reset_index()
    )

    return price_features


def build_customer_features(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    behavior = build_customer_behavior_features(transactions)
    preferences = build_customer_preference_features(transactions, articles)
    price_features = build_price_sensitivity_features(transactions)

    customer_features = behavior.merge(
        preferences,
        on="customer_id",
        how="left",
    )

    customer_features = customer_features.merge(
        price_features,
        on="customer_id",
        how="left",
    )

    customer_metadata_cols = [
        "customer_id",
        "fashion_news_binary",
        "is_active",
        "club_member_status",
        "fashion_news_frequency",
        "age",
        "age_group",
    ]

    available_cols = [
        col for col in customer_metadata_cols if col in customers.columns
    ]

    customer_features = customer_features.merge(
        customers[available_cols],
        on="customer_id",
        how="left",
    )

    return customer_features