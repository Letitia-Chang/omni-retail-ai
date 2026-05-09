import numpy as np
import pandas as pd


def create_positive_samples(transactions: pd.DataFrame) -> pd.DataFrame:
    positive_samples = transactions[["customer_id", "article_id"]].copy()
    positive_samples["purchased"] = 1
    positive_samples = positive_samples.drop_duplicates()

    return positive_samples


def create_negative_samples(
    positive_samples: pd.DataFrame,
    articles: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_state)

    positive_samples_full = positive_samples.copy()
    all_articles = articles["article_id"].dropna().unique()

    negative_samples = positive_samples_full.copy()

    negative_samples["article_id"] = np.random.choice(
        all_articles,
        size=len(positive_samples_full),
        replace=True,
    )

    negative_samples["purchased"] = 0

    positive_samples_full["pair_key"] = (
        positive_samples_full["customer_id"].astype(str)
        + "_"
        + positive_samples_full["article_id"].astype(str)
    )

    negative_samples["pair_key"] = (
        negative_samples["customer_id"].astype(str)
        + "_"
        + negative_samples["article_id"].astype(str)
    )

    purchased_keys = set(positive_samples_full["pair_key"])

    negative_samples = negative_samples[
        ~negative_samples["pair_key"].isin(purchased_keys)
    ].copy()

    negative_samples = negative_samples.drop(columns=["pair_key"])
    positive_samples_full = positive_samples_full.drop(columns=["pair_key"])

    negative_samples = negative_samples.drop_duplicates(
        subset=["customer_id", "article_id", "purchased"]
    )

    return negative_samples


def create_modeling_pairs(
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    positive_samples = create_positive_samples(transactions)
    negative_samples = create_negative_samples(
        positive_samples,
        articles,
        random_state=random_state,
    )

    modeling_pairs = pd.concat(
        [positive_samples, negative_samples],
        ignore_index=True,
    )

    modeling_pairs = modeling_pairs.drop_duplicates(
        subset=["customer_id", "article_id", "purchased"]
    )

    return modeling_pairs


def build_purchase_modeling_dataset(
    modeling_pairs: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
) -> pd.DataFrame:
    customer_feature_cols = [
        "customer_id",
        "total_transactions",
        "unique_products",
        "total_spend",
        "avg_price",
        "max_price",
        "days_since_last_purchase",
        "customer_lifetime_days",
        "purchase_frequency",
        "low_price_purchase_ratio",
        "high_price_purchase_ratio",
        "fashion_news_binary",
        "is_active",
        "age",
        "age_group",
        "customer_segment",
    ]

    product_feature_cols = [
        "article_id",
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
        "copy_angle",
    ]

    available_customer_cols = [
        col for col in customer_feature_cols if col in customers.columns
    ]

    available_product_cols = [
        col for col in product_feature_cols if col in articles.columns
    ]

    modeling_df = modeling_pairs.merge(
        customers[available_customer_cols],
        on="customer_id",
        how="left",
    )

    modeling_df = modeling_df.merge(
        articles[available_product_cols],
        on="article_id",
        how="left",
    )

    return modeling_df


def add_purchase_interaction_features(modeling_df: pd.DataFrame) -> pd.DataFrame:
    df = modeling_df.copy()

    df["price_vs_customer_avg"] = df["avg_selling_price"] - df["avg_price"]

    df["is_lower_than_customer_avg"] = (
        df["avg_selling_price"] < df["avg_price"]
    ).astype(int)

    df["is_higher_than_customer_avg"] = (
        df["avg_selling_price"] > df["avg_price"]
    ).astype(int)

    return df


def encode_purchase_features(
    modeling_df: pd.DataFrame,
    target_col: str = "purchased",
):
    drop_cols = ["customer_id", "article_id", target_col]

    X = modeling_df.drop(columns=drop_cols)
    y = modeling_df[target_col]

    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    X_encoded = pd.get_dummies(
        X,
        columns=categorical_cols,
        dummy_na=True,
    )

    X_encoded = X_encoded.fillna(0)

    return X_encoded, y