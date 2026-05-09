import joblib
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DEFAULT_CLUSTERING_FEATURES = [
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
]


DEFAULT_SEGMENT_NAME_MAP = {
    0: "High-Value One-Time Buyers",
    1: "Inactive Budget Shoppers",
    2: "Engaged Budget Shoppers",
    3: "Regular Shoppers",
    4: "Loyal High-Value Customers",
}


def train_segmentation_model(
    customer_features: pd.DataFrame,
    clustering_features: list[str] | None = None,
    n_clusters: int = 5,
    random_state: int = 42,
):
    if clustering_features is None:
        clustering_features = DEFAULT_CLUSTERING_FEATURES

    cluster_df = customer_features[["customer_id"] + clustering_features].copy()
    cluster_df[clustering_features] = cluster_df[clustering_features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df[clustering_features])

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )

    labels = model.fit_predict(X_scaled)

    result = customer_features.copy()
    result["cluster"] = labels

    return result, model, scaler, clustering_features


def assign_segment_names(
    customer_features: pd.DataFrame,
    segment_name_map: dict[int, str] | None = None,
) -> pd.DataFrame:
    if segment_name_map is None:
        segment_name_map = DEFAULT_SEGMENT_NAME_MAP

    result = customer_features.copy()
    result["customer_segment"] = result["cluster"].map(segment_name_map)

    return result


def create_cluster_summary(customer_features: pd.DataFrame) -> pd.DataFrame:
    return (
        customer_features.groupby("cluster")
        .agg(
            customer_count=("customer_id", "count"),
            avg_total_transactions=("total_transactions", "mean"),
            avg_unique_products=("unique_products", "mean"),
            avg_total_spend=("total_spend", "mean"),
            avg_price=("avg_price", "mean"),
            avg_days_since_last_purchase=("days_since_last_purchase", "mean"),
            avg_purchase_frequency=("purchase_frequency", "mean"),
            avg_low_price_ratio=("low_price_purchase_ratio", "mean"),
            avg_high_price_ratio=("high_price_purchase_ratio", "mean"),
            avg_age=("age", "mean"),
            active_rate=("is_active", "mean"),
            fashion_news_rate=("fashion_news_binary", "mean"),
        )
        .reset_index()
    )


def save_segmentation_artifacts(
    model,
    scaler,
    clustering_features: list[str],
    model_dir,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "kmeans_customer_segments.pkl")
    joblib.dump(scaler, model_dir / "customer_segment_scaler.pkl")
    joblib.dump(clustering_features, model_dir / "clustering_features.pkl")