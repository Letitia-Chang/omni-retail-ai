import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR, SEGMENTATION_MODEL_DIR
from src.features.customer_features import build_customer_features
from src.models.segmentation import (
    train_segmentation_model,
    assign_segment_names,
    create_cluster_summary,
    save_segmentation_artifacts,
)


def main():
    print("Loading processed data...")

    customers = pd.read_csv(PROCESSED_DATA_DIR / "customers.csv")
    articles = pd.read_csv(PROCESSED_DATA_DIR / "articles.csv")
    transactions = pd.read_csv(PROCESSED_DATA_DIR / "transactions.csv")

    print("Building customer features...")

    customer_features = build_customer_features(
        customers=customers,
        transactions=transactions,
        articles=articles,
    )

    print("Training segmentation model...")

    customer_segments, model, scaler, clustering_features = train_segmentation_model(
        customer_features=customer_features,
        n_clusters=5,
    )

    customer_segments = assign_segment_names(customer_segments)
    cluster_summary = create_cluster_summary(customer_segments)

    print("Saving outputs...")

    customer_segments.to_csv(
        PROCESSED_DATA_DIR / "customer_segments.csv",
        index=False,
    )

    cluster_summary.to_csv(
        PROCESSED_DATA_DIR / "customer_segment_summary.csv",
        index=False,
    )

    save_segmentation_artifacts(
        model=model,
        scaler=scaler,
        clustering_features=clustering_features,
        model_dir=SEGMENTATION_MODEL_DIR,
    )

    print("Saved customer_segments.csv")
    print("Saved customer_segment_summary.csv")
    print(customer_segments.head())


if __name__ == "__main__":
    main()