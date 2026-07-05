import pandas as pd

from src.utils.paths import PROCESSED_DATA_DIR, PURCHASE_MODEL_DIR
from src.features.purchase_features import (
    create_modeling_pairs,
    build_purchase_modeling_dataset,
    add_purchase_interaction_features,
    encode_purchase_features,
)
from src.models.purchase_intent import (
    train_purchase_intent_model,
    evaluate_purchase_model,
    get_feature_importance,
    save_purchase_model_artifacts,
    generate_purchase_predictions,
)


def main():
    print("Loading data...")

    transactions = pd.read_csv(PROCESSED_DATA_DIR / "transactions.csv")
    articles = pd.read_csv(PROCESSED_DATA_DIR / "articles_enriched.csv")
    customers = pd.read_csv(PROCESSED_DATA_DIR / "customer_segments.csv")

    print("Creating positive and negative samples...")

    modeling_pairs = create_modeling_pairs(
        transactions=transactions,
        articles=articles,
        random_state=42,
    )

    print(modeling_pairs["purchased"].value_counts())

    print("Building modeling dataset...")

    modeling_df = build_purchase_modeling_dataset(
        modeling_pairs=modeling_pairs,
        customers=customers,
        articles=articles,
    )

    modeling_df = add_purchase_interaction_features(modeling_df)

    print("Encoding features...")

    X_encoded, y = encode_purchase_features(modeling_df)

    print("Training purchase intent model...")

    model, X_train, X_test, y_train, y_test = train_purchase_intent_model(
        X=X_encoded,
        y=y,
    )

    print("Evaluating model...")

    metrics = evaluate_purchase_model(model, X_test, y_test)

    print(metrics["classification_report"])
    print("ROC-AUC:", metrics["roc_auc"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])

    print("Saving model artifacts...")

    feature_columns = X_encoded.columns.tolist()

    save_purchase_model_artifacts(
        model=model,
        feature_columns=feature_columns,
        model_dir=PURCHASE_MODEL_DIR,
    )

    feature_importance = get_feature_importance(
        model=model,
        feature_columns=feature_columns,
    )

    feature_importance.to_csv(
        PROCESSED_DATA_DIR / "purchase_model_feature_importance.csv",
        index=False,
    )

    print("Generating purchase predictions...")

    predictions = generate_purchase_predictions(
        model=model,
        X_encoded=X_encoded,
        modeling_df=modeling_df,
    )

    predictions.to_csv(
        PROCESSED_DATA_DIR / "purchase_intent_predictions.csv",
        index=False,
    )

    print("Saved purchase_intent_predictions.csv")
    print("Saved purchase_model_feature_importance.csv")
    print(predictions.head())


if __name__ == "__main__":
    main()