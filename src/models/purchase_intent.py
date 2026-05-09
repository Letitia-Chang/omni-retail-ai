import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier


def train_purchase_intent_model(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test


def evaluate_purchase_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "classification_report": classification_report(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics


def get_feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_purchase_model_artifacts(
    model,
    feature_columns: list[str],
    model_dir,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "xgboost_purchase_intent_model.pkl")
    joblib.dump(feature_columns, model_dir / "purchase_model_features.pkl")


def generate_purchase_predictions(
    model,
    X_encoded: pd.DataFrame,
    modeling_df: pd.DataFrame,
) -> pd.DataFrame:
    predictions = modeling_df[["customer_id", "article_id", "purchased"]].copy()
    predictions["purchase_probability"] = model.predict_proba(X_encoded)[:, 1]

    predictions = predictions.sort_values(
        "purchase_probability",
        ascending=False,
    ).reset_index(drop=True)

    return predictions