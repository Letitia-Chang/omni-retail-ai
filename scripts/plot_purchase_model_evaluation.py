"""Generates real evaluation charts for the purchase-intent model (ROC
curve, confusion matrix, feature importance) for reports/figures/ and the
README. Reruns the exact same deterministic pipeline as
scripts/run_purchase_intent.py (same random_state=42 throughout) purely to
get y_test/y_proba/the fitted model for plotting — it does not overwrite
the saved model artifacts or predictions CSV.

Usage: PYTHONPATH=. python scripts/plot_purchase_model_evaluation.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import RocCurveDisplay, confusion_matrix, roc_auc_score

from src.utils.paths import PROCESSED_DATA_DIR, REPORTS_DIR
from src.features.purchase_features import (
    create_modeling_pairs,
    build_purchase_modeling_dataset,
    add_purchase_interaction_features,
    encode_purchase_features,
)
from src.models.purchase_intent import (
    train_purchase_intent_model,
    get_feature_importance,
)

FIGURES_DIR = REPORTS_DIR / "figures"

TEAL = "#2f9e8f"
NAVY = "#1f3b57"


def main():
    print("Rebuilding the exact modeling dataset (random_state=42)...")
    transactions = pd.read_csv(PROCESSED_DATA_DIR / "transactions.csv")
    articles = pd.read_csv(PROCESSED_DATA_DIR / "articles_enriched.csv")
    customers = pd.read_csv(PROCESSED_DATA_DIR / "customer_segments.csv")

    modeling_pairs = create_modeling_pairs(
        transactions=transactions, articles=articles, random_state=42
    )
    modeling_df = build_purchase_modeling_dataset(
        modeling_pairs=modeling_pairs, customers=customers, articles=articles
    )
    modeling_df = add_purchase_interaction_features(modeling_df)
    X_encoded, y = encode_purchase_features(modeling_df)

    print("Training (same as run_purchase_intent.py)...")
    model, X_train, X_test, y_train, y_test = train_purchase_intent_model(
        X=X_encoded, y=y
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.3f} (should match reports/evaluation.md: 0.926)")

    # --- ROC curve ---
    fig, ax = plt.subplots(figsize=(6, 5))
    display = RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="XGBoost")
    display.line_.set_color(TEAL)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, label="Chance")
    ax.set_title("Purchase-Intent Model — ROC Curve (held-out test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "purchase_intent_roc_curve.png", dpi=150)
    plt.close(fig)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["No purchase", "Purchase"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Purchase-Intent Model — Confusion Matrix (n={cm.sum():,})")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "purchase_intent_confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- Feature importance ---
    importance = get_feature_importance(model, X_encoded.columns.tolist()).head(12)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y_pos = np.arange(len(importance))
    ax.barh(y_pos, importance["importance"][::-1], color=NAVY)
    ax.set_yticks(y_pos, importance["feature"][::-1])
    ax.set_xlabel("XGBoost feature importance")
    ax.set_title("Purchase-Intent Model — Top 12 Features")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "purchase_intent_feature_importance.png", dpi=150)
    plt.close(fig)

    print(f"Saved 3 charts to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
