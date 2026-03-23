"""
Model evaluation utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
    }


def plot_results(models: dict, X_test, y_test, results: dict):
    """Generate evaluation plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. ROC Curves
    ax = axes[0]
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = results[name]["auc_roc"]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. Model Comparison
    ax = axes[1]
    metric_names = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    x = np.arange(len(metric_names))
    width = 0.25
    for i, (name, metrics) in enumerate(results.items()):
        values = [metrics[m] for m in metric_names]
        ax.bar(x + i * width, values, width, label=name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"])
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model Comparison")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # 3. Feature Importance (best model — XGBoost or Random Forest)
    ax = axes[2]
    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    best_model = models[best_name]
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        feature_names = X_test.columns
        idx = np.argsort(importances)[-15:]
        ax.barh(
            [feature_names[i] for i in idx],
            importances[idx],
            color="steelblue",
        )
        ax.set_title(f"Feature Importance ({best_name})")
        ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {FIGURES_DIR / 'model_evaluation.png'}")
