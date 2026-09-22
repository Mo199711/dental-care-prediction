"""
Model evaluation utilities: metrics and the comparison figure.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def evaluate_model(model, X_test, y_test) -> dict:
    """Compute the headline classification metrics on the held-out set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }


def feature_importances(model, X) -> "tuple[np.ndarray, str] | None":
    """Return comparable per-feature importances for any supported model.

    A Pipeline is unwrapped to its final estimator, with `X` pushed through the
    preceding steps so the scale matches the fitted coefficients.

    Tree ensembles expose `feature_importances_`. Linear models do not, so we
    fall back to their coefficients. Raw coefficients are not comparable across
    features on different scales, so each is multiplied by the standard
    deviation of its column: the product is the change in log-odds per standard
    deviation. Behind a StandardScaler that factor is 1 and the result is simply
    the coefficient. Without this branch the importance panel renders blank
    whenever a linear model wins the benchmark.
    """
    estimator = model
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
        X = model[:-1].transform(X)

    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_), "Gain"
    if hasattr(estimator, "coef_"):
        spread = np.asarray(X).std(axis=0)
        scaled = np.abs(np.ravel(estimator.coef_)) * spread
        return scaled, "|Coefficient| (log-odds per SD)"
    return None


def plot_results(models: dict, X_test, y_test, results: dict) -> Path:
    """Render the three-panel evaluation figure used in the README."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, linewidth=2,
                label=f"{name} (AUC={results[name]['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    x = np.arange(len(metric_keys))
    width = 0.25
    for i, (name, metrics) in enumerate(results.items()):
        ax.bar(x + i * width, [metrics[k] for k in metric_keys], width, label=name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"])
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Model comparison (test set)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    best_name = max(results, key=lambda k: results[k]["auc_roc"])
    computed = feature_importances(models[best_name], X_test)
    if computed is None:
        ax.axis("off")
        ax.set_title(f"No importances available ({best_name})")
    else:
        values, unit = computed
        order = np.argsort(values)[-15:]
        ax.barh([X_test.columns[i] for i in order], values[order], color="steelblue")
        ax.set_xlabel(unit)
        ax.set_title(f"Feature importance ({best_name})")
        ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    output = FIGURES_DIR / "model_evaluation.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output}")
    return output
