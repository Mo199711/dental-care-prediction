"""
Train and compare classification models for dental care utilization.

Run from the project root:

    python src/train.py

Writes the winning model, the fitted encoders, the feature configuration and
the metrics table to `models/`, plus the evaluation figure to `figures/`.
"""

import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from evaluate import evaluate_model, feature_importances, plot_results
from features import build_training_sets, load_and_prepare, save_feature_config

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SEED = 42


def get_models() -> dict:
    """Models compared in the benchmark.

    Hyperparameters are sensible defaults rather than the output of a search:
    the data-generating process is known and close to linear, so the ranking
    below is driven by model family, not by fine tuning.

    The linear model is wrapped in a scaler. Claim amounts run to 2000 while
    utilization rates sit in [0, 1], and without standardisation lbfgs hits its
    iteration cap without converging. Scaling also makes the coefficients
    directly comparable to one another.
    """
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=SEED
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=SEED, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=SEED, eval_metric="logloss", verbosity=0
        ),
    }


def train_all() -> dict:
    """Train every model, persist the best one and the artifacts around it."""
    print("Loading and preparing data...")
    df = load_and_prepare()
    X_train, X_test, y_train, y_test, encoders, config = build_training_sets(df)

    print(f"Train set: {len(X_train):,} | Test set: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Target balance: {y_train.mean():.1%} positive")
    print(f"high_claim threshold (train only): {config['high_claim_threshold']:.2f} EUR\n")

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
        )
        print(f"  CV AUC-ROC:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["cv_auc_roc"] = round(float(cv_scores.mean()), 4)
        results[name] = metrics

        print(f"  Test AUC-ROC:  {metrics['auc_roc']:.4f}")
        print(f"  Test F1:       {metrics['f1']:.4f}")
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}\n")

    best_model_name = max(results, key=lambda name: results[name]["auc_roc"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[best_model_name], MODELS_DIR / "best_model.joblib")
    joblib.dump(encoders, MODELS_DIR / "encoders.joblib")
    joblib.dump(list(X_train.columns), MODELS_DIR / "feature_names.joblib")
    save_feature_config(config)

    payload = {"best_model": best_model_name, "metrics": results}
    (MODELS_DIR / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    # Importances are computed once, here, on the test set. The dashboard
    # scores one beneficiary at a time and cannot derive them from a single
    # row, so it reads this file instead.
    computed = feature_importances(models[best_model_name], X_test)
    if computed is not None:
        values, unit = computed
        (MODELS_DIR / "feature_importances.json").write_text(
            json.dumps(
                {
                    "model": best_model_name,
                    "unit": unit,
                    "values": {
                        name: round(float(value), 6)
                        for name, value in zip(X_test.columns, values)
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Best model: {best_model_name} (AUC-ROC: {results[best_model_name]['auc_roc']:.4f})")
    print(f"Artifacts written to {MODELS_DIR}")

    plot_results(models, X_test, y_test, results)
    return payload


if __name__ == "__main__":
    train_all()
