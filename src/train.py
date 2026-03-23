"""
Train classification models for dental care utilization prediction.
"""

import json
import joblib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

from features import load_and_prepare, engineer_features, encode_and_split
from evaluate import evaluate_model, plot_results

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def get_models():
    """Return a dict of models to train."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=1.0, random_state=42,
            eval_metric="logloss", verbosity=0
        ),
    }


def train_all():
    """Train all models and save the best one."""
    # Load & prepare
    print("Loading and preparing data...")
    df = load_and_prepare()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, encoders = encode_and_split(df)

    print(f"Train set: {len(X_train):,} | Test set: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Target balance: {y_train.mean():.1%} positive\n")

    models = get_models()
    results = {}
    best_auc = 0
    best_model_name = None

    for name, model in models.items():
        print(f"Training {name}...")

        # Cross-validation on train set
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
        )
        print(f"  CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Fit on full train set
        model.fit(X_train, y_train)

        # Evaluate on test set
        metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics

        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_model_name = name

        print(f"  Test AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  Test F1:      {metrics['f1']:.4f}")
        print(f"  Test Accuracy: {metrics['accuracy']:.4f}\n")

    # Save best model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model = models[best_model_name]
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    joblib.dump(encoders, MODELS_DIR / "encoders.joblib")
    joblib.dump(list(X_train.columns), MODELS_DIR / "feature_names.joblib")

    # Save results
    with open(MODELS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Best model: {best_model_name} (AUC-ROC: {best_auc:.4f})")
    print(f"Model saved to {MODELS_DIR / 'best_model.joblib'}")

    # Plot
    plot_results(models, X_test, y_test, results)

    return results


if __name__ == "__main__":
    train_all()
