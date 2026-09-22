"""End-to-end check on the committed artifacts.

The dashboard loads `models/` on a cold start without retraining, so a mismatch
between the committed model and the committed feature configuration would only
surface in production. These tests reproduce the dashboard's exact scoring path.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from features import apply_encoders, engineer_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REQUIRED = ["best_model.joblib", "encoders.joblib",
            "feature_names.joblib", "feature_config.json"]

pytestmark = pytest.mark.skipif(
    not all((MODELS_DIR / name).exists() for name in REQUIRED),
    reason="artifacts not built yet; run python src/train.py",
)

RAW_ROW = {
    "age": 40, "gender": "F", "income_bracket": "medium", "region": "urban",
    "has_complementary_insurance": 1, "dental_visits_3y": 3,
    "claim_amount": 300.0, "treatment_type": "preventive",
    "distance_to_provider_km": 5.0, "prev_utilization_rate": 0.6,
    "days_since_last_visit": 300, "nb_dependents": 1,
}


@pytest.fixture(scope="module")
def artifacts():
    return (
        joblib.load(MODELS_DIR / "best_model.joblib"),
        joblib.load(MODELS_DIR / "encoders.joblib"),
        joblib.load(MODELS_DIR / "feature_names.joblib"),
        json.loads((MODELS_DIR / "feature_config.json").read_text(encoding="utf-8")),
    )


def test_dashboard_path_produces_a_probability(artifacts):
    model, encoders, feature_names, config = artifacts

    row = engineer_features(pd.DataFrame([RAW_ROW]), config)
    row = apply_encoders(row, encoders)
    row = row[feature_names]

    proba = float(model.predict_proba(row)[0, 1])
    assert 0.0 <= proba <= 1.0


def test_artifacts_agree_on_the_feature_set(artifacts):
    model, encoders, feature_names, config = artifacts

    row = apply_encoders(engineer_features(pd.DataFrame([RAW_ROW]), config), encoders)
    assert set(feature_names).issubset(row.columns), (
        "the model expects a feature the current pipeline no longer produces"
    )

    expected = getattr(model, "n_features_in_", len(feature_names))
    assert expected == len(feature_names)


def test_reported_metrics_match_the_saved_best_model():
    results = json.loads((MODELS_DIR / "results.json").read_text(encoding="utf-8"))
    best = results["best_model"]
    metrics = results["metrics"]

    assert best == max(metrics, key=lambda name: metrics[name]["auc_roc"]), (
        "results.json names a best model that is not the top scorer"
    )
