"""Tests for the synthetic data pipeline and the evaluation helpers."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_pipeline import generate_beneficiaries, generate_dental_claims, generate_target
from evaluate import evaluate_model, feature_importances

EXPECTED_COLUMNS = {
    "beneficiary_id", "age", "gender", "income_bracket", "region",
    "has_complementary_insurance", "dental_visits_3y", "claim_amount",
    "treatment_type", "distance_to_provider_km", "prev_utilization_rate",
    "days_since_last_visit", "nb_dependents",
}


def _claims(seed: int = 42, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return generate_dental_claims(generate_beneficiaries(n, rng), rng)


def test_pipeline_is_reproducible():
    """The same seed must produce byte-identical data, the README claims it."""
    pd.testing.assert_frame_equal(_claims(), _claims())


def test_generated_schema_and_ranges():
    claims = _claims()
    assert EXPECTED_COLUMNS.issubset(claims.columns)
    assert claims["age"].between(18, 85).all()
    assert claims["claim_amount"].between(50, 2000).all()
    assert claims["prev_utilization_rate"].between(0, 1).all()
    assert not claims.isna().any().any()


def test_target_is_binary_and_not_degenerate():
    rng = np.random.default_rng(42)
    claims = _claims()
    target = generate_target(claims, rng)

    assert set(np.unique(target)).issubset({0, 1})
    assert 0.2 < target.mean() < 0.9, "a degenerate target would make AUC meaningless"


def _tiny_fit(model):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=list("abcd"))
    y = (X["a"] + rng.normal(scale=0.5, size=120) > 0).astype(int)
    return model.fit(X, y), X, y


def test_metrics_are_bounded():
    model, X, y = _tiny_fit(LogisticRegression(max_iter=500))
    metrics = evaluate_model(model, X, y)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "auc_roc"}
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_importances_available_for_linear_models():
    """Regression test: the figure's third panel used to render empty.

    `plot_results` asked the winning model for `feature_importances_`. Logistic
    regression has no such attribute, so when it won the benchmark the panel was
    blank. Both families must now return usable values.
    """
    linear, X, _ = _tiny_fit(LogisticRegression(max_iter=500))
    forest, _, _ = _tiny_fit(RandomForestClassifier(n_estimators=10, random_state=0))
    scaled, _, _ = _tiny_fit(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    )

    for model in (linear, forest, scaled):
        computed = feature_importances(model, X)
        assert computed is not None
        values, unit = computed
        assert len(values) == X.shape[1]
        assert np.isfinite(values).all()
        assert isinstance(unit, str) and unit
