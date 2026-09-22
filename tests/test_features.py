"""Tests for the feature layer shared by training and the dashboard."""

import numpy as np
import pandas as pd

from features import (
    TARGET,
    apply_encoders,
    build_training_sets,
    engineer_features,
    fit_encoders,
    fit_feature_config,
)


def test_single_row_matches_batch(sample_frame):
    """The dashboard scores one row at a time. It must get the same features.

    This is the regression test for the train/serve skew that shipped earlier:
    the app hard-coded `high_claim = claim_amount > 800` while training used a
    quantile of the training split. Scoring a row on its own must be identical
    to scoring it inside the full frame.
    """
    config = fit_feature_config(sample_frame)
    batch = engineer_features(sample_frame, config)

    derived = ["age_group", "high_claim", "recent_visitor",
               "access_score", "engagement_score"]

    for position in (0, 7, len(sample_frame) - 1):
        single = engineer_features(sample_frame.iloc[[position]], config)
        for column in derived:
            assert str(single[column].iloc[0]) == str(batch[column].iloc[position]), (
                f"{column} differs between single-row and batch scoring"
            )


def test_config_is_fitted_on_training_data_only(sample_frame):
    """The high_claim cut-off must not be able to see the test split."""
    train = sample_frame.iloc[:150]
    full = sample_frame

    assert fit_feature_config(train)["high_claim_threshold"] == (
        float(train["claim_amount"].quantile(0.75))
    )
    assert fit_feature_config(train) != fit_feature_config(full)


def test_engineer_features_is_pure(sample_frame):
    """Calling the function twice must not mutate the caller's frame."""
    config = fit_feature_config(sample_frame)
    before = sample_frame.copy()
    engineer_features(sample_frame, config)
    pd.testing.assert_frame_equal(sample_frame, before)


def test_unseen_category_does_not_crash(sample_frame):
    """A category absent from training maps to -1 rather than raising."""
    encoders = fit_encoders(sample_frame.drop(columns=[TARGET]))
    unseen = sample_frame.iloc[[0]].copy()
    unseen["region"] = "offshore"

    encoded = apply_encoders(unseen, encoders)
    assert encoded["region"].iloc[0] == -1


def test_train_test_split_is_disjoint_and_ordered(sample_frame):
    X_train, X_test, y_train, y_test, encoders, config = build_training_sets(
        sample_frame, test_size=0.25, seed=42
    )

    assert len(X_train) + len(X_test) == len(sample_frame)
    assert list(X_train.columns) == list(X_test.columns), (
        "column order must match, the model is indexed positionally"
    )
    assert TARGET not in X_train.columns
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert "high_claim_threshold" in config


def test_no_missing_values_after_engineering(sample_frame):
    config = fit_feature_config(sample_frame)
    engineered = engineer_features(sample_frame, config)
    encoders = fit_encoders(engineered.drop(columns=[TARGET]))
    encoded = apply_encoders(engineered, encoders)

    assert not encoded.isna().any().any()
    assert np.isfinite(encoded.select_dtypes("number").to_numpy()).all()
