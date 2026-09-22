"""
Feature engineering for dental care utilization prediction.

This module is the single source of truth for feature construction. Both the
training pipeline and the Streamlit dashboard import `engineer_features` from
here, so a beneficiary scored in the app goes through exactly the same
transformations as a row seen during training.

Every data-dependent constant (currently the `high_claim` cut-off) is fitted on
the training split only and persisted to `models/feature_config.json`, so it can
be replayed at inference time instead of being recomputed or hard-coded.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET = "utilized"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dental_claims.csv"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "models" / "feature_config.json"

AGE_BINS = [0, 25, 35, 50, 65, 100]
AGE_LABELS = ["18-25", "26-35", "36-50", "51-65", "65+"]
HIGH_CLAIM_QUANTILE = 0.75
RECENT_VISIT_DAYS = 365


def load_and_prepare(path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset and drop the identifier column."""
    df = pd.read_csv(path or DATA_PATH)
    return df.drop(columns=["beneficiary_id"], errors="ignore")


def fit_feature_config(train_df: pd.DataFrame) -> dict:
    """Derive data-dependent constants from the TRAINING split only."""
    return {
        "high_claim_threshold": float(
            train_df["claim_amount"].quantile(HIGH_CLAIM_QUANTILE)
        ),
    }


def save_feature_config(config: dict, path: Path = FEATURE_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_feature_config(path: Path = FEATURE_CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Build derived features. Pure function of `df` and `config`.

    Because no statistic is computed from `df` itself, calling this on a single
    row in the dashboard yields the same values as calling it on the training
    set. That property is enforced by tests/test_features.py.
    """
    df = df.copy()

    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS)
    df["high_claim"] = (
        df["claim_amount"] > config["high_claim_threshold"]
    ).astype(int)
    df["recent_visitor"] = (
        df["days_since_last_visit"] < RECENT_VISIT_DAYS
    ).astype(int)
    df["access_score"] = (
        df["has_complementary_insurance"] * 2
        - np.log1p(df["distance_to_provider_km"])
    ).round(3)
    df["engagement_score"] = (
        df["dental_visits_3y"] * 0.4
        + df["prev_utilization_rate"] * 0.6
    ).round(3)

    return df


def _categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def fit_encoders(train_df: pd.DataFrame) -> dict:
    """Fit one LabelEncoder per categorical column on the training split."""
    encoders = {}
    for col in _categorical_columns(train_df):
        encoder = LabelEncoder()
        encoder.fit(train_df[col].astype(str))
        encoders[col] = encoder
    return encoders


def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply fitted encoders. Unseen categories map to -1 instead of raising."""
    df = df.copy()
    for col, encoder in encoders.items():
        if col not in df.columns:
            continue
        mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
        df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
    return df


def build_training_sets(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Split first, then fit every transformation on the training split.

    Splitting before feature engineering is what keeps the `high_claim`
    threshold and the label encoders free of test-set information.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[TARGET],
    )

    config = fit_feature_config(train_df)
    train_df = engineer_features(train_df, config)
    test_df = engineer_features(test_df, config)

    encoders = fit_encoders(train_df.drop(columns=[TARGET]))
    train_df = apply_encoders(train_df, encoders)
    test_df = apply_encoders(test_df, encoders)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET]
    X_test = test_df.drop(columns=[TARGET])[X_train.columns]
    y_test = test_df[TARGET]

    return X_train, X_test, y_train, y_test, encoders, config
