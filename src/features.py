"""
Feature engineering for dental care utilization prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_prepare(path: str = None) -> pd.DataFrame:
    if path is None:
        path = str(Path(__file__).resolve().parent.parent / "data" / "dental_claims.csv")
    """Load dataset and perform basic cleaning."""
    df = pd.read_csv(path)
    df = df.drop(columns=["beneficiary_id"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features from existing columns."""
    df = df.copy()

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["18-25", "26-35", "36-50", "51-65", "65+"],
    )

    # High claim flag
    df["high_claim"] = (df["claim_amount"] > df["claim_amount"].quantile(0.75)).astype(int)

    # Recency: recent visitor vs not
    df["recent_visitor"] = (df["days_since_last_visit"] < 365).astype(int)

    # Access score (combination of distance and complementary insurance)
    df["access_score"] = (
        df["has_complementary_insurance"] * 2
        - np.log1p(df["distance_to_provider_km"])
    ).round(3)

    # Engagement score
    df["engagement_score"] = (
        df["dental_visits_3y"] * 0.4
        + df["prev_utilization_rate"] * 0.6
    ).round(3)

    return df


def encode_and_split(
    df: pd.DataFrame,
    target: str = "utilized",
    test_size: float = 0.2,
    seed: int = 42,
):
    """Encode categorical features and split into train/test sets."""
    df = df.copy()

    # Label encode categoricals
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    return X_train, X_test, y_train, y_test, encoders
