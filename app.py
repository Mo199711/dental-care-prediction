"""
Streamlit dashboard for dental care utilization prediction.

The dashboard never re-implements feature engineering. It imports the same
`engineer_features` used at training time and replays the persisted feature
configuration, so a beneficiary scored here is transformed exactly as a
training row was.
"""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "dental_claims.csv"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features import (  # noqa: E402
    FEATURE_CONFIG_PATH,
    apply_encoders,
    engineer_features,
    load_feature_config,
)

ARTIFACTS = [
    MODELS_DIR / "best_model.joblib",
    MODELS_DIR / "encoders.joblib",
    MODELS_DIR / "feature_names.joblib",
    FEATURE_CONFIG_PATH,
]


def ensure_artifacts_exist() -> None:
    """Rebuild the artifacts if a fresh deploy is missing them.

    The dataset is fully synthetic and seeded, so regenerating it reproduces
    the committed artifacts rather than silently producing a different model.
    """
    if all(path.exists() for path in ARTIFACTS):
        return

    with st.spinner("First run: generating data and training the model (1-2 min)..."):
        from data_pipeline import run_pipeline
        from train import train_all

        if not DATA_PATH.exists():
            run_pipeline()
        train_all()


@st.cache_resource
def load_artifacts():
    ensure_artifacts_exist()
    return (
        joblib.load(MODELS_DIR / "best_model.joblib"),
        joblib.load(MODELS_DIR / "encoders.joblib"),
        joblib.load(MODELS_DIR / "feature_names.joblib"),
        load_feature_config(),
    )


@st.cache_data
def load_importances() -> dict | None:
    path = MODELS_DIR / "feature_importances.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def collect_inputs() -> pd.DataFrame:
    """Read the sidebar form into a single raw row."""
    st.sidebar.header("Beneficiary profile")
    return pd.DataFrame([{
        "age": st.sidebar.slider("Age", 18, 85, 40),
        "gender": st.sidebar.selectbox("Gender", ["M", "F"]),
        "income_bracket": st.sidebar.selectbox(
            "Income bracket", ["low", "medium", "high"]),
        "region": st.sidebar.selectbox(
            "Region", ["urban", "suburban", "rural"]),
        "has_complementary_insurance": st.sidebar.selectbox(
            "Complementary insurance", [1, 0],
            format_func=lambda x: "Yes" if x else "No"),
        "dental_visits_3y": st.sidebar.slider(
            "Dental visits (last 3 years)", 0, 15, 3),
        "claim_amount": float(st.sidebar.slider(
            "Claim amount (EUR)", 50, 2000, 300)),
        "treatment_type": st.sidebar.selectbox(
            "Treatment type",
            ["preventive", "restorative", "prosthetic", "orthodontic"]),
        "distance_to_provider_km": st.sidebar.slider(
            "Distance to provider (km)", 0.0, 50.0, 5.0),
        "prev_utilization_rate": st.sidebar.slider(
            "Previous utilization rate", 0.0, 1.0, 0.6),
        "days_since_last_visit": st.sidebar.slider(
            "Days since last visit", 0, 2000, 300),
        "nb_dependents": st.sidebar.slider("Number of dependents", 0, 8, 1),
    }])


def main() -> None:
    st.set_page_config(
        page_title="Dental Care Prediction", page_icon="🦷", layout="wide"
    )
    st.title("Dental Care Utilization Prediction")
    st.markdown(
        "Predict whether a beneficiary will use their approved dental benefit. "
        "Trained on a synthetic, seeded dataset of 50,000 claims."
    )

    try:
        model, encoders, feature_names, config = load_artifacts()
    except FileNotFoundError:
        st.error("Artifacts not found. Run `python src/train.py` first.")
        return

    raw = collect_inputs()
    features = engineer_features(raw, config)
    features = apply_encoders(features, encoders)
    features = features[feature_names]

    if st.sidebar.button("Predict", type="primary"):
        proba = float(model.predict_proba(features)[0, 1])

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", "WILL USE" if proba >= 0.5 else "WILL NOT USE")
        col2.metric("Probability", f"{proba:.1%}")
        col3.metric(
            "Non-utilization risk",
            "Low" if proba >= 0.7 else ("Medium" if proba >= 0.4 else "High"),
        )
        st.progress(proba)

        importances = load_importances()
        if importances:
            st.subheader(f"What drives the model ({importances['unit']})")
            st.caption(
                f"Global importances from the {importances['model']} benchmark, "
                "not a per-prediction attribution."
            )
            st.bar_chart(
                pd.Series(importances["values"])
                .sort_values(ascending=False)
                .head(8)
            )

    st.markdown("---")
    st.subheader("Dataset overview")
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", f"{len(df):,}")
        c2.metric("Utilization rate", f"{df['utilized'].mean():.1%}")
        c3.metric("Avg claim", f"{df['claim_amount'].mean():.0f} EUR")
        c4.metric("high_claim cut-off", f"{config['high_claim_threshold']:.0f} EUR")
    else:
        st.info("Run the data pipeline to see dataset statistics.")


if __name__ == "__main__":
    main()
