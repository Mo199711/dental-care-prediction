"""
Streamlit dashboard for dental care utilization prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODELS_DIR = Path("models")


@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "best_model.joblib")
    encoders = joblib.load(MODELS_DIR / "encoders.joblib")
    feature_names = joblib.load(MODELS_DIR / "feature_names.joblib")
    return model, encoders, feature_names


def main():
    st.set_page_config(page_title="Dental Care Prediction", page_icon="🦷", layout="wide")

    st.title("Dental Care Utilization Prediction")
    st.markdown("Predict whether a beneficiary will use their approved dental care benefit.")

    try:
        model, encoders, feature_names = load_model()
    except FileNotFoundError:
        st.error("Model not found. Run `python src/train.py` first.")
        return

    # --- Sidebar: Input form ---
    st.sidebar.header("Beneficiary Profile")

    age = st.sidebar.slider("Age", 18, 85, 40)
    gender = st.sidebar.selectbox("Gender", ["M", "F"])
    income_bracket = st.sidebar.selectbox("Income Bracket", ["low", "medium", "high"])
    region = st.sidebar.selectbox("Region", ["urban", "suburban", "rural"])
    has_complementary = st.sidebar.selectbox("Complementary Insurance", [1, 0], format_func=lambda x: "Yes" if x else "No")
    dental_visits_3y = st.sidebar.slider("Dental Visits (last 3 years)", 0, 15, 3)
    claim_amount = st.sidebar.slider("Claim Amount (EUR)", 50, 2000, 300)
    treatment_type = st.sidebar.selectbox("Treatment Type", ["preventive", "restorative", "prosthetic", "orthodontic"])
    distance = st.sidebar.slider("Distance to Provider (km)", 0.0, 50.0, 5.0)
    prev_utilization = st.sidebar.slider("Previous Utilization Rate", 0.0, 1.0, 0.6)
    days_since_visit = st.sidebar.slider("Days Since Last Visit", 0, 2000, 300)
    nb_dependents = st.sidebar.slider("Number of Dependents", 0, 8, 1)

    # --- Build feature vector ---
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "income_bracket": income_bracket,
        "region": region,
        "has_complementary_insurance": has_complementary,
        "dental_visits_3y": dental_visits_3y,
        "claim_amount": claim_amount,
        "treatment_type": treatment_type,
        "distance_to_provider_km": distance,
        "prev_utilization_rate": prev_utilization,
        "days_since_last_visit": days_since_visit,
        "nb_dependents": nb_dependents,
    }])

    # Engineer features (same as training)
    input_data["age_group"] = pd.cut(
        input_data["age"], bins=[0, 25, 35, 50, 65, 100],
        labels=["18-25", "26-35", "36-50", "51-65", "65+"],
    )
    input_data["high_claim"] = (input_data["claim_amount"] > 800).astype(int)
    input_data["recent_visitor"] = (input_data["days_since_last_visit"] < 365).astype(int)
    input_data["access_score"] = (
        input_data["has_complementary_insurance"] * 2
        - np.log1p(input_data["distance_to_provider_km"])
    ).round(3)
    input_data["engagement_score"] = (
        input_data["dental_visits_3y"] * 0.4
        + input_data["prev_utilization_rate"] * 0.6
    ).round(3)

    # Encode categoricals
    for col, le in encoders.items():
        if col in input_data.columns:
            input_data[col] = le.transform(input_data[col].astype(str))

    # Ensure same feature order
    input_data = input_data[feature_names]

    # --- Predict ---
    if st.sidebar.button("Predict", type="primary"):
        proba = model.predict_proba(input_data)[0, 1]
        prediction = "WILL USE" if proba >= 0.5 else "WILL NOT USE"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction", prediction)
        with col2:
            st.metric("Probability", f"{proba:.1%}")
        with col3:
            risk = "Low" if proba >= 0.7 else ("Medium" if proba >= 0.4 else "High")
            st.metric("Non-utilization Risk", risk)

        st.progress(proba)

        # Feature importance context
        if hasattr(model, "feature_importances_"):
            st.subheader("Key Factors")
            importances = pd.Series(
                model.feature_importances_, index=feature_names
            ).sort_values(ascending=False).head(8)
            st.bar_chart(importances)

    # --- Dataset overview ---
    st.markdown("---")
    st.subheader("Dataset Overview")
    try:
        df = pd.read_csv("data/dental_claims.csv")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{len(df):,}")
        col2.metric("Utilization Rate", f"{df['utilized'].mean():.1%}")
        col3.metric("Avg Claim Amount", f"{df['claim_amount'].mean():.0f} EUR")
        col4.metric("Features", str(len(feature_names)))
    except FileNotFoundError:
        st.info("Run the data pipeline first to see dataset stats.")


if __name__ == "__main__":
    main()
