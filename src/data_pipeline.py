"""
ETL Pipeline — Generate synthetic healthcare data for dental care prediction.
Simulates a realistic dataset inspired by CPAM health insurance records.
"""

import pandas as pd
import numpy as np
from pathlib import Path

SEED = 42
N_RECORDS = 50_000
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def generate_beneficiaries(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic beneficiary profiles."""
    ages = rng.integers(18, 85, size=n)
    genders = rng.choice(["M", "F"], size=n)
    income_brackets = rng.choice(
        ["low", "medium", "high"],
        size=n,
        p=[0.35, 0.45, 0.20],
    )
    regions = rng.choice(
        ["urban", "suburban", "rural"],
        size=n,
        p=[0.50, 0.30, 0.20],
    )
    has_complementary = rng.choice([0, 1], size=n, p=[0.25, 0.75])

    return pd.DataFrame({
        "beneficiary_id": np.arange(n),
        "age": ages,
        "gender": genders,
        "income_bracket": income_brackets,
        "region": regions,
        "has_complementary_insurance": has_complementary,
    })


def generate_dental_claims(beneficiaries: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Generate dental care claims with approval and utilization status."""
    n = len(beneficiaries)

    # Historical dental visits in the past 3 years
    dental_visits_3y = rng.poisson(lam=2.5, size=n)

    # Claim amount requested (euros)
    claim_amounts = rng.uniform(50, 2000, size=n).round(2)

    # Treatment types
    treatment_types = rng.choice(
        ["preventive", "restorative", "prosthetic", "orthodontic"],
        size=n,
        p=[0.30, 0.35, 0.20, 0.15],
    )

    # Distance to nearest dental provider (km)
    distances = rng.exponential(scale=8, size=n).round(1)

    # Previous claim utilization rate
    prev_utilization_rate = rng.beta(a=3, b=2, size=n).round(3)

    # Days since last dental visit
    days_since_last_visit = rng.exponential(scale=365, size=n).astype(int)

    # Number of dependents
    nb_dependents = rng.poisson(lam=1.2, size=n)

    claims = beneficiaries.copy()
    claims["dental_visits_3y"] = dental_visits_3y
    claims["claim_amount"] = claim_amounts
    claims["treatment_type"] = treatment_types
    claims["distance_to_provider_km"] = distances
    claims["prev_utilization_rate"] = prev_utilization_rate
    claims["days_since_last_visit"] = days_since_last_visit
    claims["nb_dependents"] = nb_dependents

    return claims


def generate_target(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """
    Generate realistic binary target: 1 = benefit was used, 0 = not used.
    Probability is influenced by multiple features to ensure learnable patterns.
    """
    logit = (
        -0.5
        + 0.02 * (df["age"] - 40)
        + 0.8 * df["has_complementary_insurance"]
        + 0.15 * df["dental_visits_3y"]
        - 0.0005 * df["claim_amount"]
        + 1.2 * df["prev_utilization_rate"]
        - 0.03 * df["distance_to_provider_km"]
        - 0.001 * df["days_since_last_visit"]
        + 0.1 * df["nb_dependents"]
        + 0.3 * (df["income_bracket"] == "high").astype(float)
        - 0.2 * (df["income_bracket"] == "low").astype(float)
        + 0.2 * (df["region"] == "urban").astype(float)
        + 0.3 * (df["treatment_type"] == "preventive").astype(float)
        - 0.3 * (df["treatment_type"] == "prosthetic").astype(float)
    )
    prob = 1 / (1 + np.exp(-logit))
    # Add noise
    prob = np.clip(prob + rng.normal(0, 0.05, size=len(df)), 0.02, 0.98)
    return (rng.random(size=len(df)) < prob).astype(int)


def run_pipeline() -> pd.DataFrame:
    """Execute the full ETL pipeline."""
    rng = np.random.default_rng(SEED)

    print("Generating beneficiary profiles...")
    beneficiaries = generate_beneficiaries(N_RECORDS, rng)

    print("Generating dental claims...")
    claims = generate_dental_claims(beneficiaries, rng)

    print("Generating target variable...")
    claims["utilized"] = generate_target(claims, rng)

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "dental_claims.csv"
    claims.to_csv(output_path, index=False)
    print(f"Dataset saved: {output_path} ({len(claims):,} records)")
    print(f"Utilization rate: {claims['utilized'].mean():.1%}")

    return claims


if __name__ == "__main__":
    run_pipeline()
