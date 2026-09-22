"""Make `src/` importable so the tests mirror how the scripts import."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """A small, fully specified frame covering every categorical level."""
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "age": rng.integers(18, 85, n),
        "gender": rng.choice(["M", "F"], n),
        "income_bracket": rng.choice(["low", "medium", "high"], n),
        "region": rng.choice(["urban", "suburban", "rural"], n),
        "has_complementary_insurance": rng.choice([0, 1], n),
        "dental_visits_3y": rng.integers(0, 12, n),
        "claim_amount": rng.uniform(50, 2000, n).round(2),
        "treatment_type": rng.choice(
            ["preventive", "restorative", "prosthetic", "orthodontic"], n),
        "distance_to_provider_km": rng.exponential(8, n).round(1),
        "prev_utilization_rate": rng.beta(3, 2, n).round(3),
        "days_since_last_visit": rng.exponential(365, n).astype(int),
        "nb_dependents": rng.integers(0, 6, n),
        "utilized": rng.choice([0, 1], n),
    })
