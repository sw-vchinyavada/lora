"""Shared pytest fixtures for credit scoring tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def reset_calibration_cache():
    """Prevent calibration cache leaking between tests."""
    import src.scoring.inference as inf

    inf._CALIBRATION_CACHE = None
    yield
    inf._CALIBRATION_CACHE = None


@pytest.fixture
def sample_calibration() -> dict:
    return {"p_min": 0.05, "p_max": 0.75, "proba_std_test": 0.2, "unique_scores_test": 100}


@pytest.fixture
def synthetic_applicants() -> pd.DataFrame:
    """Small deterministic dataset slice for variation tests."""
    from src.data.zimbabwe_synthetic import generate_zimbabwe_alternative_data

    return generate_zimbabwe_alternative_data(n_samples=200, random_state=42)


@pytest.fixture
def mock_preprocessor(synthetic_applicants: pd.DataFrame):
    """Preprocessor stub using columns that differ between strong/weak applicants."""
    prep = MagicMock()
    key_cols = [
        "mm_txn_per_month", "utility_payment_rate", "util_overdue_count",
        "digital_engagement", "airtime_consistency_score", "account_age_months",
        "mm_avg_txn_usd", "mm_bill_payment_ratio", "util_electricity_consistency",
        "is_urban", "youth", "msme", "age", "mm_balance_volatility",
        "social_media_usage", "has_smartphone", "mm_tenure_months",
        "util_water_consistency", "consecutive_on_time", "payment_trend",
    ]
    prep.feature_columns_ = [c for c in key_cols if c in synthetic_applicants.columns]
    prep.numerical_features = prep.feature_columns_
    prep.categorical_features = []

    def transform(df):
        cols = [c for c in prep.feature_columns_ if c in df.columns]
        return df[cols].fillna(0).astype(float).values

    prep.transform = transform
    return prep


class FeatureSensitiveMockModel:
    """Mock LoRA model: default probability varies with input feature fingerprint."""

    def __init__(self, num_features: int):
        self.num_features = num_features
        self._weights = np.linspace(0.3, 1.7, num_features)

    def predict_proba_numpy(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        fingerprint = np.sum(X * self._weights, axis=1)
        # Higher fingerprint → lower default probability (better credit profile)
        raw = 0.64 / (1.0 + np.exp(-0.008 * (fingerprint - 400.0)))
        scaled = 0.72 - raw
        return scaled.astype(float)


@pytest.fixture
def mock_scoring_artifacts(mock_preprocessor, synthetic_applicants):
    n = len(mock_preprocessor.feature_columns_)
    model = FeatureSensitiveMockModel(num_features=n)
    return model, mock_preprocessor, synthetic_applicants
