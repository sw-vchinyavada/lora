"""
Unit tests ensuring credit scores vary meaningfully across applicants.

Guards against the common demo failure mode where every applicant receives
the same score because calibration, feature derivation, or model inference
collapses inputs to a single probability.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.scoring.inference import (
    _derive_applicant_features,
    applicant_dict_to_dataframe,
    load_score_calibration,
    probability_to_score,
    risk_band,
    score_applicant_row,
    score_features,
)


# ─── probability_to_score ───────────────────────────────────────────────────


class TestProbabilityToScore:
    def test_monotonic_decrease_with_rising_default_probability(self, sample_calibration):
        probs = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
        scores = [probability_to_score(p, sample_calibration) for p in probs]
        assert scores == sorted(scores, reverse=True)
        assert all(300 <= s <= 850 for s in scores)

    def test_distinct_scores_for_distinct_probabilities(self, sample_calibration):
        probs = np.linspace(0.05, 0.74, 20)
        scores = {probability_to_score(float(p), sample_calibration) for p in probs}
        assert len(scores) >= 15, "Calibration should spread scores across the 300–850 range"

    def test_calibration_expands_compressed_probabilities(self, sample_calibration):
        raw_tight = [0.30, 0.31, 0.32, 0.33]
        without = [probability_to_score(p, calibration=None) for p in raw_tight]
        with_cal = [probability_to_score(p, sample_calibration) for p in raw_tight]
        assert max(with_cal) - min(with_cal) >= max(without) - min(without)

    def test_clamps_to_300_850(self, sample_calibration):
        assert probability_to_score(-1.0, sample_calibration) == 850
        assert probability_to_score(99.0, sample_calibration) == 300

    def test_identical_probability_yields_identical_score(self, sample_calibration):
        a = probability_to_score(0.42, sample_calibration)
        b = probability_to_score(0.42, sample_calibration)
        assert a == b

    def test_loads_project_calibration_file(self):
        cal_path = Path("models/score_calibration.json")
        if not cal_path.exists():
            pytest.skip("Calibration file not present")
        cal = load_score_calibration()
        assert cal is not None
        assert cal["p_max"] > cal["p_min"]
        score_at_low_risk = probability_to_score(cal["p_min"], cal)
        score_at_high_risk = probability_to_score(cal["p_max"], cal)
        assert score_at_low_risk > score_at_high_risk
        assert score_at_low_risk - score_at_high_risk >= 400, "Calibration should span most of 300–850"


# ─── risk_band ──────────────────────────────────────────────────────────────


class TestRiskBand:
    @pytest.mark.parametrize(
        "proba,expected",
        [(0.0, "Low"), (0.29, "Low"), (0.3, "Medium"), (0.59, "Medium"), (0.6, "High"), (1.0, "High")],
    )
    def test_thresholds(self, proba, expected):
        assert risk_band(proba) == expected


# ─── Feature derivation (MFI portal path) ───────────────────────────────────


class TestApplicantFeatureDerivation:
    def test_different_applicants_produce_different_feature_rows(self):
        strong = applicant_dict_to_dataframe({
            "age": 28, "location": "urban", "mm_txn_per_month": 80,
            "utility_payment_rate": 0.95, "util_overdue_count": 0,
            "digital_engagement": 0.9, "social_media_usage": 0.8,
        })
        weak = applicant_dict_to_dataframe({
            "age": 55, "location": "rural", "mm_txn_per_month": 5,
            "utility_payment_rate": 0.35, "util_overdue_count": 5,
            "digital_engagement": 0.1, "social_media_usage": 0.1,
        })
        numeric_cols = [
            c for c in strong.columns
            if strong[c].dtype in ("int64", "float64") and c in weak.columns
        ]
        diffs = sum(
            1 for c in numeric_cols
            if float(strong[c].iloc[0]) != float(weak[c].iloc[0])
        )
        assert diffs >= 8, "Strong vs weak applicants should differ on many numeric fields"

    def test_derived_youth_and_age_group(self):
        young = _derive_applicant_features({"age": 22})
        old = _derive_applicant_features({"age": 58})
        assert young["youth"] == 1 and old["youth"] == 0
        assert young["age_group"] == "18-25"
        assert old["age_group"] == "55+"

    def test_derived_urban_flag(self):
        urban = _derive_applicant_features({"location": "urban"})
        rural = _derive_applicant_features({"location": "rural"})
        assert urban["is_urban"] == 1
        assert rural["is_urban"] == 0

    def test_msme_derives_business_fields(self):
        msme = _derive_applicant_features({"msme": 1, "mm_txn_per_month": 40, "account_age_months": 24})
        indiv = _derive_applicant_features({"msme": 0, "mm_txn_per_month": 40, "account_age_months": 24})
        assert msme["business_supplier_txns"] > indiv["business_supplier_txns"]
        assert msme["business_customer_txns"] > indiv["business_customer_txns"]

    def test_overdue_utilities_reduce_on_time_streak(self):
        clean = _derive_applicant_features({"util_overdue_count": 0})
        late = _derive_applicant_features({"util_overdue_count": 4})
        assert clean["consecutive_on_time"] > late["consecutive_on_time"]


# ─── score_features / score_applicant_row (mocked model) ────────────────────


class TestScoreVariationWithMockModel:
    def test_score_features_differs_for_contrasting_applicants(self, mock_scoring_artifacts, sample_calibration):
        model, prep, _ = mock_scoring_artifacts
        artifacts = (model, prep, None)

        strong = applicant_dict_to_dataframe({
            "mm_txn_per_month": 90, "utility_payment_rate": 0.98,
            "util_overdue_count": 0, "digital_engagement": 0.85,
        })
        weak = applicant_dict_to_dataframe({
            "mm_txn_per_month": 4, "utility_payment_rate": 0.25,
            "util_overdue_count": 6, "digital_engagement": 0.15,
        })

        with patch("src.scoring.inference.load_score_calibration", return_value=sample_calibration):
            with patch("src.evaluation.explainability.get_lora_input_attribution", return_value=[]):
                s_strong = score_features(strong, artifacts=artifacts, explain=False)
                s_weak = score_features(weak, artifacts=artifacts, explain=False)

        assert s_strong["score"] != s_weak["score"]
        assert s_strong["default_probability"] != s_weak["default_probability"]
        assert s_strong["score"] > s_weak["score"], "Strong payer should score higher"

    def test_many_random_applicants_yield_many_unique_scores(self, mock_scoring_artifacts, sample_calibration):
        model, prep, df = mock_scoring_artifacts
        artifacts = (model, prep, df)
        indices = np.linspace(0, len(df) - 1, 40, dtype=int)

        with patch("src.scoring.inference.load_score_calibration", return_value=sample_calibration):
            with patch("src.evaluation.explainability.get_lora_input_attribution", return_value=[]):
                scores = [
                    score_applicant_row(int(i), artifacts=artifacts)["score"]
                    for i in indices
                ]

        unique = len(set(scores))
        assert unique >= 25, f"Expected diverse scores across 40 applicants, got {unique} unique"
        assert max(scores) - min(scores) >= 80, "Score range should be materially wide"

    def test_same_applicant_is_deterministic(self, mock_scoring_artifacts, sample_calibration):
        model, prep, df = mock_scoring_artifacts
        artifacts = (model, prep, df)

        with patch("src.scoring.inference.load_score_calibration", return_value=sample_calibration):
            with patch("src.evaluation.explainability.get_lora_input_attribution", return_value=[]):
                a = score_applicant_row(7, artifacts=artifacts)
                b = score_applicant_row(7, artifacts=artifacts)

        assert a["score"] == b["score"]
        assert a["default_probability"] == b["default_probability"]

    def test_adjacent_rows_usually_differ(self, mock_scoring_artifacts, sample_calibration):
        model, prep, df = mock_scoring_artifacts
        artifacts = (model, prep, df)

        with patch("src.scoring.inference.load_score_calibration", return_value=sample_calibration):
            with patch("src.evaluation.explainability.get_lora_input_attribution", return_value=[]):
                pairs = [
                    (score_applicant_row(i, artifacts=artifacts)["score"],
                     score_applicant_row(i + 1, artifacts=artifacts)["score"])
                    for i in range(0, 30, 2)
                ]

        different = sum(1 for a, b in pairs if a != b)
        assert different >= len(pairs) * 0.6, "Most consecutive applicants should not share a score"

    def test_probability_to_score_not_constant_across_batch(self, mock_scoring_artifacts):
        model, prep, df = mock_scoring_artifacts
        X = prep.transform(df.iloc[:50])
        probas = model.predict_proba_numpy(X)
        scores = [probability_to_score(float(p)) for p in probas]
        assert len(set(scores)) > 1, "Raw probabilities must not all map to one score"


# ─── Synthetic dataset-level variation ──────────────────────────────────────


class TestSyntheticDatasetScoreSpread:
    def test_generated_data_has_feature_variance(self, synthetic_applicants):
        cols = ["mm_txn_per_month", "utility_payment_rate", "digital_engagement", "util_overdue_count"]
        thresholds = {"util_overdue_count": 3}
        for col in cols:
            min_unique = thresholds.get(col, 10)
            assert synthetic_applicants[col].nunique() > min_unique, f"{col} should vary across rows"

    def test_no_single_probability_dominates_synthetic_rows(self, synthetic_applicants, mock_scoring_artifacts, sample_calibration):
        """If every row produced the same proba, calibration alone cannot fix the demo."""
        model, prep, _ = mock_scoring_artifacts
        subset = synthetic_applicants.head(60)
        X = prep.transform(subset)

        probas = model.predict_proba_numpy(X)
        assert probas.std() > 0.05
        assert len(np.unique(np.round(probas, 3))) >= 10


# ─── Integration with trained model (optional) ───────────────────────────────


@pytest.mark.integration
class TestTrainedModelScoreVariation:
    @pytest.fixture(autouse=True)
    def require_trained_model(self):
        if not Path("models/lora/best_model.pt").exists():
            pytest.skip("Trained model not available — run scripts/train.py first")

    def test_real_model_scores_diverse_applicants(self, synthetic_applicants):
        from src.scoring import load_scoring_artifacts, score_applicant_row

        artifacts = load_scoring_artifacts()
        assert artifacts is not None

        indices = [0, 10, 25, 50, 75, 100, 150, 199]
        scores = []
        probas = []
        for idx in indices:
            result = score_applicant_row(idx, artifacts=artifacts)
            assert result is not None
            scores.append(result["score"])
            probas.append(result["default_probability"])

        assert len(set(scores)) >= 5, f"Real model returned too few unique scores: {scores}"
        assert len(set(probas)) >= 5, f"Real model returned too few unique probabilities: {probas}"
        assert max(scores) - min(scores) >= 30

    def test_contrasting_profiles_get_different_real_scores(self):
        from src.scoring import score_features, applicant_dict_to_dataframe

        high = applicant_dict_to_dataframe({
            "utility_payment_rate": 0.99, "util_overdue_count": 0,
            "mm_txn_per_month": 100, "digital_engagement": 0.95,
            "airtime_consistency_score": 0.95, "account_age_months": 60,
        })
        low = applicant_dict_to_dataframe({
            "utility_payment_rate": 0.2, "util_overdue_count": 8,
            "mm_txn_per_month": 3, "digital_engagement": 0.05,
            "airtime_consistency_score": 0.1, "account_age_months": 3,
        })

        s_high = score_features(high, explain=False)
        s_low = score_features(low, explain=False)
        assert s_high is not None and s_low is not None
        assert s_high["score"] != s_low["score"]
