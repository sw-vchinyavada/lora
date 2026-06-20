"""Tests for Live Demo applicant selection helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from app.demo_helpers import (
    DEMO_SCENARIOS,
    applicant_choice_label,
    apply_demo_scenario_filters,
    build_applicant_snapshot,
    filter_applicant_indices,
    resolve_applicant_list,
    snapshot_tagline,
)


@pytest.fixture
def mini_df():
    return pd.DataFrame({
        "gender": ["female", "male", "female", "male"],
        "age": [28, 45, 32, 19],
        "location": ["rural", "urban", "rural", "urban"],
        "msme": [1, 0, 1, 0],
        "mm_provider": ["ecocash", "onemoney", "ecocash", "ecocash"],
        "mm_txn_per_month": [40, 12, 55, 8],
        "utility_payment_rate": [0.9, 0.5, 0.88, 0.3],
        "util_overdue_count": [0, 1, 0, 4],
        "digital_engagement": [0.7, 0.4, 0.8, 0.2],
    })


class TestApplicantFiltering:
    def test_gender_filter(self, mini_df):
        idx = filter_applicant_indices(mini_df, "Female", "All", "All")
        assert list(idx) == [0, 2]

    def test_msme_filter(self, mini_df):
        idx = filter_applicant_indices(mini_df, "All", "All", "MSME")
        assert list(idx) == [0, 2]

    def test_strong_payer_extra_filter(self, mini_df):
        idx = filter_applicant_indices(mini_df, "All", "All", "All", extra_filter="strong_payer")
        assert list(idx) == [0, 2]

    def test_overdue_extra_filter(self, mini_df):
        idx = filter_applicant_indices(mini_df, "All", "All", "All", extra_filter="overdue")
        assert list(idx) == [3]


class TestDemoScenarioPresets:
    def test_all_scenarios_have_required_keys(self):
        for scenario in DEMO_SCENARIOS.values():
            assert "label" in scenario
            assert "gender" in scenario
            assert "location" in scenario
            assert "msme" in scenario
            assert "hint" in scenario

    def test_rural_msme_women_scenario_filters(self):
        gender, location, msme, hint = apply_demo_scenario_filters("rural_msme_women")
        assert gender == "Female"
        assert location == "Rural"
        assert msme == "MSME"
        assert hint

    def test_resolve_list_for_scenario(self, mini_df):
        choices, first_id, snapshot, profile = resolve_applicant_list(
            "Female", "Rural", "MSME", "rural_msme_women", df=mini_df,
        )
        assert len(choices) == 2
        assert first_id in ("0", "2")
        assert "Applicant #" in snapshot
        assert "profile-card" in profile


class TestApplicantPreview:
    def test_choice_label_is_compact(self, mini_df):
        label = applicant_choice_label(mini_df.iloc[0], 0)
        assert "#1" in label
        assert "txn/mo" in label
        assert len(label) < 60

    def test_snapshot_contains_key_metrics(self, mini_df):
        html = build_applicant_snapshot(mini_df.iloc[0], 0)
        assert "Mobile money" in html
        assert "Utility rate" in html
        assert "Digital engagement" in html
        assert "Applicant #1" in html

    def test_snapshot_tagline_varies_by_profile(self, mini_df):
        strong = snapshot_tagline(mini_df.iloc[0])
        risky = snapshot_tagline(mini_df.iloc[3])
        assert strong != risky
