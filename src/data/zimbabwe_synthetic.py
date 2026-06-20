"""
Synthetic Zimbabwe Alternative Data for Credit Scoring.

Aligned with MTECH Software Engineering Project Documentation §3.3:
- EcoCash / OneMoney mobile money (90%+ market share)
- ZESA (electricity), water, telecom utilities
- Airtime & data bundles (telecom)
- Social media / digital footprint
- Reference: raw_credit_data.csv structure

Features: ~87 engineered (demographic, mobile money, utility, telecom, business, digital)
Default rate: 8-12% (microfinance/MSME)
Missing data: 5-15%
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Zimbabwe provinces (FinScope, ZimStat)
ZIMBABWE_REGIONS = ["Harare", "Bulawayo", "Midlands", "Mashonaland_East", "Mashonaland_West",
                    "Mashonaland_Central", "Manicaland", "Masvingo", "Matabeleland_North", "Matabeleland_South"]


def generate_zimbabwe_alternative_data(
    n_samples: int = 50000,
    default_rate: float = 0.10,
    missingness_rate: float = 0.08,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic alternative data per dissertation §3.3.2.

    Mobile money (EcoCash, OneMoney): transactions, volume, savings, P2P, regularity
    Telecom: airtime topups, data bundles, consistency
    Utility (ZESA, water, telecom): payment rate, late payments, months history
    Business: supplier/customer txns, revenue trend (MSME)
    Digital: smartphone, social media usage, account age
    Demographic: gender, age, location, region, education
    """
    rng = np.random.default_rng(random_state)

    # --- Demographics (Zimbabwe: FinScope 2022, ZimStat) ---
    gender = rng.choice(["male", "female"], n_samples, p=[0.52, 0.48])
    age = rng.integers(18, 70, n_samples)
    youth = (age <= 35).astype(int)
    age_group = np.where(age <= 25, "18-25",
                 np.where(age <= 35, "26-35",
                 np.where(age <= 45, "36-45",
                 np.where(age <= 55, "46-55", "55+"))))
    location = rng.choice(["urban", "rural", "peri_urban"], n_samples, p=[0.35, 0.50, 0.15])
    is_urban = (location == "urban").astype(int)
    region = rng.choice(ZIMBABWE_REGIONS, n_samples, p=[0.23, 0.08, 0.11, 0.09, 0.08, 0.07, 0.12, 0.09, 0.07, 0.06])
    education = rng.choice(["none", "primary", "secondary", "tertiary"], n_samples, p=[0.15, 0.35, 0.40, 0.10])
    household_size = rng.integers(1, 10, n_samples)
    employment = rng.choice(["formal", "informal"], n_samples, p=[0.24, 0.76])
    sector = rng.choice(["retail", "agriculture", "services", "manufacturing", "other"], n_samples, p=[0.30, 0.35, 0.20, 0.08, 0.07])
    msme = rng.choice([0, 1], n_samples, p=[0.35, 0.65])

    # --- Mobile money (EcoCash 90%, OneMoney 10% - Reserve Bank 2024) ---
    mm_provider = rng.choice(["ecocash", "onemoney"], n_samples, p=[0.90, 0.10])
    mm_txn_per_month = rng.integers(2, 120, n_samples)
    mm_avg_txn_usd = rng.exponential(25, n_samples) + 2
    mm_total_volume_6m = mm_txn_per_month * mm_avg_txn_usd * 6 * (0.8 + rng.uniform(0, 0.4, n_samples))
    mm_incoming_ratio = rng.beta(2, 2, n_samples)
    mm_unique_recipients = rng.integers(1, 25, n_samples)
    mm_unique_senders = rng.integers(1, 25, n_samples)
    mm_transaction_regularity = rng.beta(2, 2, n_samples)
    mm_days_since_last = rng.exponential(7, n_samples).astype(int).clip(0, 90)
    mm_savings_balance_avg = rng.exponential(150, n_samples) + 10
    mm_savings_deposits_count = rng.integers(0, 20, n_samples)
    mm_balance_volatility = rng.uniform(0.05, 2.5, n_samples)
    mm_tenure_months = rng.integers(1, 120, n_samples)
    mm_p2p_ratio = rng.beta(2, 2, n_samples)
    mm_bill_payment_ratio = rng.beta(1, 3, n_samples)
    mm_merchant_ratio = np.clip(1 - mm_p2p_ratio - mm_bill_payment_ratio, 0, 1)
    mm_incoming_outgoing_ratio = rng.uniform(0.3, 3.0, n_samples)
    mm_weekend_usage_pct = rng.uniform(0.1, 0.5, n_samples)

    # --- Airtime & data (telecom - Zimbabwe Econet, NetOne, Telecel) ---
    airtime_topups_per_month = rng.integers(1, 30, n_samples)
    airtime_avg_amount_usd = rng.exponential(5, n_samples) + 1
    airtime_consistency_score = rng.beta(2, 2, n_samples)
    data_bundles_per_month = rng.integers(0, 15, n_samples)
    data_avg_bundle_usd = rng.exponential(3, n_samples) + 0.5

    # --- Utility (ZESA electricity, water, telecom bills — §3.3.2) ---
    zesa_type = rng.choice(["prepaid", "postpaid"], n_samples, p=[0.65, 0.35])
    utility_accounts_count = rng.integers(1, 4, n_samples)
    utility_payment_rate = rng.beta(3, 2, n_samples)
    utility_avg_monthly_usd = rng.exponential(25, n_samples) + 5
    utility_months_history = rng.integers(1, 84, n_samples)
    utility_late_payments_6m = rng.poisson(1, n_samples).clip(0, 12)
    util_electricity_consistency = rng.beta(2, 2, n_samples)
    util_water_consistency = rng.beta(2, 2, n_samples)
    util_telecom_consistency = rng.beta(2, 2, n_samples)
    util_overdue_count = utility_late_payments_6m
    util_avg_delay_days = rng.exponential(5, n_samples).clip(0, 60)
    util_avg_amount_zwl = rng.exponential(500, n_samples) + 50
    util_multi_service_consistency = (util_electricity_consistency + util_water_consistency + util_telecom_consistency) / 3

    # --- Business (MSME - supplier/customer txns, revenue trend) ---
    is_business_account = (msme == 1) & (rng.random(n_samples) < 0.6)
    business_supplier_txns = np.where(is_business_account, rng.integers(0, 50, n_samples), 0)
    business_customer_txns = np.where(is_business_account, rng.integers(0, 80, n_samples), 0)
    business_revenue_trend = np.where(is_business_account, rng.normal(0, 0.3, n_samples), 0.0)
    business_months_active = np.where(is_business_account, rng.integers(1, 60, n_samples), 0)

    # --- Digital footprint (social media, smartphone — §3.3.2) ---
    has_smartphone = rng.choice([0, 1], n_samples, p=[0.35, 0.65])
    social_media_usage = rng.beta(2, 2, n_samples)
    social_media_platforms = rng.integers(0, 5, n_samples)
    account_age_months = rng.integers(1, 120, n_samples)
    digital_engagement = (social_media_usage * 0.4 + has_smartphone * 0.3 + np.clip(account_age_months / 120, 0, 1) * 0.3) + rng.normal(0, 0.1, n_samples)
    digital_engagement = np.clip(digital_engagement, 0, 1)
    app_sessions_per_week = rng.integers(0, 60, n_samples)
    service_disruption_count = rng.poisson(0.3, n_samples).clip(0, 5)
    channel_diversity = rng.integers(1, 5, n_samples)

    # --- Remittance (diaspora — §3.3.2) ---
    remittance_receipt_freq = rng.choice([0, 1, 2, 3], n_samples, p=[0.5, 0.25, 0.15, 0.10])
    remittance_avg_usd = np.where(remittance_receipt_freq > 0, rng.exponential(100, n_samples) + 20, 0.0)

    # --- Derived / engineered ---
    txn_cv = mm_balance_volatility * mm_avg_txn_usd / (mm_avg_txn_usd + 1)
    payment_trend = util_multi_service_consistency - 0.5 + rng.normal(0, 0.1, n_samples)
    consecutive_on_time = (utility_late_payments_6m == 0).astype(int) * rng.integers(1, 12, n_samples)

    # --- Target: default 8-12% (correlated with payment behaviour, digital, business) ---
    default_logit = (
        -0.5 * util_multi_service_consistency
        - 0.3 * utility_payment_rate
        - 0.15 * mm_transaction_regularity
        - 0.2 * airtime_consistency_score
        - 0.15 * digital_engagement
        + 0.4 * utility_late_payments_6m / 6
        + 0.2 * (mm_days_since_last > 14).astype(float)
        + 0.15 * (location == "rural").astype(float)
        + 0.08 * youth
        - 0.1 * (employment == "formal").astype(float)
        + 0.1 * (business_revenue_trend < -0.2).astype(float)
        + 0.05 * service_disruption_count
        + rng.normal(0, 0.65, n_samples)
    )
    default_proba = 1 / (1 + np.exp(-np.clip(default_logit, -10, 10)))
    threshold = np.percentile(default_proba, 100 * (1 - default_rate))
    default = (default_proba >= threshold).astype(int)

    # --- Assemble ---
    df = pd.DataFrame({
        "gender": gender, "age": age, "youth": youth, "age_group": age_group,
        "location": location, "is_urban": is_urban, "region": region,
        "education": education, "household_size": household_size,
        "employment": employment, "sector": sector, "msme": msme,
        "mm_provider": mm_provider,
        "mm_txn_per_month": mm_txn_per_month, "mm_avg_txn_usd": mm_avg_txn_usd,
        "mm_total_volume_6m": mm_total_volume_6m, "mm_incoming_ratio": mm_incoming_ratio,
        "mm_unique_recipients": mm_unique_recipients, "mm_unique_senders": mm_unique_senders,
        "mm_transaction_regularity": mm_transaction_regularity,
        "mm_days_since_last": mm_days_since_last, "mm_savings_balance_avg": mm_savings_balance_avg,
        "mm_savings_deposits_count": mm_savings_deposits_count,
        "mm_balance_volatility": mm_balance_volatility, "mm_tenure_months": mm_tenure_months,
        "mm_p2p_ratio": mm_p2p_ratio, "mm_bill_payment_ratio": mm_bill_payment_ratio,
        "mm_merchant_ratio": mm_merchant_ratio, "mm_incoming_outgoing_ratio": mm_incoming_outgoing_ratio,
        "mm_weekend_usage_pct": mm_weekend_usage_pct,
        "airtime_topups_per_month": airtime_topups_per_month, "airtime_avg_amount_usd": airtime_avg_amount_usd,
        "airtime_consistency_score": airtime_consistency_score,
        "data_bundles_per_month": data_bundles_per_month, "data_avg_bundle_usd": data_avg_bundle_usd,
        "zesa_type": zesa_type,
        "utility_accounts_count": utility_accounts_count, "utility_payment_rate": utility_payment_rate,
        "utility_avg_monthly_usd": utility_avg_monthly_usd, "utility_months_history": utility_months_history,
        "utility_late_payments_6m": utility_late_payments_6m,
        "util_electricity_consistency": util_electricity_consistency,
        "util_water_consistency": util_water_consistency, "util_telecom_consistency": util_telecom_consistency,
        "util_overdue_count": util_overdue_count, "util_avg_delay_days": util_avg_delay_days,
        "util_avg_amount_zwl": util_avg_amount_zwl, "util_multi_service_consistency": util_multi_service_consistency,
        "is_business_account": is_business_account.astype(int),
        "business_supplier_txns": business_supplier_txns, "business_customer_txns": business_customer_txns,
        "business_revenue_trend": business_revenue_trend, "business_months_active": business_months_active,
        "account_age_months": account_age_months, "has_smartphone": has_smartphone,
        "social_media_usage": social_media_usage, "social_media_platforms": social_media_platforms,
        "digital_engagement": digital_engagement, "app_sessions_per_week": app_sessions_per_week,
        "service_disruption_count": service_disruption_count, "channel_diversity": channel_diversity,
        "remittance_receipt_freq": remittance_receipt_freq, "remittance_avg_usd": remittance_avg_usd,
        "txn_cv": txn_cv, "payment_trend": payment_trend, "consecutive_on_time": consecutive_on_time,
        "default": default,
    })

    # Income proxy for fairness evaluation §3.6.3 (quartiles from financial activity)
    activity_score = (
        0.35 * (mm_total_volume_6m / (mm_total_volume_6m.max() + 1))
        + 0.25 * (mm_savings_balance_avg / (mm_savings_balance_avg.max() + 1))
        + 0.20 * utility_payment_rate
        + 0.20 * mm_transaction_regularity
    )
    df["income_quartile"] = pd.qcut(activity_score, q=4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)

    if missingness_rate > 0:
        feature_cols = [c for c in df.columns if c != "default"]
        n_missing = int(n_samples * len(feature_cols) * missingness_rate)
        for _ in range(n_missing):
            r, c = rng.integers(0, n_samples), rng.integers(0, len(feature_cols))
            df.iloc[r, df.columns.get_loc(feature_cols[c])] = np.nan

    return df


def load_zimbabwe_synthetic(
    n_samples: int = 50000,
    data_dir: Optional[str] = None,
    force_regenerate: bool = False,
    default_rate: float = 0.10,
) -> pd.DataFrame:
    """Load or generate Zimbabwe synthetic dataset per dissertation §3.3."""
    data_dir = Path(data_dir or "data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "zimbabwe_alternative_data.csv"

    meta_path = data_dir / "zimbabwe_alternative_data.meta.json"
    needs_regen = force_regenerate or not path.exists()
    if not needs_regen and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("n_samples") != n_samples or meta.get("default_rate") != default_rate:
            needs_regen = True
    elif not needs_regen:
        existing_rows = sum(1 for _ in open(path, encoding="utf-8")) - 1
        if existing_rows != n_samples:
            needs_regen = True
        else:
            sample = pd.read_csv(path, usecols=["default"], nrows=min(1000, existing_rows))
            if sample["default"].isna().any():
                needs_regen = True

    if needs_regen:
        df = generate_zimbabwe_alternative_data(n_samples=n_samples, default_rate=default_rate)
        df.to_csv(path, index=False)
        meta_path.write_text(json.dumps({
            "n_samples": n_samples,
            "default_rate": default_rate,
            "feature_count": len(df.columns) - 1,
            "source": "MTECH Software Engineering Project Documentation §3.3",
        }, indent=2))
        print(
            f"Generated {len(df)} samples, {df['default'].sum():,} defaults "
            f"({100 * df['default'].mean():.1f}%), {len(df.columns) - 1} features -> {path}"
        )

    return pd.read_csv(path)
