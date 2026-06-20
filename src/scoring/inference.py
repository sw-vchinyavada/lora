"""Shared LoRA credit scoring inference for Gradio demo and MFI portal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

_CALIBRATION_CACHE: Optional[Dict[str, float]] = None


def load_score_calibration(models_dir: str | Path = "models") -> Optional[Dict[str, float]]:
    """Load percentile calibration saved during training (spreads scores across 300–850)."""
    global _CALIBRATION_CACHE
    if _CALIBRATION_CACHE is not None:
        return _CALIBRATION_CACHE
    path = Path(models_dir) / "score_calibration.json"
    if not path.exists():
        return None
    with open(path) as f:
        _CALIBRATION_CACHE = json.load(f)
    return _CALIBRATION_CACHE


def probability_to_score(
    proba: float,
    calibration: Optional[Dict[str, float]] = None,
) -> int:
    """Map default probability to a 300–850 credit score."""
    p = float(proba)
    cal = calibration if calibration is not None else load_score_calibration()
    if cal and cal.get("p_max", 0) > cal.get("p_min", 0):
        p = (p - cal["p_min"]) / (cal["p_max"] - cal["p_min"])
        p = max(0.0, min(1.0, p))
    return int(round(max(300, min(850, 850 - p * 550))))


def risk_band(proba: float) -> str:
    if proba < 0.3:
        return "Low"
    if proba < 0.6:
        return "Medium"
    return "High"


def load_scoring_artifacts(models_dir: str | Path = "models"):
    """Load LoRA model, preprocessor, and dataset for inference."""
    models_dir = Path(models_dir)
    model_path = models_dir / "lora" / "best_model.pt"
    prep_path = models_dir / "preprocessor.pkl"
    if not model_path.exists() or not prep_path.exists():
        return None

    from src.data import load_dataset, DataPreprocessor
    from src.models import LoRACreditScorer

    prep = DataPreprocessor.load(str(prep_path))
    lora = LoRACreditScorer.load_pretrained(str(model_path))
    load_score_calibration(models_dir)
    dataset_name = (
        (models_dir / "dataset_name.txt").read_text().strip()
        if (models_dir / "dataset_name.txt").exists()
        else "zimbabwe_synthetic"
    )
    df = load_dataset(dataset_name)
    return lora, prep, df


def _transform_row(prep, df: pd.DataFrame, sample_idx: int, lora) -> np.ndarray:
    cols = [c for c in prep.feature_columns_ if c in df.columns]
    missing = [c for c in prep.feature_columns_ if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing trained features: {missing[:5]}")
    sample = df[cols].iloc[sample_idx : sample_idx + 1]
    X = prep.transform(sample)
    if X.shape[1] != lora.num_features:
        raise ValueError(
            f"Feature mismatch: preprocessor produced {X.shape[1]} features, "
            f"model expects {lora.num_features}. Retrain with scripts/train.py."
        )
    return X


def _transform_dataframe(prep, df: pd.DataFrame, lora) -> np.ndarray:
    for c in prep.feature_columns_:
        if c not in df.columns:
            df[c] = 0 if c in prep.numerical_features else "missing"
    df = df[prep.feature_columns_]
    X = prep.transform(df)
    if X.shape[1] != lora.num_features:
        raise ValueError(
            f"Feature mismatch: preprocessor produced {X.shape[1]} features, "
            f"model expects {lora.num_features}. Retrain with scripts/train.py."
        )
    return X


def _feature_names(prep) -> List[str]:
    return list(prep.numerical_features) + list(prep.categorical_features)


def score_features(
    df: pd.DataFrame,
    artifacts: Optional[Tuple] = None,
    models_dir: str | Path = "models",
    device: str = "cpu",
    explain: bool = True,
) -> Optional[Dict[str, Any]]:
    """Score one applicant from a feature row (used by Gradio and MFI portal)."""
    loaded = artifacts or load_scoring_artifacts(models_dir)
    if loaded is None:
        return None

    lora, prep, _ = loaded
    X = _transform_dataframe(prep, df.copy(), lora)
    proba = float(lora.predict_proba_numpy(X, device)[0])
    score = probability_to_score(proba)

    top_drivers: List[Dict[str, Any]] = []
    if explain:
        from src.evaluation.explainability import get_lora_input_attribution

        top_drivers = get_lora_input_attribution(
            lora, X, _feature_names(prep), device=device, top_k=5
        )

    return {
        "score": score,
        "default_probability": round(proba, 4),
        "risk_band": risk_band(proba),
        "top_drivers": top_drivers,
    }


def score_applicant_row(
    sample_idx: int,
    artifacts: Optional[Tuple] = None,
    models_dir: str | Path = "models",
    device: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """Score one applicant by dataset row index."""
    loaded = artifacts or load_scoring_artifacts(models_dir)
    if loaded is None:
        return None

    lora, prep, df = loaded
    sample_idx = min(max(0, int(sample_idx)), len(df) - 1)
    X = _transform_row(prep, df, sample_idx, lora)
    proba = float(lora.predict_proba_numpy(X, device)[0])
    score = probability_to_score(proba)

    from src.evaluation.explainability import get_lora_input_attribution

    top_drivers = get_lora_input_attribution(
        lora, X, _feature_names(prep), device=device, top_k=5
    )

    return {
        "score": score,
        "default_probability": round(proba, 4),
        "risk_band": risk_band(proba),
        "top_drivers": top_drivers,
        "sample_idx": sample_idx,
    }


def _derive_applicant_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """Fill derived Zimbabwe alternative-data fields so applicants differ meaningfully."""
    age = int(row.get("age", 35))
    row["youth"] = 1 if age <= 35 else 0
    if not row.get("age_group"):
        if age <= 25:
            row["age_group"] = "18-25"
        elif age <= 35:
            row["age_group"] = "26-35"
        elif age <= 45:
            row["age_group"] = "36-45"
        elif age <= 55:
            row["age_group"] = "46-55"
        else:
            row["age_group"] = "55+"

    location = str(row.get("location", "urban")).lower()
    row["is_urban"] = 1 if location == "urban" else 0

    txn = float(row.get("mm_txn_per_month", 20))
    avg = float(row.get("mm_avg_txn_usd", 25.0))
    row.setdefault("mm_total_volume_6m", txn * avg * 6)
    row.setdefault("mm_incoming_ratio", 0.5)
    row.setdefault("mm_unique_recipients", max(1, int(txn * 0.3)))
    row.setdefault("mm_unique_senders", max(1, int(txn * 0.3)))
    row.setdefault("mm_transaction_regularity", float(row.get("airtime_consistency_score", 0.6)))
    row.setdefault("mm_savings_balance_avg", avg * 6)
    row.setdefault("mm_savings_deposits_count", max(0, int(txn * 0.15)))
    bill = float(row.get("mm_bill_payment_ratio", 0.3))
    p2p = float(row.get("mm_p2p_ratio", 0.5))
    row.setdefault("mm_merchant_ratio", max(0.0, 1.0 - p2p - bill))

    util_rate = float(row.get("utility_payment_rate", 0.8))
    row.setdefault("utility_accounts_count", 2 if util_rate > 0.5 else 1)
    row.setdefault("utility_avg_monthly_usd", 30.0 * util_rate + 5.0)
    row.setdefault("utility_months_history", int(row.get("account_age_months", 12)))
    overdue = int(row.get("util_overdue_count", 0))
    row.setdefault("utility_late_payments_6m", overdue)

    msme = int(row.get("msme", 0))
    row.setdefault("is_business_account", msme)
    row.setdefault("business_supplier_txns", txn * msme * 0.4)
    row.setdefault("business_customer_txns", txn * msme * 0.6)
    row.setdefault("business_revenue_trend", 0.0 if msme else -0.1)
    row.setdefault("business_months_active", int(row.get("account_age_months", 12)) * msme)

    social = float(row.get("social_media_usage", 0.5))
    row.setdefault("social_media_platforms", max(0, int(social * 4)))
    smartphone = int(row.get("has_smartphone", 1))
    acct_age = float(row.get("account_age_months", 12))
    row.setdefault(
        "digital_engagement",
        min(1.0, max(0.0, social * 0.4 + smartphone * 0.3 + min(acct_age / 120.0, 1.0) * 0.3)),
    )

    row.setdefault("remittance_receipt_freq", 0)
    row.setdefault("remittance_avg_usd", 0.0)

    vol = float(row.get("mm_balance_volatility", 0.5))
    row["txn_cv"] = vol * avg / (avg + 1)
    util_cons = float(row.get("util_multi_service_consistency", 0.7))
    row["payment_trend"] = util_cons - 0.5
    if "consecutive_on_time" not in row or row["consecutive_on_time"] is None:
        row["consecutive_on_time"] = 6 if overdue == 0 else max(0, 6 - overdue)
    return row


def applicant_dict_to_dataframe(applicant: Dict[str, Any]) -> pd.DataFrame:
    """Convert MFI portal applicant payload to a single-row dataframe."""
    aliases = {"mm_txn_freq": "mm_txn_per_month", "mm_avg_amount_usd": "mm_avg_txn_usd"}
    defaults = {
        "gender": "male", "age": 35, "youth": 0, "age_group": "26-35",
        "location": "urban", "is_urban": 1, "region": "Harare",
        "education": "secondary", "household_size": 4, "employment": "informal", "sector": "other", "msme": 0,
        "mm_provider": "ecocash", "mm_txn_per_month": 20, "mm_avg_txn_usd": 25.0,
        "mm_total_volume_6m": 3000.0, "mm_incoming_ratio": 0.5,
        "mm_unique_recipients": 6, "mm_unique_senders": 6, "mm_transaction_regularity": 0.6,
        "mm_balance_volatility": 0.5, "mm_tenure_months": 24, "mm_days_since_last": 7,
        "mm_savings_balance_avg": 150.0, "mm_savings_deposits_count": 3,
        "mm_p2p_ratio": 0.5, "mm_bill_payment_ratio": 0.3, "mm_merchant_ratio": 0.2,
        "mm_incoming_outgoing_ratio": 1.0, "mm_weekend_usage_pct": 0.25,
        "airtime_topups_per_month": 8, "airtime_avg_amount_usd": 3.0, "airtime_consistency_score": 0.6,
        "data_bundles_per_month": 3, "data_avg_bundle_usd": 2.5,
        "zesa_type": "prepaid", "utility_accounts_count": 2, "utility_payment_rate": 0.8,
        "utility_avg_monthly_usd": 29.0, "utility_months_history": 12, "utility_late_payments_6m": 0,
        "util_electricity_consistency": 0.7, "util_water_consistency": 0.7,
        "util_telecom_consistency": 0.7, "util_overdue_count": 0,
        "util_avg_delay_days": 0.0, "util_avg_amount_zwl": 500.0,
        "util_multi_service_consistency": 0.7,
        "is_business_account": 0, "business_supplier_txns": 0, "business_customer_txns": 0,
        "business_revenue_trend": -0.1, "business_months_active": 0,
        "consecutive_on_time": 6, "account_age_months": 12, "digital_engagement": 0.5,
        "has_smartphone": 1, "social_media_usage": 0.5, "social_media_platforms": 2,
        "app_sessions_per_week": 5, "service_disruption_count": 0, "channel_diversity": 2,
        "remittance_receipt_freq": 0, "remittance_avg_usd": 0.0,
        "txn_cv": 0.1, "payment_trend": 0.2,
    }
    row = {**defaults, **{k: v for k, v in applicant.items() if v is not None}}
    for old_name, new_name in aliases.items():
        if old_name in row and new_name not in row:
            row[new_name] = row[old_name]
    row = _derive_applicant_features(row)
    return pd.DataFrame([row])
