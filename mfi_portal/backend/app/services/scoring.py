"""Credit scoring service — integrates with LoRA model."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from typing import Optional, Dict, Any, List

from src.scoring.inference import probability_to_score, risk_band

_APPLICANT_ALIASES = {"mm_txn_freq": "mm_txn_per_month", "mm_avg_amount_usd": "mm_avg_txn_usd"}


def _load_model_and_prep():
    models_dir = PROJECT_ROOT / "models"
    if not (models_dir / "lora" / "best_model.pt").exists():
        return None, None
    from src.data import DataPreprocessor
    from src.models import LoRACreditScorer

    prep = DataPreprocessor.load(str(models_dir / "preprocessor.pkl"))
    lora = LoRACreditScorer.load_pretrained(str(models_dir / "lora" / "best_model.pt"))
    return lora, prep


def _applicant_to_dataframe(applicant: Dict[str, Any]) -> pd.DataFrame:
    defaults = {
        "gender": "male", "age": 35, "youth": 0, "location": "urban", "region": "Harare",
        "education": "secondary", "household_size": 4, "employment": "informal", "sector": "other", "msme": 0,
        "mm_provider": "ecocash", "mm_txn_per_month": 20, "mm_txn_freq": 20, "mm_avg_txn_usd": 25.0, "mm_avg_amount_usd": 25.0,
        "mm_balance_volatility": 0.5, "mm_tenure_months": 24, "mm_days_since_last": 7,
        "mm_p2p_ratio": 0.5, "mm_bill_payment_ratio": 0.3, "mm_merchant_ratio": 0.2,
        "mm_incoming_outgoing_ratio": 1.0, "mm_weekend_usage_pct": 0.25,
        "airtime_topups_per_month": 8, "airtime_avg_amount_usd": 3.0, "airtime_consistency_score": 0.6,
        "data_bundles_per_month": 3, "data_avg_bundle_usd": 2.5,
        "utility_payment_rate": 0.8, "zesa_type": "prepaid",
        "util_electricity_consistency": 0.7, "util_water_consistency": 0.7,
        "util_telecom_consistency": 0.7, "util_overdue_count": 0,
        "util_avg_delay_days": 0.0, "util_avg_amount_zwl": 500.0,
        "util_multi_service_consistency": 0.7,
        "consecutive_on_time": 6, "account_age_months": 12, "digital_engagement": 0.5,
        "has_smartphone": 1, "social_media_usage": 0.5,
        "app_sessions_per_week": 5, "service_disruption_count": 0, "channel_diversity": 2,
    }
    row = {**defaults, **{k: v for k, v in applicant.items() if v is not None}}
    for old_name, new_name in _APPLICANT_ALIASES.items():
        if old_name in row and new_name not in row:
            row[new_name] = row[old_name]
    age = row.get("age", 35)
    row["youth"] = 1 if age <= 35 else 0
    vol = row.get("mm_balance_volatility", 0.5)
    amt = row.get("mm_avg_txn_usd", row.get("mm_avg_amount_usd", 25.0))
    row["txn_cv"] = vol * amt / (amt + 1)
    row["payment_trend"] = row.get("util_multi_service_consistency", 0.7) - 0.5
    return pd.DataFrame([row])


def compute_score(applicant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    lora, prep = _load_model_and_prep()
    if lora is None or prep is None:
        return None

    df = _applicant_to_dataframe(applicant)
    for c in prep.feature_columns_:
        if c not in df.columns:
            df[c] = 0 if c in prep.numerical_features else "missing"
    df = df[prep.feature_columns_]
    X = prep.transform(df)
    if X.shape[1] != lora.num_features:
        return None

    proba = float(lora.predict_proba_numpy(X)[0])
    score = probability_to_score(proba)

    top_drivers: List[Dict] = []
    fi_path = PROJECT_ROOT / "results" / "metrics" / "feature_importance.json"
    if fi_path.exists():
        import json
        with open(fi_path) as f:
            data = json.load(f)
        fi = data.get("feature_importance", []) or data.get("shap_importance", [])
        key = "importance" if fi and "importance" in (fi[0] or {}) else "mean_abs_shap"
        top_drivers = [{"feature": t["feature"], "importance": t.get(key, 0)} for t in fi[:5]]

    return {
        "score": score,
        "default_probability": round(proba, 4),
        "risk_band": risk_band(proba),
        "top_drivers": top_drivers,
    }
