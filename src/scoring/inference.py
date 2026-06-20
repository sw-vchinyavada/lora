"""Shared LoRA credit scoring inference for Gradio demo and MFI portal."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def probability_to_score(proba: float) -> int:
    """Map default probability to a 300–850 credit score."""
    return int(max(300, min(850, 850 - float(proba) * 550)))


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


def score_applicant_row(
    sample_idx: int,
    artifacts: Optional[Tuple] = None,
    models_dir: str | Path = "models",
) -> Optional[Dict[str, Any]]:
    """Score one applicant by dataset row index."""
    loaded = artifacts or load_scoring_artifacts(models_dir)
    if loaded is None:
        return None

    lora, prep, df = loaded
    sample_idx = min(max(0, int(sample_idx)), len(df) - 1)
    X = _transform_row(prep, df, sample_idx, lora)
    proba = float(lora.predict_proba_numpy(X)[0])
    score = probability_to_score(proba)

    top_drivers: List[Dict[str, Any]] = []
    fi_path = Path("results/metrics/feature_importance.json")
    if fi_path.exists():
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
        "sample_idx": sample_idx,
    }
