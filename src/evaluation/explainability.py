"""
Explainable AI for credit scoring — transparency and regulatory compliance.

Per MTECH Software Engineering Project Documentation §4.4.9:
- Random Forest SHAP for global baseline comparison
- LoRA input perturbation attribution for per-applicant explanations
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from pathlib import Path


def get_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    n_samples: int = 100,
) -> Optional[Dict[str, Any]]:
    """SHAP values for baseline models (TreeExplainer for RF/XGB)."""
    try:
        import shap
    except ImportError:
        return None
    sklearn_model = getattr(model, "model", model)
    if not hasattr(sklearn_model, "predict_proba"):
        return None
    try:
        explainer = shap.TreeExplainer(sklearn_model)
    except Exception:
        return None
    X_sample = X[:n_samples] if len(X) > n_samples else X
    shap_vals = explainer.shap_values(X_sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=-1)
    mean_abs = np.asarray(mean_abs).flatten()
    idx = np.argsort(mean_abs)[::-1][:10]
    top_indices = [int(i) for i in idx.flatten().tolist() if 0 <= int(i) < len(feature_names)]
    return {
        "shap_importance": [
            {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in top_indices
        ],
    }


def get_feature_importance(
    model,
    feature_names: List[str],
    X: np.ndarray,
) -> pd.DataFrame:
    """Extract feature importance from tree/linear models."""
    imp = model.get_feature_importance()
    if len(imp) == 0:
        return pd.DataFrame()
    imp = np.abs(imp) / (np.abs(imp).sum() + 1e-9)
    idx = np.argsort(imp)[::-1]
    return pd.DataFrame({
        "feature": [feature_names[i] for i in idx if i < len(feature_names)],
        "importance": imp[idx],
    })


def get_lora_input_attribution(
    model,
    X: np.ndarray,
    feature_names: List[str],
    device: str = "cpu",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Per-applicant LoRA attribution via input perturbation.
    Measures how a small feature change shifts default probability.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    x0 = X[0:1].astype(np.float32).copy()
    base_proba = float(model.predict_proba_numpy(x0, device)[0])
    impacts = []
    for i in range(x0.shape[1]):
        x_pert = x0.copy()
        delta = max(abs(float(x0[0, i])) * 0.05, 0.01)
        x_pert[0, i] += delta
        pert_proba = float(model.predict_proba_numpy(x_pert, device)[0])
        impacts.append((feature_names[i], pert_proba - base_proba))
    impacts.sort(key=lambda t: abs(t[1]), reverse=True)
    return [
        {
            "feature": name,
            "impact": round(float(delta), 4),
            "direction": "increases risk" if delta > 0 else "decreases risk",
        }
        for name, delta in impacts[:top_k]
    ]


def get_lora_global_attribution(
    model,
    X: np.ndarray,
    feature_names: List[str],
    device: str = "cpu",
    n_samples: int = 50,
) -> List[Dict[str, Any]]:
    """Aggregate LoRA perturbation attribution across a sample for global ranking."""
    sample = X[:n_samples] if len(X) > n_samples else X
    scores = np.zeros(len(feature_names))
    for row in sample:
        attrs = get_lora_input_attribution(model, row.reshape(1, -1), feature_names, device, top_k=len(feature_names))
        for item in attrs:
            idx = feature_names.index(item["feature"])
            scores[idx] += abs(item["impact"])
    scores /= max(len(sample), 1)
    idx = np.argsort(scores)[::-1][:10]
    return [{"feature": feature_names[i], "mean_abs_impact": float(scores[i])} for i in idx]


def get_lora_attention_weights(model, X: np.ndarray) -> np.ndarray:
    """Projection layer weights as proxy for feature influence (§4.4.10)."""
    if hasattr(model, "feature_projection") and model.feature_projection is not None:
        w = model.feature_projection[0].weight.detach().cpu().numpy()
        return np.abs(w).mean(axis=0)
    return np.ones(X.shape[1]) / X.shape[1]
