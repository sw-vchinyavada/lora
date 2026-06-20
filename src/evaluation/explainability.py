"""
Explainable AI for credit scoring — transparency and regulatory compliance.

Per dissertation §3.7-3.8: attention weights, permutation importance, SHAP values.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path


def get_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    n_samples: int = 100,
) -> Optional[Dict[str, Any]]:
    """SHAP values for local interpretability (§3.7). Uses TreeExplainer for RF/XGB."""
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
        shap_vals = shap_vals[1]  # binary: use positive class
    mean_abs = np.abs(shap_vals).mean(axis=0)
    if mean_abs.ndim > 1:
        mean_abs = mean_abs.mean(axis=-1)
    mean_abs = np.asarray(mean_abs).flatten()
    idx = np.argsort(mean_abs)[::-1][:10]
    top_indices = [int(i) for i in idx.flatten().tolist() if 0 <= int(i) < len(feature_names)]
    return {
        "shap_importance": [{"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])} for i in top_indices],
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


def explain_prediction(
    X: np.ndarray,
    feature_names: List[str],
    prediction: float,
    top_k: int = 5,
) -> List[Tuple[str, float, str]]:
    """
    Simple explanation: features with largest absolute values that push toward default.
    Returns list of (feature_name, value, direction).
    """
    if len(feature_names) != X.shape[1]:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    x = X.flatten()
    # Use magnitude as proxy for influence (simplified)
    idx = np.argsort(np.abs(x))[::-1][:top_k]
    rows = []
    for i in idx:
        direction = "↑ risk" if x[i] > 0 else "↓ risk"
        rows.append((feature_names[i], float(x[i]), direction))
    return rows


def get_lora_attention_weights(model, X: np.ndarray) -> np.ndarray:
    """Placeholder: LoRA/projection layer weights as proxy for feature influence."""
    if hasattr(model, "feature_projection") and model.feature_projection is not None:
        w = model.feature_projection[0].weight.detach().cpu().numpy()
        return np.abs(w).mean(axis=0)
    return np.ones(X.shape[1]) / X.shape[1]
