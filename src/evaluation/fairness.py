"""
Fairness evaluation for credit scoring models.

Per dissertation §3.7-3.8 (MTech_Dissertation_Chapters_1-3):
- Demographic parity: difference in positive prediction rates
- Equal opportunity: difference in TPR (recall) across groups
- Equalized odds: difference in TPR and FPR across groups
- Calibration: predicted probabilities vs actual outcomes
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix


def _tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def _positive_rate(y_pred: np.ndarray) -> float:
    return y_pred.mean()


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "positive_rate": _positive_rate(y_pred),
        "tpr": _tpr(y_true, y_pred),
        "fpr": _fpr(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def demographic_parity_difference(group_metrics: Dict[str, Dict]) -> float:
    """|max_rate - min_rate|. 0 = perfect parity (§3.7)."""
    rates = [m["positive_rate"] for m in group_metrics.values()]
    return max(rates) - min(rates) if len(rates) >= 2 else 0.0


def equal_opportunity_difference(group_metrics: Dict[str, Dict]) -> float:
    """|max_TPR - min_TPR|. 0 = equal opportunity (§3.7)."""
    tprs = [m["tpr"] for m in group_metrics.values()]
    return max(tprs) - min(tprs) if len(tprs) >= 2 else 0.0


def equalized_odds_difference(group_metrics: Dict[str, Dict]) -> Dict[str, float]:
    """Max TPR diff and max FPR diff across groups (§3.7)."""
    tprs = [m["tpr"] for m in group_metrics.values()]
    fprs = [m["fpr"] for m in group_metrics.values()]
    return {
        "tpr_diff": max(tprs) - min(tprs) if len(tprs) >= 2 else 0.0,
        "fpr_diff": max(fprs) - min(fprs) if len(fprs) >= 2 else 0.0,
    }


def compute_fairness_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    feature_columns: list,
) -> Dict[str, Any]:
    """
    Fairness metrics per §3.7: demographic parity, equal opportunity, equalized odds.
    Subgroups: gender, location, age (youth/adult), MSME (§3.8).
    """
    results = {"groups": {}, "disparity": {}, "fairness_metrics": {}}

    attr_map = {}
    if "gender" in df.columns:
        attr_map["gender"] = df["gender"].astype(str)
    elif "sex" in df.columns:
        attr_map["gender"] = df["sex"].map({1: "male", 2: "female", "1": "male", "2": "female"}).fillna("other").astype(str)
    if "location" in df.columns:
        attr_map["location"] = df["location"].astype(str)
    if "youth" in df.columns:
        attr_map["age_group"] = df["youth"].map({0: "adult", 1: "youth"}).fillna("unknown")
    elif "age" in df.columns:
        attr_map["age_group"] = np.where(df["age"].values <= 35, "youth", "adult")
    if "msme" in df.columns:
        attr_map["msme"] = df["msme"].map({0: "non_msme", 1: "msme"}).fillna("unknown")

    for attr_name, attr_vals in attr_map.items():
        if attr_vals is None or len(attr_vals) != len(y_true):
            continue
        attr_vals = pd.Series(attr_vals).fillna("missing").astype(str)
        group_metrics = {}
        for g in attr_vals.unique():
            mask = (attr_vals == g).values
            if mask.sum() < 20:
                continue
            group_metrics[str(g)] = _group_metrics(y_true[mask], y_pred[mask], y_proba[mask])
        if group_metrics:
            results["groups"][attr_name] = group_metrics
            results["disparity"][f"{attr_name}_demographic_parity"] = demographic_parity_difference(group_metrics)
            results["disparity"][f"{attr_name}_equal_opportunity"] = equal_opportunity_difference(group_metrics)
            eo = equalized_odds_difference(group_metrics)
            results["disparity"][f"{attr_name}_equalized_odds_tpr"] = eo["tpr_diff"]
            results["disparity"][f"{attr_name}_equalized_odds_fpr"] = eo["fpr_diff"]

    results["fairness_metrics"] = {
        "demographic_parity": {k: v for k, v in results["disparity"].items() if "demographic_parity" in k},
        "equal_opportunity": {k: v for k, v in results["disparity"].items() if "equal_opportunity" in k},
        "equalized_odds": {k: v for k, v in results["disparity"].items() if "equalized_odds" in k},
    }
    return results
