#!/usr/bin/env python3
"""
Train LoRA Credit Scoring models.

Aligned with MTech_Dissertation_Chapters_1-3 §3 (Methodology):
- Synthetic Zimbabwe data §3.2 (50K, 45 features, 8-12% default)
- Train/val/test 70/15/15 §3.2
- Metrics: accuracy, precision, recall, F1, AUC-ROC, AUC-PR §3.7
- Fairness: demographic parity, equal opportunity, equalized odds §3.7
- Class-weighted BCE §3.6
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from src.data import load_dataset, DataPreprocessor
from src.models import (
    LogisticRegressionModel,
    RandomForestModel,
    GradientBoostingModel,
    LoRACreditScorer,
)
from src.training import LoRATrainer
from src.evaluation import compute_fairness_metrics, get_feature_importance, get_shap_values


def _metrics(y_true, y_pred, y_proba):
    """All metrics per dissertation §3.7."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
        "auc_pr": float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="zimbabwe_synthetic", choices=["zimbabwe_synthetic", "credit_card_default", "german_credit"])
    ap.add_argument("--n-samples", type=int, default=5000, help="Zimbabwe: 50000 per §3.2, 5000 for quick run")
    ap.add_argument("--output-dir", default="models")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline").mkdir(exist_ok=True)
    (out_dir / "lora").mkdir(exist_ok=True)
    (Path("results") / "metrics").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LoRA Credit Scoring — Dissertation Methodology §3")
    print("=" * 60)

    print(f"\n1. Loading {args.dataset}...")
    kwargs = {}
    if args.dataset == "zimbabwe_synthetic":
        kwargs = {"n_samples": args.n_samples, "default_rate": 0.10}
    df = load_dataset(args.dataset, **kwargs)
    if "default" not in df.columns:
        tc = next((c for c in df.columns if "default" in c.lower()), None)
        if tc:
            df["default"] = df[tc]
    print(f"   Samples: {len(df)}, Defaults: {df['default'].sum()} ({100*df['default'].mean():.1f}%)")

    print("\n2. Preprocessing (70/15/15 split §3.2)...")
    prep = DataPreprocessor(target="default", test_size=0.15, val_size=0.15)
    X, y, _ = prep.fit_transform(df)
    X_train, X_val, X_test, y_train, y_val, y_test = prep.split(X, y)
    n_feat = X_train.shape[1]
    feature_names = prep.numerical_features + list(prep.categorical_features)

    _, _, test_idx = prep.split_indices(len(y), y)
    df_fairness = df.iloc[test_idx].reset_index(drop=True)

    prep.save(str(out_dir / "preprocessor.pkl"))
    with open(out_dir / "dataset_name.txt", "w") as f:
        f.write(args.dataset)
    print(f"   Features: {n_feat}, Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    results = {}

    print("\n3. Training baselines...")
    for name, model in [
        ("logistic_regression", LogisticRegressionModel()),
        ("random_forest", RandomForestModel()),
        ("gradient_boosting", GradientBoostingModel()),
    ]:
        t0 = time.perf_counter()
        model.train(X_train, y_train)
        train_time = time.perf_counter() - t0
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        model.save_model(str(out_dir / "baseline" / f"{name}.pkl"))
        results[name] = {
            **_metrics(y_test, pred, proba),
            "train_time_sec": round(train_time, 2),
        }
        print(f"   {name}: AUC-ROC={results[name]['auc_roc']:.3f} AUC-PR={results[name]['auc_pr']:.3f} ({train_time:.1f}s)")

    print("\n4. Training LoRA (class-weighted BCE §3.6)...")
    lora = LoRACreditScorer(num_features=n_feat, hidden_dim=768)
    params = lora.count_parameters()
    t0 = time.perf_counter()
    n0, n1 = (y_train == 0).sum(), (y_train == 1).sum()
    w0, w1 = n1 / (n0 + n1) if n1 > 0 else 1.0, n0 / (n0 + n1) if n0 > 0 else 1.0
    trainer = LoRATrainer(lora, device=args.device, patience=5, class_weights=(w0, w1))
    trainer.fit(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size)
    lora_train_time = time.perf_counter() - t0
    lora.save_pretrained(str(out_dir / "lora" / "best_model.pt"))

    proba_lora = lora.predict_proba_numpy(X_test, args.device)
    pred_lora = (proba_lora > 0.5).astype(int)
    results["lora"] = {
        **_metrics(y_test, pred_lora, proba_lora),
        "trainable_params": params["trainable"],
        "total_params": params["total"],
        "train_time_sec": round(lora_train_time, 2),
    }
    print(f"   LoRA: AUC-ROC={results['lora']['auc_roc']:.3f} ({lora_train_time:.1f}s, {params['trainable_pct']:.1f}% params)")

    print("\n5. Fairness evaluation (§3.7: demographic parity, equal opportunity, equalized odds)...")
    rf_model = RandomForestModel()
    rf_model.load_model(str(out_dir / "baseline" / "random_forest.pkl"))
    fairness_lora = compute_fairness_metrics(df_fairness, y_test, pred_lora, proba_lora, feature_names)
    fairness_rf = compute_fairness_metrics(df_fairness, y_test, rf_model.predict(X_test), rf_model.predict_proba(X_test), feature_names)
    with open("results/metrics/fairness_results.json", "w") as f:
        json.dump({"lora": fairness_lora, "random_forest": fairness_rf}, f, indent=2, default=str)

    print("\n6. Explainability (§3.7: feature importance, SHAP)...")
    fi = get_feature_importance(rf_model, feature_names, X_test[:500])
    explain_data = {"feature_importance": fi.head(10).to_dict(orient="records") if len(fi) > 0 else []}
    shap_data = get_shap_values(rf_model, X_test[:100], feature_names, n_samples=100)
    if shap_data:
        explain_data["shap_importance"] = shap_data["shap_importance"]
        print(f"   SHAP computed")
    if len(fi) > 0:
        with open("results/metrics/feature_importance.json", "w") as f:
            json.dump(explain_data, f, indent=2)
        print(f"   Top: {fi.iloc[0]['feature']}")

    with open("results/metrics/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS (§3.7)")
    print("=" * 60)
    for name, m in results.items():
        print(f"  {name:22s} AUC-ROC: {m['auc_roc']:.3f}  AUC-PR: {m['auc_pr']:.3f}  F1: {m['f1']:.3f}  Prec: {m['precision']:.3f}  Rec: {m['recall']:.3f}")
    print(f"\nModels -> {out_dir} | results/metrics/")


if __name__ == "__main__":
    main()
