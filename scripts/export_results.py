#!/usr/bin/env python3
"""
Export dissertation-ready figures from training metrics.

Generates PNG charts referenced in Chapter 4 and Appendix D.
Run after training: python scripts/export_results.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd


def _load_json(name: str):
    path = Path("results/metrics") / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def export_model_comparison(out_dir: Path):
    data = _load_json("training_results.json")
    if not data:
        return
    rows = [
        {
            "Model": k.replace("_", " ").title(),
            "AUC-ROC": v["auc_roc"],
            "AUC-PR": v.get("auc_pr", 0),
            "F1": v["f1"],
        }
        for k, v in data.items()
    ]
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    width = 0.25
    ax.bar([i - width for i in x], df["AUC-ROC"], width, label="AUC-ROC", color="#0ea5e9")
    ax.bar(x, df["AUC-PR"], width, label="AUC-PR", color="#10b981")
    ax.bar([i + width for i in x], df["F1"], width, label="F1", color="#f59e0b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison — Objective 2 & 3 (§4.4.4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison.png", dpi=150)
    plt.close(fig)


def export_efficiency(out_dir: Path):
    data = _load_json("training_results.json")
    if not data or "lora" not in data:
        return
    lora = data["lora"]
    labels = ["Trainable params (M)", "Train time (min)", "Inference (ms)", "Peak memory (MB)"]
    values = [
        lora.get("trainable_params", 0) / 1e6,
        lora.get("train_time_sec", 0) / 60,
        lora.get("inference_latency_ms", 0),
        lora.get("peak_memory_mb", 0),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values, color=["#8b5cf6", "#0ea5e9", "#10b981", "#f59e0b"])
    ax.set_title("LoRA Computational Efficiency — Objective 3 (§4.4.6)")
    fig.tight_layout()
    fig.savefig(out_dir / "lora_efficiency.png", dpi=150)
    plt.close(fig)


def export_fairness(out_dir: Path):
    data = _load_json("fairness_results.json")
    if not data:
        return
    lora = data.get("lora", {})
    groups = lora.get("groups", {})
    rows = []
    for attr, gdata in groups.items():
        for gname, m in gdata.items():
            rows.append({"Attribute": attr, "Group": str(gname), "AUC": m.get("auc", 0)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5))
    for attr in df["Attribute"].unique():
        sub = df[df["Attribute"] == attr]
        ax.plot(sub["Group"], sub["AUC"], marker="o", label=attr)
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Fairness by Demographic Group — Objective 4 (§4.4.8)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fairness_by_group.png", dpi=150)
    plt.close(fig)


def export_explainability(out_dir: Path):
    data = _load_json("feature_importance.json")
    if not data:
        return
    lora = data.get("lora_global_attribution", [])
    if not lora:
        lora = data.get("feature_importance", [])
    if not lora:
        return
    df = pd.DataFrame(lora).head(12)
    y_col = "mean_abs_impact" if "mean_abs_impact" in df.columns else "importance"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["feature"], df[y_col], color="#0369a1")
    ax.invert_yaxis()
    ax.set_title("LoRA Feature Attribution — Objective 2 (§4.4.9)")
    fig.tight_layout()
    fig.savefig(out_dir / "lora_feature_attribution.png", dpi=150)
    plt.close(fig)


def export_all():
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    export_model_comparison(out_dir)
    export_efficiency(out_dir)
    export_fairness(out_dir)
    export_explainability(out_dir)
    print(f"Exported figures to {out_dir}/")


if __name__ == "__main__":
    export_all()
