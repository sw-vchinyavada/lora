#!/usr/bin/env python3
"""Quick check that credit scores vary across applicants."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring.inference import (
    applicant_dict_to_dataframe,
    load_scoring_artifacts,
    score_applicant_row,
    score_features,
)


def main() -> int:
    artifacts = load_scoring_artifacts("models")
    if artifacts is None:
        print("FAIL: models/lora/best_model.pt or preprocessor.pkl missing")
        return 1

    lora, prep, df = artifacts
    print(f"Model features: {lora.num_features} | dataset rows: {len(df):,}")

    sample_idxs = [0, 1, 100, 500, 1000, 2500, min(49999, len(df) - 1)]
    scores = []
    print("\nDataset applicants:")
    for idx in sample_idxs:
        if idx >= len(df):
            continue
        r = score_applicant_row(idx, artifacts=artifacts)
        scores.append(r["score"])
        print(f"  idx={idx:5d}  score={r['score']:3d}  proba={r['default_probability']:.4f}")

    profiles = [
        {"gender": "female", "age": 22, "location": "rural", "mm_txn_per_month": 5,
         "utility_payment_rate": 0.4, "util_overdue_count": 3},
        {"gender": "male", "age": 48, "location": "urban", "mm_txn_per_month": 80,
         "utility_payment_rate": 0.95, "util_overdue_count": 0, "msme": 1},
        {"gender": "female", "age": 35, "location": "urban", "mm_txn_per_month": 45,
         "utility_payment_rate": 0.85, "digital_engagement": 0.9},
    ]
    print("\nMFI portal-style applicants:")
    for p in profiles:
        r = score_features(applicant_dict_to_dataframe(p), artifacts=artifacts, explain=False)
        scores.append(r["score"])
        print(f"  {p['gender']:6s} age={p['age']}  score={r['score']:3d}  proba={r['default_probability']:.4f}")

    unique = len(set(scores))
    print(f"\nUnique scores in sample: {unique}/{len(scores)}")
    if unique > 1:
        print("PASS: scores differ across applicants")
        return 0
    print("FAIL: all sampled scores are identical")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
