
# Dissertation Alignment Guide

**Project:** Low-Rank Adaptation (LoRA) for Alternative Data Credit Scoring  
**Document:** `MTECH Software Engineering Project Documentation.docx`  
**Branch:** `dissertation-final-presentation`

This guide maps the dissertation's five objectives and four research questions to the implemented system, demo tabs, and generated artifacts.

---

## Objectives → Implementation

| # | Dissertation objective (§1.5) | Code / artifact | Demo tab |
|---|------------------------------|-----------------|----------|
| **1** | Design synthetic alternative data (mobile money, utilities, digital commerce) | `src/data/zimbabwe_synthetic.py`, `data/raw/zimbabwe_alternative_data.csv` | **Dataset** |
| **2** | Develop LoRA-enhanced credit scoring model | `src/models/lora_model.py`, `src/training/trainer.py`, `src/scoring/inference.py` | **Live Demo**, **Results**, **Explainability** |
| **3** | Evaluate computational efficiency (params, time, memory, latency) | `scripts/train.py` → `training_results.json` (inference_latency_ms, peak_memory_mb) | **Efficiency**, **Results** |
| **4** | Assess fairness across gender, age, location, income, MSME | `src/evaluation/fairness.py` → `fairness_results.json` | **Fairness** |
| **5** | Analyse NDS / National AI Strategy alignment; policy recommendations | Policy tab content from §6.3; MFI portal as deployment case study | **Policy**, **MFI Portal** |

---

## Research Questions → Evidence

| RQ | Question (§1.6) | Where demonstrated |
|----|-----------------|-------------------|
| **RQ1** | How can alternative data be synthesised for unbanked populations? | Dataset tab; synthetic generator with EcoCash, ZESA, telecom, MSME features |
| **RQ2** | Does LoRA maintain performance with reduced compute? | Results tab (AUC vs baselines); Efficiency tab (trainable %, latency) |
| **RQ3** | What fairness characteristics emerge? | Fairness tab; demographic parity, equal opportunity, equalized odds |
| **RQ4** | How can the system align with national priorities? | Policy tab; MFI portal deployment (Appendix C.4) |

---

## Methodology alignment (Chapter 3)

| Dissertation section | Implementation |
|---------------------|----------------|
| §3.3.3 Dataset scale (50,000 records, 70/15/15) | Default `--n-samples 50000` in `scripts/train.py` and Docker `train` service |
| §3.5.2 LoRA rank 8 | `LoRACreditScorer(lora_rank=8)` in `src/models/lora_model.py` |
| §3.6.1 Predictive metrics | accuracy, precision, recall, F1, AUC-ROC, AUC-PR |
| §3.6.2 Efficiency metrics | trainable_params, train_time_sec, inference_latency_ms, peak_memory_mb |
| §3.6.3 Fairness dimensions | gender, location, age_group, income_quartile, msme |
| §3.6.4 Policy alignment | Policy tab + MFI portal architecture |
| §4.4.9 Explainability | LoRA input attribution, RF SHAP baseline comparison |

---

## Generated artifacts (Chapter 4)

After training:

```bash
python scripts/train.py --dataset zimbabwe_synthetic --n-samples 50000 --epochs 15 --export-figures
```

| Path | Content |
|------|---------|
| `results/metrics/training_results.json` | All model metrics + LoRA efficiency |
| `results/metrics/fairness_results.json` | Subgroup fairness (LoRA + RF) |
| `results/metrics/feature_importance.json` | RF SHAP + LoRA global attribution |
| `results/figures/model_comparison.png` | §4.4.4 bar chart |
| `results/figures/lora_efficiency.png` | §4.4.6 efficiency chart |
| `results/figures/fairness_by_group.png` | §4.4.8 fairness chart |
| `results/figures/lora_feature_attribution.png` | §4.4.9 explainability chart |

---

## Panel presentation sequence

1. **Overview** (30 s) — Problem, 5 objectives, RQ1–RQ4  
2. **Dataset** (45 s) — Objective 1: 50K synthetic records, alternative data features  
3. **Live Demo** (60 s) — Filter rural MSME applicant; run LoRA score; show per-applicant drivers  
4. **Results** (45 s) — LoRA vs LR/RF/XGB; highlight competitive AUC  
5. **Efficiency** (30 s) — ~2–8% params trained; inference latency for MFI deployment  
6. **Fairness** (45 s) — Balanced performance across subgroups  
7. **Explainability** (30 s) — LoRA attribution + regulatory transparency  
8. **Policy** (30 s) — NDS1/NDS2 recommendations  
9. **MFI Portal** (optional, 60 s) — http://localhost:5174 — operational deployment  

---

## Quick commands

```bash
# Full methodology training (50K, ~20–40 min CPU)
python scripts/train.py --dataset zimbabwe_synthetic --n-samples 50000 --epochs 15 --export-figures

# Quick demo training (5K, ~5–10 min)
python scripts/train.py --dataset zimbabwe_synthetic --n-samples 5000 --epochs 10 --export-figures

# Launch Gradio demo
python -m app.main

# Docker (train + all services)
docker compose --profile train run --rm train
docker compose up -d --build
```

---

*Harare Institute of Technology · H240799Q · Supervisor: Eng A. Ndlovu · June 2026*
