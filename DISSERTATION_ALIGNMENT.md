# Dissertation Requirement Alignment

Reference: `MTech_Dissertation_Chapters_1-3_HARVARD_FINAL.md`

## Research Objectives (§1.3)

| Objective | Implementation |
|-----------|-----------------|
| 1. Identify and integrate alternative data sources (mobile money, utility bills) | `src/data/zimbabwe_synthetic.py`: 45 features — mobile money, utility payments, demographics per §3.2 |
| 2. Develop credit scoring model incorporating LoRA for efficient parameter customisation | `src/models/lora_model.py`: DistilBERT + LoRA; ~2–8% trainable params |
| 3. Compare LoRA-based model with traditional and baseline ML models | `scripts/train.py`: LR, RF, XGBoost baselines; LoRA comparison; fairness (§3.7), explainability (§3.7) |
| 4. Examine LoRA’s contribution to financial inclusion under the NDS | Policy tab; fairness by gender, location, age, MSME; NDS1/NDS2 alignment; RBZ guidelines |

## Methodology (§3)

### §3.2 Data Collection
| Requirement | Implementation |
|-------------|-----------------|
| 50,000 observations | `generate_zimbabwe_alternative_data(n_samples=50000)` |
| 45 features | 45 columns: demographic, mm, utility, behavioral, derived |
| Default rate 8–12% | `default_rate=0.10` |
| 70/15/15 split | `DataPreprocessor(test_size=0.15, val_size=0.15)` |
| Missing data 5–15% | `missingness_rate=0.08` in generator |
| Demographics (Zimbabwe) | gender, age, location, education, household, employment, sector, MSME |

### §3.3 Preprocessing
| Requirement | Implementation |
|-------------|-----------------|
| Missing: imputation | `SimpleImputer(strategy="median")` |
| Scaling: z-score | `StandardScaler()` |
| Categorical encoding | `LabelEncoder` (one-hot via expansion) |
| Feature engineering | txn_cv, payment_trend, consecutive_on_time in generator |

### §3.4 Model Architecture
| Requirement | Implementation |
|-------------|-----------------|
| Transformer + LoRA | DistilBERT (Hugging Face) + PEFT LoRA |
| LoRA on Q, V projections | `target_modules=["q_lin","v_lin"]` |
| Baselines: LR, RF | `src/models/baseline.py` |
| Full fine-tuning | Omitted (DistilBERT full ft heavy); LoRA achieves §3.5 efficiency |

### §3.5 LoRA
| Requirement | Implementation |
|-------------|-----------------|
| Rank r=8 | `lora_rank=8` |
| Alpha=16 | `lora_alpha=16` |
| A: random init, B: zero | PEFT default |
| Scaling α/r | PEFT internal |

### §3.6 Training
| Requirement | Implementation |
|-------------|-----------------|
| BCE with class weighting | `CrossEntropyLoss(weight=...)` in `LoRATrainer` |
| AdamW | `torch.optim.AdamW` |
| Early stopping | `patience=5` |
| Dropout 0.1 | In LoRACreditScorer |

### §3.7 Evaluation Metrics
| Requirement | Implementation |
|-------------|-----------------|
| Accuracy | ✓ |
| Precision | ✓ |
| Recall | ✓ |
| F1 | ✓ |
| AUC-ROC | ✓ |
| AUC-PR | ✓ |
| Trainable params | ✓ (LoRA) |
| Training time | ✓ |
| Demographic parity | ✓ |
| Equal opportunity | ✓ |
| Equalized odds | ✓ |
| Feature importance | ✓ |
| SHAP | ✓ (TreeExplainer for RF) |

### §3.8 Fairness & Interpretability
| Requirement | Implementation |
|-------------|-----------------|
| Subgroup analysis (gender, age, location) | `compute_fairness_metrics` |
| Statistical disparity | demographic_parity, equal_opportunity, equalized_odds |
| Feature importance | RF `feature_importances_` |
| SHAP (local) | `get_shap_values` |

## Regulatory Alignment

- **Data Protection Act (2021)**: Synthetic data, no PII
- **RBZ responsible AI (2024)**: Fairness testing, interpretability
- **NDS1/NDS2**: Financial inclusion, MSME, women, youth, rural
