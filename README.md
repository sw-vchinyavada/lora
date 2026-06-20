# LoRA Credit Scoring

**MSc Software Engineering Dissertation**  
*Low-Rank Adaptation for Alternative Data Credit Scoring: Enhancing Financial Inclusion under the National Development Strategy*

Harare Institute of Technology · Supervisor: Eng A. Ndlovu

---

## 1. Steps to Run the Project

### Clean Start (from scratch)

```bash
cd /path/to/lora-project

# Full reset: stop processes, clear models/results/data, reinstall, optional train
./scripts/clean_start.sh

# Include training (5K samples, ~5–10 min)
./scripts/clean_start.sh --train

# Or: make clean  (just remove generated files)
```

Then run each system in a **separate terminal** (see steps below).

---

### Prerequisites

- Python 3.10 or higher
- ~4 GB disk space
- Internet connection (first run: download DistilBERT)

### Step 1: Create Virtual Environment

```bash
cd /path/to/lora-project

# Create a fresh venv (if existing venv is broken, remove it first: rm -rf venv)
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
# venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

*Expected time: 3–8 minutes depending on connection.*

### Step 3: Train Models

```bash
# Quick run (5,000 samples, ~5–10 min on CPU)
python scripts/train.py --dataset zimbabwe_synthetic --n-samples 5000 --epochs 10

# Full run per methodology (50,000 samples, ~20–40 min)
python scripts/train.py --dataset zimbabwe_synthetic --n-samples 50000 --epochs 15
```

**Output:** Models saved to `models/`, metrics to `results/metrics/`.

### Step 4: Launch the Demo UI

```bash
python -m app.main
```

**Open http://localhost:7860** in your browser.

### Docker (Alternative)

```bash
# Train
docker compose --profile train run --rm train

# Launch UI
docker compose up -d app
```

Open **http://localhost:7860**.

### Step 5 (Optional): MFI Portal — Office UI

For microfinance staff to check credit scores in the office:

```bash
# Activate venv first
source venv/bin/activate

# Install MFI portal deps
pip install sqlalchemy python-jose[cryptography] bcrypt

# Terminal 1: Start API (port 8000)
cd mfi_portal/backend && uvicorn app.main:app --reload

# Terminal 2 (from project root): Start frontend (port 5174)
cd mfi_portal/frontend && npm install && npm run dev
```

- **Portal**: http://localhost:5174 — Login: admin/admin123 or officer/officer123
- **API docs**: http://localhost:8000/docs

See `mfi_portal/README.md` for details.

---

## 2. Demo Guide — Meeting Each Objective

Use this flow when presenting to the dissertation panel. Each tab demonstrates one or more research objectives.

| Order | Tab | What to do | Objective(s) |
|-------|-----|------------|--------------|
| 1 | **Overview** | State the problem: 89% credit excluded. Introduce LoRA + alternative data. | All (context) |
| 2 | **Live Demo** | Pick a customer index (e.g. 100). Click **Predict**. Show score, risk, top drivers. | 1, 2 |
| 3 | **Results** | Click **View results**. Highlight LoRA AUC vs baselines, parameter efficiency (~2–8%). | 2, 3 |
| 4 | **Fairness** | Click **View fairness**. Show performance by gender, location, youth, MSME. | 4 |
| 5 | **Explainability** | Click **View explainability**. Show feature importance chart. | 2, 4 |
| 6 | **Dataset** | Click **View dataset**. Describe Zimbabwe synthetic data: mobile money, utility, demographics. | 1 |
| 7 | **Policy** | Read recommendations. Link to NDS1/NDS2. | 4 |

### Suggested Script (30–60 seconds per tab)

1. **Overview:** *"89% of Zimbabweans lack formal credit. We use LoRA with alternative data—mobile money, utilities—to score credit for the unbanked."*
2. **Live Demo:** *"Here’s customer 100: female, rural, MSME. The model predicts a score of X with low/medium/high risk. Top drivers are utility consistency and transaction frequency."*
3. **Results:** *"LoRA matches or beats logistic regression and random forest. Only ~3% of parameters are trained—ideal for resource-limited settings."*
4. **Fairness:** *"Performance is balanced across gender, location, age, and MSME status, in line with RBZ fairness expectations."*
5. **Explainability:** *"Feature importance and SHAP give transparent explanations for each decision, supporting regulatory review."*
6. **Dataset:** *"We use synthetic Zimbabwe data with mobile money and utility payments, aligned with our methodology."*
7. **Policy:** *"The Policy tab outlines NDS-aligned recommendations for policymakers, banks, and tech providers."*

---

## 3. Objectives → Where to Demonstrate

| # | Research Objective (from §1.3) | Where to demo |
|---|--------------------------------|---------------|
| **1** | Identify and integrate alternative data sources (mobile money, utility bills) | **Dataset** tab: feature list. **Live Demo**: customer profile. **Overview**: mobile money, utility, demographics. |
| **2** | Develop credit scoring model incorporating LoRA for efficient parameter customisation | **Results** tab: LoRA efficiency (~2–8% params). **Overview**: parameter reduction. **Explainability**: transparent decisions. |
| **3** | Compare LoRA-based model with traditional and baseline ML models | **Results** tab: bar chart, LR/RF/XGB vs LoRA; AUC-ROC, AUC-PR, F1. **Overview**: model comparison. |
| **4** | Examine LoRA’s contribution to financial inclusion under the NDS | **Fairness** tab: gender, location, age, MSME. **Policy** tab: NDS1/NDS2 recommendations. **Overview**: NDS alignment. |

---

## 4. Datasets

| Dataset | Samples | Use |
|---------|---------|-----|
| **zimbabwe_synthetic** (default) | 5K or 50K | Zimbabwe alternative data — mobile money, utility, demographics |
| **credit_card_default** | 30K | UCI benchmark |
| **german_credit** | 1K | UCI benchmark |

---

## 5. Project Structure

```
lora-project/
├── README.md              # This file
├── app/main.py            # Demo UI
├── scripts/train.py       # Training pipeline
├── src/
│   ├── data/              # Loader, preprocessor, zimbabwe_synthetic
│   ├── models/            # LR, RF, XGB, LoRA
│   ├── training/          # LoRA trainer
│   └── evaluation/        # Fairness, explainability
├── results/metrics/       # training_results, fairness_results, feature_importance
└── requirements.txt
```

---

## 6. Troubleshooting

| Issue | Fix |
|-------|-----|
| `venv/bin/pip: cannot execute` | Recreate venv: `rm -rf venv && python3 -m venv venv` |
| `ModuleNotFoundError: gradio` | `pip install -r requirements.txt` (with venv active) |
| `ImportError: get_shap_values` | Ensure latest `src/evaluation/__init__.py` exports `get_shap_values` |
| UI shows "Train first" | Run `python scripts/train.py --dataset zimbabwe_synthetic` |
| Port 7860 in use | `python -m app.main --server-port 7861` |

---

## References

- Hu et al. (2021) *LoRA: Low-Rank Adaptation of Large Language Models*
- Zimbabwe National Development Strategy (NDS1, NDS2)
- Methodology: `MTech_Dissertation_Chapters_1-3_HARVARD_FINAL.md` §3
