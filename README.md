# LoRA Credit Scoring

**MSc Software Engineering Dissertation**  
*Low-Rank Adaptation for Alternative Data Credit Scoring: Enhancing Financial Inclusion under the National Development Strategy*

Harare Institute of Technology · Supervisor: Eng A. Ndlovu

---

## 1. Steps to Run the Project

Two ways to run the system:

| Approach | Best for | What you get |
|----------|----------|--------------|
| **[Docker](#docker-recommended)** | Panel demo, Windows, minimal setup | Both frontends + API in one command |
| **[Local (venv)](#local-setup-venv)** | Development, training tweaks | Full control; manual terminal setup |

---

### Docker (Recommended)

Runs **both frontends** and the MFI API without installing Python or Node locally.

#### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) or Docker Engine + Compose (Linux)
- ~8 GB free disk (images include PyTorch)
- Internet on first build (downloads base images and DistilBERT weights during training)
- **Start Docker Desktop** before running any `docker compose` command

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  docker compose up -d --build                                   │
├─────────────────┬─────────────────────┬─────────────────────────┤
│  app            │  mfi-frontend       │  mfi-api                │
│  Gradio demo    │  React + nginx      │  FastAPI + LoRA scoring │
│  :7860          │  :5174              │  :8000                  │
│                 │  /api ──proxy──►    │  SQLite in mfi_portal/  │
└────────┬────────┴──────────┬──────────┴──────────┬──────────────┘
         │                   │                      │
         └───────────────────┴──────────────────────┘
                    shared host volumes
              ./models  ./results  ./data  ./mfi_portal/data
```

| Service | Container | Host URL | Role |
|---------|-----------|----------|------|
| `app` | Gradio | http://localhost:7860 | Dissertation demo — fairness, explainability, live scoring |
| `mfi-frontend` | nginx + React | http://localhost:5174 | MFI office portal — applicants, score inquiries, reports |
| `mfi-api` | FastAPI | http://localhost:8000/docs | REST API for the portal; loads LoRA from `./models` |
| `train` | one-shot job | — | Trains models; not started by default (uses `--profile train`) |

#### First-time setup

From the project root (`lora/`):

```bash
# 1. Train models — full methodology (50K per §3.3.3)
docker compose --profile train run --rm train

# Or quick demo training (5K, faster)
docker compose --profile train-quick run --rm train-quick

# 2. Build images and start all three services
docker compose up -d --build
```

**Windows (PowerShell)** — same commands; ensure Docker Desktop is running.

*Training takes ~5–15 minutes on CPU depending on hardware. The first `docker compose build` may take 10–20 minutes while PyTorch and dependencies download.*

#### Verify services are up

```bash
docker compose ps
```

All three services should show `running` (and `healthy` once ready). Then open:

| URL | Login / notes |
|-----|----------------|
| http://localhost:7860 | Gradio demo — no login |
| http://localhost:5174 | MFI portal — `admin` / `admin123` or `officer` / `officer123` |
| http://localhost:8000/docs | Swagger API reference |

#### Day-to-day commands

```bash
# Start (after first build)
docker compose up -d

# Stop all services
docker compose down

# View logs (all services, follow mode)
docker compose logs -f

# Logs for one service
docker compose logs -f app
docker compose logs -f mfi-api
docker compose logs -f mfi-frontend

# Rebuild after code changes
docker compose up -d --build

# Rebuild a single service
docker compose up -d --build mfi-frontend
```

#### Start individual services

```bash
docker compose up -d app                    # Gradio demo only
docker compose up -d mfi-api mfi-frontend # MFI portal only (API + UI)
```

The MFI frontend depends on `mfi-api` being healthy; start the API first if running them separately.

#### Retrain models (Docker)

```bash
docker compose --profile train run --rm train

# Optional: override training args (50K samples, methodology run)
docker compose --profile train run --rm train \
  python scripts/train.py --dataset zimbabwe_synthetic --n-samples 50000 --epochs 15
```

Restart services after retraining so they pick up new weights:

```bash
docker compose restart app mfi-api
```

#### Data persistence

Docker mounts host folders into containers. Nothing is lost when you `docker compose down`:

| Host path | Used by | Contents |
|-----------|---------|----------|
| `./models/` | `app`, `mfi-api`, `train` | LoRA weights, preprocessor |
| `./results/` | `app`, `mfi-api`, `train` | Metrics, fairness, feature importance |
| `./data/` | `app`, `train` | Generated datasets |
| `./mfi_portal/data/` | `mfi-api` | SQLite DB (`mfi.db`), applicants, score history |

#### Panel demo flow (both UIs)

1. **Gradio** (http://localhost:7860) — walk through Overview → Live Demo → Results → Fairness → Explainability (see [§2 Demo Guide](#2-demo-guide--meeting-each-objective)).
2. **MFI portal** (http://localhost:5174) — log in as officer, add an applicant, run a score inquiry, show dashboard and reports.

---

### Local setup (venv)

#### Clean Start (from scratch)

*Linux/macOS only* — on Windows, follow Steps 1–4 below manually (remove `venv`, `models/`, and `results/` if resetting).

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

- Python 3.10 or higher ([python.org](https://www.python.org/downloads/); on Windows, enable **Add python.exe to PATH** during install)
- ~4 GB disk space
- Internet connection (first run: download DistilBERT)

> **Windows note:** `python3` often points to a Microsoft Store placeholder and fails with *“Python was not found”*. Use `python` or `py` instead (see Step 1). To disable the stub: **Settings → Apps → Advanced app settings → App execution aliases** — turn off `python.exe` and `python3.exe`.

### Step 1: Create Virtual Environment

**Linux / macOS**

```bash
cd /path/to/lora-project

# Create a fresh venv (if broken, remove first: rm -rf venv)
python3 -m venv venv

# Activate
source venv/bin/activate
```

**Windows (PowerShell)**

```powershell
cd C:\path\to\lora-project

# Create a fresh venv (if broken, remove first: Remove-Item -Recurse -Force venv)
python -m venv venv
# Or pin a version: py -3.11 -m venv venv

# Activate
.\venv\Scripts\Activate.ps1
```

If activation is blocked by execution policy, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Windows (Command Prompt)**

```cmd
cd C:\path\to\lora-project
python -m venv venv
venv\Scripts\activate.bat
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

*Expected time: 3–8 minutes depending on connection.*

### Step 3: Train Models

```bash
# Quick run (5,000 samples, ~5–10 min on CPU)
python scripts/train.py --dataset zimbabwe_synthetic --n-samples 5000 --epochs 10 --export-figures

# Full run per methodology (50,000 samples, ~20–40 min)
python scripts/train.py --dataset zimbabwe_synthetic --n-samples 50000 --epochs 15 --export-figures
```

**Output:** Models saved to `models/`, metrics to `results/metrics/`.

### Step 4: Launch the Demo UI

```bash
python -m app.main
```

**Open http://localhost:7860** in your browser.

For the MFI office portal without Docker, see [Step 5](#step-5-optional-mfi-portal--office-ui) or `mfi_portal/README.md`.

### Step 5 (Optional): MFI Portal — Office UI

For microfinance staff to check credit scores in the office:

```bash
# Activate venv first (Linux/macOS: source venv/bin/activate)
# Windows PowerShell: .\venv\Scripts\Activate.ps1

# Install MFI portal deps
pip install sqlalchemy python-jose[cryptography] passlib[bcrypt]

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

Use this flow when presenting to the dissertation panel. Each tab maps to the five objectives in **MTECH Software Engineering Project Documentation** §1.5.

| Order | Tab | What to do | Objective(s) |
|-------|-----|------------|--------------|
| 1 | **Overview** | State the problem (89% credit excluded), five objectives, RQ1–RQ4 | All (context) |
| 2 | **Dataset** | Show 50K synthetic records; mobile money, utilities, digital commerce | **1** |
| 3 | **Live Demo** | Filter applicant, review profile, run LoRA score | **2** |
| 4 | **Results** | Compare LoRA AUC vs LR/RF/XGB baselines | **2**, **3** |
| 5 | **Efficiency** | Trainable params (~2–8%), inference latency, memory | **3** |
| 6 | **Fairness** | Performance by gender, location, age, income quartile, MSME | **4** |
| 7 | **Explainability** | LoRA per-applicant drivers + global attribution | **2** |
| 8 | **Policy** | NDS1/NDS2 and National AI Strategy recommendations | **5** |
| 9 | **MFI Portal** | http://localhost:5174 — deployment (Appendix C.4) | **5** |

See **`DISSERTATION_ALIGNMENT.md`** for the full chapter mapping.

### Suggested Script (30–60 seconds per tab)

1. **Overview:** *"89% of Zimbabweans lack formal credit. We use LoRA with alternative data—mobile money, utilities—to score credit for the unbanked."*
2. **Live Demo:** *"Here’s a rural MSME applicant — female, EcoCash user, mixed utility history. After reviewing her profile, the model predicts a score of X with low/medium/high risk. Top drivers are utility consistency and transaction frequency."*
3. **Results:** *"LoRA matches or beats logistic regression and random forest. Only ~3% of parameters are trained—ideal for resource-limited settings."*
4. **Fairness:** *"Performance is balanced across gender, location, age, and MSME status, in line with RBZ fairness expectations."*
5. **Explainability:** *"Feature importance and SHAP give transparent explanations for each decision, supporting regulatory review."*
6. **Dataset:** *"We use synthetic Zimbabwe data with mobile money and utility payments, aligned with our methodology."*
7. **Policy:** *"The Policy tab outlines NDS-aligned recommendations for policymakers, banks, and tech providers."*

---

## 3. Objectives → Where to Demonstrate

| # | Research Objective (§1.5) | Where to demo |
|---|---------------------------|---------------|
| **1** | Synthetic alternative data framework | **Dataset** tab |
| **2** | LoRA-enhanced credit scoring model | **Live Demo**, **Results**, **Explainability** |
| **3** | Computational efficiency | **Efficiency**, **Results** |
| **4** | Fairness across demographic dimensions | **Fairness** tab |
| **5** | NDS / National AI Strategy alignment | **Policy**, **MFI Portal** |

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
├── README.md                      # Setup and demo guide
├── DISSERTATION_ALIGNMENT.md      # Objectives ↔ code mapping (§1.5)
├── MTECH Software Engineering Project Documentation.docx
├── docker-compose.yml             # app + mfi-api + mfi-frontend + train profiles
├── Dockerfile                     # Gradio demo image
├── app/main.py                    # Demo UI (8 tabs aligned to 5 objectives)
├── scripts/
│   ├── train.py                   # Training pipeline (§3 methodology)
│   └── export_results.py          # Dissertation figures → results/figures/
├── mfi_portal/
│   ├── backend/                   # FastAPI (Dockerfile)
│   ├── frontend/                  # React + Vite (Dockerfile, nginx.conf)
│   └── data/                      # SQLite DB (runtime, Docker volume)
├── src/
│   ├── data/                      # Loader, preprocessor, zimbabwe_synthetic
│   ├── models/                    # LR, RF, XGB, LoRA
│   ├── training/                  # LoRA trainer
│   ├── evaluation/                # Fairness, explainability
│   └── scoring/                   # Shared inference (Gradio + MFI)
├── models/                        # Trained weights (generated)
├── results/metrics/               # training_results, fairness_results, feature_importance
├── results/figures/               # PNG charts for Chapter 4 (generated)
└── requirements.txt
```

---

## 6. Troubleshooting

### Local (venv)

| Issue | Fix |
|-------|-----|
| `Python was not found` (Windows) | Use `python -m venv venv` or `py -3.11 -m venv venv`, not `python3`. Install from [python.org](https://www.python.org/downloads/) if `python` is also missing. |
| `venv/bin/pip: cannot execute` (Linux/macOS) | Recreate venv: `rm -rf venv && python3 -m venv venv` |
| Broken venv (Windows) | `Remove-Item -Recurse -Force venv` then `python -m venv venv` |
| `Activate.ps1` blocked (Windows) | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again |
| `ModuleNotFoundError: gradio` | `pip install -r requirements.txt` (with venv active) |
| `ImportError: get_shap_values` | Ensure latest `src/evaluation/__init__.py` exports `get_shap_values` |
| UI shows "Train first" | Run `python scripts/train.py --dataset zimbabwe_synthetic` or see Docker training below |
| Port 7860 in use | `python -m app.main --server-port 7861` |

### Docker

| Issue | Fix |
|-------|-----|
| `failed to connect to the docker API` / `dockerDesktopLinuxEngine` | Start **Docker Desktop** and wait until it reports "Running", then retry |
| `docker compose` not found | Install Docker Desktop (includes Compose v2) or use `docker-compose` on older installs |
| Gradio shows "Train first" | Run `docker compose --profile train run --rm train`, then `docker compose restart app` |
| Score always ~570–580 for every applicant | Old model/preprocessor mismatch. **Retrain:** `python scripts/train.py --dataset zimbabwe_synthetic` then restart the app. Scores should spread across ~400–800. |
| `mfi-frontend` stuck on "Starting" | Check API health: `docker compose logs mfi-api`. API must pass healthcheck before UI starts |
| Port already in use (7860, 5174, 8000) | Stop conflicting process or change the host port in `docker-compose.yml` (e.g. `"7861:7860"`) |
| Slow first build | Normal — PyTorch and transformers are large. Subsequent builds use cache |
| Changes not reflected | Rebuild: `docker compose up -d --build` (or rebuild the specific service) |
| Reset MFI applicant data | Stop containers, delete `mfi_portal/data/mfi.db`, run `docker compose up -d mfi-api` (re-seeds admin users) |
| View container errors | `docker compose logs -f <service>` where `<service>` is `app`, `mfi-api`, or `mfi-frontend` |

---

## References

- Hu et al. (2021) *LoRA: Low-Rank Adaptation of Large Language Models*
- Zimbabwe National Development Strategy (NDS1, NDS2)
- Methodology: `MTECH Software Engineering Project Documentation.docx` §3
- Alignment guide: `DISSERTATION_ALIGNMENT.md`
