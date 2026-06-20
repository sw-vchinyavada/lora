# MFI Credit Scoring Portal

Microfinance institution office UI for checking credit scores using the LoRA-based credit scoring model.

## Features

- **Login** — JWT authentication (admin, officer roles)
- **Applicant management** — Add, edit, search applicants
- **Credit score inquiry** — Run LoRA model on applicant data
- **Score history** — Per-applicant inquiry log
- **Reports** — Dashboard stats, activity feed, CSV export
- **Database** — SQLite (dev) / PostgreSQL (prod)

---

## Quick Start with Docker

The MFI portal is designed to run alongside the Gradio dissertation demo via Docker Compose at the **project root** (`lora/`).

### Prerequisites

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
2. Trained models in `./models/` (see training step below)

### Commands

```bash
cd /path/to/lora

# First time: train models (required for live scoring)
docker compose --profile train run --rm train

# Start Gradio demo + MFI API + MFI frontend
docker compose up -d --build
```

### URLs

| UI | URL | Credentials |
|----|-----|-------------|
| **MFI portal** | http://localhost:5174 | `admin` / `admin123` or `officer` / `officer123` |
| **API (Swagger)** | http://localhost:8000/docs | Use login endpoint to obtain JWT |
| **Gradio demo** | http://localhost:7860 | No login (separate `app` service) |

### How it works in Docker

```
Browser → http://localhost:5174 (mfi-frontend, nginx)
              │
              └─ /api/* proxied to → mfi-api:8000 (FastAPI)
                                           │
                                           ├─ reads ./models/lora/best_model.pt
                                           ├─ reads ./results/metrics/feature_importance.json
                                           └─ SQLite at ./mfi_portal/data/mfi.db
```

- **`mfi-frontend`** — production build of the React app; nginx serves static files and proxies `/api` to the backend container.
- **`mfi-api`** — FastAPI; shares the same LoRA weights and preprocessor as the Gradio demo via mounted `./models` and `./results`.
- **`app`** — independent Gradio service for the research demo (fairness, explainability, etc.).

### MFI-only (without Gradio)

```bash
docker compose up -d mfi-api mfi-frontend
```

### Useful commands

```bash
docker compose ps                    # service status
docker compose logs -f mfi-api       # API logs
docker compose logs -f mfi-frontend  # frontend/nginx logs
docker compose restart mfi-api       # after retraining models
docker compose down                  # stop all services
```

### Retrain and refresh scores

```bash
docker compose --profile train run --rm train
docker compose restart mfi-api
```

Scoring requires `models/lora/best_model.pt` and `models/preprocessor.pkl` on the host.

---

## Local Development (without Docker)

### Prerequisites

1. **Train models first** (from project root):
   ```bash
   python scripts/train.py --dataset zimbabwe_synthetic --epochs 10
   ```

2. **Project venv** with main `requirements.txt` installed, plus MFI backend deps.

### Activate venv

From lora-project root:

- Linux/macOS: `source venv/bin/activate`
- Windows PowerShell: `.\venv\Scripts\Activate.ps1`
- Windows CMD: `venv\Scripts\activate.bat`

### Backend (API) — Terminal 1

```bash
cd mfi_portal/backend
pip install sqlalchemy python-jose[cryptography] passlib[bcrypt]
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend — Terminal 2

From project root:

```bash
cd mfi_portal/frontend
npm install
npm run dev
```

Vite dev server runs on http://localhost:5174 and proxies `/api` to http://127.0.0.1:8000 (see `vite.config.js`).

### Demo credentials

| Username | Password |
|----------|----------|
| admin    | admin123 |
| officer  | officer123 |

---

## Project Structure

```
mfi_portal/
├── backend/
│   ├── Dockerfile         # Built from project root context
│   ├── app/
│   │   ├── api/           # auth, applicants, reports
│   │   ├── core/          # config, security
│   │   ├── db/            # models, database
│   │   ├── schemas/
│   │   └── services/      # scoring (LoRA integration)
│   └── requirements.txt
├── frontend/
│   ├── Dockerfile         # Node build + nginx
│   ├── nginx.conf         # SPA + /api proxy (Docker)
│   ├── vite.config.js     # dev proxy (local)
│   └── src/
│       ├── components/
│       ├── context/
│       └── pages/
├── data/                  # SQLite DB (created at runtime; Docker volume)
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | /api/health | Health check (used by Docker) |
| POST   | /api/auth/login | Login (JSON) |
| POST   | /api/auth/token | OAuth2 form (Swagger) |
| GET    | /api/applicants | List applicants |
| POST   | /api/applicants | Create applicant |
| GET    | /api/applicants/{id} | Get applicant |
| PATCH  | /api/applicants/{id} | Update applicant |
| POST   | /api/applicants/{id}/score | Run credit score |
| GET    | /api/applicants/{id}/history | Score history |
| GET    | /api/reports/dashboard | Dashboard stats |
| GET    | /api/reports/activity | Activity log |
| GET    | /api/reports/export/csv | Export CSV |

---

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| MFI_SECRET_KEY | dev-secret-... | JWT signing key |
| MFI_DATABASE_URL | sqlite:///.../mfi_portal/data/mfi.db | Database URL |
| PORT | 8000 | API port (local uvicorn) |

In Docker, `MFI_DATABASE_URL` is set in `docker-compose.yml` to `sqlite:////app/mfi_portal/data/mfi.db`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Login works but score fails | Train models; check `models/lora/best_model.pt` exists |
| 502 / network error on portal | Ensure `mfi-api` is running: `docker compose ps` and `docker compose logs mfi-api` |
| Empty applicant list after reset | Expected if `mfi.db` was deleted; create new applicants in the UI |
| Local dev: API connection refused | Start backend on port 8000 before the Vite dev server |
| CORS errors (local only) | Backend allows all origins; use Vite proxy (`/api`) rather than calling `:8000` directly from the browser |

See also the main [README.md](../README.md) Docker section for shared issues (Docker Desktop, ports, training).
