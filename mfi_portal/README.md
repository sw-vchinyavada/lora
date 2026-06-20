# MFI Credit Scoring Portal

Microfinance institution office UI for checking credit scores using the LoRA-based credit scoring model.

## Features

- **Login** — JWT authentication (admin, officer roles)
- **Applicant management** — Add, edit, search applicants
- **Credit score inquiry** — Run LoRA model on applicant data
- **Score history** — Per-applicant inquiry log
- **Reports** — Dashboard stats, activity feed, CSV export
- **Database** — SQLite (dev) / PostgreSQL (prod)

## Prerequisites

1. **Train models first** (from project root):
   ```bash
   python scripts/train.py --dataset zimbabwe_synthetic --epochs 10
   ```

2. **Install dependencies**

## Quick Start

**Use the project venv first:** `source venv/bin/activate` (from lora-project root)

### Backend (API)

```bash
cd mfi_portal/backend
pip install sqlalchemy python-jose[cryptography] passlib[bcrypt]
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend (separate terminal, from project root)

```bash
cd mfi_portal/frontend
npm install
npm run dev
```

- **API**: http://localhost:8000
- **Frontend**: http://localhost:5174
- **API docs**: http://localhost:8000/docs

### Demo credentials

| Username | Password |
|----------|----------|
| admin    | admin123 |
| officer  | officer123 |

## Project Structure

```
mfi_portal/
├── backend/           # FastAPI
│   ├── app/
│   │   ├── api/       # auth, applicants, reports
│   │   ├── core/      # config, security
│   │   ├── db/        # models, database
│   │   ├── schemas/
│   │   └── services/  # scoring (LoRA integration)
│   └── requirements.txt
├── frontend/          # React + Vite + Tailwind
│   └── src/
│       ├── components/
│       ├── context/
│       └── pages/
├── data/             # SQLite DB (created at runtime)
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
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

## Environment

| Variable | Default | Description |
|----------|---------|--------------|
| MFI_SECRET_KEY | dev-secret-... | JWT signing key |
| MFI_DATABASE_URL | sqlite:///... | Database URL |
| PORT | 8000 | API port |
