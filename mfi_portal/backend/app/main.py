"""
MFI Credit Scoring Portal — REST API

Microfinance institution dashboard backend.
Run: uvicorn app.main:app --reload
"""
import sys
from pathlib import Path

# Add lora-project root for src imports (scoring service)
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import engine, Base, get_db
from app.db.models import User, Applicant, ScoreInquiry, AuditLog
from app.api import auth, applicants, reports

# Create tables
Base.metadata.create_all(bind=engine)

# Seed default user if empty
def _seed_admin():
    from app.db.database import SessionLocal
    from app.core.security import get_password_hash
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin = User(
                username="admin",
                email="admin@mfi.local",
                hashed_password=get_password_hash("admin123"),
                full_name="Administrator",
                role="admin",
            )
            db.add(admin)
            officer = User(
                username="officer",
                email="officer@mfi.local",
                hashed_password=get_password_hash("officer123"),
                full_name="Loan Officer",
                role="officer",
            )
            db.add(officer)
            db.commit()
    finally:
        db.close()

_seed_admin()

app = FastAPI(
    title="MFI Credit Scoring Portal",
    description="API for microfinance institutions to check credit scores (LoRA-based)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(applicants.router, prefix="/api")
app.include_router(reports.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "mfi-portal"}
