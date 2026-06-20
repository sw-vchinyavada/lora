"""Application configuration."""
import os
from pathlib import Path

# Project root (lora-project)
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # app/core -> 4 levels up
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MFI_DATA_DIR = Path(__file__).resolve().parents[3] / "data"  # mfi_portal/data
MFI_DATA_DIR.mkdir(parents=True, exist_ok=True)

SECRET_KEY = os.getenv("MFI_SECRET_KEY", "dev-secret-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8 hours

DATABASE_URL = os.getenv("MFI_DATABASE_URL", f"sqlite:///{MFI_DATA_DIR / 'mfi.db'}")
