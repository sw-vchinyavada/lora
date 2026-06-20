"""Database setup."""
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app.core.config import DATABASE_URL

# Ensure data dir exists for SQLite
url = DATABASE_URL
if url.startswith("sqlite"):
    path = Path(url.replace("sqlite:///", ""))
    path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(url, connect_args={"check_same_thread": False} if "sqlite" in url else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
