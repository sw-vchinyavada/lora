"""SQLAlchemy models for MFI portal."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, JSON
from sqlalchemy.orm import relationship

from .database import Base


class User(Base):
    """MFI staff user for login."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    email = Column(String(128), unique=True, index=True)
    hashed_password = Column(String(128), nullable=False)
    full_name = Column(String(128))
    role = Column(String(32), default="officer")  # officer, admin, manager
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)


class Applicant(Base):
    """Credit applicant / customer record."""
    __tablename__ = "applicants"

    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(String(32), unique=True, index=True, nullable=False)  # MFI reference
    national_id = Column(String(64), index=True)  # Anonymised or hash in production
    full_name = Column(String(128))
    gender = Column(String(16))
    age = Column(Integer)
    location = Column(String(64))  # urban, rural, peri_urban
    employment = Column(String(32))
    sector = Column(String(64))
    msme = Column(Integer)  # 0 or 1
    education = Column(String(32))  # none, primary, secondary, tertiary
    region = Column(String(64))  # Harare, Bulawayo, Midlands, etc.
    # Mobile money (EcoCash, OneMoney)
    mm_provider = Column(String(32))
    mm_txn_freq = Column(Integer)
    mm_txn_per_month = Column(Integer)
    mm_weekend_usage_pct = Column(Float)
    mm_avg_amount_usd = Column(Float)
    mm_balance_volatility = Column(Float)
    mm_tenure_months = Column(Integer)
    mm_p2p_ratio = Column(Float)
    mm_bill_payment_ratio = Column(Float)
    mm_merchant_ratio = Column(Float)
    # Telecom (airtime, data bundles)
    airtime_topups_per_month = Column(Integer)
    airtime_avg_amount_usd = Column(Float)
    airtime_consistency_score = Column(Float)
    data_bundles_per_month = Column(Integer)
    data_avg_bundle_usd = Column(Float)
    # Utility (ZESA, water, telecom)
    utility_payment_rate = Column(Float)
    zesa_type = Column(String(16))  # prepaid, postpaid
    util_electricity_consistency = Column(Float)
    util_water_consistency = Column(Float)
    util_telecom_consistency = Column(Float)
    util_overdue_count = Column(Integer)
    util_avg_delay_days = Column(Float)
    util_multi_service_consistency = Column(Float)
    consecutive_on_time = Column(Integer)
    # Digital footprint
    has_smartphone = Column(Integer)  # 0 or 1
    social_media_usage = Column(Float)
    # Behavioral
    account_age_months = Column(Integer)
    digital_engagement = Column(Float)
    channel_diversity = Column(Integer)
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"))

    score_history = relationship("ScoreInquiry", back_populates="applicant", order_by="desc(ScoreInquiry.created_at)")


class ScoreInquiry(Base):
    """Record of each credit score check."""
    __tablename__ = "score_inquiries"

    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(Integer, ForeignKey("applicants.id"), nullable=False)
    score = Column(Integer)  # 300–850
    default_probability = Column(Float)
    risk_band = Column(String(16))  # Low, Medium, High
    top_drivers = Column(JSON)  # [{"feature": "...", "importance": ...}]
    model_used = Column(String(32), default="lora")
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(Integer, ForeignKey("users.id"))
    notes = Column(Text)

    applicant = relationship("Applicant", back_populates="score_history")


class AuditLog(Base):
    """Audit trail for compliance."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(64))  # login, score_inquiry, applicant_create, report_export
    entity_type = Column(String(32))
    entity_id = Column(String(64))
    details = Column(JSON)
    ip_address = Column(String(45))
    created_at = Column(DateTime, default=datetime.utcnow)
