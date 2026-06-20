"""Applicant schemas."""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class ApplicantBase(BaseModel):
    applicant_id: str
    national_id: Optional[str] = None
    full_name: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    location: Optional[str] = None
    employment: Optional[str] = None
    sector: Optional[str] = None
    msme: Optional[int] = None
    education: Optional[str] = None
    region: Optional[str] = None
    mm_provider: Optional[str] = None
    mm_txn_freq: Optional[int] = None
    mm_txn_per_month: Optional[int] = None
    mm_avg_amount_usd: Optional[float] = None
    mm_balance_volatility: Optional[float] = None
    mm_tenure_months: Optional[int] = None
    mm_weekend_usage_pct: Optional[float] = None
    mm_p2p_ratio: Optional[float] = None
    mm_bill_payment_ratio: Optional[float] = None
    mm_merchant_ratio: Optional[float] = None
    airtime_topups_per_month: Optional[int] = None
    airtime_avg_amount_usd: Optional[float] = None
    airtime_consistency_score: Optional[float] = None
    data_bundles_per_month: Optional[int] = None
    data_avg_bundle_usd: Optional[float] = None
    utility_payment_rate: Optional[float] = None
    zesa_type: Optional[str] = None
    has_smartphone: Optional[int] = None
    social_media_usage: Optional[float] = None
    util_electricity_consistency: Optional[float] = None
    util_water_consistency: Optional[float] = None
    util_telecom_consistency: Optional[float] = None
    util_overdue_count: Optional[int] = None
    util_avg_delay_days: Optional[float] = None
    util_multi_service_consistency: Optional[float] = None
    consecutive_on_time: Optional[int] = None
    account_age_months: Optional[int] = None
    digital_engagement: Optional[float] = None
    channel_diversity: Optional[int] = None


class ApplicantCreate(ApplicantBase):
    pass


class ApplicantUpdate(BaseModel):
    full_name: Optional[str] = None
    national_id: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    location: Optional[str] = None
    employment: Optional[str] = None
    sector: Optional[str] = None
    msme: Optional[int] = None
    education: Optional[str] = None
    region: Optional[str] = None
    mm_provider: Optional[str] = None
    mm_txn_freq: Optional[int] = None
    mm_txn_per_month: Optional[int] = None
    mm_avg_amount_usd: Optional[float] = None
    mm_balance_volatility: Optional[float] = None
    mm_tenure_months: Optional[int] = None
    mm_weekend_usage_pct: Optional[float] = None
    mm_p2p_ratio: Optional[float] = None
    mm_bill_payment_ratio: Optional[float] = None
    mm_merchant_ratio: Optional[float] = None
    airtime_topups_per_month: Optional[int] = None
    airtime_avg_amount_usd: Optional[float] = None
    airtime_consistency_score: Optional[float] = None
    data_bundles_per_month: Optional[int] = None
    data_avg_bundle_usd: Optional[float] = None
    utility_payment_rate: Optional[float] = None
    zesa_type: Optional[str] = None
    has_smartphone: Optional[int] = None
    social_media_usage: Optional[float] = None
    util_electricity_consistency: Optional[float] = None
    util_water_consistency: Optional[float] = None
    util_telecom_consistency: Optional[float] = None
    util_overdue_count: Optional[int] = None
    util_avg_delay_days: Optional[float] = None
    util_multi_service_consistency: Optional[float] = None
    consecutive_on_time: Optional[int] = None
    account_age_months: Optional[int] = None
    digital_engagement: Optional[float] = None
    channel_diversity: Optional[int] = None


class ApplicantResponse(ApplicantBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ScoreResult(BaseModel):
    score: int
    default_probability: float
    risk_band: str
    top_drivers: List[dict]
