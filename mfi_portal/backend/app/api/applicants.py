"""Applicants API."""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Applicant, ScoreInquiry, User
from app.api.deps import get_current_user
from app.schemas.applicant import ApplicantCreate, ApplicantUpdate, ApplicantResponse, ScoreResult
from app.services.scoring import compute_score

router = APIRouter(prefix="/applicants", tags=["applicants"])


def _to_response(a: Applicant) -> dict:
    return ApplicantResponse(
        id=a.id,
        applicant_id=a.applicant_id,
        national_id=a.national_id,
        full_name=a.full_name,
        gender=a.gender,
        age=a.age,
        location=a.location,
        employment=a.employment,
        sector=a.sector,
        msme=a.msme,
        education=getattr(a, "education", None),
        region=getattr(a, "region", None),
        mm_provider=getattr(a, "mm_provider", None),
        mm_txn_freq=a.mm_txn_freq,
        mm_txn_per_month=getattr(a, "mm_txn_per_month", None),
        mm_avg_amount_usd=a.mm_avg_amount_usd,
        mm_balance_volatility=a.mm_balance_volatility,
        mm_tenure_months=a.mm_tenure_months,
        mm_weekend_usage_pct=getattr(a, "mm_weekend_usage_pct", None),
        mm_p2p_ratio=a.mm_p2p_ratio,
        mm_bill_payment_ratio=a.mm_bill_payment_ratio,
        mm_merchant_ratio=a.mm_merchant_ratio,
        airtime_topups_per_month=getattr(a, "airtime_topups_per_month", None),
        airtime_avg_amount_usd=getattr(a, "airtime_avg_amount_usd", None),
        airtime_consistency_score=getattr(a, "airtime_consistency_score", None),
        data_bundles_per_month=getattr(a, "data_bundles_per_month", None),
        data_avg_bundle_usd=getattr(a, "data_avg_bundle_usd", None),
        utility_payment_rate=getattr(a, "utility_payment_rate", None),
        zesa_type=getattr(a, "zesa_type", None),
        has_smartphone=getattr(a, "has_smartphone", None),
        social_media_usage=getattr(a, "social_media_usage", None),
        util_electricity_consistency=a.util_electricity_consistency,
        util_water_consistency=a.util_water_consistency,
        util_telecom_consistency=a.util_telecom_consistency,
        util_overdue_count=a.util_overdue_count,
        util_avg_delay_days=a.util_avg_delay_days,
        util_multi_service_consistency=a.util_multi_service_consistency,
        consecutive_on_time=a.consecutive_on_time,
        account_age_months=a.account_age_months,
        digital_engagement=a.digital_engagement,
        channel_diversity=a.channel_diversity,
        created_at=a.created_at,
    ).model_dump()


_SCORE_FIELDS = [
    "gender", "age", "location", "region", "employment", "sector", "msme", "education",
    "mm_provider", "mm_txn_freq", "mm_txn_per_month", "mm_avg_amount_usd", "mm_avg_txn_usd",
    "mm_balance_volatility", "mm_tenure_months", "mm_weekend_usage_pct",
    "mm_p2p_ratio", "mm_bill_payment_ratio", "mm_merchant_ratio",
    "airtime_topups_per_month", "airtime_avg_amount_usd", "airtime_consistency_score",
    "data_bundles_per_month", "data_avg_bundle_usd",
    "utility_payment_rate", "zesa_type",
    "util_electricity_consistency", "util_water_consistency", "util_telecom_consistency",
    "util_overdue_count", "util_avg_delay_days", "util_multi_service_consistency",
    "consecutive_on_time", "account_age_months", "digital_engagement", "channel_diversity",
    "has_smartphone", "social_media_usage",
]

def _applicant_to_dict(a: Applicant) -> dict:
    return {k: getattr(a, k, None) for k in _SCORE_FIELDS}


@router.get("", response_model=List[dict])
def list_applicants(
    q: Optional[str] = Query(None),
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    qry = db.query(Applicant)
    if q:
        qry = qry.filter(
            Applicant.applicant_id.ilike(f"%{q}%") |
            Applicant.full_name.ilike(f"%{q}%") |
            Applicant.national_id.ilike(f"%{q}%")
        )
    applicants = qry.order_by(Applicant.created_at.desc()).offset(skip).limit(limit).all()
    return [_to_response(a) for a in applicants]


@router.post("", response_model=dict)
def create_applicant(
    data: ApplicantCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if db.query(Applicant).filter(Applicant.applicant_id == data.applicant_id).first():
        raise HTTPException(status_code=400, detail="Applicant ID already exists")
    a = Applicant(
        applicant_id=data.applicant_id,
        national_id=data.national_id,
        full_name=data.full_name,
        gender=data.gender,
        age=data.age,
        location=data.location,
        employment=data.employment,
        sector=data.sector,
        msme=data.msme,
        education=data.education,
        region=data.region,
        mm_provider=data.mm_provider,
        mm_txn_freq=data.mm_txn_freq,
        mm_txn_per_month=data.mm_txn_per_month,
        mm_avg_amount_usd=data.mm_avg_amount_usd,
        mm_balance_volatility=data.mm_balance_volatility,
        mm_tenure_months=data.mm_tenure_months,
        mm_weekend_usage_pct=data.mm_weekend_usage_pct,
        mm_p2p_ratio=data.mm_p2p_ratio,
        mm_bill_payment_ratio=data.mm_bill_payment_ratio,
        mm_merchant_ratio=data.mm_merchant_ratio,
        airtime_topups_per_month=data.airtime_topups_per_month,
        airtime_avg_amount_usd=data.airtime_avg_amount_usd,
        airtime_consistency_score=data.airtime_consistency_score,
        data_bundles_per_month=data.data_bundles_per_month,
        data_avg_bundle_usd=data.data_avg_bundle_usd,
        utility_payment_rate=data.utility_payment_rate,
        zesa_type=data.zesa_type,
        has_smartphone=data.has_smartphone,
        social_media_usage=data.social_media_usage,
        util_electricity_consistency=data.util_electricity_consistency,
        util_water_consistency=data.util_water_consistency,
        util_telecom_consistency=data.util_telecom_consistency,
        util_overdue_count=data.util_overdue_count,
        util_avg_delay_days=data.util_avg_delay_days,
        util_multi_service_consistency=data.util_multi_service_consistency,
        consecutive_on_time=data.consecutive_on_time,
        account_age_months=data.account_age_months,
        digital_engagement=data.digital_engagement,
        channel_diversity=data.channel_diversity,
        created_by=user.id,
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return _to_response(a)


@router.get("/{applicant_id}", response_model=dict)
def get_applicant(
    applicant_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    a = db.query(Applicant).filter(Applicant.applicant_id == applicant_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Applicant not found")
    return _to_response(a)


@router.patch("/{applicant_id}", response_model=dict)
def update_applicant(
    applicant_id: str,
    data: ApplicantUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    a = db.query(Applicant).filter(Applicant.applicant_id == applicant_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Applicant not found")
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(a, k, v)
    db.commit()
    db.refresh(a)
    return _to_response(a)


@router.post("/{applicant_id}/score", response_model=dict)
def get_score(
    applicant_id: str,
    notes: Optional[str] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    a = db.query(Applicant).filter(Applicant.applicant_id == applicant_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Applicant not found")
    applicant_dict = _applicant_to_dict(a)
    result = compute_score(applicant_dict)
    if result is None:
        raise HTTPException(status_code=503, detail="Scoring model not available. Run: python scripts/train.py --dataset zimbabwe_synthetic")
    # Save to score history
    si = ScoreInquiry(
        applicant_id=a.id,
        score=result["score"],
        default_probability=result["default_probability"],
        risk_band=result["risk_band"],
        top_drivers=result["top_drivers"],
        created_by=user.id,
        notes=notes,
    )
    db.add(si)
    db.commit()
    return {"applicant_id": applicant_id, **result}


@router.post("/score/quick", response_model=dict)
def quick_score(
    data: dict,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Score from raw data without saving applicant (e.g. preliminary check)."""
    result = compute_score(data)
    if result is None:
        raise HTTPException(status_code=503, detail="Scoring model not available")
    return result


@router.get("/{applicant_id}/history")
def get_score_history(
    applicant_id: str,
    limit: int = 20,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    a = db.query(Applicant).filter(Applicant.applicant_id == applicant_id).first()
    if not a:
        raise HTTPException(status_code=404, detail="Applicant not found")
    inquiries = db.query(ScoreInquiry).filter(ScoreInquiry.applicant_id == a.id).order_by(ScoreInquiry.created_at.desc()).limit(limit).all()
    return [{"score": s.score, "risk_band": s.risk_band, "created_at": s.created_at.isoformat()} for s in inquiries]
