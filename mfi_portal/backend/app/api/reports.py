"""Reports API."""
from datetime import datetime, timedelta
from io import BytesIO, StringIO
from typing import Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.database import get_db
from app.db.models import Applicant, ScoreInquiry, User
from app.api.deps import get_current_user

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/dashboard")
def dashboard(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Summary stats for dashboard."""
    total_applicants = db.query(Applicant).count()
    total_scores = db.query(ScoreInquiry).count()
    today = datetime.utcnow().date()
    scores_today = db.query(ScoreInquiry).filter(
        func.date(ScoreInquiry.created_at) == today
    ).count()
    # Risk distribution
    low = db.query(ScoreInquiry).filter(ScoreInquiry.risk_band == "Low").count()
    med = db.query(ScoreInquiry).filter(ScoreInquiry.risk_band == "Medium").count()
    high = db.query(ScoreInquiry).filter(ScoreInquiry.risk_band == "High").count()
    return {
        "total_applicants": total_applicants,
        "total_score_inquiries": total_scores,
        "scores_today": scores_today,
        "risk_distribution": {"low": low, "medium": med, "high": high},
    }


@router.get("/activity")
def activity_log(
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Recent score inquiries for activity feed."""
    since = datetime.utcnow() - timedelta(days=days)
    rows = db.query(ScoreInquiry, Applicant).join(Applicant).filter(
        ScoreInquiry.created_at >= since
    ).order_by(ScoreInquiry.created_at.desc()).limit(limit).all()
    return [
        {
            "applicant_id": a.applicant_id,
            "full_name": a.full_name,
            "score": s.score,
            "risk_band": s.risk_band,
            "created_at": s.created_at.isoformat(),
        }
        for s, a in rows
    ]


@router.get("/export/csv")
def export_csv(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Export score inquiries as CSV."""
    since = datetime.utcnow() - timedelta(days=days)
    rows = db.query(ScoreInquiry, Applicant).join(Applicant).filter(
        ScoreInquiry.created_at >= since
    ).order_by(ScoreInquiry.created_at.desc()).all()

    import csv
    buf = StringIO()
    w = csv.writer(buf)
    w.writerow(["applicant_id", "full_name", "score", "risk_band", "default_probability", "created_at"])
    for s, a in rows:
        w.writerow([a.applicant_id, a.full_name or "", s.score, s.risk_band, s.default_probability, s.created_at.isoformat()])
    buf.seek(0)
    return StreamingResponse(
        BytesIO(buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=credit_scores_{datetime.utcnow().strftime('%Y%m%d')}.csv"},
    )
