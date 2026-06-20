"""Credit scoring service — uses shared src.scoring inference."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, Optional

from src.scoring.inference import applicant_dict_to_dataframe, score_features


def compute_score(applicant: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    df = applicant_dict_to_dataframe(applicant)
    return score_features(df, models_dir=PROJECT_ROOT / "models")
