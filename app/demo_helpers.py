"""Live Demo applicant selection — pure helpers (no Gradio dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

DEMO_SCENARIOS = {
    "browse_all": {
        "label": "Browse all applicants",
        "gender": "All", "location": "All", "msme": "All",
        "hint": "Full synthetic dataset — pick anyone from the dropdown.",
    },
    "rural_msme_women": {
        "label": "Inclusion focus · Rural MSME women",
        "gender": "Female", "location": "Rural", "msme": "MSME",
        "hint": "NDS1/NDS2 priority group: women entrepreneurs in rural areas.",
    },
    "urban_youth": {
        "label": "Youth · Urban digital natives",
        "gender": "All", "location": "Urban", "msme": "Non-MSME",
        "hint": "Young urban applicants — mobile money and app usage tend to be higher.",
    },
    "strong_payers": {
        "label": "Strong payers · High utility consistency",
        "gender": "All", "location": "All", "msme": "All",
        "hint": "Applicants with utility payment rate above 85% — typically stronger credit signals.",
        "extra_filter": "strong_payer",
    },
    "risk_signals": {
        "label": "Risk signals · Overdue utilities",
        "gender": "All", "location": "All", "msme": "All",
        "hint": "Applicants with overdue bills — useful for contrasting scores in the panel.",
        "extra_filter": "overdue",
    },
}


def load_demo_dataframe() -> pd.DataFrame:
    dataset_name = "zimbabwe_synthetic"
    if Path("models/dataset_name.txt").exists():
        dataset_name = Path("models/dataset_name.txt").read_text().strip()
    from src.data import load_dataset
    return load_dataset(dataset_name)


def fmt_value(value, suffix="") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}" if abs(value) < 1000 else f"{value:,.0f}{suffix}"
    if isinstance(value, (int, np.integer)):
        if value in (0, 1) and suffix == "":
            return "Yes" if value == 1 else "No"
        return f"{value}{suffix}"
    text = str(value).strip()
    return text.replace("_", " ").title() if text else "—"


def applicant_choice_label(row: pd.Series, idx: int) -> str:
    gender = fmt_value(row.get("gender"))[0] if fmt_value(row.get("gender")) != "—" else "?"
    age = fmt_value(row.get("age"))
    location = fmt_value(row.get("location"))[:3]
    provider = fmt_value(row.get("mm_provider"))[:3]
    txn = fmt_value(row.get("mm_txn_per_month"))
    return f"#{idx + 1} · {gender} · {age}y · {location} · {provider} · {txn} txn/mo"


def applicant_initials(row: pd.Series) -> str:
    gender = str(row.get("gender", "?"))[0].upper()
    return gender if gender in "FM" else "A"


def snapshot_tagline(row: pd.Series) -> str:
    util = float(row.get("utility_payment_rate", 0) or 0)
    overdue = int(row.get("util_overdue_count", 0) or 0)
    digital = float(row.get("digital_engagement", 0) or 0)
    if overdue >= 3:
        return "Several overdue utility bills — expect a lower score unless mobile money compensates."
    if util >= 0.85 and digital >= 0.6:
        return "Strong alternative-data profile: consistent utilities and active digital footprint."
    if util < 0.5:
        return "Weak utility payment history — key risk signal for alternative-data scoring."
    if row.get("msme") == 1:
        return "MSME operator — business transaction patterns feed into the LoRA model."
    return "Typical applicant — review mobile money and utility signals before scoring."


def build_applicant_snapshot(row: pd.Series, idx: int) -> str:
    initials = applicant_initials(row)
    name = f"Applicant #{idx + 1}"
    subtitle = " · ".join(filter(None, [
        fmt_value(row.get("gender")),
        f"{fmt_value(row.get('age'))} years",
        fmt_value(row.get("location")),
        "MSME" if row.get("msme") == 1 else "Individual",
        fmt_value(row.get("mm_provider")),
    ]))

    def metric(label: str, value: str) -> str:
        return f'<div class="snapshot-metric"><span>{label}</span><strong>{value}</strong></div>'

    util_pct = f"{100 * float(row.get('utility_payment_rate', 0) or 0):.0f}%"
    digital_pct = f"{100 * float(row.get('digital_engagement', 0) or 0):.0f}%"

    return f"""
    <div class="applicant-snapshot">
      <div class="snapshot-avatar">{initials}</div>
      <div class="snapshot-main">
        <h3>{name}</h3>
        <div class="snapshot-tagline">{subtitle}</div>
        <p class="snapshot-tagline" style="margin-top:0;font-style:italic;">{snapshot_tagline(row)}</p>
        <div class="snapshot-metrics">
          {metric("Mobile money", f"{fmt_value(row.get('mm_txn_per_month'))}/mo")}
          {metric("Utility rate", util_pct)}
          {metric("Digital engagement", digital_pct)}
          {metric("Overdue bills", fmt_value(row.get('util_overdue_count')))}
        </div>
      </div>
    </div>
    """


def empty_snapshot(message: str = "Choose an applicant from the dropdown to preview their profile.") -> str:
    return f'<div class="snapshot-empty">{message}</div>'


def filter_applicant_indices(
    df: pd.DataFrame,
    gender: str,
    location: str,
    msme: str,
    extra_filter: str = "",
) -> np.ndarray:
    mask = pd.Series(True, index=df.index)
    if gender != "All":
        mask &= df["gender"].astype(str).str.lower() == gender.lower()
    if location != "All":
        mask &= df["location"].astype(str).str.lower() == location.lower()
    if msme == "MSME":
        mask &= df["msme"] == 1
    elif msme == "Non-MSME":
        mask &= df["msme"] == 0
    if extra_filter == "strong_payer":
        mask &= df["utility_payment_rate"].astype(float) >= 0.85
    elif extra_filter == "overdue":
        mask &= df["util_overdue_count"].astype(int) >= 2
    return df.index[mask].to_numpy()


def build_profile_html(row: pd.Series, idx: int) -> str:
    badges = [
        fmt_value(row.get("gender")),
        f"{fmt_value(row.get('age'))} yrs" if pd.notna(row.get("age")) else None,
        fmt_value(row.get("location")),
        "MSME" if row.get("msme") == 1 else "Individual",
        fmt_value(row.get("region")) if pd.notna(row.get("region")) else None,
    ]
    badge_html = "".join(f'<span class="profile-badge">{b}</span>' for b in badges if b and b != "—")

    def block(title: str, rows: list) -> str:
        items = "".join(
            f'<div class="profile-row"><span>{label}</span><span>{value}</span></div>'
            for label, value in rows
        )
        return f'<div class="profile-block"><h4>{title}</h4>{items}</div>'

    sections = [
        block("Personal", [
            ("Education", fmt_value(row.get("education"))),
            ("Employment", fmt_value(row.get("employment"))),
            ("Sector", fmt_value(row.get("sector"))),
            ("Household size", fmt_value(row.get("household_size"))),
            ("Youth (≤35)", fmt_value(row.get("youth"))),
        ]),
        block("Mobile money", [
            ("Provider", fmt_value(row.get("mm_provider"))),
            ("Transactions / month", fmt_value(row.get("mm_txn_per_month"))),
            ("Avg. transaction", f"${fmt_value(row.get('mm_avg_txn_usd'))}"),
            ("Tenure (months)", fmt_value(row.get("mm_tenure_months"))),
            ("Bill payment ratio", fmt_value(row.get("mm_bill_payment_ratio"))),
        ]),
        block("Utilities & telecom", [
            ("Utility payment rate", fmt_value(row.get("utility_payment_rate"))),
            ("Electricity consistency", fmt_value(row.get("util_electricity_consistency"))),
            ("Water consistency", fmt_value(row.get("util_water_consistency"))),
            ("Overdue bills", fmt_value(row.get("util_overdue_count"))),
            ("Airtime consistency", fmt_value(row.get("airtime_consistency_score"))),
        ]),
        block("Digital & behaviour", [
            ("Smartphone", fmt_value(row.get("has_smartphone"))),
            ("Digital engagement", fmt_value(row.get("digital_engagement"))),
            ("App sessions / week", fmt_value(row.get("app_sessions_per_week"))),
            ("Account age (months)", fmt_value(row.get("account_age_months"))),
            ("On-time streak", fmt_value(row.get("consecutive_on_time"))),
        ]),
    ]

    return f"""
    <div class="profile-card">
      <div class="profile-header">
        <div>
          <h3>Applicant profile</h3>
          <p style="margin:0.35rem 0 0; color:#64748b; font-size:0.9rem;">Review alternative-data signals before running the credit check.</p>
        </div>
        <div style="text-align:right;">
          <div style="font-size:0.8rem; color:#64748b;">Record</div>
          <div style="font-size:1.1rem; font-weight:700; color:#0f172a;">#{idx + 1}</div>
        </div>
      </div>
      <div style="margin-bottom:1rem;">{badge_html}</div>
      <div class="profile-grid">{''.join(sections)}</div>
    </div>
    """


def scenario_extra_filter(scenario_key: str) -> str:
    return DEMO_SCENARIOS.get(scenario_key, DEMO_SCENARIOS["browse_all"]).get("extra_filter", "")


def resolve_applicant_list(
    gender: str,
    location: str,
    msme: str,
    scenario_key: str = "browse_all",
    df: Optional[pd.DataFrame] = None,
) -> Tuple[list, Optional[str], Optional[str], Optional[str]]:
    """
    Build dropdown choices, snapshot HTML, and profile HTML for the Live Demo.

    Returns (choices, first_id, snapshot_html, profile_html) or error placeholders.
    """
    extra = scenario_extra_filter(scenario_key)
    if df is None:
        df = load_demo_dataframe()

    indices = filter_applicant_indices(df, gender, location, msme, extra)
    if len(indices) == 0:
        return [], None, empty_snapshot("No applicants match — try a different scenario or filter."), (
            '<div class="profile-card"><p style="margin:0;color:#64748b;">No applicants match these filters.</p></div>'
        )

    choices = [(applicant_choice_label(df.loc[i], i), str(i)) for i in indices[:400]]
    first_id = choices[0][1]
    first_idx = int(first_id)
    row = df.loc[first_idx]
    note = f"{len(choices):,} applicants in list"
    if len(indices) > 400:
        note += f" (showing 400 of {len(indices):,})"
    profile = build_profile_html(row, first_idx) + (
        f'<p style="margin:0.75rem 0 0; color:#64748b; font-size:0.85rem;">{note}</p>'
    )
    return choices, first_id, build_applicant_snapshot(row, first_idx), profile


def apply_demo_scenario_filters(scenario_key: str) -> Tuple[str, str, str, str]:
    """Return gender, location, msme, hint for a curated scenario."""
    scenario = DEMO_SCENARIOS.get(scenario_key, DEMO_SCENARIOS["browse_all"])
    return scenario["gender"], scenario["location"], scenario["msme"], scenario.get("hint", "")
