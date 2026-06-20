"""
LoRA Credit Scoring — Dissertation Demo UI

Presentation-ready interface for research objectives.
Run: python -m app.main
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ─── Config ───────────────────────────────────────────────────────────────

CUSTOM_CSS = """
:root { --radius: 16px; --card-bg: rgba(255,255,255,0.95); }
.gradio-container { max-width: 960px !important; margin: auto; }
.gr-box { border-radius: var(--radius) !important; }
.gr-button-primary { border-radius: 12px !important; font-weight: 600 !important; }
h1 { font-size: 2rem !important; letter-spacing: -0.02em !important; }
.gr-tabs { border-radius: var(--radius) !important; }
.gr-tab-item { font-weight: 500 !important; }
.hero { 
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0c4a6e 100%); 
  color: white; padding: 2rem; border-radius: var(--radius); margin-bottom: 1.5rem;
  text-align: center; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}
.hero h2 { margin: 0 0 0.5rem; font-size: 1.5rem; font-weight: 600; }
.hero .stats { display: flex; flex-wrap: wrap; justify-content: center; gap: 1.5rem; margin-top: 1rem; }
.hero .stat { background: rgba(255,255,255,0.15); padding: 0.5rem 1rem; border-radius: 10px; font-size: 0.95rem; }
.demo-guide { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }
.metric-card { background: linear-gradient(145deg, #f8fafc, #f1f5f9); border-radius: 12px; padding: 1rem; border-left: 4px solid #0ea5e9; margin: 0.5rem 0; }
.metric-card .val { font-size: 1.5rem; font-weight: 700; color: #0f172a; }
.metric-card .lbl { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.demo-step { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: #64748b; margin: 0 0 0.35rem; }
.profile-card {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 1.25rem 1.5rem;
  box-shadow: 0 1px 3px rgba(15,23,42,0.06);
}
.profile-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid #f1f5f9; }
.profile-header h3 { margin: 0; font-size: 1.15rem; color: #0f172a; }
.profile-badge { display: inline-block; background: #e0f2fe; color: #0369a1; font-size: 0.75rem; font-weight: 600; padding: 0.2rem 0.55rem; border-radius: 999px; margin-right: 0.35rem; }
.profile-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.85rem 1.25rem; }
.profile-block { background: #f8fafc; border-radius: 10px; padding: 0.85rem 1rem; }
.profile-block h4 { margin: 0 0 0.55rem; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; }
.profile-row { display: flex; justify-content: space-between; gap: 0.75rem; font-size: 0.9rem; padding: 0.18rem 0; }
.profile-row span:first-child { color: #64748b; }
.profile-row span:last-child { color: #0f172a; font-weight: 600; text-align: right; }
.score-panel { background: linear-gradient(180deg, #f8fafc 0%, #fff 100%); border: 1px dashed #cbd5e1; border-radius: 14px; padding: 1.25rem; margin-top: 0.5rem; }
.score-panel-empty { color: #64748b; text-align: center; padding: 2rem 1rem; font-size: 0.95rem; }
"""

DEMO_FLOW = """
**Panel presentation flow (aligned with dissertation §1.5):**
1. **Overview** → Problem (89% credit excluded), 5 objectives, research questions RQ1–RQ4
2. **Dataset** → Objective 1: synthetic alternative data (mobile money, utilities, digital commerce)
3. **Live Demo** → Objective 2: LoRA credit score with per-applicant drivers
4. **Results** → Objectives 2 & 3: LoRA vs baselines (AUC-ROC, AUC-PR, F1)
5. **Efficiency** → Objective 3: trainable params, training time, inference latency
6. **Fairness** → Objective 4: gender, age, location, income quartile, MSME
7. **Explainability** → LoRA feature attribution + RF SHAP comparison
8. **Policy** → Objective 5: NDS1/NDS2 and National AI Strategy recommendations
9. **MFI Portal** (http://localhost:5174) → deployment architecture §C.4
"""


def _load_demo_dataframe() -> pd.DataFrame:
    """Load applicant dataset once for the Live Demo tab."""
    dataset_name = "zimbabwe_synthetic"
    if Path("models/dataset_name.txt").exists():
        dataset_name = Path("models/dataset_name.txt").read_text().strip()
    from src.data import load_dataset
    return load_dataset(dataset_name)


def _fmt(value, suffix="") -> str:
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


def _applicant_choice_label(row: pd.Series, idx: int) -> str:
    gender = _fmt(row.get("gender"))
    age = _fmt(row.get("age"))
    location = _fmt(row.get("location"))
    msme = "MSME" if row.get("msme") == 1 else "Individual"
    provider = _fmt(row.get("mm_provider"))
    return f"Applicant {idx + 1} — {gender}, {age} yrs · {location} · {msme} · {provider}"


def _filter_applicant_indices(df: pd.DataFrame, gender: str, location: str, msme: str) -> np.ndarray:
    mask = pd.Series(True, index=df.index)
    if gender != "All":
        mask &= df["gender"].astype(str).str.lower() == gender.lower()
    if location != "All":
        mask &= df["location"].astype(str).str.lower() == location.lower()
    if msme == "MSME":
        mask &= df["msme"] == 1
    elif msme == "Non-MSME":
        mask &= df["msme"] == 0
    return df.index[mask].to_numpy()


def _build_profile_html(row: pd.Series, idx: int) -> str:
    badges = [
        _fmt(row.get("gender")),
        f"{_fmt(row.get('age'))} yrs" if pd.notna(row.get("age")) else None,
        _fmt(row.get("location")),
        "MSME" if row.get("msme") == 1 else "Individual",
        _fmt(row.get("region")) if pd.notna(row.get("region")) else None,
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
            ("Education", _fmt(row.get("education"))),
            ("Employment", _fmt(row.get("employment"))),
            ("Sector", _fmt(row.get("sector"))),
            ("Household size", _fmt(row.get("household_size"))),
            ("Youth (≤35)", _fmt(row.get("youth"))),
        ]),
        block("Mobile money", [
            ("Provider", _fmt(row.get("mm_provider"))),
            ("Transactions / month", _fmt(row.get("mm_txn_per_month"))),
            ("Avg. transaction", f"${_fmt(row.get('mm_avg_txn_usd'))}"),
            ("Tenure (months)", _fmt(row.get("mm_tenure_months"))),
            ("Bill payment ratio", _fmt(row.get("mm_bill_payment_ratio"))),
        ]),
        block("Utilities & telecom", [
            ("Utility payment rate", _fmt(row.get("utility_payment_rate"))),
            ("Electricity consistency", _fmt(row.get("util_electricity_consistency"))),
            ("Water consistency", _fmt(row.get("util_water_consistency"))),
            ("Overdue bills", _fmt(row.get("util_overdue_count"))),
            ("Airtime consistency", _fmt(row.get("airtime_consistency_score"))),
        ]),
        block("Digital & behaviour", [
            ("Smartphone", _fmt(row.get("has_smartphone"))),
            ("Digital engagement", _fmt(row.get("digital_engagement"))),
            ("App sessions / week", _fmt(row.get("app_sessions_per_week"))),
            ("Account age (months)", _fmt(row.get("account_age_months"))),
            ("On-time streak", _fmt(row.get("consecutive_on_time"))),
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


def get_applicant_choices(gender: str, location: str, msme: str):
    """Populate the applicant dropdown from simple filters."""
    try:
        df = _load_demo_dataframe()
    except Exception as e:
        return gr.Dropdown(choices=[], value=None, label="Select applicant"), f"Could not load dataset: {e}"

    indices = _filter_applicant_indices(df, gender, location, msme)
    if len(indices) == 0:
        return gr.Dropdown(choices=[], value=None, label="Select applicant"), (
            '<div class="profile-card"><p style="margin:0;color:#64748b;">No applicants match these filters. Try broader options.</p></div>'
        )

    choices = [(_applicant_choice_label(df.loc[i], i), str(i)) for i in indices[:400]]
    first_id = choices[0][1]
    profile = _build_profile_html(df.loc[int(first_id)], int(first_id))
    note = f"Showing {len(choices):,} of {len(indices):,} matching applicants."
    if len(indices) > 400:
        note += " Narrow filters to see a shorter list."
    return gr.Dropdown(choices=choices, value=first_id, label="Select applicant", filterable=True), profile + f'<p style="margin:0.75rem 0 0; color:#64748b; font-size:0.85rem;">{note}</p>'


def show_applicant_profile(applicant_id: str):
    """Load profile only — no model inference yet."""
    if not applicant_id:
        return (
            '<div class="profile-card"><p style="margin:0;color:#64748b;">Choose an applicant from the list above.</p></div>',
            "Select an applicant to enable scoring.",
            None,
            "",
        )
    try:
        df = _load_demo_dataframe()
        idx = int(applicant_id)
        row = df.iloc[idx]
    except Exception:
        return (
            '<div class="profile-card"><p style="margin:0;color:#64748b;">Applicant not found.</p></div>',
            "Select an applicant to enable scoring.",
            None,
            "",
        )
    return (
        _build_profile_html(row, idx),
        "Profile loaded. Click **Check credit score** when ready.",
        None,
        "",
    )


def pick_random_applicant(gender: str, location: str, msme: str):
    """Pick a random applicant from the current filter — useful for panel demos."""
    try:
        df = _load_demo_dataframe()
    except Exception as e:
        return None, f"Could not load dataset: {e}", "Select an applicant to enable scoring.", None, ""

    indices = _filter_applicant_indices(df, gender, location, msme)
    if len(indices) == 0:
        return None, (
            '<div class="profile-card"><p style="margin:0;color:#64748b;">No applicants match these filters.</p></div>'
        ), "Select an applicant to enable scoring.", None, ""

    idx = int(np.random.choice(indices))
    choices = [(_applicant_choice_label(df.loc[i], i), str(i)) for i in indices[:400]]
    return (
        gr.Dropdown(choices=choices, value=str(idx), label="Select applicant", filterable=True),
        _build_profile_html(df.loc[idx], idx),
        "Profile loaded. Click **Check credit score** when ready.",
        None,
        "",
    )


def run_credit_score(applicant_id: str):
    """Run LoRA inference after the user has reviewed the applicant profile."""
    if not applicant_id:
        return "Select an applicant and review their profile first.", None, ""

    out = load_model_and_prep()
    if out[0] is None:
        return (
            "⚠️ **Train models first**\n\n```bash\npython scripts/train.py --dataset zimbabwe_synthetic\n```",
            None,
            "",
        )

    from src.scoring import score_applicant_row

    try:
        result = score_applicant_row(int(applicant_id), artifacts=out[:3])
    except ValueError as e:
        return f"⚠️ **Model out of date** — retrain to match current features.\n\n`{e}`", None, ""

    if result is None:
        return "⚠️ **Could not score applicant.**", None, ""

    proba = result["default_probability"]
    score = result["score"]
    risk = "Low ✓" if proba < 0.3 else "Medium" if proba < 0.6 else "High ✗"
    risk_color = "#10b981" if proba < 0.3 else "#f59e0b" if proba < 0.6 else "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Credit Score (300–850)"},
        gauge={
            "axis": {"range": [300, 850], "tickwidth": 1},
            "bar": {"color": "#0ea5e9"},
            "steps": [
                {"range": [300, 580], "color": "#fef2f2"},
                {"range": [580, 670], "color": "#fef9c3"},
                {"range": [670, 850], "color": "#dcfce7"},
            ],
            "threshold": {"line": {"color": risk_color, "width": 3}, "thickness": 0.85, "value": score},
        },
        number={"suffix": "", "font": {"size": 36}},
    ))
    fig.update_layout(height=260, margin=dict(l=25, r=25, t=50, b=25), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, system-ui, sans-serif"))

    explain = ""
    top = result.get("top_drivers") or []
    if top:
        parts = []
        for t in top[:4]:
            feat = t.get("feature", "")
            direction = t.get("direction", "")
            impact = t.get("impact", t.get("importance", 0))
            parts.append(f"*{feat}* ({direction}, Δ{abs(float(impact)):.3f})")
        explain = "**LoRA drivers for this applicant:** " + " → ".join(parts)

    result_text = f"### {risk}\n**Default probability:** {proba:.1%}  |  **Score:** {score}"
    return result_text, fig, explain


def load_model_and_prep():
    models_dir = Path("models")
    if not (models_dir / "lora" / "best_model.pt").exists():
        return None, None, None, 0
    from src.data import load_dataset, DataPreprocessor
    from src.models import LoRACreditScorer

    prep = DataPreprocessor.load(str(models_dir / "preprocessor.pkl"))
    lora = LoRACreditScorer.load_pretrained(str(models_dir / "lora" / "best_model.pt"))
    dataset_name = (models_dir / "dataset_name.txt").read_text().strip() if (models_dir / "dataset_name.txt").exists() else "zimbabwe_synthetic"
    df = load_dataset(dataset_name)
    return lora, prep, df, len(df) - 1


def model_comparison():
    path = Path("results/metrics/training_results.json")
    if not path.exists():
        return None, "", "Run training first: `python scripts/train.py`"

    with open(path) as f:
        r = json.load(f)

    lora = r.get("lora", {})
    cards_html = ""
    if lora:
        pct = 100 * lora.get("trainable_params", 0) / max(lora.get("total_params", 1), 1)
        cards_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.75rem; margin-bottom: 1rem;">
          <div style="background: linear-gradient(145deg,#f8fafc,#f1f5f9); border-radius: 12px; padding: 1rem; border-left: 4px solid #0ea5e9;">
            <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase;">AUC-ROC</div>
            <div style="font-size:1.5rem; font-weight:700;">{lora.get('auc_roc', 0):.3f}</div>
          </div>
          <div style="background: linear-gradient(145deg,#f8fafc,#f1f5f9); border-radius: 12px; padding: 1rem; border-left: 4px solid #10b981;">
            <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase;">AUC-PR</div>
            <div style="font-size:1.5rem; font-weight:700;">{lora.get('auc_pr', 0):.3f}</div>
          </div>
          <div style="background: linear-gradient(145deg,#f8fafc,#f1f5f9); border-radius: 12px; padding: 1rem; border-left: 4px solid #f59e0b;">
            <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase;">F1</div>
            <div style="font-size:1.5rem; font-weight:700;">{lora.get('f1', 0):.3f}</div>
          </div>
          <div style="background: linear-gradient(145deg,#f8fafc,#f1f5f9); border-radius: 12px; padding: 1rem; border-left: 4px solid #8b5cf6;">
            <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase;">Params trained</div>
            <div style="font-size:1.5rem; font-weight:700;">~{pct:.1f}%</div>
          </div>
        </div>
        """

    rows = [{"Model": k.replace("_", " ").title(), "AUC-ROC": v["auc_roc"], "AUC-PR": v.get("auc_pr", 0), "F1": v["f1"]} for k, v in r.items()]
    df = pd.DataFrame(rows)
    y_cols = [c for c in ["AUC-ROC", "AUC-PR", "F1"] if c in df.columns]
    fig = px.bar(df, x="Model", y=y_cols, barmode="group",
                 color_discrete_sequence=["#0ea5e9", "#10b981", "#f59e0b"][:len(y_cols)],
                 title="Model Comparison (§3.6.1, §4.4.4)")
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.9)", font=dict(size=12),
                      xaxis_tickangle=-15, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    eff = ""
    if lora and "trainable_params" in lora:
        eff = (
            f"**LoRA efficiency (Objective 3):** {lora['trainable_params']:,} / {lora['total_params']:,} params "
            f"(~{100 * lora['trainable_params'] / lora['total_params']:.1f}%) · "
            f"Train: {lora.get('train_time_sec', 0):.0f}s · "
            f"Inference: {lora.get('inference_latency_ms', '—')}ms · "
            f"Peak memory: {lora.get('peak_memory_mb', '—')} MB"
        )

    return fig, cards_html, eff


def efficiency_dashboard():
    path = Path("results/metrics/training_results.json")
    if not path.exists():
        return None, "Run training first: `python scripts/train.py`"

    with open(path) as f:
        r = json.load(f)
    lora = r.get("lora", {})
    if not lora:
        return None, "LoRA results not found."

    labels = ["Trainable params (M)", "Training time (min)", "Inference (ms)", "Peak memory (MB)"]
    values = [
        lora.get("trainable_params", 0) / 1e6,
        lora.get("train_time_sec", 0) / 60,
        lora.get("inference_latency_ms", 0),
        lora.get("peak_memory_mb", 0),
    ]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=["#8b5cf6", "#0ea5e9", "#10b981", "#f59e0b"]))
    fig.update_layout(
        title="Computational Efficiency — Objective 3 (§3.6.2, §4.4.6)",
        height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Value",
    )
    pct = 100 * lora.get("trainable_params", 0) / max(lora.get("total_params", 1), 1)
    note = (
        f"**Objective 3:** LoRA trains only **~{pct:.1f}%** of parameters while maintaining competitive AUC. "
        f"Inference latency **{lora.get('inference_latency_ms', '—')} ms** per applicant supports "
        f"real-time MFI deployment in resource-constrained settings."
    )
    return fig, note


def fairness_dashboard():
    path = Path("results/metrics/fairness_results.json")
    if not path.exists():
        return None, "Run training first to see fairness metrics (§3.6.3).", None

    with open(path) as f:
        data = json.load(f)
    lora = data.get("lora", {})
    groups = lora.get("groups", {})
    fm = lora.get("fairness_metrics", {})

    if not groups:
        return None, "Use **zimbabwe_synthetic** dataset for gender, location, age, income quartile, MSME subgroups (§3.6.3)."

    rows = []
    for attr, gdata in groups.items():
        for gname, m in gdata.items():
            rows.append({"Attribute": attr, "Group": str(gname), "AUC": m.get("auc", 0), "TPR": m.get("tpr", 0)})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return None, str(lora)

    fig = px.bar(df, x="Group", y=["AUC", "TPR"], color="Attribute", barmode="group",
                 color_discrete_sequence=["#0ea5e9", "#10b981", "#f59e0b", "#8b5cf6"],
                 title="Performance by Protected Group (§3.6.3, §4.4.8)")
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-20, font=dict(size=11))

    parts = []
    for kind, d in fm.items():
        if d:
            parts.append(f"**{kind.replace('_', ' ').title()}:** " + " · ".join(f"{k.split('_')[0]}: {v:.3f}" for k, v in list(d.items())[:3]))
    return fig, "\n\n".join(parts) if parts else "Fairness metrics computed."


def explainability_tab():
    path = Path("results/metrics/feature_importance.json")
    if not path.exists():
        return None, "Run training first. LoRA attribution and RF SHAP computed in §4.4.9."

    with open(path) as f:
        data = json.load(f)
    fi = data.get("lora_global_attribution") or data.get("feature_importance") or data.get("shap_importance", [])
    if not fi:
        return None, "No explainability data."

    df = pd.DataFrame(fi)
    y_col = "mean_abs_impact" if "mean_abs_impact" in df.columns else (
        "importance" if "importance" in df.columns else "mean_abs_shap"
    )
    fig = px.bar(df.head(12), x="feature", y=y_col,
                 color=y_col, color_continuous_scale=["#e0f2fe", "#0ea5e9", "#0369a1"],
                 title="LoRA Global Feature Attribution — Objective 2 (§4.4.9)")
    fig.update_layout(height=360, xaxis_tickangle=-35, paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig, (
        "**Live Demo** shows *per-applicant* LoRA drivers. This chart shows aggregate attribution. "
        "RF SHAP values are retained for baseline comparison (§4.4.9)."
    )


def dataset_info():
    try:
        from src.data import load_dataset
        dn = "zimbabwe_synthetic"
        if Path("models/dataset_name.txt").exists():
            dn = Path("models/dataset_name.txt").read_text().strip()
        df = load_dataset(dn)
    except Exception as e:
        return str(e), None

    n = len(df)
    tc = "default" if "default" in df.columns else next((c for c in df.columns if "default" in c.lower()), None)
    defs = int(df[tc].sum()) if tc else 0
    pct = 100 * defs / n if n else 0

    desc = "**Zimbabwe alternative data** (synthetic §3.3): mobile money, utility payments, digital commerce, demographics — **Objective 1**." if dn == "zimbabwe_synthetic" else f"**{dn}** — validation benchmark (§3.3.5)."
    summary = f"{desc}\n\n**{n:,}** samples  ·  **{defs:,}** defaults ({pct:.1f}%)  ·  **{len(df.columns)-1}** features"
    table_html = df.head(40).to_html(classes="table-auto", index=False)
    table = f'<div style="overflow-x:auto; overflow-y:auto; max-height:400px; max-width:100%;">{table_html}</div>'
    return summary, table


# ─── UI ──────────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(
    primary_hue="sky",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%)",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e2e8f0",
    block_label_background_fill="#0f172a",
    block_label_text_color="#ffffff",
    block_label_border_width="0px",
    block_label_radius="8px",
    button_primary_background_fill="linear-gradient(90deg, #0369a1 0%, #0ea5e9 100%)",
    button_primary_border_color="#0284c7",
)

with gr.Blocks(title="LoRA Credit Scoring | MSc Dissertation") as demo:

    gr.Markdown("# LoRA Credit Scoring")
    gr.Markdown("*Alternative Data · Zimbabwe Financial Inclusion · NDS1/NDS2*")

    gr.HTML("""
    <div class="hero">
      <h2>Low-Rank Adaptation for Alternative Data Credit Scoring</h2>
      <p style="opacity: 0.9; margin: 0;">Enhancing financial inclusion under the National Development Strategy</p>
      <div class="stats">
        <span class="stat">89% credit excluded</span>
        <span class="stat">~2–8% params trained (LoRA)</span>
        <span class="stat">MSME · Women · Youth · Rural</span>
      </div>
    </div>
    """)

    with gr.Accordion("📋 Demo guide for panel", open=False):
        gr.Markdown(DEMO_FLOW)

    with gr.Tabs() as tabs:
        with gr.TabItem("📋 Overview"):
            gr.Markdown("""
            ### Research aim (§1.5)
            Develop and evaluate a **LoRA-enhanced alternative data credit scoring system** that expands financial inclusion in Zimbabwe while maintaining computational efficiency, fairness, and NDS alignment.

            ### Specific objectives
            | # | Objective | Demo tab |
            |---|-----------|----------|
            | **1** | Synthetic alternative data framework (mobile money, utilities, digital commerce) | **Dataset** |
            | **2** | LoRA-enhanced credit scoring model | **Live Demo**, **Results**, **Explainability** |
            | **3** | Computational efficiency (params, time, latency, memory) | **Efficiency**, **Results** |
            | **4** | Fairness across gender, age, location, income, MSME | **Fairness** |
            | **5** | Policy alignment with NDS, National AI Strategy, RBZ | **Policy** |

            ### Research questions
            | RQ | Question | Addressed in |
            |----|----------|--------------|
            | **RQ1** | How can alternative data be synthesised for unbanked populations? | Dataset, Live Demo |
            | **RQ2** | Does LoRA maintain performance with reduced compute? | Results, Efficiency |
            | **RQ3** | What fairness characteristics emerge and how are they mitigated? | Fairness |
            | **RQ4** | How can the system align with national development priorities? | Policy, MFI Portal |

            ---
            *MTECH Software Engineering Project Documentation · Hu et al. (2021) · Zimbabwe NDS1/NDS2*
            """)

        with gr.TabItem("🔮 Live Demo"):
            gr.Markdown(
                "Walk through a realistic credit inquiry: **find an applicant → review their profile → run the LoRA score**."
            )

            gr.Markdown('<p class="demo-step">Step 1 · Find an applicant</p>')
            with gr.Row():
                filter_gender = gr.Dropdown(
                    ["All", "Female", "Male"], value="All", label="Gender", scale=1,
                )
                filter_location = gr.Dropdown(
                    ["All", "Urban", "Rural"], value="All", label="Location", scale=1,
                )
                filter_msme = gr.Dropdown(
                    ["All", "MSME", "Non-MSME"], value="All", label="Business type", scale=1,
                )
            with gr.Row():
                applicant_select = gr.Dropdown(
                    label="Select applicant",
                    choices=[],
                    filterable=True,
                    scale=3,
                )
                random_btn = gr.Button("Pick random", scale=1)
                apply_filters_btn = gr.Button("Apply filters", variant="secondary", scale=1)

            gr.Markdown('<p class="demo-step">Step 2 · Review applicant details</p>')
            profile_out = gr.HTML(
                '<div class="profile-card"><p style="margin:0;color:#64748b;">Use the filters above, then choose an applicant to load their profile.</p></div>'
            )

            gr.Markdown('<p class="demo-step">Step 3 · Run credit check</p>')
            score_hint = gr.Markdown("*Review the profile first, then run the credit check.*")
            with gr.Row():
                with gr.Column(scale=1):
                    score_btn = gr.Button("Check credit score", variant="primary", size="lg")
                    out_text = gr.Markdown()
                    out_explain = gr.Markdown()
                with gr.Column(scale=1):
                    out_plot = gr.Plot()

            demo.load(
                get_applicant_choices,
                [filter_gender, filter_location, filter_msme],
                [applicant_select, profile_out],
            )
            apply_filters_btn.click(
                get_applicant_choices,
                [filter_gender, filter_location, filter_msme],
                [applicant_select, profile_out],
            )
            filter_gender.change(
                get_applicant_choices,
                [filter_gender, filter_location, filter_msme],
                [applicant_select, profile_out],
            )
            filter_location.change(
                get_applicant_choices,
                [filter_gender, filter_location, filter_msme],
                [applicant_select, profile_out],
            )
            filter_msme.change(
                get_applicant_choices,
                [filter_gender, filter_location, filter_msme],
                [applicant_select, profile_out],
            )
            applicant_select.change(
                show_applicant_profile,
                applicant_select,
                [profile_out, score_hint, out_plot, out_text],
            )
            random_btn.click(
                pick_random_applicant,
                [filter_gender, filter_location, filter_msme],
                [applicant_select, profile_out, score_hint, out_plot, out_text],
            )
            score_btn.click(
                run_credit_score,
                applicant_select,
                [out_text, out_plot, out_explain],
            )

        with gr.TabItem("📈 Results"):
            gr.Markdown("**Objectives 2 & 3** — LoRA vs baselines: AUC-ROC, AUC-PR, F1 (§4.4.4).")
            comp_btn = gr.Button("View results", variant="primary")
            comp_cards = gr.HTML()
            comp_plot = gr.Plot()
            comp_note = gr.Markdown()
            comp_btn.click(model_comparison, None, [comp_plot, comp_cards, comp_note])

        with gr.TabItem("⚡ Efficiency"):
            gr.Markdown("**Objective 3** — Trainable parameters, training time, inference latency, memory (§3.6.2).")
            eff_btn = gr.Button("View efficiency", variant="primary")
            eff_plot = gr.Plot()
            eff_note = gr.Markdown()
            eff_btn.click(efficiency_dashboard, None, [eff_plot, eff_note])

        with gr.TabItem("⚖️ Fairness"):
            gr.Markdown("**Objective 4** — Performance across gender, location, age, income quartile, MSME (§3.6.3).")
            fair_btn = gr.Button("View fairness", variant="primary")
            fair_note = gr.Markdown()
            fair_plot = gr.Plot()
            fair_btn.click(fairness_dashboard, None, [fair_plot, fair_note])

        with gr.TabItem("🔍 Explainability"):
            gr.Markdown("**Objective 2** — LoRA feature attribution and RF SHAP for regulators (§4.4.9).")
            exp_btn = gr.Button("View explainability", variant="primary")
            exp_note = gr.Markdown()
            exp_plot = gr.Plot()
            exp_btn.click(explainability_tab, None, [exp_plot, exp_note])

        with gr.TabItem("📊 Dataset"):
            gr.Markdown("**Objective 1** — Synthetic alternative data framework (§3.3).")
            ds_btn = gr.Button("View dataset", variant="primary")
            ds_text = gr.Markdown()
            ds_table = gr.HTML()
            ds_btn.click(dataset_info, None, [ds_text, ds_table])

        with gr.TabItem("📜 Policy"):
            gr.Markdown("""
            ### Objective 5 — Policy alignment (§6.3, Appendix E)

            **Financial institutions & fintech (§6.3.1)**
            - Partner with EcoCash and OneMoney for consented alternative data access
            - Deploy LoRA adapters for cost-efficient scoring on modest hardware
            - Publish explainability summaries for applicants and regulators

            **Policymakers & regulators (§6.3.2)**
            - Establish regulatory sandboxes for alternative-data credit pilots (Appendix E.1)
            - Integrate fairness monitoring (gender, geography, income quartile) into RBZ oversight
            - Advance national credit data infrastructure (Appendix E.2)

            **Researchers (§6.3.3)**
            - Extend LoRA rank sensitivity and cross-validation (Appendix D)
            - Validate synthetic findings on real Zimbabwe fintech datasets when available

            ---
            **NDS1/NDS2 · National AI Strategy:** Financial inclusion >90% · MSME credit · Women & youth entrepreneurship · Responsible AI
            """)

    gr.Markdown("---")
    gr.Markdown("*Harare Institute of Technology · Supervisor: Eng A. Ndlovu · LoRA (Hu et al., 2021)*")

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, theme=THEME)