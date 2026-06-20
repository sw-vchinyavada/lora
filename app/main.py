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
from plotly.subplots import make_subplots

from app.demo_helpers import (
    DEMO_SCENARIOS,
    apply_demo_scenario_filters,
    build_applicant_snapshot,
    build_profile_html,
    empty_snapshot,
    load_demo_dataframe,
    resolve_applicant_list,
    scenario_extra_filter,
    filter_applicant_indices,
    applicant_choice_label,
)


# ─── Config ───────────────────────────────────────────────────────────────

CUSTOM_CSS = """
:root { --radius: 16px; --card-bg: rgba(255,255,255,0.95); }
.gradio-container { max-width: 1120px !important; margin: auto; }
.presentation-script {
  background: #fffbeb; border: 1px solid #fcd34d; border-left: 4px solid #f59e0b;
  border-radius: 12px; padding: 1rem 1.25rem; margin: 0.75rem 0 1rem;
  font-size: 0.95rem; line-height: 1.65; color: #422006;
}
.presentation-script h4 { margin: 0 0 0.5rem; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; color: #92400e; }
.presentation-script p { margin: 0.35rem 0; }
.chart-caption { font-size: 0.88rem; color: #64748b; margin: 0.25rem 0 0.75rem; line-height: 1.5; }
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
.applicant-snapshot {
  display: flex; align-items: stretch; gap: 1rem; flex-wrap: wrap;
  background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
  color: #fff; border-radius: 16px; padding: 1.1rem 1.35rem; margin: 0.75rem 0 1rem;
  box-shadow: 0 4px 20px rgba(15,23,42,0.18);
}
.snapshot-avatar {
  width: 56px; height: 56px; border-radius: 50%; background: rgba(255,255,255,0.15);
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.15rem;
  flex-shrink: 0; border: 2px solid rgba(255,255,255,0.25);
}
.snapshot-main { flex: 1; min-width: 200px; }
.snapshot-main h3 { margin: 0; font-size: 1.05rem; font-weight: 600; }
.snapshot-tagline { margin: 0.25rem 0 0.65rem; font-size: 0.88rem; opacity: 0.85; line-height: 1.45; }
.snapshot-metrics { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.snapshot-metric {
  background: rgba(255,255,255,0.12); border-radius: 8px; padding: 0.35rem 0.65rem;
  font-size: 0.78rem; white-space: nowrap;
}
.snapshot-metric strong { display: block; font-size: 0.95rem; font-weight: 700; margin-top: 0.1rem; }
.snapshot-empty { color: #64748b; text-align: center; padding: 1.25rem; font-size: 0.92rem;
  background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 14px; margin: 0.75rem 0 1rem; }
.demo-scenario-hint { font-size: 0.88rem; color: #64748b; margin: 0.35rem 0 0.75rem; line-height: 1.5; }
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

CHART_COLORS = ["#0284c7", "#059669", "#d97706", "#7c3aed", "#db2777"]
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#f8fafc",
    font=dict(size=14, family="Inter, system-ui, sans-serif", color="#0f172a"),
    margin=dict(l=56, r=32, t=88, b=72),
    title=dict(font=dict(size=17), x=0.5, xanchor="center"),
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5, font=dict(size=13)),
)

OVERVIEW_SCRIPT = """
<div class="presentation-script">
  <h4>Presentation script — Overview tab</h4>
  <p>Good morning. This dissertation addresses a core barrier to financial inclusion in Zimbabwe: roughly <strong>89 percent</strong> of adults lack access to formal credit, largely because they have no traditional credit bureau history.</p>
  <p>My aim is to develop and evaluate a <strong>LoRA-enhanced credit scoring system</strong> that uses alternative data — mobile money, utility payments, and digital behaviour — instead of conventional bureau records.</p>
  <p>The work is organised around <strong>five objectives</strong>: building a synthetic alternative-data framework; implementing LoRA scoring; measuring computational efficiency; assessing fairness across gender, location, age, income, and MSME status; and aligning recommendations with NDS1, NDS2, and the National AI Strategy.</p>
  <p>Research questions RQ1 through RQ4 map directly to the tabs in this demo — dataset design, LoRA performance, fairness, and policy alignment.</p>
</div>
"""

LIVE_DEMO_SCRIPT = """
<div class="presentation-script">
  <h4>Presentation script — Live Demo tab</h4>
  <p>I start with a <strong>demo scenario</strong> — for example a rural MSME woman — which narrows the pool to applicants aligned with NDS inclusion targets. The dropdown then lists real individuals from the synthetic dataset; each label shows gender, age, location, and mobile-money provider at a glance.</p>
  <p>The <strong>selected applicant preview</strong> card summarises the three signals an MFI officer cares about before scoring: mobile money activity, utility payment consistency, and digital engagement. The full profile below adds detail if the panel asks.</p>
  <p>When I click <strong>Check credit score</strong>, the LoRA model returns a score on the 300-to-850 scale, a default probability, and the top feature drivers for this individual — supporting both the lending decision and regulatory explainability.</p>
</div>
"""

DATASET_SCRIPT = """
<div class="presentation-script">
  <h4>Presentation script — Dataset tab</h4>
  <p>Objective 1 required a dataset that mirrors Zimbabwe's alternative-data landscape without exposing real customer records. I built a <strong>synthetic framework</strong> with mobile money transactions, utility and airtime consistency, digital commerce signals, and demographic attributes including gender, location, youth status, and MSME classification.</p>
  <p>The table below shows sample rows. The default rate reflects the class imbalance typical of credit portfolios — most applicants repay, but the model must still identify higher-risk cases.</p>
  <p>This synthetic approach lets me test methodology and fairness metrics before real fintech partnerships are in place — which aligns with the regulatory sandbox recommendations in Chapter 6.</p>
</div>
"""

POLICY_SCRIPT = """
<div class="presentation-script">
  <h4>Presentation script — Policy tab</h4>
  <p>Objective 5 connects the technical work to national priorities. For <strong>financial institutions</strong>, the recommendation is to partner with EcoCash and OneMoney for consented data access, deploy LoRA adapters on modest hardware, and publish explainability summaries for applicants.</p>
  <p>For <strong>policymakers and the Reserve Bank</strong>, I recommend regulatory sandboxes for alternative-data pilots, routine fairness monitoring by gender and geography, and investment in shared credit infrastructure.</p>
  <p>For <strong>researchers</strong>, the next step is validating these findings on real Zimbabwe fintech data and extending LoRA rank sensitivity analysis — both outlined in the appendices.</p>
  <p>Together, these measures support NDS targets on financial inclusion above 90 percent, MSME credit access, and responsible AI deployment.</p>
</div>
"""


def _presentation_script(title: str, paragraphs: list[str]) -> str:
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return f'<div class="presentation-script"><h4>{title}</h4>{body}</div>'


def _human_feature(name: str) -> str:
    labels = {
        "mm_txn_per_month": "Mobile money transactions / month",
        "mm_balance_volatility": "Mobile money balance volatility",
        "mm_weekend_usage_pct": "Weekend mobile money usage",
        "utility_payment_rate": "Utility payment rate",
        "util_water_consistency": "Water bill consistency",
        "util_electricity_consistency": "Electricity bill consistency",
        "airtime_consistency_score": "Airtime purchase consistency",
        "account_age_months": "Account age (months)",
        "is_urban": "Urban location",
        "age_group": "Age group",
        "social_media_usage": "Social media usage",
    }
    if name in labels:
        return labels[name]
    return name.replace("_", " ").title()


def _apply_chart_style(fig, height: int = 420, y_title: str = "Score") -> go.Figure:
    fig.update_layout(height=height, **CHART_LAYOUT)
    fig.update_xaxes(tickfont=dict(size=13), title_font=dict(size=14), gridcolor="#e2e8f0", linecolor="#cbd5e1")
    fig.update_yaxes(
        tickfont=dict(size=13), title=y_title, title_font=dict(size=14),
        gridcolor="#e2e8f0", linecolor="#cbd5e1", zerolinecolor="#e2e8f0",
    )
    return fig


def _load_demo_dataframe() -> pd.DataFrame:
    return load_demo_dataframe()


def get_applicant_choices(
    gender: str,
    location: str,
    msme: str,
    scenario_key: str = "browse_all",
):
    """Populate the applicant dropdown from filters and optional demo scenario."""
    try:
        choices, first_id, snapshot, profile = resolve_applicant_list(
            gender, location, msme, scenario_key,
        )
    except Exception as e:
        return (
            gr.Dropdown(choices=[], value=None, label="Choose applicant"),
            empty_snapshot(f"Could not load dataset: {e}"),
            '<div class="profile-card"><p style="margin:0;color:#64748b;">Dataset unavailable.</p></div>',
        )

    if not choices:
        return (
            gr.Dropdown(choices=[], value=None, label="Choose applicant"),
            snapshot,
            profile,
        )

    return (
        gr.Dropdown(choices=choices, value=first_id, label="Choose applicant", filterable=True),
        snapshot,
        profile,
    )


def apply_demo_scenario(scenario_key: str):
    """Set filters from a curated demo scenario and reload the applicant list."""
    gender, location, msme, hint = apply_demo_scenario_filters(scenario_key)
    dropdown, snapshot, profile = get_applicant_choices(gender, location, msme, scenario_key)
    return gender, location, msme, dropdown, snapshot, profile, hint


def show_applicant_profile(applicant_id: str):
    """Load snapshot and full profile — no model inference yet."""
    if not applicant_id:
        return (
            empty_snapshot(),
            '<div class="profile-card"><p style="margin:0;color:#64748b;">Choose an applicant from the dropdown above.</p></div>',
            "Select an applicant, review the preview, then run the credit check.",
            None,
            "",
        )
    try:
        df = _load_demo_dataframe()
        idx = int(applicant_id)
        row = df.iloc[idx]
    except Exception:
        return (
            empty_snapshot("Applicant not found."),
            '<div class="profile-card"><p style="margin:0;color:#64748b;">Applicant not found.</p></div>',
            "Select an applicant to enable scoring.",
            None,
            "",
        )
    return (
        build_applicant_snapshot(row, idx),
        build_profile_html(row, idx),
        "Preview loaded. Click **Check credit score** when ready.",
        None,
        "",
    )


def pick_random_applicant(
    gender: str,
    location: str,
    msme: str,
    scenario_key: str = "browse_all",
):
    """Pick a random applicant from the current filter — useful for panel demos."""
    extra = scenario_extra_filter(scenario_key)
    try:
        df = _load_demo_dataframe()
    except Exception as e:
        return (
            None,
            empty_snapshot(str(e)),
            '<div class="profile-card"><p style="margin:0;color:#64748b;">Dataset unavailable.</p></div>',
            "Select an applicant to enable scoring.",
            None,
            "",
        )

    indices = filter_applicant_indices(df, gender, location, msme, extra)
    if len(indices) == 0:
        return (
            None,
            empty_snapshot("No applicants match."),
            '<div class="profile-card"><p style="margin:0;color:#64748b;">No applicants match these filters.</p></div>',
            "Select an applicant to enable scoring.",
            None,
            "",
        )

    idx = int(np.random.choice(indices))
    choices = [(applicant_choice_label(df.loc[i], i), str(i)) for i in indices[:400]]
    row = df.loc[idx]
    return (
        gr.Dropdown(choices=choices, value=str(idx), label="Choose applicant", filterable=True),
        build_applicant_snapshot(row, idx),
        build_profile_html(row, idx),
        "Preview loaded. Click **Check credit score** when ready.",
        None,
        "",
    )


def run_credit_score(applicant_id: str):
    """Run LoRA inference after the user has reviewed the applicant profile."""
    if not applicant_id:
        return "Select an applicant and review their profile first.", None, "", ""

    out = load_model_and_prep()
    if out[0] is None:
        return (
            "⚠️ **Train models first**\n\n```bash\npython scripts/train.py --dataset zimbabwe_synthetic\n```",
            None,
            "",
            "",
        )

    from src.scoring import score_applicant_row

    try:
        result = score_applicant_row(int(applicant_id), artifacts=out[:3])
    except ValueError as e:
        return f"⚠️ **Model out of date** — retrain to match current features.\n\n`{e}`", None, "", ""

    if result is None:
        return "⚠️ **Could not score applicant.**", None, "", ""

    proba = result["default_probability"]
    score = result["score"]
    risk = "Low ✓" if proba < 0.3 else "Medium" if proba < 0.6 else "High ✗"
    risk_color = "#10b981" if proba < 0.3 else "#f59e0b" if proba < 0.6 else "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={"text": "Credit Score (300–850)", "font": {"size": 16}},
        gauge={
            "axis": {"range": [300, 850], "tickwidth": 2, "tickmode": "linear", "dtick": 100, "tickfont": {"size": 12}},
            "bar": {"color": "#0284c7", "thickness": 0.28},
            "steps": [
                {"range": [300, 580], "color": "#fecaca"},
                {"range": [580, 670], "color": "#fde68a"},
                {"range": [670, 850], "color": "#bbf7d0"},
            ],
            "threshold": {"line": {"color": risk_color, "width": 4}, "thickness": 0.85, "value": score},
        },
        number={"suffix": " / 850", "font": {"size": 42, "color": "#0f172a"}},
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=40, t=70, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=14),
    )
    fig.add_annotation(
        x=0.5, y=-0.08, xref="paper", yref="paper", showarrow=False,
        text="<b>Red zone</b> 300–580 · <b>Amber</b> 580–670 · <b>Green</b> 670–850",
        font=dict(size=12, color="#64748b"),
    )

    explain = ""
    top = result.get("top_drivers") or []
    driver_lines = []
    if top:
        parts = []
        for t in top[:4]:
            feat = _human_feature(t.get("feature", ""))
            direction = t.get("direction", "")
            impact = t.get("impact", t.get("importance", 0))
            parts.append(f"*{feat}* ({direction}, Δ{abs(float(impact)):.3f})")
            driver_lines.append(
                f"<strong>{feat}</strong> pushes the score {direction.lower()} "
                f"(impact {abs(float(impact)):.3f})."
            )
        explain = "**LoRA drivers for this applicant:** " + " → ".join(parts)

    result_text = f"### {risk}\n**Default probability:** {proba:.1%}  |  **Score:** {score}"

    risk_word = "low" if proba < 0.3 else "medium" if proba < 0.6 else "high"
    script = _presentation_script(
        "Presentation script — this applicant's score",
        [
            f"The model assigns a credit score of <strong>{score}</strong> out of 850, "
            f"with a default probability of <strong>{proba:.1%}</strong>. "
            f"That places this applicant in the <strong>{risk_word} risk</strong> band.",
            "The gauge uses the same colour logic a panel would expect: red below 580, amber up to 670, and green above that.",
        ] + (driver_lines[:3] if driver_lines else [
            "The top feature drivers show which alternative-data signals most influenced this decision — "
            "supporting both the loan officer and regulatory explainability requirements."
        ]),
    )
    return result_text, fig, explain, script


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
        return None, "", "Run training first: `python scripts/train.py`", ""

    with open(path) as f:
        r = json.load(f)

    lora = r.get("lora", {})
    pct = 100 * lora.get("trainable_params", 0) / max(lora.get("total_params", 1), 1) if lora else 0
    cards_html = ""
    if lora:
        cards_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin-bottom: 1rem;">
          <div style="background: linear-gradient(145deg,#f8fafc,#f1f5f9); border-radius: 12px; padding: 1rem; border-left: 4px solid #0284c7;">
            <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase;">AUC-ROC</div>
            <div style="font-size:1.6rem; font-weight:700;">{lora.get('auc_roc', 0):.3f}</div>
            <div style="font-size:0.75rem; color:#64748b;">Ranking ability (0–1)</div>
          </div>
          <div style="background: linear-gradient(145deg,#f8fafc,#f1f5f9); border-radius: 12px; padding: 1rem; border-left: 4px solid #059669;">
            <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase;">AUC-PR</div>
            <div style="font-size:1.6rem; font-weight:700;">{lora.get('auc_pr', 0):.3f}</div>
            <div style="font-size:0.75rem; color:#64748b;">Imbalanced defaults</div>
          </div>
          <div style="background: linear-gradient(145deg,#f8fafc,#f1f5f9); border-radius: 12px; padding: 1rem; border-left: 4px solid #d97706;">
            <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase;">F1</div>
            <div style="font-size:1.6rem; font-weight:700;">{lora.get('f1', 0):.3f}</div>
            <div style="font-size:0.75rem; color:#64748b;">Precision–recall balance</div>
          </div>
          <div style="background: linear-gradient(145deg,#f8fafc,#f1f5f9); border-radius: 12px; padding: 1rem; border-left: 4px solid #7c3aed;">
            <div style="font-size:0.75rem; color:#64748b; text-transform:uppercase;">Params trained</div>
            <div style="font-size:1.6rem; font-weight:700;">~{pct:.1f}%</div>
            <div style="font-size:0.75rem; color:#64748b;">LoRA efficiency</div>
          </div>
        </div>
        """

    rows = [{"Model": k.replace("_", " ").title(), "AUC-ROC": v["auc_roc"], "AUC-PR": v.get("auc_pr", 0), "F1": v["f1"]} for k, v in r.items()]
    df = pd.DataFrame(rows)
    y_cols = [c for c in ["AUC-ROC", "AUC-PR", "F1"] if c in df.columns]
    fig = px.bar(
        df, x="Model", y=y_cols, barmode="group",
        color_discrete_sequence=CHART_COLORS[:len(y_cols)],
        title="Model comparison — LoRA vs baselines",
        labels={"value": "Score (0–1)", "variable": "Metric"},
        text_auto=".3f",
    )
    fig.update_traces(textfont_size=12, textposition="outside", cliponaxis=False)
    _apply_chart_style(fig, height=440, y_title="Score (0 = worst, 1 = best)")
    fig.update_layout(
        xaxis_title="Model",
        yaxis=dict(range=[0, min(1.08, df[y_cols].max().max() * 1.15 + 0.05)]),
    )

    eff = ""
    if lora and "trainable_params" in lora:
        eff = (
            f"**LoRA efficiency (Objective 3):** {lora['trainable_params']:,} / {lora['total_params']:,} params "
            f"(~{100 * lora['trainable_params'] / lora['total_params']:.1f}%) · "
            f"Train: {lora.get('train_time_sec', 0):.0f}s · "
            f"Inference: {lora.get('inference_latency_ms', '—')}ms · "
            f"Peak memory: {lora.get('peak_memory_mb', '—')} MB"
        )

    best_auc = df.loc[df["AUC-ROC"].idxmax(), "Model"]
    best_auc_val = df["AUC-ROC"].max()
    lora_row = df[df["Model"] == "Lora"]
    lora_auc = lora_row["AUC-ROC"].iloc[0] if len(lora_row) else 0
    lora_f1 = lora_row["F1"].iloc[0] if len(lora_row) else 0
    script = _presentation_script(
        "Presentation script — Results chart",
        [
            "This grouped bar chart compares four models on three standard credit-scoring metrics, all on a zero-to-one scale.",
            "<strong>AUC-ROC</strong> measures how well each model ranks risky applicants above safe ones. "
            "<strong>AUC-PR</strong> is more informative when defaults are rare. "
            "<strong>F1</strong> balances precision and recall at the chosen threshold.",
            f"In this run, <strong>{best_auc}</strong> achieves the highest AUC-ROC at <strong>{best_auc_val:.3f}</strong>. "
            f"LoRA scores <strong>{lora_auc:.3f}</strong> on AUC-ROC and <strong>{lora_f1:.3f}</strong> on F1 — "
            "the key point for Objective 3 is that this performance is reached while training only about "
            f"<strong>{pct:.1f}%</strong> of model parameters, which matters for deployment on modest MFI hardware.",
        ],
    )
    return fig, cards_html, eff, script


def efficiency_dashboard():
    path = Path("results/metrics/training_results.json")
    if not path.exists():
        return None, "Run training first: `python scripts/train.py`", ""

    with open(path) as f:
        r = json.load(f)
    lora = r.get("lora", {})
    if not lora:
        return None, "LoRA results not found.", ""

    trainable = lora.get("trainable_params", 0)
    total = lora.get("total_params", 1)
    frozen = max(total - trainable, 0)
    pct = 100 * trainable / max(total, 1)
    train_min = lora.get("train_time_sec", 0) / 60
    infer_ms = lora.get("inference_latency_ms", 0)
    peak_mb = lora.get("peak_memory_mb", 0)

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.42, 0.58],
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=("Parameters trained vs frozen", "Runtime metrics (actual values labelled)"),
    )
    fig.add_trace(
        go.Pie(
            labels=["Trainable (LoRA)", "Frozen (pre-trained)"],
            values=[trainable, frozen],
            marker=dict(colors=["#0284c7", "#cbd5e1"]),
            textinfo="label+percent",
            textfont=dict(size=13),
            hole=0.45,
        ),
        row=1, col=1,
    )
    metric_labels = ["Training time (min)", "Inference (ms)", "Peak memory (MB)"]
    metric_values = [train_min, infer_ms, peak_mb]
    metric_text = [f"{train_min:.1f} min", f"{infer_ms:.1f} ms", f"{peak_mb:.2f} MB"]
    fig.add_trace(
        go.Bar(
            x=metric_labels, y=metric_values,
            marker_color=["#7c3aed", "#059669", "#d97706"],
            text=metric_text, textposition="outside", textfont=dict(size=13),
            cliponaxis=False,
        ),
        row=1, col=2,
    )
    fig.update_layout(
        height=420,
        title=dict(text="Computational efficiency — Objective 3", x=0.5, xanchor="center", font=dict(size=17)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font=dict(size=14, family="Inter, system-ui, sans-serif"),
        margin=dict(l=40, r=40, t=90, b=60),
        showlegend=False,
    )
    fig.update_yaxes(title_text="Value (see bar labels for units)", row=1, col=2, gridcolor="#e2e8f0")
    fig.update_xaxes(tickangle=-12, row=1, col=2)

    note = (
        f"**Objective 3:** LoRA trains only **~{pct:.1f}%** of parameters ({trainable:,} of {total:,}) "
        f"while maintaining competitive AUC. Inference latency **{infer_ms:.1f} ms** per applicant supports "
        f"real-time MFI deployment in resource-constrained settings."
    )
    script = _presentation_script(
        "Presentation script — Efficiency charts",
        [
            "Objective 3 asks whether LoRA reduces compute without sacrificing scoring quality. "
            "The donut chart on the left makes the parameter story easy to explain: only the blue slice is updated during fine-tuning; "
            f"the rest of DistilBERT stays frozen. In this run that is about <strong>{pct:.1f}%</strong> of weights.",
            f"The bar chart on the right reports three deployment metrics with their real units: "
            f"training took <strong>{train_min:.1f} minutes</strong>, each inference call takes about "
            f"<strong>{infer_ms:.1f} milliseconds</strong>, and peak memory was <strong>{peak_mb:.2f} MB</strong>.",
            "Together, these figures support the claim that the system can run on modest hardware at branch level — "
            "which is important for MFIs serving rural and peri-urban clients.",
        ],
    )
    return fig, note, script


def fairness_dashboard():
    path = Path("results/metrics/fairness_results.json")
    if not path.exists():
        return None, "Run training first to see fairness metrics (§3.6.3).", ""

    with open(path) as f:
        data = json.load(f)
    lora = data.get("lora", {})
    groups = lora.get("groups", {})
    fm = lora.get("fairness_metrics", {})

    if not groups:
        return None, "Use **zimbabwe_synthetic** dataset for gender, location, age, income quartile, MSME subgroups (§3.6.3).", ""

    attrs = [a for a in groups.keys() if groups[a]]
    n_attrs = len(attrs)
    n_cols = 2
    n_rows = (n_attrs + 1) // 2
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[a.replace("_", " ").title() for a in attrs],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    script_bits = []
    for i, attr in enumerate(attrs):
        row, col = i // n_cols + 1, i % n_cols + 1
        gdata = groups[attr]
        gnames = [str(g).replace("_", " ").title() for g in gdata.keys()]
        aucs = [gdata[g].get("auc", 0) for g in gdata.keys()]
        tprs = [gdata[g].get("tpr", 0) for g in gdata.keys()]
        fig.add_trace(
            go.Bar(name="AUC", x=gnames, y=aucs, marker_color="#0284c7",
                   text=[f"{v:.2f}" for v in aucs], textposition="outside", legendgroup="AUC",
                   showlegend=(i == 0), offsetgroup="auc"),
            row=row, col=col,
        )
        fig.add_trace(
            go.Bar(name="TPR", x=gnames, y=tprs, marker_color="#059669",
                   text=[f"{v:.2f}" for v in tprs], textposition="outside", legendgroup="TPR",
                   showlegend=(i == 0), offsetgroup="tpr"),
            row=row, col=col,
        )
        auc_spread = max(aucs) - min(aucs) if aucs else 0
        script_bits.append(
            f"For <strong>{attr.replace('_', ' ')}</strong>, AUC ranges from "
            f"<strong>{min(aucs):.2f}</strong> to <strong>{max(aucs):.2f}</strong> "
            f"(spread {auc_spread:.2f})."
        )

    fig.update_layout(
        height=max(420, 220 * n_rows),
        title=dict(text="Fairness by protected group — Objective 4", x=0.5, xanchor="center", font=dict(size=17)),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font=dict(size=13, family="Inter, system-ui, sans-serif"),
        margin=dict(l=48, r=32, t=100, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="Score (0–1)", range=[0, 1.08], gridcolor="#e2e8f0")
    fig.add_annotation(
        x=0.5, y=-0.06, xref="paper", yref="paper", showarrow=False,
        text="<b>AUC</b> = ranking quality per group · <b>TPR</b> = true positive rate (defaults correctly flagged)",
        font=dict(size=12, color="#64748b"),
    )

    parts = []
    for kind, d in fm.items():
        if d:
            parts.append(f"**{kind.replace('_', ' ').title()}:** " + " · ".join(f"{k.split('_')[0]}: {v:.3f}" for k, v in list(d.items())[:3]))
    note = "\n\n".join(parts) if parts else "Fairness metrics computed."

    script = _presentation_script(
        "Presentation script — Fairness charts",
        [
            "Objective 4 examines whether the model performs consistently across protected groups defined in the dissertation: "
            "gender, location, age band, income quartile, and MSME status.",
            "Each small chart shows two bars per subgroup. <strong>AUC</strong> tells us whether the model ranks risk equally well within that group. "
            "<strong>TPR</strong> — true positive rate — shows what fraction of actual defaults the model catches; large gaps here can indicate disparate impact.",
        ] + script_bits[:4] + [
            "These results inform mitigation strategies discussed in Chapter 5 — such as threshold adjustment and ongoing monitoring under RBZ oversight."
        ],
    )
    return fig, note, script


def explainability_tab():
    path = Path("results/metrics/feature_importance.json")
    if not path.exists():
        return None, "Run training first. LoRA attribution and RF SHAP computed in §4.4.9.", ""

    with open(path) as f:
        data = json.load(f)
    fi = data.get("lora_global_attribution") or data.get("feature_importance") or data.get("shap_importance", [])
    if not fi:
        return None, "No explainability data.", ""

    df = pd.DataFrame(fi).head(10).copy()
    y_col = "mean_abs_impact" if "mean_abs_impact" in df.columns else (
        "importance" if "importance" in df.columns else "mean_abs_shap"
    )
    df["label"] = df["feature"].map(_human_feature)
    df = df.sort_values(y_col, ascending=True)

    fig = px.bar(
        df, x=y_col, y="label", orientation="h",
        color=y_col, color_continuous_scale=["#bae6fd", "#0284c7", "#075985"],
        title="Top 10 global feature drivers — LoRA attribution",
        labels={y_col: "Mean absolute impact", "label": "Feature"},
        text=df[y_col].map(lambda v: f"{v:.4f}"),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=460)
    _apply_chart_style(fig, height=460, y_title="")
    fig.update_layout(margin=dict(l=180, r=60, t=88, b=60))

    top3 = df.tail(3)["label"].tolist()[::-1]
    note = (
        "**Live Demo** shows *per-applicant* LoRA drivers. This chart shows aggregate attribution across the test set. "
        "RF SHAP values are retained for baseline comparison (§4.4.9)."
    )
    script = _presentation_script(
        "Presentation script — Explainability chart",
        [
            "Regulators and applicants need to understand why a score was assigned. "
            "This horizontal bar chart ranks the ten features with the largest average impact on LoRA predictions across the evaluation set.",
            f"The strongest global drivers in this run are <strong>{top3[0]}</strong>, "
            f"<strong>{top3[1] if len(top3) > 1 else '—'}</strong>, and "
            f"<strong>{top3[2] if len(top3) > 2 else '—'}</strong> — "
            "mostly utility consistency, mobile-money behaviour, and location-related signals rather than opaque latent features.",
            "In the Live Demo tab I complement this aggregate view with per-applicant drivers, "
            "which is what an MFI officer would see at decision time.",
        ],
    )
    return fig, note, script


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
    script = DATASET_SCRIPT.replace(
        "The table below shows sample rows.",
        f"The table below shows sample rows from <strong>{n:,}</strong> synthetic applicants, "
        f"including <strong>{defs:,}</strong> simulated defaults ({pct:.1f}%).",
    )
    summary = f"{desc}\n\n**{n:,}** samples  ·  **{defs:,}** defaults ({pct:.1f}%)  ·  **{len(df.columns)-1}** features\n\n{script}"
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

    with gr.Accordion("🎤 Presentation scripts — read aloud while practising", open=True):
        gr.Markdown(
            "Each tab includes a **yellow script box** with wording you can read during the panel demo. "
            "Chart tabs also generate a **dynamic script** after you click the view button, using your trained model numbers."
        )
        gr.Markdown(DEMO_FLOW)

    with gr.Tabs() as tabs:
        with gr.TabItem("📋 Overview"):
            gr.HTML(OVERVIEW_SCRIPT)
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
            gr.HTML(LIVE_DEMO_SCRIPT)
            gr.Markdown(
                "Walk through a realistic credit inquiry: **pick a scenario → choose an applicant → preview → run the LoRA score**."
            )

            gr.Markdown('<p class="demo-step">Step 1 · Demo scenario</p>')
            scenario_select = gr.Dropdown(
                choices=[(v["label"], k) for k, v in DEMO_SCENARIOS.items()],
                value="browse_all",
                label="Presentation scenario",
                info="Curated groups for the panel — auto-fills filters below",
            )
            scenario_hint = gr.Markdown(
                DEMO_SCENARIOS["browse_all"]["hint"],
                elem_classes=["demo-scenario-hint"],
            )

            gr.Markdown('<p class="demo-step">Step 2 · Refine & choose applicant</p>')
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
                    label="Choose applicant",
                    choices=[],
                    filterable=True,
                    info="Type to search — each row shows key alternative-data signals",
                    scale=4,
                )
                random_btn = gr.Button("🎲 Random", scale=1)

            gr.Markdown('<p class="demo-step">Step 3 · Selected applicant preview</p>')
            snapshot_out = gr.HTML(empty_snapshot())

            gr.Markdown('<p class="demo-step">Step 4 · Full profile (optional detail)</p>')
            profile_out = gr.HTML(
                '<div class="profile-card"><p style="margin:0;color:#64748b;">Select a scenario and applicant to load their profile.</p></div>'
            )

            gr.Markdown('<p class="demo-step">Step 5 · Run credit check</p>')
            score_hint = gr.Markdown("*Review the preview, then run the credit check.*")
            with gr.Row():
                with gr.Column(scale=1):
                    score_btn = gr.Button("Check credit score", variant="primary", size="lg")
                    out_text = gr.Markdown()
                    out_explain = gr.Markdown()
                with gr.Column(scale=1):
                    out_plot = gr.Plot(label="Credit score gauge")
            score_script = gr.HTML("")

            filter_inputs = [filter_gender, filter_location, filter_msme, scenario_select]

            demo.load(
                get_applicant_choices,
                filter_inputs,
                [applicant_select, snapshot_out, profile_out],
            )
            scenario_select.change(
                apply_demo_scenario,
                scenario_select,
                [filter_gender, filter_location, filter_msme, applicant_select, snapshot_out, profile_out, scenario_hint],
            )
            for filt in (filter_gender, filter_location, filter_msme):
                filt.change(
                    get_applicant_choices,
                    filter_inputs,
                    [applicant_select, snapshot_out, profile_out],
                )
            applicant_select.change(
                show_applicant_profile,
                applicant_select,
                [snapshot_out, profile_out, score_hint, out_plot, out_text],
            )
            random_btn.click(
                pick_random_applicant,
                filter_inputs,
                [applicant_select, snapshot_out, profile_out, score_hint, out_plot, out_text],
            )
            score_btn.click(
                run_credit_score,
                applicant_select,
                [out_text, out_plot, out_explain, score_script],
            )

        with gr.TabItem("📈 Results"):
            gr.Markdown(
                "**Objectives 2 & 3** — LoRA vs baselines on AUC-ROC, AUC-PR, and F1. "
                "Higher bars are better; all metrics are on a 0–1 scale."
            )
            comp_btn = gr.Button("View results", variant="primary")
            comp_cards = gr.HTML()
            comp_script = gr.HTML("")
            comp_plot = gr.Plot(label="Model comparison chart")
            comp_note = gr.Markdown()
            comp_btn.click(model_comparison, None, [comp_plot, comp_cards, comp_note, comp_script])

        with gr.TabItem("⚡ Efficiency"):
            gr.Markdown(
                "**Objective 3** — How much of the model is actually trained, and how fast is inference? "
                "Left: parameter split. Right: runtime with labelled units."
            )
            eff_btn = gr.Button("View efficiency", variant="primary")
            eff_script = gr.HTML("")
            eff_plot = gr.Plot(label="Efficiency dashboard")
            eff_note = gr.Markdown()
            eff_btn.click(efficiency_dashboard, None, [eff_plot, eff_note, eff_script])

        with gr.TabItem("⚖️ Fairness"):
            gr.Markdown(
                "**Objective 4** — Separate charts per protected attribute. "
                "Compare **AUC** (ranking quality) and **TPR** (defaults caught) across subgroups."
            )
            fair_btn = gr.Button("View fairness", variant="primary")
            fair_script = gr.HTML("")
            fair_plot = gr.Plot(label="Fairness by subgroup")
            fair_note = gr.Markdown()
            fair_btn.click(fairness_dashboard, None, [fair_plot, fair_note, fair_script])

        with gr.TabItem("🔍 Explainability"):
            gr.Markdown(
                "**Objective 2** — Global LoRA feature attribution (horizontal bars = easier to read). "
                "Per-applicant drivers appear in **Live Demo**."
            )
            exp_btn = gr.Button("View explainability", variant="primary")
            exp_script = gr.HTML("")
            exp_plot = gr.Plot(label="Feature attribution chart")
            exp_note = gr.Markdown()
            exp_btn.click(explainability_tab, None, [exp_plot, exp_note, exp_script])

        with gr.TabItem("📊 Dataset"):
            gr.HTML(DATASET_SCRIPT)
            gr.Markdown("**Objective 1** — Synthetic alternative data framework (§3.3).")
            ds_btn = gr.Button("View dataset", variant="primary")
            ds_text = gr.Markdown()
            ds_table = gr.HTML()
            ds_btn.click(dataset_info, None, [ds_text, ds_table])

        with gr.TabItem("📜 Policy"):
            gr.HTML(POLICY_SCRIPT)
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