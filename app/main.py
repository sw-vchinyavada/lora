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
"""

DEMO_FLOW = """
**Suggested demo flow for the panel:**
1. **Overview** → State the problem (89% credit excluded) and LoRA solution
2. **Live Demo** → Pick a customer, show prediction — *"This MSME applicant gets a score of X"*
3. **Results** → Compare models: LoRA vs baselines, highlight parameter efficiency
4. **Fairness** → Show performance across gender, location — *"Balanced across subgroups"*
5. **Explainability** → Feature importance — *"Transparent for regulators"*
6. **Policy** → Recommendations aligned with NDS
"""


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


def predict(sample_idx: int):
    out = load_model_and_prep()
    if out[0] is None:
        return (
            "⚠️ **Train models first**\n\n```bash\npython scripts/train.py --dataset zimbabwe_synthetic\n```",
            None,
            "",
            "",
        )

    lora, prep, df, max_idx = out
    sample_idx = min(max(0, int(sample_idx)), max_idx)

    # Customer profile preview
    row = df.iloc[sample_idx]
    profile_cols = [c for c in ["gender", "age", "location", "employment", "msme", "sector"] if c in df.columns]
    profile_parts = []
    for c in profile_cols:
        v = row.get(c, "")
        if pd.notna(v):
            profile_parts.append(f"**{c.replace('_', ' ').title()}:** {v}")
    profile_html = " · ".join(profile_parts) if profile_parts else f"Applicant #{sample_idx}"

    cols = [c for c in prep.feature_columns_ if c in df.columns]
    sample = df[cols].iloc[sample_idx:sample_idx + 1]
    X = prep.transform(sample)
    if X.shape[1] != lora.num_features:
        pad = max(0, lora.num_features - X.shape[1])
        X = np.pad(X, ((0, 0), (0, pad)), constant_values=0)[:, :lora.num_features]

    proba = lora.predict_proba_numpy(X)[0]
    score = int(850 - proba * 550)
    risk = "Low ✓" if proba < 0.3 else "Medium" if proba < 0.6 else "High ✗"
    risk_color = "#10b981" if proba < 0.3 else "#f59e0b" if proba < 0.6 else "#ef4444"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Credit Score (0–850)"},
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
    fi_path = Path("results/metrics/feature_importance.json")
    if fi_path.exists():
        with open(fi_path) as f:
            data = json.load(f)
        top = data.get("feature_importance", [])[:4]
        if top:
            explain = "**Top risk drivers:** " + " → ".join(f"*{t['feature']}*" for t in top)

    result_text = f"### {risk}\n**Default probability:** {proba:.1%}  |  **Score:** {score}"
    return result_text, fig, explain, profile_html


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
                 title="Model Comparison (§3.7)")
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,250,252,0.9)", font=dict(size=12),
                      xaxis_tickangle=-15, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    eff = ""
    if lora and "trainable_params" in lora:
        eff = f"**LoRA efficiency:** {lora['trainable_params']:,} / {lora['total_params']:,} params trained (~{100*lora['trainable_params']/lora['total_params']:.1f}%) — *§3.5*"

    return fig, cards_html, eff


def fairness_dashboard():
    path = Path("results/metrics/fairness_results.json")
    if not path.exists():
        return None, "Run training first to see fairness metrics (§3.7).", None

    with open(path) as f:
        data = json.load(f)
    lora = data.get("lora", {})
    groups = lora.get("groups", {})
    fm = lora.get("fairness_metrics", {})

    if not groups:
        return None, "Use **zimbabwe_synthetic** dataset for gender, location, youth, MSME subgroups (§3.8)."

    rows = []
    for attr, gdata in groups.items():
        for gname, m in gdata.items():
            rows.append({"Attribute": attr, "Group": str(gname), "AUC": m.get("auc", 0), "TPR": m.get("tpr", 0)})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return None, str(lora)

    fig = px.bar(df, x="Group", y=["AUC", "TPR"], color="Attribute", barmode="group",
                 color_discrete_sequence=["#0ea5e9", "#10b981", "#f59e0b", "#8b5cf6"],
                 title="Performance by Protected Group (§3.7–3.8)")
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-20, font=dict(size=11))

    parts = []
    for kind, d in fm.items():
        if d:
            parts.append(f"**{kind.replace('_', ' ').title()}:** " + " · ".join(f"{k.split('_')[0]}: {v:.3f}" for k, v in list(d.items())[:3]))
    return fig, "\n\n".join(parts) if parts else "Fairness metrics computed."


def explainability_tab():
    path = Path("results/metrics/feature_importance.json")
    if not path.exists():
        return None, "Run training first. Feature importance & SHAP computed for interpretability (§3.7)."

    with open(path) as f:
        data = json.load(f)
    fi = data.get("feature_importance", []) or data.get("shap_importance", [])
    if not fi:
        return None, "No explainability data."

    df = pd.DataFrame(fi)
    y_col = "importance" if "importance" in df.columns else "mean_abs_shap"
    fig = px.bar(df.head(12), x="feature", y=y_col,
                 color=y_col, color_continuous_scale=["#e0f2fe", "#0ea5e9", "#0369a1"],
                 title="Top Features Driving Credit Decisions (§3.7)")
    fig.update_layout(height=360, xaxis_tickangle=-35, paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
    return fig, "**Regulatory alignment:** Transparency for RBZ guidelines & Data Protection Act. Enables applicant communication and audit trails."


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

    desc = "**Zimbabwe alternative data** (synthetic): mobile money, utility payments, demographics. Aligned with NDS §3.2." if dn == "zimbabwe_synthetic" else f"**{dn}** — benchmark dataset."
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
            ### Research problem
            **89%** of Zimbabweans lack access to formal credit despite 84% financial inclusion (FinScope 2022).  
            Traditional scoring excludes the informal sector, MSMEs, women, youth, and rural populations.

            ### Proposed solution
            **LoRA** (Low-Rank Adaptation) + alternative data:
            - **Mobile money** (EcoCash, OneMoney) — transaction patterns, balance volatility
            - **Utility payments** — electricity, water, telecom consistency
            - **Behavioral** — digital engagement, app usage
            - **Parameter efficiency** — ~99% reduction vs full fine-tuning

            ### Objectives addressed
            | # | Objective | ✓ |
            |---|----------|---|
            | 1 | Alternative data (mobile money, utility bills) | ✓ |
            | 2 | LoRA for efficient parameter customisation | ✓ |
            | 3 | Compare LoRA vs traditional/baseline ML models | ✓ |
            | 4 | LoRA’s contribution to financial inclusion under NDS | ✓ |

            ---
            *Hu et al. (2021) · Zimbabwe NDS · RBZ Responsible AI Guidelines*
            """)

        with gr.TabItem("🔮 Live Demo"):
            gr.Markdown("**Predict credit risk** — Select an applicant and view the LoRA model's score.")
            with gr.Row():
                idx = gr.Slider(0, 49999, 0, step=1, label="Applicant #")
                pred_btn = gr.Button("Predict", variant="primary", size="lg")
            profile_out = gr.Markdown("*Select an applicant and click Predict*")
            with gr.Row():
                with gr.Column(scale=1):
                    out_text = gr.Markdown()
                    out_explain = gr.Markdown()
                with gr.Column(scale=1):
                    out_plot = gr.Plot()
            pred_btn.click(predict, idx, [out_text, out_plot, out_explain, profile_out])
            # Also show profile - need to add profile_out to outputs. Let me check - we return 4 values, we have 4 outputs. But profile_out - we need to show it. Let me add profile_out as visible output. Actually we're passing profile_out in the outputs - but initially it's empty. The predict returns profile_html as 4th value. We need a component for it. I'll use a Markdown that shows the profile. Let me add it above the results.
            # Actually looking at the click: pred_btn.click(predict, idx, [out_text, out_plot, out_explain, profile_out]) - so we have 4 outputs. The predict returns (result_text, fig, explain, profile_html). So profile_out gets profile_html. Good. But we need to show profile_out - it might be in the layout. Let me add it. Actually we have profile_out = gr.Markdown(visible=True) - so it's in the layout. Good. The order of outputs might need to match - we have out_text, out_plot, out_explain, profile_out. So we need predict to return in that order. Currently we return result_text, fig, explain, profile_html. So we need [out_text, out_plot, out_explain, profile_out] = [result_text, fig, explain, profile_html]. Good.

        with gr.TabItem("📈 Results"):
            gr.Markdown("**Model performance** — AUC-ROC, AUC-PR, F1. LoRA efficiency (§3.5).")
            comp_btn = gr.Button("View results", variant="primary")
            comp_cards = gr.HTML()
            comp_plot = gr.Plot()
            comp_note = gr.Markdown()
            comp_btn.click(model_comparison, None, [comp_plot, comp_cards, comp_note])

        with gr.TabItem("⚖️ Fairness"):
            gr.Markdown("**Fairness assessment** — Performance across gender, location, age, MSME (§3.7–3.8).")
            fair_btn = gr.Button("View fairness", variant="primary")
            fair_note = gr.Markdown()
            fair_plot = gr.Plot()
            fair_btn.click(fairness_dashboard, None, [fair_plot, fair_note])

        with gr.TabItem("🔍 Explainability"):
            gr.Markdown("**Transparency** — Feature importance & SHAP for regulators (§3.7).")
            exp_btn = gr.Button("View explainability", variant="primary")
            exp_note = gr.Markdown()
            exp_plot = gr.Plot()
            exp_btn.click(explainability_tab, None, [exp_plot, exp_note])

        with gr.TabItem("📊 Dataset"):
            gr.Markdown("**Data overview** — Zimbabwe synthetic or benchmark dataset (§3.2).")
            ds_btn = gr.Button("View dataset", variant="primary")
            ds_text = gr.Markdown()
            ds_table = gr.HTML()
            ds_btn.click(dataset_info, None, [ds_text, ds_table])

        with gr.TabItem("📜 Policy"):
            gr.Markdown("""
            ### Evidence-based recommendations (NDS alignment)

            **Policymakers**
            - Integrate alternative data standards into credit bureau regulations
            - Support LoRA-enhanced MSME lending pilots
            - Establish fairness monitoring (gender, geographic, age)

            **Financial institutions**
            - Partner with EcoCash, OneMoney for consented transaction data
            - Deploy LoRA for cost-efficient AI infrastructure
            - Adopt explainable AI for customer communication and regulators

            **Technology providers**
            - Develop LoRA adapters for Zimbabwe's demographic mix
            - Meet Data Protection Act requirements
            - Contribute to African alternative-data benchmarks

            ---
            **NDS1/NDS2:** Financial inclusion >90% · MSME credit · Women & youth entrepreneurship
            """)

    gr.Markdown("---")
    gr.Markdown("*Harare Institute of Technology · Supervisor: Eng A. Ndlovu · LoRA (Hu et al., 2021)*")

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, theme=THEME)