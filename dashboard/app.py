"""
Dashboard Streamlit - Optimisation du ROI Marketing
====================================================
Cible utilisateur : CMO / Responsable Marketing / Direction Financiere.

Lancement :
    cd "Projet Data Science"
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (  # noqa: E402
    ALL_FEATURES,
    INFLUENCER_CATEGORIES,
    NUMERIC_FEATURES,
    TARGET,
)

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="ROI Marketing — Dashboard CMO",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette centrale
BLUE_DARK   = "#1e3a5f"
BLUE_MED    = "#2563eb"
BLUE_LIGHT  = "#dbeafe"
RED         = "#e74c3c"
GREEN       = "#10b981"
AMBER       = "#f59e0b"
GRAY_BG     = "#f1f5f9"
CARD_BG     = "#ffffff"
TEXT_DARK   = "#1e293b"
TEXT_MUTED  = "#64748b"

PLOTLY_TEMPLATE = "plotly_white"
CHART_FONT = dict(family="Inter, sans-serif", size=13, color=TEXT_DARK)

MODEL_COLORS = {
    "Linear Regression":  "#6366f1",
    "Random Forest":      "#f59e0b",
    "XGBoost":            "#10b981",
    "MLP (Deep Learning)":"#e74c3c",
}

# =============================================================================
# CSS
# =============================================================================

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* --- App background --- */
[data-testid="stAppViewContainer"] > .main {{
    background: {GRAY_BG};
}}

/* --- Sidebar dark gradient --- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {BLUE_DARK} 0%, #0f2347 100%) !important;
}}
[data-testid="stSidebar"] * {{
    color: #e2e8f0 !important;
}}
[data-testid="stSidebar"] .stRadio > label {{
    color: #94a3b8 !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.12) !important;
}}

/* --- Page hero header --- */
.page-hero {{
    background: linear-gradient(135deg, {BLUE_DARK} 0%, {BLUE_MED} 100%);
    color: white;
    padding: 1.6rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 20px rgba(30,58,95,0.25);
}}
.page-hero h1 {{
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.01em;
}}
.page-hero p {{
    font-size: 0.92rem;
    margin: 0;
    opacity: 0.82;
}}

/* --- KPI card --- */
.kpi-card {{
    background: {CARD_BG};
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.04);
    border-top: 4px solid {BLUE_MED};
    height: 100%;
}}
.kpi-card.green  {{ border-top-color: {GREEN}; }}
.kpi-card.amber  {{ border-top-color: {AMBER}; }}
.kpi-card.red    {{ border-top-color: {RED};   }}
.kpi-label {{
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {TEXT_MUTED};
    margin-bottom: 0.35rem;
}}
.kpi-value {{
    font-size: 2rem;
    font-weight: 700;
    color: {TEXT_DARK};
    line-height: 1.1;
}}
.kpi-sub {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    margin-top: 0.25rem;
}}

/* --- Section card --- */
.section-card {{
    background: {CARD_BG};
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    margin-bottom: 1.2rem;
}}
.section-title {{
    font-size: 1rem;
    font-weight: 600;
    color: {TEXT_DARK};
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid {BLUE_LIGHT};
}}

/* --- Insight cards --- */
.insight-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
    margin-bottom: 0.5rem;
}}
.insight-card {{
    background: {CARD_BG};
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    border-left: 5px solid {BLUE_MED};
}}
.insight-card.green {{ border-left-color: {GREEN}; }}
.insight-card.amber {{ border-left-color: {AMBER}; }}
.insight-number {{
    font-size: 2.2rem;
    font-weight: 700;
    color: {BLUE_MED};
    line-height: 1;
    margin-bottom: 0.3rem;
}}
.insight-number.green {{ color: {GREEN}; }}
.insight-number.amber {{ color: {AMBER}; }}
.insight-title {{
    font-size: 0.85rem;
    font-weight: 600;
    color: {TEXT_DARK};
    margin-bottom: 0.25rem;
}}
.insight-desc {{
    font-size: 0.78rem;
    color: {TEXT_MUTED};
    line-height: 1.5;
}}

/* --- Filter bar --- */
.filter-bar {{
    background: {CARD_BG};
    border-radius: 12px;
    padding: 1rem 1.5rem 0.5rem 1.5rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    margin-bottom: 1.2rem;
}}

/* --- Result card (simulation) --- */
.result-hero {{
    background: linear-gradient(135deg, {GREEN} 0%, #059669 100%);
    color: white;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(16,185,129,0.3);
    margin-bottom: 1rem;
}}
.result-hero .big-number {{
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
}}
.result-hero .result-label {{
    font-size: 0.9rem;
    opacity: 0.88;
    margin-top: 0.3rem;
}}

/* --- Model badge --- */
.badge {{
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.03em;
}}
.badge-green  {{ background: #d1fae5; color: #065f46; }}
.badge-red    {{ background: #fee2e2; color: #991b1b; }}
.badge-blue   {{ background: {BLUE_LIGHT}; color: #1e40af; }}

/* --- Sidebar brand --- */
.sidebar-brand {{
    text-align: center;
    padding: 1rem 0 0.5rem 0;
}}
.sidebar-brand .brand-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: white !important;
    letter-spacing: -0.01em;
}}
.sidebar-brand .brand-sub {{
    font-size: 0.72rem;
    color: #94a3b8 !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}
.sidebar-stat {{
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    margin: 0.3rem 0;
    font-size: 0.8rem;
}}
.sidebar-stat .s-label {{ color: #94a3b8 !important; font-size: 0.7rem; }}
.sidebar-stat .s-value {{ color: white !important; font-weight: 600; }}

/* --- Streamlit overrides --- */
[data-testid="metric-container"] {{
    background: {CARD_BG};
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
}}
div[data-testid="stMetricValue"] > div {{
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    color: {TEXT_DARK} !important;
}}
div[data-testid="stMetricLabel"] > div {{
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {TEXT_MUTED} !important;
}}
div[data-testid="stMetricDelta"] svg {{
    width: 14px;
}}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS GRAPHIQUES
# =============================================================================

def apply_chart_style(fig, height=420, margin=None):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        font=CHART_FONT,
        height=height,
        margin=margin or dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=11))
    return fig


def page_hero(title: str, subtitle: str = ""):
    sub = f'<p>{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="page-hero"><h1>{title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", color: str = "blue"):
    color_cls = {"blue": "", "green": "green", "amber": "amber", "red": "red"}.get(color, "")
    st.markdown(
        f"""<div class="kpi-card {color_cls}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


# =============================================================================
# CHARGEMENT (cached)
# =============================================================================

MODELS_PATH    = PROJECT_ROOT / "models"
DATA_RAW       = PROJECT_ROOT / "data" / "raw" / "marketing_and_sales.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_PATH   = PROJECT_ROOT / "reports"


@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_RAW).dropna(subset=[TARGET])


@st.cache_data
def load_test_data():
    X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
    y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv").squeeze()
    return X_test, y_test


@st.cache_resource
def load_final_model():
    return joblib.load(MODELS_PATH / "final_model.joblib")


@st.cache_resource
def load_all_models():
    return {
        "Linear Regression":   joblib.load(MODELS_PATH / "linear_regression.joblib"),
        "Random Forest":       joblib.load(MODELS_PATH / "random_forest.joblib"),
        "XGBoost":             joblib.load(MODELS_PATH / "xgboost.joblib"),
        "MLP (Deep Learning)": joblib.load(MODELS_PATH / "mlp_deep_learning.joblib"),
    }


@st.cache_data
def load_evaluation_results():
    return pd.read_csv(REPORTS_PATH / "evaluation_results.csv")


@st.cache_resource
def get_shap_explainer(_pipeline):
    import shap
    return shap.TreeExplainer(_pipeline.named_steps["model"])


df            = load_raw_data()
X_test, y_test = load_test_data()
pipeline      = load_final_model()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-title">ROI Marketing</div>
        <div class="brand-sub">Dashboard CMO &nbsp;·&nbsp; EFREI M1</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Accueil & KPI",
            "Exploration des donnees",
            "Simulation budgetaire",
            "Comparaison des modeles",
            "Explication SHAP",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;color:#94a3b8;margin-bottom:0.5rem;">Modele en production</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-stat">
        <div class="s-label">Algorithme</div>
        <div class="s-value">XGBoost Regressor</div>
    </div>
    <div class="sidebar-stat">
        <div class="s-label">R² (test)</div>
        <div class="s-value">0.9985</div>
    </div>
    <div class="sidebar-stat">
        <div class="s-label">MAPE (test)</div>
        <div class="s-value">1.84 %</div>
    </div>
    <div class="sidebar-stat">
        <div class="s-label">Taille dataset</div>
        <div class="s-value">{len(df):,} campagnes</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 1 — ACCUEIL & KPI
# =============================================================================

if page == "Accueil & KPI":
    page_hero(
        "Tableau de bord — Performance Marketing",
        "Vue executive des indicateurs cles · Donnees issues du dataset Kaggle Advertising & Sales",
    )

    # KPI row
    total_budget = (df["TV"] + df["Radio"] + df["Social Media"]).mean()
    avg_sales    = df["Sales"].mean()
    avg_roi      = avg_sales / total_budget if total_budget > 0 else 0
    best_infl    = df.groupby("Influencer")["Sales"].mean().idxmax()
    tv_corr      = df[["TV", "Sales"]].corr().iloc[0, 1]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Campagnes analysees", f"{len(df):,}", "dataset complet", "blue")
    with c2:
        kpi_card("Budget moyen / campagne", f"{total_budget:.0f} M$", "TV + Radio + Social Media", "amber")
    with c3:
        kpi_card("Ventes moyennes", f"{avg_sales:.1f} M$", "variable cible Sales", "green")
    with c4:
        kpi_card("ROI moyen", f"x {avg_roi:.2f}", "Sales / Budget total", "green")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-card"><div class="section-title">Distribution des ventes (variable cible)</div>', unsafe_allow_html=True)
        fig = px.histogram(
            df, x="Sales", nbins=50,
            color_discrete_sequence=[BLUE_MED],
            marginal="box",
            opacity=0.85,
        )
        fig.add_vline(x=avg_sales, line_dash="dash", line_color=RED,
                      annotation_text=f"Moy. {avg_sales:.1f}", annotation_font_color=RED)
        fig.add_vline(x=df["Sales"].median(), line_dash="dot", line_color=GREEN,
                      annotation_text=f"Med. {df['Sales'].median():.1f}", annotation_font_color=GREEN)
        apply_chart_style(fig, height=360, margin=dict(l=10, r=10, t=10, b=10))
        fig.update_layout(xaxis_title="Sales (M$)", yaxis_title="Campagnes", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-card"><div class="section-title">Repartition par type d\'Influencer</div>', unsafe_allow_html=True)
        infl_counts = df["Influencer"].value_counts().reset_index()
        infl_counts.columns = ["Influencer", "count"]
        fig = px.pie(
            infl_counts, values="count", names="Influencer",
            color_discrete_sequence=["#2563eb", "#10b981", "#f59e0b", "#e74c3c"],
            hole=0.52,
        )
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          textfont=dict(size=12))
        apply_chart_style(fig, height=360, margin=dict(l=10, r=10, t=10, b=10))
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Insights
    st.markdown('<div class="section-title" style="margin-top:0.5rem;">Insights cles pour le pilotage</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="insight-grid">
        <div class="insight-card">
            <div class="insight-number">{tv_corr:.3f}</div>
            <div class="insight-title">Correlation TV — Sales</div>
            <div class="insight-desc">Quasi-lineaire. Chaque euro investi en TV se traduit mecaniquement par une hausse des ventes. Levier numero 1.</div>
        </div>
        <div class="insight-card green">
            <div class="insight-number green">1.84 %</div>
            <div class="insight-title">Erreur moyenne du modele (MAPE)</div>
            <div class="insight-desc">XGBoost predit les ventes avec une precision de 1.84%. Fiable pour planifier les budgets.</div>
        </div>
        <div class="insight-card amber">
            <div class="insight-number amber">0.13 %</div>
            <div class="insight-title">Impact de l'Influencer (SHAP)</div>
            <div class="insight-desc">Le type d'influenceur (Nano / Mega) a un impact negligeable sur les ventes. Inutile de surpayer.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sales par influencer (bonus)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-card"><div class="section-title">Sales moyenne par type d\'Influencer</div>', unsafe_allow_html=True)
    infl_agg = df.groupby("Influencer")["Sales"].agg(["mean", "std"]).reset_index()
    infl_agg.columns = ["Influencer", "mean_sales", "std_sales"]
    infl_agg = infl_agg.sort_values("mean_sales", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=infl_agg["mean_sales"], y=infl_agg["Influencer"],
        orientation="h",
        marker_color=[BLUE_MED, GREEN, AMBER, RED],
        error_x=dict(type="data", array=infl_agg["std_sales"], color="#94a3b8"),
        text=[f"{v:.1f} M$" for v in infl_agg["mean_sales"]],
        textposition="outside",
    ))
    apply_chart_style(fig, height=260, margin=dict(l=10, r=60, t=10, b=10))
    fig.update_layout(xaxis_title="Sales moyenne (M$)", yaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 2 — EXPLORATION DES DONNEES
# =============================================================================

elif page == "Exploration des donnees":
    page_hero(
        "Exploration interactive du dataset",
        "Filtrez, segmentez et visualisez les relations entre budgets et ventes",
    )

    # Filtres
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        infl_filter = st.multiselect(
            "Type d'Influencer",
            options=INFLUENCER_CATEGORIES,
            default=INFLUENCER_CATEGORIES,
        )
    with fc2:
        tv_range = st.slider(
            "Budget TV (M$)",
            float(df["TV"].min()), float(df["TV"].max()),
            (float(df["TV"].min()), float(df["TV"].max())),
        )
    with fc3:
        sales_range = st.slider(
            "Ventes (M$)",
            float(df["Sales"].min()), float(df["Sales"].max()),
            (float(df["Sales"].min()), float(df["Sales"].max())),
        )
    st.markdown('</div>', unsafe_allow_html=True)

    mask = (
        df["Influencer"].isin(infl_filter)
        & df["TV"].between(*tv_range)
        & df["Sales"].between(*sales_range)
    )
    df_f = df[mask]

    pct_shown = len(df_f) / len(df) * 100
    col_info, _ = st.columns([2, 3])
    with col_info:
        st.info(f"**{len(df_f):,} campagnes** selectionnees · {pct_shown:.1f}% du dataset")

    # Scatter + Boxplot side by side
    col_sc, col_box = st.columns(2)

    with col_sc:
        st.markdown('<div class="section-card"><div class="section-title">Relation Budget — Ventes</div>', unsafe_allow_html=True)
        feature_x = st.selectbox("Variable X", NUMERIC_FEATURES, index=0, key="eda_x")
        fig = px.scatter(
            df_f, x=feature_x, y="Sales",
            color="Influencer",
            opacity=0.55,
            color_discrete_sequence=[BLUE_MED, GREEN, AMBER, RED],
            hover_data=NUMERIC_FEATURES,
        )
        # Droite de regression globale (numpy, pas statsmodels)
        _x = df_f[feature_x].dropna().values
        _y = df_f.loc[df_f[feature_x].notna(), "Sales"].values
        if len(_x) > 1:
            _m, _b = np.polyfit(_x, _y, 1)
            _xr = np.array([_x.min(), _x.max()])
            fig.add_trace(go.Scatter(
                x=_xr, y=_m * _xr + _b,
                mode="lines", name="Tendance",
                line=dict(color=RED, width=2, dash="dash"),
                showlegend=True,
            ))
        apply_chart_style(fig, height=400)
        fig.update_layout(xaxis_title=f"{feature_x} (M$)", yaxis_title="Sales (M$)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_box:
        st.markdown('<div class="section-card"><div class="section-title">Distribution Sales par Influencer</div>', unsafe_allow_html=True)
        fig = px.box(
            df_f, x="Influencer", y="Sales",
            color="Influencer",
            category_orders={"Influencer": INFLUENCER_CATEGORIES},
            color_discrete_sequence=[BLUE_MED, GREEN, AMBER, RED],
            points="outliers",
        )
        apply_chart_style(fig, height=400)
        fig.update_layout(xaxis_title="", yaxis_title="Sales (M$)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Correlation heatmap
    st.markdown('<div class="section-card"><div class="section-title">Matrice de correlation (Pearson)</div>', unsafe_allow_html=True)
    corr = df_f[NUMERIC_FEATURES + ["Sales"]].corr()
    fig = px.imshow(
        corr, text_auto=".3f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
    )
    fig.update_traces(textfont=dict(size=14, color="white"))
    apply_chart_style(fig, height=350, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Donnees brutes filtrees (100 premieres lignes)"):
        st.dataframe(
            df_f.head(100).style.background_gradient(subset=["Sales"], cmap="Blues"),
            use_container_width=True,
        )


# =============================================================================
# PAGE 3 — SIMULATION BUDGETAIRE
# =============================================================================

elif page == "Simulation budgetaire":
    page_hero(
        "Simulateur budgetaire",
        "Prediction des ventes en temps reel · Optimisez votre mix media",
    )

    col_in, col_out = st.columns([1, 1.6], gap="large")

    with col_in:
        st.markdown('<div class="section-card"><div class="section-title">Parametres de la campagne</div>', unsafe_allow_html=True)

        tv_budget = st.slider(
            "Budget TV (M$)",
            min_value=float(df["TV"].min()), max_value=float(df["TV"].max()),
            value=float(df["TV"].median()), step=1.0,
        )
        radio_budget = st.slider(
            "Budget Radio (M$)",
            min_value=float(df["Radio"].min()), max_value=float(df["Radio"].max()),
            value=float(df["Radio"].median()), step=0.5,
        )
        sm_budget = st.slider(
            "Budget Social Media (M$)",
            min_value=float(df["Social Media"].min()), max_value=float(df["Social Media"].max()),
            value=float(df["Social Media"].median()), step=0.1,
        )
        influencer_type = st.selectbox(
            "Type d'Influencer", options=INFLUENCER_CATEGORIES, index=2,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Budget breakdown donut
        total_budget = tv_budget + radio_budget + sm_budget
        st.markdown('<div class="section-card"><div class="section-title">Repartition du budget</div>', unsafe_allow_html=True)
        bdf = pd.DataFrame({
            "Canal":  ["TV", "Radio", "Social Media"],
            "Budget": [tv_budget, radio_budget, sm_budget],
        })
        bdf["Part_%"] = bdf["Budget"] / total_budget * 100
        fig_donut = px.pie(
            bdf, values="Budget", names="Canal",
            color_discrete_sequence=[RED, BLUE_MED, GREEN],
            hole=0.55,
        )
        fig_donut.update_traces(
            textinfo="percent+label",
            textposition="outside",
            textfont=dict(size=12),
        )
        fig_donut.add_annotation(
            text=f"<b>{total_budget:.0f}</b><br><span style='font-size:10px'>M$ total</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=TEXT_DARK),
        )
        apply_chart_style(fig_donut, height=280, margin=dict(l=10, r=10, t=10, b=10))
        fig_donut.update_layout(showlegend=False)
        st.plotly_chart(fig_donut, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_out:
        scenario = pd.DataFrame([{
            "TV": tv_budget, "Radio": radio_budget,
            "Social Media": sm_budget, "Influencer": influencer_type,
        }])
        prediction      = float(pipeline.predict(scenario)[0])
        roi             = prediction / total_budget if total_budget > 0 else 0
        avg_sales_ds    = df["Sales"].mean()
        ecart_pct       = (prediction - avg_sales_ds) / avg_sales_ds * 100
        ecart_sign      = "+" if ecart_pct >= 0 else ""
        ecart_color     = GREEN if ecart_pct >= 0 else RED

        # Result hero card
        st.markdown(f"""
        <div class="result-hero">
            <div class="result-label">Ventes predites (XGBoost)</div>
            <div class="big-number">{prediction:.1f} M$</div>
            <div class="result-label" style="margin-top:0.5rem;">
                ROI estime : <b>x {roi:.2f}</b> &nbsp;·&nbsp;
                vs marche : <b style="color:{'#d1fae5' if ecart_pct>=0 else '#fee2e2'}">{ecart_sign}{ecart_pct:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge prediction
        st.markdown('<div class="section-card"><div class="section-title">Positionnement vs distribution historique</div>', unsafe_allow_html=True)
        q25 = df["Sales"].quantile(0.25)
        q75 = df["Sales"].quantile(0.75)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            delta={"reference": avg_sales_ds, "valueformat": ".1f",
                   "increasing": {"color": GREEN}, "decreasing": {"color": RED}},
            number={"suffix": " M$", "font": {"size": 28, "color": TEXT_DARK}},
            gauge={
                "axis": {"range": [df["Sales"].min(), df["Sales"].max()],
                         "tickcolor": TEXT_MUTED, "tickfont": {"size": 10}},
                "bar":  {"color": GREEN if ecart_pct >= 0 else AMBER, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [df["Sales"].min(), q25], "color": "#fee2e2"},
                    {"range": [q25, q75],               "color": "#fef3c7"},
                    {"range": [q75, df["Sales"].max()],  "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": BLUE_MED, "width": 3},
                    "thickness": 0.8,
                    "value": avg_sales_ds,
                },
            },
        ))
        fig_gauge.update_layout(
            height=260,
            margin=dict(l=30, r=30, t=20, b=10),
            paper_bgcolor="white",
            font=CHART_FONT,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.caption(f"Bande rouge = Q0-Q25 · jaune = Q25-Q75 · verte = Q75-Q100 · trait bleu = moyenne ({avg_sales_ds:.1f} M$)")
        st.markdown('</div>', unsafe_allow_html=True)

        # Histogramme positionnel
        st.markdown('<div class="section-card"><div class="section-title">Position dans la distribution (test set)</div>', unsafe_allow_html=True)
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df["Sales"], nbinsx=50,
            marker_color=BLUE_LIGHT, marker_line_color=BLUE_MED,
            marker_line_width=0.4, name="Historique",
        ))
        fig_hist.add_vline(
            x=prediction, line_width=2.5, line_color=GREEN,
            annotation_text=f"Prediction {prediction:.1f} M$",
            annotation_font_color=GREEN, annotation_position="top right",
        )
        fig_hist.add_vline(
            x=avg_sales_ds, line_width=1.5, line_dash="dot", line_color=TEXT_MUTED,
            annotation_text=f"Moy. {avg_sales_ds:.1f}", annotation_font_color=TEXT_MUTED,
        )
        apply_chart_style(fig_hist, height=240, margin=dict(l=10, r=10, t=30, b=10))
        fig_hist.update_layout(xaxis_title="Sales (M$)", yaxis_title="", showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 4 — COMPARAISON DES MODELES
# =============================================================================

elif page == "Comparaison des modeles":
    page_hero(
        "Comparaison des modeles",
        "Evaluation sur le test set (donnees jamais vues) — anti data leakage garanti",
    )

    eval_df = load_evaluation_results()

    # Tableau stylé
    st.markdown('<div class="section-card"><div class="section-title">Tableau comparatif des performances (test set)</div>', unsafe_allow_html=True)

    def color_verdict(val):
        if val == "OK":
            return "background-color:#d1fae5;color:#065f46;font-weight:600"
        return "background-color:#fee2e2;color:#991b1b;font-weight:600"

    styled = (
        eval_df.style
        .highlight_min(subset=["RMSE_test", "MAE_test", "MAPE_test_%"], color="#bbf7d0")
        .highlight_max(subset=["R2_test"],                               color="#bbf7d0")
        .applymap(color_verdict, subset=["Verdict"])
        .format({
            "MAE_train": "{:.3f}", "MAE_test": "{:.3f}",
            "RMSE_train": "{:.3f}", "RMSE_test": "{:.3f}",
            "R2_train": "{:.4f}", "R2_test": "{:.4f}",
            "MAPE_test_%": "{:.2f}%",
            "Gap_R2": "{:+.4f}",
        })
    )
    st.dataframe(styled, use_container_width=True)
    st.success("**Modele retenu : XGBoost** — meilleur RMSE, R² et MAPE sur le test set, zero overfitting.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Charts row
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-card"><div class="section-title">RMSE test (bas = meilleur)</div>', unsafe_allow_html=True)
        df_sorted = eval_df.sort_values("RMSE_test", ascending=True)
        colors = [MODEL_COLORS.get(m, BLUE_MED) for m in df_sorted["Model"]]
        fig = go.Figure(go.Bar(
            x=df_sorted["RMSE_test"], y=df_sorted["Model"],
            orientation="h", marker_color=colors,
            text=[f"{v:.3f}" for v in df_sorted["RMSE_test"]],
            textposition="outside",
        ))
        apply_chart_style(fig, height=300, margin=dict(l=10, r=50, t=10, b=10))
        fig.update_layout(xaxis_title="RMSE", yaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-card"><div class="section-title">R² test (haut = meilleur)</div>', unsafe_allow_html=True)
        df_sorted = eval_df.sort_values("R2_test", ascending=False)
        colors = [MODEL_COLORS.get(m, BLUE_MED) for m in df_sorted["Model"]]
        fig = go.Figure(go.Bar(
            x=df_sorted["R2_test"], y=df_sorted["Model"],
            orientation="h", marker_color=colors,
            text=[f"{v:.4f}" for v in df_sorted["R2_test"]],
            textposition="outside",
        ))
        apply_chart_style(fig, height=300, margin=dict(l=10, r=60, t=10, b=10))
        fig.update_layout(xaxis_title="R²", yaxis_title="", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="section-card"><div class="section-title">Radar — Vue multi-criteres</div>', unsafe_allow_html=True)
        max_r2   = eval_df["R2_test"].max()
        min_rmse = eval_df["RMSE_test"].min()
        min_mape = eval_df["MAPE_test_%"].min()
        categories = ["R²", "RMSE (inv.)", "MAPE (inv.)", "Stabilite"]
        fig = go.Figure()
        for _, row in eval_df.iterrows():
            stab = 1.0 if row["Verdict"] == "OK" else 0.6
            fig.add_trace(go.Scatterpolar(
                r=[
                    row["R2_test"] / max_r2,
                    min_rmse / row["RMSE_test"],
                    min_mape / row["MAPE_test_%"],
                    stab,
                ],
                theta=categories,
                fill="toself",
                name=row["Model"],
                line=dict(color=MODEL_COLORS.get(row["Model"], BLUE_MED)),
                opacity=0.55,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template=PLOTLY_TEMPLATE, height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            font=CHART_FONT,
            legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Predictions vs Realite superposees
    st.markdown('<div class="section-card"><div class="section-title">Predictions vs Ventes reelles — 4 modeles (test set)</div>', unsafe_allow_html=True)
    all_models = load_all_models()
    fig = go.Figure()
    for name, model in all_models.items():
        y_pred = model.predict(X_test)
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers", name=name,
            marker=dict(size=5, opacity=0.45, color=MODEL_COLORS.get(name, BLUE_MED)),
        ))
    lims = [float(y_test.min()), float(y_test.max())]
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines", name="Prediction parfaite",
        line=dict(color=RED, dash="dash", width=2),
    ))
    apply_chart_style(fig, height=460)
    fig.update_layout(xaxis_title="Sales reelles (M$)", yaxis_title="Sales predites (M$)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PAGE 5 — EXPLICATION SHAP
# =============================================================================

elif page == "Explication SHAP":
    page_hero(
        "Explicabilite des predictions (SHAP)",
        "Comprendre pourquoi le modele predit une valeur donnee · Vue globale et locale",
    )

    # Importance globale
    interp_path = REPORTS_PATH / "interpretability_results.csv"
    if interp_path.exists():
        interp_df = pd.read_csv(interp_path)

        st.markdown('<div class="section-card"><div class="section-title">Importance globale des variables — 3 methodes comparees</div>', unsafe_allow_html=True)

        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            methods = [
                ("Native_Importance_norm",     "Feature Importance (XGBoost)", BLUE_MED),
                ("Permutation_R2_loss_norm",   "Permutation Importance",       AMBER),
                ("SHAP_mean_abs_norm",         "SHAP (valeur absolue moy.)",   GREEN),
            ]
            x_vals = np.arange(len(interp_df))
            fig = go.Figure()
            width = 0.25
            for i, (col, label, color) in enumerate(methods):
                fig.add_trace(go.Bar(
                    x=interp_df["feature"],
                    y=interp_df[col],
                    name=label,
                    marker_color=color,
                    text=[f"{v:.1f}%" for v in interp_df[col]],
                    textposition="outside",
                    textfont=dict(size=10),
                ))
            apply_chart_style(fig, height=350, margin=dict(l=10, r=10, t=30, b=10))
            fig.update_layout(
                barmode="group",
                yaxis_title="Importance relative (%)",
                xaxis_title="",
                legend=dict(orientation="h", y=1.12, font=dict(size=11)),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("**Importance relative (%) par methode :**")
            display = interp_df[["feature", "Native_Importance_norm",
                                  "Permutation_R2_loss_norm", "SHAP_mean_abs_norm"]].copy()
            display.columns = ["Variable", "Native", "Permutation", "SHAP"]
            st.dataframe(
                display.style.background_gradient(
                    subset=["Native", "Permutation", "SHAP"], cmap="Blues"
                ).format("{:.2f}%", subset=["Native", "Permutation", "SHAP"]),
                use_container_width=True,
                height=300,
            )
            st.markdown(
                f'<div style="background:{BLUE_LIGHT};border-radius:8px;padding:0.7rem 1rem;'
                f'font-size:0.8rem;color:{BLUE_DARK};margin-top:0.5rem;">'
                "Les 3 methodes convergent : <b>TV domine</b> avec plus de 90% de l'importance. "
                "L'Influencer est negligeable (&lt;1%)."
                '</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Explication locale
    st.markdown('<div class="section-card"><div class="section-title">Vue locale — Expliquer une prediction specifique</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.85rem;color:#64748b;margin-bottom:1rem;">'
        "Ajustez les budgets ci-dessous : le graphe waterfall decompose la prediction "
        "variable par variable.</p>",
        unsafe_allow_html=True,
    )

    col_ctrl, col_shap = st.columns([1, 2], gap="large")

    with col_ctrl:
        tv_s  = st.slider("TV (M$)", float(df["TV"].min()), float(df["TV"].max()),
                          float(df["TV"].median()), step=1.0, key="shap_tv")
        rad_s = st.slider("Radio (M$)", float(df["Radio"].min()), float(df["Radio"].max()),
                          float(df["Radio"].median()), step=0.5, key="shap_rad")
        sm_s  = st.slider("Social Media (M$)", float(df["Social Media"].min()),
                          float(df["Social Media"].max()),
                          float(df["Social Media"].median()), step=0.1, key="shap_sm")
        inf_s = st.selectbox("Influencer", INFLUENCER_CATEGORIES, index=2, key="shap_inf")

    with col_shap:
        import shap

        scenario = pd.DataFrame([{
            "TV": tv_s, "Radio": rad_s, "Social Media": sm_s, "Influencer": inf_s,
        }])
        pred_shap = float(pipeline.predict(scenario)[0])

        preprocessor   = pipeline.named_steps["preprocessor"]
        feature_names  = preprocessor.get_feature_names_out().tolist()
        explainer      = get_shap_explainer(pipeline)
        scen_tr        = preprocessor.transform(scenario)
        shap_vals      = explainer.shap_values(scen_tr)
        expected_val   = float(explainer.expected_value)
        if not np.isscalar(expected_val):
            expected_val = float(expected_val[0])

        contrib = pd.DataFrame({
            "feature":    feature_names,
            "shap_value": shap_vals[0],
            "abs_value":  np.abs(shap_vals[0]),
        }).sort_values("abs_value", ascending=True)
        contrib = contrib[contrib["abs_value"] > 0.001]

        bar_colors = [GREEN if v > 0 else RED for v in contrib["shap_value"]]

        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            x=contrib["shap_value"],
            y=contrib["feature"],
            orientation="h",
            marker_color=bar_colors,
            marker_line_color="white",
            marker_line_width=0.5,
            text=[f"{v:+.2f} M$" for v in contrib["shap_value"]],
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig_shap.add_vline(x=0, line_dash="solid", line_color=TEXT_MUTED, line_width=1.5)
        apply_chart_style(fig_shap, height=360, margin=dict(l=10, r=80, t=60, b=10))
        fig_shap.update_layout(
            title=dict(
                text=(
                    f"<b>Prediction : {pred_shap:.2f} M$</b>  "
                    f"<span style='font-size:12px;color:{TEXT_MUTED}'>"
                    f"Base = {expected_val:.2f} M$  →  Delta = {pred_shap - expected_val:+.2f} M$"
                    f"</span>"
                ),
                font=dict(size=14),
                x=0,
            ),
            xaxis_title="Contribution a la prediction (M$)",
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown(
            f'<span class="badge badge-green">Vert = impact positif sur les ventes</span>'
            f'&nbsp;&nbsp;'
            f'<span class="badge badge-red">Rouge = impact negatif</span>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)
