"""
DataLab Pro - Dashboard Intelligent
Version 3.1 | Production-Ready | Minimalist | Moderne
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# --- Imports internes ---
from src.data.data_analysis import (
    auto_detect_column_types,
    detect_useless_columns,
    compute_global_metrics
)
from src.data.image_processing import (
    analyze_image_quality,
    analyze_image_distribution
)
from utils.system_utils import cleanup_memory
from monitoring.state_managers import init, AppPage, STATE

# --- Configuration ---
st.set_page_config(
    page_title="DataLab Pro | Dashboard",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure session is refreshed (metrics/timeouts) on page load
init()

# --- CSS Moderne & Léger ---
st.markdown("""
<style>
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --success: #4facfe;
        --warning: #43e97b;
        --danger: #fa709a;
        --radius: 16px;
        --shadow: 0 4px 20px rgba(0,0,0,0.08);
        --transition: all 0.3s ease;
    }
    .header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .card {
        background: white;
        padding: 2rem;
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        margin: 1rem 0;
        transition: var(--transition);
    }
    .card:hover { transform: translateY(-4px); box-shadow: 0 8px 30px rgba(0,0,0,0.12); }
    .metric-card {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        padding: 1.5rem;
        border-radius: var(--radius);
        text-align: center;
        font-weight: 700;
    }
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-success { background: #d4edda; color: #155724; }
    .badge-info { background: #d1ecf1; color: #0c5460; }
    .badge-danger { background: #f8d7da; color: #721c24; }
    .stTabs [data-baseweb="tab"] { border-radius: 12px 12px 0 0; padding: 1rem 2rem; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, var(--primary), var(--secondary)); color: white; }
    #MainMenu, footer, .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ========================================
# FONCTIONS UTILITAIRES
# ========================================
def show_status():
    icon = "photo" if STATE.images else "table"
    text = "IMAGES" if STATE.images else "TABULAIRE"
    cls = "badge-success" if STATE.images else "badge-info"
    st.markdown(f'<div class="status-badge {cls}"> {icon} {text}</div>', unsafe_allow_html=True)

def show_system():
    try:
        mem = STATE.metrics.memory_percent
        color = "#28a745" if mem < 70 else "#ffc107" if mem < 85 else "#dc3545"
        st.markdown(f'<div style="text-align:center;"><div style="font-size:0.9rem;color:#666;margin-bottom:0.5rem;">SYSTÈME</div><div style="width:60px;height:60px;border-radius:50%;background:conic-gradient({color} 0% {mem}%,#e9ecef {mem}% 100%);margin:0 auto;display:flex;align-items:center;justify-content:center;"><div style="width:45px;height:45px;border-radius:50%;background:white;display:flex;align-items:center;justify-content:center;font-weight:bold;color:{color};">{mem:.0f}%</div></div></div>', unsafe_allow_html=True)
    except:
        st.markdown("<div style='text-align:center;color:#6c757d;font-size:0.8rem;'>N/A</div>", unsafe_allow_html=True)

# ========================================
# ACCÈS & HEADER
# ========================================
if not STATE.loaded:
    st.error("Aucun dataset chargé")
    if st.button("Retour Accueil", type="primary"):
        STATE.switch(AppPage.HOME)
    st.stop()

col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
with col1:
    st.markdown('<div class="header">Dashboard Exploratoire</div>', unsafe_allow_html=True)
    st.markdown(f"**{STATE.dtype.value.upper()}** • `{STATE.data.name or 'Dataset sans nom'}`")
with col2: show_status()
with col3: show_system()
with col4:
    if STATE.tabular:
        if st.button("ML", use_container_width=True): STATE.switch(AppPage.ML_TRAINING)
    else:
        if st.button("Vision", use_container_width=True): STATE.switch(AppPage.CV_TRAINING)
    if st.button("Accueil", use_container_width=True): STATE.switch(AppPage.HOME)

st.markdown("---")

# ========================================
# VUE D'ENSEMBLE
# ========================================
st.markdown("## Vue d'Ensemble")

if STATE.images:
    d = STATE.data
    cols = st.columns(4)
    cols[0].markdown(f"<div class='metric-card'>Images<br><b>{d.img_count:,}</b></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='metric-card'>Taille<br><b>{d.img_shape[1]}×{d.img_shape[2]}</b></div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='metric-card'>Classes<br><b>{d.n_classes}</b></div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div class='metric-card'>Mémoire<br><b>{d.X.nbytes/(1024**2):.1f} MB</b></div>", unsafe_allow_html=True)

    st.markdown("### Distribution des Classes")
    counts = Counter(d.y)
    if len(counts) == 2 and set(counts.keys()) == {0, 1}:
        fig = go.Figure(data=[go.Pie(labels=['Normal', 'Anomalie'], values=list(counts.values()), hole=0.5)])
    else:
        fig = px.pie(names=[f"Classe {k}" for k in counts], values=list(counts.values()))
    st.plotly_chart(fig, use_container_width=True)

else:  # Tabular
    df = STATE.data.df
    metrics = compute_global_metrics(df)
    cols = st.columns(4)
    cols[0].markdown(f"<div class='metric-card'>Lignes<br><b>{metrics['n_rows']:,}</b></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='metric-card'>Colonnes<br><b>{metrics['n_cols']}</b></div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='metric-card'>Manquants<br><b>{metrics['missing_percentage']:.1f}%</b></div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div class='metric-card'>Doublons<br><b>{metrics['duplicate_rows']:,}</b></div>", unsafe_allow_html=True)

    st.markdown("### Types de Colonnes")
    types = auto_detect_column_types(df)
    cols = st.columns(4)
    for i, (k, label, icon, color) in enumerate([
        ('numeric', 'Numériques', 'numbers', '#667eea'),
        ('categorical', 'Catégorielles', 'text', '#f093fb'),
        ('text_or_high_cardinality', 'Texte', 'book', '#4facfe'),
        ('datetime', 'Dates', 'calendar', '#43e97b')
    ]):
        with cols[i]:
            count = len(types.get(k, []))
            st.markdown(f"<div style='text-align:center;padding:1.5rem;background:{color};color:white;border-radius:{'var(--radius)'};'><div style='font-size:2rem;'>{icon}</div><div style='font-size:0.9rem;margin:0.5rem 0;'>{label}</div><div style='font-size:1.8rem;font-weight:800;'>{count}</div></div>", unsafe_allow_html=True)

st.markdown("---")

# ========================================
# ANALYSE PAR TYPE
# ========================================
if STATE.images:
    tab1, tab2, tab3 = st.tabs(["Échantillons", "Qualité", "Distributions"])

    with tab1:
        st.markdown("### Échantillons")
        n = st.slider("Images", 1, 12, 6)
        mode = st.radio("Mode", ["Aléatoire", "Par classe"], horizontal=True)
        X, y = STATE.data.X, STATE.data.y
        if mode == "Aléatoire":
            idxs = np.random.choice(len(X), n, replace=False)
        else:
            idxs = []
            for c in np.unique(y):
                cls_idx = np.where(y == c)[0]
                idxs.extend(np.random.choice(cls_idx, min(2, len(cls_idx)), replace=False))
            idxs = np.random.choice(idxs, n, replace=False)
        cols = st.columns(3)
        for i, idx in enumerate(idxs):
            with cols[i % 3]:
                label = "Normal" if y[idx] == 0 and len(np.unique(y)) == 2 else f"Classe {y[idx]}"
                st.image(X[idx], caption=f"{label} | Index {idx}")

    with tab2:
        st.markdown("### Qualité")
        if st.button("Analyser", type="primary"):
            with st.spinner("Analyse en cours..."):
                res = analyze_image_quality(STATE.data.X, sample_size=200)
                if 'error' in res:
                    st.error(res['error'])
                else:
                    cols = st.columns(2)
                    with cols[0]:
                        fig = go.Figure(go.Histogram(x=res['brightness']['values'], nbinsx=30, name="Luminosité"))
                        fig.add_vline(res['brightness']['mean'], line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
                    with cols[1]:
                        fig = go.Figure(go.Histogram(x=res['contrast']['values'], nbinsx=30, name="Contraste"))
                        fig.add_vline(res['contrast']['mean'], line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Distributions")
        if st.button("Calculer", type="primary"):
            with st.spinner("Calcul en cours..."):
                res = analyze_image_distribution(STATE.data.X)
                fig = go.Figure()
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
                for i, (ch, data) in enumerate(res['channels'].items()):
                    fig.add_trace(go.Histogram(x=data['data'], name=ch, opacity=0.7, marker_color=colors[i]))
                fig.update_layout(barmode='overlay', title="Distribution par Canal")
                st.plotly_chart(fig, use_container_width=True)

else:  # Tabular
    tab1, tab2, tab3 = st.tabs(["Qualité", "Corrélations", "Nettoyage"])

    with tab1:
        st.markdown("### Qualité")
        df = STATE.data.df
        miss = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        if miss.sum() > 0:
            fig = px.bar(x=miss.index[:15], y=miss.values[:15], title="Valeurs Manquantes (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("Aucune valeur manquante")

    with tab2:
        st.markdown("### Corrélations")
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Moins de 2 colonnes numériques")

    with tab3:
        st.markdown("### Nettoyage")
        if st.button("Détecter colonnes inutiles"):
            useless = detect_useless_columns(df)
            if useless:
                st.warning(f"{len(useless)} colonnes inutiles")
                st.write(useless)
            else:
                st.success("Aucune colonne inutile")

# ========================================
# FOOTER
# ========================================
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
with col1: st.caption(f"{STATE.get('time', 'N/A')}")
with col2: st.caption(f"Erreurs: {STATE.metrics.errors}")
with col3: st.caption(f"Données: {STATE.dtype.value.upper()}")
with col4:
    if st.button("Optimiser Mémoire"):
        cleanup_memory()
        st.success("Mémoire libérée")
        st.rerun()