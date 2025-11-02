"""
DataLab Pro - Page d'Accueil
Version 3.1 | Production-Ready | Minimalist | Moderne
"""
import sys
import os
# Ajout de la racine du projet à sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

# --- Imports internes ---
from src.data.data_loader import load_data
from src.data.image_processing import (
    detect_dataset_structure,
    load_images_flexible,
    get_dataset_info
)
from utils.system_utils import get_system_metrics, cleanup_memory
from monitoring.state_managers import init, AppPage, STATE

# --- Configuration Streamlit ---
st.set_page_config(
    page_title="DataLab Pro",
    page_icon="test-tube",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- CSS Moderne & Léger ---
st.markdown("""
<style>
    :root {
        --primary: #667eea;
        --secondary: #764ba2;
        --success: #4facfe;
        --warning: #43e97b;
        --dark: #2c3e50;
        --radius: 16px;
        --shadow: 0 4px 20px rgba(0,0,0,0.08);
        --transition: all 0.3s ease;
    }
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
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
    .upload-zone {
        border: 3px dashed var(--primary);
        border-radius: var(--radius);
        padding: 3rem;
        text-align: center;
        background: #f8f9ff;
        margin: 1.5rem 0;
    }
    .btn-primary {
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        width: 100%;
    }
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
    .badge-dark { background: #d6d8db; color: #1b1e21; }
    .stButton > button { border-radius: 50px; font-weight: 600; }
    #MainMenu, footer, .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ========================================
# FONCTIONS UTILITAIRES
# ========================================
def show_status_badge():
    # CORRECTION: Utiliser les propriétés au lieu de STATE directement
    if STATE.loaded:  
        icon = "photo" if STATE.images else "table"  
        text = "IMAGES" if STATE.images else "TABULAIRE"  
        cls = "badge-success" if STATE.images else "badge-info"
    else:
        icon, text, cls = "hourglass", "EN ATTENTE", "badge-dark"
    st.markdown(f'<div class="status-badge {cls}"> {icon} {text}</div>', unsafe_allow_html=True)

def show_system_status():
    try:
        mem = get_system_metrics()["memory_percent"]
        color = "#28a745" if mem < 70 else "#ffc107" if mem < 85 else "#dc3545"
        st.markdown(f"""
        <div style="text-align:center;">
            <div style="font-size:0.9rem;color:#666;margin-bottom:0.5rem;">SYSTÈME</div>
            <div style="width:60px;height:60px;border-radius:50%;background:conic-gradient({color} 0% {mem}%,#e9ecef {mem}% 100%);margin:0 auto;display:flex;align-items:center;justify-content:center;">
                <div style="width:45px;height:45px;border-radius:50%;background:white;display:flex;align-items:center;justify-content:center;font-weight:bold;color:{color};">{mem:.0f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except:
        st.markdown("<div style='text-align:center;color:#6c757d;font-size:0.8rem;'>Indisponible</div>", unsafe_allow_html=True)

# ========================================
# CHARGEMENT DONNÉES
# ========================================
def handle_tabular_upload(file):
    if file.size > STATE.config.MAX_FILE_MB * 1024**2:
        st.error(f"Fichier trop grand (> {STATE.config.MAX_FILE_MB} MB)")
        return
    if st.session_state.get("last_file") == file.name:
        st.info("Fichier déjà chargé")
        return

    with st.spinner("Chargement..."):
        df, report, df_raw = load_data(file, sanitize_for_display=True)
        if df is None or df.empty:
            st.error("Fichier vide ou illisible")
            return
        if STATE.set_tabular(df, df_raw, file.name):
            st.session_state.last_file = file.name
            st.success(f"{len(df):,} lignes chargées")
            if STATE.switch(AppPage.DASHBOARD):
                st.rerun()

def handle_image_upload(data_dir: str):
    if not os.path.exists(data_dir):
        st.warning("Dossier introuvable")
        return
    structure = detect_dataset_structure(data_dir)
    if structure["type"] == "invalid":
        st.error("Structure invalide")
        return

    with st.spinner("Chargement des images..."):
        X, y = load_images_flexible(data_dir, target_size=(256, 256))
        if len(X) == 0:
            st.error("Aucune image trouvée")
            return
        X_norm = X / 255.0 if X.max() > 1 else X.copy()
        info = get_dataset_info(data_dir)
        if STATE.set_images(X, X_norm, y, data_dir, structure, info):
            st.balloons()
            st.success(f"{len(X):,} images chargées")
            if STATE.switch(AppPage.DASHBOARD):
                st.rerun()

# ========================================
# RENDU PRINCIPAL
# ========================================
# === CORRECTIONS DANS render_home() ===

def render_home():
    # === HERO ===
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown('<div class="main-header">DataLab Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Analyse • ML • Vision • Production</div>', unsafe_allow_html=True)
    with col2: show_status_badge()
    with col3: show_system_status()

    st.markdown("---")

    # === SÉLECTION DONNÉES ===
    st.markdown("## Commencez Votre Analyse")
    tab1, tab2 = st.tabs(["Données Tabulaires", "Données Images"])

    with tab1:
        st.markdown("#### Glissez-déposez votre fichier")
        uploaded = st.file_uploader(
            "CSV, Excel, Parquet, JSON, Feather",
            type=list(STATE.config.SUPPORTED_EXT),
            key="tabular",
            label_visibility="collapsed"
        )
        if uploaded:
            handle_tabular_upload(uploaded)

    with tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            default_path = os.path.join(os.path.dirname(__file__), "..", "data", "mvtec_ad")
            data_dir = st.text_input("Dossier dataset", value=default_path or "")
        with col2:
            st.write("<br>", unsafe_allow_html=True)
            if st.button("Charger", use_container_width=True):
                handle_image_upload(data_dir)

        # Exemples
        examples = ["bottle", "cable", "capsule", "metal_nut"]
        selected = st.selectbox("Ou choisissez un exemple", examples, format_func=lambda x: x.title())
        example_path = os.path.join(project_root, "src", "data", "mvtec_ad", selected)
        if st.button("Charger Exemple", use_container_width=True):
            handle_image_upload(example_path)

    st.markdown("---")

    # === ÉTAT ACTUEL ===
    # CORRECTION: Utiliser STATE.loaded au lieu de STATE dans la condition
    if STATE.loaded:  # ✅ Correct - utilise la propriété
        st.markdown("## Données Chargées")
        # CORRECTION: Utiliser STATE.images au lieu de STATE dans la condition
        if STATE.images:  # ✅ Correct - utilise la propriété
            d = STATE.data  # ✅ Correct - utilise la propriété
            cols = st.columns(4)
            cols[0].markdown(f"<div class='metric-card'>Images<br><b>{d.img_count:,}</b></div>", unsafe_allow_html=True)
            cols[1].markdown(f"<div class='metric-card'>Taille<br><b>{d.img_shape[1]}×{d.img_shape[2]}</b></div>", unsafe_allow_html=True)
            cols[2].markdown(f"<div class='metric-card'>Classes<br><b>{d.n_classes}</b></div>", unsafe_allow_html=True)
            cols[3].markdown(f"<div class='metric-card'>Mémoire<br><b>{d.X.nbytes/(1024**2):.1f} MB</b></div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Explorer Dashboard", type="primary", use_container_width=True):
                    STATE.switch(AppPage.DASHBOARD)
            with col2:
                if st.button("Vision IA", use_container_width=True):
                    STATE.switch(AppPage.CV_TRAINING)
            with col3:
                if st.button("Nouveau", use_container_width=True):
                    STATE.reset_all()
                    st.rerun()

        else:  # Tabular
            # CORRECTION: Utiliser STATE.data au lieu d'accéder directement à st.session_state
            df = STATE.data.df  # ✅ Correct - utilise la propriété
            cols = st.columns(4)
            cols[0].markdown(f"<div class='metric-card'>Lignes<br><b>{len(df):,}</b></div>", unsafe_allow_html=True)
            cols[1].markdown(f"<div class='metric-card'>Colonnes<br><b>{len(df.columns)}</b></div>", unsafe_allow_html=True)
            mem = df.memory_usage(deep=True).sum() / (1024**2)
            cols[2].markdown(f"<div class='metric-card'>Mémoire<br><b>{mem:.1f} MB</b></div>", unsafe_allow_html=True)
            cols[3].markdown(f"<div class='metric-card'>Type<br><b>{'Dask' if hasattr(df, 'npartitions') else 'Pandas'}</b></div>", unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Analyser", type="primary", use_container_width=True):
                    STATE.switch(AppPage.DASHBOARD)
            with col2:
                if st.button("Machine Learning", use_container_width=True):
                    STATE.switch(AppPage.ML_TRAINING)
            with col3:
                if st.button("Nouveau", use_container_width=True):
                    STATE.reset_all()
                    st.rerun()
    else:
        st.markdown("## Fonctionnalités")
        cols = st.columns(3)
        with cols[0]:
            st.markdown('<div class="card"><h4>Tabulaires</h4><ul><li>CSV, Excel, Parquet</li><li>Analyse auto</li><li>ML intégré</li></ul></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown('<div class="card"><h4>Images</h4><ul><li>MVTec AD</li><li>Anomalies</li><li>Classification</li></ul></div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown('<div class="card"><h4>Production</h4><ul><li>Thread-safe</li><li>Timeout</li><li>Logs</li></ul></div>', unsafe_allow_html=True)

    # === FOOTER ===
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.caption(f"{time.strftime('%H:%M:%S')}")
    with col2: st.caption(f"Erreurs: {STATE.metrics.errors}")  # ✅ Correct - utilise la propriété
    with col3: st.caption(f"Données: {STATE.dtype.value.upper() if STATE.loaded else 'Aucune'}")  # ✅ Correct
    with col4:
        if st.button("Optimiser Mémoire", key="cleanup"):
            cleanup_memory()
            st.success("Mémoire libérée")
            st.rerun()

# ========================================
# EXÉCUTION
# ========================================
if __name__ == "__main__":
    try:
        render_home()
    except Exception as e:
        st.error("Erreur critique. Redémarrez.")
        st.code(f"{type(e).__name__}: {e}")