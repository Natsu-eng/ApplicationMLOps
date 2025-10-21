"""
Dashboard exploratoire pour l'analyse de données.
Version optimisée pour la production avec monitoring et gestion d'erreurs robuste.
"""
import plotly.graph_objects as go
import plotly.express as px
import os
import streamlit as st
import pandas as pd
import time
import logging
import re
import psutil
import gc
import numpy as np
from functools import wraps
from typing import Dict, List, Any
from PIL import Image
from scipy import ndimage  # Importé au top pour cohérence

from visions import Image as VisionsImage

from helpers.anomaly_helpers import preview_images
from helpers.data_validators import DataValidator
from helpers.data_samplers import DataSampler
from helpers.navigation_manager import setup_navigation, require_data
from monitoring.state_managers import DashboardStateManager
from monitoring.decorators import monitor_performance
from monitoring.system_monitor import SystemMonitor, get_system_metrics
from src.config.constants import ANOMALY_CONFIG
from src.data.data_analysis import (
    compute_if_dask,
    is_dask_dataframe,
    auto_detect_column_types,
    detect_useless_columns,
    cleanup_memory
)
from src.data.image_processing import load_images_from_folder
from src.evaluation.exploratory_plots import (
    create_simple_correlation_heatmap,
    plot_overview_metrics,
    plot_missing_values_overview,
    plot_cardinality_overview,
    plot_distribution,
    plot_bivariate_analysis,
    plot_correlation_heatmap
)

from src.shared.logging import get_logger
logger = get_logger(__name__)

# Configuration Streamlit pour la production
st.set_page_config(
    page_title="Dashboard Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto"
)

# CSS personnalisé pour un style moderne
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .tab-content {
        padding: 1rem;
        background: #f9f9f9;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .btn-primary {
        background-color: #3498db;
        color: white;
    }
    .btn-primary:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# Constantes de configuration
class Config:
    MAX_PREVIEW_ROWS = 100
    MAX_SAMPLE_SIZE = 15000
    MAX_BIVARIATE_SAMPLE = 10000
    MEMORY_CHECK_INTERVAL = 180  # 3 minutes
    CACHE_TTL = 600  # 10 minutes
    TIMEOUT_THRESHOLD = 30
    MEMORY_WARNING = 85
    MEMORY_CRITICAL = 90

def setup_production_environment():
    """Configuration optimisée pour l'environnement de production."""
    if 'production_setup_done' not in st.session_state:
        st.session_state.production_setup_done = True
        if os.getenv('STREAMLIT_ENV') == 'production':
            hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
            .stAlert > div {padding: 0.5rem;}
            .element-container {margin: 0.5rem 0;}
            </style>
            """
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)

setup_production_environment()

# === FONCTIONS DE DÉTECTION INTELLIGENTE ===

def get_loaded_data_type():
    """
    Détermine le type de données chargées de manière robuste.
    Retourne: "tabular", "images", ou "none"
    """
    has_tabular = 'df' in st.session_state and st.session_state.df is not None
    has_images = 'X' in st.session_state and st.session_state.X is not None and 'y' in st.session_state and st.session_state.y is not None
    
    if has_images and has_tabular:
        logger.warning("Les deux types de données sont chargés, priorité aux images")
        return "images"
    elif has_images:
        return "images"
    elif has_tabular:
        return "tabular"
    else:
        return "none"

def validate_loaded_data():
    """Valide l'intégrité des données chargées."""
    data_type = get_loaded_data_type()
    
    if data_type == "images":
        X = st.session_state.get("X")
        y = st.session_state.get("y")
        if X is None or y is None:
            return False, "Données images corrompues"
        if len(X) == 0 or len(y) == 0:
            return False, "Aucune image chargée"
        if len(X) != len(y):
            return False, "Incohérence entre images et labels"
        return True, "Images validées"
    
    elif data_type == "tabular":
        df = st.session_state.get("df")
        if df is None:
            return False, "DataFrame non chargé"
        if len(df) == 0:
            return False, "DataFrame vide"
        return True, "Données tabulaires validées"
    
    return False, "Aucune donnée chargée"

# === INITIALISATION ===
DashboardStateManager.initialize()

# Configuration de la navigation dynamique
setup_navigation()

st.title("📊 Dashboard Exploratoire")

# Vérification robuste des données
data_type = get_loaded_data_type()

if data_type == "none":
    st.error("📊 Aucun dataset chargé")
    st.info("Chargez un dataset depuis la page d'accueil pour commencer l'analyse.")
    if st.button("🏠 Retour à l'accueil"):
        st.switch_page("main.py")
    st.stop()

# Validation des données
is_valid, validation_msg = validate_loaded_data()
if not is_valid:
    st.error(f"❌ {validation_msg}")
    st.stop()

# === CONFIGURATION DES TABS ADAPTATIVE ===
if data_type == "images":
    tabs = st.tabs(["📊 Statistiques", "🔍 Qualité Images", "🖼️ Échantillons", "📈 Distributions"])
else:  # data_type == "tabular"
    tabs = st.tabs([
        "📈 Qualité", "🔬 Variables", "🔗 Relations", "🌐 Corrélations", 
        "📄 Aperçu", "🗑️ Nettoyage"
    ])

# === HEADER ADAPTATIF ===
if data_type == "images":
    # Header pour l'analyse d'images
    X = st.session_state["X"]
    y = st.session_state["y"]
    data_dir = st.session_state.get("data_dir", "N/A")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📷 Images", f"{len(X):,}")
    with col2:
        st.metric("📐 Dimensions", f"{X.shape[1]}×{X.shape[2]}")
    with col3:
        n_classes = len(np.unique(y))
        st.metric("🎯 Classes", n_classes)
    with col4:
        memory_mb = X.nbytes / (1024**2)
        st.metric("💾 Mémoire", f"{memory_mb:.1f} MB")
        
else:
    # Header pour l'analyse tabulaire
    df = st.session_state.df
    
    try:
        df = DataValidator.validate_dataframe(df)
    except Exception as e:
        st.error(f"Erreur validation données: {str(e)}")
        st.stop()

    SystemMonitor().check_resources()

    @monitor_performance("dataset_hashing")
    def get_dataset_hash(df) -> str:
        """Génère un hash stable du dataset."""
        try:
            if is_dask_dataframe(df):
                return f"dask_{hash(tuple(sorted(df.columns)))}_{df.npartitions}_{st.session_state.dashboard_version}"
            else:
                shape_hash = hash((df.shape[0], df.shape[1]))
                return f"pandas_{hash(tuple(sorted(df.columns)))}_{shape_hash}_{st.session_state.dashboard_version}"
        except Exception as e:
            logger.warning(f"Hash calculation fallback: {e}")
            return f"fallback_{int(time.time())}"

    current_hash = get_dataset_hash(df)
    if st.session_state.dataset_hash != current_hash:
        logger.info(f"Dataset changed: {current_hash}")
        st.session_state.dataset_hash = current_hash
        st.session_state.column_types = None
        DashboardStateManager.reset_selections()

    @st.cache_data(ttl=Config.CACHE_TTL, max_entries=20, show_spinner=False)
    @monitor_performance("global_metrics")
    def compute_global_metrics(_df) -> Dict[str, Any]:
        """Calcule les métriques globales avec gestion robuste."""
        try:
            n_rows = compute_if_dask(_df.shape[0]) if hasattr(_df, 'shape') else len(_df)
            n_cols = _df.shape[1] if hasattr(_df, 'shape') else 0
            try:
                total_missing = compute_if_dask(_df.isna().sum().sum())
                missing_percentage = (total_missing / (n_rows * n_cols)) * 100 if (n_rows * n_cols) > 0 else 0
            except Exception:
                total_missing = 0
                missing_percentage = 0
            try:
                duplicate_rows = compute_if_dask(_df.duplicated().sum())
            except Exception:
                duplicate_rows = 0
            memory_usage = "N/A"
            try:
                if not is_dask_dataframe(_df):
                    memory_bytes = compute_if_dask(_df.memory_usage(deep=True).sum())
                    memory_usage = memory_bytes / (1024**2)
                else:
                    memory_usage = f"Dask ({_df.npartitions} partitions)"
            except Exception:
                pass
            return {
                'n_rows': n_rows,
                'n_cols': n_cols,
                'missing_percentage': missing_percentage,
                'duplicate_rows': duplicate_rows,
                'memory_usage': memory_usage
            }
        except Exception as e:
            logger.error(f"Global metrics error: {e}")
            return {'n_rows': 0, 'n_cols': 0, 'missing_percentage': 0, 'duplicate_rows': 0, 'memory_usage': 'Error'}

    @st.cache_data(ttl=Config.CACHE_TTL, max_entries=10)
    @monitor_performance("column_detection")
    def cached_auto_detect_column_types(_df) -> Dict[str, List]:
        """Cache la détection des types de colonnes."""
        try:
            result = auto_detect_column_types(_df)
            required_keys = ['numeric', 'categorical', 'text_or_high_cardinality', 'datetime']
            for key in required_keys:
                if key not in result:
                    result[key] = []
            return result
        except Exception as e:
            logger.error(f"Column type detection failed: {e}")
            return {key: [] for key in ['numeric', 'categorical', 'text_or_high_cardinality', 'datetime']}

    if st.session_state.column_types is None:
        with st.spinner("🔍 Analyse des types de colonnes..."):
            st.session_state.column_types = cached_auto_detect_column_types(df)

    column_types = st.session_state.column_types

    st.header("📋 Vue d'ensemble du jeu de données")

    try:
        overview_metrics = compute_global_metrics(df)
        fig = plot_overview_metrics(overview_metrics)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
        else:
            st.info("📊 Métriques globales non disponibles")
    except Exception as e:
        st.error(f"Erreur métriques: {str(e)[:100]}")

    col_count = len(df.columns)
    if col_count > 8:
        col_info = f"**{col_count} colonnes** : {', '.join(list(df.columns)[:6])}... +{col_count-6}"
    else:
        col_info = f"**{col_count} colonnes** : {', '.join(list(df.columns))}"
    st.info(col_info)

# === CONTENU DES TABS ===

if data_type == "images":
    # Protection de la page (requiert X et y)
    @require_data("X", "y")
    def images_dashboard():
        # === TAB IMAGES ===
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.title("🔍 Analyse Visuelle - Computer Vision")
        
        # === RÉCUPÉRATION DES DONNÉES ===
        X = st.session_state["X"]
        y = st.session_state["y"]
        data_dir = st.session_state.get("data_dir", "N/A")
        structure = st.session_state.get("dataset_structure", {})
        
        # === HEADER AVEC INFOS ===
        col_header1, col_header2, col_header3, col_header4 = st.columns(4)
        
        with col_header1:
            st.metric("📷 Images", f"{len(X):,}")
        
        with col_header2:
            st.metric("📐 Dimensions", f"{X.shape[1]}×{X.shape[2]}")
        
        with col_header3:
            n_classes = len(np.unique(y))
            st.metric("🎯 Classes", n_classes)
        
        with col_header4:
            memory_mb = X.nbytes / (1024**2)
            st.metric("💾 Mémoire", f"{memory_mb:.1f} MB")
        
        st.markdown("---")
        
        # === TABS D'ANALYSE ===
        analysis_tabs = tabs  # Utilise les tabs définies plus haut
        
        # === TAB 1: STATISTIQUES ===
        with analysis_tabs[0]:
            st.subheader("📊 Analyse Statistique Complète")
            
            # Distribution des classes
            from collections import Counter
            label_counts = Counter(y)
            
            col_chart, col_details = st.columns([2, 1])
            
            with col_chart:
                # Déterminer le type de classification
                unique_labels = sorted(label_counts.keys())
                
                if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
                    # Cas binaire (anomalie detection)
                    label_names = ['Normal', 'Anomalie']
                    colors = ['#2ecc71', '#e74c3c']
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=label_names,
                            y=[label_counts[0], label_counts[1]],
                            marker_color=colors,
                            text=[label_counts[0], label_counts[1]],
                            textposition='auto',
                            textfont=dict(size=14, color='white')
                        )
                    ])
                    fig.update_layout(
                        title="Répartition Normal vs Anomalie",
                        xaxis_title="Type",
                        yaxis_title="Nombre d'images",
                        template="plotly_white",
                        height=400
                    )
                    
                else:
                    # Cas multi-classe
                    label_names = [f"Classe {label}" for label in unique_labels]
                    counts = [label_counts[label] for label in unique_labels]
                    
                    fig = px.bar(
                        x=label_names,
                        y=counts,
                        title="Distribution des Classes",
                        labels={'x': 'Classe', 'y': 'Nombre d\'images'},
                        text=counts
                    )
                    fig.update_traces(textposition='outside')
                    fig.update_layout(template="plotly_white", height=400)
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col_details:
                st.markdown("#### 📋 Détails par Classe")
                
                total = len(y)
                
                for label in unique_labels:
                    count = label_counts[label]
                    percentage = (count / total) * 100
                    
                    if len(unique_labels) == 2 and set(unique_labels) == {0, 1}:
                        label_name = "Normal" if label == 0 else "Anomalie"
                        color = "🟢" if label == 0 else "🔴"
                    else:
                        label_name = f"Classe {label}"
                        color = "🔵"
                    
                    st.metric(
                        f"{color} {label_name}",
                        f"{count:,}",
                        f"{percentage:.1f}%"
                    )
                
                # Déséquilibre
                st.markdown("---")
                imbalance_ratio = max(label_counts.values()) / min(label_counts.values())
                
                if imbalance_ratio > 3:
                    st.warning(f"⚠️ Dataset déséquilibré\n\nRatio: **{imbalance_ratio:.1f}:1**")
                    st.caption("Considérez l'augmentation de données ou l'utilisation de poids de classe")
                elif imbalance_ratio > 1.5:
                    st.info(f"ℹ️ Léger déséquilibre\n\nRatio: **{imbalance_ratio:.1f}:1**")
                else:
                    st.success(f"✅ Dataset équilibré\n\nRatio: **{imbalance_ratio:.1f}:1**")
            
            # Informations techniques
            st.markdown("---")
            st.subheader("🔧 Informations Techniques")
            
            col_tech1, col_tech2, col_tech3 = st.columns(3)
            
            with col_tech1:
                st.markdown("**Format des images**")
                st.write(f"- Shape: `{X.shape}`")
                st.write(f"- Dtype: `{X.dtype}`")
                st.write(f"- Channels: `{X.shape[3] if len(X.shape) == 4 else 1}`")
            
            with col_tech2:
                st.markdown("**Plage de valeurs**")
                st.write(f"- Min: `{X.min():.4f}`")
                st.write(f"- Max: `{X.max():.4f}`")
                st.write(f"- Mean: `{X.mean():.4f}`")
                st.write(f"- Std: `{X.std():.4f}`")
            
            with col_tech3:
                st.markdown("**Dataset**")
                st.write(f"- Structure: `{structure.get('type', 'N/A')}`")
                st.write(f"- Chemin: `{os.path.basename(data_dir)}`")
                st.write(f"- Mémoire: `{memory_mb:.1f} MB`")
        
        # === TAB 2: QUALITÉ ===
        with analysis_tabs[1]:
            st.subheader("🔍 Analyse de Qualité des Images")
            
            # Paramètres
            col_param1, col_param2 = st.columns(2)
            
            with col_param1:
                sample_size = st.slider(
                    "Nombre d'images à analyser",
                    min_value=50,
                    max_value=min(500, len(X)),
                    value=min(100, len(X)),
                    help="Plus le nombre est élevé, plus l'analyse est précise mais lente"
                )
            
            with col_param2:
                if st.button("🔄 Rafraîchir l'analyse", key="refresh_quality"):
                    st.rerun()
            
            # Échantillonnage
            indices = np.random.choice(len(X), sample_size, replace=False)
            sample_X = X[indices]
            
            # Normalisation pour analyse
            if sample_X.max() > 1.0:
                sample_normalized = sample_X / 255.0
            else:
                sample_normalized = sample_X
            
            # === CALCUL DES MÉTRIQUES ===
            with st.spinner("📊 Calcul des métriques de qualité..."):
                # Luminosité (brightness)
                brightness = np.mean(sample_normalized, axis=(1, 2, 3)) * 255
                
                # Contraste (standard deviation)
                contrast = np.std(sample_normalized, axis=(1, 2, 3)) * 255
                
                # Netteté (Laplacian variance)
                sharpness = []
                for img in sample_normalized[:50]:  # Limiter pour performance
                    gray = np.mean(img, axis=2)
                    laplacian = ndimage.laplace(gray)
                    sharpness.append(laplacian.var())
                sharpness = np.array(sharpness) * 1000  # Scaling pour visualisation
            
            # === VISUALISATIONS ===
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Luminosité
                fig_brightness = go.Figure()
                fig_brightness.add_trace(go.Histogram(
                    x=brightness,
                    nbinsx=30,
                    name="Luminosité",
                    marker_color='lightblue',
                    opacity=0.7
                ))
                fig_brightness.add_vline(
                    x=brightness.mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Moyenne: {brightness.mean():.1f}"
                )
                fig_brightness.update_layout(
                    title="Distribution de la Luminosité",
                    xaxis_title="Luminosité (0-255)",
                    yaxis_title="Fréquence",
                    template="plotly_white",
                    height=350
                )
                st.plotly_chart(fig_brightness, use_container_width=True)
                
                # Métriques luminosité
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("📊 Moyenne", f"{brightness.mean():.1f}")
                with col_m2:
                    st.metric("📊 Écart-type", f"{brightness.std():.1f}")
            
            with col_viz2:
                # Contraste
                fig_contrast = go.Figure()
                fig_contrast.add_trace(go.Histogram(
                    x=contrast,
                    nbinsx=30,
                    name="Contraste",
                    marker_color='lightcoral',
                    opacity=0.7
                ))
                fig_contrast.add_vline(
                    x=contrast.mean(),
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Moyenne: {contrast.mean():.1f}"
                )
                fig_contrast.update_layout(
                    title="Distribution du Contraste",
                    xaxis_title="Contraste (std)",
                    yaxis_title="Fréquence",
                    template="plotly_white",
                    height=350
                )
                st.plotly_chart(fig_contrast, use_container_width=True)
                
                # Métriques contraste
                col_m3, col_m4 = st.columns(2)
                with col_m3:
                    st.metric("📊 Moyenne", f"{contrast.mean():.1f}")
                with col_m4:
                    st.metric("📊 Écart-type", f"{contrast.std():.1f}")
            
            # Netteté
            if len(sharpness) > 0:
                st.markdown("---")
                st.markdown("#### 🎯 Analyse de Netteté (échantillon de 50 images)")
                
                fig_sharpness = go.Figure()
                fig_sharpness.add_trace(go.Histogram(
                    x=sharpness,
                    nbinsx=20,
                    marker_color='lightgreen',
                    opacity=0.7
                ))
                fig_sharpness.update_layout(
                    title="Distribution de la Netteté",
                    xaxis_title="Variance du Laplacien",
                    yaxis_title="Fréquence",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig_sharpness, use_container_width=True)
                
                col_sharp1, col_sharp2 = st.columns(2)
                with col_sharp1:
                    st.metric("📊 Netteté Moyenne", f"{sharpness.mean():.2f}")
                with col_sharp2:
                    blur_threshold = 1.0
                    blurry_count = np.sum(sharpness < blur_threshold)
                    if blurry_count > 0:
                        st.warning(f"⚠️ {blurry_count} image(s) potentiellement floue(s)")
                    else:
                        st.success("✅ Toutes les images sont nettes")
            
            # === DÉTECTION D'ANOMALIES DE QUALITÉ ===
            st.markdown("---")
            st.subheader("⚠️ Détection d'Images Problématiques")
            
            # Seuils
            dark_threshold = 50
            bright_threshold = 200
            low_contrast_threshold = 20
            
            # Détection
            dark_images = np.where(brightness < dark_threshold)[0]
            bright_images = np.where(brightness > bright_threshold)[0]
            low_contrast_images = np.where(contrast < low_contrast_threshold)[0]
            
            col_issue1, col_issue2, col_issue3 = st.columns(3)
            
            with col_issue1:
                if len(dark_images) > 0:
                    st.warning(f"🌑 **{len(dark_images)} image(s) trop sombre(s)**")
                    st.caption(f"Luminosité < {dark_threshold}")
                else:
                    st.success("✅ Pas d'images trop sombres")
            
            with col_issue2:
                if len(bright_images) > 0:
                    st.warning(f"☀️ **{len(bright_images)} image(s) surexposée(s)**")
                    st.caption(f"Luminosité > {bright_threshold}")
                else:
                    st.success("✅ Pas d'images surexposées")
            
            with col_issue3:
                if len(low_contrast_images) > 0:
                    st.warning(f"🎭 **{len(low_contrast_images)} image(s) faible contraste**")
                    st.caption(f"Contraste < {low_contrast_threshold}")
                else:
                    st.success("✅ Bon contraste général")
            
            # Affichage des images problématiques
            if len(dark_images) > 0 or len(bright_images) > 0 or len(low_contrast_images) > 0:
                st.markdown("#### 📸 Aperçu des Images Problématiques")
                
                problematic_indices = list(set(
                    list(dark_images[:3]) + 
                    list(bright_images[:3]) + 
                    list(low_contrast_images[:3])
                ))
                
                if problematic_indices:
                    cols = st.columns(min(3, len(problematic_indices)))
                    
                    for idx, col in zip(problematic_indices, cols):
                        with col:
                            img_idx = indices[idx]
                            image = X[img_idx]
                            
                            # Classification du problème
                            problems = []
                            if idx in dark_images:
                                problems.append("🌑 Sombre")
                            if idx in bright_images:
                                problems.append("☀️ Surexposé")
                            if idx in low_contrast_images:
                                problems.append("🎭 Faible contraste")
                            
                            st.image(
                                image, 
                                caption=f"Image {img_idx}: {', '.join(problems)}",
                                use_column_width=True
                            )
        
        # === TAB 3: ÉCHANTILLONS ===
        with analysis_tabs[2]:
            st.subheader("🖼️ Exploration Visuelle des Images")
            
            # Configuration
            col_config1, col_config2, col_config3 = st.columns(3)
            
            with col_config1:
                n_samples = st.slider(
                    "Nombre d'images à afficher",
                    min_value=1,
                    max_value=20,
                    value=6
                )
            
            with col_config2:
                sample_mode = st.selectbox(
                    "Mode d'échantillonnage",
                    options=["Aléatoire", "Par classe", "Problématiques"],
                    help="Choisissez comment sélectionner les images"
                )
            
            with col_config3:
                if st.button("🔄 Nouvel échantillon", key="refresh_samples"):
                    st.rerun()
            
            # Sélection des indices
            if sample_mode == "Aléatoire":
                sample_indices = np.random.choice(len(X), n_samples, replace=False)
            elif sample_mode == "Par classe":
                # Échantillon équilibré par classe
                unique_classes = np.unique(y)
                samples_per_class = max(1, n_samples // len(unique_classes))
                
                sample_indices = []
                for cls in unique_classes:
                    class_indices = np.where(y == cls)[0]
                    if len(class_indices) > 0:
                        selected = np.random.choice(
                            class_indices, 
                            min(samples_per_class, len(class_indices)), 
                            replace=False
                        )
                        sample_indices.extend(selected)
                
                # Compléter si nécessaire
                if len(sample_indices) < n_samples:
                    remaining = n_samples - len(sample_indices)
                    additional = np.random.choice(
                        [i for i in range(len(X)) if i not in sample_indices],
                        min(remaining, len(X) - len(sample_indices)),
                        replace=False
                    )
                    sample_indices.extend(additional)
            else:  # Problématiques
                # Utiliser l'analyse de qualité précédente
                if 'brightness' in locals() and len(brightness) > 0:
                    # Prendre les 3 plus sombres, 3 plus claires, etc.
                    darkest = np.argsort(brightness)[:min(2, len(brightness))]
                    brightest = np.argsort(brightness)[-min(2, len(brightness)):]
                    lowest_contrast = np.argsort(contrast)[:min(2, len(contrast))]
                    
                    problematic_sample_indices = list(set(
                        list(darkest) + list(brightest) + list(lowest_contrast)
                    ))
                    
                    sample_indices = [indices[i] for i in problematic_sample_indices[:n_samples]]
                else:
                    sample_indices = np.random.choice(len(X), n_samples, replace=False)
            
            # Affichage des images
            st.markdown(f"#### 📸 {n_samples} Images Sélectionnées")
            
            # Configuration de la grille
            n_cols = 3
            n_rows = (n_samples + n_cols - 1) // n_cols
            
            for row in range(n_rows):
                cols = st.columns(n_cols)
                for col_idx in range(n_cols):
                    img_idx = row * n_cols + col_idx
                    if img_idx < len(sample_indices):
                        with cols[col_idx]:
                            idx = sample_indices[img_idx]
                            image = X[idx]
                            label = y[idx]
                            
                            # Déterminer le type de label
                            if len(np.unique(y)) == 2 and set(np.unique(y)) == {0, 1}:
                                label_text = "Normal" if label == 0 else "Anomalie"
                                color = "🟢" if label == 0 else "🔴"
                            else:
                                label_text = f"Classe {label}"
                                color = "🔵"
                            
                            st.image(
                                image,
                                caption=f"{color} {label_text} (Index: {idx})",
                                use_column_width=True
                            )
            
            # Informations détaillées
            with st.expander("📋 Détails des Images Sélectionnées", expanded=False):
                details_data = []
                for idx in sample_indices:
                    image = X[idx]
                    details_data.append({
                        "Index": idx,
                        "Classe": y[idx],
                        "Label": "Normal" if y[idx] == 0 else "Anomalie" if len(np.unique(y)) == 2 else f"Classe {y[idx]}",
                        "Dimensions": f"{image.shape[0]}×{image.shape[1]}",
                        "Valeur Min": f"{image.min():.2f}",
                        "Valeur Max": f"{image.max():.2f}",
                        "Moyenne": f"{image.mean():.2f}"
                    })
                
                st.dataframe(pd.DataFrame(details_data), use_container_width=True)
            
        # === TAB 4: DISTRIBUTIONS ===
        with analysis_tabs[3]:
            st.subheader("📈 Analyse des Distributions")
            
            # Distribution des canaux de couleur
            st.markdown("#### 🎨 Distribution des Canaux de Couleur")
            
            # Échantillonnage pour performance
            sample_size_dist = min(100, len(X))
            dist_indices = np.random.choice(len(X), sample_size_dist, replace=False)
            sample_dist = X[dist_indices]
            
            # Normalisation si nécessaire
            if sample_dist.max() > 1.0:
                sample_dist = sample_dist / 255.0
            
            # Calcul des distributions par canal
            channels = ['Rouge', 'Vert', 'Bleu'] if sample_dist.shape[-1] == 3 else ['Niveau de Gris']
            
            fig_channels = go.Figure()
            
            for i, channel_name in enumerate(channels):
                if sample_dist.shape[-1] == 3:
                    channel_data = sample_dist[:, :, :, i].flatten()
                else:
                    channel_data = sample_dist.flatten()
                
                fig_channels.add_trace(go.Histogram(
                    x=channel_data,
                    name=channel_name,
                    opacity=0.7,
                    nbinsx=50
                ))
            
            fig_channels.update_layout(
                title="Distribution des Valeurs par Canal",
                xaxis_title="Valeur du Pixel",
                yaxis_title="Fréquence",
                barmode='overlay',
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_channels, use_container_width=True)
            
            # Heatmap de corrélation entre canaux
            if sample_dist.shape[-1] == 3:
                st.markdown("#### 🔥 Corrélation entre Canaux")
                
                # Calcul des corrélations
                red = sample_dist[:, :, :, 0].flatten()
                green = sample_dist[:, :, :, 1].flatten()
                blue = sample_dist[:, :, :, 2].flatten()
                
                corr_matrix = np.corrcoef([red, green, blue])
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=channels,
                    y=channels,
                    colorscale='Blues',
                    text=[[f'{corr:.3f}' for corr in row] for row in corr_matrix],
                    texttemplate="%{text}",
                    textfont={"size": 14}
                ))
                
                fig_heatmap.update_layout(
                    title="Matrice de Corrélation entre Canaux",
                    height=400
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Distribution spatiale
            st.markdown("#### 🗺️ Distribution Spatiale des Intensités")
            
            # Calcul de l'intensité moyenne par position
            if sample_dist.shape[-1] == 3:
                intensity = np.mean(sample_dist, axis=3)  # Moyenne sur les canaux
            else:
                intensity = sample_dist.squeeze()
            
            mean_intensity_map = np.mean(intensity, axis=0)
            
            fig_spatial = go.Figure(data=go.Heatmap(
                z=mean_intensity_map,
                colorscale='Viridis',
                colorbar=dict(title="Intensité Moyenne")
            ))
            
            fig_spatial.update_layout(
                title="Carte d'Intensité Moyenne Spatiale",
                xaxis_title="Position X",
                yaxis_title="Position Y",
                height=400
            )
            
            st.plotly_chart(fig_spatial, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    images_dashboard()

else:
    # Protection de la page (requiert df)
    @require_data("df")
    def tabular_dashboard():
        with tabs[0]:  # Qualité
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.subheader("📊 Qualité des Données")
            col1, col2 = st.columns(2)
            with col1:
                try:
                    missing_fig = plot_missing_values_overview(df)
                    if missing_fig:
                        st.plotly_chart(missing_fig, use_container_width=True, config={'responsive': True})
                    else:
                        st.success("✅ Aucune valeur manquante")
                except Exception as e:
                    st.error("Erreur valeurs manquantes")
                    logger.error(f"Missing values plot: {e}")
            with col2:
                try:
                    cardinality_fig = plot_cardinality_overview(df, column_types)
                    if cardinality_fig:
                        st.plotly_chart(cardinality_fig, use_container_width=True, config={'responsive': True})
                    else:
                        st.info("📊 Cardinalité uniforme")
                except Exception as e:
                    st.error("Erreur cardinalité")
                    logger.error(f"Cardinality plot: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with tabs[1]:  # Variables
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.subheader("🔍 Analyse Univariée")
            available_columns = list(df.columns)
            if not available_columns:
                st.warning("Aucune colonne disponible")
            else:
                if not st.session_state.selected_univar_col or st.session_state.selected_univar_col not in available_columns:
                    st.session_state.selected_univar_col = available_columns[0]
                selected_col = st.selectbox(
                    "Variable à analyser :",
                    options=available_columns,
                    index=available_columns.index(st.session_state.selected_univar_col),
                    format_func=lambda x: f"{x} ({'Numérique' if x in column_types.get('numeric', []) else 'Catégorielle'})",
                    key="univar_selector"
                )
                if selected_col != st.session_state.selected_univar_col:
                    st.session_state.selected_univar_col = selected_col
                if selected_col:
                    try:
                        sample_df = DataSampler.get_sample(df)
                        col_data = sample_df[selected_col].dropna()
                        if col_data.empty:
                            st.warning(f"Aucune donnée valide pour **{selected_col}**")
                        else:
                            if selected_col in column_types.get('numeric', []):
                                stats_cols = st.columns(4)
                                with stats_cols[0]:
                                    st.metric("Moyenne", f"{col_data.mean():.3f}")
                                with stats_cols[1]:
                                    st.metric("Médiane", f"{col_data.median():.3f}")
                                with stats_cols[2]:
                                    st.metric("Écart-type", f"{col_data.std():.3f}")
                                with stats_cols[3]:
                                    st.metric("Uniques", f"{col_data.nunique():,}")
                                with st.spinner("📊 Génération du graphique..."):
                                    fig = plot_distribution(col_data, selected_col)
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
                                    else:
                                        st.info("Graphique non disponible")
                            else:
                                value_counts = col_data.value_counts().head(20)
                                if not value_counts.empty:
                                    col_chart, col_table = st.columns([2, 1])
                                    with col_table:
                                        df_display = value_counts.reset_index()
                                        df_display.columns = ['Valeur', 'Count']
                                        st.dataframe(df_display, height=400, use_container_width=True)
                                    with col_chart:
                                        fig = px.bar(
                                            x=value_counts.index.astype(str),
                                            y=value_counts.values,
                                            labels={'x': selected_col, 'y': 'Fréquence'},
                                            title=f"Distribution de {selected_col}"
                                        )
                                        fig.update_layout(template="plotly_white", height=400)
                                        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
                                else:
                                    st.info("Aucune donnée à afficher")
                    except Exception as e:
                        st.error(f"Erreur analyse de {selected_col}")
                        logger.error(f"Univariate analysis error for {selected_col}: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

        with tabs[2]:  # Relations
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.subheader("🔗 Relations entre Variables")
            available_columns = list(df.columns)
            if len(available_columns) >= 2:
                if not st.session_state.selected_bivar_col1 or st.session_state.selected_bivar_col1 not in available_columns:
                    st.session_state.selected_bivar_col1 = available_columns[0]
                if not st.session_state.selected_bivar_col2 or st.session_state.selected_bivar_col2 not in available_columns:
                    st.session_state.selected_bivar_col2 = available_columns[1] if len(available_columns) > 1 else available_columns[0]
                col1, col2 = st.columns(2)
                with col1:
                    var1 = st.selectbox(
                        "Variable 1",
                        options=available_columns,
                        index=available_columns.index(st.session_state.selected_bivar_col1),
                        key="bivar_var1"
                    )
                    if var1 != st.session_state.selected_bivar_col1:
                        st.session_state.selected_bivar_col1 = var1
                with col2:
                    var2 = st.selectbox(
                        "Variable 2",
                        options=available_columns,
                        index=available_columns.index(st.session_state.selected_bivar_col2),
                        key="bivar_var2"
                    )
                    if var2 != st.session_state.selected_bivar_col2:
                        st.session_state.selected_bivar_col2 = var2
                if var1 != var2:
                    try:
                        type1 = 'numeric' if var1 in column_types.get('numeric', []) else 'categorical'
                        type2 = 'numeric' if var2 in column_types.get('numeric', []) else 'categorical'
                        sample_df = DataSampler.get_sample(df, Config.MAX_BIVARIATE_SAMPLE)
                        if not sample_df[[var1, var2]].empty:
                            with st.spinner("📊 Génération de l'analyse bivariée..."):
                                biv_fig = plot_bivariate_analysis(sample_df, var1, var2, type1, type2)
                                if biv_fig:
                                    st.plotly_chart(biv_fig, use_container_width=True, config={'responsive': True})
                                else:
                                    st.info("Graphique non disponible pour cette combinaison")
                        else:
                            st.warning("Données insuffisantes")
                    except Exception as e:
                        st.error("Erreur analyse bivariée")
                        logger.error(f"Bivariate analysis error: {e}")
                else:
                    st.warning("Sélectionnez deux variables différentes")
            else:
                st.warning("Au moins 2 colonnes nécessaires")
            st.markdown('</div>', unsafe_allow_html=True)

        with tabs[3]:  # Corrélations
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.subheader("🌐 Matrice de Corrélations")
            col_config1, col_config2 = st.columns(2)
            with col_config1:
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if numeric_cols:
                    target_col = st.selectbox(
                        "Variable cible (optionnelle)",
                        options=[None] + numeric_cols,
                        key="corr_target_select"
                    )
                else:
                    st.warning("Aucune variable numérique disponible")
                    target_col = None
            with col_config2:
                use_simple_mode = st.checkbox("Mode simple (recommandé)", value=True, key="simple_mode")
            if st.button("🔄 Générer la matrice", type="primary", key="generate_corr"):
                with st.spinner("📊 Calcul des corrélations..."):
                    try:
                        sample_df = DataSampler.get_sample(df, max_rows=3000)
                        if use_simple_mode:
                            corr_fig, used_cols = create_simple_correlation_heatmap(sample_df, max_cols=15)
                        else:
                            corr_fig, used_cols = plot_correlation_heatmap(
                                sample_df,
                                target_column=target_col,
                                task_type="classification" if target_col else None
                            )
                        if corr_fig:
                            st.plotly_chart(corr_fig, use_container_width=True, config={'responsive': True})
                            st.success(f"✅ Matrice générée avec {len(used_cols)} variables")
                        else:
                            st.warning("❌ Impossible de générer la matrice")
                            st.info("Essayez avec le mode simple activé")
                    except Exception as e:
                        st.error("Erreur lors du calcul des corrélations")
                        logger.error(f"Correlation error: {e}")
                        try:
                            st.info("Tentative avec méthode alternative...")
                            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 1:
                                corr_matrix = sample_df[numeric_cols].corr()
                                fig = px.imshow(corr_matrix, text_auto=".2f", aspect="auto")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Pas assez de variables numériques")
                        except Exception as fallback_e:
                            st.error("Échec de toutes les méthodes")
            st.markdown('</div>', unsafe_allow_html=True)

        with tabs[4]:  # Aperçu
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.subheader("📄 Aperçu des Données Brutes")
            try:
                raw_df = st.session_state.get('df_raw', df)
                total_rows = compute_if_dask(raw_df.shape[0])
                col_config1, col_config2, col_config3 = st.columns(3)
                with col_config1:
                    preview_size = st.slider(
                        "Nombre de lignes à afficher",
                        min_value=10,
                        max_value=min(500, total_rows),
                        value=min(Config.MAX_PREVIEW_ROWS, total_rows),
                        key="preview_size_slider"
                    )
                with col_config2:
                    show_from_start = st.radio("Affichage", ["Début", "Échantillon"], key="preview_type")
                with col_config3:
                    show_dtypes = st.checkbox("Afficher les types", value=False, key="show_dtypes")
                if show_from_start == "Début":
                    display_df = compute_if_dask(raw_df.head(preview_size))
                else:
                    display_df = DataSampler.get_sample(raw_df, preview_size)
                display_df_truncated = display_df.copy()
                for col in display_df_truncated.select_dtypes(include=['object']).columns:
                    display_df_truncated[col] = display_df_truncated[col].astype(str).apply(
                        lambda x: x[:50] + "..." if len(str(x)) > 50 else x
                    )
                st.dataframe(display_df_truncated, height=400, use_container_width=True)
                if show_dtypes:
                    with st.expander("📊 Types de données des colonnes"):
                        dtypes_info = []
                        for col in display_df.columns:
                            dtype = str(display_df[col].dtype)
                            non_null_count = display_df[col].count()
                            total_count = len(display_df[col])
                            null_percentage = ((total_count - non_null_count) / total_count * 100) if total_count > 0 else 0
                            dtypes_info.append({
                                'Colonne': col,
                                'Type': dtype,
                                'Non-null': non_null_count,
                                'Null (%)': f"{null_percentage:.1f}%"
                            })
                        dtypes_df = pd.DataFrame(dtypes_info)
                        st.dataframe(dtypes_df, use_container_width=True)
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.caption(f"📊 {len(display_df)} lignes affichées sur {total_rows:,} total")
                with info_col2:
                    st.caption(f"📋 {len(display_df.columns)} colonnes")
                with info_col3:
                    if len(display_df) > 0 and len(display_df.columns) > 0:
                        missing_pct = (display_df.isnull().sum().sum() / (len(display_df) * len(display_df.columns))) * 100
                        st.caption(f"🕳️ {missing_pct:.1f}% valeurs manquantes")
                    else:
                        st.caption("🕳️ Données insuffisantes")
                st.markdown("---")
                col_download1, col_download2 = st.columns([3, 1])
                with col_download2:
                    if st.button("💾 Télécharger l'échantillon", key="download_sample"):
                        try:
                            csv = display_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Télécharger CSV",
                                data=csv,
                                file_name=f"echantillon_donnees_{time.strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv",
                                key="download_csv"
                            )
                        except Exception as download_error:
                            st.error("❌ Erreur lors de la préparation du téléchargement")
                            logger.error(f"Download error: {download_error}")
            except Exception as e:
                st.error("❌ Erreur lors de l'affichage de l'aperçu des données")
                logger.error(f"Data preview error: {e}")
                try:
                    st.info("🔄 Tentative de chargement simplifié...")
                    fallback_df = compute_if_dask(df.head(50))
                    st.dataframe(fallback_df, height=300, use_container_width=True)
                    st.caption(f"📊 Affichage de secours: 50 premières lignes sur {compute_if_dask(df.shape[0]):,} total")
                except Exception as fallback_error:
                    st.error("🚨 Impossible d'afficher les données")
                    logger.error(f"Fallback preview also failed: {fallback_error}")
            st.markdown('</div>', unsafe_allow_html=True)

        with tabs[5]:  # Nettoyage
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.subheader("🗑️ Nettoyage des Données")
            st.markdown("### 🔍 Détection des colonnes inutiles")
            mode = st.radio(
                "Mode de sélection des colonnes à supprimer :",
                options=["Automatique", "Manuelle"],
                horizontal=True
            )
            cols_to_remove = []
            if mode == "Automatique":
                col_detect, col_action = st.columns([2, 1])
                with col_detect:
                    if st.button("🔎 Analyser les colonnes inutiles", key="analyze_useless"):
                        with st.spinner("Analyse en cours..."):
                            try:
                                useless_cols = detect_useless_columns(df, threshold_missing=0.7)
                                st.session_state.useless_candidates = useless_cols
                                if useless_cols:
                                    st.success(f"✅ {len(useless_cols)} colonne(s) potentiellement inutile(s) détectée(s)")
                                    st.write(
                                        "**Colonnes détectées:**",
                                        ", ".join(useless_cols[:5]) + ("..." if len(useless_cols) > 5 else "")
                                    )
                                    cols_to_remove = useless_cols
                                else:
                                    st.info("🎉 Aucune colonne inutile détectée")
                            except Exception as e:
                                st.error("❌ Erreur lors de l'analyse")
                                logger.error(f"Useless columns detection error: {e}")
                with col_action:
                    if st.session_state.useless_candidates:
                        if st.button("🗑️ Supprimer les colonnes inutiles", type="primary"):
                            try:
                                valid_cols = [col for col in st.session_state.useless_candidates if col in df.columns]
                                if valid_cols:
                                    if is_dask_dataframe(df):
                                        new_df = df.drop(columns=valid_cols).persist()
                                    else:
                                        new_df = df.drop(columns=valid_cols)
                                    st.session_state.df = new_df
                                    st.session_state.useless_candidates = []
                                    st.session_state.column_types = None
                                    st.session_state.dashboard_version += 1
                                    st.success(f"✅ {len(valid_cols)} colonne(s) supprimée(s)")
                                    st.rerun()
                                else:
                                    st.warning("⚠️ Aucune colonne valide à supprimer")
                            except Exception as e:
                                st.error("❌ Erreur lors de la suppression")
                                logger.error(f"Column removal error: {e}")
            else:
                all_cols = df.columns.tolist()
                cols_to_remove = st.multiselect(
                    "Sélectionnez les colonnes à supprimer",
                    options=all_cols,
                    default=[]
                )
                if cols_to_remove:
                    if st.button("🗑️ Supprimer les colonnes sélectionnées", type="primary"):
                        try:
                            valid_cols = [col for col in cols_to_remove if col in df.columns]
                            if valid_cols:
                                if is_dask_dataframe(df):
                                    new_df = df.drop(columns=valid_cols).persist()
                                else:
                                    new_df = df.drop(columns=valid_cols)
                                st.session_state.df = new_df
                                st.session_state.column_types = None
                                st.session_state.dashboard_version += 1
                                st.success(f"✅ {len(valid_cols)} colonne(s) supprimée(s)")
                                st.rerun()
                            else:
                                st.warning("⚠️ Aucune colonne valide à supprimer")
                        except Exception as e:
                            st.error("❌ Erreur lors de la suppression")
                            logger.error(f"Manual column removal error: {e}")
            st.markdown("---")
            st.markdown("### ✏️ Renommage des colonnes")
            col_rename1, col_rename2 = st.columns(2)
            with col_rename1:
                if df.columns.tolist():
                    col_to_rename = st.selectbox(
                        "Colonne à renommer",
                        options=df.columns.tolist(),
                        key="rename_select"
                    )
                else:
                    col_to_rename = None
                    st.warning("Aucune colonne disponible")
            with col_rename2:
                new_name = st.text_input(
                    "Nouveau nom",
                    placeholder="Nouveau nom de colonne",
                    key="rename_input"
                )
            col_add, col_clear = st.columns(2)
            with col_add:
                if st.button("➕ Ajouter au plan de renommage", key="add_rename"):
                    if col_to_rename and new_name and DataValidator.is_valid_column_name(new_name):
                        if new_name not in df.columns:
                            if (col_to_rename, new_name) not in st.session_state.rename_list:
                                st.session_state.rename_list.append((col_to_rename, new_name))
                                st.success(f"✅ {col_to_rename} → {new_name} ajouté")
                            else:
                                st.warning("⚠️ Ce renommage est déjà planifié")
                        else:
                            st.error("❌ Ce nom de colonne existe déjà")
                    else:
                        st.error("❌ Nom invalide ou vide")
            with col_clear:
                if st.button("🗑️ Vider la liste", key="clear_renames"):
                    st.session_state.rename_list = []
                    st.success("✅ Liste vidée")
            if st.session_state.rename_list:
                st.markdown("**📋 Renommages planifiés:**")
                rename_df = pd.DataFrame(st.session_state.rename_list, columns=["Ancien nom", "Nouveau nom"])
                st.dataframe(rename_df, use_container_width=True)
                if st.button("✅ Appliquer tous les renommages", type="primary", key="apply_renames"):
                    try:
                        rename_dict = dict(st.session_state.rename_list)
                        valid_renames = {old: new for old, new in rename_dict.items() if old in df.columns}
                        if valid_renames:
                            if is_dask_dataframe(df):
                                new_df = df.rename(columns=valid_renames).persist()
                            else:
                                new_df = df.rename(columns=valid_renames)
                            st.session_state.df = new_df
                            st.session_state.rename_list = []
                            st.session_state.column_types = None
                            st.session_state.dashboard_version += 1
                            st.success(f"✅ {len(valid_renames)} colonne(s) renommée(s)")
                            st.rerun()
                        else:
                            st.warning("⚠️ Aucun renommage valide à appliquer")
                    except Exception as e:
                        st.error("❌ Erreur lors du renommage")
                        logger.error(f"Rename error: {e}")
            st.markdown("---")
            st.markdown("### 🛠️ Actions de maintenance")
            col_maint1, col_maint2 = st.columns(2)
            with col_maint1:
                if st.button("🧹 Nettoyer la mémoire", key="cleanup_mem"):
                    try:
                        cleanup_memory()
                        st.success("✅ Mémoire nettoyée")
                    except Exception as e:
                        st.error("❌ Erreur de nettoyage")
            with col_maint2:
                if st.button("🔄 Rafraîchir l'analyse", key="refresh_analysis"):
                    try:
                        st.session_state.column_types = None
                        st.cache_data.clear()
                        st.success("✅ Analyse rafraîchie")
                        st.rerun()
                    except Exception as e:
                        st.error("❌ Erreur de rafraîchissement")
            st.markdown('</div>', unsafe_allow_html=True)

    tabular_dashboard()

# === FOOTER ADAPTATIF ===
st.markdown("---")
footer_cols = st.columns(4)

with footer_cols[0]:
    if data_type == "images":
        st.caption(f"📷 {len(X):,} images")
    else:
        try:
            n_rows = compute_if_dask(df.shape[0])
            n_cols = df.shape[1]
            st.caption(f"📊 {n_rows:,} × {n_cols}")
        except:
            st.caption("📊 Données non disponibles")

with footer_cols[1]:
    if data_type == "images":
        memory_mb = X.nbytes / (1024**2)
        st.caption(f"💾 {memory_mb:.1f} MB")
    else:
        try:
            if not is_dask_dataframe(df):
                memory_mb = compute_if_dask(df.memory_usage(deep=True).sum()) / (1024**2)
                st.caption(f"💾 {memory_mb:.1f} MB")
            else:
                st.caption(f"💾 {df.npartitions} partitions")
        except:
            st.caption("💾 N/A")

with footer_cols[2]:
    try:
        sys_mem = psutil.virtual_memory().percent
        status = "🔴" if sys_mem > Config.MEMORY_CRITICAL else "🟡" if sys_mem > Config.MEMORY_WARNING else "🟢"
        st.caption(f"{status} {sys_mem:.0f}% RAM")
    except:
        st.caption("🔧 RAM: N/A")

with footer_cols[3]:
    st.caption(f"🕒 {time.strftime('%H:%M:%S')}")

gc.collect()