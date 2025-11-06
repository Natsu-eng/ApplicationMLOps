"""
Page d'évaluation des modèles - Version Moderne et Production-Ready
Interface utilisateur avancée avec animations, thèmes sombres et visualisations interactives
"""
import logging
import os
import pickle
import traceback
import numpy as np
import streamlit as st
import pandas as pd
import time
import json
import plotly.express as px
import plotly.graph_objects as go
import gc
import concurrent.futures
from typing import Dict, Optional, List, Any
from src.evaluation.model_plots import ModelEvaluationVisualizer
from src.evaluation.metrics import get_system_metrics
from utils.report_generator import generate_pdf_report
from src.config.constants import TRAINING_CONSTANTS, VISUALIZATION_CONSTANTS
from datetime import datetime
from monitoring.decorators import monitor_operation
from src.shared.logging import get_logger
from monitoring.state_managers import init, AppPage

STATE = init()

logger = get_logger(__name__)

# Import MLflow avec gestion robuste
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

# Configuration page avec thème moderne
st.set_page_config(
    page_title="Évaluation ML Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS MODERNE AVEC ANIMATIONS ET GLASSMORPHISM (Optimisé pour Fluidité)
# ============================================================================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables CSS */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --dark-bg: #0f172a;
        --card-bg: rgba(255, 255, 255, 0.05);
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border-color: rgba(255, 255, 255, 0.1);
        --transition-fast: all 0.2s ease;
        --transition-smooth: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Reset et Base */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        box-sizing: border-box;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        min-height: 100vh;
    }
    
    /* Header Principal avec Animation Fade-In */
    .modern-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 24px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        animation: fadeIn 0.8s ease-out;
        opacity: 0;
        animation-fill-mode: forwards;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.98); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .modern-header h1 {
        background: linear-gradient(135deg, #fff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    
    .modern-header p {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Cartes Métriques Glassmorphism Optimisées */
    .metric-glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1.8rem;
        margin: 0.8rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: var(--transition-smooth);
        position: relative;
        overflow: hidden;
    }
    
    .metric-glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        opacity: 0;
        transition: var(--transition-fast);
    }
    
    .metric-glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.3);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .metric-glass-card:hover::before {
        opacity: 1;
    }
    
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-secondary);
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #fff 0%, #e0e7ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    
    .metric-subtitle {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.5);
        margin-top: 0.5rem;
    }
    
    /* Badge Statut Animé (Plus Fluide) */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        animation: pulseSmooth 2s infinite ease-in-out;
    }
    
    @keyframes pulseSmooth {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.05); opacity: 0.9; }
    }
    
    .status-success {
        background: linear-gradient(135deg, var(--success-color) 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, var(--warning-color) 0%, #d97706 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.3);
    }
    
    .status-danger {
        background: linear-gradient(135deg, var(--danger-color) 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3);
    }
    
    /* Onglets Modernes avec Fade-In */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--card-bg);
        padding: 0.5rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
        animation: fadeIn 0.6s ease-out;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: transparent;
        border-radius: 12px;
        color: var(--text-secondary);
        font-weight: 500;
        transition: var(--transition-fast);
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.08);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white !important;
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
    }
    
    /* Conteneur de Graphiques Optimisé */
    .plot-container {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        transition: var(--transition-smooth);
        opacity: 0;
        animation: fadeIn 0.5s 0.2s ease-out forwards;
    }
    
    .plot-container:hover {
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.25);
    }
    
    /* Boutons Modernes Fluides */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: var(--transition-smooth);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 6px 30px rgba(99, 102, 241, 0.4);
    }
    
    /* DataFrames Modernes */
    .dataframe {
        background: var(--card-bg) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        overflow: hidden;
        transition: var(--transition-fast);
    }
    
    .dataframe thead tr {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(99, 102, 241, 0.08);
    }
    
    /* Expanders Modernes */
    .streamlit-expanderHeader {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid var(--border-color);
        color: var(--text-primary) !important;
        font-weight: 500;
        transition: var(--transition-fast);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* Selectbox Modernes */
    .stSelectbox > div > div {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        color: var(--text-primary);
        transition: var(--transition-fast);
    }
    
    /* Scrollbar Personnalisée Fluide */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%);
    }
    
    /* Media Queries pour Responsivité (Mobile-Friendly) */
    @media (max-width: 768px) {
        .modern-header {
            padding: 1.5rem;
        }
        .metric-glass-card {
            padding: 1.2rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            font-size: 0.9rem;
        }
        .plot-container {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS UTILITAIRES CORRIGÉES (Avec Plus de Fluidité)
# ============================================================================

def log_structured(level: str, message: str, extra: Dict = None):
    """Utilisation directe du logger pour simplicité"""
    try:
        log_level = getattr(logging, level.upper())
        logger.log(log_level, message)
    except Exception as e:
        print(f"Logging error: {str(e)[:200]}")

@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def cached_plot(fig, plot_key: str):
    """Cache les graphiques avec gestion robuste et feedback"""
    with st.spinner("Chargement du graphique..."):
        try:
            if fig is None:
                logger.info(f"WARNING: Figure None dans cached_plot. Plot key: {plot_key}")
                return None
            return fig
        except Exception as e:
            logger.error(f"Erreur dans le cache graphique. Clé: {plot_key}, Erreur: {str(e)[:200]}")
            return fig

def render_modern_header():
    """Affiche un en-tête moderne et animé"""
    st.markdown("""
    <div class="modern-header">
        <h1>📊 Évaluation ML Pro</h1>
        <p>Analyse avancée et visualisation interactive de vos modèles d'apprentissage automatique</p>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(label: str, value: str, subtitle: str = "", status: str = "default"):
    """Rend une carte métrique moderne avec glassmorphism"""
    status_class = ""
    if status == "success":
        status_class = "status-success"
    elif status == "warning":
        status_class = "status-warning"
    elif status == "danger":
        status_class = "status-danger"
    
    badge_html = f'<span class="status-badge {status_class}">{value}</span>' if status != "default" else f'<div class="metric-value">{value}</div>'
    
    return f"""
    <div class="metric-glass-card">
        <div class="metric-label">{label}</div>
        {badge_html}
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """

def display_modern_metrics_header(validation_result: Dict[str, Any]):
    """Affiche l'en-tête avec métriques modernes"""
    try:
        successful_models = validation_result.get("successful_models", [])
        failed_models = validation_result.get("failed_models", [])
        total_models = len(successful_models) + len(failed_models)
        
        best_model_name = validation_result.get("best_model", "N/A")
        task_type = validation_result.get("task_type", "unknown").upper()
        
        success_rate = (len(successful_models) / total_models * 100) if total_models > 0 else 0
        
        # Détermination du statut
        if success_rate > 80:
            status = "success"
        elif success_rate > 50:
            status = "warning"
        else:
            status = "danger"
        
        # Métrique principale selon le type de tâche
        if task_type == "CLUSTERING":
            metric_title = "Silhouette"
            best_score = max([m.get('metrics', {}).get('silhouette_score', 0) for m in successful_models], default=0)
        elif task_type == "REGRESSION":
            metric_title = "R² Score"
            best_score = max([m.get('metrics', {}).get('r2', 0) for m in successful_models], default=0)
        else:
            metric_title = "Accuracy"
            best_score = max([m.get('metrics', {}).get('accuracy', 0) for m in successful_models], default=0)
        
        # Affichage en 4 colonnes responsives
        cols = st.columns([1,1,1,1])
        
        with cols[0]:
            st.markdown(
                render_metric_card(
                    "Taux de Réussite",
                    f"{success_rate:.1f}%",
                    f"{len(successful_models)}/{total_models} modèles",
                    status
                ),
                unsafe_allow_html=True
            )
        
        with cols[1]:
            st.markdown(
                render_metric_card(
                    "Meilleur Modèle",
                    best_model_name,
                    f"Type: {task_type.title()}"
                ),
                unsafe_allow_html=True
            )
        
        with cols[2]:
            st.markdown(
                render_metric_card(
                    f"Meilleur {metric_title}",
                    f"{best_score:.3f}",
                    "Score optimal atteint"
                ),
                unsafe_allow_html=True
            )
        
        with cols[3]:
            try:
                import psutil
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_gb = memory.available / (1024**3)
                
                mem_status = "success" if memory_percent < 70 else "warning" if memory_percent < 85 else "danger"
                
                st.markdown(
                    render_metric_card(
                        "Mémoire Système",
                        f"{memory_percent:.1f}%",
                        f"{memory_gb:.1f} GB disponible",
                        mem_status
                    ),
                    unsafe_allow_html=True
                )
            except:
                st.markdown(
                    render_metric_card(
                        "Système",
                        "✅ Prêt",
                        "Opérationnel"
                    ),
                    unsafe_allow_html=True
                )
    
    except Exception as e:
        st.error(f"❌ Erreur affichage métriques: {str(e)[:100]}")

@monitor_operation
def display_model_details(visualizer, model_result: Dict[str, Any], task_type: str):
    """Affiche les détails complets d'un modèle avec UI moderne et chargement fluide"""
    try:
        model_name = model_result.get('model_name', 'Unknown')
        unique_id = f"{model_name}_{int(time.time())}"
        
        # En-tête moderne
        st.markdown(f"""
        <div class="plot-container" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);">
            <h3 style="color: white; margin: 0;">🔍 {model_name}</h3>
            <p style="color: rgba(255, 255, 255, 0.7); margin-top: 0.5rem;">Analyse détaillée des performances</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Diagnostic avec tooltip
        with st.expander("🔧 Diagnostic des Données", expanded=False):
            diagnostic_info = {
                "model_name": model_name,
                "task_type": task_type,
                "has_model": model_result.get('model') is not None,
                "has_metrics": bool(model_result.get('metrics')),
                "has_X_train": model_result.get('X_train') is not None,
                "has_X_test": model_result.get('X_test') is not None,
                "has_y_train": model_result.get('y_train') is not None,
                "has_y_test": model_result.get('y_test') is not None,
                "has_labels": model_result.get('labels') is not None,
            }
            st.json(diagnostic_info)
        
        # Validation
        has_model = model_result.get('model') is not None
        has_metrics = bool(model_result.get('metrics'))
        
        if not has_model or not has_metrics:
            st.error("❌ Données insuffisantes pour l'analyse")
            return
        
        # Métriques avec colonnes responsives
        st.markdown("---")
        st.markdown("#### 📊 Métriques de Performance")
        
        metrics = model_result.get('metrics', {})
        
        if task_type == 'classification':
            cols = st.columns(4)
            cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
            cols[1].metric("Precision", f"{metrics.get('precision', 0):.3f}")
            cols[2].metric("Recall", f"{metrics.get('recall', 0):.3f}")
            cols[3].metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
        
        elif task_type == 'regression':
            cols = st.columns(3)
            cols[0].metric("R² Score", f"{metrics.get('r2', 0):.3f}")
            cols[1].metric("MAE", f"{metrics.get('mae', 0):.3f}")
            cols[2].metric("RMSE", f"{metrics.get('rmse', 0):.3f}")
        
        elif task_type == 'clustering':
            cols = st.columns(3)
            cols[0].metric("Silhouette", f"{metrics.get('silhouette_score', 0):.3f}")
            cols[1].metric("Groupes", str(metrics.get('n_clusters', 'N/A')))
            db_score = metrics.get('davies_bouldin_score', 'N/A')
            cols[2].metric("Index DB", f"{db_score:.3f}" if isinstance(db_score, (int, float)) else str(db_score))
        
        # Visualisations avec chargement asynchrone
        st.markdown("---")
        st.markdown("#### 📈 Visualisations")
        
        has_viz_data = False
        if task_type == 'clustering':
            has_viz_data = model_result.get('X_train') is not None and model_result.get('labels') is not None
        else:
            has_viz_data = model_result.get('X_test') is not None and model_result.get('y_test') is not None
        
        if not has_viz_data:
            st.warning("⚠️ Données de visualisation non disponibles")
            return
        
        def load_plot(func):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(func)
                return future.result()
        
        if task_type == 'clustering':
            cols = st.columns(2)
            with cols[0]:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown("**🔮 Visualisation des Clusters**")
                try:
                    cluster_plot = load_plot(lambda: visualizer.create_cluster_visualization(model_result))
                    if cluster_plot:
                        st.plotly_chart(cached_plot(cluster_plot, f"cluster_{unique_id}"), use_container_width=True)
                    else:
                        st.info("ℹ️ Visualisation non disponible")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)[:100]}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown("**📊 Analyse Silhouette**")
                try:
                    silhouette_plot = load_plot(lambda: visualizer.create_silhouette_analysis(model_result))
                    if silhouette_plot:
                        st.plotly_chart(cached_plot(silhouette_plot, f"silhouette_{unique_id}"), use_container_width=True)
                    else:
                        st.info("ℹ️ Analyse non disponible")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)[:100]}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif task_type == 'classification':
            cols = st.columns(2)
            with cols[0]:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown("**📊 Matrice de Confusion**")
                try:
                    cm_plot = load_plot(lambda: visualizer.create_confusion_matrix(model_result))
                    if cm_plot:
                        st.plotly_chart(cached_plot(cm_plot, f"cm_{unique_id}"), use_container_width=True)
                    else:
                        st.info("ℹ️ Matrice non disponible")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)[:100]}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown("**📈 Courbe ROC**")
                try:
                    roc_plot = load_plot(lambda: visualizer.create_roc_curve(model_result))
                    if roc_plot:
                        st.plotly_chart(cached_plot(roc_plot, f"roc_{unique_id}"), use_container_width=True)
                    else:
                        st.info("ℹ️ Courbe ROC non disponible")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)[:100]}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown("**🎯 Importance des Features**")
            try:
                feature_plot = load_plot(lambda: visualizer.create_feature_importance_plot(model_result))
                if feature_plot:
                    st.plotly_chart(cached_plot(feature_plot, f"feature_{unique_id}"), use_container_width=True)
                else:
                    st.info("ℹ️ Importance des features non disponible")
            except Exception as e:
                st.error(f"❌ Erreur: {str(e)[:100]}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif task_type == 'regression':
            cols = st.columns(2)
            with cols[0]:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown("**📉 Graphique des Résidus**")
                try:
                    residuals_plot = load_plot(lambda: visualizer.create_residuals_plot(model_result))
                    if residuals_plot:
                        st.plotly_chart(cached_plot(residuals_plot, f"residuals_{unique_id}"), use_container_width=True)
                    else:
                        st.info("ℹ️ Graphique non disponible")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)[:100]}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown("**🎯 Prédictions vs Réelles**")
                try:
                    pred_plot = load_plot(lambda: visualizer.create_predicted_vs_actual(model_result))
                    if pred_plot:
                        st.plotly_chart(cached_plot(pred_plot, f"pred_{unique_id}"), use_container_width=True)
                    else:
                        st.info("ℹ️ Graphique non disponible")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)[:100]}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Informations complémentaires
        st.markdown("---")
        st.markdown("#### ℹ️ Informations Complémentaires")
        
        cols = st.columns(3)
        
        with cols[0]:
            st.markdown("**⏱️ Performances**")
            training_time = model_result.get('training_time', 0)
            st.metric("Temps d'entraînement", f"{training_time:.2f}s")
            
        with cols[1]:
            st.markdown("**📊 Données**")
            if model_result.get('X_train') is not None:
                n_samples = len(model_result['X_train'])
                st.metric("Échantillons d'entraînement", f"{n_samples:,}")
                
        with cols[2]:
            st.markdown("**🔧 Statut**")
            if model_result.get('success', False):
                st.metric("Entraînement", "✅ Réussi")
            else:
                st.metric("Entraînement", "❌ Échoué")

    except Exception as e:
        logger.error(f"Erreur détaillée dans {model_name}. Erreur : {str(e)}, Type de tâche : {task_type}")
        st.error(f"❌ Erreur critique dans l'analyse du modèle: {str(e)}")

def display_overview_tab(validation_result: Dict[str, Any], results_data: List[Dict]):
    """Affiche l'onglet Vue d'ensemble avec métriques adaptées"""
    st.markdown("### 📊 Vue d'Ensemble des Performances")
    
    successful_models = validation_result.get("successful_models", [])
    task_type = validation_result.get("task_type", "classification")
    
    if not successful_models:
        st.info("ℹ️ Aucun modèle à afficher dans la vue d'ensemble")
        return
    
    # Graphique de comparaison
    st.markdown("#### 📈 Comparaison des Modèles")
    
    model_names = []
    scores = []
    metric_label = ""
    
    for model in successful_models:
        name = model.get('model_name', 'Unknown')
        metrics = model.get('metrics', {})
        
        if task_type == 'clustering':
            score = metrics.get('silhouette_score', 0)
            metric_label = "Score Silhouette"
        elif task_type == 'regression':
            score = metrics.get('r2', 0)
            metric_label = "R² Score"
        else:
            score = metrics.get('accuracy', 0)
            metric_label = "Accuracy"
        
        model_names.append(name)
        scores.append(score)
    
    if model_names and scores:
        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=scores,
                text=[f'{score:.3f}' for score in scores],
                textposition='auto',
                marker_color=['#28a745' if score == max(scores) else '#17a2b8' for score in scores]
            )
        ])
        
        fig.update_layout(
            title=f"Comparaison des Modèles - {metric_label}",
            xaxis_title="Modèles",
            yaxis_title=metric_label,
            template="plotly_white",
            height=500,
            transition_duration=500  # Animation fluide pour Plotly
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Tableau récapitulatif
    st.markdown("#### 📋 Tableau Récapitulatif")
    
    summary_data = []
    for model in successful_models:
        metrics = model.get('metrics', {})
        
        if task_type == 'clustering':
            row = {
                'Modèle': model.get('model_name', 'Unknown'),
                'Silhouette': f"{metrics.get('silhouette_score', 0):.3f}",
                'Groupes': str(metrics.get('n_clusters', 'N/A')),
                'Index DB': f"{metrics.get('davies_bouldin_score', 'N/A'):.3f}" if isinstance(metrics.get('davies_bouldin_score'), (int, float)) else 'N/A',
                'Temps (s)': f"{model.get('training_time', 0):.2f}"
            }
        elif task_type == 'regression':
            row = {
                'Modèle': model.get('model_name', 'Unknown'),
                'R² Score': f"{metrics.get('r2', 0):.3f}",
                'MAE': f"{metrics.get('mae', 0):.3f}",
                'RMSE': f"{metrics.get('rmse', 0):.3f}",
                'Temps (s)': f"{model.get('training_time', 0):.2f}"
            }
        else:
            row = {
                'Modèle': model.get('model_name', 'Unknown'),
                'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                'Precision': f"{metrics.get('precision', 0):.3f}",
                'Recall': f"{metrics.get('recall', 0):.3f}",
                'F1-Score': f"{metrics.get('f1_score', 0):.3f}",
                'Temps (s)': f"{model.get('training_time', 0):.2f}"
            }
        
        summary_data.append(row)
    
    if summary_data:
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

def display_mlflow_tab():
    """Affiche l'onglet MLflow avec interface moderne"""
    st.markdown("### 🔗 Exploration des Runs MLflow")
    
    if not MLFLOW_AVAILABLE:
        st.error("🚫 MLflow non disponible")
        st.info("Installez MLflow pour accéder aux runs: `pip install mlflow`")
        return
    
    # Synchronisation des runs
    mlflow_runs = getattr(st.session_state, 'mlflow_runs', [])
    
    if not mlflow_runs:
        st.warning("⚠️ Aucun run MLflow disponible")
        st.info("""
        **Pour générer des runs MLflow:**
        1. Allez dans l'onglet **Configuration ML**
        2. Chargez un dataset et configurez l'entraînement
        3. **Assurez-vous que MLflow est activé** dans les paramètres avancés
        4. Lancez l'entraînement des modèles
        5. Revenez sur cette page pour voir les runs
        """)
        return
    
    st.success(f"**📊 {len(mlflow_runs)} runs MLflow disponibles**")
    
    # Filtres avec multiselect fluide
    cols = st.columns(2)
    with cols[0]:
        status_filter = st.multiselect(
            "Filtrer par statut",
            options=['FINISHED', 'RUNNING', 'FAILED'],
            default=['FINISHED'],
            key="mlflow_status_filter"
        )
    
    # Affichage des runs
    run_data = []
    for run in mlflow_runs:
        if isinstance(run, dict) and run.get('status') in status_filter:
            run_id = run.get('run_id', 'N/A')
            model_name = run.get('tags', {}).get('mlflow.runName') or run.get('model_name', 'Unknown')
            metrics = run.get('metrics', {})
            
            row = {
                'Run ID': run_id[:8] + '...',
                'Modèle': model_name,
                'Statut': run.get('status', 'UNKNOWN')
            }
            
            # Ajout des métriques principales
            for metric in ['accuracy', 'f1', 'precision', 'recall', 'r2', 'rmse']:
                if metric in metrics:
                    row[metric] = f"{metrics[metric]:.3f}"
            
            run_data.append(row)
    
    if run_data:
        st.dataframe(pd.DataFrame(run_data), use_container_width=True)
    else:
        st.info("ℹ️ Aucun run ne correspond aux filtres")

def safe_main():
    """Point d'entrée principal sécurisé avec sidebar pour filtres globaux"""
    try:
        logger.info("🚀 Démarrage de la page d'évaluation")
        
        # Sidebar pour filtres globaux (améliore la fluidité utilisateur)
        with st.sidebar:
            st.markdown("### ⚙️ Filtres Globaux")
            task_filter = st.selectbox("Type de Tâche", ["Tous", "Classification", "Regression", "Clustering"])
            st.markdown("---")
            st.info("Utilisez ces filtres pour personnaliser l'affichage.")
        
        # En-tête moderne
        render_modern_header()
        
        # Validation des données
        if not hasattr(STATE, 'training_results') or STATE.training_results is None:
            st.error("🚫 Aucun résultat d'entraînement disponible")
            st.info("""
            **Pour utiliser cette page :**
            1. Allez dans l'onglet **'Configuration ML'**
            2. Chargez un dataset et configurez l'entraînement  
            3. Lancez l'entraînement des modèles
            4. Revenez sur cette page pour analyser les résultats
            """)
            if st.button("⚙️ Aller à l'Entraînement", type="primary", use_container_width=True):
                st.switch_page("pages/2_training.py")
            return

        # Extraction des résultats
        training_results = STATE.training_results
        if not hasattr(training_results, 'results'):
            st.error("❌ Format invalide des résultats d'entraînement")
            return

        results_data = training_results.results
        if not results_data or not isinstance(results_data, list):
            st.error("📭 Aucun résultat détaillé disponible")
            return

        # Initialisation du visualizer
        task_type = getattr(STATE, 'task_type', 'classification')
        if task_filter != "Tous" and task_filter.lower() != task_type:
            st.warning(f"⚠️ Type de tâche filtré ({task_filter}) ne correspond pas aux données ({task_type.upper()}). Affichage complet.")
        st.success(f"🔧 **Type de tâche détecté :** {task_type.upper()}")

        try:
            visualizer = ModelEvaluationVisualizer(results_data)
            validation_result = visualizer.validation_result
            validation_result["task_type"] = task_type
        except Exception as e:
            st.error(f"❌ Erreur lors de l'initialisation du visualiseur : {str(e)}")
            return

        if not validation_result.get("has_results", False):
            st.error("📭 Aucune donnée valide trouvée dans les résultats")
            return

        # En-tête des métriques
        display_modern_metrics_header(validation_result)

        # Configuration des onglets
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Vue d'Ensemble", 
            "🔍 Détails Modèles",
            "📈 Métriques", 
            "🔗 MLflow"
        ])

        with tab1:
            display_overview_tab(validation_result, results_data)

        with tab2:
            st.markdown("### 🔍 Analyse Détaillée par Modèle")
            
            successful_models = []
            for result in results_data:
                if isinstance(result, dict) and result.get('success', False):
                    successful_models.append(result)
            
            if successful_models:
                model_names = [m.get('model_name', f'Modèle_{i}') for i, m in enumerate(successful_models)]
                selected_idx = st.selectbox(
                    "Sélectionnez un modèle à analyser :",
                    range(len(model_names)),
                    format_func=lambda x: model_names[x],
                    key="model_selector_main"
                )
                
                if 0 <= selected_idx < len(successful_models):
                    selected_model = successful_models[selected_idx]
                    display_model_details(visualizer, selected_model, task_type)
            else:
                st.warning("⚠️ Aucun modèle n'a terminé avec succès l'entraînement")

        with tab3:
            st.markdown("### 📈 Analyse des Métriques Détaillées")
            
            successful_models = validation_result.get("successful_models", [])
            if successful_models:
                model_names = [m.get('model_name', f'Modèle_{i}') for i, m in enumerate(successful_models)]
                selected_model_name = st.selectbox(
                    "Sélectionnez un modèle pour voir ses métriques détaillées:",
                    options=model_names,
                    key="metrics_model_selector"
                )
                
                selected_model = next((m for m in successful_models if m.get('model_name') == selected_model_name), None)
                
                if selected_model:
                    metrics = selected_model.get('metrics', {})
                    st.markdown(f"#### 📊 Métriques pour **{selected_model_name}**")
                    
                    if task_type == 'clustering':
                        cols = st.columns(2)
                        with cols[0]:
                            if 'silhouette_score' in metrics:
                                st.metric("Score Silhouette", f"{metrics['silhouette_score']:.3f}")
                            if 'calinski_harabasz_score' in metrics:
                                st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz_score']:.3f}")
                        with cols[1]:
                            if 'davies_bouldin_score' in metrics:
                                st.metric("Davies-Bouldin", f"{metrics['davies_bouldin_score']:.3f}")
                            if 'n_clusters' in metrics:
                                st.metric("Nombre de Clusters", f"{metrics['n_clusters']}")
                    
                    elif task_type == 'regression':
                        cols = st.columns(2)
                        with cols[0]:
                            if 'r2' in metrics:
                                st.metric("R² Score", f"{metrics['r2']:.3f}")
                            if 'mae' in metrics:
                                st.metric("MAE", f"{metrics['mae']:.3f}")
                        with cols[1]:
                            if 'rmse' in metrics:
                                st.metric("RMSE", f"{metrics['rmse']:.3f}")
                    
                    else:
                        cols = st.columns(2)
                        with cols[0]:
                            if 'accuracy' in metrics:
                                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                            if 'precision' in metrics:
                                st.metric("Precision", f"{metrics['precision']:.3f}")
                        with cols[1]:
                            if 'recall' in metrics:
                                st.metric("Recall", f"{metrics['recall']:.3f}")
                            if 'f1_score' in metrics:
                                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")
                    
                    # Métriques brutes
                    with st.expander("🔍 Métriques Brutes (JSON)", expanded=False):
                        st.json(metrics)

        with tab4:
            display_mlflow_tab()

        # Nettoyage mémoire
        gc.collect()

    except Exception as e:
        logger.error(f"Erreur critique dans main(). Erreur : {str(e)}, Traceback : {traceback.format_exc()[:500]}")
        st.error("❌ Erreur critique dans la page d'évaluation")
        
        with st.expander("🔧 Détails Techniques (Debug)", expanded=False):
            st.code(traceback.format_exc())
            
        if st.button("🔄 Redémarrer l'Application", type="primary"):
            st.rerun()

# Point d'entrée
if __name__ == "__main__":
    safe_main()