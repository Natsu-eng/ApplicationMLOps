"""
Page Streamlit: Évaluation Détection d'Anomalies - Premium Dashboard
Version complète avec design moderne et analyse approfondie
"""

import streamlit as st # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import time
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore
import plotly.express as px # type: ignore
from sklearn.metrics import ( # type: ignore    
    confusion_matrix, classification_report, roc_auc_score, 
    f1_score, precision_score, recall_score, accuracy_score,
    roc_curve, precision_recall_curve
)

# Imports métier
try:
    from src.evaluation.anomaly_typing import AnomalyTypeAnalyzer
    from src.config.anomaly_taxonomy import ANOMALY_TAXONOMY
    from src.evaluation.computer_vision_metrics import (
        compute_anomaly_metrics, compute_reconstruction_metrics
    )
    from src.shared.logging import get_logger
    from src.config.constants import ANOMALY_CONFIG
except ImportError:
    # Fallback pour développement
    class AnomalyTypeAnalyzer:
        def compute_metrics_by_anomaly_type(self, y_true, y_pred, types, threshold):
            return {}
        def generate_type_specific_recommendations(self, metrics):
            return []
        def create_performance_heatmap(self, metrics):
            return go.Figure()
        def create_category_summary(self, metrics):
            return pd.DataFrame()
    
    ANOMALY_TAXONOMY = {}
    ANOMALY_CONFIG = {"MLFLOW_ENABLED": False}
    
    def get_logger(name):
        # Utiliser le système de logging centralisé même en fallback
        try:
            from src.shared.logging import get_logger
            return get_logger(name)
        except ImportError:
            import logging
            return logging.getLogger(name)

import torch # type: ignore
from scipy.ndimage import zoom  # type: ignore

from monitoring.state_managers import init, AppPage

# ✅ IMPORTS UI CENTRALISÉS
from ui.anomaly_evaluation_styles import AnomalyEvaluationStyles
from helpers.ui_components.anomaly_evaluation import (
    safe_convert_history,
    analyze_false_positives,
    get_performance_status,
    create_performance_summary,
    generate_recommendations,
    create_performance_radar,
    plot_error_distribution
)
from helpers.anomaly_prediction_helpers import robust_predict_with_preprocessor

STATE = init()
logger = get_logger(__name__)

# ============================================================================
# CONFIGURATION STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Evaluation Dashboard | DataLab Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CSS ULTRA-MODERNE
# ============================================================================

# ✅ Injection CSS centralisé
st.markdown(AnomalyEvaluationStyles.get_css(), unsafe_allow_html=True)


# ============================================================================
# VÉRIFICATIONS INITIALES
# ============================================================================
# Note: Toutes les fonctions métier sont importées depuis les helpers:
# - helpers/ui_components/anomaly_evaluation.py
# - helpers/anomaly_prediction_helpers.py
# ============================================================================

if not hasattr(STATE, 'training_results') or STATE.training_results is None:
    st.error("❌ Aucun modèle entraîné")
    st.info("💡 Veuillez d'abord entraîner un modèle dans la section Computer Vision")
    if st.button("🚀 Aller à l'Entraînement", type="primary"):
        st.switch_page("pages/4_training_computer.py")
    st.stop()

if not isinstance(STATE.training_results, dict):
    st.error("❌ Format invalide des résultats d'entraînement")
    st.info("Type reçu: " + str(type(STATE.training_results)))
    st.stop()

if 'model' not in STATE.training_results:
    st.error("❌ Modèle manquant dans les résultats")
    st.info("Clés disponibles: " + str(list(STATE.training_results.keys())))
    st.stop()

# Récupération données avec fallbacks
try:
    model = STATE.training_results["model"]
    history = safe_convert_history(STATE.training_results.get("history", {}))
    
    # Accès safe à model_config (avec fallback depuis training_results)
    if not hasattr(STATE, 'model_config') or STATE.model_config is None:
        # Fallback: récupérer depuis training_results
        if isinstance(STATE.training_results, dict) and "model_config" in STATE.training_results:
            STATE.model_config = STATE.training_results["model_config"]
            logger.info("✅ model_config récupéré depuis training_results")
        else:
            st.error("❌ Configuration du modèle manquante")
            st.info("💡 Veuillez relancer l'entraînement pour générer la configuration")
            st.stop()
    
    model_type = STATE.model_config.get("model_type", "autoencoder") if isinstance(STATE.model_config, dict) else getattr(STATE.model_config, "model_type", "autoencoder")
    
    # Récupération safe du preprocessor
    preprocessor = STATE.training_results.get("preprocessor")
    
    # Vérification données test avec accès via STATE.data
    if not hasattr(STATE.data, 'X_test') or STATE.data.X_test is None:
        st.error("❌ Données de test (X_test) manquantes")
        st.info("Veuillez relancer l'entraînement pour générer les données de test")
        st.stop()
    
    if not hasattr(STATE.data, 'y_test') or STATE.data.y_test is None:
        st.error("❌ Labels de test (y_test) manquants")
        st.stop()
    
    X_test = STATE.data.X_test
    y_test = STATE.data.y_test
    
    # VALIDATION : Cohérence des données
    if len(X_test) != len(y_test):
        st.error(f"❌ Incohérence: {len(X_test)} images mais {len(y_test)} labels")
        st.stop()
    
    logger.info(f"✅ Données chargées: {len(X_test)} échantillons, modèle: {model_type}")
        
except KeyError as e:
    st.error(f"❌ Clé manquante dans les résultats: {e}")
    st.info("Structure attendue: training_results[model, history, preprocessor]")
    st.stop()
except Exception as e:
    st.error(f"❌ Erreur chargement: {str(e)}")
    logger.error(f"Erreur chargement données: {e}", exc_info=True)
    st.stop()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    threshold = st.slider(
        "**Seuil de Classification**",
        0.0, 1.0, 0.5, 0.01,
        help="Niveau de confiance requis"
    )
    
    if threshold < 0.3:
        st.error("🔻 Seuil Bas - Plus de détection")
    elif threshold > 0.7:
        st.warning("🔺 Seuil Élevé - Plus de précision")
    else:
        st.success("✅ Seuil Optimal")
    
    st.markdown("---")
    
    st.markdown("### 📊 Options")
    show_error_analysis = st.checkbox("Analyse Erreurs", True)
    show_recommendations = st.checkbox("Recommandations", True)
    n_samples_viz = st.slider("Échantillons", 1, 12, 6)
    
    st.markdown("---")
    
    st.markdown("### 🔧 Infos")
    st.metric("Type", model_type)
    st.metric("Échantillons", len(X_test))


# ============================================================================
# INTERFACE PRINCIPALE
# ============================================================================

# Hero Header
st.markdown(f'''
<div class="hero-header">
    <h1 class="hero-title">📊 Dashboard d'Évaluation</h1>
    <p class="hero-subtitle">Analyse approfondie des performances en détection d'anomalies</p>
</div>
''', unsafe_allow_html=True)

# Prédictions
with st.spinner("🔮 Calcul des prédictions..."):
    # ✅ Utilisation du helper centralisé avec STATE pour récupération shapes originales
    prediction_results = robust_predict_with_preprocessor(
        model, X_test, preprocessor, model_type,
        return_localization=True, STATE=STATE
    )
    y_pred_proba = prediction_results["y_pred_proba"]
    y_pred_binary = prediction_results["y_pred_binary"]

# Métriques
with st.spinner("📈 Calcul des métriques..."):
    try:
        # Validation prédictions
        if not isinstance(prediction_results, dict):
            st.error("❌ Format invalide des prédictions")
            st.stop()
        
        y_pred_proba = prediction_results.get("y_pred_proba")
        y_pred_binary = prediction_results.get("y_pred_binary")
        
        if y_pred_proba is None or y_pred_binary is None:
            st.error("❌ Prédictions manquantes")
            st.stop()
        
        # Validation cohérence
        if len(y_pred_binary) != len(y_test):
            st.error(f"❌ Incohérence: {len(y_pred_binary)} prédictions pour {len(y_test)} labels")
            st.stop()
        
        # Calcul métriques avec gestion erreurs individuelles
        metrics = {}
        
        # AUC-ROC (nécessite au moins 2 classes dans y_test)
        try:
            if len(np.unique(y_test)) >= 2:
                metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
            else:
                logger.warning("⚠️ AUC-ROC impossible: une seule classe dans y_test")
                metrics['auc_roc'] = 0.5
        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul AUC-ROC: {e}")
            metrics['auc_roc'] = 0.5
        
        # F1-Score
        try:
            metrics['f1_score'] = f1_score(y_test, y_pred_binary, zero_division=0)
        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul F1: {e}")
            metrics['f1_score'] = 0.0
        
        # Precision
        try:
            metrics['precision'] = precision_score(y_test, y_pred_binary, zero_division=0)
        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul Precision: {e}")
            metrics['precision'] = 0.0
        
        # Recall
        try:
            metrics['recall'] = recall_score(y_test, y_pred_binary, zero_division=0)
        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul Recall: {e}")
            metrics['recall'] = 0.0
        
        # Accuracy
        try:
            metrics['accuracy'] = accuracy_score(y_test, y_pred_binary)
        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul Accuracy: {e}")
            metrics['accuracy'] = 0.0
        
        # Specificity (nécessite au moins une classe 0)
        try:
            if np.any(y_test == 0):
                metrics['specificity'] = recall_score(
                    1 - y_test, 1 - y_pred_binary, zero_division=0
                )
            else:
                logger.warning("⚠️ Specificity impossible: aucune classe 0")
                metrics['specificity'] = 0.0
        except Exception as e:
            logger.warning(f"⚠️ Erreur calcul Specificity: {e}")
            metrics['specificity'] = 0.0
        
        # Métriques spécifiques autoencoder
        if model_type in ["autoencoder", "conv_autoencoder", "variational_autoencoder", "denoising_autoencoder"]:
            reconstructed = prediction_results.get("reconstructed", X_test.copy())
            reconstruction_errors = prediction_results.get('reconstruction_errors')
            
            if reconstruction_errors is not None:
                metrics['reconstruction_error'] = float(np.mean(reconstruction_errors))
                metrics['reconstruction_std'] = float(np.std(reconstruction_errors))
                metrics['adaptive_threshold'] = prediction_results.get('adaptive_threshold', 0.5)
        
        # Matrice de confusion
        try:
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred_binary).tolist()
        except Exception as e:
            logger.warning(f"⚠️ Erreur matrice confusion: {e}")
            metrics['confusion_matrix'] = [[0, 0], [0, 0]]
        
        logger.info(f"✅ Métriques calculées: {list(metrics.keys())}")
        
        # Alerte fallback
        if prediction_results.get("fallback"):
            st.warning("⚠️ **Attention**: Prédictions en mode fallback (aléatoires). Résultats non fiables.")
        
    except Exception as e:
        st.error(f"❌ Erreur calcul métriques: {str(e)}")
        logger.error(f"Erreur métriques: {e}", exc_info=True)
        
        # Métriques par défaut en cas d'échec total
        metrics = {
            'auc_roc': 0.5,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'accuracy': 0.0,
            'specificity': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]]
        }

# Analyse
error_analysis = analyze_false_positives(X_test, y_test, y_pred_binary)
performance_summary = create_performance_summary(metrics, error_analysis)
recommendations = generate_recommendations(metrics, model_type, error_analysis, performance_summary)


# ============================================================================
# SECTION: MÉTRIQUES PRINCIPALES
# ============================================================================

st.markdown("### 📊 Métriques de Performance")

col1, col2, col3, col4 = st.columns(4)

main_metrics = [
    ("AUC-ROC", "auc_roc", "🎯", col1),
    ("F1-Score", "f1_score", "⚡", col2),
    ("Precision", "precision", "🎪", col3),
    ("Recall", "recall", "🔍", col4)
]

for label, metric_key, icon, col in main_metrics:
    with col:
        value = metrics.get(metric_key, 0)
        status, status_text = get_performance_status(value, metric_key)
        
        st.markdown(f'''
        <div class="metric-card-premium">
            <span class="metric-icon">{icon}</span>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value:.3f}</div>
            <span class="status-badge badge-{status}">{status_text}</span>
        </div>
        ''', unsafe_allow_html=True)


# ============================================================================
# SECTION: RÉSUMÉ PERFORMANCE
# ============================================================================

st.markdown("### 🎯 Résumé Exécutif")

col_summary1, col_summary2, col_summary3 = st.columns([2, 1, 1])

with col_summary1:
    overall_score = performance_summary["overall_score"]
    status = performance_summary["status"]
    
    st.markdown(f'''
    <div class="panel-card">
        <div class="panel-header">
            <span class="panel-icon">📈</span>
            <h3 class="panel-title">Performance Globale</h3>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 3rem; font-weight: 800; color: #1f2937; margin: 1rem 0;">
                {overall_score:.1%}
            </div>
            <span class="status-badge badge-{status}" style="font-size: 1rem; padding: 0.75rem 1.5rem;">
                {get_performance_status(overall_score, "auc_roc")[1]}
            </span>
        </div>
        <div class="progress-wrapper" style="margin-top: 1.5rem;">
            <div class="progress-bar progress-{status}" style="width: {overall_score*100}%;"></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

with col_summary2:
    st.markdown(f'''
    <div class="panel-card">
        <div class="panel-header">
            <span class="panel-icon">🚀</span>
            <h3 class="panel-title">Production</h3>
        </div>
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">{"✅" if performance_summary["production_ready"] else "⚠️"}</div>
            <div style="font-weight: 600; color: #6b7280; margin-top: 0.5rem;">
                {"Prêt" if performance_summary["production_ready"] else "Optimisation requise"}
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

with col_summary3:
    risk_colors = {"low": "#10b981", "medium": "#f59e0b", "high": "#ef4444"}
    risk_labels = {"low": "Faible", "medium": "Moyen", "high": "Élevé"}
    risk_level = performance_summary["risk_level"]
    
    st.markdown(f'''
    <div class="panel-card">
        <div class="panel-header">
            <span class="panel-icon">🛡️</span>
            <h3 class="panel-title">Risque</h3>
        </div>
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.5rem; color: {risk_colors[risk_level]};">●</div>
            <div style="font-weight: 600; color: {risk_colors[risk_level]}; margin-top: 0.5rem;">
                {risk_labels[risk_level]}
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


# ============================================================================
# SECTION: RADAR CHART
# ============================================================================

st.markdown("### 🎯 Analyse Multidimensionnelle")

col_radar1, col_radar2 = st.columns([2, 1])

with col_radar1:
    fig_radar = create_performance_radar(metrics)
    st.plotly_chart(fig_radar, use_container_width=True)

with col_radar2:
    st.markdown("#### 📋 Détails")
    
    detail_metrics = {
        "Accuracy": metrics.get('accuracy', 0),
        "Specificity": metrics.get('specificity', 0),
        "Erreur Totale": error_analysis['total_errors']
    }
    
    for label, value in detail_metrics.items():
        if isinstance(value, float):
            st.metric(label, f"{value:.3f}")
        else:
            st.metric(label, value)


# ============================================================================
# ONGLETS PRINCIPAUX
# ============================================================================

tabs = st.tabs([
    "📊 Métriques Détaillées",
    "🔍 Analyse des Erreurs",
    "🎯 Modèle vs Réalité",  # ✅ NOUVEAU: Visualisation comparée
    "💡 Recommandations",
    "🎨 Visualisations",
    "📋 Rapport"
])


# TAB 1: MÉTRIQUES DÉTAILLÉES
with tabs[0]:
    st.markdown("### 📊 Métriques Complètes")
    
    if metrics:
        col_metrics1, col_metrics2 = st.columns(2)
        
        with col_metrics1:
            st.markdown("#### 🎯 Métriques de Classification")
            
            classification_metrics = {
                "AUC-ROC": metrics.get('auc_roc', 0),
                "F1-Score": metrics.get('f1_score', 0),
                "Precision": metrics.get('precision', 0),
                "Recall": metrics.get('recall', 0),
                "Accuracy": metrics.get('accuracy', 0),
                "Specificity": metrics.get('specificity', 0)
            }
            
            for metric, value in classification_metrics.items():
                status, status_text = get_performance_status(value, metric.lower().replace('-', '_'))
                st.markdown(f'''
                <div style="display: flex; justify-content: space-between; align-items: center; 
                            padding: 0.75rem; margin: 0.5rem 0; background: #f9fafb; border-radius: 8px;">
                    <span style="font-weight: 600;">{metric}</span>
                    <div>
                        <span style="font-size: 1.25rem; font-weight: 700; margin-right: 0.5rem;">{value:.3f}</span>
                        <span class="status-badge badge-{status}" style="font-size: 0.7rem; padding: 0.25rem 0.5rem;">
                            {status_text}
                        </span>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        with col_metrics2:
            st.markdown("#### 📈 Métriques d'Erreur")
            
            error_metrics = {
                "Faux Positifs": error_analysis['fp_count'],
                "Faux Négatifs": error_analysis['fn_count'],
                "Taux FP": f"{error_analysis['fp_rate']:.1%}",
                "Taux FN": f"{error_analysis['fn_rate']:.1%}",
                "Erreurs Totales": error_analysis['total_errors'],
                "Précisions Correctes": error_analysis['tp_count'] + error_analysis['tn_count']
            }
            
            for metric, value in error_metrics.items():
                st.markdown(f'''
                <div style="display: flex; justify-content: space-between; align-items: center; 
                            padding: 0.75rem; margin: 0.5rem 0; background: #f9fafb; border-radius: 8px;">
                    <span style="font-weight: 600;">{metric}</span>
                    <span style="font-size: 1.25rem; font-weight: 700;">{value}</span>
                </div>
                ''', unsafe_allow_html=True)
    
    # Matrice de confusion
    if 'confusion_matrix' in metrics:
        st.markdown("---")
        st.markdown("#### 📋 Matrice de Confusion")
        
        cm = np.array(metrics['confusion_matrix'])
        labels = ["Normal", "Anomalie"]
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            showscale=True
        ))
        
        fig_cm.update_layout(
            title="Matrice de Confusion",
            xaxis_title="Prédiction",
            yaxis_title="Réalité",
            height=400
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)


# TAB 2: ANALYSE DES ERREURS
with tabs[1]:
    st.markdown("### 🔍 Analyse Détaillée des Erreurs")
    
    if show_error_analysis:
        # Distribution
        col_err1, col_err2 = st.columns([1, 1])
        
        with col_err1:
            fig_pie = plot_error_distribution(error_analysis)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_err2:
            st.markdown("#### 📊 Statistiques")
            
            st.markdown(f'''
            <div class="error-box error-fp">
                <h4 style="margin: 0 0 0.5rem 0;">❌ Faux Positifs</h4>
                <div style="font-size: 2rem; font-weight: 800;">{error_analysis['fp_count']}</div>
                <div style="opacity: 0.8;">Taux: {error_analysis['fp_rate']:.1%}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(f'''
            <div class="error-box error-fn">
                <h4 style="margin: 0 0 0.5rem 0;">⚠️ Faux Négatifs</h4>
                <div style="font-size: 2rem; font-weight: 800;">{error_analysis['fn_count']}</div>
                <div style="opacity: 0.8;">Taux: {error_analysis['fn_rate']:.1%}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown(f'''
            <div class="error-box error-tp">
                <h4 style="margin: 0 0 0.5rem 0;">✅ Prédictions Correctes</h4>
                <div style="font-size: 2rem; font-weight: 800;">
                    {error_analysis['tp_count'] + error_analysis['tn_count']}
                </div>
                <div style="opacity: 0.8;">
                    {(1 - (error_analysis['total_errors'] / len(y_test))):.1%} de précision
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Exemples d'images
        st.markdown("---")
        st.markdown("#### 🖼️ Exemples d'Erreurs")
        
        if len(error_analysis["false_positives"]) > 0:
            st.markdown("##### ❌ Faux Positifs (Normales classées comme Anomalies)")
            sample_fp = error_analysis["false_positives"][:min(n_samples_viz, len(error_analysis["false_positives"]))]
            
            cols_fp = st.columns(len(sample_fp))
            for i, idx in enumerate(sample_fp):
                with cols_fp[i]:
                    if len(X_test.shape) > 3:
                        # Normaliser l'image pour affichage
                        img = X_test[idx]
                        img_display = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img
                        st.image(img_display, caption=f"FP #{idx}", use_column_width=True)
                    st.caption(f"Confiance: {y_pred_proba[idx]:.3f}")
        
        if len(error_analysis["false_negatives"]) > 0:
            st.markdown("##### ⚠️ Faux Négatifs (Anomalies manquées)")
            sample_fn = error_analysis["false_negatives"][:min(n_samples_viz, len(error_analysis["false_negatives"]))]
            
            cols_fn = st.columns(len(sample_fn))
            for i, idx in enumerate(sample_fn):
                with cols_fn[i]:
                    if len(X_test.shape) > 3:
                        img = X_test[idx]
                        img_display = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img
                        st.image(img_display, caption=f"FN #{idx}", use_column_width=True)
                    st.caption(f"Confiance: {y_pred_proba[idx]:.3f}")


# TAB 3: MODÈLE VS RÉALITÉ
with tabs[2]:
    st.markdown("### 🎯 Visualisation Modèle vs Réalité")
    st.markdown("""
    <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <p style="margin: 0; color: #0369a1;">
            <strong>🔍 Cette section compare ce que le modèle voit avec la réalité:</strong><br>
            • <strong>Prédiction du modèle:</strong> Ce que le modèle a détecté<br>
            • <strong>Label réel:</strong> La vérité terrain<br>
            • <strong>Heatmaps:</strong> Où le modèle localise les anomalies<br>
            • <strong>Type d'erreur:</strong> Classification des types de défauts (si disponible)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Récupération des heatmaps depuis prediction_results
    heatmaps = prediction_results.get("heatmaps")
    error_maps = prediction_results.get("error_maps")
    binary_masks = prediction_results.get("binary_masks")
    
    has_localization = heatmaps is not None and error_maps is not None
    
    if has_localization:
        st.success(f"✅ Localisation disponible: {len(heatmaps)} images avec heatmaps")
    else:
        st.warning("⚠️ Heatmaps non disponibles pour ce modèle")
    
    # Sélection d'échantillons à visualiser
    st.markdown("---")
    st.markdown("#### 🖼️ Échantillons Détaillés")
    
    # Créer des catégories d'échantillons
    tp_indices = error_analysis["true_positives"]
    tn_indices = error_analysis["true_negatives"]
    fp_indices = error_analysis["false_positives"]
    fn_indices = error_analysis["false_negatives"]
    
    sample_categories = {
        "✅ Vrais Positifs (Anomalies détectées correctement)": tp_indices,
        "✅ Vrais Négatifs (Normales détectées correctement)": tn_indices,
        "❌ Faux Positifs (Normales classées comme anomalies)": fp_indices,
        "⚠️ Faux Négatifs (Anomalies manquées)": fn_indices
    }
    
    for category_name, indices in sample_categories.items():
        if len(indices) == 0:
            continue
        
        st.markdown(f"##### {category_name} ({len(indices)} échantillons)")
        
        # Sélectionner jusqu'à n_samples_viz échantillons
        sample_indices = indices[:n_samples_viz]
        
        # Créer une grille de visualisations
        for idx in sample_indices:
            col_viz1, col_viz2 = st.columns([1, 1])
            
            with col_viz1:
                st.markdown("**📷 Image Originale**")
                
                # Affichage image originale
                img = X_test[idx]
                
                # Normalisation pour affichage
                if img.dtype != np.uint8:
                    if img.max() > 1:
                        img_display = img / 255.0
                    else:
                        img_display = img
                    img_display = np.clip(img_display, 0, 1)
                else:
                    img_display = img / 255.0 if img.max() > 1 else img
                
                # Gestion format (channels_first vs channels_last)
                if len(img_display.shape) == 3 and img_display.shape[0] in [1, 3]:
                    # channels_first -> channels_last
                    img_display = np.transpose(img_display, (1, 2, 0))
                elif len(img_display.shape) == 2:
                    img_display = np.stack([img_display, img_display, img_display], axis=-1)
                
                # Conversion grayscale -> RGB si nécessaire
                if img_display.shape[-1] == 1:
                    img_display = np.repeat(img_display, 3, axis=-1)
                
                st.image(img_display, use_container_width=True)
                
                # Informations labels
                label_real = y_test[idx]
                pred_real = y_pred_binary[idx]
                proba = y_pred_proba[idx]
                
                st.markdown(f"""
                <div style="background: #f9fafb; padding: 0.75rem; border-radius: 6px; margin-top: 0.5rem;">
                    <div style="font-size: 0.9rem;">
                        <strong>Label réel:</strong> 
                        <span style="color: {'#10b981' if label_real == 0 else '#ef4444'}; font-weight: 700;">
                            {'✅ Normal' if label_real == 0 else '❌ Anomalie'}
                        </span><br>
                        <strong>Prédiction modèle:</strong> 
                        <span style="color: {'#10b981' if pred_real == 0 else '#ef4444'}; font-weight: 700;">
                            {'✅ Normal' if pred_real == 0 else '❌ Anomalie'}
                        </span><br>
                        <strong>Confiance:</strong> {proba:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_viz2:
                st.markdown("**🔥 Heatmap de Localisation**")
                
                if has_localization and idx < len(heatmaps):
                    try:
                        # Récupérer heatmap pour cet index
                        heatmap = heatmaps[idx]
                        error_map = error_maps[idx]
                        
                        # S'assurer que la heatmap a les bonnes dimensions
                        if len(heatmap.shape) == 2:
                            # Aligner heatmap à l'image si nécessaire
                            img_h, img_w = img_display.shape[:2]
                            if heatmap.shape != (img_h, img_w):
                                zoom_factors = (img_h / heatmap.shape[0], img_w / heatmap.shape[1])
                                heatmap = zoom(heatmap, zoom_factors, order=1)
                                error_map = zoom(error_map, zoom_factors, order=1)
                            
                            # Créer visualisation avec Plotly
                            fig_heatmap = go.Figure()
                            
                            # Image de base
                            img_for_plot = (img_display * 255).astype(np.uint8)
                            fig_heatmap.add_trace(go.Image(z=img_for_plot))
                            
                            # Heatmap superposée
                            fig_heatmap.add_trace(go.Heatmap(
                                z=heatmap,
                                colorscale="Jet",
                                opacity=0.6,
                                showscale=True,
                                colorbar=dict(title="Score anomalie")
                            ))
                            
                            fig_heatmap.update_layout(
                                title=f"Localisation Anomalie (Index {idx})",
                                xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                height=400,
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                            
                            # Informations heatmap
                            max_error = float(error_map.max())
                            mean_error = float(error_map.mean())
                            
                            st.markdown(f"""
                            <div style="background: #f9fafb; padding: 0.75rem; border-radius: 6px; margin-top: 0.5rem;">
                                <div style="font-size: 0.9rem;">
                                    <strong>Erreur max:</strong> {max_error:.4f}<br>
                                    <strong>Erreur moyenne:</strong> {mean_error:.4f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Mask binaire si disponible
                            if binary_masks is not None and idx < len(binary_masks):
                                binary_mask = binary_masks[idx]
                                
                                # Aligner mask si nécessaire
                                if binary_mask.shape != (img_h, img_w):
                                    zoom_factors = (img_h / binary_mask.shape[0], img_w / binary_mask.shape[1])
                                    binary_mask = zoom(binary_mask, zoom_factors, order=0)
                                
                                # Afficher le mask binaire
                                st.markdown("**🎯 Masque Binaire (Région détectée)**")
                                mask_for_display = (binary_mask * 255).astype(np.uint8)
                                st.image(mask_for_display, use_container_width=True, clamp=True)
                        
                        else:
                            st.warning("Format de heatmap non supporté")
                    
                    except Exception as e:
                        logger.error(f"Erreur génération heatmap pour index {idx}: {e}", exc_info=True)
                        st.warning(f"Impossible de générer heatmap: {str(e)}")
                else:
                    st.info("Heatmap non disponible pour cet échantillon")
            
            st.markdown("---")
    
    # Résumé statistique
    st.markdown("---")
    st.markdown("#### 📊 Résumé Statistique")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric("Vrais Positifs", len(tp_indices), 
                 f"{len(tp_indices)/max(len(y_test), 1)*100:.1f}%")
    
    with col_stat2:
        st.metric("Vrais Négatifs", len(tn_indices),
                 f"{len(tn_indices)/max(len(y_test), 1)*100:.1f}%")
    
    with col_stat3:
        st.metric("Faux Positifs", len(fp_indices),
                 f"{len(fp_indices)/max(len(y_test), 1)*100:.1f}%")
    
    with col_stat4:
        st.metric("Faux Négatifs", len(fn_indices),
                 f"{len(fn_indices)/max(len(y_test), 1)*100:.1f}%")


# TAB 4: RECOMMANDATIONS
with tabs[2]:
    st.markdown("### 💡 Recommandations Intelligentes")
    
    if show_recommendations and recommendations:
        # Grouper par priorité
        high_recs = [r for r in recommendations if r["priority"] == "high"]
        medium_recs = [r for r in recommendations if r["priority"] == "medium"]
        low_recs = [r for r in recommendations if r["priority"] == "low"]
        
        if high_recs:
            st.markdown("#### 🔴 Actions Critiques")
            for rec in high_recs:
                st.markdown(f'''
                <div class="recommendation-card rec-priority-high">
                    <div class="rec-title">
                        <span>{rec['icon']}</span>
                        <span>{rec['action']}</span>
                    </div>
                    <p style="margin: 0.5rem 0 0 0; color: #6b7280;">{rec['message']}</p>
                </div>
                ''', unsafe_allow_html=True)
        
        if medium_recs:
            st.markdown("#### 🟡 Améliorations Recommandées")
            for rec in medium_recs:
                st.markdown(f'''
                <div class="recommendation-card rec-priority-medium">
                    <div class="rec-title">
                        <span>{rec['icon']}</span>
                        <span>{rec['action']}</span>
                    </div>
                    <p style="margin: 0.5rem 0 0 0; color: #6b7280;">{rec['message']}</p>
                </div>
                ''', unsafe_allow_html=True)
        
        if low_recs:
            st.markdown("#### 🔵 Optimisations")
            for rec in low_recs:
                st.markdown(f'''
                <div class="recommendation-card">
                    <div class="rec-title">
                        <span>{rec['icon']}</span>
                        <span>{rec['action']}</span>
                    </div>
                    <p style="margin: 0.5rem 0 0 0; color: #6b7280;">{rec['message']}</p>
                </div>
                ''', unsafe_allow_html=True)
    
    # Points forts et faibles
    st.markdown("---")
    st.markdown("### 📈 Analyse SWOT")
    
    col_swot1, col_swot2 = st.columns(2)
    
    with col_swot1:
        st.markdown("#### ✅ Points Forts")
        strengths = performance_summary.get("strengths", [])
        if strengths:
            for strength in strengths:
                st.success(f"✓ {strength}")
        else:
            st.info("Analyse en cours...")
    
    with col_swot2:
        st.markdown("#### ⚠️ Points d'Amélioration")
        weaknesses = performance_summary.get("weaknesses", [])
        if weaknesses:
            for weakness in weaknesses:
                st.warning(f"→ {weakness}")
        else:
            st.success("Aucun point faible détecté!")


# TAB 5: VISUALISATIONS
with tabs[4]:
    st.markdown("### 🎨 Visualisations Avancées")
    
    # Courbes ROC et PR
    st.markdown("#### 📈 Courbes de Performance")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # Courbe ROC
        try:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'ROC (AUC={metrics.get("auc_roc", 0):.3f})',
                line=dict(color='#6366f1', width=3)
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Aléatoire',
                line=dict(color='gray', dash='dash')
            ))
            
            fig_roc.update_layout(
                title="Courbe ROC",
                xaxis_title="Taux Faux Positifs",
                yaxis_title="Taux Vrais Positifs",
                height=400
            )
            
            st.plotly_chart(fig_roc, use_container_width=True)
        except Exception as e:
            st.warning(f"Impossible de générer courbe ROC: {e}")
    
    with col_viz2:
        # Courbe Precision-Recall
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=recall_curve, y=precision_curve,
                mode='lines',
                name='Precision-Recall',
                line=dict(color='#10b981', width=3),
                fill='tonexty'
            ))
            
            fig_pr.update_layout(
                title="Courbe Precision-Recall",
                xaxis_title="Recall",
                yaxis_title="Precision",
                height=400
            )
            
            st.plotly_chart(fig_pr, use_container_width=True)
        except Exception as e:
            st.warning(f"Impossible de générer courbe PR: {e}")
    
    # Distribution des scores
    st.markdown("---")
    st.markdown("#### 📊 Distribution des Scores de Confiance")
    
    fig_hist = go.Figure()
    
    fig_hist.add_trace(go.Histogram(
        x=y_pred_proba[y_test == 0],
        name='Normal',
        marker_color='#3b82f6',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig_hist.add_trace(go.Histogram(
        x=y_pred_proba[y_test == 1],
        name='Anomalie',
        marker_color='#ef4444',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig_hist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Seuil: {threshold}",
        annotation_position="top"
    )
    
    fig_hist.update_layout(
        title="Distribution des Scores par Classe",
        xaxis_title="Score de Confiance",
        yaxis_title="Fréquence",
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)


# TAB 6: RAPPORT
with tabs[5]:
    st.markdown("### 📋 Rapport d'Évaluation")
    
    # Résumé exécutif
    st.markdown("#### 📄 Résumé Exécutif")
    
    st.markdown(f'''
    <div class="panel-card">
        <p><strong>Date:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        <p><strong>Type de Modèle:</strong> {model_type}</p>
        <p><strong>Échantillons Testés:</strong> {len(X_test):,}</p>
        <p><strong>Score Global:</strong> {performance_summary["overall_score"]:.1%}</p>
        <p><strong>Statut Production:</strong> {"✅ Prêt" if performance_summary["production_ready"] else "⚠️ Optimisation requise"}</p>
        <p><strong>Niveau de Risque:</strong> {performance_summary["risk_level"].title()}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Tableau récapitulatif
    st.markdown("#### 📊 Tableau Récapitulatif")
    
    summary_data = {
        "Métrique": list(metrics.keys()),
        "Valeur": [f"{v:.3f}" if isinstance(v, float) else str(v) for v in metrics.values()]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Export
    st.markdown("---")
    st.markdown("#### 💾 Export du Rapport")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("📥 JSON", use_container_width=True):
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "model_type": model_type,
                "threshold": threshold,
                "metrics": {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                           for k, v in metrics.items()},
                "performance_summary": performance_summary,
                "error_analysis": {k: int(v) if isinstance(v, (int, np.integer)) else float(v) if isinstance(v, (float, np.floating)) else v
                                  for k, v in error_analysis.items() if k not in ['false_positives', 'false_negatives', 'true_positives', 'true_negatives']}
            }
            
            json_str = json.dumps(report_data, indent=2, default=str)
            st.download_button(
                "⬇️ Télécharger JSON",
                json_str,
                "evaluation_report.json",
                "application/json",
                use_container_width=True
            )
    
    with col_export2:
        if st.button("📥 CSV", use_container_width=True):
            csv_data = summary_df.to_csv(index=False)
            st.download_button(
                "⬇️ Télécharger CSV",
                csv_data,
                "evaluation_metrics.csv",
                "text/csv",
                use_container_width=True
            )
    
    with col_export3:
        if st.button("📥 Markdown", use_container_width=True):
            md_content = f"""# Rapport d'Évaluation
            
## Résumé
- **Date**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
- **Score Global**: {performance_summary["overall_score"]:.1%}
- **Production Ready**: {"✅ Oui" if performance_summary["production_ready"] else "⚠️ Non"}

## Métriques
{summary_df.to_markdown(index=False)}

## Recommandations
"""
            for rec in recommendations[:5]:
                md_content += f"\n- **{rec['action']}**: {rec['message']}"
            
            st.download_button(
                "⬇️ Télécharger MD",
                md_content,
                "evaluation_report.md",
                "text/markdown",
                use_container_width=True
            )


# ============================================================================
# FOOTER & NAVIGATION
# ============================================================================

st.markdown("---")

col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)

with col_nav1:
    if st.button("🏠 Dashboard", use_container_width=True):
        st.switch_page("pages/1_dashboard.py")

with col_nav2:
    if st.button("🔙 Entraînement", use_container_width=True):
        st.switch_page("pages/4_training_computer.py")

with col_nav3:
    if st.button("🔄 Nouvelle Évaluation", use_container_width=True):
        st.rerun()

with col_nav4:
    if st.button("💾 Sauvegarder Session", type="primary", use_container_width=True):
        try:
            session_data = {
                "metrics": metrics,
                "error_analysis": error_analysis,
                "performance_summary": performance_summary,
                "timestamp": datetime.now().isoformat()
            }
            
            session_file = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            st.success(f"✅ Session sauvegardée: {session_file}")
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")

# Footer info
st.markdown("---")
st.caption(f"🕒 Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')} | DataLab Pro v2.0 Premium")