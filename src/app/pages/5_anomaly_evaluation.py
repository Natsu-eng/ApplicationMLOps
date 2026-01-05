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

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Imports métier
try:
    from src.evaluation.anomaly_typing import AnomalyTypeAnalyzer
    from src.config.anomaly_taxonomy import ANOMALY_TAXONOMY
    from src.evaluation.computer_vision_metrics import (
        compute_anomaly_metrics, compute_reconstruction_metrics
    )
    from src.evaluation.localization_utils import resize_error_map
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
from monitoring.mlflow_collector import get_mlflow_collector

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
    
    # Récupération mode depuis model_config
    model_type = STATE.model_config.get("model_type", "autoencoder") if isinstance(STATE.model_config, dict) else getattr(STATE.model_config, "model_type", "autoencoder")
    is_autoencoder = model_type in ["autoencoder", "conv_autoencoder", "variational_autoencoder", "denoising_autoencoder", "patch_core"]
    
    # Vérification conditionnelle selon type de modèle
    if not hasattr(STATE.data, 'y_test') or STATE.data.y_test is None:
        if is_autoencoder:
            # Mode non supervisé: y_test peut être None, on utilisera reconstruction error
            logger.info("⚠️ y_test manquant, évaluation basée sur reconstruction error uniquement")
            y_test = None  # Évaluation sans labels pour autoencoders
        else:
            # Mode supervisé: y_test obligatoire
            st.error("❌ Labels de test (y_test) manquants (mode supervisé)")
            st.stop()
    else:
        y_test = STATE.data.y_test
    
    X_test = STATE.data.X_test
    
    # VALIDATION : Cohérence des données (seulement si y_test présent)
    if y_test is not None and len(X_test) != len(y_test):
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
        
        # Validation cohérence (seulement si y_test présent)
        if y_test is not None and len(y_pred_binary) != len(y_test):
            st.error(f"❌ Incohérence: {len(y_pred_binary)} prédictions pour {len(y_test)} labels")
            st.stop()
        
        # Calcul métriques avec gestion erreurs individuelles
        metrics = {}
        
        # Métriques de classification (uniquement si y_test présent)
        if y_test is not None:
            # Détection du nombre de classes pour déterminer le mode (binaire vs multiclasse)
            n_classes = len(np.unique(y_test))
            is_multiclass = n_classes > 2
            
            # Détermination de l'average pour les métriques
            if is_multiclass:
                average_mode = 'weighted'  # ou 'macro', 'micro' selon préférence
                logger.info(f"✅ Mode multiclasse détecté ({n_classes} classes) - utilisation average='{average_mode}'")
            else:
                average_mode = 'binary'
                logger.info(f"✅ Mode binaire détecté - utilisation average='{average_mode}'")
            
            try:
                if n_classes >= 2:
                    if is_multiclass:
                        # Récupérer les probabilités par classe depuis prediction_results
                        y_proba_multiclass = prediction_results.get("class_probabilities")
                        
                        if y_proba_multiclass is not None and y_proba_multiclass.shape[1] == n_classes:
                            # Binariser y_test pour one-vs-rest
                            y_test_bin = label_binarize(y_test, classes=range(n_classes))
                            
                            # Calcul AUC ROC one-vs-rest
                            auc_scores = []
                            for i in range(n_classes):
                                if len(np.unique(y_test_bin[:, i])) >= 2:
                                    try:
                                        auc_i = roc_auc_score(y_test_bin[:, i], y_proba_multiclass[:, i])
                                        auc_scores.append(auc_i)
                                    except:
                                        auc_scores.append(0.5)
                                else:
                                    auc_scores.append(0.5)
                            
                            # Moyenne pondérée
                            metrics['auc_roc'] = np.mean(auc_scores)
                            metrics['auc_roc_per_class'] = auc_scores
                            
                            logger.info(f"✅ AUC-ROC multiclasse calculé: {metrics['auc_roc']:.3f}")
                        else:
                            # Fallback si pas de probabilités par classe
                            logger.warning("⚠️ Probabilités multiclasse non disponibles")
                            if len(np.unique(y_test)) >= 2:
                                metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
                            else:
                                metrics['auc_roc'] = 0.5
                    else:
                        # Binaire
                        if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 2:
                            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
                        else:
                            metrics['auc_roc'] = roc_auc_score(y_test, y_pred_proba)
                else:
                    metrics['auc_roc'] = 0.5
            except Exception as e:
                logger.warning(f"⚠️ Erreur calcul AUC-ROC: {e}")
                metrics['auc_roc'] = 0.5
            
            # F1-Score
            try:
                metrics['f1_score'] = f1_score(
                    y_test, y_pred_binary, 
                    average=average_mode, 
                    zero_division=0
                )
            except Exception as e:
                logger.warning(f"⚠️ Erreur calcul F1: {e}")
                metrics['f1_score'] = 0.0
            
            # Precision
            try:
                metrics['precision'] = precision_score(
                    y_test, y_pred_binary, 
                    average=average_mode, 
                    zero_division=0
                )
            except Exception as e:
                logger.warning(f"⚠️ Erreur calcul Precision: {e}")
                metrics['precision'] = 0.0
            
            # Recall
            try:
                metrics['recall'] = recall_score(
                    y_test, y_pred_binary, 
                    average=average_mode, 
                    zero_division=0
                )
            except Exception as e:
                logger.warning(f"⚠️ Erreur calcul Recall: {e}")
                metrics['recall'] = 0.0
            
            # Accuracy
            try:
                metrics['accuracy'] = accuracy_score(y_test, y_pred_binary)
            except Exception as e:
                logger.warning(f"⚠️ Erreur calcul Accuracy: {e}")
                metrics['accuracy'] = 0.0
            
            # Specificity (uniquement pour binaire, ou calcul par classe pour multiclasse)
            try:
                if is_multiclass:
                    # Pour multiclasse: calculer recall par classe puis moyenne
                    # Specificity = recall de la classe négative (classe 0)
                    if np.any(y_test == 0):
                        # Calculer recall pour classe 0 (vrais négatifs)
                        y_test_binary = (y_test == 0).astype(int)
                        y_pred_binary_neg = (y_pred_binary == 0).astype(int)
                        metrics['specificity'] = recall_score(
                            y_test_binary, y_pred_binary_neg, 
                            zero_division=0
                        )
                    else:
                        logger.warning("⚠️ Specificity impossible: aucune classe 0")
                        metrics['specificity'] = 0.0
                else:
                    # Binaire: calcul standard
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
            
            # Matrice de confusion
            try:
                metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred_binary).tolist()
            except Exception as e:
                logger.warning(f"⚠️ Erreur matrice confusion: {e}")
                # Matrice par défaut selon nombre de classes
                default_size = max(n_classes, 2)
                metrics['confusion_matrix'] = [[0] * default_size for _ in range(default_size)]
        else:
            # Mode non supervisé: métriques de classification non disponibles
            logger.info("ℹ️ Mode non supervisé: métriques de classification non calculées")
            metrics['auc_roc'] = None
            metrics['f1_score'] = None
            metrics['precision'] = None
            metrics['recall'] = None
            metrics['accuracy'] = None
            metrics['specificity'] = None
            metrics['confusion_matrix'] = None
        
        # Métriques spécifiques autoencoder
        if model_type in ["autoencoder", "conv_autoencoder", "variational_autoencoder", "denoising_autoencoder"]:
            reconstructed = prediction_results.get("reconstructed", X_test.copy())
            reconstruction_errors = prediction_results.get('reconstruction_errors')
            
            if reconstruction_errors is not None:
                metrics['reconstruction_error'] = float(np.mean(reconstruction_errors))
                metrics['reconstruction_std'] = float(np.std(reconstruction_errors))
                metrics['adaptive_threshold'] = prediction_results.get('adaptive_threshold', 0.5)
        
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

# Analyse (conditionnelle selon présence y_test)
if y_test is not None:
    error_analysis = analyze_false_positives(X_test, y_test, y_pred_binary)
else:
    # Mode non supervisé: analyse basée uniquement sur reconstruction errors
    error_analysis = {
        'total_errors': 0,
        'false_positives': [],
        'false_negatives': [],
        'true_positives': [],
        'true_negatives': []
    }
    logger.info("ℹ️ Mode non supervisé: analyse d'erreurs limitée (pas de labels)")

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
# SECTION: CLASSES DISPONIBLES (NOUVELLE SECTION)
# ============================================================================

# ✅ SECTION AJOUTÉE ICI - Juste après les métriques principales
if 'class_names' in prediction_results:
    st.markdown("### 🏷️ Classes Disponibles")
    
    class_names = prediction_results['class_names']
    is_multiclass = prediction_results.get('is_multiclass', False)
    
    if is_multiclass:
        st.info(f"🔀 **Mode Multiclasse** - {len(class_names)} classes disponibles")
    
    # Afficher les classes sous forme de badges
    badges_html = ""
    for i, class_name in enumerate(class_names):
        # Attribution de couleurs:
        # - Classe 0 (Normal): Vert
        # - Classe 1 (Anomalie): Rouge
        # - Classes 2+: Couleurs alternées du spectre
        color = "#10b981" if i == 0 else ("#ef4444" if i == 1 else f"hsl({(i * 60) % 360}, 70%, 60%)")
        badges_html += f'''
        <span style="background: {color}; color: white; padding: 0.3rem 0.8rem; 
                     border-radius: 20px; margin: 0.2rem; display: inline-block; 
                     font-size: 0.9rem; font-weight: 600;">
            {class_name}
        </span>'''
    
    st.markdown(f'<div style="margin: 1rem 0;">{badges_html}</div>', unsafe_allow_html=True)
    
    # Statistiques par classe si multiclasse et labels disponibles
    if is_multiclass and y_test is not None:
        st.markdown("#### 📊 Distribution par Classe")
        
        col_dist1, col_dist2, col_dist3 = st.columns(3)
        
        with col_dist1:
            # Distribution réelle
            unique, counts = np.unique(y_test, return_counts=True)
            dist_df = pd.DataFrame({
                "Classe": [class_names[i] if i < len(class_names) else f"Classe {i}" for i in unique],
                "Échantillons": counts,
                "Pourcentage": [c/len(y_test)*100 for c in counts]
            })
            st.dataframe(dist_df, hide_index=True, width='stretch')
        
        with col_dist2:
            # Distribution prédite
            unique_pred, counts_pred = np.unique(y_pred_binary, return_counts=True)
            dist_pred_df = pd.DataFrame({
                "Classe": [class_names[i] if i < len(class_names) else f"Classe {i}" for i in unique_pred],
                "Prédictions": counts_pred,
                "Pourcentage": [c/len(y_pred_binary)*100 for c in counts_pred]
            })
            st.dataframe(dist_pred_df, hide_index=True, width='stretch')
        
        with col_dist3:
            # Top 3 classes les plus prédites
            if len(unique_pred) > 0:
                most_common_idx = unique_pred[np.argmax(counts_pred)]
                most_common_name = class_names[most_common_idx] if most_common_idx < len(class_names) else f"Classe {most_common_idx}"
                st.metric("Classe la plus fréquente", most_common_name, f"{max(counts_pred)/len(y_pred_binary)*100:.1f}%")


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
    st.plotly_chart(fig_radar, width='stretch')

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
    "🎯 Modèle vs Réalité",  
    "💡 Recommandations",
    "🎨 Visualisations",
    "🔗 MLflow",  
    "📋 Rapport",
    "🔥 Grad-CAM"
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
        
        st.plotly_chart(fig_cm, width='stretch')


# TAB 2: ANALYSE DES ERREURS
with tabs[1]:
    st.markdown("### 🔍 Analyse Détaillée des Erreurs")
    
    if show_error_analysis:
        # Distribution
        col_err1, col_err2 = st.columns([1, 1])
        
        with col_err1:
            fig_pie = plot_error_distribution(error_analysis)
            st.plotly_chart(fig_pie, width='stretch')
        
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
                    {(1 - (error_analysis['total_errors'] / max(len(y_test) if y_test is not None else len(X_test), 1))):.1%} de précision
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
                        st.image(img_display, caption=f"FP #{idx}", width='stretch')
                    st.caption(f"Confiance: {y_pred_proba[idx]:.3f}" if y_pred_proba[idx] is not None else "Confiance: N/A")
        
        if len(error_analysis["false_negatives"]) > 0:
            st.markdown("##### ⚠️ Faux Négatifs (Anomalies manquées)")
            sample_fn = error_analysis["false_negatives"][:min(n_samples_viz, len(error_analysis["false_negatives"]))]
            
            cols_fn = st.columns(len(sample_fn))
            for i, idx in enumerate(sample_fn):
                with cols_fn[i]:
                    if len(X_test.shape) > 3:
                        img = X_test[idx]
                        img_display = (img - img.min()) / (img.max() - img.min()) if img.max() > img.min() else img
                        st.image(img_display, caption=f"FN #{idx}", width='stretch')
                    st.caption(f"Confiance: {y_pred_proba[idx]:.3f}" if y_pred_proba[idx] is not None else "Confiance: N/A")


# TAB 3: MODÈLE VS RÉALITÉ
with tabs[2]:
    st.markdown("### 🎯 Visualisation Modèle vs Réalité")
    
    # Récupération des noms de classes depuis les résultats
    class_names = prediction_results.get("class_names", ["Normal", "Anomalie"])
    is_multiclass = prediction_results.get("is_multiclass", False)
    
    st.markdown(f"""
    <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
        <p style="margin: 0; color: #0369a1;">
            <strong>🔍 Cette section compare ce que le modèle voit avec la réalité:</strong><br>
            • <strong>Prédiction du modèle:</strong> {prediction_results.get('y_pred_class_names', ['N/A'])[0] if len(prediction_results.get('y_pred_class_names', [])) > 0 else 'N/A'}<br>
            • <strong>Label réel:</strong> {"Disponible" if y_test is not None else "Non disponible"}<br>
            • <strong>Heatmaps:</strong> {"✅ Disponibles" if prediction_results.get("heatmaps") is not None else "❌ Non disponibles"}<br>
            • <strong>Mode:</strong> {"🔀 Multiclasse" if is_multiclass else "⚪ Binaire"}<br>
            • <strong>Classes:</strong> {', '.join(class_names) if class_names else 'Non spécifié'}
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
    
    # Créer des catégories d'échantillons (si y_test disponible)
    if y_test is not None:
        tp_indices = error_analysis["true_positives"]
        tn_indices = error_analysis["true_negatives"]
        fp_indices = error_analysis["false_positives"]
        fn_indices = error_analysis["false_negatives"]
        
        # Noms de classes réels si disponibles
        y_test_class_names = None
        if class_names is not None and y_test is not None:
            # S'assurer que les indices sont dans les limites
            valid_indices = [i for i in y_test if i < len(class_names)]
            if len(valid_indices) == len(y_test):
                y_test_class_names = [class_names[i] for i in y_test]
        
        sample_categories = {
            "✅ Prédictions Correctes (Positives)": tp_indices,
            "✅ Prédictions Correctes (Négatives)": tn_indices,
            "❌ Faux Positifs": fp_indices,
            "⚠️ Faux Négatifs": fn_indices
        }
    else:
        # Mode non supervisé: afficher toutes les prédictions
        all_indices = list(range(min(n_samples_viz * 2, len(X_test))))
        sample_categories = {
            "📊 Toutes les Prédictions": all_indices
        }
    
    # Récupération des noms de classes prédits
    y_pred_class_names = prediction_results.get("y_pred_class_names", [])
    
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
                
                st.image(img_display, width='stretch')
                
                # Informations labels
                if y_test is not None:
                    label_real = y_test[idx] if idx < len(y_test) else None
                    if y_test_class_names and idx < len(y_test_class_names):
                        label_real_name = y_test_class_names[idx]
                    else:
                        label_real_name = f"Classe {label_real}" if label_real is not None else "Inconnu"
                else:
                    label_real = None
                    label_real_name = "Non disponible"
                
                pred_real = y_pred_binary[idx] if idx < len(y_pred_binary) else None
                proba = y_pred_proba[idx] if idx < len(y_pred_proba) else None
                
                if idx < len(y_pred_class_names):
                    pred_name = y_pred_class_names[idx]
                else:
                    pred_name = f"Classe {pred_real}" if pred_real is not None else "Inconnu"
                
                # Déterminer la couleur selon si la prédiction est correcte
                if y_test is not None and label_real is not None and pred_real is not None:
                    is_correct = (label_real == pred_real) if not is_multiclass else (label_real == pred_real)
                    status_color = "#10b981" if is_correct else "#ef4444"
                    status_icon = "✅" if is_correct else "❌"
                else:
                    status_color = "#6b7280"
                    status_icon = "🔍"
                
                st.markdown(f"""
                <div style="background: #f9fafb; padding: 0.75rem; border-radius: 6px; margin-top: 0.5rem;">
                    <div style="font-size: 0.9rem;">
                        <strong>{status_icon} Statut:</strong> 
                        <span style="color: {status_color}; font-weight: 700;">
                            {"Correct" if y_test is not None and label_real is not None and pred_real is not None and label_real == pred_real else "Vérification requise"}
                        </span><br>
                        <strong>Label réel:</strong> 
                        <span style="color: #3b82f6; font-weight: 700;">{label_real_name}</span><br>
                        <strong>Prédiction modèle:</strong> 
                        <span style="color: #ef4444; font-weight: 700;">{pred_name}</span><br>
                        <strong>Confiance:</strong> {f"{proba:.3f}" if proba is not None else "N/A"}
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
                                logger.debug(
                                    f"🔧 Resize heatmap: {heatmap.shape} → ({img_h}, {img_w})"
                                )
                                
                                # Utilisation de la fonction centralisée
                                from src.evaluation.localization_utils import resize_error_map
                                heatmap = resize_error_map(
                                    heatmap,
                                    target_size=(img_h, img_w),
                                    method="bilinear"
                                )
                                
                                error_map = resize_error_map(
                                    error_map,
                                    target_size=(img_h, img_w),
                                    method="bilinear"
                                )
                            
                            # Crée la visualisation avec Plotly
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
                            
                            # Titre avec information de classe
                            title_text = f"Index {idx}"
                            if idx < len(y_pred_class_names):
                                title_text += f" | {y_pred_class_names[idx]}"
                            
                            fig_heatmap.update_layout(
                                title=title_text,
                                xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                height=400,
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            
                            st.plotly_chart(fig_heatmap, width='stretch')
                            
                            # Informations heatmap
                            max_error = float(error_map.max()) if error_map is not None else 0
                            mean_error = float(error_map.mean()) if error_map is not None else 0
                            
                            st.markdown(f"""
                            <div style="background: #f9fafb; padding: 0.75rem; border-radius: 6px; margin-top: 0.5rem;">
                                <div style="font-size: 0.9rem;">
                                    <strong>Erreur max:</strong> {max_error:.4f}<br>
                                    <strong>Erreur moyenne:</strong> {mean_error:.4f}<br>
                                    <strong>Type d'anomalie:</strong> {pred_name}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Mask binaire si disponible
                            if binary_masks is not None and idx < len(binary_masks):
                                binary_mask = binary_masks[idx]
                                
                                # Resize mask avec fonction centralisée
                                if binary_mask.shape != (img_h, img_w):
                                    from src.evaluation.localization_utils import resize_error_map
                                    binary_mask = resize_error_map(
                                        binary_mask,
                                        target_size=(img_h, img_w),
                                        method="nearest"  # Nearest pour masks binaires
                                    )
                                
                                # Afficher le mask binaire
                                st.markdown("**🎯 Masque Binaire (Région détectée)**")
                                mask_for_display = (binary_mask * 255).astype(np.uint8)
                                st.image(mask_for_display, width='stretch', clamp=True)
                                
                        else:
                            st.warning("Format de heatmap non supporté")
                    
                    except Exception as e:
                        logger.error(f"Erreur génération heatmap pour index {idx}: {e}", exc_info=True)
                        st.warning(f"Impossible de générer heatmap: {str(e)}")
                else:
                    st.info("Heatmap non disponible pour cet échantillon")
            
            st.markdown("---")
    
    # Résumé statistique (si y_test disponible)
    if y_test is not None:
        st.markdown("---")
        st.markdown("#### 📊 Résumé Statistique")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        n_total = len(y_test)
        
        with col_stat1:
            st.metric("Vrais Positifs", len(tp_indices), 
                     f"{len(tp_indices)/max(n_total, 1)*100:.1f}%")
        
        with col_stat2:
            st.metric("Vrais Négatifs", len(tn_indices),
                     f"{len(tn_indices)/max(n_total, 1)*100:.1f}%")
        
        with col_stat3:
            st.metric("Faux Positifs", len(fp_indices),
                     f"{len(fp_indices)/max(n_total, 1)*100:.1f}%")
        
        with col_stat4:
            st.metric("Faux Négatifs", len(fn_indices),
                     f"{len(fn_indices)/max(n_total, 1)*100:.1f}%")
        
        # Matrice de confusion détaillée pour multiclasse
        if is_multiclass and class_names:
            st.markdown("---")
            st.markdown("#### 🎯 Matrice de Confusion Détail par Classe")
            
            from sklearn.metrics import confusion_matrix
            
            try:
                cm = confusion_matrix(y_test, y_pred_binary)
                
                # Créer une figure avec les noms de classes
                fig_cm_detailed = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=class_names,
                    y=class_names,
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 12},
                    showscale=True
                ))
                
                fig_cm_detailed.update_layout(
                    title="Matrice de Confusion Détail par Classe",
                    xaxis_title="Prédiction",
                    yaxis_title="Réalité",
                    height=500
                )
                
                st.plotly_chart(fig_cm_detailed, width='stretch')
                
            except Exception as e:
                st.warning(f"Impossible d'afficher la matrice de confusion détaillée: {e}")


# TAB 4: RECOMMANDATIONS
with tabs[3]:
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
    
    # Dans TAB 5: VISUALISATIONS, modifier la section ROC pour multiclasse
    with col_viz1:
        # Courbe ROC (uniquement si y_test présent)
        if y_test is not None:
            try:              
                n_classes = len(np.unique(y_test))
                
                if n_classes == 2:
                    # Binaire
                    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC (AUC={roc_auc:.3f})',
                        line=dict(color='#6366f1', width=3)
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Aléatoire',
                        line=dict(color='gray', dash='dash')
                    ))
                    
                    fig_roc.update_layout(
                        title="Courbe ROC (Binaire)",
                        xaxis_title="Taux Faux Positifs",
                        yaxis_title="Taux Vrais Positifs",
                        height=400
                    )
                    
                else:
                    # Multiclasse: courbes ROC one-vs-rest
                    # Binariser les labels
                    y_test_bin = label_binarize(y_test, classes=range(n_classes))
                    
                    # Récupérer les probabilités par classe
                    y_proba = prediction_results.get("class_probabilities")
                    
                    if y_proba is not None and y_proba.shape[1] == n_classes:
                        # Calculer ROC pour chaque classe
                        fig_roc = go.Figure()
                        
                        for i in range(n_classes):
                            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                            roc_auc = auc(fpr, tpr)
                            
                            class_name = class_names[i] if i < len(class_names) else f"Classe {i}"
                            
                            fig_roc.add_trace(go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name=f'{class_name} (AUC={roc_auc:.3f})',
                                line=dict(width=2)
                            ))
                        
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Aléatoire',
                            line=dict(color='gray', dash='dash')
                        ))
                        
                        fig_roc.update_layout(
                            title="Courbes ROC Multiclasse (One-vs-Rest)",
                            xaxis_title="Taux Faux Positifs",
                            yaxis_title="Taux Vrais Positifs",
                            height=400
                        )
                    else:
                        st.warning("Probabilités par classe non disponibles pour multiclasse")
                        fig_roc = go.Figure()
                
                if fig_roc.data:
                    st.plotly_chart(fig_roc, width='stretch')
                    
            except Exception as e:
                st.warning(f"Impossible de générer courbe ROC: {e}")
        else:
            st.info("ℹ️ Courbe ROC non disponible en mode non supervisé (pas de labels)")
    
    with col_viz2:
        # Courbe Precision-Recall (uniquement si y_test présent)
        if y_test is not None:
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
                
                st.plotly_chart(fig_pr, width='stretch')
            except Exception as e:
                st.warning(f"Impossible de générer courbe PR: {e}")
        else:
            st.info("ℹ️ Courbe Precision-Recall non disponible en mode non supervisé (pas de labels)")
    
    # Distribution des scores
    st.markdown("---")
    st.markdown("#### 📊 Distribution des Scores de Confiance")
    
    fig_hist = go.Figure()
    
    if y_test is not None:
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
    else:
        # Mode non supervisé: distribution globale
        fig_hist.add_trace(go.Histogram(
            x=y_pred_proba,
            name='Scores de Reconstruction',
            marker_color='#6366f1',
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
    
    st.plotly_chart(fig_hist, width='stretch')


# TAB 6: MLFLOW
with tabs[5]:
    st.markdown("### 🔗 Exploration des Runs MLflow")
    
    # Récupération des runs depuis le collecteur
    try:
        collector = get_mlflow_collector()
        mlflow_runs = collector.get_runs(limit=50, sort_by='collected_at', reverse=True)
    except Exception as e:
        logger.warning(f"Erreur récupération runs MLflow: {e}")
        mlflow_runs = []
    
    if not mlflow_runs:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #6c757d;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📭</div>
            <h3 style="color: #495057;">Aucun Run MLflow Disponible</h3>
            <p>Lancez un entraînement depuis la page <strong>Training Computer Vision</strong> pour générer des runs MLflow</p>
            <p style="margin-top: 1rem; font-size: 0.9rem; color: #868e96;">
                Les runs MLflow contiennent toutes les informations sur l'entraînement :<br>
                métriques, paramètres, configuration, et historique complet
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"**📊 {len(mlflow_runs)} runs MLflow disponibles**")
        
        # Filtres
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filtre par statut
            status_options = sorted(list(set(run.get('status', 'UNKNOWN') for run in mlflow_runs)))
            selected_status = st.multiselect(
                "Filtrer par statut",
                options=status_options,
                default=status_options,
                help="Sélectionnez les statuts à afficher"
            )
        
        with col2:
            # Filtre par modèle
            model_options = sorted(list(set(run.get('model_name', 'Unknown') for run in mlflow_runs)))
            selected_models = st.multiselect(
                "Filtrer par modèle",
                options=model_options,
                default=model_options,
                help="Sélectionnez les modèles à afficher"
            )
        
        with col3:
            # Filtre par type (Computer Vision spécifique)
            cv_types = []
            for run in mlflow_runs:
                model_name = run.get('model_name', '').lower()
                if any(x in model_name for x in ['autoencoder', 'vae', 'patch', 'siamese']):
                    cv_types.append('Anomaly Detection')
                elif any(x in model_name for x in ['cnn', 'resnet', 'vgg', 'efficientnet']):
                    cv_types.append('Classification')
                else:
                    cv_types.append('Autre')
            
            type_options = sorted(list(set(cv_types)))
            selected_types = st.multiselect(
                "Filtrer par type",
                options=type_options,
                default=type_options,
                help="Type de modèle Computer Vision"
            )
        
        # Application des filtres
        filtered_runs = []
        for i, run in enumerate(mlflow_runs):
            if not isinstance(run, dict):
                continue
            
            status = run.get('status', 'UNKNOWN')
            model_name = run.get('model_name', 'Unknown')
            
            # Filtre statut
            if status not in selected_status:
                continue
            
            # Filtre modèle
            if model_name not in selected_models:
                continue
            
            # Filtre type
            model_name_lower = model_name.lower()
            run_type = 'Autre'
            if any(x in model_name_lower for x in ['autoencoder', 'vae', 'patch', 'siamese']):
                run_type = 'Anomaly Detection'
            elif any(x in model_name_lower for x in ['cnn', 'resnet', 'vgg', 'efficientnet']):
                run_type = 'Classification'
            
            if run_type not in selected_types:
                continue
            
            filtered_runs.append((i, run))
        
        st.info(f"**{len(filtered_runs)}** runs correspondant aux filtres")
        
        # Affichage des runs filtrés
        if filtered_runs:
            for idx, (original_idx, run) in enumerate(filtered_runs[:20]):  # Limiter à 20 runs
                run_id = run.get('run_id', 'N/A')
                model_name = run.get('model_name', 'Unknown')
                status = run.get('status', 'UNKNOWN')
                metrics = run.get('metrics', {})
                params = run.get('params', {})
                tags = run.get('tags', {})
                
                # Détermination du type
                model_name_lower = model_name.lower()
                if any(x in model_name_lower for x in ['autoencoder', 'vae', 'patch', 'siamese']):
                    run_type_badge = "🔍 Anomaly Detection"
                    run_type_color = "#f5576c"
                elif any(x in model_name_lower for x in ['cnn', 'resnet', 'vgg', 'efficientnet']):
                    run_type_badge = "🎯 Classification"
                    run_type_color = "#4facfe"
                else:
                    run_type_badge = "🤖 Autre"
                    run_type_color = "#6c757d"
                
                # Statut coloré
                status_colors = {
                    'FINISHED': '#10b981',
                    'RUNNING': '#3b82f6',
                    'FAILED': '#ef4444',
                    'KILLED': '#f59e0b'
                }
                status_color = status_colors.get(status, '#6c757d')
                
                with st.expander(
                    f"{run_type_badge} | {model_name} | {status}",
                    expanded=(idx == 0)
                ):
                    # En-tête avec informations principales
                    col_header1, col_header2, col_header3 = st.columns([2, 1, 1])
                    
                    with col_header1:
                        st.markdown(f"""
                        <div style="padding: 0.5rem; background: #f8f9fa; border-radius: 5px;">
                            <strong>Run ID:</strong> <code>{run_id[:16]}...</code><br>
                            <strong>Type:</strong> <span style="color: {run_type_color};">{run_type_badge}</span><br>
                            <strong>Statut:</strong> <span style="color: {status_color};">{status}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_header2:
                        if 'collected_at' in run:
                            collected_at = run['collected_at']
                            if isinstance(collected_at, str):
                                try:
                                    dt = datetime.fromisoformat(collected_at.replace('Z', '+00:00'))
                                    st.write(f"**Date:** {dt.strftime('%d/%m/%Y %H:%M')}")
                                except:
                                    st.write(f"**Date:** {collected_at[:10]}")
                    
                    with col_header3:
                        if 'training_time' in metrics:
                            st.write(f"**Durée:** {metrics['training_time']:.1f}s")
                        elif 'training_time' in params:
                            st.write(f"**Durée:** {params['training_time']}s")
                    
                    st.markdown("---")
                    
                    # Métriques principales
                    col_metrics1, col_metrics2 = st.columns(2)
                    
                    with col_metrics1:
                        st.markdown("#### 📈 Métriques d'Entraînement")
                        
                        # Métriques spécifiques Computer Vision
                        cv_metrics = {}
                        standard_metrics = {}
                        
                        for key, value in metrics.items():
                            if any(x in key.lower() for x in ['loss', 'accuracy', 'f1', 'precision', 'recall', 'auc']):
                                standard_metrics[key] = value
                            elif any(x in key.lower() for x in ['reconstruction', 'error', 'threshold']):
                                cv_metrics[key] = value
                            else:
                                standard_metrics[key] = value
                        
                        # Affichage métriques standard
                        if standard_metrics:
                            for metric, value in list(standard_metrics.items())[:5]:
                                if isinstance(value, (int, float)):
                                    st.metric(metric.replace('_', ' ').title(), f"{value:.4f}")
                                else:
                                    st.write(f"**{metric.replace('_', ' ').title()}:** {value}")
                        
                        # Métriques spécifiques CV
                        if cv_metrics:
                            st.markdown("**🔍 Métriques Anomaly Detection:**")
                            for metric, value in list(cv_metrics.items())[:3]:
                                if isinstance(value, (int, float)):
                                    st.write(f"- {metric.replace('_', ' ').title()}: `{value:.4f}`")
                    
                    with col_metrics2:
                        st.markdown("#### ⚙️ Paramètres d'Entraînement")
                        
                        # Paramètres importants
                        important_params = [
                            'epochs', 'batch_size', 'learning_rate', 'optimizer',
                            'scheduler', 'model_type', 'num_classes'
                        ]
                        
                        displayed_params = {}
                        for param in important_params:
                            if param in params:
                                displayed_params[param] = params[param]
                        
                        # Afficher les paramètres importants
                        for param, value in displayed_params.items():
                            st.write(f"**{param.replace('_', ' ').title()}:** `{value}`")
                        
                        # Afficher autres paramètres (limité)
                        other_params = {k: v for k, v in params.items() if k not in important_params}
                        if other_params:
                            with st.expander(f"Autres paramètres ({len(other_params)})"):
                                for param, value in list(other_params.items())[:10]:
                                    st.write(f"- **{param}:** `{value}`")
                    
                    st.markdown("---")
                    
                    # Tags et métadonnées
                    if tags:
                        st.markdown("#### 🏷️ Tags et Métadonnées")
                        col_tags1, col_tags2 = st.columns(2)
                        
                        with col_tags1:
                            for key, value in list(tags.items())[:5]:
                                st.write(f"**{key}:** `{value}`")
                        
                        with col_tags2:
                            if len(tags) > 5:
                                with st.expander(f"Tous les tags ({len(tags)})"):
                                    for key, value in tags.items():
                                        st.write(f"- **{key}:** `{value}`")
                    
                    # Boutons d'action
                    col_action1, col_action2, col_action3 = st.columns(3)
                    
                    with col_action1:
                        if st.button("📖 Voir Détails Complets", key=f"details_{run_id}", width='stretch'):
                            st.json({
                                "run_id": run_id,
                                "model_name": model_name,
                                "status": status,
                                "metrics": metrics,
                                "params": params,
                                "tags": tags
                            })
                    
                    with col_action2:
                        # Lien vers MLflow UI si disponible
                        try:
                            import mlflow
                            tracking_uri = mlflow.get_tracking_uri()
                            if tracking_uri and tracking_uri != "file://":
                                mlflow_url = f"{tracking_uri}/#/experiments/{run.get('experiment_id', '0')}/runs/{run_id}"
                                st.markdown(f"[🔗 Ouvrir dans MLflow UI]({mlflow_url})")
                        except:
                            pass
                    
                    with col_action3:
                        # Export JSON
                        export_data = {
                            "run_id": run_id,
                            "model_name": model_name,
                            "status": status,
                            "metrics": metrics,
                            "params": params,
                            "tags": tags,
                            "collected_at": run.get('collected_at', '')
                        }
                        json_str = json.dumps(export_data, indent=2, ensure_ascii=False, default=str)
                        st.download_button(
                            "💾 Export JSON",
                            json_str,
                            f"mlflow_run_{run_id[:8]}_{datetime.now().strftime('%Y%m%d')}.json",
                            "application/json",
                            key=f"export_{run_id}",
                            width='stretch'
                        )
        else:
            st.warning("⚠️ Aucun run ne correspond aux filtres sélectionnés")


# TAB 7: RAPPORT
with tabs[6]:
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
    st.dataframe(summary_df, width='stretch')
    
    # Export
    st.markdown("---")
    st.markdown("#### 💾 Export du Rapport")
    
    col_export1, col_export2, col_export3 = st.columns(3)
    
    with col_export1:
        if st.button("📥 JSON", width='stretch'):
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
                width='stretch'
            )
    
    with col_export2:
        if st.button("📥 CSV", width='stretch'):
            csv_data = summary_df.to_csv(index=False)
            st.download_button(
                "⬇️ Télécharger CSV",
                csv_data,
                "evaluation_metrics.csv",
                "text/csv",
                width='stretch'
            )
    
    with col_export3:
        if st.button("📥 Markdown", width='stretch'):
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
                width='stretch'
            )

# ============================================================================
# TAB 7: GRAD-CAM (index 7 car c'est le 8ème onglet)
# ============================================================================
with tabs[7]:
    st.markdown("### 🔥 Grad-CAM - Visualisation des Zones d'Attention")
    
    # Vérifier compatibilité
    model_type_lower = model_type.lower()
    compatible_models = [
        'transfer_learning', 'resnet', 'vgg', 'efficientnet',
        'mobilenet', 'densenet', 'simple_cnn', 'custom_resnet', 'cnn'
    ]
    
    is_compatible = any(model in model_type_lower for model in compatible_models)
    
    if not is_compatible:
        st.info(f"""
        ℹ️ **Grad-CAM non disponible pour ce type de modèle**
        
        **Modèle actuel:** {model_type}
        
        Grad-CAM nécessite un modèle convolutionnel (CNN).
        
        **Modèles compatibles:**
        - Transfer Learning (ResNet, VGG, EfficientNet, MobileNet, DenseNet)
        - CNNs custom
        - Classification multiclasse/binaire avec CNN
        
        **Pour utiliser Grad-CAM:**
        1. Entraînez un modèle CNN
        2. Revenez sur cette page
        """)
        
        # Afficher quand même l'interface mais désactivée
        st.markdown("---")
        st.warning("⚠️ Interface Grad-CAM désactivée (modèle non compatible)")
        
        # Afficher un aperçu de ce que Grad-CAM fait
        with st.expander("👀 Aperçu de ce que Grad-CAM ferait avec un modèle compatible"):
            st.markdown("""
            ### 🔬 Visualisation des zones d'attention
            
            Avec un modèle compatible, vous verriez:
            
            **🎨 Modes disponibles:**
            - **Overlay**: Heatmap superposée à l'image
            - **Side-by-side**: Image | Heatmap | Overlay
            - **Heatmap**: Carte de chaleur seule
            
            **🖼️ Sélection d'images:**
            - Aléatoire
            - Premières N
            - Erreurs uniquement (mal classées)
            - Indices spécifiques
            
            **📊 Fonctionnalités:**
            - Export PNG/NPZ
            - Statistiques détaillées
            - Visualisation batch
            """)
        
        st.stop()
    
    # ====================================================================
    # VOTRE CODE GRAD-CAM COMPLET ICI
    # ====================================================================
    
    st.markdown("""
    <div class="info-box">
    <strong>🔬 Qu'est-ce que Grad-CAM ?</strong><br>
    Grad-CAM (Gradient-weighted Class Activation Mapping) visualise les régions 
    de l'image qui ont le plus influencé la prédiction du modèle.
    
    - 🔴 **Rouge/Jaune**: Régions importantes (forte activation)
    - 🔵 **Bleu/Violet**: Régions peu importantes (faible activation)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ====================================================================
    # CONFIGURATION
    # ====================================================================
    st.markdown("#### ⚙️ Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Mode de visualisation
        mode = st.selectbox(
            "🎨 Mode de visualisation",
            options=["overlay", "side_by_side", "heatmap"],
            index=0,
            help=(
                "**overlay**: Superposition heatmap sur image\n\n"
                "**side_by_side**: Image | Heatmap | Overlay\n\n"
                "**heatmap**: Heatmap seule"
            )
        )
    
    with col2:
        # Transparence
        alpha = st.slider(
            "🎚️ Transparence heatmap",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = image visible, 1 = heatmap visible"
        )
    
    with col3:
        # Nombre d'images
        n_samples = st.number_input(
            "📊 Nombre d'images",
            min_value=1,
            max_value=min(20, len(X_test)),
            value=min(6, len(X_test)),
            step=1
        )
    
    # ====================================================================
    # SÉLECTION D'IMAGES
    # ====================================================================
    st.markdown("---")
    st.markdown("#### 🖼️ Sélection des images")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        selection_mode = st.radio(
            "Mode de sélection",
            options=["Aléatoire", "Premières N", "Erreurs uniquement", "Indices spécifiques"],
            index=0,
            help=(
                "**Aléatoire**: Échantillon aléatoire\n\n"
                "**Premières N**: Premières images du dataset\n\n"
                "**Erreurs uniquement**: Images mal classées\n\n"
                "**Indices spécifiques**: Choisir manuellement"
            )
        )
    
    with col2:
        if selection_mode == "Indices spécifiques":
            indices_str = st.text_input(
                "Indices (séparés par des virgules)",
                value="0,1,2,3,4,5",
                help="Ex: 0,5,10,15,20"
            )
    
    # ====================================================================
    # GÉNÉRATION
    # ====================================================================
    st.markdown("---")
    
    if st.button("🔥 Générer Grad-CAM", type="primary", use_container_width=True):
        with st.spinner("⏳ Génération des visualisations Grad-CAM..."):
            try:
                # Sélection des indices
                if selection_mode == "Aléatoire":
                    indices = np.random.choice(len(X_test), size=n_samples, replace=False)
                
                elif selection_mode == "Premières N":
                    indices = np.arange(n_samples)
                
                elif selection_mode == "Erreurs uniquement":
                    # Utiliser prediction_results au lieu de pred_result
                    if prediction_results and prediction_results.get('y_pred_binary') is not None and y_test is not None:
                        y_pred = prediction_results['y_pred_binary']
                        error_indices = np.where(y_pred != y_test)[0]
                        
                        if len(error_indices) == 0:
                            st.warning("⚠️ Aucune erreur de prédiction trouvée! Toutes les prédictions sont correctes.")
                            st.stop()
                        
                        indices = error_indices[:n_samples]
                        st.info(f"ℹ️ {len(error_indices)} erreurs trouvées, affichage de {len(indices)}")
                    else:
                        st.error("❌ Prédictions non disponibles pour filtrer les erreurs")
                        st.stop()
                
                elif selection_mode == "Indices spécifiques":
                    try:
                        indices = [int(i.strip()) for i in indices_str.split(",")]
                        indices = [i for i in indices if 0 <= i < len(X_test)]
                        
                        if len(indices) == 0:
                            st.error("❌ Aucun indice valide trouvé")
                            st.stop()
                    except ValueError:
                        st.error("❌ Format d'indices invalide. Utilisez des nombres séparés par des virgules.")
                        st.stop()
                
                # Extraction des données
                X_selected = X_test[indices]
                y_selected = y_test[indices] if y_test is not None else None
                
                pred_selected = None
                if prediction_results and prediction_results.get('y_pred_binary') is not None:
                    pred_selected = prediction_results['y_pred_binary'][indices]
                
                # Récupération noms de classes
                class_names = None
                if prediction_results and prediction_results.get('class_names'):
                    class_names = prediction_results['class_names']
                elif hasattr(STATE, 'training_results') and STATE.training_results:
                    if isinstance(STATE.training_results, dict):
                        class_names = STATE.training_results.get('class_names')
                
                # Vérifier que STATE.trained_model existe
                if not hasattr(STATE, 'trained_model') or STATE.trained_model is None:
                    STATE.trained_model = model
                
                # Import Grad-CAM
                from src.evaluation.grad_cam import generate_gradcam_visualization
                
                # Génération Grad-CAM
                result = generate_gradcam_visualization(
                    model=STATE.trained_model,
                    images=X_selected,
                    predictions=pred_selected,
                    ground_truths=y_selected,
                    class_names=class_names,
                    target_layer=None,  # Auto-détection
                    mode=mode,
                    alpha=alpha,
                    use_cuda=torch.cuda.is_available()
                )
                
                if result['success']:
                    st.success(f"✅ {len(indices)} visualisations Grad-CAM générées avec succès!")
                    
                    # Affichage de la figure
                    st.pyplot(result['figure'])
                    
                    # Métriques de l'échantillon
                    st.markdown("---")
                    st.markdown("#### 📊 Statistiques de l'échantillon")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Images visualisées", len(indices))
                    
                    with col2:
                        if pred_selected is not None and y_selected is not None:
                            accuracy = (pred_selected == y_selected).mean()
                            st.metric("Accuracy", f"{accuracy:.2%}")
                    
                    with col3:
                        if pred_selected is not None and y_selected is not None:
                            errors = (pred_selected != y_selected).sum()
                            st.metric("Erreurs", errors)
                    
                    with col4:
                        if pred_selected is not None and y_selected is not None:
                            correct = (pred_selected == y_selected).sum()
                            st.metric("Correctes", correct)
                    
                    # Distribution des prédictions
                    if pred_selected is not None and class_names:
                        st.markdown("---")
                        st.markdown("#### 📈 Distribution des Prédictions")
                        
                        import matplotlib.pyplot as plt
                        import pandas as pd
                        
                        pred_counts = pd.Series(pred_selected).value_counts().sort_index()
                        pred_df = pd.DataFrame({
                            'Classe': [class_names[i] if i < len(class_names) else f"Classe_{i}" for i in pred_counts.index],
                            'Nombre': pred_counts.values
                        })
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.dataframe(pred_df, use_container_width=True)
                        
                        with col2:
                            fig_dist = plt.figure(figsize=(8, 6))
                            plt.bar(pred_df['Classe'], pred_df['Nombre'], color='steelblue', alpha=0.7)
                            plt.xlabel('Classe Prédite')
                            plt.ylabel('Nombre d\'images')
                            plt.title('Distribution des Prédictions')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig_dist)
                    
                    # Export
                    st.markdown("---")
                    st.markdown("#### 💾 Export des Visualisations")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Export PNG
                        import io
                        buf = io.BytesIO()
                        result['figure'].savefig(buf, format='png', dpi=150, bbox_inches='tight')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📥 Télécharger Figure (PNG)",
                            data=buf,
                            file_name=f"gradcam_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Export NPZ (données brutes)
                        buf_npz = io.BytesIO()
                        np.savez_compressed(
                            buf_npz,
                            cams=np.array(result['cams']),
                            indices=indices,
                            predictions=pred_selected,
                            ground_truths=y_selected
                        )
                        buf_npz.seek(0)
                        
                        st.download_button(
                            label="📥 Télécharger Données (NPZ)",
                            data=buf_npz,
                            file_name=f"gradcam_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz",
                            mime="application/x-npz",
                            use_container_width=True
                        )
                    
                    # Stockage dans STATE
                    STATE.gradcam_results = result
                    STATE.gradcam_indices = indices
                    
                else:
                    st.error(f"❌ Erreur génération Grad-CAM: {result.get('error', 'Inconnue')}")
                    st.info("💡 Vérifiez que le modèle est compatible et que les données sont correctement chargées.")
            
            except Exception as e:
                st.error(f"❌ Erreur inattendue: {e}")
                logger.error(f"Erreur Grad-CAM UI: {e}", exc_info=True)
                
                with st.expander("🔍 Détails de l'erreur (debug)"):
                    st.code(f"{type(e).__name__}: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    # ====================================================================
    # AIDE
    # ====================================================================
    st.markdown("---")
    
    with st.expander("ℹ️ À propos de Grad-CAM"):
        st.markdown("""
        ### 🔬 Comment ça marche ?
        
        1. **Forward Pass**: L'image passe dans le modèle
        2. **Backward Pass**: Calcul des gradients par rapport à la classe prédite
        3. **Pondération**: Les activations sont pondérées par les gradients
        4. **Visualisation**: Génération d'une heatmap montrant les zones importantes
        
        ### 📊 Interprétation
        
        - **Rouge/Jaune**: Le modèle se concentre fortement sur ces zones
        - **Vert**: Zones d'importance moyenne
        - **Bleu/Violet**: Zones ignorées par le modèle
        
        ### ✅ Utilisations
        
        - **Debugging**: Comprendre pourquoi le modèle se trompe
        - **Validation**: Vérifier que le modèle regarde les bonnes zones
        - **Interprétabilité**: Expliquer les prédictions aux non-experts
        - **Amélioration**: Identifier les biais du modèle
        
        ### 📚 Références
        
        - [Grad-CAM Paper (2017)](https://arxiv.org/abs/1610.02391)
        - [Grad-CAM++ (2018)](https://arxiv.org/abs/1710.11063)
        - [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)
        
        ### ⚠️ Limitations
        
        - Fonctionne uniquement avec des CNNs (modèles convolutionnels)
        - Ne montre pas toujours toutes les zones importantes
        - Peut être trompeur pour des modèles mal entraînés
        - La résolution de la heatmap dépend de la couche convolutionnelle
        
        ### 💡 Conseils
        
        - Comparez plusieurs images pour identifier des patterns
        - Vérifiez si les zones importantes correspondent à l'intuition humaine
        - Utilisez mode "Erreurs uniquement" pour comprendre les échecs
        - Exportez les visualisations pour documentation/présentation
        """)
    
    

# ============================================================================
# FOOTER & NAVIGATION
# ============================================================================

st.markdown("---")

col_nav1, col_nav2, col_nav3, col_nav4 = st.columns(4)

with col_nav1:
    if st.button("🏠 Dashboard", width='stretch'):
        st.switch_page("pages/1_dashboard.py")

with col_nav2:
    if st.button("🔙 Entraînement", width='stretch'):
        st.switch_page("pages/4_training_computer.py")

with col_nav3:
    if st.button("🔄 Nouvelle Évaluation", width='stretch'):
        st.rerun()

with col_nav4:
    if st.button("💾 Sauvegarder Session", type="primary", width='stretch'):
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