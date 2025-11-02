"""
Page Streamlit: Évaluation Détection d'Anomalies - Premium Dashboard
Version complète avec design moderne et analyse approfondie
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    f1_score, precision_score, recall_score, accuracy_score
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
        import logging
        return logging.getLogger(name)

import torch # type: ignore

from monitoring.state_managers import init, AppPage
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

st.markdown("""
<style>
    /* Variables */
    :root {
        --primary: #6366f1;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #3b82f6;
        --bg-card: #ffffff;
        --shadow: 0 1px 3px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    /* Base */
    .block-container {
        padding: 1.5rem 2.5rem !important;
        max-width: 1600px;
    }
    
    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: rgba(255,255,255,0.1);
        border-radius: 50%;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        opacity: 0.95;
        margin: 0.5rem 0 0 0;
        position: relative;
        z-index: 1;
    }
    
    /* Metric Cards Premium */
    .metric-card-premium {
    	background: white;
    	border-radius: 12px;
    	padding: 1.5rem;
    	box-shadow: var(--shadow);
    	border: 1px solid #e5e7eb;
    	transition: box-shadow 0.2s ease;
    }
    .metric-card-premium:hover {
    	box-shadow: var(--shadow-lg);
    }
    .metric-card-premium::before {
    	content: '';
    	position: absolute;
    	top: 0;
    	left: 0;
    	width: 100%;
    	height: 3px;
    	background: var(--primary);
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1f2937;
        margin: 0.5rem 0;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-trend {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }
    
    .trend-up {
        background: #d1fae5;
        color: #065f46;
    }
    
    .trend-down {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 999px;
        font-size: 0.875rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-excellent {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        border: 2px solid #10b981;
    }
    
    .badge-good {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        color: #1e40af;
        border: 2px solid #3b82f6;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        color: #92400e;
        border: 2px solid #f59e0b;
    }
    
    .badge-critical {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        border: 2px solid #ef4444;
    }
    
    /* Panel Cards */
    .panel-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: var(--shadow);
        border: 1px solid #e5e7eb;
        margin-bottom: 1.5rem;
    }
    
    .panel-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #f3f4f6;
    }
    
    .panel-icon {
        font-size: 1.75rem;
    }
    
    .panel-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0;
    }
    
    /* Recommendation Cards */
    .recommendation-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 4px solid #3b82f6;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateX(4px);
        box-shadow: var(--shadow);
    }
    
    .rec-priority-high {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-left-color: #ef4444;
    }
    
    .rec-priority-medium {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left-color: #f59e0b;
    }
    
    .rec-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Error Analysis */
    .error-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid;
    }
    
    .error-fp {
        background: #fef2f2;
        border-color: #fca5a5;
    }
    
    .error-fn {
        background: #fef3c7;
        border-color: #fcd34d;
    }
    
    .error-tp {
        background: #d1fae5;
        border-color: #6ee7b7;
    }
    
    /* Progress Indicator */
    .progress-wrapper {
        background: #f3f4f6;
        border-radius: 999px;
        height: 12px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 999px;
        transition: width 0.5s ease;
    }
    
    .progress-excellent {
        background: linear-gradient(90deg, #10b981, #059669);
    }
    
    .progress-good {
        background: linear-gradient(90deg, #3b82f6, #2563eb);
    }
    
    .progress-warning {
        background: linear-gradient(90deg, #f59e0b, #d97706);
    }
    
    .progress-critical {
        background: linear-gradient(90deg, #ef4444, #dc2626);
    }
    
    /* Tabs Moderne */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f9fafb;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: var(--shadow);
    }
    
    /* Images Gallery */
    .image-gallery {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .image-item {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: var(--shadow);
        transition: transform 0.3s ease;
    }
    
    .image-item:hover {
        transform: scale(1.05);
    }
    
    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1rem;
        background: #f9fafb;
        border-radius: 12px;
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 800;
        color: #1f2937;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.25rem;
    }
    
    /* Animations */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-in {
        animation: slideIn 0.5s ease-out;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem !important;
        }
        .hero-title {
            font-size: 1.75rem;
        }
        .metric-value {
            font-size: 1.75rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FONCTIONS MÉTIER
# ============================================================================

def safe_convert_history(history):
    """Corrige l'historique d'entraînement."""
    if not history:
        return {}
    
    fixed_history = {}
    for key, value in history.items():
        if isinstance(value, bool):
            fixed_history[key] = [1.0 if value else 0.0]
        elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
            cleaned = []
            for item in value:
                if isinstance(item, bool):
                    cleaned.append(1.0 if item else 0.0)
                elif isinstance(item, (int, float)):
                    cleaned.append(float(item))
                else:
                    cleaned.append(0.0)
            fixed_history[key] = cleaned
        else:
            fixed_history[key] = value
    
    return fixed_history


def robust_predict_with_preprocessor(model, X_test, preprocessor, model_type):
    """
    Prédictions robustes avec gestion complète des cas edge.
    - Gestion preprocessor None
    - Validation des shapes
    - Try-except sur chaque transformation
    - Logs détaillés des échecs
    """
    try:
        # Preprocessing avec gestion None
        if preprocessor is not None:
            try:
                # Tenter transformation avec preprocessor
                X_processed = preprocessor.transform(X_test, output_format="channels_first")
                logger.info(f"✅ Preprocessing réussi: {X_processed.shape}")
            except AttributeError as e:
                # Preprocessor sans méthode transform
                logger.warning(f"⚠️ Preprocessor sans transform(): {e}")
                X_processed = X_test.copy()
            except Exception as e:
                # Erreur transformation
                logger.warning(f"⚠️ Erreur preprocessing, utilisation données brutes: {e}")
                X_processed = X_test.copy()
        else:
            logger.info("ℹ️ Pas de preprocessor, utilisation données brutes")
            X_processed = X_test.copy()
        
        # Validation shape
        if len(X_processed.shape) != 4:
            logger.error(f"❌ Shape invalide: {X_processed.shape}, attendu: (N, C, H, W)")
            # Tentative de correction
            if len(X_processed.shape) == 3:
                # Ajouter dimension channel
                X_processed = np.expand_dims(X_processed, axis=1)
                logger.info(f"✅ Shape corrigée: {X_processed.shape}")
        
        # Device avec gestion CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.info(f"🖥️ Device: {device}, Shape entrée: {X_processed.shape}")
        
        # Conversion tensor avec dtype explicite
        try:
            X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
        except Exception as e:
            logger.error(f"❌ Erreur conversion tensor: {e}")
            # Tentative de correction dtype
            X_processed = X_processed.astype(np.float32)
            X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if model_type in ["autoencoder", "conv_autoencoder", "variational_autoencoder", "denoising_autoencoder"]:
                # AUTOENCODER BRANCH
                try:
                    reconstructed = model(X_tensor)
                    reconstructed_np = reconstructed.cpu().numpy()
                    
                    # Calcul erreur de reconstruction safe
                    reconstruction_errors = np.mean(
                        (X_processed - reconstructed_np) ** 2,
                        axis=(1, 2, 3) if len(X_processed.shape) == 4 else (1,)
                    )
                    
                    # Normalisation avec protection division par zéro
                    max_error = np.max(reconstruction_errors)
                    if max_error > 0:
                        y_pred_proba = reconstruction_errors / max_error
                    else:
                        logger.warning("⚠️ Erreur reconstruction nulle, utilisation valeurs uniformes")
                        y_pred_proba = np.ones(len(reconstruction_errors)) * 0.5
                    
                    # Seuil adaptatif basé sur distribution
                    threshold = np.median(y_pred_proba) + np.std(y_pred_proba)
                    threshold = np.clip(threshold, 0.3, 0.7)  # Entre 0.3 et 0.7
                    
                    y_pred_binary = (y_pred_proba > threshold).astype(int)
                    
                    logger.info(
                        f"✅ Prédictions autoencoder: {len(y_pred_binary)} samples, "
                        f"seuil: {threshold:.3f}, anomalies: {y_pred_binary.sum()}"
                    )
                    
                    return {
                        "y_pred_proba": y_pred_proba,
                        "y_pred_binary": y_pred_binary,
                        "reconstruction_errors": reconstruction_errors,
                        "reconstructed": reconstructed_np,
                        "adaptive_threshold": threshold,
                        "success": True
                    }
                
                except Exception as e:
                    logger.error(f"❌ Erreur prédiction autoencoder: {e}", exc_info=True)
                    raise
            
            else:
                # CLASSIFICATION BRANCH
                try:
                    output = model(X_tensor)
                    
                    # Gestion multiple formats output
                    if hasattr(output, 'logits'):
                        y_proba = torch.softmax(output.logits, dim=1).cpu().numpy()
                    elif isinstance(output, tuple):
                        # Certains modèles retournent (logits, features)
                        y_proba = torch.softmax(output[0], dim=1).cpu().numpy()
                    else:
                        y_proba = torch.softmax(output, dim=1).cpu().numpy()
                    
                    # Extraction probabilité classe positive
                    if y_proba.shape[1] == 2:
                        y_pred_proba = y_proba[:, 1]
                    elif y_proba.shape[1] == 1:
                        y_pred_proba = y_proba[:, 0]
                    else:
                        # Multi-classes: prendre max
                        y_pred_proba = np.max(y_proba, axis=1)
                    
                    y_pred_binary = (y_pred_proba > 0.5).astype(int)
                    
                    logger.info(
                        f"✅ Prédictions classification: {len(y_pred_binary)} samples, "
                        f"anomalies: {y_pred_binary.sum()}"
                    )
                    
                    return {
                        "y_pred_proba": y_pred_proba,
                        "y_pred_binary": y_pred_binary,
                        "class_probabilities": y_proba,
                        "success": True
                    }
                
                except Exception as e:
                    logger.error(f"❌ Erreur prédiction classification: {e}", exc_info=True)
                    raise
        
    except Exception as e:
        logger.error(f"❌ Erreur critique prédiction: {e}", exc_info=True)
        
        # Génération prédictions aléatoires réalistes
        logger.warning("⚠️ Utilisation fallback: prédictions aléatoires")
        
        if model_type in ["autoencoder", "conv_autoencoder"]:
            # Pour autoencoder: distribution normale autour de 0.3
            reconstruction_errors = np.random.normal(0.3, 0.15, len(X_test))
            reconstruction_errors = np.clip(reconstruction_errors, 0, 1)
            
            threshold = 0.5
            
            return {
                "y_pred_proba": reconstruction_errors,
                "y_pred_binary": (reconstruction_errors > threshold).astype(int),
                "reconstruction_errors": reconstruction_errors,
                "reconstructed": X_test.copy(),
                "adaptive_threshold": threshold,
                "success": False,
                "fallback": True
            }
        else:
            # Pour classification: distribution uniforme biaisée
            y_pred_proba = np.random.beta(2, 5, len(X_test))  # Biais vers valeurs basses
            
            return {
                "y_pred_proba": y_pred_proba,
                "y_pred_binary": (y_pred_proba > 0.5).astype(int),
                "success": False,
                "fallback": True
            }


def analyze_false_positives(X_test, y_test, y_pred_binary):
    """Analyse des erreurs."""
    false_positives = np.where((y_test == 0) & (y_pred_binary == 1))[0]
    false_negatives = np.where((y_test == 1) & (y_pred_binary == 0))[0]
    true_positives = np.where((y_test == 1) & (y_pred_binary == 1))[0]
    true_negatives = np.where((y_test == 0) & (y_pred_binary == 0))[0]
    
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_positives": true_positives,
        "true_negatives": true_negatives,
        "fp_count": len(false_positives),
        "fn_count": len(false_negatives),
        "tp_count": len(true_positives),
        "tn_count": len(true_negatives),
        "fp_rate": len(false_positives) / max(len(y_test[y_test == 0]), 1),
        "fn_rate": len(false_negatives) / max(len(y_test[y_test == 1]), 1),
        "total_errors": len(false_positives) + len(false_negatives)
    }


def get_performance_status(metric_value, metric_type):
    """Retourne le statut de performance."""
    if metric_type == "auc_roc":
        if metric_value >= 0.9: return "excellent", "🎯 Excellent"
        elif metric_value >= 0.8: return "good", "✅ Bon"
        elif metric_value >= 0.7: return "warning", "⚠️ Moyen"
        else: return "critical", "❌ Critique"
    elif metric_type in ["f1_score", "precision", "recall"]:
        if metric_value >= 0.85: return "excellent", "🎯 Excellent"
        elif metric_value >= 0.75: return "good", "✅ Bon"
        elif metric_value >= 0.6: return "warning", "⚠️ Moyen"
        else: return "critical", "❌ Critique"
    else:
        if metric_value >= 0.8: return "good", "✅ Bon"
        elif metric_value >= 0.6: return "warning", "⚠️ Moyen"
        else: return "critical", "❌ Critique"


def create_performance_summary(metrics, error_analysis):
    """Crée un résumé des performances."""
    weights = {
        'auc_roc': 0.25,
        'f1_score': 0.25,
        'precision': 0.20,
        'recall': 0.20,
        'specificity': 0.10
    }
    
    total_score = sum(metrics.get(k, 0) * v for k, v in weights.items() if k in metrics)
    valid_weight = sum(v for k, v in weights.items() if k in metrics)
    overall_score = total_score / valid_weight if valid_weight > 0 else 0
    
    summary = {
        "overall_score": overall_score,
        "production_ready": overall_score >= 0.75,
        "risk_level": "low" if overall_score >= 0.85 else "medium" if overall_score >= 0.75 else "high",
        "strengths": [],
        "weaknesses": []
    }
    
    if overall_score >= 0.85:
        summary["status"] = "excellent"
        summary["strengths"] = ["Performances exceptionnelles", "Prêt production"]
    elif overall_score >= 0.75:
        summary["status"] = "good"
        summary["strengths"] = ["Bonnes performances"]
        summary["weaknesses"] = ["Optimisations possibles"]
    elif overall_score >= 0.6:
        summary["status"] = "warning"
        summary["weaknesses"] = ["Optimisations nécessaires"]
    else:
        summary["status"] = "critical"
        summary["weaknesses"] = ["Re-entraînement recommandé"]
    
    return summary


def generate_recommendations(metrics, model_type, error_analysis, performance_summary):
    """Génère des recommandations."""
    recommendations = []
    
    if performance_summary["overall_score"] < 0.6:
        recommendations.append({
            "priority": "high",
            "icon": "🔴",
            "action": "Re-entraînement complet",
            "message": "Performances insuffisantes. Re-entraîner avec plus de données."
        })
    
    if metrics.get('recall', 1) < 0.7:
        recommendations.append({
            "priority": "high",
            "icon": "🔍",
            "action": "Améliorer détection",
            "message": "Rappel faible. Anomalies manquées. Ajuster le seuil."
        })
    
    if metrics.get('precision', 1) < 0.7:
        recommendations.append({
            "priority": "medium",
            "icon": "⚖️",
            "action": "Réduire faux positifs",
            "message": "Trop de faux positifs. Augmenter seuil ou améliorer données."
        })
    
    if error_analysis.get('fp_rate', 0) > 0.1:
        recommendations.append({
            "priority": "medium",
            "icon": "📊",
            "action": "Analyser faux positifs",
            "message": f"Taux FP élevé ({error_analysis['fp_rate']:.1%}). Examiner images."
        })
    
    if performance_summary["production_ready"]:
        recommendations.append({
            "priority": "low",
            "icon": "🚀",
            "action": "Déploiement production",
            "message": "Modèle prêt. Configurer monitoring."
        })
    
    return recommendations


def create_performance_radar(metrics):
    """Crée un radar chart."""
    categories = ['AUC-ROC', 'F1-Score', 'Precision', 'Recall', 'Specificity']
    values = [
        metrics.get('auc_roc', 0),
        metrics.get('f1_score', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('specificity', 0)
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.3)',
        line=dict(color='#6366f1', width=3),
        name='Performance'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        height=400,
        title="Analyse Multidimensionnelle"
    )
    
    return fig


def plot_error_distribution(error_analysis):
    """Graphique distribution erreurs."""
    labels = ['Vrais Positifs', 'Faux Positifs', 'Vrais Négatifs', 'Faux Négatifs']
    values = [
        error_analysis['tp_count'],
        error_analysis['fp_count'],
        error_analysis['tn_count'],
        error_analysis['fn_count']
    ]
    
    colors = ['#10b981', '#ef4444', '#3b82f6', '#f59e0b']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker_colors=colors,
        textinfo='label+percent+value'
    )])
    
    fig.update_layout(
        title="Distribution des Prédictions",
        height=400
    )
    
    return fig


# ============================================================================
# VÉRIFICATIONS INITIALES
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
    
    # Accès safe à model_config
    if not hasattr(STATE, 'model_config') or STATE.model_config is None:
        st.error("❌ Configuration du modèle manquante")
        st.stop()
    
    model_type = STATE.model_config.get("model_type", "autoencoder")
    
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
    <h1 class="hero-title">📊 Dashboard d'Évaluation Premium</h1>
    <p class="hero-subtitle">Analyse approfondie des performances en détection d'anomalies</p>
</div>
''', unsafe_allow_html=True)

# Prédictions
with st.spinner("🔮 Calcul des prédictions..."):
    prediction_results = robust_predict_with_preprocessor(model, X_test, preprocessor, model_type)
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


# TAB 3: RECOMMANDATIONS
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


# TAB 4: VISUALISATIONS
with tabs[3]:
    st.markdown("### 🎨 Visualisations Avancées")
    
    # Courbes ROC et PR
    st.markdown("#### 📈 Courbes de Performance")
    
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        # Simulation courbe ROC
        from sklearn.metrics import roc_curve
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
        # Simulation courbe Precision-Recall
        from sklearn.metrics import precision_recall_curve
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


# TAB 5: RAPPORT
with tabs[4]:
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