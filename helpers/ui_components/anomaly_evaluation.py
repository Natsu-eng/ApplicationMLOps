"""
Helpers pour la page d'évaluation d'anomalies
Fonctions métier extraites de 5_anomaly_evaluation.py
"""
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional
from src.shared.logging import get_logger

logger = get_logger(__name__)


def safe_convert_history(history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Corrige l'historique d'entraînement.
    
    Args:
        history: Historique brut de l'entraînement
    
    Returns:
        Historique corrigé avec valeurs numériques
    """
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


def analyze_false_positives(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred_binary: np.ndarray
) -> Dict[str, Any]:
    """
    Analyse des erreurs de classification.
    
    Args:
        X_test: Images de test
        y_test: Labels réels
        y_pred_binary: Prédictions binaires
    
    Returns:
        Dictionnaire avec analyse complète des erreurs
    """
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


def get_performance_status(
    metric_value: float,
    metric_type: str
) -> tuple[str, str]:
    """
    Retourne le statut de performance basé sur une métrique.
    
    Args:
        metric_value: Valeur de la métrique
        metric_type: Type de métrique (auc_roc, f1_score, etc.)
    
    Returns:
        Tuple (status, status_text) avec le statut et son label
    """
    if metric_type == "auc_roc":
        if metric_value >= 0.9:
            return "excellent", "🎯 Excellent"
        elif metric_value >= 0.8:
            return "good", "✅ Bon"
        elif metric_value >= 0.7:
            return "warning", "⚠️ Moyen"
        else:
            return "critical", "❌ Critique"
    elif metric_type in ["f1_score", "precision", "recall"]:
        if metric_value >= 0.85:
            return "excellent", "🎯 Excellent"
        elif metric_value >= 0.75:
            return "good", "✅ Bon"
        elif metric_value >= 0.6:
            return "warning", "⚠️ Moyen"
        else:
            return "critical", "❌ Critique"
    else:
        if metric_value >= 0.8:
            return "good", "✅ Bon"
        elif metric_value >= 0.6:
            return "warning", "⚠️ Moyen"
        else:
            return "critical", "❌ Critique"


def create_performance_summary(
    metrics: Dict[str, float],
    error_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Crée un résumé des performances globales.
    
    Args:
        metrics: Dictionnaire de métriques
        error_analysis: Analyse des erreurs
    
    Returns:
        Résumé avec score global, statut production, etc.
    """
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


def generate_recommendations(
    metrics: Dict[str, float],
    model_type: str,
    error_analysis: Dict[str, Any],
    performance_summary: Dict[str, Any]
) -> list[Dict[str, str]]:
    """
    Génère des recommandations basées sur les performances.
    
    Args:
        metrics: Métriques calculées
        model_type: Type de modèle
        error_analysis: Analyse des erreurs
        performance_summary: Résumé des performances
    
    Returns:
        Liste de recommandations avec priorité
    """
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


def create_performance_radar(metrics: Dict[str, float]) -> go.Figure:
    """
    Crée un graphique radar des performances.
    
    Args:
        metrics: Dictionnaire de métriques
    
    Returns:
        Figure Plotly avec graphique radar
    """
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


def plot_error_distribution(error_analysis: Dict[str, Any]) -> go.Figure:
    """
    Graphique de distribution des erreurs.
    
    Args:
        error_analysis: Analyse des erreurs
    
    Returns:
        Figure Plotly avec graphique en camembert
    """
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



