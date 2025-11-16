"""
Module de visualisation des modèles ML - Version Production
Visualisations avancées pour l'évaluation des modèles avec gestion robuste des erreurs
"""

import tempfile
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import gc
from datetime import datetime
from pathlib import Path

# Import des modules déplacés
from src.shared.logging import get_logger
from monitoring.decorators import monitor_operation, timeout
from monitoring.state_managers import STATE
from monitoring.system_monitor import get_system_metrics
from utils.formatters import format_metric_value

# Configuration des imports conditionnels avec fallbacks robustes
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    import shap # type: ignore
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import psutil # type: ignore
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from matplotlib import cm
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import (
        silhouette_samples, silhouette_score, 
        confusion_matrix, roc_curve, precision_recall_curve, auc,
        mean_squared_error, mean_absolute_error, r2_score
    )
    from sklearn.model_selection import learning_curve, validation_curve
    from sklearn.metrics.pairwise import euclidean_distances
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False

# Import des constantes avec fallback
try:
    from src.config.constants import (
        VISUALIZATION_CONSTANTS, LOGGING_CONSTANTS, 
        VALIDATION_CONSTANTS, TRAINING_CONSTANTS
    )
except ImportError:
    # Fallback des constantes en cas d'import échoué
    VISUALIZATION_CONSTANTS = {
        "PLOTLY_TEMPLATE": "plotly_white",
        "MAX_SAMPLES": 10000,
        "TRAIN_SIZES": np.linspace(0.1, 1.0, 10),
        "COLOR_PALETTE": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
        "HEATMAP_COLORMAP": "Viridis"
    }
    LOGGING_CONSTANTS = {
        "DEFAULT_LOG_LEVEL": "INFO",
        "LOG_DIR": "logs",
        "LOG_FILE": "model_plots.log",
        "CONSOLE_LOGGING": True,
        "SLOW_OPERATION_THRESHOLD": 30,
        "HIGH_MEMORY_THRESHOLD": 500
    }
    VALIDATION_CONSTANTS = {
        "MIN_SAMPLES_PLOT": 10,
        "MAX_FEATURES_PLOT": 50
    }
    TRAINING_CONSTANTS = {
        "RANDOM_STATE": 42,
        "CV_FOLDS": 5,
        "N_JOBS": -1
    }

# Initialisation du logger
logger = get_logger(__name__)

def _safe_get(obj: Any, keys: List[str], default: Optional[Any] = None) -> Optional[Any]:
    """Récupération sécurisée d'attributs imbriqués - VERSION UNIFIÉE"""
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
    return current if current is not None else default


def _create_empty_plot(message: str, height: int = 400) -> go.Figure:
    """Crée un graphique vide avec message d'erreur - SPÉCIFIQUE AUX VISUALISATIONS"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color="#e74c3c"),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="#e74c3c",
        borderwidth=1
    )
    fig.update_layout(
        title="Visualisation non disponible",
        template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
        height=height,
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='rgba(240, 240, 240, 0.1)',
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig


def _generate_color_palette(n_colors: int) -> List[str]:
    """Génère une palette de couleurs - VERSION UNIFIÉE"""
    if MATPLOTLIB_AVAILABLE and n_colors <= 20:
        try:
            cmap = cm.get_cmap('viridis', n_colors)
            return [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                   for r, g, b, _ in cmap(np.linspace(0, 1, n_colors))]
        except Exception:
            pass
    
    # Fallback vers une palette prédéfinie
    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    
    # Génération de couleurs supplémentaires si nécessaire
    import colorsys
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
    
    return colors


""" def _safe_get_model_task_type(model_result: Dict) -> str:
    Détection robuste du type de tâche - VERSION ULTRA ROBUSTE

    try:
        # PRIORITÉ ABSOLUE : Utiliser STATE.task_type si disponible
        if hasattr(STATE, 'task_type') and STATE.task_type in ['classification', 'regression', 'clustering']:
            return STATE.task_type
        
        # Fallback : analyse des métriques
        metrics = model_result.get('metrics', {})
        
        # Clustering metrics
        if any(metric in metrics for metric in ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']):
            return 'clustering'
        
        # Classification metrics
        if any(metric in metrics for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']):
            return 'classification'
        
        # Regression metrics  
        if any(metric in metrics for metric in ['r2', 'mse', 'mae', 'rmse']):
            return 'regression'
        
        # Fallback final
        return 'unknown'
        
    except Exception as e:
        print(f"DEBUG - Erreur détection type tâche: {e}")
        return 'unknown' """

def _safe_get(obj: Any, keys: List[str], default: Optional[Any] = None) -> Optional[Any]:
    """Récupération sécurisée d'attributs imbriqués"""
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
    return current if current is not None else default


def _generate_color_palette(n_colors: int) -> List[str]:
    """Génère une palette de couleurs"""
    if MATPLOTLIB_AVAILABLE and n_colors <= 20:
        try:
            cmap = cm.get_cmap('viridis', n_colors)
            return [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' 
                   for r, g, b, _ in cmap(np.linspace(0, 1, n_colors))]
        except Exception:
            pass
    
    # Fallback vers une palette prédéfinie
    base_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
        '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
    ]
    
    if n_colors <= len(base_colors):
        return base_colors[:n_colors]
    
    # Génération de couleurs supplémentaires si nécessaire
    import colorsys
    colors = []
    for i in range(n_colors):
        hue = i / n_colors
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
        colors.append(f'rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})')
    
    return colors

# Classe principale de visualisation
class ModelEvaluationVisualizer:
    """
    Visualisateur robuste pour l'évaluation des modèles ML
    Gère les visualisations pour classification, régression et clustering
    """
    
    def __init__(self, ml_results: List[Dict[str, Any]]):
        self.ml_results = ml_results or []
        self.validation_result = self._validate_data()
        self._temp_dir = Path(tempfile.gettempdir()) / "ml_plots_cache"
        self._temp_dir.mkdir(exist_ok=True)
        self._plot_cache = {}
        
        logger.info("Visualizer initialisé", {
            "n_results": len(self.ml_results),
            "task_type": self.validation_result.get("task_type", "unknown")
        })

    @monitor_operation
    def _validate_data(self) -> Dict[str, Any]:
        """
        Valide les résultats ML fournis
        1. Vérifie la présence de résultats
        2. Sépare modèles réussis et échoués
        3. Détecte le type de tâche ML
        4. Identifie le meilleur modèle
        5. Calcule un résumé des métriques
        6. Gère les erreurs et avertissements
        """
        validation = {
            "has_results": False,
            "results_count": 0,
            "task_type": "unknown",
            "best_model": None,
            "successful_models": [],
            "failed_models": [],
            "errors": [],
            "warnings": [],
            "metrics_summary": {}
        }
        
        try:
            if not self.ml_results:
                validation["errors"].append("Aucun résultat ML fourni")
                logger.error("❌ ml_results vide")
                return validation
            
            validation["results_count"] = len(self.ml_results)
            validation["has_results"] = True
            
            # Séparation modèles
            for result in self.ml_results:
                if not isinstance(result, dict):
                    validation["warnings"].append(f"Résultat non-dict ignoré: {type(result)}")
                    continue
                
                model_name = result.get('model_name', 'Unknown')
                
                # Vérification erreur
                has_error = (
                    result.get('metrics', {}).get('error') is not None or
                    not result.get('success', False)
                )
                
                if has_error:
                    validation["failed_models"].append(result)
                    logger.debug(f"Modèle échoué: {model_name}")
                else:
                    # Validation métriques présentes
                    metrics = result.get('metrics', {})
                    if not metrics or not isinstance(metrics, dict):
                        validation["warnings"].append(f"{model_name}: metrics invalides")
                        validation["failed_models"].append(result)
                    else:
                        validation["successful_models"].append(result)
                        logger.debug(f"Modèle réussi: {model_name}")
            
            logger.info(
                f"✅ Validation: {len(validation['successful_models'])} réussis, "
                f"{len(validation['failed_models'])} échoués"
            )
            
            # Détection task_type
            if validation["successful_models"]:
                validation["task_type"] = self._detect_task_type(validation["successful_models"])
                
                # Best model
                validation["best_model"] = self._find_best_model(
                    validation["successful_models"],
                    validation["task_type"]
                )
                
                # Résumé métriques
                validation["metrics_summary"] = self._compute_metrics_summary(
                    validation["successful_models"],
                    validation["task_type"]
                )
        
        except Exception as e:
            validation["errors"].append(f"Erreur validation: {str(e)}")
            logger.error(f"❌ Validation échouée: {e}", exc_info=True)
        
        return validation

    def _detect_task_type(self, successful_models: List[Dict]) -> str:
        """
        Détecte le type de tâche ML parmi les modèles réussis
        1. STATE.task_type (le plus fiable)
        2. Analyse task_type dans chaque résultat
        3. Analyse structure données (labels vs y_test)
        4. Analyse métriques
        """
        if not successful_models:
            return "unknown"
        
        # PRIORITÉ 1: STATE.task_type (source unique de vérité)
        try:
            from monitoring.state_managers import STATE
            if hasattr(STATE, 'task_type') and STATE.task_type in ['classification', 'regression', 'clustering']:
                logger.info(f"✅ Type tâche depuis STATE: {STATE.task_type}")
                return STATE.task_type
        except Exception as e:
            logger.warning(f"⚠️ Erreur accès STATE.task_type: {e}")
        
        # PRIORITÉ 2: task_type explicite dans résultats
        task_types_found = []
        for model in successful_models:
            explicit_type = model.get('task_type')
            if explicit_type and explicit_type in ['classification', 'regression', 'clustering']:
                task_types_found.append(explicit_type)
        
        if task_types_found:
            from collections import Counter
            most_common = Counter(task_types_found).most_common(1)
            detected = most_common[0][0]
            logger.info(f"✅ Type tâche depuis résultats: {detected}")
            return detected
        
        # PRIORITÉ 3: Analyse structure données
        first_model = successful_models[0]
        
        has_labels = first_model.get('labels') is not None
        has_y_test = first_model.get('y_test') is not None
        
        if has_labels and not has_y_test:
            logger.info("✅ Type tâche: clustering (labels présents)")
            return 'clustering'
        
        if has_y_test:
            try:
                y_test = first_model['y_test']
                
                # Conversion robuste
                if hasattr(y_test, 'values'):
                    y_array = y_test.values
                else:
                    y_array = np.array(y_test)
                
                y_array = y_array.ravel()
                unique_vals = np.unique(y_array)
                n_unique = len(unique_vals)
                n_total = len(y_array)
                
                # Heuristique: classification si peu de valeurs uniques
                is_classification = (
                    n_unique <= 20 or
                    (n_unique / n_total < 0.1 and n_unique > 1)
                )
                
                detected = 'classification' if is_classification else 'regression'
                logger.info(f"✅ Type tâche par y_test: {detected} ({n_unique} valeurs uniques)")
                return detected
            
            except Exception as e:
                logger.error(f"❌ Erreur analyse y_test: {e}")
        
        # PRIORITÉ 4: Analyse métriques
        metrics = first_model.get('metrics', {})
        
        clustering_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        classification_metrics = ['accuracy', 'precision', 'recall', 'f1', 'f1_score', 'auc']
        regression_metrics = ['r2', 'mse', 'mae', 'rmse']
        
        if any(m in metrics for m in clustering_metrics):
            logger.info("✅ Type tâche par métriques: clustering")
            return 'clustering'
        elif any(m in metrics for m in classification_metrics):
            logger.info("✅ Type tâche par métriques: classification")
            return 'classification'
        elif any(m in metrics for m in regression_metrics):
            logger.info("✅ Type tâche par métriques: regression")
            return 'regression'
        
        logger.warning("⚠️ Type tâche non détecté, fallback: classification")
        return 'classification'

    def _find_best_model(self, models: List[Dict], task_type: str) -> Optional[str]:
        """Trouve le meilleur modèle selon le type de tâche"""
        if not models:
            return None
            
        try:
            if task_type == 'classification':
                best_model = max(models, key=lambda x: (
                    _safe_get(x, ['metrics', 'accuracy'], 0),
                    _safe_get(x, ['metrics', 'f1'], 0)
                ))
            elif task_type == 'regression':
                best_model = max(models, key=lambda x: _safe_get(x, ['metrics', 'r2'], -999))
            elif task_type == 'clustering':
                best_model = max(models, key=lambda x: _safe_get(x, ['metrics', 'silhouette_score'], -999))
            else:
                best_model = models[0]
            
            return _safe_get(best_model, ['model_name'])
        except Exception as e:
            logger.warning(f"Erreur recherche meilleur modèle: {str(e)}")
            return _safe_get(models[0], ['model_name'])

    def _compute_metrics_summary(self, models: List[Dict], task_type: str) -> Dict[str, Any]:
        """Calcule un résumé des métriques pour tous les modèles"""
        summary = {}
        
        try:
            if task_type == 'classification':
                accuracies = [_safe_get(m, ['metrics', 'accuracy'], 0) for m in models]
                f1_scores = [_safe_get(m, ['metrics', 'f1'], 0) for m in models]
                summary = {
                    'accuracy_mean': float(np.mean(accuracies)),
                    'accuracy_std': float(np.std(accuracies)),
                    'f1_mean': float(np.mean(f1_scores)),
                    'f1_std': float(np.std(f1_scores))
                }
            elif task_type == 'regression':
                r2_scores = [_safe_get(m, ['metrics', 'r2'], 0) for m in models]
                rmse_scores = [_safe_get(m, ['metrics', 'rmse'], 0) for m in models]
                summary = {
                    'r2_mean': float(np.mean(r2_scores)),
                    'r2_std': float(np.std(r2_scores)),
                    'rmse_mean': float(np.mean(rmse_scores)),
                    'rmse_std': float(np.std(rmse_scores))
                }
            elif task_type == 'clustering':
                silhouette_scores = [_safe_get(m, ['metrics', 'silhouette_score'], 0) for m in models]
                summary = {
                    'silhouette_mean': float(np.mean(silhouette_scores)),
                    'silhouette_std': float(np.std(silhouette_scores))
                }
        except Exception as e:
            logger.warning(f"Erreur calcul résumé métriques: {str(e)}")
        
        return summary

    # Méthodes principales de visualisation
    @monitor_operation
    @timeout(seconds=60)
    def create_comparison_plot(self) -> go.Figure:
        """Crée un graphique de comparaison des modèles"""
        try:
            successful_models = self.validation_result["successful_models"]
            
            if not successful_models:
                return _create_empty_plot("Aucun modèle valide à comparer")
            
            model_names = [_safe_get(r, ['model_name'], f'Modèle_{i}') 
                         for i, r in enumerate(successful_models)]
            
            task_type = self.validation_result["task_type"]
            
            if task_type == 'classification':
                return self._create_classification_comparison(successful_models, model_names)
            elif task_type == 'regression':
                return self._create_regression_comparison(successful_models, model_names)
            elif task_type == 'clustering':
                return self._create_clustering_comparison(successful_models, model_names)
            else:
                return _create_empty_plot(f"Type de tâche '{task_type}' non supporté")
                
        except Exception as e:
            logger.error(f"Graphique comparaison échoué: {str(e)}")
            return _create_empty_plot(f"Erreur création graphique: {str(e)}")

    def _create_classification_comparison(self, models: List[Dict], model_names: List[str]) -> go.Figure:
        """Graphique de comparaison pour classification"""
        metrics_data = {
            'Accuracy': [_safe_get(m, ['metrics', 'accuracy'], 0) for m in models],
            'F1-Score': [_safe_get(m, ['metrics', 'f1'], 0) for m in models],
            'Precision': [_safe_get(m, ['metrics', 'precision'], 0) for m in models],
            'Recall': [_safe_get(m, ['metrics', 'recall'], 0) for m in models]
        }
        
        fig = go.Figure()
        colors = _generate_color_palette(len(metrics_data))
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            fig.add_trace(go.Bar(
                name=metric_name,
                x=model_names, 
                y=values,
                marker_color=colors[i],
                text=[f"{v:.3f}" for v in values],
                textposition='auto',
                hovertemplate=f"{metric_name}: %{{y:.3f}}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Comparaison des Modèles - Classification",
            xaxis_title="Modèles",
            yaxis_title="Score",
            height=500,
            template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode='closest'
        )
        fig.update_xaxes(tickangle=45)
        
        return fig

    def _create_regression_comparison(self, models: List[Dict], model_names: List[str]) -> go.Figure:
        """Graphique de comparaison pour régression"""
        r2_scores = [_safe_get(m, ['metrics', 'r2'], 0) for m in models]
        rmse_scores = [_safe_get(m, ['metrics', 'rmse'], 0) for m in models]
        mae_scores = [_safe_get(m, ['metrics', 'mae'], 0) for m in models]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² Score (plus haut = mieux)', 'Erreurs (plus bas = mieux)'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # R² scores
        fig.add_trace(
            go.Bar(x=model_names, y=r2_scores, name='R²', 
                   marker_color='#2ecc71', 
                   text=[f"{v:.3f}" for v in r2_scores],
                   textposition='auto'),
            row=1, col=1
        )
        
        # Erreurs
        fig.add_trace(
            go.Bar(x=model_names, y=rmse_scores, name='RMSE', 
                   marker_color='#e74c3c',
                   text=[f"{v:.3f}" for v in rmse_scores],
                   textposition='auto'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=model_names, y=mae_scores, name='MAE', 
                   marker_color='#f39c12',
                   text=[f"{v:.3f}" for v in mae_scores],
                   textposition='auto'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Comparaison des Modèles - Régression",
            height=500,
            template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.update_xaxes(tickangle=45)
        
        return fig

    def _create_clustering_comparison(self, models: List[Dict], model_names: List[str]) -> go.Figure:
        """Graphique de comparaison pour clustering"""
        silhouette_scores = [_safe_get(m, ['metrics', 'silhouette_score'], 0) for m in models]
        n_clusters = [_safe_get(m, ['metrics', 'n_clusters'], 0) for m in models]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Score de Silhouette', 'Nombre de Clusters'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Scores de silhouette avec code couleur
        colors_sil = []
        for score in silhouette_scores:
            if score > 0.5:
                colors_sil.append('#27ae60')  # Vert - bon
            elif score > 0.3:
                colors_sil.append('#f39c12')  # Orange - moyen
            else:
                colors_sil.append('#e74c3c')  # Rouge - faible
        
        fig.add_trace(
            go.Bar(x=model_names, y=silhouette_scores, name='Silhouette',
                   marker_color=colors_sil,
                   text=[f"{v:.3f}" for v in silhouette_scores],
                   textposition='auto'),
            row=1, col=1
        )
        
        # Nombre de clusters
        fig.add_trace(
            go.Bar(x=model_names, y=n_clusters, name='Clusters',
                   marker_color='#3498db',
                   text=[f"{int(v)}" for v in n_clusters],
                   textposition='auto'),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Comparaison des Modèles - Clustering",
            height=500,
            template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        
        return fig

    @monitor_operation
    @timeout(seconds=60)
    def create_performance_distribution(self) -> go.Figure:
        """Crée un histogramme de distribution des performances"""
        try:
            successful_models = self.validation_result["successful_models"]
            
            if not successful_models:
                return _create_empty_plot("Aucune donnée de performance disponible")
            
            task_type = self.validation_result["task_type"]
            
            if task_type == 'classification':
                values = [_safe_get(m, ['metrics', 'accuracy'], 0) for m in successful_models]
                title = "Distribution des Scores d'Accuracy"
                x_title = "Score d'Accuracy"
                color = '#3498db'
                
            elif task_type == 'regression':
                values = [_safe_get(m, ['metrics', 'r2'], 0) for m in successful_models]
                title = "Distribution des Scores R²"
                x_title = "Score R²"
                color = '#2ecc71'
                
            elif task_type == 'clustering':
                values = [_safe_get(m, ['metrics', 'silhouette_score'], 0) for m in successful_models]
                title = "Distribution des Scores de Silhouette"
                x_title = "Score de Silhouette"
                color = '#9b59b6'
            else:
                return _create_empty_plot("Type de tâche non supporté")
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=values, 
                nbinsx=min(15, len(values)), 
                marker_color=color,
                opacity=0.7,
                name="Distribution",
                hovertemplate="Score: %{x:.3f}<br>Fréquence: %{y}<extra></extra>"
            ))
            
            # Ligne de moyenne
            mean_val = np.mean(values)
            fig.add_vline(
                x=mean_val, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Moyenne: {mean_val:.3f}",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title=title,
                xaxis_title=x_title,
                yaxis_title="Nombre de Modèles",
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                height=450,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Distribution performances échouée: {str(e)}")
            return _create_empty_plot(f"Erreur: {str(e)}")

    @monitor_operation
    @timeout(seconds=120)
    def create_feature_importance_plot_fixed(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Crée un graphique d'importance des features avec gestion stricte des noms
        1. Extraction robuste des importances
        2. Gestion stricte des noms de features
        3. Création graphique avec noms lisibles
        4. Gestion des erreurs et cas particuliers
        """
        try:
            model = model_result.get('model')
            feature_names = model_result.get('feature_names', [])
            model_name = model_result.get('model_name', 'Modèle')
            
            if model is None:
                return _create_empty_plot("Modèle non disponible")
            
            # Extraction modèle du pipeline
            actual_model = model
            if hasattr(model, 'named_steps'):
                pipeline_steps = list(model.named_steps.keys())
                if pipeline_steps:
                    model_step = pipeline_steps[-1]
                    actual_model = model.named_steps[model_step]
                    logger.debug(f"Modèle extrait: {model_step}")
            
            # Extraction importances
            importances = None
            method_used = ""
            
            if hasattr(actual_model, 'feature_importances_'):
                importances = actual_model.feature_importances_
                method_used = "Feature Importances"
            elif hasattr(actual_model, 'coef_'):
                coef = actual_model.coef_
                if coef.ndim == 1:
                    importances = np.abs(coef)
                elif coef.ndim == 2:
                    importances = np.mean(np.abs(coef), axis=0)
                else:
                    return _create_empty_plot("Format coefficients invalide")
                method_used = "Coefficients"
            else:
                return _create_empty_plot("Importance non disponible")
            
            # Validation
            if importances is None or len(importances) == 0:
                return _create_empty_plot("Importances vides")
            
            if np.all(importances == 0):
                return _create_empty_plot("Toutes les importances sont nulles")
            
            if np.any(np.isnan(importances)):
                logger.warning("⚠️ NaN détectés, remplacement par 0")
                importances = np.nan_to_num(importances, nan=0.0)
            
            # 🎯 GESTION STRICTE DES NOMS DE FEATURES
            n_importances = len(importances)
            
            if not feature_names:
                logger.warning(f"⚠️ feature_names vide, génération automatique pour {n_importances} features")
                feature_names = [f'Feature_{i}' for i in range(n_importances)]
            
            elif len(feature_names) != n_importances:
                logger.warning(
                    f"⚠️ Incohérence: {len(feature_names)} noms vs {n_importances} importances"
                )
                
                # Stratégie de récupération
                if len(feature_names) < n_importances:
                    # Ajouter des noms génériques
                    missing_count = n_importances - len(feature_names)
                    feature_names = feature_names + [
                        f'Feature_{i}' for i in range(len(feature_names), n_importances)
                    ]
                    logger.warning(f"   Ajout de {missing_count} noms génériques")
                else:
                    # Tronquer les noms
                    feature_names = feature_names[:n_importances]
                    logger.warning(f"   Troncature à {n_importances} noms")
            
            # Création DataFrame
            try:
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
            except Exception as e:
                logger.error(f"❌ Erreur création DataFrame: {e}")
                return _create_empty_plot(f"Erreur: {str(e)}")
            
            # Tri et limitation
            importance_df = importance_df.sort_values('importance', ascending=True)
            
            max_features = 20
            top_n = min(max_features, len(importance_df))
            importance_df = importance_df.tail(top_n)
            
            if importance_df.empty:
                return _create_empty_plot("DataFrame vide")
            
            # Création graphique avec NOMS LISIBLES
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],  # ✅ NOMS RÉELS
                orientation='h',
                marker=dict(
                    color=importance_df['importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f"{imp:.4f}" for imp in importance_df['importance']],
                textposition='auto',
                textfont=dict(size=10, color='white'),
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Top {top_n} Features - {model_name}<br><sub>{method_used}</sub>",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=max(400, top_n * 30),
                template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
                margin=dict(l=200, r=100, t=100, b=50),  # ✅ Marge gauche pour noms longs
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(tickfont=dict(size=11))  # ✅ Taille police lisible
            )
            
            logger.info(f"✅ Feature importance créé: {model_name}, {top_n} features")
            return fig
        
        except Exception as e:
            logger.error(f"❌ Feature importance échoué: {e}", exc_info=True)
            return _create_empty_plot(f"Erreur: {str(e)[:100]}")
    

    @monitor_operation
    @timeout(seconds=180)
    def create_shap_analysis(self, model_result: Dict[str, Any], max_samples: int = 1000) -> go.Figure:
        """Crée une analyse SHAP des features - VERSION CORRIGÉE"""
        if not SHAP_AVAILABLE:
            return _create_empty_plot("SHAP non disponible. Installez avec: pip install shap")
        
        try:
            model = _safe_get(model_result, ['model'])
            X_sample = _safe_get(model_result, ['X_sample'])  
            feature_names = _safe_get(model_result, ['feature_names'], [])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')
            
            if model is None or X_sample is None:
                return _create_empty_plot("Données manquantes pour l'analyse SHAP")
            
            # Conversion robuste des données
            try:
                X_sample = np.array(X_sample)
                if X_sample.size == 0:
                    return _create_empty_plot("Données X_sample vides")
            except Exception as e:
                return _create_empty_plot(f"Erreur conversion données: {str(e)}")
            
            # Échantillonnage pour les performances
            n_samples = min(max_samples, len(X_sample))
            if n_samples < 2:
                return _create_empty_plot("Trop peu d'échantillons pour l'analyse SHAP")
                
            X_shap = X_sample[:n_samples]
            
            # Extraction du modèle du pipeline
            actual_model = model
            if hasattr(model, 'named_steps'):
                pipeline_steps = list(model.named_steps.keys())
                if pipeline_steps:
                    actual_model = model.named_steps[pipeline_steps[-1]]
            
            # Vérification que le modèle peut faire des prédictions
            if not hasattr(actual_model, 'predict') and not hasattr(actual_model, 'predict_proba'):
                return _create_empty_plot("Modèle ne supporte pas les prédictions")
            
            # Calcul des valeurs SHAP
            explainer = None
            shap_values = None
            
            try:
                # Essayer TreeExplainer pour les modèles tree-based
                if hasattr(actual_model, 'feature_importances_') or hasattr(actual_model, 'tree_'):
                    try:
                        explainer = shap.TreeExplainer(actual_model)
                        shap_values = explainer.shap_values(X_shap)
                    except Exception as tree_error:
                        logger.warning(f"TreeExplainer échoué: {str(tree_error)}")
                        # Continuer avec d'autres explainers
                        
                # Essayer LinearExplainer pour les modèles linéaires
                if shap_values is None and hasattr(actual_model, 'coef_'):
                    try:
                        explainer = shap.LinearExplainer(actual_model, X_shap)
                        shap_values = explainer.shap_values(X_shap)
                    except Exception as linear_error:
                        logger.warning(f"LinearExplainer échoué: {str(linear_error)}")
                
                # Fallback KernelExplainer
                if shap_values is None:
                    try:
                        background = shap.sample(X_shap, min(10, len(X_shap)))
                        if hasattr(actual_model, 'predict_proba'):
                            explainer = shap.KernelExplainer(actual_model.predict_proba, background)
                        else:
                            explainer = shap.KernelExplainer(actual_model.predict, background)
                        shap_values = explainer.shap_values(X_shap)
                    except Exception as kernel_error:
                        logger.error(f"KernelExplainer échoué: {str(kernel_error)}")
                        return _create_empty_plot(f"Erreur calcul SHAP: {str(kernel_error)[:100]}")
                
                # Gestion des formats de sortie SHAP
                if shap_values is None:
                    return _create_empty_plot("Impossible de calculer les valeurs SHAP")
                    
                if isinstance(shap_values, list):
                    # Prendre la première classe pour la classification binaire
                    shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
                
                return self._create_shap_summary_plot(shap_values, X_shap, feature_names, model_name)
                
            except Exception as shap_error:
                logger.error(f"Erreur calcul SHAP: {str(shap_error)}")
                return _create_empty_plot(f"Erreur calcul SHAP: {str(shap_error)[:100]}")
            
        except Exception as e:
            logger.error(f"Analyse SHAP échouée: {str(e)}")
            return _create_empty_plot(f"Erreur analyse SHAP: {str(e)}")

    def _create_shap_summary_plot(self, shap_values: np.ndarray, X: np.ndarray, 
                                feature_names: List[str], model_name: str) -> go.Figure:
        """Crée un graphique summary SHAP personnalisé"""
        try:
            if len(shap_values.shape) != 2:
                return _create_empty_plot("Format de valeurs SHAP incorrect")
                
            # Calcul de l'importance moyenne absolue
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Noms de features par défaut si manquants
            if not feature_names or len(feature_names) != shap_values.shape[1]:
                feature_names = [f'Feature_{i}' for i in range(shap_values.shape[1])]
            
            # Sélection des top features
            top_n = min(15, len(mean_abs_shap))
            top_indices = np.argsort(mean_abs_shap)[-top_n:]
            
            fig = go.Figure()
            
            for idx, feature_idx in enumerate(top_indices):
                feature_name = feature_names[feature_idx]
                shap_vals = shap_values[:, feature_idx]
                feature_vals = X[:, feature_idx]
                
                # Normalisation des valeurs de feature pour la couleur
                if len(np.unique(feature_vals)) > 1:
                    norm_vals = (feature_vals - np.min(feature_vals)) / (np.max(feature_vals) - np.min(feature_vals))
                else:
                    norm_vals = np.zeros_like(feature_vals)
                
                fig.add_trace(go.Scatter(
                    x=shap_vals,
                    y=[idx] * len(shap_vals),
                    mode='markers',
                    marker=dict(
                        color=norm_vals,
                        colorscale='RdYlBu',
                        size=6,
                        opacity=0.6,
                        line=dict(width=0.5, color='white'),
                        colorbar=dict(title="Valeur Feature<br>(normalisée)", x=1.02) 
                    ),
                    name=feature_name,
                    showlegend=False,
                    hovertemplate=(
                        f'<b>{feature_name}</b><br>'
                        f'SHAP: %{{x:.4f}}<br>'
                        f'Valeur: %{{marker.color:.3f}}<extra></extra>'
                    )
                ))
            
            fig.update_layout(
                title=f"SHAP Summary - {model_name}",
                xaxis_title="Valeur SHAP (impact sur la prédiction)",
                yaxis=dict(
                    tickmode='array',
                    tickvals=list(range(len(top_indices))),
                    ticktext=[feature_names[i] for i in top_indices],
                    title="Features"
                ),
                height=600,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=False,
                margin=dict(l=150, r=100, t=80, b=50)
            )
            
            # Ligne verticale à zéro
            fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            return fig
            
        except Exception as e:
            return _create_empty_plot(f"Erreur création SHAP: {str(e)}")

    @monitor_operation
    @timeout(seconds=120)
    def create_confusion_matrix(self, model_result: Dict[str, Any]) -> go.Figure:
        """Crée une matrice de confusion"""
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour la matrice de confusion")

            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données de test manquantes")

            # Conversion des données
            if isinstance(y_test, (pd.Series, pd.DataFrame)):
                y_test = y_test.values
            y_test = np.array(y_test).ravel()

            # Prédictions
            y_pred = model.predict(X_test)

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            
            # Noms des classes
            unique_labels = np.unique(np.concatenate([y_test, y_pred]))
            class_names = [str(label) for label in unique_labels]

            # Création du heatmap
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=class_names,
                y=class_names,
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12},
                colorscale='Blues',
                hovertemplate=(
                    'Classe Réelle: %{y}<br>'
                    'Classe Prédite: %{x}<br>'
                    'Nombre: %{z}<extra></extra>'
                )
            ))

            fig.update_layout(
                title=f"Matrice de Confusion - {model_name}",
                xaxis_title="Classe Prédite",
                yaxis_title="Classe Réelle",
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                annotations=[
                    dict(
                        x=xi, y=yi, text=str(val),
                        xref='x1', yref='y1',
                        font=dict(color='white' if val > cm.max() / 2 else 'black'),
                        showarrow=False
                    ) for yi, row in enumerate(cm) for xi, val in enumerate(row)
                ]
            )

            logger.info("Matrice de confusion créée", {"model": model_name})
            return fig

        except Exception as e:
            logger.error(f"Matrice de confusion échouée: {str(e)}")
            return _create_empty_plot(f"Erreur matrice de confusion: {str(e)}")

    @monitor_operation
    @timeout(seconds=120)
    def create_roc_curve(self, model_result: Dict[str, Any]) -> go.Figure:
        """Crée une courbe ROC"""
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour la courbe ROC")

            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données de test manquantes")

            if not hasattr(model, 'predict_proba'):
                return _create_empty_plot("Modèle ne supporte pas predict_proba")

            # Conversion des données
            if isinstance(y_test, (pd.Series, pd.DataFrame)):
                y_test = y_test.values
            y_test = np.array(y_test).ravel()

            # Probabilités prédites
            y_score = model.predict_proba(X_test)

            # Courbe ROC (binary ou multi-class)
            if y_score.shape[1] == 2:
                # Cas binaire
                fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
                roc_auc = auc(fpr, tpr)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    line=dict(color='#2ecc71', width=3),
                    name=f'ROC (AUC = {roc_auc:.3f})',
                    hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
                ))
                
            else:
                # Cas multi-class
                fig = go.Figure()
                n_classes = y_score.shape[1]
                
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_test == i, y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'Classe {i} (AUC = {roc_auc:.3f})',
                        hovertemplate=f'Classe {i}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>'
                    ))

            # Ligne diagonale de référence
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Aléatoire',
                showlegend=False
            ))

            fig.update_layout(
                title=f"Courbe ROC - {model_name}",
                xaxis_title="Taux de Faux Positifs (FPR)",
                yaxis_title="Taux de Vrais Positifs (TPR)",
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=True
            )

            logger.info("Courbe ROC créée", {"model": model_name})
            return fig

        except Exception as e:
            logger.error(f"Courbe ROC échouée: {str(e)}")
            return _create_empty_plot(f"Erreur courbe ROC: {str(e)}")

    @monitor_operation
    @timeout(seconds=120)
    def create_cluster_visualization(self, model_result: Dict[str, Any]) -> go.Figure:
        """Visualisation 2D des clusters"""
        try:
            X = _safe_get(model_result, ['X_sample'])
            labels = _safe_get(model_result, ['labels'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')
            
            if X is None or labels is None:
                return _create_empty_plot("Données manquantes pour la visualisation")
            
            X = np.array(X)
            labels = np.array(labels)
            
            # Filtrage des données valides
            valid_mask = ~np.isnan(labels)
            if not np.any(valid_mask):
                return _create_empty_plot("Aucune donnée valide")
                
            X = X[valid_mask]
            labels = labels[valid_mask]
            
            # Réduction de dimension si nécessaire
            if X.shape[1] > 2 and SKLEARN_AVAILABLE:
                try:
                    # Essai PCA d'abord
                    pca = PCA(n_components=2, random_state=TRAINING_CONSTANTS["RANDOM_STATE"])
                    X_reduced = pca.fit_transform(X)
                    x_label = f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)"
                    y_label = f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"
                except Exception:
                    # Fallback vers les deux premières features
                    X_reduced = X[:, :2]
                    x_label = "Feature 1"
                    y_label = "Feature 2"
            else:
                X_reduced = X[:, :2] if X.shape[1] >= 2 else X
                x_label = "Feature 1"
                y_label = "Feature 2" if X.shape[1] >= 2 else "Feature 1"
            
            unique_labels = np.unique(labels)
            colors = _generate_color_palette(len(unique_labels))
            
            fig = go.Figure()
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                if np.sum(mask) == 0:
                    continue
                    
                if label == -1:
                    # Points de bruit
                    color = 'gray'
                    name = 'Bruit'
                    size = 6
                    opacity = 0.4
                else:
                    color = colors[i % len(colors)]
                    name = f'Cluster {int(label)}'
                    size = 8
                    opacity = 0.7
                
                fig.add_trace(go.Scatter(
                    x=X_reduced[mask, 0], 
                    y=X_reduced[mask, 1] if X_reduced.shape[1] > 1 else np.zeros(np.sum(mask)),
                    mode='markers',
                    name=name,
                    marker=dict(
                        color=color,
                        size=size,
                        line=dict(width=0.5, color='white'),
                        opacity=opacity
                    ),
                    hovertemplate=(
                        f'<b>{name}</b><br>'
                        f'X: %{{x:.2f}}<br>'
                        f'Y: %{{y:.2f}}<extra></extra>'
                    )
                ))
            
            fig.update_layout(
                title=f"Visualisation des Clusters - {model_name}",
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Visualisation clusters échouée: {str(e)}")
            return _create_empty_plot(f"Erreur visualisation clusters: {str(e)}")

    @monitor_operation
    @timeout(seconds=90)
    def create_silhouette_analysis(self, model_result: Dict[str, Any]) -> go.Figure:
        """Analyse silhouette pour le clustering"""
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis pour l'analyse silhouette")
                
            X = _safe_get(model_result, ['X_sample'])
            labels = _safe_get(model_result, ['labels'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')
            
            if X is None or labels is None:
                return _create_empty_plot("Données manquantes pour l'analyse silhouette")
            
            X = np.array(X)
            labels = np.array(labels)
            
            # Filtrage des données valides
            valid_mask = ~np.isnan(labels) & (labels != -1)
            if not np.any(valid_mask):
                return _create_empty_plot("Aucune donnée valide")
                
            X = X[valid_mask]
            labels = labels[valid_mask]
            
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return _create_empty_plot("Au moins 2 clusters requis")
            
            # Calcul des scores silhouette
            silhouette_vals = silhouette_samples(X, labels)
            avg_score = silhouette_score(X, labels)
            
            fig = go.Figure()
            y_lower = 10
            
            for i, label in enumerate(unique_labels):
                cluster_sil = silhouette_vals[labels == label]
                cluster_sil.sort()
                
                cluster_size = len(cluster_sil)
                y_upper = y_lower + cluster_size
                
                color = _generate_color_palette(len(unique_labels))[i]
                
                fig.add_trace(go.Scatter(
                    x=cluster_sil,
                    y=np.arange(y_lower, y_upper),
                    mode='lines',
                    line=dict(width=2, color=color),
                    name=f'Cluster {int(label)} ({cluster_size} pts)',
                    fill='tozerox',
                    fillcolor=color,
                    opacity=0.7,
                    hovertemplate=(
                        f'Cluster {int(label)}<br>'
                        f'Score Silhouette: %{{x:.3f}}<extra></extra>'
                    )
                ))
                
                y_lower = y_upper + 10
            
            # Ligne du score moyen
            fig.add_vline(
                x=avg_score, 
                line_dash="dash", 
                line_color="red", 
                line_width=3,
                annotation_text=f"Score moyen: {avg_score:.3f}",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title=f"Analyse Silhouette - {model_name}",
                xaxis_title="Coefficient de Silhouette",
                yaxis_title="Échantillons (par cluster)",
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=True,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Analyse silhouette échouée: {str(e)}")
            return _create_empty_plot(f"Erreur analyse silhouette: {str(e)}")

    @monitor_operation
    @timeout(seconds=60)
    def create_residuals_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """Graphique des résidus pour la régression"""
        try:
            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données de test manquantes")

            # Conversion des données
            X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
            y_test = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test

            # Prédictions et résidus
            y_pred = pd.Series(model.predict(X_test))
            residuals = y_test - y_pred

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_pred,
                y=residuals,
                mode='markers',
                marker=dict(
                    color='#e74c3c', 
                    size=8, 
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                name='Résidus',
                hovertemplate=(
                    'Prédiction: %{x:.3f}<br>'
                    'Résidu: %{y:.3f}<extra></extra>'
                )
            ))

            # Ligne horizontale à zéro
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.8)

            fig.update_layout(
                title=f"Analyse des Résidus - {model_name}",
                xaxis_title="Valeurs Prédites",
                yaxis_title="Résidus (Réel - Prédit)",
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=True
            )

            return fig

        except Exception as e:
            logger.error(f"Graphique résidus échoué: {str(e)}")
            return _create_empty_plot(f"Erreur graphique résidus: {str(e)}")

    @monitor_operation
    @timeout(seconds=60)
    def create_predicted_vs_actual(self, model_result: Dict[str, Any]) -> go.Figure:
        """Graphique prédictions vs valeurs réelles"""
        try:
            model = _safe_get(model_result, ['model'])
            X_test = _safe_get(model_result, ['X_test'])
            y_test = _safe_get(model_result, ['y_test'])
            model_name = _safe_get(model_result, ['model_name'], 'Modèle')

            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données de test manquantes")

            # Conversion des données
            X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test
            y_test = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test

            # Prédictions
            y_pred = pd.Series(model.predict(X_test))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                marker=dict(
                    color='#2ecc71', 
                    size=8, 
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                name='Prédictions',
                hovertemplate=(
                    'Réel: %{x:.3f}<br>'
                    'Prédit: %{y:.3f}<extra></extra>'
                )
            ))

            # Ligne y=x
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='y = x',
                showlegend=False
            ))

            fig.update_layout(
                title=f"Prédictions vs Réelles - {model_name}",
                xaxis_title="Valeurs Réelles",
                yaxis_title="Valeurs Prédites",
                height=500,
                template=VISUALIZATION_CONSTANTS["PLOTLY_TEMPLATE"],
                showlegend=True
            )

            return fig

        except Exception as e:
            logger.error(f"Graphique prédictions vs réelles échoué: {str(e)}")
            return _create_empty_plot(f"Erreur graphique prédictions: {str(e)}")

    @monitor_operation
    @timeout(seconds=120)
    def create_learning_curve(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Crée une courbe d'apprentissage pour le modèle donné
        ️ Gère les erreurs et retourne un graphique vide en cas d'échec
        ️ Utilise des constantes de configuration pour les paramètres
        ️ Supporte les modèles de classification et régression
        ️ Optimisé pour la performance et la robustesse
        ️ Journalisation détaillée des étapes et erreurs
        """
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis")
            
            model = model_result.get('model')
            X_train = model_result.get('X_train')
            y_train = model_result.get('y_train')
            model_name = model_result.get('model_name', 'Modèle')
            
            if model is None or X_train is None or y_train is None:
                return _create_empty_plot("Données d'entraînement manquantes")
            
            # Conversion robuste
            if isinstance(X_train, np.ndarray):
                X_train = pd.DataFrame(X_train)
            
            if isinstance(y_train, (pd.DataFrame, pd.Series)):
                y_train = y_train.values
            
            y_train = np.array(y_train).ravel()
            
            # Validation taille
            if len(X_train) != len(y_train):
                return _create_empty_plot(
                    f"Incohérence dimensions: X_train={len(X_train)}, y_train={len(y_train)}"
                )
            
            if len(X_train) < 10:
                return _create_empty_plot("Trop peu d'échantillons (< 10)")
            
            # Calcul courbe d'apprentissage
            train_sizes = np.linspace(0.1, 1.0, 10)
            cv_folds = min(5, TRAINING_CONSTANTS.get("CV_FOLDS", 5))
            random_state = TRAINING_CONSTANTS.get("RANDOM_STATE", 42)
            
            try:
                train_sizes, train_scores, test_scores = learning_curve(
                    model,
                    X_train,
                    y_train,
                    train_sizes=train_sizes,
                    cv=cv_folds,
                    n_jobs=1,  # Séquentiel pour éviter conflits
                    random_state=random_state,
                    error_score='raise'
                )
            except Exception as lc_error:
                logger.error(f"❌ learning_curve échouée: {lc_error}")
                return _create_empty_plot(f"Erreur calcul: {str(lc_error)[:100]}")
            
            # Calcul statistiques
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            
            # Création graphique
            fig = go.Figure()
            
            # Courbe train
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_scores_mean,
                mode='lines+markers',
                name='Score entraînement',
                line=dict(color='#3498db', width=3),
                marker=dict(size=8),
                error_y=dict(
                    type='data',
                    array=train_scores_std,
                    visible=True,
                    color='#3498db',
                    thickness=1.5,
                    width=3
                ),
                hovertemplate='Train: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>'
            ))
            
            # Courbe test
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=test_scores_mean,
                mode='lines+markers',
                name='Score validation',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=8),
                error_y=dict(
                    type='data',
                    array=test_scores_std,
                    visible=True,
                    color='#e74c3c',
                    thickness=1.5,
                    width=3
                ),
                hovertemplate='Val: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Courbe d'Apprentissage - {model_name}",
                xaxis_title="Nombre d'échantillons d'entraînement",
                yaxis_title="Score",
                height=500,
                template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            logger.info(f"✅ Learning curve créée: {model_name}")
            return fig
        
        except Exception as e:
            logger.error(f"❌ Learning curve échouée: {e}", exc_info=True)
            return _create_empty_plot(f"Erreur: {str(e)[:100]}")


    @monitor_operation
    @timeout(seconds=120)
    def create_precision_recall_curve(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Courbe Precision-Recall pour classification
        Utile pour classes déséquilibrées
        """
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis")
            
            model = model_result.get('model')
            X_test = model_result.get('X_test')
            y_test = model_result.get('y_test')
            model_name = model_result.get('model_name', 'Modèle')
            
            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données manquantes")
            
            if not hasattr(model, 'predict_proba'):
                return _create_empty_plot("predict_proba non disponible")
            
            # Conversion
            if isinstance(y_test, (pd.Series, pd.DataFrame)):
                y_test = y_test.values
            y_test = np.array(y_test).ravel()
            
            # Probabilités
            y_score = model.predict_proba(X_test)
            
            fig = go.Figure()
            
            if y_score.shape[1] == 2:
                # Cas binaire
                precision, recall, _ = precision_recall_curve(y_test, y_score[:, 1])
                pr_auc = auc(recall, precision)
                
                fig.add_trace(go.Scatter(
                    x=recall,
                    y=precision,
                    mode='lines',
                    line=dict(color='#667eea', width=3),
                    name=f'PR Curve (AUC = {pr_auc:.3f})',
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.2)',
                    hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'
                ))
            else:
                # Multi-classe
                for i in range(y_score.shape[1]):
                    precision, recall, _ = precision_recall_curve(y_test == i, y_score[:, i])
                    pr_auc = auc(recall, precision)
                    
                    fig.add_trace(go.Scatter(
                        x=recall,
                        y=precision,
                        mode='lines',
                        name=f'Classe {i} (AUC = {pr_auc:.3f})',
                        hovertemplate=f'Classe {i}<br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>'
                    ))
            
            fig.update_layout(
                title=f"Courbe Precision-Recall - {model_name}",
                xaxis_title="Recall",
                yaxis_title="Precision",
                height=500,
                template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            logger.info(f"✅ PR curve créée: {model_name}")
            return fig
        
        except Exception as e:
            logger.error(f"❌ PR curve échouée: {e}", exc_info=True)
            return _create_empty_plot(f"Erreur: {str(e)[:100]}")
        

    @monitor_operation
    @timeout(seconds=60)
    def create_error_distribution(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Distribution des erreurs pour régression
        """
        try:
            model = model_result.get('model')
            X_test = model_result.get('X_test')
            y_test = model_result.get('y_test')
            model_name = model_result.get('model_name', 'Modèle')
            
            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données manquantes")
            
            # Prédictions
            y_pred = model.predict(X_test)
            
            # Erreurs
            errors = y_test - y_pred
            
            # Création graphique
            fig = go.Figure()
            
            # Histogramme des erreurs
            fig.add_trace(go.Histogram(
                x=errors,
                nbinsx=50,
                marker=dict(
                    color='#667eea',
                    line=dict(color='white', width=1)
                ),
                opacity=0.7,
                name='Erreurs',
                hovertemplate='Erreur: %{x:.2f}<br>Fréquence: %{y}<extra></extra>'
            ))
            
            # Ligne de moyenne
            mean_error = np.mean(errors)
            fig.add_vline(
                x=mean_error,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"Moyenne: {mean_error:.3f}",
                annotation_position="top right"
            )
            
            # Ligne médiane
            median_error = np.median(errors)
            fig.add_vline(
                x=median_error,
                line_dash="dash",
                line_color="green",
                line_width=2,
                annotation_text=f"Médiane: {median_error:.3f}",
                annotation_position="bottom right"
            )
            
            fig.update_layout(
                title=f"Distribution des Erreurs - {model_name}",
                xaxis_title="Erreur (Réel - Prédit)",
                yaxis_title="Fréquence",
                height=500,
                template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            logger.info(f"✅ Distribution erreurs créée: {model_name}")
            return fig
        
        except Exception as e:
            logger.error(f"❌ Distribution erreurs échouée: {e}", exc_info=True)
            return _create_empty_plot(f"Erreur: {str(e)[:100]}")


    @monitor_operation
    @timeout(seconds=120)
    def create_calibration_plot(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Courbe de calibration pour classification
        Montre si les probabilités prédites sont fiables
        """
        try:
            if not SKLEARN_AVAILABLE:
                return _create_empty_plot("scikit-learn requis")
            
            from sklearn.calibration import calibration_curve
            
            model = model_result.get('model')
            X_test = model_result.get('X_test')
            y_test = model_result.get('y_test')
            model_name = model_result.get('model_name', 'Modèle')
            
            if model is None or X_test is None or y_test is None:
                return _create_empty_plot("Données manquantes")
            
            if not hasattr(model, 'predict_proba'):
                return _create_empty_plot("predict_proba non disponible")
            
            # Conversion
            if isinstance(y_test, (pd.Series, pd.DataFrame)):
                y_test = y_test.values
            y_test = np.array(y_test).ravel()
            
            # Probabilités (classe positive)
            y_proba = model.predict_proba(X_test)
            
            if y_proba.shape[1] != 2:
                return _create_empty_plot("Calibration uniquement pour classification binaire")
            
            y_proba = y_proba[:, 1]
            
            # Calcul courbe de calibration
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')
            
            fig = go.Figure()
            
            # Courbe de calibration
            fig.add_trace(go.Scatter(
                x=prob_pred,
                y=prob_true,
                mode='lines+markers',
                line=dict(color='#667eea', width=3),
                marker=dict(size=10),
                name='Calibration',
                hovertemplate='Prédit: %{x:.3f}<br>Observé: %{y:.3f}<extra></extra>'
            ))
            
            # Diagonale parfaite
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='gray', dash='dash', width=2),
                name='Calibration parfaite',
                showlegend=True
            ))
            
            fig.update_layout(
                title=f"Courbe de Calibration - {model_name}",
                xaxis_title="Probabilité Moyenne Prédite",
                yaxis_title="Fraction de Positifs",
                height=500,
                template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            logger.info(f"✅ Calibration plot créé: {model_name}")
            return fig
        
        except Exception as e:
            logger.error(f"❌ Calibration plot échoué: {e}", exc_info=True)
            return _create_empty_plot(f"Erreur: {str(e)[:100]}")
        
    
    @monitor_operation
    @timeout(seconds=120)
    def create_feature_correlation_matrix(self, model_result: Dict[str, Any]) -> go.Figure:
        """
        Matrice de corrélation des features
        """
        try:
            X_train = model_result.get('X_train')
            feature_names = model_result.get('feature_names', [])
            model_name = model_result.get('model_name', 'Modèle')
            
            if X_train is None:
                return _create_empty_plot("Données X_train manquantes")
            
            # Conversion DataFrame
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train, columns=feature_names if feature_names else None)
            
            # Sélection features numériques uniquement
            numeric_features = X_train.select_dtypes(include=['number'])
            
            if numeric_features.empty:
                return _create_empty_plot("Aucune feature numérique")
            
            # Limitation nombre de features
            max_features = 15
            if len(numeric_features.columns) > max_features:
                numeric_features = numeric_features.iloc[:, :max_features]
            
            # Calcul corrélation
            corr_matrix = numeric_features.corr()
            
            # Création heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Corrélation"),
                hovertemplate='%{x} vs %{y}<br>Corrélation: %{z:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Matrice de Corrélation - {model_name}",
                xaxis_title="Features",
                yaxis_title="Features",
                height=600,
                template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
                xaxis=dict(tickangle=-45),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            logger.info(f"✅ Matrice corrélation créée: {model_name}")
            return fig
        
        except Exception as e:
            logger.error(f"❌ Matrice corrélation échouée: {e}", exc_info=True)
            return _create_empty_plot(f"Erreur: {str(e)[:100]}")
        
    
    @monitor_operation
    @timeout(seconds=120)
    def create_radar_comparison(self) -> go.Figure:
        """
        Graphique radar pour comparaison multi-modèles
        """
        try:
            successful_models = self.validation_result["successful_models"]
            
            if not successful_models or len(successful_models) < 2:
                return _create_empty_plot("Au moins 2 modèles requis")
            
            task_type = self.validation_result["task_type"]
            
            # Sélection métriques selon task_type
            if task_type == 'classification':
                metric_keys = ['accuracy', 'precision', 'recall', 'f1_score']
                metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            elif task_type == 'regression':
                metric_keys = ['r2']
                metric_labels = ['R² Score']
                
                # Pour régression, on ne peut pas faire un radar pertinent
                return _create_empty_plot("Radar chart non pertinent pour régression")
            elif task_type == 'clustering':
                metric_keys = ['silhouette_score']
                metric_labels = ['Silhouette']
                return _create_empty_plot("Radar chart non pertinent pour clustering")
            else:
                return _create_empty_plot(f"Task type {task_type} non supporté")
            
            fig = go.Figure()
            
            colors = _generate_color_palette(len(successful_models))
            
            for i, model in enumerate(successful_models[:5]):  # Limiter à 5 modèles
                model_name = model.get('model_name', f'Modèle_{i}')
                metrics = model.get('metrics', {})
                
                values = [metrics.get(key, 0) for key in metric_keys]
                
                # Fermer le radar
                values_closed = values + [values[0]]
                labels_closed = metric_labels + [metric_labels[0]]
                
                fig.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=labels_closed,
                    fill='toself',
                    name=model_name,
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{model_name}</b><br>%{{theta}}: %{{r:.3f}}<extra></extra>'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        showgrid=True,
                        gridcolor='rgba(0,0,0,0.1)'
                    )
                ),
                title="Comparaison Radar des Modèles",
                showlegend=True,
                height=600,
                template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            logger.info("✅ Radar comparison créé")
            return fig
        
        except Exception as e:
            logger.error(f"❌ Radar comparison échoué: {e}", exc_info=True)
            return _create_empty_plot(f"Erreur: {str(e)[:100]}")
        
    
    @monitor_operation
    @timeout(seconds=60)
    def create_time_vs_performance_plot(self) -> go.Figure:
        """
        Scatter plot: Temps d'entraînement vs Performance
        Aide à choisir le meilleur compromis
        """
        try:
            successful_models = self.validation_result["successful_models"]
            
            if not successful_models:
                return _create_empty_plot("Aucun modèle disponible")
            
            task_type = self.validation_result["task_type"]
            
            # Métrique principale
            metric_key = {
                'classification': 'accuracy',
                'regression': 'r2',
                'clustering': 'silhouette_score'
            }.get(task_type, 'accuracy')
            
            metric_label = {
                'classification': 'Accuracy',
                'regression': 'R² Score',
                'clustering': 'Silhouette'
            }.get(task_type, 'Score')
            
            # Extraction données
            model_names = []
            training_times = []
            performances = []
            
            for model in successful_models:
                model_names.append(model.get('model_name', 'Unknown'))
                training_times.append(model.get('training_time', 0))
                performances.append(model.get('metrics', {}).get(metric_key, 0))
            
            # Création scatter
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=training_times,
                y=performances,
                mode='markers+text',
                text=model_names,
                textposition='top center',
                marker=dict(
                    size=15,
                    color=performances,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=metric_label),
                    line=dict(color='white', width=2)
                ),
                hovertemplate=(
                    '<b>%{text}</b><br>'
                    f'Temps: %{{x:.2f}}s<br>'
                    f'{metric_label}: %{{y:.3f}}<extra></extra>'
                )
            ))
            
            fig.update_layout(
                title="Compromis Temps d'Entraînement vs Performance",
                xaxis_title="Temps d'Entraînement (secondes)",
                yaxis_title=metric_label,
                height=500,
                template=VISUALIZATION_CONSTANTS.get("PLOTLY_TEMPLATE", "plotly_white"),
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            logger.info("✅ Time vs performance plot créé")
            return fig
        
        except Exception as e:
            logger.error(f"❌ Time vs performance plot échoué: {e}", exc_info=True)
            return _create_empty_plot(f"Erreur: {str(e)[:100]}")


    @monitor_operation
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Retourne un DataFrame de comparaison des modèles"""
        comparison_data = []
        
        for result in self.ml_results:
            model_name = _safe_get(result, ['model_name'], 'Unknown')
            training_time = _safe_get(result, ['training_time'], 0)
            metrics = _safe_get(result, ['metrics'], {})
            
            has_error = _safe_get(metrics, ['error']) is not None
            
            row = {
                'Modèle': model_name,
                'Statut': '❌ Échec' if has_error else '✅ Succès',
                'Temps (s)': f"{training_time:.2f}" if isinstance(training_time, (int, float)) else 'N/A'
            }
            
            if not has_error:
                task_type = self.validation_result["task_type"]
                
                if task_type == 'classification':
                    row.update({
                        'Accuracy': format_metric_value(_safe_get(metrics, ['accuracy'])),
                        'F1-Score': format_metric_value(_safe_get(metrics, ['f1'])),
                        'Precision': format_metric_value(_safe_get(metrics, ['precision'])),
                        'Recall': format_metric_value(_safe_get(metrics, ['recall'])),
                        'AUC': format_metric_value(_safe_get(metrics, ['auc']))
                    })
                elif task_type == 'regression':
                    row.update({
                        'R²': format_metric_value(_safe_get(metrics, ['r2'])),
                        'MAE': format_metric_value(_safe_get(metrics, ['mae'])),
                        'RMSE': format_metric_value(_safe_get(metrics, ['rmse'])),
                        'MSE': format_metric_value(_safe_get(metrics, ['mse']))
                    })
                elif task_type == 'clustering':
                    row.update({
                        'Silhouette': format_metric_value(_safe_get(metrics, ['silhouette_score'])),
                        'N_Clusters': format_metric_value(_safe_get(metrics, ['n_clusters'])),
                        'DB_Index': format_metric_value(_safe_get(metrics, ['davies_bouldin_score']))
                    })
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Nettoyage des types de données
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        logger.info("DataFrame comparaison généré", {"n_models": len(df)})
        return df

    @monitor_operation
    def get_export_data(self) -> Dict[str, Any]:
        """Prépare les données pour l'export"""
        try:
            models_data = []
            
            for result in self.validation_result["successful_models"]:
                model_data = {
                    "model_name": _safe_get(result, ["model_name"], "Unknown"),
                    "task_type": self.validation_result["task_type"],
                    "training_time": _safe_get(result, ["training_time"], 0),
                    "metrics": {}
                }
                
                metrics = _safe_get(result, ["metrics"], {})
                
                # Filtrage des métriques non-sérialisables
                for key, value in metrics.items():
                    if not isinstance(value, (dict, list, np.ndarray)) and key != 'error':
                        try:
                            if isinstance(value, (np.integer, np.floating)):
                                model_data["metrics"][key] = float(value)
                            else:
                                model_data["metrics"][key] = value
                        except (TypeError, ValueError):
                            continue
                
                models_data.append(model_data)
            
            export_data = {
                "export_timestamp": time.time(),
                "export_date": datetime.now().isoformat(),
                "task_type": self.validation_result["task_type"],
                "best_model": self.validation_result["best_model"],
                "total_models": len(self.validation_result["successful_models"]),
                "failed_models": len(self.validation_result["failed_models"]),
                "success_rate": len(self.validation_result["successful_models"]) / self.validation_result["results_count"] * 100 if self.validation_result["results_count"] > 0 else 0,
                "global_statistics": self.validation_result["metrics_summary"],
                "models": models_data,
                "system_info": get_system_metrics(),
                "warnings": self.validation_result["warnings"],
                "errors": self.validation_result["errors"]
            }
            
            logger.info("Données d'export préparées", {
                "n_models": len(models_data),
                "task_type": self.validation_result["task_type"]
            })
            
            gc.collect()
            return export_data
            
        except Exception as e:
            logger.error(f"Préparation données export échouée: {str(e)}")
            return {
                "error": str(e), 
                "export_timestamp": time.time(),
                "task_type": self.validation_result.get("task_type", "unknown")
            }

    def cleanup(self):
        """Nettoie les ressources temporaires"""
        try:
            # Nettoyage du cache
            self._plot_cache.clear()
            
            # Nettoyage des fichiers temporaires
            for file_path in self._temp_dir.glob("*.png"):
                try:
                    file_path.unlink()
                except Exception:
                    pass
            
            logger.info("Ressources nettoyées")
        except Exception as e:
            logger.warning(f"Nettoyage partiellement échoué: {str(e)}")
# Fonctions utilitaires exportées
def create_model_comparison(ml_results: List[Dict[str, Any]]) -> go.Figure:
    """Fonction utilitaire pour créer un graphique de comparaison"""
    visualizer = ModelEvaluationVisualizer(ml_results)
    return visualizer.create_comparison_plot()

def create_feature_importance_plot(model_result: Dict[str, Any]) -> go.Figure:
    """Fonction utilitaire pour créer un graphique d'importance des features"""
    visualizer = ModelEvaluationVisualizer([model_result])
    return visualizer.create_feature_importance_plot(model_result)

# Export des symboles principaux
__all__ = [
    'ModelEvaluationVisualizer',
    'create_model_comparison',
    'create_feature_importance_plot',
    'get_system_metrics',
    'logger'
]

# Point d'entrée pour les tests
if __name__ == "__main__":
    # Exemple d'utilisation
    sample_results = [
        {
            "model_name": "RandomForest",
            "training_time": 10.5,
            "metrics": {
                "accuracy": 0.85,
                "f1": 0.83,
                "precision": 0.84,
                "recall": 0.82
            }
        }
    ]
    
    visualizer = ModelEvaluationVisualizer(sample_results)
    fig = visualizer.create_comparison_plot()
    
    if fig:
        print("Graphique créé avec succès!")
    else:
        print("Échec de création du graphique")
    
    visualizer.cleanup()