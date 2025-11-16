"""
ML Factory Pro - Page Évaluation ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Imports internes
from src.evaluation.model_plots import ModelEvaluationVisualizer
from src.shared.logging import get_logger
from monitoring.state_managers import init, AppPage, STATE
from ui.styles import UIStyles
from helpers.ui_components import UIComponents
from monitoring.decorators import monitor_operation

# Initialisation
STATE = init()
logger = get_logger(__name__)

# Configuration page
st.set_page_config(
    page_title="ML Factory Pro | Évaluation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown(UIStyles.get_main_css(), unsafe_allow_html=True)

st.markdown("""
<style>
    /* Métriques horizontales */
    .metrics-horizontal {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .metric-card-horizontal {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card-horizontal:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.4);
    }
    
    .metric-icon-horizontal {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-value-horizontal {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    .metric-label-horizontal {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Graphiques */
    .plot-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .plot-container-modern {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 📦 DATA CLASSES
# ============================================================================

@dataclass
class ValidationResult:
    is_valid: bool
    successful_models: List[Dict]
    failed_models: List[Dict]
    task_type: str
    best_model: Optional[str]
    errors: List[str]
    warnings: List[str]


# ============================================================================
# 🔧 FONCTIONS UTILITAIRES
# ============================================================================

def get_mlflow_runs_robust() -> List[Dict[str, Any]]:
    """Récupération MLflow multi-sources"""
    mlflow_runs = []
    
    try:
        if hasattr(st.session_state, 'mlflow_runs') and st.session_state.mlflow_runs:
            mlflow_runs = st.session_state.mlflow_runs
            logger.info(f"✅ {len(mlflow_runs)} runs depuis session_state")
            return mlflow_runs
    except Exception as e:
        logger.warning(f"⚠️ session_state.mlflow_runs: {e}")
    
    try:
        if hasattr(STATE, 'mlflow_runs') and STATE.mlflow_runs:
            mlflow_runs = STATE.mlflow_runs
            logger.info(f"✅ {len(mlflow_runs)} runs depuis STATE")
            if not hasattr(st.session_state, 'mlflow_runs'):
                st.session_state.mlflow_runs = mlflow_runs
            return mlflow_runs
    except Exception as e:
        logger.warning(f"⚠️ STATE.mlflow_runs: {e}")
    
    try:
        if hasattr(STATE, 'training') and hasattr(STATE.training, 'mlflow_runs'):
            if STATE.training.mlflow_runs:
                mlflow_runs = STATE.training.mlflow_runs
                logger.info(f"✅ {len(mlflow_runs)} runs depuis STATE.training")
                return mlflow_runs
    except Exception as e:
        logger.warning(f"⚠️ STATE.training.mlflow_runs: {e}")
    
    logger.warning("❌ Aucune source MLflow disponible")
    return []


def validate_training_results(results_data: Any) -> ValidationResult:
    """Validation stricte"""
    successful_models = []
    failed_models = []
    task_type = 'unknown'
    best_model = None
    errors = []
    warnings = []
    
    try:
        if hasattr(results_data, 'results') and hasattr(results_data, 'summary'):
            results_list = results_data.results
            
            for result in results_list:
                if not isinstance(result, dict):
                    continue
                if result.get('success', False):
                    successful_models.append(result)
                else:
                    failed_models.append(result)
            
            if successful_models:
                task_type = successful_models[0].get('task_type', 'unknown')
            
            if hasattr(results_data, 'summary'):
                best_model = results_data.summary.get('best_model')
        
        elif isinstance(results_data, list):
            for result in results_data:
                if not isinstance(result, dict):
                    continue
                if result.get('success', False):
                    successful_models.append(result)
                else:
                    failed_models.append(result)
            
            if successful_models:
                task_type = successful_models[0].get('task_type', 'unknown')
                
                metric_key = {
                    'classification': 'accuracy',
                    'regression': 'r2',
                    'clustering': 'silhouette_score'
                }.get(task_type, 'accuracy')
                
                best = max(
                    successful_models,
                    key=lambda x: x.get('metrics', {}).get(metric_key, -float('inf'))
                )
                best_model = best.get('model_name')
        else:
            errors.append(f"Format non supporté: {type(results_data)}")
    
    except Exception as e:
        errors.append(f"Erreur validation: {str(e)}")
        logger.error(f"❌ Validation: {e}", exc_info=True)
    
    is_valid = len(successful_models) > 0 and len(errors) == 0
    
    return ValidationResult(is_valid, successful_models, failed_models, task_type, best_model, errors, warnings)


# ============================================================================
# 🎨 COMPOSANTS UI
# ============================================================================
def render_metrics_horizontal(validation):
    """Dashboard métriques horizontal"""
    try:
        total = len(validation['successful_models']) + len(validation['failed_models'])
        success_rate = (len(validation['successful_models']) / total * 100) if total > 0 else 0
        
        task_type = validation.get('task_type', 'classification')
        metric_key = {
            'clustering': 'silhouette_score',
            'regression': 'r2',
            'classification': 'accuracy'
        }.get(task_type, 'accuracy')
        
        best_score = max(
            [m.get('metrics', {}).get(metric_key, 0) for m in validation['successful_models']],
            default=0
        )
        
        # Calcul temps moyen
        avg_time = np.mean([
            m.get('training_time', 0) 
            for m in validation['successful_models']
        ]) if validation['successful_models'] else 0
        
        # MÉTRIQUES HORIZONTALES
        st.markdown('<div class="metrics-horizontal">', unsafe_allow_html=True)
        
        metrics_data = [
            {
                'icon': '✅',
                'value': f"{success_rate:.0f}%",
                'label': 'Taux de Réussite',
                'bg': 'linear-gradient(135deg, #28a745 0%, #20c997 100%)'
            },
            {
                'icon': '🏆',
                'value': validation.get('best_model', 'N/A'),
                'label': 'Meilleur Modèle',
                'bg': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
            },
            {
                'icon': '📈',
                'value': f"{best_score:.3f}",
                'label': f'Meilleur Score ({metric_key})',
                'bg': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
            },
            {
                'icon': '⏱️',
                'value': f"{avg_time:.1f}s",
                'label': 'Temps Moyen',
                'bg': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)'
            },
            {
                'icon': '🤖',
                'value': f"{len(validation['successful_models'])}",
                'label': 'Modèles Réussis',
                'bg': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'
            }
        ]
        
        for metric in metrics_data:
            st.markdown(
                f"""
                <div class="metric-card-horizontal" style="background: {metric['bg']};">
                    <div class="metric-icon-horizontal">{metric['icon']}</div>
                    <div class="metric-value-horizontal">{metric['value']}</div>
                    <div class="metric-label-horizontal">{metric['label']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Erreur métriques: {str(e)}")

def create_advanced_comparison(visualizer, validation):
    """Graphique de comparaison avancé"""
    try:
        successful_models = validation['successful_models']
        task_type = validation['task_type']
        
        model_names = [m.get('model_name', 'Unknown') for m in successful_models]
        
        if task_type == 'classification':
            metrics_keys = ['accuracy', 'precision', 'recall', 'f1_score']
            metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        elif task_type == 'regression':
            metrics_keys = ['r2', 'mae', 'rmse']
            metrics_labels = ['R²', 'MAE', 'RMSE']
        else:
            metrics_keys = ['silhouette_score']
            metrics_labels = ['Silhouette']
        
        # Création subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Comparaison Métriques',
                'Temps d\'Entraînement',
                'Distribution Performances',
                'Radar Comparatif'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "scatterpolar"}]
            ]
        )
        
        # 1. Comparaison métriques
        for i, (key, label) in enumerate(zip(metrics_keys[:2], metrics_labels[:2])):
            values = [m.get('metrics', {}).get(key, 0) for m in successful_models]
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=values,
                    name=label,
                    text=[f"{v:.3f}" for v in values],
                    textposition='auto'
                ),
                row=1, col=1
            )
        
        # 2. Temps d'entraînement
        times = [m.get('training_time', 0) for m in successful_models]
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=times,
                name='Temps (s)',
                marker_color='lightblue',
                text=[f"{t:.1f}s" for t in times],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # 3. Distribution performances
        for key, label in zip(metrics_keys[:2], metrics_labels[:2]):
            values = [m.get('metrics', {}).get(key, 0) for m in successful_models]
            fig.add_trace(
                go.Box(
                    y=values,
                    name=label,
                    boxmean='sd'
                ),
                row=2, col=1
            )
        
        # 4. Radar chart
        if len(metrics_keys) >= 3:
            for model in successful_models[:3]:  # Top 3 modèles
                values = [model.get('metrics', {}).get(key, 0) for key in metrics_keys]
                values_closed = values + [values[0]]
                labels_closed = metrics_labels + [metrics_labels[0]]
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=values_closed,
                        theta=labels_closed,
                        fill='toself',
                        name=model.get('model_name', 'Unknown')
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            template="plotly_white",
            title_text="Dashboard Comparatif Avancé"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Erreur graphique avancé: {str(e)}")

def create_mlflow_dashboard(mlflow_runs):
    """Dashboard MLflow enrichi"""
    if not mlflow_runs:
        st.info("ℹ️ Aucun run MLflow disponible")
        return
    
    try:
        st.markdown("### 🔗 Dashboard MLflow")
        
        # Métriques MLflow
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Runs", len(mlflow_runs))
        
        with col2:
            finished = len([r for r in mlflow_runs if r.get('status') == 'FINISHED'])
            st.metric("✅ Terminés", finished)
        
        with col3:
            models = set(r.get('model_name', 'Unknown') for r in mlflow_runs)
            st.metric("🤖 Modèles Uniques", len(models))
        
        with col4:
            avg_time = np.mean([
                r.get('metrics', {}).get('training_time', 0) 
                for r in mlflow_runs
            ])
            st.metric("⏱️ Temps Moyen", f"{avg_time:.1f}s")
        
        # Graphique évolution runs
        st.markdown("#### 📈 Évolution des Performances")
        
        df_runs = pd.DataFrame([
            {
                'model_name': r.get('model_name', 'Unknown'),
                'accuracy': r.get('metrics', {}).get('accuracy', 0),
                'timestamp': r.get('start_time', 0)
            }
            for r in mlflow_runs
        ])
        
        if not df_runs.empty:
            fig = go.Figure()
            
            for model in df_runs['model_name'].unique():
                df_model = df_runs[df_runs['model_name'] == model]
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(df_model))),
                        y=df_model['accuracy'],
                        mode='lines+markers',
                        name=model,
                        line=dict(width=3),
                        marker=dict(size=10)
                    )
                )
            
            fig.update_layout(
                title="Évolution des Scores par Modèle",
                xaxis_title="Run #",
                yaxis_title="Accuracy",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Erreur dashboard MLflow: {str(e)}")


def render_hero_section():
    """Hero section compact"""
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0 1rem 0;">
        <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   font-size: 2.5rem; font-weight: 800; margin: 0;">
            📊 Évaluation ML Pro
        </h1>
        <p style="color: #666; font-size: 1rem; margin-top: 0.5rem;">
            Analyse Complète des Performances
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_metrics_dashboard(validation: ValidationResult):
    """Dashboard 4 métriques compact"""
    try:
        total = len(validation.successful_models) + len(validation.failed_models)
        success_rate = (len(validation.successful_models) / total * 100) if total > 0 else 0
        
        metric_key = {
            'clustering': 'silhouette_score',
            'regression': 'r2',
            'classification': 'accuracy'
        }.get(validation.task_type, 'accuracy')
        
        metric_name = {
            'clustering': 'Silhouette',
            'regression': 'R²',
            'classification': 'Accuracy'
        }.get(validation.task_type, 'Score')
        
        best_score = max(
            [m.get('metrics', {}).get(metric_key, 0) for m in validation.successful_models],
            default=0
        )
        
        color = "#28a745" if success_rate > 80 else "#ffc107" if success_rate > 50 else "#dc3545"
        
        st.markdown('<div class="metrics-compact">', unsafe_allow_html=True)
        
        # Carte 1
        st.markdown(f"""
        <div class="metric-card-compact" style="--card-color: {color};">
            <div class="metric-icon-compact">✅</div>
            <div class="metric-label-compact">Taux de Réussite</div>
            <div class="metric-value-compact" style="color: {color};">{success_rate:.1f}%</div>
            <div class="metric-subtitle-compact">{len(validation.successful_models)}/{total} modèles</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Carte 2
        st.markdown(f"""
        <div class="metric-card-compact" style="--card-color: #667eea;">
            <div class="metric-icon-compact">🏆</div>
            <div class="metric-label-compact">Meilleur Modèle</div>
            <div class="metric-value-compact" style="font-size: 1.5rem;">{validation.best_model or 'N/A'}</div>
            <div class="metric-subtitle-compact">Type: {validation.task_type.title()}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Carte 3
        st.markdown(f"""
        <div class="metric-card-compact" style="--card-color: #17a2b8;">
            <div class="metric-icon-compact">📈</div>
            <div class="metric-label-compact">Meilleur {metric_name}</div>
            <div class="metric-value-compact">{best_score:.3f}</div>
            <div class="metric-subtitle-compact">Score optimal</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Carte 4
        try:
            import psutil # type: ignore
            mem = psutil.virtual_memory()
            mem_pct = mem.percent
            mem_gb = mem.available / (1024**3)
            mem_color = "#28a745" if mem_pct < 70 else "#ffc107" if mem_pct < 85 else "#dc3545"
            
            st.markdown(f"""
            <div class="metric-card-compact" style="--card-color: {mem_color};">
                <div class="metric-icon-compact">💻</div>
                <div class="metric-label-compact">Mémoire Système</div>
                <div class="metric-value-compact">{mem_pct:.1f}%</div>
                <div class="metric-subtitle-compact">{mem_gb:.1f} GB dispo</div>
            </div>
            """, unsafe_allow_html=True)
        except:
            st.markdown("""
            <div class="metric-card-compact" style="--card-color: #28a745;">
                <div class="metric-icon-compact">✅</div>
                <div class="metric-label-compact">Système</div>
                <div class="metric-value-compact">OK</div>
                <div class="metric-subtitle-compact">Opérationnel</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Erreur métriques: {str(e)[:100]}")


def render_model_comparison_chart(validation: ValidationResult):
    """Graphique de comparaison moderne"""
    try:
        model_names = []
        scores = []
        
        metric_label, metric_key = {
            'clustering': ('Score Silhouette', 'silhouette_score'),
            'regression': ('R² Score', 'r2'),
            'classification': ('Accuracy', 'accuracy')
        }.get(validation.task_type, ('Accuracy', 'accuracy'))
        
        for model in validation.successful_models:
            model_names.append(model.get('model_name', 'Unknown'))
            scores.append(model.get('metrics', {}).get(metric_key, 0))
        
        if not model_names:
            st.info("ℹ️ Aucun modèle à comparer")
            return
        
        max_score = max(scores)
        colors = ['#28a745' if score == max_score else '#667eea' for score in scores]
        
        fig = go.Figure(data=[
            go.Bar(
                x=model_names,
                y=scores,
                text=[f'{score:.3f}' for score in scores],
                textposition='auto',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title={
                'text': f"Comparaison des Modèles - {metric_label}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#333', 'family': 'Inter'}
            },
            xaxis=dict(
                title="Modèles",
                tickangle=-45,
                showgrid=False
            ),
            yaxis=dict(
                title=metric_label,
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)'
            ),
            template="plotly_white",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=80, r=40, t=80, b=120)
        )
        
        st.plotly_chart(fig, use_container_width=True, key="comparison_chart_main")
    
    except Exception as e:
        st.error(f"❌ Erreur graphique: {str(e)[:100]}")


def render_summary_table(validation: ValidationResult):
    """Tableau récapitulatif professionnel"""
    try:
        summary_data = []
        
        for model in validation.successful_models:
            metrics = model.get('metrics', {})
            
            row = {
                'Modèle': model.get('model_name', 'Unknown'),
                'Temps (s)': f"{model.get('training_time', 0):.2f}"
            }
            
            if validation.task_type == 'clustering':
                row.update({
                    'Silhouette': f"{metrics.get('silhouette_score', 0):.3f}",
                    'Clusters': str(metrics.get('n_clusters', 'N/A')),
                    'Index DB': f"{metrics.get('davies_bouldin_score', 0):.3f}" 
                               if isinstance(metrics.get('davies_bouldin_score'), (int, float)) 
                               else 'N/A'
                })
            elif validation.task_type == 'regression':
                row.update({
                    'R²': f"{metrics.get('r2', 0):.3f}",
                    'MAE': f"{metrics.get('mae', 0):.3f}",
                    'RMSE': f"{metrics.get('rmse', 0):.3f}"
                })
            else:  # classification
                row.update({
                    'Accuracy': f"{metrics.get('accuracy', 0):.3f}",
                    'Precision': f"{metrics.get('precision', 0):.3f}",
                    'Recall': f"{metrics.get('recall', 0):.3f}",
                    'F1': f"{metrics.get('f1_score', 0):.3f}"
                })
            
            summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=min(500, len(df) * 45 + 50)
            )
    
    except Exception as e:
        st.error(f"❌ Erreur tableau: {str(e)[:100]}")


@monitor_operation
def render_model_details(
    visualizer: ModelEvaluationVisualizer,
    model_result: Dict[str, Any],
    task_type: str
):
    """Affichage détaillé d'un modèle avec visualisations"""
    try:
        model_name = model_result.get('model_name', 'Unknown')
        unique_id = f"{model_name}_{int(time.time() * 1000)}"
        
        # En-tête
        st.markdown(f"""
        <div class="plot-container">
            <div class="plot-title">
                🔍 {model_name} - Analyse Détaillée
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Validation données
        if not model_result.get('model') or not model_result.get('metrics'):
            st.error("❌ Données insuffisantes")
            return
        
        # Métriques principales
        st.markdown("#### 📊 Métriques de Performance")
        
        metrics = model_result.get('metrics', {})
        
        if task_type == 'classification':
            cols = st.columns(4)
            metrics_list = [
                ('Accuracy', 'accuracy'),
                ('Precision', 'precision'),
                ('Recall', 'recall'),
                ('F1-Score', 'f1_score')
            ]
            
            for col, (label, key) in zip(cols, metrics_list):
                value = metrics.get(key, 0)
                col.metric(label, f"{value:.3f}")
        
        elif task_type == 'regression':
            cols = st.columns(3)
            metrics_list = [
                ('R² Score', 'r2'),
                ('MAE', 'mae'),
                ('RMSE', 'rmse')
            ]
            
            for col, (label, key) in zip(cols, metrics_list):
                value = metrics.get(key, 0)
                col.metric(label, f"{value:.3f}")
        
        elif task_type == 'clustering':
            cols = st.columns(3)
            cols[0].metric("Silhouette", f"{metrics.get('silhouette_score', 0):.3f}")
            cols[1].metric("Clusters", str(metrics.get('n_clusters', 'N/A')))
            db = metrics.get('davies_bouldin_score', 'N/A')
            cols[2].metric("Index DB", f"{db:.3f}" if isinstance(db, (int, float)) else str(db))
        
        # Visualisations
        st.markdown("---")
        st.markdown("#### 📈 Visualisations")
        
        def safe_plot(plot_func, plot_name: str):
            """Génère un plot avec gestion d'erreurs"""
            try:
                with st.spinner(f"Génération {plot_name}..."):
                    fig = plot_func()
                    return fig
            except Exception as e:
                logger.error(f"Erreur {plot_name}: {e}")
                return None
        
        if task_type == 'clustering':
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown('<div class="plot-title">🔮 Visualisation Clusters</div>', unsafe_allow_html=True)
                fig = safe_plot(
                    lambda: visualizer.create_cluster_visualization(model_result),
                    "Clusters"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"cluster_{unique_id}")
                else:
                    st.info("ℹ️ Visualisation non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown('<div class="plot-title">📊 Analyse Silhouette</div>', unsafe_allow_html=True)
                fig = safe_plot(
                    lambda: visualizer.create_silhouette_analysis(model_result),
                    "Silhouette"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"sil_{unique_id}")
                else:
                    st.info("ℹ️ Analyse non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif task_type == 'classification':
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown('<div class="plot-title">📊 Matrice de Confusion</div>', unsafe_allow_html=True)
                fig = safe_plot(
                    lambda: visualizer.create_confusion_matrix(model_result),
                    "Confusion Matrix"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"cm_{unique_id}")
                else:
                    st.info("ℹ️ Matrice non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown('<div class="plot-title">📈 Courbe ROC</div>', unsafe_allow_html=True)
                fig = safe_plot(
                    lambda: visualizer.create_roc_curve(model_result),
                    "ROC Curve"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"roc_{unique_id}")
                else:
                    st.info("ℹ️ Courbe non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.markdown('<div class="plot-title">🎯 Importance des Features</div>', unsafe_allow_html=True)
            fig = safe_plot(
                lambda: visualizer.create_feature_importance_plot(model_result),
                "Feature Importance"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True, key=f"feat_{unique_id}")
            else:
                st.info("ℹ️ Importance non disponible")
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif task_type == 'regression':
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown('<div class="plot-title">📉 Résidus</div>', unsafe_allow_html=True)
                fig = safe_plot(
                    lambda: visualizer.create_residuals_plot(model_result),
                    "Residuals"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"res_{unique_id}")
                else:
                    st.info("ℹ️ Graphique non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                st.markdown('<div class="plot-title">🎯 Prédictions vs Réelles</div>', unsafe_allow_html=True)
                fig = safe_plot(
                    lambda: visualizer.create_predicted_vs_actual(model_result),
                    "Predictions"
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"pred_{unique_id}")
                else:
                    st.info("ℹ️ Graphique non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Infos complémentaires
        st.markdown("---")
        st.markdown("#### ℹ️ Informations Complémentaires")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            training_time = model_result.get('training_time', 0)
            st.metric("⏱️ Temps d'entraînement", f"{training_time:.2f}s")
        
        with col2:
            if model_result.get('X_train') is not None:
                n_samples = len(model_result['X_train'])
                st.metric("📊 Échantillons train", f"{n_samples:,}")
        
        with col3:
            status = "✅ Réussi" if model_result.get('success', False) else "❌ Échoué"
            st.metric("🔧 Statut", status)
        
        # Infos SMOTE et déséquilibre
        if model_result.get('smote_applied'):
            st.info("ℹ️ **SMOTE activé** pour ce modèle")
        
        if model_result.get('imbalance_ratio'):
            ratio = model_result['imbalance_ratio']
            st.info(f"📊 **Ratio de déséquilibre**: {ratio:.2f}:1")
    
    except Exception as e:
        logger.error(f"❌ Erreur détails modèle: {e}", exc_info=True)
        st.error(f"❌ Erreur affichage: {str(e)[:100]}")


def render_mlflow_tab():
    """Onglet MLflow avec design moderne"""
    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
    st.markdown('<div class="plot-title">🔗 Exploration des Runs MLflow</div>', unsafe_allow_html=True)
    
    try:
        import mlflow # type: ignore
        from mlflow.tracking import MlflowClient # type: ignore
        MLFLOW_AVAILABLE = True
    except ImportError:
        MLFLOW_AVAILABLE = False
    
    if not MLFLOW_AVAILABLE:
        st.error("🚫 MLflow non disponible")
        st.info("📦 Installez MLflow: `pip install mlflow`")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Récupération runs
    with st.spinner("🔄 Synchronisation des runs MLflow..."):
        mlflow_runs = get_mlflow_runs_robust()
    
    if not mlflow_runs:
        st.warning("⚠️ Aucun run MLflow disponible")
        
        st.info("""
        **💡 Pour générer des runs MLflow:**
        1. Allez dans **Configuration ML** (Training)
        2. Lancez un entraînement de modèles
        3. Revenez ici pour voir les résultats
        
        **🔍 Sources vérifiées:**
        - ✅ `st.session_state.mlflow_runs`
        - ✅ `STATE.mlflow_runs`
        - ✅ `STATE.training.mlflow_runs`
        - ✅ Recherche MLflow API
        """)
        
        with st.expander("🔧 Diagnostic Avancé", expanded=False):
            diagnostic = {
                'session_state.mlflow_runs': hasattr(st.session_state, 'mlflow_runs'),
                'STATE.mlflow_runs': hasattr(STATE, 'mlflow_runs'),
                'STATE.training.mlflow_runs': (
                    hasattr(STATE, 'training') and 
                    hasattr(STATE.training, 'mlflow_runs')
                ),
                'MLFLOW_AVAILABLE': MLFLOW_AVAILABLE
            }
            
            if hasattr(st.session_state, 'mlflow_runs'):
                diagnostic['session_state count'] = len(st.session_state.mlflow_runs)
            if hasattr(STATE, 'mlflow_runs'):
                diagnostic['STATE count'] = len(STATE.mlflow_runs)
            
            st.json(diagnostic)
        
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    st.success(f"**📊 {len(mlflow_runs)} runs MLflow disponibles**")
    
    # Filtres
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        status_filter = st.multiselect(
            "Filtrer par statut",
            options=['FINISHED', 'RUNNING', 'FAILED', 'SCHEDULED'],
            default=['FINISHED'],
            key="mlflow_status_filter_v3"
        )
    
    with col2:
        unique_models = set()
        for run in mlflow_runs:
            if isinstance(run, dict):
                model_name = (
                    run.get('tags', {}).get('mlflow.runName') or 
                    run.get('model_name', 'Unknown')
                )
                unique_models.add(model_name)
        
        model_filter = st.multiselect(
            "Filtrer par modèle",
            options=sorted(list(unique_models)),
            default=None,
            key="mlflow_model_filter_v3"
        )
    
    with col3:
        sort_by = st.selectbox(
            "Trier par",
            options=['Date (récent)', 'Nom', 'Score'],
            key="mlflow_sort_v3"
        )
    
    # Filtrage
    filtered_runs = []
    for run in mlflow_runs:
        if not isinstance(run, dict):
            continue
        
        if run.get('status') not in status_filter:
            continue
        
        if model_filter:
            model_name = (
                run.get('tags', {}).get('mlflow.runName') or 
                run.get('model_name', 'Unknown')
            )
            if model_name not in model_filter:
                continue
        
        filtered_runs.append(run)
    
    # Tri
    if sort_by == 'Date (récent)':
        filtered_runs.sort(key=lambda x: x.get('start_time', 0), reverse=True)
    elif sort_by == 'Nom':
        filtered_runs.sort(key=lambda x: x.get('model_name', 'Unknown'))
    elif sort_by == 'Score':
        filtered_runs.sort(
            key=lambda x: max(x.get('metrics', {}).values(), default=0),
            reverse=True
        )
    
    if not filtered_runs:
        st.info("ℹ️ Aucun run ne correspond aux filtres")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Tableau
    run_data = []
    for run in filtered_runs:
        run_id = run.get('run_id', 'N/A')
        model_name = (
            run.get('tags', {}).get('mlflow.runName') or 
            run.get('model_name', 'Unknown')
        )
        metrics = run.get('metrics', {})
        params = run.get('params', {})
        
        row = {
            'Run ID': run_id[:8] + '...' if len(run_id) > 8 else run_id,
            'Modèle': model_name,
            'Statut': run.get('status', 'UNKNOWN')
        }
        
        # Métriques
        for metric in ['accuracy', 'f1', 'precision', 'recall', 'r2', 'rmse', 'silhouette_score']:
            if metric in metrics:
                row[metric.upper()] = f"{metrics[metric]:.3f}"
        
        # Params
        if 'use_smote' in params:
            row['SMOTE'] = '✅' if params['use_smote'] == 'true' else '❌'
        
        if 'optimize_hyperparams' in params:
            row['Optim HP'] = '✅' if params['optimize_hyperparams'] == 'true' else '❌'
        
        run_data.append(row)
    
    if run_data:
        df_runs = pd.DataFrame(run_data)
        
        st.markdown(f"**📊 Affichage de {len(df_runs)} runs**")
        
        st.dataframe(
            df_runs,
            use_container_width=True,
            hide_index=True,
            height=min(600, len(df_runs) * 45 + 50)
        )
        
        # Export
        if st.button("📥 Exporter en CSV", key="export_mlflow_v3"):
            csv = df_runs.to_csv(index=False)
            st.download_button(
                label="💾 Télécharger CSV",
                data=csv,
                file_name=f"mlflow_runs_{int(time.time())}.csv",
                mime="text/csv"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# 🚀 FONCTION PRINCIPALE
# ============================================================================

def main():
    """Point d'entrée principal"""
    try:
        # Hero section
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 3rem; font-weight: 800;">
                📊 Évaluation ML Pro
            </h1>
            <p style="color: #666; font-size: 1.2rem;">
                Analyse Avancée des Performances
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.markdown("### ⚙️ Options")
            show_failed = st.checkbox("Afficher modèles échoués", value=False)
            show_mlflow = st.checkbox("Dashboard MLflow", value=True)
            
            if st.button("🔄 Rafraîchir", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        # Récupération résultats
        training_results = None
        results_data = None
        
        if hasattr(STATE, 'training_results') and STATE.training_results:
            training_results = STATE.training_results
        elif hasattr(STATE, 'ml_results') and STATE.ml_results:
            results_data = STATE.ml_results
        else:
            st.error("🚫 Aucun résultat d'entraînement")
            st.info("Lancez un entraînement depuis **Training ML**")
            if st.button("⚙️ Aller au Training", type="primary"):
                STATE.switch(AppPage.ML_TRAINING)
            return
        
        # Extraction
        if training_results:
            if hasattr(training_results, 'results'):
                results_data = training_results.results
            else:
                st.error("❌ Format invalide")
                return
        
        # Validation
        from helpers.data_validators import DataValidator
        
        validation = {
            'successful_models': [],
            'failed_models': [],
            'task_type': getattr(STATE, 'task_type', 'classification')
        }
        
        for r in results_data:
            if r.get('success'):
                validation['successful_models'].append(r)
            else:
                validation['failed_models'].append(r)
        
        if not validation['successful_models']:
            st.error("❌ Aucun modèle réussi")
            return
        
        # Meilleur modèle
        task_type = validation['task_type']
        metric_key = {
            'classification': 'accuracy',
            'regression': 'r2',
            'clustering': 'silhouette_score'
        }.get(task_type, 'accuracy')
        
        best_model_result = max(
            validation['successful_models'],
            key=lambda x: x.get('metrics', {}).get(metric_key, 0)
        )
        validation['best_model'] = best_model_result.get('model_name', 'Unknown')
        
        # === MÉTRIQUES HORIZONTALES ===
        render_metrics_horizontal(validation)
        
        # === VISUALISEUR ===
        visualizer = ModelEvaluationVisualizer(results_data)
        
        # === GRAPHIQUE AVANCÉ ===
        st.markdown("---")
        create_advanced_comparison(visualizer, validation)
        
        # === GRILLE GRAPHIQUES ===
        st.markdown("---")
        st.markdown("### 📊 Analyses Détaillées")
        
        st.markdown('<div class="plot-grid">', unsafe_allow_html=True)
        
        # Graphique 1: Comparaison standard
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        fig1 = visualizer.create_comparison_plot()
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Graphique 2: Distribution performances
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        fig2 = visualizer.create_performance_distribution()
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Graphique 3: Temps vs Performance
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        fig3 = visualizer.create_time_vs_performance_plot()
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # === DASHBOARD MLFLOW ===
        if show_mlflow:
            st.markdown("---")
            mlflow_runs = STATE.get_mlflow_runs()
            create_mlflow_dashboard(mlflow_runs)
        
        # === TABLEAU RÉCAPITULATIF ===
        st.markdown("---")
        st.markdown("### 📋 Tableau Récapitulatif")
        df_comparison = visualizer.get_comparison_dataframe()
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # === EXPORT ===
        st.markdown("---")
        st.markdown("### 📥 Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_comparison.to_csv(index=False)
            st.download_button(
                "💾 Télécharger CSV",
                csv,
                f"evaluation_{int(time.time())}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            export_data = visualizer.get_export_data()
            import json
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                "💾 Télécharger JSON",
                json_str,
                f"evaluation_{int(time.time())}.json",
                "application/json",
                use_container_width=True
            )
    
    except Exception as e:
        logger.error(f"❌ Erreur page évaluation: {e}", exc_info=True)
        st.error(f"❌ Erreur: {str(e)}")


if __name__ == "__main__":
    main()