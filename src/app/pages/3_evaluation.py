"""
📊 ML Factory Pro - Page Évaluation ML v4.0
Version Complète et Moderne avec Toutes les Fonctionnalités
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import tempfile
import base64

# Imports internes
from src.evaluation.model_plots import ModelEvaluationVisualizer
from src.shared.logging import get_logger
from monitoring.state_managers import init, AppPage, STATE
from ui.styles import UIStyles
from ui.evaluation_styles import EvaluationStyles
from helpers.ui_components import UIComponents

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
st.markdown(EvaluationStyles.get_evaluation_css(), unsafe_allow_html=True)


# ============================================================================
# 🎨 FONCTIONS UTILITAIRES AVANCÉES
# ============================================================================

def get_metric_badge(value: float, metric_type: str) -> str:
    """Retourne un badge HTML selon la performance avec couleurs adaptatives"""
    if value is None:
        return '<span class="metric-badge badge-neutral">N/A</span>'
    
    if metric_type in ['accuracy', 'precision', 'recall', 'f1_score', 'r2', 'silhouette_score']:
        if value >= 0.9:
            return f'<span class="metric-badge badge-excellent">{value:.3f}</span>'
        elif value >= 0.75:
            return f'<span class="metric-badge badge-good">{value:.3f}</span>'
        elif value >= 0.6:
            return f'<span class="metric-badge badge-fair">{value:.3f}</span>'
        else:
            return f'<span class="metric-badge badge-poor">{value:.3f}</span>'
    elif metric_type in ['mae', 'rmse']:
        if value <= 0.1:
            return f'<span class="metric-badge badge-excellent">{value:.3f}</span>'
        elif value <= 0.3:
            return f'<span class="metric-badge badge-good">{value:.3f}</span>'
        elif value <= 0.5:
            return f'<span class="metric-badge badge-fair">{value:.3f}</span>'
        else:
            return f'<span class="metric-badge badge-poor">{value:.3f}</span>'
    else:
        return f'<span class="metric-badge badge-neutral">{value:.3f}</span>'


def get_progress_bar(value: float, metric_type: str) -> str:
    """Retourne une barre de progression visuelle"""
    if value is None:
        return '<div class="progress-container"><div class="progress-bar" style="width: 0%"></div></div>'
    
    width = min(100, max(0, value * 100))
    
    if metric_type in ['accuracy', 'precision', 'recall', 'f1_score', 'r2', 'silhouette_score']:
        if value >= 0.9:
            progress_class = "progress-excellent"
        elif value >= 0.75:
            progress_class = "progress-good"
        elif value >= 0.6:
            progress_class = "progress-fair"
        else:
            progress_class = "progress-poor"
    else:
        progress_class = "progress-fair"
    
    return f'<div class="progress-container"><div class="progress-bar {progress_class}" style="width: {width}%"></div></div>'


def render_metrics_dashboard_horizontal(validation: Dict[str, Any]):
    """Dashboard de métriques horizontal compact"""
    try:
        total_models = len(validation['successful_models']) + len(validation['failed_models'])
        success_rate = (len(validation['successful_models']) / total_models * 100) if total_models > 0 else 0
        
        task_type = validation.get('task_type', 'classification')
        metric_key = {
            'clustering': 'silhouette_score',
            'regression': 'r2',
            'classification': 'accuracy'
        }.get(task_type, 'accuracy')
        
        # Calcul du meilleur score
        best_score = max(
            [m.get('metrics', {}).get(metric_key, 0) for m in validation['successful_models']],
            default=0
        )
        
        # Temps moyen d'entraînement
        avg_time = np.mean([
            m.get('training_time', 0) 
            for m in validation['successful_models']
        ]) if validation['successful_models'] else 0
        
        # Score moyen
        avg_score = np.mean([
            m.get('metrics', {}).get(metric_key, 0) 
            for m in validation['successful_models']
        ]) if validation['successful_models'] else 0

        st.markdown('<div class="metrics-horizontal-compact">', unsafe_allow_html=True)
        
        # Carte 1: Taux de réussite
        color = "#28a745" if success_rate > 80 else "#ffc107" if success_rate > 50 else "#dc3545"
        st.markdown(f"""
        <div class="metric-card-horizontal" style="--card-color: {color};">
            <div class="metric-icon-horizontal">✅</div>
            <div class="metric-value-horizontal" style="color: {color};">{success_rate:.0f}%</div>
            <div class="metric-label-horizontal">Taux Réussite</div>
            <div class="metric-subtitle-horizontal">{len(validation['successful_models'])}/{total_models}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Carte 2: Meilleur modèle
        st.markdown(f"""
        <div class="metric-card-horizontal" style="--card-color: #667eea;">
            <div class="metric-icon-horizontal">🏆</div>
            <div class="metric-value-horizontal" style="font-size: 1.3rem;">{validation.get('best_model', 'N/A')}</div>
            <div class="metric-label-horizontal">Meilleur Modèle</div>
            <div class="metric-subtitle-horizontal">{task_type.title()}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Carte 3: Meilleur score
        st.markdown(f"""
        <div class="metric-card-horizontal" style="--card-color: #17a2b8;">
            <div class="metric-icon-horizontal">📈</div>
            <div class="metric-value-horizontal">{best_score:.3f}</div>
            <div class="metric-label-horizontal">Meilleur Score</div>
            <div class="metric-subtitle-horizontal">{metric_key}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Carte 4: Score moyen
        st.markdown(f"""
        <div class="metric-card-horizontal" style="--card-color: #f39c12;">
            <div class="metric-icon-horizontal">📊</div>
            <div class="metric-value-horizontal">{avg_score:.3f}</div>
            <div class="metric-label-horizontal">Score Moyen</div>
            <div class="metric-subtitle-horizontal">Tous modèles</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Carte 5: Temps moyen
        st.markdown(f"""
        <div class="metric-card-horizontal" style="--card-color: #6f42c1;">
            <div class="metric-icon-horizontal">⏱️</div>
            <div class="metric-value-horizontal">{avg_time:.1f}s</div>
            <div class="metric-label-horizontal">Temps Moyen</div>
            <div class="metric-subtitle-horizontal">Par modèle</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Erreur métriques: {str(e)[:100]}")


def create_complex_comparison_table(validation: Dict[str, Any]):
    """Tableau de comparaison complexe avec design avancé"""
    try:
        successful_models = validation['successful_models']
        failed_models = validation['failed_models']
        task_type = validation['task_type']
        best_model_name = validation.get('best_model')
        
        # En-tête du tableau
        st.markdown("""
        <div class="complex-table-container">
            <div class="table-header-modern">
                <div>
                    <div class="table-title">📊 Comparaison Détaillée des Modèles</div>
                    <div class="table-subtitle">Analyse complète des performances et métriques</div>
                </div>
                <div style="font-size: 0.8rem; opacity: 0.9;">
                    🟢 Meilleur modèle • 🔴 Échecs
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Construction du tableau
        table_html = '<table class="complex-table">'
        
        # En-têtes selon le type de tâche
        if task_type == 'classification':
            headers = ['Modèle', 'Statut', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Temps (s)', 'Performance']
        elif task_type == 'regression':
            headers = ['Modèle', 'Statut', 'R² Score', 'MAE', 'RMSE', 'R² Adj.', 'Temps (s)', 'Performance']
        else:  # clustering
            headers = ['Modèle', 'Statut', 'Silhouette', 'Clusters', 'DB Index', 'Calinski', 'Temps (s)', 'Performance']
        
        table_html += '<thead><tr>'
        for header in headers:
            table_html += f'<th>{header}</th>'
        table_html += '</tr></thead><tbody>'
        
        # Lignes des modèles réussis
        for model in successful_models:
            metrics = model.get('metrics', {})
            model_name = model.get('model_name', 'Unknown')
            training_time = model.get('training_time', 0)
            
            # Classe CSS pour le meilleur modèle
            row_class = 'best-model-row' if model_name == best_model_name else ''
            
            table_html += f'<tr class="{row_class}">'
            
            # Colonne Modèle
            table_html += f'<td><strong>{model_name}</strong>'
            if model_name == best_model_name:
                table_html += '&nbsp;<span style="color: #28a745; font-size: 0.8rem;">👑</span>'
            table_html += '</td>'
            
            # Colonne Statut
            table_html += '<td><span class="status-indicator status-success">✅ Succès</span></td>'
            
            # Métriques selon le type de tâche
            if task_type == 'classification':
                table_html += f'<td>{get_metric_badge(metrics.get("accuracy", 0), "accuracy")}</td>'
                table_html += f'<td>{get_metric_badge(metrics.get("precision", 0), "precision")}</td>'
                table_html += f'<td>{get_metric_badge(metrics.get("recall", 0), "recall")}</td>'
                table_html += f'<td>{get_metric_badge(metrics.get("f1_score", 0), "f1_score")}</td>'
                table_html += f'<td>{get_metric_badge(metrics.get("roc_auc", 0), "accuracy")}</td>'
            elif task_type == 'regression':
                table_html += f'<td>{get_metric_badge(metrics.get("r2", 0), "r2")}</td>'
                table_html += f'<td>{get_metric_badge(metrics.get("mae", 0), "mae")}</td>'
                table_html += f'<td>{get_metric_badge(metrics.get("rmse", 0), "rmse")}</td>'
                table_html += f'<td>{get_metric_badge(metrics.get("r2_adj", 0), "r2")}</td>'
            else:  # clustering
                table_html += f'<td>{get_metric_badge(metrics.get("silhouette_score", 0), "silhouette_score")}</td>'
                table_html += f'<td>{metrics.get("n_clusters", "N/A")}</td>'
                table_html += f'<td>{get_metric_badge(metrics.get("davies_bouldin_score", 0), "accuracy")}</td>'
                table_html += f'<td>{get_metric_badge(metrics.get("calinski_harabasz_score", 0), "accuracy")}</td>'
            
            # Colonne Temps
            table_html += f'<td>{training_time:.1f}s</td>'
            
            # Colonne Performance (barre visuelle)
            main_metric = {
                'classification': 'accuracy',
                'regression': 'r2',
                'clustering': 'silhouette_score'
            }.get(task_type, 'accuracy')
            
            score = metrics.get(main_metric, 0)
            table_html += f'<td>{get_progress_bar(score, main_metric)}</td>'
            
            table_html += '</tr>'
        
        # Lignes des modèles échoués
        for model in failed_models:
            model_name = model.get('model_name', 'Unknown')
            training_time = model.get('training_time', 0)
            error_msg = model.get('error', 'Erreur inconnue')
            
            table_html += f'<tr class="failed-model-row">'
            table_html += f'<td><strong>{model_name}</strong></td>'
            table_html += f'<td><span class="status-indicator status-failed">❌ Échec</span></td>'
            
            # Colonnes vides pour les métriques
            empty_cols = 6 if task_type == 'clustering' else 5
            table_html += f'<td colspan="{empty_cols}" style="color: #dc3545; font-style: italic;">{error_msg[:80]}...</td>'
            
            table_html += f'<td>{training_time:.1f}s</td>'
            table_html += '<td><div class="progress-container"><div class="progress-bar progress-poor" style="width: 0%"></div></div></td>'
            table_html += '</tr>'
        
        table_html += '</tbody></table></div>'
        
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Légende
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            st.markdown("""
            <div style="font-size: 0.8rem; color: #6c757d; margin-top: 0.5rem;">
                <strong>Légende:</strong> 
                <span class="metric-badge badge-excellent">Excellent</span>
                <span class="metric-badge badge-good">Bon</span>
                <span class="metric-badge badge-fair">Moyen</span>
                <span class="metric-badge badge-poor">Faible</span>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Erreur tableau comparaison: {str(e)[:100]}")


def _empty_fig(msg: str) -> go.Figure:
    """Retourne une figure vide avec un message"""
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color="red")
    )
    fig.update_layout(
        height=300,
        template="plotly_white",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


# ============================================================================
# 🎨 FONCTIONS PRINCIPALES PAR ONGLET - VERSION COMPLÈTE
# ============================================================================

def render_tab_overview(validation: Dict[str, Any], visualizer: ModelEvaluationVisualizer):
    """Onglet Vue d'ensemble avec 6-8 graphiques horizontaux"""
    st.markdown("## 🎯 Vue d'Ensemble des Performances")
    
    # Métriques horizontales compactes
    render_metrics_dashboard_horizontal(validation)
    
    # Tableau de comparaison complexe
    create_complex_comparison_table(validation)
    
    # Graphiques principaux en grille 2x3
    st.markdown('<div class="section-header">📊 Visualisations des Performances</div>', unsafe_allow_html=True)
    
    # Première ligne de graphiques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        st.markdown('**📈 Comparaison des Scores**')
        try:
            fig1 = visualizer.create_comparison_plot()
            st.plotly_chart(fig1, use_container_width=True, key="overview_comparison")
        except Exception as e:
            st.error(f"❌ Erreur graphique: {str(e)[:100]}")
            st.plotly_chart(_empty_fig("Graphique non disponible"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        st.markdown('**⏱️ Temps vs Performance**')
        try:
            fig2 = visualizer.create_time_vs_performance_plot()
            st.plotly_chart(fig2, use_container_width=True, key="overview_time")
        except Exception as e:
            st.error(f"❌ Erreur graphique: {str(e)[:100]}")
            st.plotly_chart(_empty_fig("Graphique non disponible"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        st.markdown('**📊 Distribution des Performances**')
        try:
            fig3 = visualizer.create_performance_distribution()
            st.plotly_chart(fig3, use_container_width=True, key="overview_dist")
        except Exception as e:
            st.error(f"❌ Erreur graphique: {str(e)[:100]}")
            st.plotly_chart(_empty_fig("Graphique non disponible"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Deuxième ligne de graphiques
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        st.markdown('**🎯 Radar de Comparaison**')
        try:
            fig4 = visualizer.create_radar_comparison()
            st.plotly_chart(fig4, use_container_width=True, key="overview_radar")
        except Exception as e:
            st.error(f"❌ Erreur graphique: {str(e)[:100]}")
            st.plotly_chart(_empty_fig("Graphique non disponible"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        st.markdown('**📈 Courbes d\'Apprentissage**')
        try:
            # Prendre le meilleur modèle pour la courbe d'apprentissage
            best_model_name = validation.get('best_model')
            best_model = next((m for m in validation['successful_models'] 
                             if m.get('model_name') == best_model_name), 
                             validation['successful_models'][0] if validation['successful_models'] else None)
            if best_model:
                fig5 = visualizer.create_learning_curve(best_model)
                st.plotly_chart(fig5, use_container_width=True, key="overview_learning")
            else:
                st.plotly_chart(_empty_fig("Aucun modèle disponible"), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Erreur graphique: {str(e)[:100]}")
            st.plotly_chart(_empty_fig("Graphique non disponible"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col6:
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        st.markdown('**🔍 Importance des Features (Top)**')
        try:
            best_model_name = validation.get('best_model')
            best_model = next((m for m in validation['successful_models'] 
                             if m.get('model_name') == best_model_name), 
                             validation['successful_models'][0] if validation['successful_models'] else None)
            if best_model:
                fig6 = visualizer.create_feature_importance_plot_fixed(best_model)
                st.plotly_chart(fig6, use_container_width=True, key="overview_feat")
            else:
                st.plotly_chart(_empty_fig("Aucun modèle disponible"), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Erreur graphique: {str(e)[:100]}")
            st.plotly_chart(_empty_fig("Graphique non disponible"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def render_tab_details(validation: Dict[str, Any], visualizer: ModelEvaluationVisualizer):
    """Onglet Détails par Modèle avec 4-6 graphiques spécifiques"""
    st.markdown("## 🔍 Analyse Détaillée par Modèle")
    
    successful_models = validation['successful_models']
    task_type = validation['task_type']
    
    if not successful_models:
        st.info("ℹ️ Aucun modèle disponible pour analyse détaillée")
        return
    
    # Sélecteur de modèle avec style
    model_names = [m.get('model_name', 'Unknown') for m in successful_models]
    selected_model_name = st.selectbox(
        "📌 Sélectionnez un modèle à analyser",
        options=model_names,
        key="detail_model_select"
    )
    
    # Trouver le modèle sélectionné
    selected_model = next(
        (m for m in successful_models if m.get('model_name') == selected_model_name),
        None
    )
    
    if not selected_model:
        st.error("❌ Modèle introuvable")
        return
    
    # En-tête du modèle
    is_best = selected_model_name == validation.get('best_model')
    best_badge = " 🏆" if is_best else ""
    
    st.markdown(f"""
    <div style="background: {'linear-gradient(135deg, #28a74520 0%, #20c99720 100%)' if is_best else 'white'}; 
                border-radius: 15px; padding: 1.5rem; margin: 1rem 0; 
                border: 2px solid {'#28a745' if is_best else '#e9ecef'};">
        <h3 style="margin: 0; color: #2d3748;">
            🤖 {selected_model_name}{best_badge}
        </h3>
        <p style="margin: 0.5rem 0 0 0; color: #6c757d;">
            {task_type.title()} • Temps d'entraînement: {selected_model.get('training_time', 0):.2f}s
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Métriques détaillées
    metrics = selected_model.get('metrics', {})
    
    st.markdown('<div class="section-header">📊 Métriques de Performance</div>', unsafe_allow_html=True)
    
    # Affichage des métriques selon le type de tâche
    if task_type == 'classification':
        display_metrics = [
            ('Accuracy', 'accuracy', metrics.get('accuracy', 0)),
            ('Precision', 'precision', metrics.get('precision', 0)),
            ('Recall', 'recall', metrics.get('recall', 0)),
            ('F1-Score', 'f1_score', metrics.get('f1_score', 0)),
            ('ROC-AUC', 'accuracy', metrics.get('roc_auc', 0))
        ]
        cols = st.columns(5)
        
    elif task_type == 'regression':
        display_metrics = [
            ('R² Score', 'r2', metrics.get('r2', 0)),
            ('MAE', 'mae', metrics.get('mae', 0)),
            ('RMSE', 'rmse', metrics.get('rmse', 0)),
            ('R² Adj.', 'r2', metrics.get('r2_adj', 0))
        ]
        cols = st.columns(4)
        
    else:  # clustering
        display_metrics = [
            ('Silhouette', 'silhouette_score', metrics.get('silhouette_score', 0)),
            ('Clusters', 'neutral', metrics.get('n_clusters', 0)),
            ('DB Index', 'accuracy', metrics.get('davies_bouldin_score', 0)),
            ('Calinski', 'accuracy', metrics.get('calinski_harabasz_score', 0))
        ]
        cols = st.columns(4)
    
    # Affichage des métriques
    for col, (label, metric_type, value) in zip(cols, display_metrics):
        with col:
            st.markdown(f"**{label}**")
            
            if metric_type == 'neutral':
                # Cas spécial pour les valeurs non-métriques
                st.markdown(f"<div style='font-size: 1.2rem; font-weight: bold; color: #667eea;'>{value}</div>", unsafe_allow_html=True)
            else:
                # Cas normal pour les métriques
                st.markdown(get_metric_badge(value, metric_type), unsafe_allow_html=True)
                
                # Barre de progression seulement pour les métriques principales
                if metric_type in ['accuracy', 'precision', 'recall', 'f1_score', 'r2', 'silhouette_score']:
                    st.markdown(get_progress_bar(value, metric_type), unsafe_allow_html=True)
    
    # Visualisations spécifiques au type de tâche
    st.markdown('<div class="section-header">📈 Visualisations Spécifiques</div>', unsafe_allow_html=True)
    
    try:
        if task_type == 'classification':
            # Ligne 1: Matrice de confusion et ROC
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
                st.markdown('**📊 Matrice de Confusion**')
                try:
                    fig = visualizer.create_confusion_matrix(selected_model)
                    st.plotly_chart(fig, use_container_width=True, key=f"cm_{selected_model_name}")
                except Exception:
                    st.info("ℹ️ Matrice de confusion non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
                st.markdown('**📈 Courbe ROC**')
                try:
                    fig = visualizer.create_roc_curve(selected_model)
                    st.plotly_chart(fig, use_container_width=True, key=f"roc_{selected_model_name}")
                except Exception:
                    st.info("ℹ️ Courbe ROC non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Ligne 2: Precision-Recall et Calibration
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
                st.markdown('**📊 Courbe Precision-Recall**')
                try:
                    fig = visualizer.create_precision_recall_curve(selected_model)
                    st.plotly_chart(fig, use_container_width=True, key=f"pr_{selected_model_name}")
                except Exception:
                    st.info("ℹ️ Courbe Precision-Recall non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
                st.markdown('**⚖️ Courbe de Calibration**')
                try:
                    fig = visualizer.create_calibration_plot(selected_model)
                    st.plotly_chart(fig, use_container_width=True, key=f"calib_{selected_model_name}")
                except Exception:
                    st.info("ℹ️ Courbe de calibration non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Importance des features
            st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
            st.markdown('**🎯 Importance des Features**')
            try:
                fig = visualizer.create_feature_importance_plot_fixed(selected_model)
                st.plotly_chart(fig, use_container_width=True, key=f"feat_{selected_model_name}")
            except Exception:
                st.info("ℹ️ Importance des features non disponible")
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif task_type == 'regression':
            # Ligne 1: Résidus et Prédictions vs Réelles
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
                st.markdown('**📉 Analyse des Résidus**')
                try:
                    fig = visualizer.create_residuals_plot(selected_model)
                    st.plotly_chart(fig, use_container_width=True, key=f"res_{selected_model_name}")
                except Exception:
                    st.info("ℹ️ Graphique des résidus non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
                st.markdown('**🎯 Prédictions vs Réelles**')
                try:
                    fig = visualizer.create_predicted_vs_actual(selected_model)
                    st.plotly_chart(fig, use_container_width=True, key=f"pred_{selected_model_name}")
                except Exception:
                    st.info("ℹ️ Graphique prédictions vs réelles non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Ligne 2: Distribution erreurs et Importance features
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
                st.markdown('**📊 Distribution des Erreurs**')
                try:
                    fig = visualizer.create_error_distribution(selected_model)
                    st.plotly_chart(fig, use_container_width=True, key=f"err_{selected_model_name}")
                except Exception:
                    st.info("ℹ️ Distribution des erreurs non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
                st.markdown('**🎯 Importance des Features**')
                try:
                    fig = visualizer.create_feature_importance_plot_fixed(selected_model)
                    st.plotly_chart(fig, use_container_width=True, key=f"feat_{selected_model_name}")
                except Exception:
                    st.info("ℹ️ Importance des features non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif task_type == 'clustering':
            # Ligne 1: Visualisation clusters et Analyse silhouette
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
                st.markdown('**🔮 Visualisation des Clusters**')
                try:
                    fig = visualizer.create_cluster_visualization(selected_model)
                    st.plotly_chart(fig, use_container_width=True, key=f"cluster_{selected_model_name}")
                except Exception:
                    st.info("ℹ️ Visualisation clusters non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
                st.markdown('**📊 Analyse Silhouette**')
                try:
                    fig = visualizer.create_silhouette_analysis(selected_model)
                    st.plotly_chart(fig, use_container_width=True, key=f"sil_{selected_model_name}")
                except Exception:
                    st.info("ℹ️ Analyse silhouette non disponible")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Matrice de corrélation
            st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
            st.markdown('**📈 Matrice de Corrélation**')
            try:
                fig = visualizer.create_feature_correlation_matrix(selected_model)
                st.plotly_chart(fig, use_container_width=True, key=f"corr_{selected_model_name}")
            except Exception:
                st.info("ℹ️ Matrice de corrélation non disponible")
            st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Erreur visualisations: {str(e)[:100]}")


def render_tab_advanced(validation: Dict[str, Any], visualizer: ModelEvaluationVisualizer):
    """Onglet Analyse Avancée avec 3-5 graphiques avancés"""
    st.markdown("## 🔬 Analyse Avancée")
    
    successful_models = validation['successful_models']
    
    if not successful_models:
        st.info("ℹ️ Aucun modèle disponible pour l'analyse avancée")
        return
    
    # Sélecteur de modèle pour l'analyse avancée
    model_names = [m.get('model_name', 'Unknown') for m in successful_models]
    selected_model_name = st.selectbox(
        "🔍 Sélectionnez un modèle pour l'analyse avancée",
        options=model_names,
        key="advanced_model_select"
    )
    
    selected_model = next(
        (m for m in successful_models if m.get('model_name') == selected_model_name),
        None
    )
    
    if not selected_model:
        st.error("❌ Modèle introuvable")
        return
    
    st.markdown(f"### 🤖 Analyse Avancée - {selected_model_name}")
    
    # Graphiques avancés
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        st.markdown('**🧠 Analyse SHAP**')
        try:
            fig = visualizer.create_shap_analysis(selected_model)
            st.plotly_chart(fig, use_container_width=True, key=f"shap_{selected_model_name}")
        except Exception as e:
            st.info(f"ℹ️ Analyse SHAP non disponible: {str(e)[:100]}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        st.markdown('**📚 Courbe d\'Apprentissage**')
        try:
            fig = visualizer.create_learning_curve(selected_model)
            st.plotly_chart(fig, use_container_width=True, key=f"learn_{selected_model_name}")
        except Exception as e:
            st.info(f"ℹ️ Courbe d'apprentissage non disponible: {str(e)[:100]}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Matrice de corrélation des features
    st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
    st.markdown('**📊 Matrice de Corrélation des Features**')
    try:
        fig = visualizer.create_feature_correlation_matrix(selected_model)
        st.plotly_chart(fig, use_container_width=True, key=f"corr_adv_{selected_model_name}")
    except Exception as e:
        st.info(f"ℹ️ Matrice de corrélation non disponible: {str(e)[:100]}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Distribution des erreurs (pour régression) ou autre analyse
    task_type = validation['task_type']
    if task_type == 'regression':
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        st.markdown('**📈 Distribution des Erreurs**')
        try:
            fig = visualizer.create_error_distribution(selected_model)
            st.plotly_chart(fig, use_container_width=True, key=f"err_adv_{selected_model_name}")
        except Exception as e:
            st.info(f"ℹ️ Distribution des erreurs non disponible: {str(e)[:100]}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyse de calibration (pour classification)
    if task_type == 'classification':
        st.markdown('<div class="plot-container-modern">', unsafe_allow_html=True)
        st.markdown('**⚖️ Courbe de Calibration**')
        try:
            fig = visualizer.create_calibration_plot(selected_model)
            st.plotly_chart(fig, use_container_width=True, key=f"calib_adv_{selected_model_name}")
        except Exception as e:
            st.info(f"ℹ️ Courbe de calibration non disponible: {str(e)[:100]}")
        st.markdown('</div>', unsafe_allow_html=True)


def render_tab_mlflow(mlflow_runs: List[Dict[str, Any]]):
    """Onglet MLflow avec tableau et filtres"""
    st.markdown("## 🔗 Exploration des Runs MLflow")
    
    if not mlflow_runs:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #6c757d;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📭</div>
            <h3 style="color: #495057;">Aucun Run MLflow Disponible</h3>
            <p>Lancez un entraînement depuis la page Training pour générer des runs MLflow</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.success(f"**📊 {len(mlflow_runs)} runs MLflow disponibles**")
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filtre par statut
        status_options = list(set(run.get('status', 'UNKNOWN') for run in mlflow_runs))
        selected_status = st.multiselect(
            "Filtrer par statut",
            options=status_options,
            default=status_options
        )
    
    with col2:
        # Filtre par modèle
        model_options = list(set(run.get('model_name', 'Unknown') for run in mlflow_runs))
        selected_models = st.multiselect(
            "Filtrer par modèle",
            options=model_options,
            default=model_options
        )
    
    with col3:
        # Filtre par métrique
        st.write("Trier par:")
        sort_options = ['date', 'score', 'temps']
        sort_by = st.selectbox("Critère de tri", options=sort_options)
    
    # Application des filtres
    filtered_runs = [
        run for run in mlflow_runs 
        if run.get('status') in selected_status 
        and run.get('model_name') in selected_models
    ]
    
    st.info(f"**{len(filtered_runs)}** runs correspondant aux filtres")
    
    # Affichage des runs filtrés
    for i, run in enumerate(filtered_runs[:20]):  # Limiter à 20 runs
        if not isinstance(run, dict):
            continue
            
        run_id = run.get('run_id', 'N/A')
        model_name = run.get('model_name', 'Unknown')
        status = run.get('status', 'UNKNOWN')
        metrics = run.get('metrics', {})
        params = run.get('params', {})
        
        with st.expander(f"🧪 {model_name} - {status}", expanded=i==0):
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.write("**📈 Métriques principales:**")
                for metric, value in list(metrics.items())[:3]:
                    st.write(f"- {metric}: `{value:.4f}`")
            
            with col2:
                st.write("**⚙️ Paramètres:**")
                for param, value in list(params.items())[:3]:
                    st.write(f"- {param}: `{value}`")
            
            with col3:
                st.write("**📋 Infos:**")
                st.write(f"- Run ID: `{run_id[:8]}...`")
                st.write(f"- Statut: `{status}`")
                
                # Bouton pour voir les détails
                if st.button("📖 Détails", key=f"details_{run_id}", use_container_width=True):
                    st.write("**Métriques complètes:**")
                    st.json(metrics)
                    st.write("**Paramètres complets:**")
                    st.json(params)


def render_tab_export(validation: Dict[str, Any], visualizer: ModelEvaluationVisualizer):
    """Onglet Export avec CSV, JSON et téléchargement modèle"""
    st.markdown("## 📥 Export des Résultats")
    
    successful_models = validation['successful_models']
    best_model_name = validation.get('best_model')
    
    # Section export données
    st.markdown("### 💾 Export des Données d'Évaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Format CSV")
        st.markdown("Tableau de comparaison des modèles avec toutes les métriques")
        
        try:
            df_comparison = visualizer.get_comparison_dataframe()
            csv = df_comparison.to_csv(index=False)
            
            st.download_button(
                "📥 Télécharger CSV Complet",
                csv,
                f"evaluation_comparison_{int(time.time())}.csv",
                "text/csv",
                use_container_width=True,
                key="export_csv"
            )
        except Exception as e:
            st.error(f"❌ Erreur export CSV: {str(e)[:100]}")
    
    with col2:
        st.markdown("#### 📄 Format JSON")
        st.markdown("Données complètes avec métriques détaillées et statistiques")
        
        try:
            export_data = visualizer.get_export_data()
            json_str = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📥 Télécharger JSON Complet",
                json_str,
                f"evaluation_complete_{int(time.time())}.json",
                "application/json",
                use_container_width=True,
                key="export_json"
            )
        except Exception as e:
            st.error(f"❌ Erreur export JSON: {str(e)[:100]}")
    
    # Section export modèle
    st.markdown("### 🤖 Export du Modèle Entraîné")
    
    if best_model_name:
        best_model = next((m for m in successful_models 
                          if m.get('model_name') == best_model_name), None)
        
        if best_model and best_model.get('model'):
            st.success(f"**🏆 Modèle sélectionné pour l'export:** {best_model_name}")
            
            col3, col4 = st.columns(2)
            
            with col3:
                # Export modèle pickle
                try:
                    import pickle
                    model_bytes = pickle.dumps(best_model['model'])
                    st.download_button(
                        "💾 Télécharger Modèle (Pickle)",
                        model_bytes,
                        f"model_{best_model_name}_{int(time.time())}.pkl",
                        "application/octet-stream",
                        use_container_width=True,
                        key="export_model_pkl"
                    )
                except Exception as e:
                    st.error(f"❌ Erreur export modèle: {str(e)[:100]}")
            
            with col4:
                # Export rapport détaillé
                try:
                    report_data = {
                        "model_name": best_model_name,
                        "export_date": datetime.now().isoformat(),
                        "task_type": validation['task_type'],
                        "metrics": best_model.get('metrics', {}),
                        "training_time": best_model.get('training_time', 0),
                        "performance_summary": f"Meilleur modèle avec score {max(best_model.get('metrics', {}).values()):.3f}"
                    }
                    
                    report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        "📋 Rapport du Modèle",
                        report_json,
                        f"model_report_{best_model_name}_{int(time.time())}.json",
                        "application/json",
                        use_container_width=True,
                        key="export_report"
                    )
                except Exception as e:
                    st.error(f"❌ Erreur création rapport: {str(e)[:100]}")
        else:
            st.warning("ℹ️ Le modèle entraîné n'est pas disponible pour l'export")
    else:
        st.warning("ℹ️ Aucun modèle optimal identifié pour l'export")


def render_tab_debug(validation: Dict[str, Any], visualizer: ModelEvaluationVisualizer):
    """Onglet DEBUG avec tous les graphiques dans des expanders"""
    st.markdown("## 🐛 Mode DEBUG - Tous les Graphiques")
    
    successful_models = validation['successful_models']
    
    if not successful_models:
        st.info("ℹ️ Aucun modèle disponible pour le debug")
        return
    
    # Option pour voir tous les graphiques
    show_all = st.checkbox("📋 Afficher TOUS les graphiques pour TOUS les modèles", value=False)
    
    models_to_debug = successful_models if show_all else [successful_models[0]]
    
    for model in models_to_debug:
        model_name = model.get('model_name', 'Unknown')
        
        with st.expander(f"🔧 DEBUG - {model_name}", expanded=not show_all):
            st.markdown(f"### 📊 Graphiques pour {model_name}")
            
            # Liste de toutes les méthodes de visualisation disponibles
            plot_methods = [
                ('Comparaison', 'create_comparison_plot'),
                ('Temps vs Performance', 'create_time_vs_performance_plot'), 
                ('Distribution Performance', 'create_performance_distribution'),
                ('Radar Comparaison', 'create_radar_comparison'),
                ('Feature Importance', 'create_feature_importance_plot_fixed'),
                ('Matrice Confusion', 'create_confusion_matrix'),
                ('Courbe ROC', 'create_roc_curve'),
                ('Courbe Precision-Recall', 'create_precision_recall_curve'),
                ('Courbe Calibration', 'create_calibration_plot'),
                ('Résidus', 'create_residuals_plot'),
                ('Prédictions vs Réelles', 'create_predicted_vs_actual'),
                ('Distribution Erreurs', 'create_error_distribution'),
                ('Visualisation Clusters', 'create_cluster_visualization'),
                ('Analyse Silhouette', 'create_silhouette_analysis'),
                ('Matrice Corrélation', 'create_feature_correlation_matrix'),
                ('Courbe Apprentissage', 'create_learning_curve'),
                ('Analyse SHAP', 'create_shap_analysis')
            ]
            
            # Création des graphiques
            cols = st.columns(2)
            col_idx = 0
            
            for plot_name, method_name in plot_methods:
                with cols[col_idx]:
                    try:
                        if hasattr(visualizer, method_name):
                            method = getattr(visualizer, method_name)
                            
                            # Appel avec ou sans paramètre selon la méthode
                            if method_name in ['create_comparison_plot', 'create_time_vs_performance_plot', 
                                             'create_performance_distribution', 'create_radar_comparison']:
                                fig = method()
                            else:
                                fig = method(model)
                            
                            if fig:
                                st.markdown(f"**{plot_name}**")
                                st.plotly_chart(fig, use_container_width=True, 
                                              key=f"debug_{method_name}_{model_name}")
                            else:
                                st.info(f"ℹ️ {plot_name} non disponible")
                        else:
                            st.warning(f"⚠️ Méthode {method_name} non trouvée")
                    
                    except Exception as e:
                        st.error(f"❌ Erreur {plot_name}: {str(e)[:100]}")
                
                col_idx = (col_idx + 1) % 2


# ============================================================================
# 🚀 FONCTION PRINCIPALE - VERSION COMPLÈTE
# ============================================================================

def main():
    """Point d'entrée principal - Version complète et moderne"""
    try:
        # Hero section
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 3rem; font-weight: 800; margin: 0;">
                📊 Évaluation ML Pro
            </h1>
            <p style="color: #666; font-size: 1.1rem; margin-top: 0.5rem;">
                Analyse Complète et Tableaux Détaillés - Version 4.0
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Récupération des résultats
        training_results = None
        results_data = None
        
        if hasattr(STATE, 'training_results') and STATE.training_results:
            training_results = STATE.training_results
        elif hasattr(STATE, 'ml_results') and STATE.ml_results:
            results_data = STATE.ml_results
        else:
            st.error("🚫 Aucun résultat d'entraînement disponible")
            st.info("💡 Lancez un entraînement depuis la page **Training ML**")
            
            if st.button("⚙️ Aller au Training", type="primary", use_container_width=True):
                STATE.switch(AppPage.ML_TRAINING)
            return
        
        # Extraction des données
        if training_results:
            if hasattr(training_results, 'results'):
                results_data = training_results.results
            else:
                st.error("❌ Format de résultat invalide")
                return
        
        # Validation des résultats
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
            st.error("❌ Aucun modèle n'a réussi l'entraînement")
            return
        
        # Détermination du meilleur modèle
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
        
        # Initialisation du visualiseur
        visualizer = ModelEvaluationVisualizer(results_data)
        
        # Récupération des runs MLflow
        mlflow_runs = getattr(STATE, 'mlflow_runs', [])
        
        # ============================================
        # 🎨 ONGLETS PRINCIPAUX COMPLETS
        # ============================================
        
        tab_overview, tab_details, tab_advanced, tab_mlflow, tab_export, tab_debug = st.tabs([
            "📊 Vue d'Ensemble",
            "🔍 Détails", 
            "🔬 Analyse Avancée",
            "🔗 MLflow",
            "📥 Export",
            "🐛 DEBUG"
        ])
        
        with tab_overview:
            render_tab_overview(validation, visualizer)
        
        with tab_details:
            render_tab_details(validation, visualizer)
        
        with tab_advanced:
            render_tab_advanced(validation, visualizer)
        
        with tab_mlflow:
            render_tab_mlflow(mlflow_runs)
        
        with tab_export:
            render_tab_export(validation, visualizer)
        
        with tab_debug:
            render_tab_debug(validation, visualizer)
    
    except Exception as e:
        logger.error(f"❌ Erreur page évaluation: {e}", exc_info=True)
        st.error(f"❌ Une erreur s'est produite: {str(e)[:200]}")
        
        if st.button("🔄 Rafraîchir la page"):
            st.rerun()


if __name__ == "__main__":
    main()