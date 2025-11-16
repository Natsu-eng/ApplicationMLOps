"""
🚀 ML FACTORY PRO - Interface Moderne pour ML Classique (Tabular Data)
Design unifié avec Computer Vision Training - Production Ready
Version: 3.0.0 | StateManager Refactorisé
"""
from src.shared.logging import get_logger

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
from typing import Dict, List, Any, Optional
from collections import Counter

# Imports de la logique métier
from orchestrators.ml_training_orchestrator import (
    ml_training_orchestrator,
    MLTrainingContext,
    MLTrainingResult
)
from src.models.catalog import MODEL_CATALOG
from src.data.data_analysis import detect_imbalance, auto_detect_column_types
from helpers.data_validators import DataValidator
from utils.system_utils import get_system_metrics as check_system_resources
from monitoring.state_managers import init, AppPage, TrainingStep
from monitoring.mlflow_collector import get_mlflow_collector

# Import des composants UI centralisés
from helpers.ui_components import UIComponents, TargetAnalysisHelpers
from ui.styles import UIStyles

# Initialisation du StateManager
STATE = init()
logger = get_logger(__name__)

# Configuration Streamlit
st.set_page_config(
    page_title="ML Factory Pro | ML Classique",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application du CSS moderne
st.markdown(UIStyles.get_main_css(), unsafe_allow_html=True)


class MLTrainingWorkflowPro:
    """
    Workflow moderne pour ML Classique (Tabular Data).
    Architecture refactorisée avec StateManager professionnel.
    """
    
    def __init__(self):
        self.logger = logger
        self.ui_components = UIComponents()
        self.target_helpers = TargetAnalysisHelpers()
    
    def render_header(self):
        """En-tête professionnel avec navigation et métriques"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown('<div class="main-header">🎯 ML Factory Pro</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">Workflow Intelligent pour ML Classique (Tabular Data)</div>', unsafe_allow_html=True)
        
        with col2:
            progress = ((STATE.current_step + 1) / 6) * 100
            st.markdown(UIStyles.render_progress_bar(progress, STATE.current_step + 1, 6), unsafe_allow_html=True)
        
        with col3:
            sys_metrics = check_system_resources()
            st.markdown(UIStyles.render_system_metrics(sys_metrics["memory_percent"]), unsafe_allow_html=True)
    
    def render_workflow_progress(self):
        """Barre de progression avec étapes détaillées"""
        steps = [
            {"name": "📊 Données", "icon": "📊", "description": "Dataset et Analyse"},
            {"name": "🎯 Cible", "icon": "🎯", "description": "Variable à Prédire"},
            {"name": "⚖️ Déséquilibre", "icon": "⚖️", "description": "Analyse Classes"},
            {"name": "🔧 Préprocess", "icon": "🔧", "description": "Transformation"},
            {"name": "🤖 Modèles", "icon": "🤖", "description": "Sélection Algos"},
            {"name": "🚀 Lancement", "icon": "🚀", "description": "Entraînement"}
        ]
        
        current_step = STATE.current_step
        
        st.markdown("### 📋 Workflow d'Entraînement")
        
        cols = st.columns(len(steps))
        for idx, (col, step) in enumerate(zip(cols, steps)):
            with col:
                if idx < current_step:
                    status = "completed"
                    status_icon = "✅"
                    status_color = "#28a745"
                    status_text = "Terminé"
                elif idx == current_step:
                    status = "active"
                    status_icon = "🔵"
                    status_color = "#667eea"
                    status_text = "En cours"
                else:
                    status = "pending"
                    status_icon = "⚪"
                    status_color = "#6c757d"
                    status_text = "À venir"
                
                st.markdown(
                    f"""
                    <div class="progress-step {status}">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{step['icon']}</div>
                        <div style="font-weight: bold; color: {status_color}; margin-bottom: 0.25rem; font-size: 0.9rem;">
                            {step['name']}
                        </div>
                        <div style="font-size: 0.75rem; color: #666;">{step['description']}</div>
                        <div style="font-size: 0.7rem; color: {status_color}; margin-top: 0.5rem;">
                            {status_icon} {status_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
    
    def render_dataset_analysis_step(self):
        """Étape 1: Analyse du dataset chargé - Version StateManager"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("📊 Étape 1: Analyse du Dataset")
        
        # Vérification dataset avec StateManager
        if not STATE.loaded or STATE.data.df is None:
            st.error("❌ Aucun dataset chargé")
            st.info("💡 Veuillez charger un dataset depuis le dashboard principal.")
            if st.button("📊 Aller au Dashboard", type="primary", use_container_width=True):
                STATE.switch(AppPage.DASHBOARD)
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        df = STATE.data.df
        
        # Nettoyage automatique des colonnes problématiques
        with st.spinner("🔍 Analyse automatique des colonnes en cours..."):
            constant_cols = []
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if df[col].std() == 0:
                    constant_cols.append(col)
            
            identifier_cols = [col for col in df.columns if df[col].nunique() == len(df)]
            high_missing_cols = [col for col in df.columns if df[col].isnull().mean() > 0.8]
            
            cols_to_remove = list(set(constant_cols + identifier_cols + high_missing_cols))
            cols_to_keep = [col for col in df.columns if col not in cols_to_remove]
        
        # Application du nettoyage automatique
        if cols_to_remove:
            st.markdown("### 🧹 Nettoyage Automatique des Colonnes")
            
            df_cleaned = df[cols_to_keep].copy()
            n_removed = len(cols_to_remove)
            
            st.success(f"✅ **{n_removed} colonne(s)** supprimée(s) automatiquement")
            
            with st.expander("📋 Détail des colonnes supprimées", expanded=True):
                if constant_cols:
                    st.error(f"**{len(constant_cols)} colonne(s) constante(s)**:")
                    for col in constant_cols:
                        st.markdown(f"- `{col}` (variance nulle)")
                
                if identifier_cols:
                    st.error(f"**{len(identifier_cols)} colonne(s) identifiante(s)**:")
                    for col in identifier_cols:
                        st.markdown(f"- `{col}` (100% valeurs uniques)")
                
                if high_missing_cols:
                    st.error(f"**{len(high_missing_cols)} colonne(s) avec trop de valeurs manquantes**:")
                    for col in high_missing_cols:
                        missing_pct = df[col].isnull().mean() * 100
                        st.markdown(f"- `{col}` ({missing_pct:.1f}% manquant)")
            
            # Mise à jour via StateManager
            df = df_cleaned
            STATE.data.df = df_cleaned
        else:
            st.success("✅ Aucune colonne problématique détectée - Dataset conservé intact")
        
        # Validation des données
        validation_result = DataValidator.validate_dataframe_for_ml(df)
        
        if not validation_result['is_valid']:
            st.error("❌ Dataset non compatible avec l'analyse ML")
            with st.expander("🔍 Détails des problèmes", expanded=True):
                for issue in validation_result['issues']:
                    st.error(f"• {issue}")
            
            if st.button("🔄 Recharger un nouveau dataset", type="primary"):
                STATE.switch(AppPage.DASHBOARD)
            
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Métriques principales avec UIComponents
        st.subheader("📈 Statistiques Principales")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics_data = [
            {"icon": "📏", "label": "Lignes", "value": f"{len(df):,}", "color": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"},
            {"icon": "📋", "label": "Colonnes", "value": f"{len(df.columns)}", "color": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"},
            {"icon": "💾", "label": "Mémoire", "value": f"{df.memory_usage(deep=True).sum() / (1024**2):.1f} MB", "color": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"},
            {"icon": "🕳️", "label": "Manquant", "value": f"{df.isnull().mean().mean() * 100:.1f}%", "color": "#28a745" if df.isnull().mean().mean() * 100 < 5 else "#ffc107"},
            {"icon": "🔢", "label": "Numériques", "value": f"{len(df.select_dtypes(include='number').columns)}", "color": "linear-gradient(135deg, #fa709a 0%, #fee140 100%)"}
        ]
        
        for col, metric in zip([col1, col2, col3, col4, col5], metrics_data):
            with col:
                st.markdown(
                    f"""
                    <div class='metric-card' style='background: {metric["color"]};'>
                        <h3>{metric['icon']}</h3>
                        <h4>{metric['label']}</h4>
                        <h2>{metric['value']}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Détection automatique des types de colonnes
        st.markdown("---")
        st.subheader("🔍 Analyse Automatique des Colonnes")
        
        with st.spinner("🤖 Analyse en cours..."):
            column_types = auto_detect_column_types(df)
        
        col_type1, col_type2, col_type3 = st.columns(3)
        
        with col_type1:
            st.markdown("**🔢 Colonnes Numériques**")
            if column_types.get('numeric'):
                st.success(f"✅ {len(column_types['numeric'])} colonnes détectées")
                with st.expander("📋 Voir les colonnes", expanded=False):
                    for col in column_types['numeric'][:15]:
                        st.markdown(f"- `{col}`")
                    if len(column_types['numeric']) > 15:
                        st.caption(f"... et {len(column_types['numeric']) - 15} autres")
            else:
                st.info("ℹ️ Aucune colonne numérique")
        
        with col_type2:
            st.markdown("**📝 Colonnes Catégorielles**")
            if column_types.get('categorical'):
                st.success(f"✅ {len(column_types['categorical'])} colonnes détectées")
                with st.expander("📋 Voir les colonnes", expanded=False):
                    for col in column_types['categorical'][:15]:
                        n_unique = df[col].nunique()
                        st.markdown(f"- `{col}` ({n_unique} valeurs)")
                    if len(column_types['categorical']) > 15:
                        st.caption(f"... et {len(column_types['categorical']) - 15} autres")
            else:
                st.info("ℹ️ Aucune colonne catégorielle")
        
        with col_type3:
            st.markdown("**📅 Colonnes Temporelles**")
            if column_types.get('datetime'):
                st.success(f"✅ {len(column_types['datetime'])} colonnes détectées")
                with st.expander("📋 Voir les colonnes", expanded=False):
                    for col in column_types['datetime']:
                        st.markdown(f"- `{col}`")
            else:
                st.info("ℹ️ Aucune colonne temporelle")
        
        # Initialisation robuste des features
        feature_list = df.columns.tolist()
        
        st.markdown("---")
        st.subheader("🎯 Features Disponibles")
        
        st.info(f"**{len(feature_list)} features** détectées automatiquement")
        
        with st.expander("📋 Liste complète des features", expanded=False):
            cols_display = st.columns(2)
            for idx, feature in enumerate(feature_list):
                with cols_display[idx % 2]:
                    col_type = "🔢" if feature in column_types.get('numeric', []) else "📝"
                    st.markdown(f"{col_type} `{feature}`")
        
        # Navigation avec StateManager
        st.markdown("---")
        if st.button("💾 Valider et Continuer ➡️", type="primary", use_container_width=True):
            # Sauvegarde via StateManager
            STATE.dataset_loaded = True
            STATE.dataset_info = {
                'n_rows': len(df),
                'n_cols': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
                'missing_pct': df.isnull().mean().mean() * 100,
                'column_types': column_types,
                'features_initial': feature_list,
                'cleaning_applied': len(cols_to_remove) > 0 if 'cols_to_remove' in locals() else False,
                'cols_removed': cols_to_remove if 'cols_to_remove' in locals() else []
            }
            
            STATE.feature_list = feature_list
            STATE.current_step = 1
            
            st.success("✅ Dataset validé et nettoyé avec succès!")
            time.sleep(0.5)
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_target_selection_step(self):
        """Étape 2: Sélection de la variable cible avec StateManager"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🎯 Étape 2: Sélection de la Cible")
        
        df = STATE.data.df
        
        # Sélection du type de tâche
        st.subheader("📋 Type de Problème")
        st.markdown("Sélectionnez le type d'apprentissage adapté à votre objectif")
        
        task_options = {
            "classification": {
                "name": "Classification Supervisée",
                "description": "Prédire des catégories (ex: spam/non-spam, fraude/normal, sentiment analysis)",
                "icon": "🎯",
                "color": "#28a745",
                "examples": "• Détection de fraude\n• Classification d'emails\n• Diagnostic médical"
            },
            "regression": {
                "name": "Régression Supervisée",
                "description": "Prédire des valeurs numériques continues (ex: prix, température, scores)",
                "icon": "📈",
                "color": "#17a2b8",
                "examples": "• Prédiction de prix\n• Estimation de ventes\n• Forecast météo"
            },
            "clustering": {
                "name": "Clustering Non Supervisé",
                "description": "Découvrir des groupes naturels dans les données sans labels prédéfinis",
                "icon": "🔍",
                "color": "#6c757d",
                "examples": "• Segmentation clients\n• Détection d'anomalies\n• Analyse de comportements"
            }
        }
        
        cols = st.columns(3)
        for idx, (task_key, task_info) in enumerate(task_options.items()):
            with cols[idx]:
                is_selected = STATE.task_type == task_key
                card_class = "task-card selected" if is_selected else "task-card"
                
                st.markdown(
                    f"""
                    <div class="{card_class}">
                        <div class="icon">{task_info['icon']}</div>
                        <h3 style="color: {task_info['color']}; margin: 0 0 0.5rem 0;">
                            {task_info['name']}
                        </h3>
                        <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0;">
                            {task_info['description']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                if st.button(
                    "✅ Sélectionné" if is_selected else "📝 Sélectionner",
                    key=f"task_{task_key}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    STATE.task_type = task_key
                    STATE.target_column = None
                    STATE.feature_list = []
                    st.success(f"✅ {task_info['name']} sélectionné")
                    time.sleep(0.3)
                    st.rerun()
                
                if is_selected:
                    with st.expander("💡 Cas d'usage", expanded=False):
                        st.markdown(task_info['examples'])
        
        st.markdown("---")
        
        # Configuration spécifique selon le type de tâche
        task_type = STATE.task_type
        
        if task_type in ['classification', 'regression']:
            st.subheader("🎯 Variable Cible (Y)")
            
            # Utilisation des helpers pour la sélection des cibles
            if task_type == 'classification':
                available_targets = self.target_helpers.get_classification_targets(df)
                help_text = "📊 Colonne avec classes à prédire (≤50 valeurs uniques recommandé)"
            else:
                available_targets = self.target_helpers.get_regression_targets(df)
                help_text = "📈 Colonne numérique avec valeurs continues (>10 valeurs uniques)"
            
            if not available_targets:
                st.error(f"❌ Aucune variable cible appropriée trouvée pour **{task_type}**")
                st.markdown(
                    """
                    **Critères requis:**
                    - **Classification**: Colonnes catégorielles ou numériques avec ≤50 classes
                    - **Régression**: Colonnes numériques avec >10 valeurs uniques
                    
                    💡 **Suggestion**: Vérifiez vos données ou changez de type de tâche
                    """
                )
            else:
                target_column = st.selectbox(
                    "Sélectionnez la variable à prédire",
                    options=[None] + available_targets,
                    index=([None] + available_targets).index(STATE.target_column)
                    if STATE.target_column in available_targets else 0,
                    help=help_text
                )
                
                if target_column:
                    STATE.target_column = target_column
                    
                    # Analyse de la cible avec visualisations
                    st.markdown("---")
                    st.subheader("📊 Analyse de la Variable Cible")
                    
                    col_info1, col_info2 = st.columns([2, 1])
                    
                    with col_info1:
                        if task_type == 'classification':
                            n_classes = df[target_column].nunique()
                            class_dist = df[target_column].value_counts()
                            
                            if n_classes <= 20:
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=class_dist.index.astype(str),
                                        y=class_dist.values,
                                        text=class_dist.values,
                                        textposition='auto',
                                        marker=dict(
                                            color=class_dist.values,
                                            colorscale='Viridis',
                                            line=dict(color='white', width=1)
                                        ),
                                        hovertemplate='<b>Classe: %{x}</b><br>Échantillons: %{y}<extra></extra>'
                                    )
                                ])
                                
                                fig.update_layout(
                                    title="Distribution des Classes",
                                    xaxis_title="Classe",
                                    yaxis_title="Nombre d'échantillons",
                                    template="plotly_white",
                                    height=400,
                                    showlegend=False
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(f"ℹ️ Trop de classes ({n_classes}) pour afficher le graphique")
                        
                        else:  # Régression
                            fig = self.ui_components.create_modern_histogram(df[target_column], '#667eea')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col_info2:
                        if task_type == 'classification':
                            st.markdown(
                                f"""
                                <div class='metric-card'>
                                    <h3>🎯</h3>
                                    <h4>Nombre de Classes</h4>
                                    <h2>{n_classes}</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Analyse du déséquilibre avec UIComponents
                            imbalance_info = detect_imbalance(df, target_column)
                            
                            if imbalance_info.get('is_imbalanced', False):
                                ratio = imbalance_info.get('imbalance_ratio', 0)
                                imbalance_level = self.ui_components.get_imbalance_level(ratio)
                                
                                st.markdown(
                                    f"""
                                    <div class='metric-card' style='background: {imbalance_level["color"]};'>
                                        <h3>{imbalance_level["icon"]}</h3>
                                        <h4>Déséquilibre</h4>
                                        <h2>{imbalance_level["label"]}</h2>
                                        <p style='margin-top: 0.5rem; font-size: 0.9rem;'>Ratio: {ratio:.1f}:1</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                STATE.imbalance_config['imbalance_detected'] = True
                                STATE.imbalance_config['imbalance_ratio'] = ratio
                                
                                st.info("💡 Nous analyserons ce déséquilibre à l'étape suivante")
                            else:
                                st.markdown(
                                    """
                                    <div class='metric-card' style='background: #28a745;'>
                                        <h3>✅</h3>
                                        <h4>Équilibre</h4>
                                        <h2>Bon</h2>
                                        <p style='margin-top: 0.5rem; font-size: 0.9rem;'>Classes équilibrées</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                STATE.imbalance_config['imbalance_detected'] = False
                        
                        else:  # Régression
                            target_stats = df[target_column].describe()
                            
                            st.markdown(
                                f"""
                                <div class='metric-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
                                    <h3>📊</h3>
                                    <h4>Moyenne</h4>
                                    <h2>{target_stats['mean']:.2f}</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            st.markdown(
                                f"""
                                <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                                    <h3>📈</h3>
                                    <h4>Écart-type</h4>
                                    <h2>{target_stats['std']:.2f}</h2>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                
                    st.subheader("📊 Variables Explicatives (X)")
                    if target_column:
                        available_features = [col for col in df.columns if col != target_column]
                        
                        auto_features = st.checkbox("Sélection automatique des features", value=True, key="auto_features")
                        
                        if auto_features:
                            column_types = auto_detect_column_types(df)
                            numeric_features = column_types.get('numeric', [])
                            categorical_features = [col for col in column_types.get('categorical', []) 
                                                if df[col].nunique() <= 50]
                            recommended_features = numeric_features + categorical_features
                            recommended_features = [col for col in recommended_features if col in available_features]
                            recommended_features = recommended_features[:50]
                            
                            STATE.feature_list = recommended_features
                            
                            st.success(f"✅ {len(recommended_features)} features sélectionnées automatiquement")
                            
                            with st.expander("📋 Voir les features sélectionnées", expanded=False):
                                for feat in recommended_features[:20]:
                                    st.markdown(f"- `{feat}`")
                                if len(recommended_features) > 20:
                                    st.caption(f"... et {len(recommended_features) - 20} autres")
                        else:
                            selected_features = st.multiselect(
                                "Sélectionnez les variables explicatives",
                                options=available_features,
                                default=STATE.feature_list if STATE.feature_list else [],
                                key="manual_features"
                            )
                            
                            STATE.feature_list = selected_features
                        
                        if STATE.feature_list and len(STATE.feature_list) > 0:
                            st.info(f"**{len(STATE.feature_list)}** features sélectionnées")
                        else:
                            st.warning("⚠️ Veuillez sélectionner au moins une feature")
                            
        else:  # Clustering
            STATE.target_column = None
            st.success("✅ **Clustering Non Supervisé** sélectionné")
            st.markdown(
                """
                <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                            padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;'>
                    <h4>🔍 À propos du Clustering</h4>
                    <p>Le clustering identifie automatiquement des groupes naturels dans vos données sans nécessiter de labels prédéfinis.</p>
                    <p><strong>Cas d'usage:</strong></p>
                    <ul>
                        <li>🛒 Segmentation de clients</li>
                        <li>🔍 Détection d'anomalies</li>
                        <li>📊 Analyse exploratoire de données</li>
                        <li>🎯 Identification de patterns cachés</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour", use_container_width=True):
                STATE.current_step = 0
                st.rerun()
        
        with col_nav2:
            can_continue = (
                (task_type in ['classification', 'regression'] and STATE.target_column) or
                task_type == 'clustering'
            )
            
            if st.button(
                "💾 Continuer ➡️",
                type="primary",
                use_container_width=True,
                disabled=not can_continue
            ):
                if can_continue:
                    STATE.current_step = 2
                    st.success("✅ Configuration de la cible sauvegardée!")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error("⚠️ Veuillez sélectionner une variable cible")
        
        st.markdown('</div>', unsafe_allow_html=True)
    

    def render_imbalance_analysis_step(self):
        """Étape 3: Analyse et correction du déséquilibre"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("⚖️ Étape 3: Gestion du Déséquilibre")
        
        df = STATE.data.df
        task_type = STATE.task_type
        target_column = STATE.target_column
        
        # Si pas classification, skip automatiquement
        if task_type != 'classification':
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #17a2b815 0%, #13849615 100%); 
                            padding: 2rem; border-radius: 15px; text-align: center;'>
                    <h3>ℹ️ Cette étape ne s'applique qu'à la classification</h3>
                    <p>Type actuel: <strong>{task_type.upper()}</strong></p>
                    <p>Vous pouvez passer directement à l'étape suivante.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_nav1, col_nav2 = st.columns(2)
            with col_nav1:
                if st.button("⬅️ Retour", use_container_width=True):
                    STATE.current_step = 1
                    st.rerun()
            with col_nav2:
                if st.button("Passer cette étape ➡️", type="primary", use_container_width=True):
                    STATE.current_step = 3
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Analyse du déséquilibre
        imbalance_info = detect_imbalance(df, target_column)
        class_counts = df[target_column].value_counts()
        total_samples = len(df)
        ratio = imbalance_info.get('imbalance_ratio', 1.0)
        
        # Niveau déséquilibre dynamique
        def get_imbalance_level_dynamic(ratio: float) -> Dict[str, str]:
            """Retourne niveau, couleur, icône selon le ratio"""
            if ratio < 1.5:
                return {
                    "level": "Équilibré",
                    "color": "#28a745",
                    "icon": "✅",
                    "severity": "low",
                    "description": "Distribution saine, pas d'action requise"
                }
            elif ratio < 3:
                return {
                    "level": "Léger",
                    "color": "#ffc107",
                    "icon": "⚠️",
                    "severity": "medium",
                    "description": "Déséquilibre modéré, poids de classe recommandés"
                }
            elif ratio < 10:
                return {
                    "level": "Modéré",
                    "color": "#ff9800",
                    "icon": "⚠️",
                    "severity": "high",
                    "description": "Déséquilibre important, SMOTE + poids recommandés"
                }
            else:
                return {
                    "level": "Sévère",
                    "color": "#dc3545",
                    "icon": "❌",
                    "severity": "critical",
                    "description": "Déséquilibre critique, resampling obligatoire"
                }
        
        imbalance_level = get_imbalance_level_dynamic(ratio)
        
        # MÉTRIQUES PRINCIPALES DYNAMIQUES
        st.subheader("📊 Analyse du Déséquilibre")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div class='metric-card' style='background: {imbalance_level["color"]}; animation: pulse 2s infinite;'>
                    <h3 style='font-size: 2.5rem;'>{imbalance_level["icon"]}</h3>
                    <h4>Niveau de Déséquilibre</h4>
                    <h2>{imbalance_level["level"]}</h2>
                    <p style='margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.9;'>
                        {imbalance_level["description"]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <h3>⚖️</h3>
                    <h4>Ratio de Déséquilibre</h4>
                    <h2>{ratio:.1f}:1</h2>
                    <p style='margin-top: 0.5rem; font-size: 0.85rem; opacity: 0.9;'>
                        Classe majoritaire vs minoritaire
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <h3>📊</h3>
                    <h4>Échantillons Total</h4>
                    <h2>{total_samples:,}</h2>
                    <p style='margin-top: 0.5rem; font-size: 0.85rem; opacity: 0.9;'>
                        Lignes d'entraînement
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # GRAPHIQUE DISTRIBUTION DYNAMIQUE
        st.markdown("---")
        st.subheader("📈 Distribution des Classes")
        
        # Création graphique avec couleurs adaptées au déséquilibre
        colors = []
        max_count = class_counts.max()
        min_count = class_counts.min()
        
        for count in class_counts.values:
            if count == max_count and ratio > 2:
                colors.append('#dc3545')  # Rouge pour classe majoritaire
            elif count == min_count and ratio > 2:
                colors.append('#ffc107')  # Jaune pour classe minoritaire
            else:
                colors.append('#667eea')  # Bleu standard
        
        fig = go.Figure(data=[
            go.Bar(
                x=class_counts.index.astype(str),
                y=class_counts.values,
                text=class_counts.values,
                textposition='auto',
                marker=dict(
                    color=colors,
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>Classe: %{x}</b><br>Échantillons: %{y}<br>Pourcentage: %{customdata:.1f}%<extra></extra>',
                customdata=[(count/total_samples)*100 for count in class_counts.values]
            )
        ])
        
        fig.update_layout(
            title={
                'text': f"Distribution des Classes (Ratio {ratio:.1f}:1)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': imbalance_level['color']}
            },
            xaxis_title="Classe",
            yaxis_title="Nombre d'échantillons",
            template="plotly_white",
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Ligne de référence pour équilibre
        avg_count = total_samples / len(class_counts)
        fig.add_hline(
            y=avg_count,
            line_dash="dash",
            line_color="green",
            annotation_text="Équilibre idéal",
            annotation_position="top right"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ALERTES CONTEXTUELLES
        if ratio > 10:
            st.error(
                f"🚨 **ATTENTION** : Déséquilibre CRITIQUE détecté (ratio {ratio:.1f}:1)\n\n"
                f"La classe minoritaire représente seulement **{(class_counts.min()/total_samples)*100:.1f}%** des données. "
                f"**Actions fortement recommandées** : SMOTE + Poids de classe"
            )
        elif ratio > 3:
            st.warning(
                f"⚠️ **Déséquilibre modéré** détecté (ratio {ratio:.1f}:1)\n\n"
                f"Nous recommandons d'activer **SMOTE** et les **poids de classe** pour de meilleures performances."
            )
        elif ratio > 1.5:
            st.info(
                f"ℹ️ **Léger déséquilibre** détecté (ratio {ratio:.1f}:1)\n\n"
                f"Les **poids de classe** suffisent généralement pour ce niveau."
            )
        else:
            st.success(
                f"✅ **Distribution équilibrée** (ratio {ratio:.1f}:1)\n\n"
                f"Aucune correction nécessaire, mais vous pouvez tester les options ci-dessous."
            )
        
        # Stratégies de correction
        st.markdown("---")
        st.markdown("### 🎯 Stratégies de Correction")
        
        col_strat1, col_strat2 = st.columns(2)
        
        with col_strat1:
            st.markdown("#### ⚖️ Poids de Classe Automatiques")
            st.markdown(
                """
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745;'>
                    <p><strong>Principe:</strong> Ajuste la fonction de perte pour donner plus d'importance aux classes minoritaires.</p>
                    <p><strong>✅ Avantages:</strong></p>
                    <ul>
                        <li>Ne modifie pas les données</li>
                        <li>Rapide à appliquer</li>
                        <li>Fonctionne avec tous les modèles</li>
                    </ul>
                    <p><strong>⚠️ Limites:</strong> Peut sur-ajuster sur classes rares si ratio > 10:1</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # SUGGESTION AUTOMATIQUE basée sur ratio
            default_class_weights = ratio >= 2
            
            use_class_weights = st.checkbox(
                "✅ Activer les poids de classe",
                value=default_class_weights,
                help=f"{'✅ RECOMMANDÉ' if ratio >= 2 else '⚪ Optionnel'} (ratio actuel: {ratio:.1f}:1)",
                key="use_class_weights_checkbox"
            )
            
            STATE.imbalance_config['use_class_weights'] = use_class_weights
            
            if use_class_weights:
                st.success("✅ Les poids seront calculés automatiquement : sklearn 'balanced' mode")
                
                # Aperçu des poids calculés
                from sklearn.utils.class_weight import compute_class_weight
                weights = compute_class_weight('balanced', classes=np.unique(df[target_column]), y=df[target_column])
                weight_dict = dict(zip(np.unique(df[target_column]), weights))
                
                with st.expander("🔍 Aperçu des poids calculés", expanded=False):
                    for cls, weight in weight_dict.items():
                        st.markdown(f"- Classe `{cls}`: poids **{weight:.2f}x**")
        
        with col_strat2:
            st.markdown("#### 🎭 SMOTE (Suréchantillonnage Synthétique)")
            st.markdown(
                """
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #17a2b8;'>
                    <p><strong>Principe:</strong> Génère des exemples synthétiques pour les classes minoritaires via interpolation.</p>
                    <p><strong>✅ Avantages:</strong></p>
                    <ul>
                        <li>Augmente les données de manière intelligente</li>
                        <li>Améliore la généralisation</li>
                        <li>Réduit l'overfitting sur minorité</li>
                    </ul>
                    <p><strong>⚠️ Limites:</strong> Peut introduire du bruit si classes se chevauchent</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # VALIDATION AUTOMATIQUE k_neighbors
            min_class_count = class_counts.min()
            max_k_safe = max(1, min_class_count - 1)
            
            # Suggestion automatique
            default_smote = ratio >= 3 and min_class_count > 5
            
            use_smote = st.checkbox(
                "✅ Activer SMOTE",
                value=default_smote,
                help=f"{'✅ RECOMMANDÉ' if ratio >= 3 else '⚪ Optionnel'} (ratio: {ratio:.1f}:1, min classe: {min_class_count})",
                key="use_smote_checkbox"
            )
            
            smote_k_neighbors = 5
            
            if use_smote:
                if min_class_count < 6:
                    st.error(
                        f"❌ **SMOTE IMPOSSIBLE** : Classe minoritaire trop petite ({min_class_count} échantillons)\n\n"
                        f"SMOTE nécessite au moins **6 échantillons** par classe. "
                        f"Veuillez collecter plus de données ou utiliser uniquement les poids de classe."
                    )
                    use_smote = False
                    STATE.preprocessing_config['use_smote'] = False
                    STATE.imbalance_config['use_smote'] = False
                else:
                    st.markdown("**⚙️ Configuration SMOTE**")
                    
                    # k_neighbors avec validation dynamique
                    smote_k_neighbors = st.slider(
                        "Nombre de voisins (k)",
                        min_value=1,
                        max_value=min(20, max_k_safe),
                        value=min(5, max_k_safe),
                        help=f"Maximum autorisé: {max_k_safe} (basé sur classe minoritaire de {min_class_count} échantillons)"
                    )
                    
                    # 🆕 ESTIMATION PRÉCISE du nombre de samples générés
                    minority_classes = class_counts[class_counts < class_counts.max()]
                    estimated_synthetic = 0
                    for count in minority_classes:
                        estimated_synthetic += (class_counts.max() - count)
                    
                    st.info(
                        f"💡 **Estimation** : ~**{estimated_synthetic:,} exemples synthétiques** seront générés\n\n"
                        f"📊 Nouvelle distribution après SMOTE:\n"
                        f"- Classe majoritaire: {class_counts.max():,} (inchangé)\n"
                        f"- Classes minoritaires: ≈{class_counts.max():,} (équilibrées)\n"
                        f"- **Total après SMOTE**: ≈{total_samples + estimated_synthetic:,} lignes"
                    )
                    
                    # SAUVEGARDE DANS STATE avec validation
                    STATE.preprocessing_config['use_smote'] = True
                    STATE.preprocessing_config['smote_k_neighbors'] = smote_k_neighbors
                    STATE.preprocessing_config['smote_sampling_strategy'] = 'auto'
                    
                    STATE.imbalance_config['use_smote'] = True
                    STATE.imbalance_config['smote_k_neighbors'] = smote_k_neighbors
                    
                    st.success(f"✅ SMOTE configuré (k={smote_k_neighbors})")
            else:
                STATE.preprocessing_config['use_smote'] = False
                STATE.imbalance_config['use_smote'] = False
        
        # RECOMMANDATIONS INTELLIGENTES
        if ratio > 5:
            st.markdown("---")
            st.markdown("### 💡 Recommandations Personnalisées")
            
            recommendations = []
            
            if ratio > 10:
                recommendations.append(
                    "🚨 **CRITIQUE** : Ratio > 10:1 → Activez **SMOTE + Poids de classe** obligatoirement"
                )
                recommendations.append(
                    "📊 **Alternative** : Si possible, collectez plus de données pour la classe minoritaire"
                )
            
            if min_class_count < 50:
                recommendations.append(
                    f"⚠️ Classe minoritaire très petite ({min_class_count} échantillons) → Risque d'overfitting élevé"
                )
            
            if use_smote and use_class_weights:
                recommendations.append(
                    "✅ **Configuration optimale** détectée : SMOTE + Poids combinés pour maximum d'efficacité"
                )
            elif ratio > 3 and not (use_smote or use_class_weights):
                recommendations.append(
                    "⚠️ **Attention** : Aucune correction activée avec ratio élevé → Performances réduites attendues"
                )
            
            if len(class_counts) > 10:
                recommendations.append(
                    f"🎯 **Multi-classes** détecté ({len(class_counts)} classes) → SMOTE appliquera une stratégie intelligente"
                )
            
            for rec in recommendations:
                st.warning(rec)
        
        # TABLEAU RÉCAPITULATIF
        st.markdown("---")
        st.markdown("### 📋 Récapitulatif Configuration")
        
        recap_data = {
            "Paramètre": [
                "Ratio déséquilibre",
                "Niveau",
                "Classe majoritaire",
                "Classe minoritaire",
                "Poids de classe",
                "SMOTE",
                "k_neighbors SMOTE"
            ],
            "Valeur": [
                f"{ratio:.2f}:1",
                imbalance_level['level'],
                f"{class_counts.max():,} échantillons",
                f"{class_counts.min():,} échantillons",
                "✅ Activé" if use_class_weights else "❌ Désactivé",
                "✅ Activé" if use_smote else "❌ Désactivé",
                f"{smote_k_neighbors}" if use_smote else "N/A"
            ],
            "Statut": [
                imbalance_level['icon'],
                imbalance_level['icon'],
                "📊",
                "⚠️" if class_counts.min() < 50 else "📊",
                "✅" if use_class_weights else "⚪",
                "✅" if use_smote else "⚪",
                "✅" if use_smote else "⚪"
            ]
        }
        
        st.dataframe(
            pd.DataFrame(recap_data),
            hide_index=True,
            use_container_width=True
        )
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour", use_container_width=True):
                STATE.current_step = 1
                st.rerun()
        
        with col_nav2:
            if st.button("💾 Sauvegarder et Continuer ➡️", type="primary", use_container_width=True):
                # SAUVEGARDE COMPLÈTE avec validation
                STATE.imbalance_config.update({
                    'use_class_weights': use_class_weights,
                    'use_smote': use_smote,
                    'smote_k_neighbors': smote_k_neighbors if use_smote else 5,
                    'smote_sampling_strategy': 'auto',
                    'imbalance_ratio': float(ratio),
                    'imbalance_level': imbalance_level['level'],
                    'min_class_count': int(class_counts.min()),
                    'max_class_count': int(class_counts.max())
                })
                
                # Log pour debug
                logger.info(f"✅ Imbalance config sauvegardée: {STATE.imbalance_config}")
                
                STATE.current_step = 3
                st.success("✅ Configuration du déséquilibre sauvegardée avec succès!")
                time.sleep(0.5)
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    def render_preprocessing_step(self):
        """Étape 4: Configuration du prétraitement avec StateManager"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🔧 Étape 4: Prétraitement des Données")
        
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
                <h4>📋 À propos du Prétraitement</h4>
                <p>Les transformations seront appliquées <strong>séparément</strong> sur train/test pour éviter le <em>data leakage</em>.</p>
                <p>✅ <strong>Bonne pratique:</strong> fit() sur train, transform() sur val/test</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Analyse des features sélectionnées
        df = STATE.data.df
        feature_list = STATE.feature_list
        
        if feature_list:
            numeric_features = [col for col in feature_list 
                            if col in df.select_dtypes(include=['number']).columns]
            categorical_features = [col for col in feature_list 
                                if col not in numeric_features]
        else:
            numeric_features = df.select_dtypes(include=['number']).columns.tolist()
            categorical_features = df.select_dtypes(exclude=['number']).columns.tolist()
        
        n_numeric = len(numeric_features)
        n_categorical = len(categorical_features)
        
        # Résumé des features
        st.markdown("### 📊 Analyse des Variables Sélectionnées")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("🔢 Variables Numériques", n_numeric)
            if n_numeric > 0:
                with st.expander("📋 Voir les variables", expanded=False):
                    for col in numeric_features[:10]:
                        st.markdown(f"- `{col}`")
                    if n_numeric > 10:
                        st.caption(f"... et {n_numeric - 10} autres")
        
        with col_info2:
            st.metric("📝 Variables Catégorielles", n_categorical)
            if n_categorical > 0:
                with st.expander("📋 Voir les variables", expanded=False):
                    for col in categorical_features[:10]:
                        st.markdown(f"- `{col}`")
                    if n_categorical > 10:
                        st.caption(f"... et {n_categorical - 10} autres")
        
        with col_info3:
            total_features = n_numeric + n_categorical
            st.metric("📊 Total Features", total_features)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧩 Gestion des Valeurs Manquantes")
            
            if n_numeric > 0:
                st.markdown("**Variables Numériques**")
                STATE.preprocessing_config['numeric_imputation'] = st.selectbox(
                    "Stratégie d'imputation",
                    options=['mean', 'median', 'constant', 'knn'],
                    index=['mean', 'median', 'constant', 'knn'].index(
                        STATE.preprocessing_config.get('numeric_imputation', 'mean')
                    ),
                    help="• **mean**: Moyenne\n• **median**: Médiane (robuste)\n• **constant**: 0\n• **knn**: k-voisins",
                    key="numeric_imp_select"
                )
            else:
                st.info("ℹ️ Aucune variable numérique sélectionnée")
                STATE.preprocessing_config['numeric_imputation'] = 'mean'
            
            if n_categorical > 0:
                st.markdown("**Variables Catégorielles**")
                STATE.preprocessing_config['categorical_imputation'] = st.selectbox(
                    "Stratégie d'imputation",
                    options=['most_frequent', 'constant'],
                    index=['most_frequent', 'constant'].index(
                        STATE.preprocessing_config.get('categorical_imputation', 'most_frequent')
                    ),
                    help="• **most_frequent**: Mode\n• **constant**: 'missing'",
                    key="cat_imp_select"
                )
            else:
                st.info("ℹ️ Aucune variable catégorielle sélectionnée")
                STATE.preprocessing_config['categorical_imputation'] = 'most_frequent'
            
            st.markdown("---")
            
            st.subheader("🧹 Nettoyage des Colonnes")
            
            STATE.preprocessing_config['remove_constant_cols'] = st.checkbox(
                "🗑️ Supprimer colonnes constantes",
                value=STATE.preprocessing_config.get('remove_constant_cols', True),
                help="Élimine colonnes sans variance"
            )
            
            STATE.preprocessing_config['remove_identifier_cols'] = st.checkbox(
                "🔑 Supprimer colonnes identifiantes",
                value=STATE.preprocessing_config.get('remove_identifier_cols', True),
                help="Élimine colonnes avec 100% valeurs uniques"
            )
        
        with col2:
            st.subheader("📏 Normalisation des Features")
            
            if n_numeric > 0:
                STATE.preprocessing_config['scale_features'] = st.checkbox(
                    "✅ Activer la normalisation",
                    value=STATE.preprocessing_config.get('scale_features', True),
                    help=f"⚡ Recommandé pour SVM, KNN, réseaux de neurones\n\n📊 S'appliquera aux {n_numeric} variables numériques"
                )
                
                if STATE.preprocessing_config.get('scale_features', True):
                    STATE.preprocessing_config['scaling_method'] = st.selectbox(
                        "Méthode de normalisation",
                        options=['standard', 'minmax', 'robust'],
                        index=['standard', 'minmax', 'robust'].index(
                            STATE.preprocessing_config.get('scaling_method', 'standard')
                        ),
                        help=(
                            "• **standard**: (x-mean)/std → Centre à 0, variance 1\n"
                            "• **minmax**: [0,1] → Normalisation min-max\n"
                            "• **robust**: Médiane et IQR → Résistant aux outliers"
                        )
                    )
                    
                    st.info(f"📊 **{n_numeric}** variables numériques seront normalisées")
                    
                    if n_categorical > 0:
                        st.success(
                            f"✅ **{n_categorical}** variables catégorielles seront encodées "
                            f"(One-Hot ou Label Encoding) mais **PAS** normalisées"
                        )
            else:
                STATE.preprocessing_config['scale_features'] = False
                st.warning(
                    "⚠️ **Normalisation désactivée**\n\n"
                    "Aucune variable numérique dans votre sélection. "
                    "La normalisation ne s'applique qu'aux variables numériques."
                )
            
            st.markdown("---")
            
            st.subheader("🔍 Réduction Dimensionnelle")
            
            if n_numeric > 10:
                STATE.preprocessing_config['pca_preprocessing'] = st.checkbox(
                    "🎯 Activer PCA",
                    value=STATE.preprocessing_config.get('pca_preprocessing', False),
                    help=f"Réduction dimensionnelle pour {n_numeric} variables numériques (>10)"
                )
                
                if STATE.preprocessing_config.get('pca_preprocessing', False):
                    st.success(f"✅ PCA sera appliqué sur les **{n_numeric}** variables numériques")
                    
                    pca_variance_threshold = st.slider(
                        "Seuil de variance expliquée (%)",
                        min_value=70,
                        max_value=99,
                        value=STATE.preprocessing_config.get('pca_variance_threshold', 95),
                        help="Pourcentage de variance à conserver"
                    )
                    STATE.preprocessing_config['pca_variance_threshold'] = pca_variance_threshold
            else:
                STATE.preprocessing_config['pca_preprocessing'] = False
                if n_numeric > 0:
                    st.info(f"ℹ️ PCA non recommandé ({n_numeric} variables numériques < 10)")
                else:
                    st.info("ℹ️ PCA ne s'applique qu'aux variables numériques")
        
        # Récapitulatif des transformations
        st.markdown("---")
        st.subheader("📋 Récapitulatif des Transformations")
        
        transformations = []
        
        if n_numeric > 0:
            transformations.append(
                f"🔢 **Variables Numériques ({n_numeric}):**\n"
                f"  - Imputation: `{STATE.preprocessing_config.get('numeric_imputation', 'mean')}`\n"
                f"  - Normalisation: `{'✅ ' + STATE.preprocessing_config.get('scaling_method', 'standard') if STATE.preprocessing_config.get('scale_features') else '❌ Désactivée'}`"
            )
            
            if STATE.preprocessing_config.get('pca_preprocessing', False):
                variance = STATE.preprocessing_config.get('pca_variance_threshold', 95)
                transformations.append(f"  - PCA: `✅ {variance}% variance`")
        
        if n_categorical > 0:
            transformations.append(
                f"📝 **Variables Catégorielles ({n_categorical}):**\n"
                f"  - Imputation: `{STATE.preprocessing_config.get('categorical_imputation', 'most_frequent')}`\n"
                f"  - Encodage: `✅ Automatique (One-Hot/Label)`\n"
                f"  - Normalisation: `❌ Non applicable`"
            )
        
        if STATE.preprocessing_config.get('remove_constant_cols', True):
            transformations.append("🧹 **Nettoyage:** Suppression colonnes constantes")
        
        if STATE.preprocessing_config.get('remove_identifier_cols', True):
            transformations.append("🧹 **Nettoyage:** Suppression colonnes identifiantes")
        
        if transformations:
            for transform in transformations:
                st.markdown(transform)
        else:
            st.info("ℹ️ Aucune transformation configurée")
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour", use_container_width=True):
                STATE.current_step = 2
                st.rerun()
        
        with col_nav2:
            if st.button("💾 Sauvegarder et Continuer ➡️", type="primary", use_container_width=True):
                STATE.preprocessing_config['n_numeric_features'] = n_numeric
                STATE.preprocessing_config['n_categorical_features'] = n_categorical
                STATE.preprocessing_config['numeric_features'] = numeric_features
                STATE.preprocessing_config['categorical_features'] = categorical_features
                
                STATE.current_step = 4
                st.success("✅ Configuration du prétraitement sauvegardée!")
                time.sleep(0.3)
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    def render_model_selection_step(self):
        """Étape 5: Sélection des algorithmes avec StateManager"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🤖 Étape 5: Sélection des Modèles")
        
        task_type = STATE.task_type
        
        # Récupération des modèles disponibles
        available_models = MODEL_CATALOG.get(task_type, {})
        
        if not available_models:
            st.error(f"❌ Aucun modèle disponible pour la tâche '{task_type}'")
            st.info("💡 Vérifiez la configuration du catalogue de modèles.")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        st.markdown(
            f"""
            <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
                <h4>🎯 Stratégie de Sélection</h4>
                <p>Nous vous recommandons de sélectionner <strong>3-5 modèles</strong> variés pour une comparaison robuste.</p>
                <p>✅ <strong>Bonnes pratiques:</strong> Combinez modèles simples (baseline) et complexes (performance)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Organisation des modèles par catégorie
        model_categories = {
            "🧠 Classiques": [],
            "🌳 Ensemble": [],
            "📈 Linéaires": [],
            "🔍 Clustering": []
        }
        
        for model_name, config in available_models.items():
            category = config.get('category', '🧠 Classiques')
            if category not in model_categories:
                category = '🧠 Classiques'
            model_categories[category].append((model_name, config))
        
        # Affichage des modèles par catégorie
        selected_models = STATE.selected_models.copy() if STATE.selected_models else []
        
        for category, models in model_categories.items():
            if not models:
                continue
                
            st.markdown(f"### {category}")
            
            cols = st.columns(3)
            col_idx = 0
            
            for model_name, config in models:
                with cols[col_idx]:
                    is_selected = model_name in selected_models
                    
                    complexity = config.get('complexity', 'medium')
                    if complexity == 'low':
                        color = "#28a745"
                        complexity_icon = "🟢"
                    elif complexity == 'high':
                        color = "#dc3545" 
                        complexity_icon = "🔴"
                    else:
                        color = "#ffc107"
                        complexity_icon = "🟡"
                    
                    card_class = "model-card selected" if is_selected else "model-card"
                    
                    st.markdown(
                        f"""
                        <div class="{card_class}" onclick="this.classList.toggle('selected')">
                            <h4 style="color: {color}; margin: 0 0 0.5rem 0;">{model_name}</h4>
                            <p style="color: #666; font-size: 0.85rem; margin: 0.5rem 0;">
                                {config.get('description', 'Description non disponible')}
                            </p>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                                <span style="font-size: 0.8rem; color: #999;">
                                    {complexity_icon} {complexity.upper()}
                                </span>
                                <span style="font-size: 0.8rem; color: #667eea;">
                                    ⚡ {config.get('training_speed', 'medium')}
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    if st.button(
                        "✅ Sélectionné" if is_selected else "📝 Sélectionner",
                        key=f"select_{model_name}",
                        use_container_width=True,
                        type="primary" if is_selected else "secondary"
                    ):
                        if is_selected:
                            selected_models.remove(model_name)
                        else:
                            selected_models.append(model_name)
                        STATE.selected_models = selected_models
                        st.success(f"{'✅ Ajouté' if not is_selected else '❌ Retiré'} : {model_name}")
                        time.sleep(0.5)
                        st.rerun()
                
                col_idx = (col_idx + 1) % 3
        
        # Résumé de la sélection
        st.markdown("---")
        st.subheader("📋 Résumé de la Sélection")
        
        if selected_models:
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            
            with col_sum1:
                n_models = len(selected_models)
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <h3>🤖</h3>
                        <h4>Modèles Sélectionnés</h4>
                        <h2>{n_models}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_sum2:
                complexities = []
                for model_name in selected_models:
                    config = available_models[model_name]
                    complexity = config.get('complexity', 'medium')
                    if complexity == 'low':
                        complexities.append(1)
                    elif complexity == 'medium':
                        complexities.append(2)
                    else:
                        complexities.append(3)
                
                avg_complexity = np.mean(complexities) if complexities else 0
                if avg_complexity < 1.5:
                    complexity_level = "Faible"
                    color = "#28a745"
                elif avg_complexity < 2.5:
                    complexity_level = "Moyenne"
                    color = "#ffc107"
                else:
                    complexity_level = "Élevée"
                    color = "#dc3545"
                
                st.markdown(
                    f"""
                    <div class='metric-card' style='background: {color};'>
                        <h3>📊</h3>
                        <h4>Complexité Moyenne</h4>
                        <h2>{complexity_level}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_sum3:
                base_time = len(selected_models) * 30
                if STATE.optimize_hyperparams:
                    base_time *= 3
                if STATE.preprocessing_config.get('pca_preprocessing', False):
                    base_time *= 1.2
                
                minutes = max(1, int(base_time / 60))
                
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <h3>⏱️</h3>
                        <h4>Temps Estimé</h4>
                        <h2>{minutes} min</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with st.expander("📋 Détail des modèles sélectionnés", expanded=True):
                cols = st.columns(3)
                for idx, model_name in enumerate(selected_models):
                    with cols[idx % 3]:
                        config = available_models[model_name]
                        st.markdown(f"**{model_name}**")
                        st.caption(f"• {config.get('description', '')}")
                        st.caption(f"• Complexité: {config.get('complexity', 'medium')}")
                        st.caption(f"• Vitesse: {config.get('training_speed', 'medium')}")
            
            if len(selected_models) > 5:
                st.warning("⚠️ Nombre élevé de modèles sélectionnés")
                st.info("💡 Pour un entraînement plus rapide, sélectionnez 3-5 modèles maximum")
            
            if len(selected_models) == 1:
                st.info("💡 Nous recommandons de sélectionner au moins 2-3 modèles pour comparaison")
        
        else:
            st.warning("⚠️ Aucun modèle sélectionné")
            st.info("💡 Sélectionnez au moins un modèle pour continuer")
        
        # Configuration avancée
        st.markdown("---")
        st.subheader("⚙️ Configuration Avancée")
        
        col_adv1, col_adv2 = st.columns(2)
        
        with col_adv1:
            if task_type != 'clustering':
                test_size = st.slider(
                    "Pourcentage de test",
                    min_value=10,
                    max_value=40,
                    value=STATE.test_size,
                    help="Pourcentage de données réservées pour l'évaluation finale"
                )
                STATE.test_size = test_size
                st.info(f"📊 Split: {100-test_size}% train, {test_size}% test")
            else:
                st.info("🔍 Clustering: 100% des données utilisées (pas de split)")
        
        with col_adv2:
            optimize = st.checkbox(
                "🔍 Optimisation des hyperparamètres",
                value=STATE.optimize_hyperparams,
                help="Recherche automatique des meilleurs paramètres (×3 temps d'entraînement)"
            )
            STATE.optimize_hyperparams = optimize
            
            if optimize:
                st.warning("⏰ Temps d'entraînement multiplié par 3-5x")
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour", use_container_width=True):
                STATE.current_step = 3
                st.rerun()
        
        with col_nav2:
            can_continue = len(selected_models) > 0
            
            if st.button(
                "💾 Sauvegarder et Continuer ➡️",
                type="primary",
                use_container_width=True,
                disabled=not can_continue
            ):
                if can_continue:
                    STATE.current_step = 5
                    st.success("✅ Sélection des modèles sauvegardée!")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error("⚠️ Veuillez sélectionner au moins un modèle")
        
        st.markdown('</div>', unsafe_allow_html=True)


    def _estimate_training_resources(self, df, n_models, sys_metrics):
        """Estime les ressources nécessaires pour l'entraînement"""
        if df is None:
            return {
                "has_enough_resources": True,
                "estimated_memory_mb": 0,
                "available_memory_mb": sys_metrics.get("memory_total_mb", 1000),
                "warnings": ["Dataset non disponible pour estimation"],
                "issues": []
            }
        
        n_samples = len(df)
        n_features = len(df.columns)
        
        # Estimation mémoire basique
        base_memory = n_samples * n_features * 8 / (1024**2)  # MB
        estimated_memory = base_memory * n_models * (3 if STATE.optimize_hyperparams else 1)
        
        # Ressources disponibles
        available_memory = (100 - sys_metrics.get("memory_percent", 80)) / 100 * sys_metrics.get("memory_total_mb", 1000)
        
        # Seuils
        memory_ok = estimated_memory < available_memory * 0.7
        cpu_ok = sys_metrics.get("cpu_percent", 0) < 80
        
        issues = []
        warnings = []
        
        if not memory_ok:
            issues.append(f"Mémoire insuffisante: {estimated_memory:.1f}MB estimés vs {available_memory:.1f}MB disponibles")
        
        if not cpu_ok:
            warnings.append("CPU très utilisé - entraînement potentiellement lent")
        
        if n_models > 5:
            warnings.append("Nombre élevé de modèles - temps d'entraînement prolongé")
        
        if n_samples > 100000 and n_models > 3:
            warnings.append("Grand dataset avec plusieurs modèles - temps d'entraînement très long")
        
        return {
            "has_enough_resources": memory_ok and cpu_ok,
            "estimated_memory_mb": estimated_memory,
            "available_memory_mb": available_memory,
            "warnings": warnings,
            "issues": issues
        }

    def render_training_launch_step(self):
        """Étape 6: Lancement de l'entraînement avec StateManager"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🚀 Étape 6: Lancement de l'Entraînement")
        
        # Récapitulatif de la configuration
        st.subheader("📋 Récapitulatif de la Configuration")
        
        col_recap1, col_recap2 = st.columns(2)
        
        with col_recap1:
            st.markdown("#### 🎯 Configuration de la Tâche")
            st.markdown(f"- **Type**: `{STATE.task_type.upper()}`")
            if STATE.target_column:
                st.markdown(f"- **Variable cible**: `{STATE.target_column}`")
            
            feature_list = STATE.feature_list
            st.markdown(f"- **Features**: `{len(feature_list)}` variables")
            
            if STATE.task_type != 'clustering':
                test_size = STATE.test_size
                st.markdown(f"- **Split test**: `{test_size}%`")
        
        with col_recap2:
            st.markdown("#### 🤖 Configuration des Modèles")
            
            selected_models = STATE.selected_models
            st.markdown(f"- **Modèles sélectionnés**: `{len(selected_models)}`")
            
            optimize = STATE.optimize_hyperparams
            st.markdown(f"- **Optimisation HP**: `{'✅ Oui' if optimize else '❌ Non'}`")
            
            if STATE.task_type == 'classification':
                use_smote = STATE.preprocessing_config.get('use_smote', False)
                use_weights = STATE.imbalance_config.get('use_class_weights', False)
                
                st.markdown(f"- **SMOTE**: `{'✅ Activé' if use_smote else '❌ Désactivé'}`")
                st.markdown(f"- **Poids de classe**: `{'✅ Activés' if use_weights else '❌ Désactivés'}`")
        
        st.markdown("---")
        st.subheader("🔍 Vérification Finale")
        
        validation_issues = []
        
        # Vérification des données
        if not STATE.loaded or STATE.data.df is None:
            validation_issues.append("❌ Aucun dataset chargé")
        
        # Vérification de la configuration
        if STATE.task_type in ['classification', 'regression']:
            if not STATE.target_column:
                validation_issues.append("❌ Variable cible non définie")
            
            if not feature_list or len(feature_list) == 0:
                validation_issues.append("❌ Aucune feature sélectionnée")
                
                # Diagnostic approfondi
                if st.checkbox("🔍 Diagnostic approfondi", value=True, key="deep_debug"):
                    st.warning("🔍 **Diagnostic des features manquantes**")
                    
                    possible_sources = {
                        "STATE.feature_list": STATE.feature_list,
                        "STATE.data.feature_list": STATE.data.feature_list,
                        "Colonnes du DataFrame": list(STATE.data.df.columns) if STATE.loaded and STATE.data.df is not None else None
                    }
                    
                    st.json(possible_sources)
                    
                    # Suggestion de récupération automatique
                    if STATE.data.df is not None and STATE.target_column:
                        auto_features = [col for col in STATE.data.df.columns if col != STATE.target_column]
                        if auto_features:
                            st.info(f"💡 **{len(auto_features)} features détectées automatiquement**")
                            if st.button("🔧 Utiliser ces features automatiquement", key="auto_fix_features"):
                                STATE.feature_list = auto_features
                                st.success(f"✅ {len(auto_features)} features restaurées!")
                                time.sleep(1)
                                st.rerun()
        else:
            # Clustering : pas besoin de target ni features spécifiques
            pass
        
        if not selected_models or len(selected_models) == 0:
            validation_issues.append("❌ Aucun modèle sélectionné")
        
        can_launch = len(validation_issues) == 0
        
        if validation_issues:
            st.error("**Problèmes de configuration détectés:**")
            for issue in validation_issues:
                st.error(issue)
            st.info("💡 Revenez aux étapes précédentes pour corriger")
        else:
            st.success("✅ Configuration valide - Prêt pour l'entraînement!")
            
            # Estimation des ressources
            st.markdown("---")
            st.subheader("💻 Estimation des Ressources")
            
            try:
                sys_metrics = check_system_resources()
                resource_check = self._estimate_training_resources(STATE.data.df, len(selected_models), sys_metrics)
                
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    has_resources = resource_check.get("has_enough_resources", False)
                    status_color = "#28a745" if has_resources else "#dc3545"
                    status_icon = "✅" if has_resources else "❌"
                    st.markdown(
                        f"""
                        <div class='metric-card' style='background: {status_color};'>
                            <h3>{status_icon}</h3>
                            <h4>Ressources Système</h4>
                            <h2>{'Suffisantes' if has_resources else 'Insuffisantes'}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    can_launch = can_launch and has_resources
                
                with col_res2:
                    n_models = len(selected_models)
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                            <h3>🤖</h3>
                            <h4>Modèles à Entraîner</h4>
                            <h2>{n_models}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with col_res3:
                    base_time = n_models * 30
                    if optimize:
                        base_time *= 3
                    minutes = max(1, int(base_time / 60))
                    
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                            <h3>⏱️</h3>
                            <h4>Temps Estimé</h4>
                            <h2>{minutes} min</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                if resource_check.get("warnings"):
                    with st.expander("⚠️ Avertissements Système", expanded=True):
                        for warning in resource_check["warnings"]:
                            st.warning(warning)
                
                if not has_resources:
                    with st.expander("🔍 Détails des Problèmes", expanded=True):
                        for issue in resource_check.get("issues", []):
                            st.error(issue)
                    
                    st.error("❌ Ressources système insuffisantes")
                    st.info("💡 Fermez d'autres applications ou réduisez le nombre de modèles")
            
            except Exception as e:
                st.warning(f"⚠️ Impossible de vérifier les ressources: {e}")
        
        # Bouton de lancement
        st.markdown("---")
        
        col_launch1, col_launch2, col_launch3 = st.columns([1, 2, 1])
        
        with col_launch2:
            if st.button(
                "🚀 Lancer l'Entraînement",
                type="primary",
                use_container_width=True,
                disabled=not can_launch,
                key="launch_training"
            ):
                self.launch_training()
        
        # Navigation
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour", use_container_width=True):
                STATE.current_step = 4
                st.rerun()
        
        with col_nav2:
            if st.button("🔄 Recommencer", use_container_width=True, type="secondary"):
                STATE.current_step = 0
                STATE.workflow_complete = False
                st.success("✅ Workflow réinitialisé!")
                time.sleep(0.5)
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    def launch_training(self):
        """Lance l'entraînement avec l'orchestrateur via StateManager - VERSION CORRIGÉE"""
        try:
            # Préparation des paramètres avec StateManager
            df = STATE.data.df
            
            # VALIDATION CRITIQUE des données
            if df is None or df.empty:
                st.error("❌ Aucun dataset chargé pour l'entraînement")
                return
            
            target_column = STATE.target_column
            feature_list = STATE.feature_list
            task_type = STATE.task_type
            test_size = STATE.test_size / 100.0
            selected_models = STATE.selected_models
            optimize = STATE.optimize_hyperparams
            preprocessing_config = STATE.preprocessing_config
            use_smote = preprocessing_config.get('use_smote', False)
            
            # VALIDATION configuration minimale
            if not selected_models:
                st.error("❌ Aucun modèle sélectionné")
                return
            
            if task_type in ['classification', 'regression'] and not target_column:
                st.error("❌ Variable cible non définie")
                return
            
            if not feature_list:
                st.error("❌ Aucune feature sélectionnée")
                return

            # Création du contexte d'entraînement
            context = MLTrainingContext(
                df=df,
                target_column=target_column,
                feature_list=feature_list,
                task_type=task_type,
                test_size=test_size,
                model_names=selected_models,
                optimize_hyperparams=optimize,
                preprocessing_config=preprocessing_config,
                use_smote=use_smote,
                metadata={
                    'session_id': str(hash(str(datetime.now()))),
                    'user_agent': 'streamlit_app'
                }
            )
            
            # Interface de progression
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()
            
            # Lancement
            status_text.text("🚀 Initialisation de l'entraînement...")
            progress_bar.progress(10)
            
            # EXÉCUTION AVEC GESTION D'ERREUR ROBUSTE
            try:
                result = ml_training_orchestrator.train(context)
                
                # VALIDATION du résultat
                if result is None:
                    raise ValueError("L'orchestrateur a retourné None")
                    
            except Exception as training_error:
                st.error(f"❌ Erreur pendant l'entraînement: {str(training_error)}")
                logger.error(f"Training execution error: {training_error}", exc_info=True)
                
                # Création d'un résultat d'erreur
                result = MLTrainingResult(
                    success=False,
                    results=[],
                    summary={},
                    errors=[str(training_error)],
                    warnings=[],
                    training_time=0,
                    metadata={'error': True}
                )
            
            # Mise à jour interface
            status_text.text("✅ Entraînement terminé!")
            progress_bar.progress(100)
            
            # SAUVEGARDE SÉCURISÉE des résultats
            if result is not None:
                STATE.training_results = result
                STATE.workflow_complete = True
                
                # 🆕 SYNCHRONISATION FINALE depuis collecteur
                collector = get_mlflow_collector()
                final_runs = collector.get_runs()
                
                if final_runs:
                    logger.info(f"📊 {len(final_runs)} runs MLflow disponibles dans collecteur")
                    
                    # Synchronisation explicite vers toutes les sources
                    try:
                        from monitoring.state_managers import sync_mlflow_runs_all_sources
                        sync_counters = sync_mlflow_runs_all_sources(final_runs)
                    except Exception as sync_error:
                        logger.warning(f"Sync MLflow non disponible: {sync_error}")
                        sync_counters = {'total_synchronized': len(final_runs) if final_runs else 0}
                    
                    if sync_counters['total_synchronized'] > 0:
                        st.success(
                            f"✅ {sync_counters['total_synchronized']} runs MLflow synchronisés "
                            f"vers tous les états"
                        )
                    else:
                        st.info("ℹ️ Tous les runs MLflow déjà synchronisés")
                
                # Affichage des résultats
                self.display_training_results(result, results_container)
            else:
                st.error("❌ Aucun résultat disponible après l'entraînement")
                
        except Exception as e:
            st.error(f"❌ Erreur inattendue lors du lancement de l'entraînement: {str(e)}")
            logger.error(f"Unexpected error in launch_training: {e}", exc_info=True)

    def display_training_results(self, result: MLTrainingResult, container):
        """Affiche les résultats de l'entraînement - VERSION COMPLÈTE CORRIGÉE"""
        
        # VALIDATION ROBUSTE du résultat
        if result is None:
            container.error("❌ Aucun résultat d'entraînement disponible")
            logger.error("display_training_results: result est None")
            return
        
        # VALIDATION des attributs essentiels
        if not hasattr(result, 'successful_models') or not hasattr(result, 'results'):
            container.error("❌ Format de résultat invalide")
            logger.error(f"Résultat invalide: {type(result)} - {dir(result)}")
            return
        
        with container.container():
            st.markdown("## 📊 Résultats de l'Entraînement")
            
            # UTILISATION SÉCURISÉE des attributs
            n_successful = len(result.successful_models)
            n_total = len(result.results)
            
            # ========================================================================
            # 🆕 DÉTECTION DU MEILLEUR MODÈLE (VERSION CORRIGÉE)
            # ========================================================================
            best_model_data = None
            best_score = 0.0
            task_type = STATE.task_type
            
            # Détermination de la métrique clé
            metric_key = {
                'classification': 'accuracy',
                'regression': 'r2',
                'clustering': 'silhouette_score'
            }.get(task_type, 'accuracy')
            
            logger.debug(f"🔍 Recherche meilleur modèle avec métrique: {metric_key}")
            
            # PARCOURS DES MODÈLES RÉUSSIS (ce sont des DICTS)
            if result.successful_models:
                for model_result in result.successful_models:
                    # Vérification que c'est bien un dict
                    if not isinstance(model_result, dict):
                        logger.warning(f"⚠️ Résultat modèle non-dict: {type(model_result)}")
                        continue
                    
                    # Extraction des métriques
                    metrics = model_result.get('metrics', {})
                    if not isinstance(metrics, dict):
                        logger.warning(f"⚠️ Métriques non-dict pour {model_result.get('model_name', 'Unknown')}")
                        continue
                    
                    # Récupération du score
                    score = metrics.get(metric_key, 0)
                    if score is None:
                        score = 0.0
                    
                    logger.debug(f"   {model_result.get('model_name', 'Unknown')}: {metric_key}={score}")
                    
                    # Comparaison
                    if score > best_score:
                        best_score = float(score)
                        best_model_data = model_result
                
                if best_model_data:
                    logger.info(
                        f"✅ Meilleur modèle: {best_model_data.get('model_name', 'Unknown')} "
                        f"({metric_key}={best_score:.3f})"
                    )
                else:
                    logger.warning("⚠️ Aucun meilleur modèle trouvé")
            else:
                logger.warning("⚠️ Aucun modèle réussi")
            
            # ========================================================================
            # MÉTRIQUES PRINCIPALES (VERSION CORRIGÉE)
            # ========================================================================
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            
            with col_res1:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <h3>🤖</h3>
                        <h4>Modèles Réussis</h4>
                        <h2>{n_successful}/{n_total}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_res2:
                training_time = getattr(result, 'training_time', 0)
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <h3>⏱️</h3>
                        <h4>Temps Total</h4>
                        <h2>{training_time:.1f}s</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_res3:
                # ✅ UTILISATION DE best_model_data (DICT)
                if best_model_data:
                    st.markdown(
                        f"""
                        <div class='metric-card' style='background: linear-gradient(135deg, #28a745 0%, #20c997 100%);'>
                            <h3>🏆</h3>
                            <h4>Meilleur Score</h4>
                            <h2>{best_score:.3f}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                            <h3>🏆</h3>
                            <h4>Meilleur Score</h4>
                            <h2>N/A</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            with col_res4:
                # UTILISATION DE best_model_data (DICT)
                if best_model_data:
                    model_name = best_model_data.get('model_name', 'Inconnu')
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                            <h3>👑</h3>
                            <h4>Meilleur Modèle</h4>
                            <h2 style="font-size: 1.5rem;">{model_name}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                            <h3>👑</h3>
                            <h4>Meilleur Modèle</h4>
                            <h2>Inconnu</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # ========================================================================
            # 🎨 TABLEAU RÉCAPITULATIF STYLÉ (VERSION CORRIGÉE)
            # ========================================================================
            st.markdown("---")
            st.markdown("### 📋 Détail des Performances")
            
            # CSS pour le tableau stylé
            st.markdown("""
            <style>
                .styled-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 1.5rem 0;
                    font-size: 0.95rem;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    border-radius: 10px;
                    overflow: hidden;
                }
                
                .styled-table thead tr {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-align: left;
                    font-weight: 600;
                }
                
                .styled-table thead th {
                    padding: 1rem;
                    text-transform: uppercase;
                    font-size: 0.85rem;
                    letter-spacing: 0.5px;
                }
                
                .styled-table tbody tr {
                    border-bottom: 1px solid #e9ecef;
                    transition: all 0.3s ease;
                }
                
                .styled-table tbody tr:nth-child(even) {
                    background-color: rgba(102, 126, 234, 0.05);
                }
                
                .styled-table tbody tr:hover {
                    background-color: rgba(102, 126, 234, 0.15);
                    transform: scale(1.01);
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }
                
                .styled-table tbody td {
                    padding: 1rem;
                }
                
                .styled-table tbody tr.best-model {
                    background: linear-gradient(90deg, rgba(40, 167, 69, 0.15) 0%, transparent 100%);
                    border-left: 4px solid #28a745;
                }
                
                .styled-table tbody tr.failed-model {
                    background: linear-gradient(90deg, rgba(220, 53, 69, 0.1) 0%, transparent 100%);
                    border-left: 4px solid #dc3545;
                }
                
                .metric-badge {
                    display: inline-block;
                    padding: 0.3rem 0.8rem;
                    border-radius: 20px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    white-space: nowrap;
                }
                
                .badge-excellent {
                    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
                    color: white;
                }
                
                .badge-good {
                    background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
                    color: white;
                }
                
                .badge-fair {
                    background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
                    color: #333;
                }
                
                .badge-poor {
                    background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
                    color: white;
                }
                
                .status-success {
                    color: #28a745;
                    font-weight: 600;
                }
                
                .status-failed {
                    color: #dc3545;
                    font-weight: 600;
                }
            </style>
            """, unsafe_allow_html=True)
            
            # Construction du HTML du tableau
            table_html = '<table class="styled-table"><thead><tr>'
            
            # En-têtes dynamiques selon task_type
            headers = ['Modèle', 'Statut', 'Temps (s)']
            
            if task_type == 'classification':
                headers.extend(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
            elif task_type == 'regression':
                headers.extend(['R²', 'MAE', 'RMSE'])
            elif task_type == 'clustering':
                headers.extend(['Silhouette', 'N_Clusters', 'DB Index'])
            
            for header in headers:
                table_html += f'<th>{header}</th>'
            
            table_html += '</tr></thead><tbody>'
            
            # LIGNES DES MODÈLES RÉUSSIS (VERSION CORRIGÉE)
            for model_result in result.successful_models:
                if not isinstance(model_result, dict):
                    continue
                
                metrics = model_result.get('metrics', {})
                if not isinstance(metrics, dict):
                    metrics = {}
                
                model_name = model_result.get('model_name', 'Inconnu')
                training_time_model = model_result.get('training_time', 0)
                
                # Classe CSS spéciale pour le meilleur modèle
                row_class = 'best-model' if best_model_data and model_name == best_model_data.get('model_name') else ''
                
                table_html += f'<tr class="{row_class}">'
                table_html += f'<td><strong>{model_name}</strong></td>'
                table_html += f'<td><span class="status-success">✅ Succès</span></td>'
                table_html += f'<td>{training_time_model:.1f}s</td>'
                
                # Métriques selon task_type - VERSION CORRIGÉE
                if task_type == 'classification':
                    accuracy = metrics.get('accuracy', 0)
                    precision = metrics.get('precision', 0)
                    recall = metrics.get('recall', 0)
                    f1 = metrics.get('f1_score', 0)
                    roc_auc = metrics.get('roc_auc')
                    
                    # ✅ CORRECTION : Formatage conditionnel séparé
                    roc_auc_str = f"{roc_auc:.3f}" if roc_auc is not None else "N/A"
                    
                    # Badge coloré pour accuracy
                    if accuracy >= 0.9:
                        acc_badge = f'<span class="metric-badge badge-excellent">{accuracy:.3f}</span>'
                    elif accuracy >= 0.75:
                        acc_badge = f'<span class="metric-badge badge-good">{accuracy:.3f}</span>'
                    elif accuracy >= 0.6:
                        acc_badge = f'<span class="metric-badge badge-fair">{accuracy:.3f}</span>'
                    else:
                        acc_badge = f'<span class="metric-badge badge-poor">{accuracy:.3f}</span>'
                    
                    table_html += f'<td>{acc_badge}</td>'
                    table_html += f'<td>{precision:.3f}</td>'
                    table_html += f'<td>{recall:.3f}</td>'
                    table_html += f'<td>{f1:.3f}</td>'
                    table_html += f'<td>{roc_auc_str}</td>'  # ✅ CORRIGÉ
                    
                elif task_type == 'regression':
                    r2 = metrics.get('r2', 0)
                    mae = metrics.get('mae', 0)
                    rmse = metrics.get('rmse', 0)
                    
                    # Badge coloré pour R²
                    if r2 >= 0.8:
                        r2_badge = f'<span class="metric-badge badge-excellent">{r2:.3f}</span>'
                    elif r2 >= 0.6:
                        r2_badge = f'<span class="metric-badge badge-good">{r2:.3f}</span>'
                    elif r2 >= 0.4:
                        r2_badge = f'<span class="metric-badge badge-fair">{r2:.3f}</span>'
                    else:
                        r2_badge = f'<span class="metric-badge badge-poor">{r2:.3f}</span>'
                    
                    table_html += f'<td>{r2_badge}</td>'
                    table_html += f'<td>{mae:.3f}</td>'
                    table_html += f'<td>{rmse:.3f}</td>'
                    
                elif task_type == 'clustering':
                    silhouette = metrics.get('silhouette_score', 0)
                    n_clusters = metrics.get('n_clusters', 'N/A')
                    db_index = metrics.get('davies_bouldin_score')
                    
                    # CORRECTION : Formatage conditionnel séparé
                    db_index_str = f"{db_index:.3f}" if db_index is not None else "N/A"
                    
                    # Badge coloré pour Silhouette
                    if silhouette >= 0.7:
                        sil_badge = f'<span class="metric-badge badge-excellent">{silhouette:.3f}</span>'
                    elif silhouette >= 0.5:
                        sil_badge = f'<span class="metric-badge badge-good">{silhouette:.3f}</span>'
                    elif silhouette >= 0.3:
                        sil_badge = f'<span class="metric-badge badge-fair">{silhouette:.3f}</span>'
                    else:
                        sil_badge = f'<span class="metric-badge badge-poor">{silhouette:.3f}</span>'
                    
                    table_html += f'<td>{sil_badge}</td>'
                    table_html += f'<td>{n_clusters}</td>'
                    table_html += f'<td>{db_index_str}</td>'  # ✅ CORRIGÉ
                
                table_html += '</tr>'
            
            # LIGNES DES MODÈLES ÉCHOUÉS
            for model_result in result.failed_models:
                if not isinstance(model_result, dict):
                    continue
                
                model_name = model_result.get('model_name', 'Inconnu')
                training_time_model = model_result.get('training_time', 0)
                error_msg = model_result.get('error', 'Erreur inconnue')
                
                table_html += f'<tr class="failed-model">'
                table_html += f'<td><strong>{model_name}</strong></td>'
                table_html += f'<td><span class="status-failed">❌ Échec</span></td>'
                table_html += f'<td>{training_time_model:.1f}s</td>'
                table_html += f'<td colspan="{len(headers)-3}" style="color: #dc3545; font-style: italic;">{error_msg[:100]}</td>'
                table_html += '</tr>'
            
            table_html += '</tbody></table>'
            
            # Affichage du tableau
            if result.successful_models or result.failed_models:
                st.markdown(table_html, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Aucun résultat à afficher")
            
            # ========================================================================
            # RECOMMANDATIONS
            # ========================================================================
            summary = getattr(result, 'summary', {})
            if summary and summary.get('recommendations'):
                st.markdown("---")
                st.markdown("### 💡 Recommandations")
                
                for recommendation in summary['recommendations']:
                    st.info(recommendation)
            
            # ========================================================================
            # INFORMATIONS COMPLÉMENTAIRES (SMOTE, IMBALANCE, etc.)
            # ========================================================================
            if task_type == 'classification' and best_model_data:
                st.markdown("---")
                st.markdown("### ℹ️ Informations Complémentaires")
                
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    smote_applied = best_model_data.get('smote_applied', False)
                    st.markdown(
                        f"""
                        <div style='padding: 1rem; border-radius: 10px; 
                                    background: {"linear-gradient(135deg, #28a74520 0%, #20c99720 100%)" if smote_applied else "rgba(0,0,0,0.05)"};
                                    border-left: 4px solid {"#28a745" if smote_applied else "#6c757d"};'>
                            <strong>SMOTE</strong><br>
                            <span style='font-size: 1.5rem;'>{"✅ Activé" if smote_applied else "❌ Désactivé"}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with col_info2:
                    imbalance_ratio = best_model_data.get('imbalance_ratio')
                    if imbalance_ratio:
                        color = "#dc3545" if imbalance_ratio > 10 else "#ffc107" if imbalance_ratio > 3 else "#28a745"
                        st.markdown(
                            f"""
                            <div style='padding: 1rem; border-radius: 10px; 
                                        background: {color}20;
                                        border-left: 4px solid {color};'>
                                <strong>Ratio Déséquilibre</strong><br>
                                <span style='font-size: 1.5rem; color: {color};'>{imbalance_ratio:.1f}:1</span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                with col_info3:
                    n_features = best_model_data.get('n_features', 0)
                    st.markdown(
                        f"""
                        <div style='padding: 1rem; border-radius: 10px; 
                                    background: rgba(102, 126, 234, 0.1);
                                    border-left: 4px solid #667eea;'>
                            <strong>Features Utilisées</strong><br>
                            <span style='font-size: 1.5rem; color: #667eea;'>{n_features}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # ========================================================================
            # BOUTON ANALYSE DÉTAILLÉE
            # ========================================================================
            st.markdown("---")
            if st.button("📈 Voir l'Analyse Détaillée des Résultats", type="primary", use_container_width=True):
                # SAUVEGARDE CORRECTE dans STATE
                STATE.ml_results = result.results
                STATE.switch(AppPage.ML_EVALUATION)

    def render_complete_step(self):
        """Étape finale après entraînement complet"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🎉 Entraînement Terminé!")
        
        if STATE.training_results:
            self.display_training_results(STATE.training_results, st)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Voir l'Analyse", type="primary", use_container_width=True):
                STATE.switch(AppPage.ML_EVALUATION)
        
        with col2:
            if st.button("🔄 Nouvel Entraînement", use_container_width=True):
                self.initialize_session_state()
                st.rerun()
        
        with col3:
            if st.button("🏠 Retour à l'Accueil", use_container_width=True):
                STATE.switch(AppPage.HOME)
        
        st.markdown('</div>', unsafe_allow_html=True)

    def run(self):
        """Exécute le workflow complet avec StateManager"""
        self.render_header()
        self.render_workflow_progress()
        
        if STATE.workflow_complete and STATE.training_results:
            self.render_complete_step()
        else:
            # Routing des étapes avec StateManager
            steps = [
                self.render_dataset_analysis_step,
                self.render_target_selection_step,
                self.render_imbalance_analysis_step,
                self.render_preprocessing_step,
                self.render_model_selection_step,
                self.render_training_launch_step
            ]
            
            current_step = STATE.current_step
            if 0 <= current_step < len(steps):
                steps[current_step]()
            else:
                STATE.current_step = 0
                st.rerun()

def debug_feature_state():
    """Fonction de debug pour l'état des features avec StateManager"""
    if st.sidebar.checkbox("🐛 Mode Debug Features", value=False):
        st.sidebar.markdown("### 🐛 État des Features")
        
        feature_sources = {
            "STATE.feature_list": STATE.feature_list,
            "STATE.data.feature_list": STATE.data.feature_list,
            "Dataset columns": STATE.data.df.columns.tolist() if STATE.loaded and STATE.data.df is not None else "N/A",
            "Target column": STATE.target_column,
            "Task type": STATE.task_type
        }
        
        for source, value in feature_sources.items():
            if isinstance(value, list):
                st.sidebar.write(f"**{source}**: {len(value)} items")
                if value and len(value) > 0:
                    st.sidebar.write(f"First 5: {value[:5]}")
            else:
                st.sidebar.write(f"**{source}**: {value}")
        
        # Bouton de réinitialisation
        if st.sidebar.button("🔄 Reset Feature State"):
            STATE.feature_list = []
            STATE.data.feature_list = []
            st.sidebar.success("Feature state reset!")
            time.sleep(1)
            st.rerun()

# Point d'entrée de l'application   
def main():
    """Fonction principale de l'application avec StateManager"""
    try:
        debug_feature_state()
        
        workflow = MLTrainingWorkflowPro()
        workflow.run()
    except Exception as e:
        st.error(f"❌ Erreur critique dans l'application: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)
        
        if st.button("🔄 Redémarrer l'Application"):
            st.rerun()

if __name__ == "__main__":
    main()