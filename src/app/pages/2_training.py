"""
🚀 ML FACTORY PRO - Interface Moderne pour ML Classique (Tabular Data)
Design unifié avec Computer Vision Training - Production Ready
Version: 2.0.0
"""
import os
import logging
from src.config.constants import LOGGING_CONSTANTS

# Configuration des logs
log_dir = LOGGING_CONSTANTS.get("LOG_DIR", "logs")
log_file = LOGGING_CONSTANTS.get("LOG_FILE", "training.log")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(os.path.join(log_dir, log_file), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
from src.shared.logging import StructuredLogger
from helpers.data_validators import DataValidator
from utils.system_utils import get_system_metrics as check_system_resources
from monitoring.state_managers import init, AppPage
STATE = init()

logger = StructuredLogger(__name__)

# Configuration Streamlit
st.set_page_config(
    page_title="ML Factory Pro | ML Classique",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS moderne identique à Computer Vision
st.markdown("""
<style>
    /* Reset et Base */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Principal */
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        animation: fadeInDown 0.6s ease-out;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .workflow-step-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        margin-bottom: 1.5rem;
        animation: fadeIn 0.4s ease-out;
    }
    
    .model-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .model-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .model-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        border-color: #667eea;
    }
    
    .model-card:hover::before {
        opacity: 1;
    }
    
    .model-card.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .model-card.selected::after {
        content: '✓';
        position: absolute;
        top: 10px;
        right: 10px;
        background: #667eea;
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 14px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
    }
    
    .metric-card h4 {
        margin: 0.5rem 0;
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .metric-card h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
        transition: transform 0.2s ease;
    }
    
    .status-badge:hover {
        transform: scale(1.1);
    }
    
    .badge-success { 
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white; 
        box-shadow: 0 2px 4px rgba(40, 167, 69, 0.3);
    }
    
    .badge-warning { 
        background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%);
        color: #333; 
        box-shadow: 0 2px 4px rgba(255, 193, 7, 0.3);
    }
    
    .badge-danger { 
        background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
        color: white; 
        box-shadow: 0 2px 4px rgba(220, 53, 69, 0.3);
    }
    
    .badge-info { 
        background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
        color: white; 
        box-shadow: 0 2px 4px rgba(23, 162, 184, 0.3);
    }
    
    /* Progress Steps */
    .progress-step {
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .progress-step.active {
        background: #f8f9ff;
        border: 2px solid #667eea;
        transform: scale(1.05);
    }
    
    .progress-step.completed {
        background: #e8f5e9;
        border: 2px solid #28a745;
    }
    
    .progress-step.pending {
        background: white;
        border: 2px solid #e0e0e0;
        opacity: 0.7;
    }
    
    /* Task Selection Cards */
    .task-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid transparent;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }
    
    .task-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .task-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
    }
    
    .task-card:hover::before {
        opacity: 1;
    }
    
    .task-card.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    .task-card .icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5568d3 0%, #6a3d91 100%);
    }
    
    /* Dataframes */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Selectbox & Inputs */
    .stSelectbox, .stSlider, .stCheckbox {
        margin-bottom: 1rem;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: 8px;
        border-left-width: 4px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)


class MLTrainingWorkflowPro:
    """
    Workflow moderne pour ML Classique (Tabular Data).
    Architecture identique à Computer Vision pour cohérence UX.
    """
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
    
    def render_header(self):
        """En-tête professionnel avec navigation et métriques"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown('<div class="main-header">🎯 ML Factory Pro</div>', unsafe_allow_html=True)
            st.markdown('<div class="sub-header">Workflow Intelligent pour ML Classique (Tabular Data)</div>', unsafe_allow_html=True)
        
        with col2:
            progress = ((STATE.current_step + 1) / 6) * 100
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Progression</div>
                    <div style="background: #e0e0e0; border-radius: 10px; height: 8px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                                    width: {progress}%; height: 100%; transition: width 0.3s ease;"></div>
                    </div>
                    <div style="font-size: 0.8rem; color: #667eea; margin-top: 0.5rem; font-weight: 600;">
                        Étape {STATE.current_step + 1}/6
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            sys_metrics = check_system_resources()
            memory_color = "#28a745" if sys_metrics["memory_percent"] < 70 else "#ffc107" if sys_metrics["memory_percent"] < 85 else "#dc3545"
            
            st.markdown(
                f"""
                <div style="text-align: center;">
                    <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">Système</div>
                    <div style="display: flex; justify-content: center; align-items: center; gap: 0.5rem;">
                        <div style="width: 40px; height: 40px; border-radius: 50%; 
                                    background: {memory_color}; display: flex; align-items: center; 
                                    justify-content: center; color: white; font-weight: bold;">
                            {sys_metrics["memory_percent"]:.0f}
                        </div>
                        <div style="text-align: left;">
                            <div style="font-size: 0.8rem; color: #666;">RAM</div>
                            <div style="font-size: 0.7rem; color: #999;">
                                {100 - sys_metrics["memory_percent"]:.0f}% libre
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
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
    
    # ============================================================================
    # ÉTAPE 1: ANALYSE DU DATASET
    # ============================================================================
    
    def render_dataset_analysis_step(self):
        """Étape 1: Analyse du dataset chargé - VERSION CORRIGÉE"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("📊 Étape 1: Analyse du Dataset")
        
        # Vérification dataset
        if not STATE.loaded or STATE.data.df is None:
            st.error("❌ Aucun dataset chargé")
            st.info("💡 Veuillez charger un dataset depuis le dashboard principal.")
            if st.button("📊 Aller au Dashboard", type="primary", use_container_width=True):
                st.switch_page("pages/1_dashboard.py")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        df = STATE.data.df
        
        # ========================================================================
        # 🆕 CORRECTION CRITIQUE : NETTOYAGE AUTOMATIQUE DES COLONNES INUTILES
        # ========================================================================
        
        # Sauvegarde du dataset original pour référence
        original_shape = df.shape
        original_columns = df.columns.tolist()
        
        # Détection automatique des colonnes problématiques
        with st.spinner("🔍 Analyse automatique des colonnes en cours..."):
            # Colonnes constantes (sans variance)
            constant_cols = []
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if df[col].std() == 0:
                    constant_cols.append(col)
            
            # Colonnes identifiantes (100% valeurs uniques)
            identifier_cols = [col for col in df.columns if df[col].nunique() == len(df)]
            
            # Colonnes avec trop de valeurs manquantes (>80%)
            high_missing_cols = [col for col in df.columns if df[col].isnull().mean() > 0.8]
            
            # Colonnes à supprimer
            cols_to_remove = list(set(constant_cols + identifier_cols + high_missing_cols))
            cols_to_keep = [col for col in df.columns if col not in cols_to_remove]
        
        # Application automatique du nettoyage
        if cols_to_remove:
            st.markdown("### 🧹 Nettoyage Automatique des Colonnes")
            
            df_cleaned = df[cols_to_keep].copy()
            n_removed = len(cols_to_remove)
            
            st.success(f"✅ **{n_removed} colonne(s)** supprimée(s) automatiquement")
            
            # Affichage détaillé des colonnes supprimées
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
            
            # Mise à jour du DataFrame
            df = df_cleaned
            STATE.data.df = df_cleaned
            
            st.info(f"📊 **Dimensions mises à jour :** {original_shape} → {df.shape}")
        
        else:
            st.success("✅ Aucune colonne problématique détectée - Dataset conservé intact")
        
        # ========================================================================
        # SUITE DU CODE EXISTANT (validation, métriques, etc.)
        # ========================================================================
        
        # Validation avec DataValidator
        validation_result = DataValidator.validate_dataframe_for_ml(df)
        
        if not validation_result['is_valid']:
            st.error("❌ Dataset non compatible avec l'analyse ML")
            with st.expander("🔍 Détails des problèmes", expanded=True):
                for issue in validation_result['issues']:
                    st.error(f"• {issue}")
            
            if st.button("🔄 Recharger un nouveau dataset", type="primary"):
                st.switch_page("pages/1_dashboard.py")
            
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Métriques principales avec design moderne
        st.subheader("📈 Statistiques Principales")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(
                f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);'>
                    <h3>📏</h3>
                    <h4>Lignes</h4>
                    <h2>{len(df):,}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);'>
                    <h3>📋</h3>
                    <h4>Colonnes</h4>
                    <h2>{len(df.columns)}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            st.markdown(
                f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);'>
                    <h3>💾</h3>
                    <h4>Mémoire</h4>
                    <h2>{memory_mb:.1f} MB</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col4:
            missing_pct = df.isnull().mean().mean() * 100
            missing_color = "#28a745" if missing_pct < 5 else "#ffc107" if missing_pct < 20 else "#dc3545"
            st.markdown(
                f"""
                <div class='metric-card' style='background: {missing_color};'>
                    <h3>🕳️</h3>
                    <h4>Manquant</h4>
                    <h2>{missing_pct:.1f}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col5:
            numeric_cols = len(df.select_dtypes(include='number').columns)
            st.markdown(
                f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);'>
                    <h3>🔢</h3>
                    <h4>Numériques</h4>
                    <h2>{numeric_cols}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Détection automatique des types de colonnes SUR LE DATASET NETTOYÉ
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
        
        # ========================================================================
        # 🆕 INITIALISATION ROBUSTE DE FEATURE_LIST
        # ========================================================================
        
        # Détermination automatique des features (toutes les colonnes restantes)
        feature_list = df.columns.tolist()
        
        st.markdown("---")
        st.subheader("🎯 Features Disponibles")
        
        st.info(f"**{len(feature_list)} features** détectées automatiquement")
        
        with st.expander("📋 Liste complète des features", expanded=False):
            # Affichage organisé des features
            cols_display = st.columns(2)
            for idx, feature in enumerate(feature_list):
                with cols_display[idx % 2]:
                    col_type = "🔢" if feature in column_types.get('numeric', []) else "📝"
                    st.markdown(f"{col_type} `{feature}`")
        
        # Navigation
        st.markdown("---")
        if st.button("💾 Valider et Continuer ➡️", type="primary", use_container_width=True):
            # 🆕 SAUVEGARDE ROBUSTE DANS TOUS LES ENDROITS NÉCESSAIRES
            STATE.dataset_loaded = True
            STATE.dataset_info = {
                'n_rows': len(df),
                'n_cols': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
                'missing_pct': df.isnull().mean().mean() * 100,
                'column_types': column_types,
                'features_initial': feature_list,  # 🆕 Sauvegarde explicite
                'cleaning_applied': len(cols_to_remove) > 0 if 'cols_to_remove' in locals() else False,
                'cols_removed': cols_to_remove if 'cols_to_remove' in locals() else []
            }
            
            # 🆕 INITIALISATION EXPLICITE DE FEATURE_LIST
            STATE.feature_list = feature_list
            
            # Debug optionnel
            if st.session_state.get('debug_mode', False):
                st.json({
                    "feature_list_saved": STATE.feature_list,
                    "length": len(STATE.feature_list),
                    "first_10": STATE.feature_list[:10]
                })
            
            STATE.current_step = 1
            st.success("✅ Dataset validé et nettoyé avec succès!")
            time.sleep(0.5)
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ============================================================================
    # ÉTAPE 2: SÉLECTION DE LA CIBLE
    # ============================================================================
    
    def render_target_selection_step(self):
        """Étape 2: Sélection de la variable cible et du type de tâche"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🎯 Étape 2: Sélection de la Cible")
        
        df = STATE.data.df
        
        # Sélection du type de tâche avec cards modernes
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
            
            # Filtrage des colonnes selon le type
            if task_type == 'classification':
                available_targets = [
                    col for col in df.columns
                    if df[col].nunique() <= 50 or not pd.api.types.is_numeric_dtype(df[col])
                ]
                help_text = "📊 Colonne avec classes à prédire (≤50 valeurs uniques recommandé)"
            else:
                available_targets = [
                    col for col in df.columns
                    if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10
                ]
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
                            
                            # Graphique de distribution
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
                            # Histogramme pour la distribution
                            fig = go.Figure(data=[
                                go.Histogram(
                                    x=df[target_column],
                                    nbinsx=50,
                                    marker=dict(
                                        color='#667eea',
                                        line=dict(color='white', width=1)
                                    ),
                                    hovertemplate='Valeur: %{x}<br>Fréquence: %{y}<extra></extra>'
                                )
                            ])
                            
                            fig.update_layout(
                                title="Distribution de la Variable Cible",
                                xaxis_title=target_column,
                                yaxis_title="Fréquence",
                                template="plotly_white",
                                height=400
                            )
                            
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
                            
                            # Analyse du déséquilibre
                            imbalance_info = detect_imbalance(df, target_column)
                            
                            if imbalance_info.get('is_imbalanced', False):
                                ratio = imbalance_info.get('imbalance_ratio', 0)
                                
                                if ratio > 10:
                                    color = "#dc3545"
                                    level = "Critique"
                                    icon = "🚨"
                                elif ratio > 5:
                                    color = "#fd7e14"
                                    level = "Élevé"
                                    icon = "⚠️"
                                else:
                                    color = "#ffc107"
                                    level = "Modéré"
                                    icon = "ℹ️"
                                
                                st.markdown(
                                    f"""
                                    <div class='metric-card' style='background: {color};'>
                                        <h3>{icon}</h3>
                                        <h4>Déséquilibre</h4>
                                        <h2>{level}</h2>
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
                            
                            # Valeurs manquantes
                            missing_count = df[target_column].isnull().sum()
                            if missing_count > 0:
                                missing_pct = (missing_count / len(df)) * 100
                                st.warning(f"⚠️ {missing_count} valeurs manquantes ({missing_pct:.1f}%)")
                
                    st.subheader("📊 Variables Explicatives (X)")
                    if target_column:
                        # Pour la classification et la régression, on propose de sélectionner les features
                        available_features = [col for col in df.columns if col != target_column]
                        
                        # Option de sélection automatique ou manuelle
                        auto_features = st.checkbox("Sélection automatique des features", value=True, key="auto_features")
                        
                        if auto_features:
                            # Détection automatique des types de colonnes
                            column_types = auto_detect_column_types(df)
                            numeric_features = column_types.get('numeric', [])
                            categorical_features = [col for col in column_types.get('categorical', []) 
                                                if df[col].nunique() <= 50]
                            recommended_features = numeric_features + categorical_features
                            recommended_features = [col for col in recommended_features if col in available_features]
                            recommended_features = recommended_features[:50]
                            
                            # Sauvegarde dans l'état
                            STATE.feature_list = recommended_features
                            
                            st.success(f"✅ {len(recommended_features)} features sélectionnées automatiquement")
                            
                            # Affichage des features sélectionnées
                            with st.expander("📋 Voir les features sélectionnées", expanded=False):
                                for feat in recommended_features[:20]:
                                    st.markdown(f"- `{feat}`")
                                if len(recommended_features) > 20:
                                    st.caption(f"... et {len(recommended_features) - 20} autres")
                        else:
                            # Sélection manuelle
                            selected_features = st.multiselect(
                                "Sélectionnez les variables explicatives",
                                options=available_features,
                                default=STATE.feature_list if hasattr(STATE, 'feature_list') and STATE.feature_list else [],
                                key="manual_features"
                            )
                            
                            # Sauvegarde dans l'état
                            STATE.feature_list = selected_features
                        
                        # Feedback sur la sélection des features
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
    
    # ============================================================================
    # ÉTAPE 3: GESTION DU DÉSÉQUILIBRE
    # ============================================================================
    
    def render_imbalance_analysis_step(self):
        """Étape 3: Analyse et correction du déséquilibre (classification uniquement)"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("⚖️ Étape 3: Gestion du Déséquilibre")
        
        df = STATE.data.df
        task_type = STATE.task_type
        target_column = STATE.target_column
        
        # Si pas classification, skip automatiquement
        if task_type != 'classification':
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #17a2b815 0%, #138496 15 100%); 
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
        
        # Statistiques des classes
        class_counts = df[target_column].value_counts()
        total_samples = len(df)
        
        # Métriques principales
        st.subheader("📊 Analyse du Déséquilibre")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ratio = imbalance_info.get('imbalance_ratio', 1.0)
            
            if ratio > 10:
                color = "#dc3545"
                icon = "🚨"
                level = "Critique"
                gradient = "linear-gradient(135deg, #dc3545 0%, #c82333 100%)"
            elif ratio > 5:
                color = "#fd7e14"
                icon = "⚠️"
                level = "Élevé"
                gradient = "linear-gradient(135deg, #fd7e14 0%, #e8590c 100%)"
            elif ratio > 2:
                color = "#ffc107"
                icon = "ℹ️"
                level = "Modéré"
                gradient = "linear-gradient(135deg, #ffc107 0%, #ff9800 100%)"
            else:
                color = "#28a745"
                icon = "✅"
                level = "Faible"
                gradient = "linear-gradient(135deg, #28a745 0%, #20c997 100%)"
            
            st.markdown(
                f"""
                <div class='metric-card' style='background: {gradient}; animation: pulse 2s infinite;'>
                    <h3 style='font-size: 2.5rem;'>{icon}</h3>
                    <h4>Niveau de Déséquilibre</h4>
                    <h2>{level}</h2>
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
                        Images d'entraînement
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Graphique de distribution interactif
        st.markdown("---")
        st.subheader("📈 Distribution des Classes")
        
        fig = go.Figure()
        
        # Couleurs dynamiques selon la taille
        colors = ['#2ecc71' if i == class_counts.idxmax() else '#e74c3c' if i == class_counts.idxmin() else '#3498db'
                  for i in class_counts.index]
        
        fig.add_trace(go.Bar(
            x=class_counts.index.astype(str),
            y=class_counts.values,
            text=[f"{count:,}<br>({count/total_samples*100:.1f}%)" for count in class_counts.values],
            textposition='auto',
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>Classe: %{x}</b><br>Échantillons: %{y}<br>Pourcentage: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "Distribution des Classes dans le Dataset",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Classe",
            yaxis_title="Nombre d'échantillons",
            template="plotly_white",
            height=450,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🎯 Stratégies de Correction")
    
        col_strat1, col_strat2 = st.columns(2)
        
        with col_strat1:
            st.markdown("#### ⚖️ Poids de Classe Automatiques")
            st.markdown(
                """
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745;'>
                    <p><strong>Principe:</strong> Ajuste la fonction de perte pour donner plus d'importance aux classes minoritaires.</p>
                    <p><strong>Avantage:</strong> Ne modifie pas les données, rapide</p>
                    <p><strong>Inconvénient:</strong> Peut sur-ajuster les classes rares</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # ✅ CORRECTION : Initialisation sécurisée
            if not hasattr(STATE, 'imbalance_config') or STATE.imbalance_config is None:
                STATE.imbalance_config = {}
            
            ratio = imbalance_info.get('imbalance_ratio', 1.0)
            
            use_class_weights = st.checkbox(
                "✅ Activer les poids de classe",
                value=ratio > 2,
                help="Recommandé pour ratios > 2:1"
            )
            
            if use_class_weights:
                st.success("✅ Les poids seront calculés automatiquement lors de l'entraînement")
                
                # ✅ CORRECTION : Sauvegarde uniforme
                STATE.imbalance_config['use_class_weights'] = True
                
                # Aperçu des poids
                with st.expander("👁️ Aperçu des poids (estimation)", expanded=False):
                    weights = len(df) / (len(class_counts) * class_counts)
                    for cls, weight in weights.items():
                        st.markdown(f"- **Classe {cls}**: `{weight:.3f}` (×{weight:.1f} importance)")
            else:
                STATE.imbalance_config['use_class_weights'] = False
        
        with col_strat2:
            st.markdown("#### 🎭 SMOTE (Suréchantillonnage Synthétique)")
            st.markdown(
                """
                <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #17a2b8;'>
                    <p><strong>Principe:</strong> Génère des exemples synthétiques pour les classes minoritaires.</p>
                    <p><strong>Avantage:</strong> Augmente les données, améliore la généralisation</p>
                    <p><strong>Inconvénient:</strong> Peut introduire du bruit</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            use_smote = st.checkbox(
                "✅ Activer SMOTE",
                value=ratio > 3,
                help="Recommandé pour ratios > 3:1"
            )
            
            smote_k_neighbors = 5  # Valeur par défaut
            
            if use_smote:
                min_class_count = class_counts.min()
                
                st.markdown("**⚙️ Configuration SMOTE**")
                
                smote_k_neighbors = st.slider(
                    "Nombre de voisins (k)",
                    min_value=1,
                    max_value=min(20, max(1, min_class_count - 1)),
                    value=min(5, max(1, min_class_count - 1)),
                    help="Nombre de plus proches voisins utilisés"
                )
                
                st.info(f"💡 SMOTE générera ~{int((class_counts.max() - class_counts.min()) * 0.8):,} exemples synthétiques")
                
                # ✅ CORRECTION : Sauvegarde uniforme via dict
                if not hasattr(STATE, 'preprocessing_config') or STATE.preprocessing_config is None:
                    STATE.preprocessing_config = {}
                
                STATE.preprocessing_config['use_smote'] = True
                STATE.preprocessing_config['smote_k_neighbors'] = smote_k_neighbors
                STATE.imbalance_config['use_smote'] = True
                
                if min_class_count < smote_k_neighbors:
                    st.warning(f"⚠️ Classe minoritaire trop petite ({min_class_count} samples) pour k={smote_k_neighbors}")
            else:
                if not hasattr(STATE, 'preprocessing_config') or STATE.preprocessing_config is None:
                    STATE.preprocessing_config = {}
                
                STATE.preprocessing_config['use_smote'] = False
                STATE.imbalance_config['use_smote'] = False
        
        # Recommandations
        if ratio > 5:
            st.markdown("---")
            st.markdown("### 💡 Recommandations")
            st.warning(
                f"""
                ⚠️ **Déséquilibre élevé détecté (ratio: {ratio:.1f}:1)**
                
                Nous vous recommandons **fortement** d'activer au moins une stratégie:
                - ✅ **Poids de classe**: Rapide et efficace
                - ✅ **SMOTE**: Utile si peu de données minoritaires
                - 🎯 **Les deux combinés**: Pour déséquilibre critique (>10:1)
                """
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
                # ✅ CORRECTION : Sauvegarde complète avec mise à jour
                STATE.imbalance_config.update({
                    'use_class_weights': use_class_weights,
                    'use_smote': use_smote,
                    'smote_k_neighbors': smote_k_neighbors if use_smote else 5,
                    'imbalance_ratio': float(ratio)
                })
                
                STATE.current_step = 3
                st.success("✅ Configuration du déséquilibre sauvegardée!")
                time.sleep(0.3)
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================================
    # ÉTAPE 4: PRÉTRAITEMENT
    # ============================================================================
    
    def render_preprocessing_step(self):
        """Étape 4: Configuration du prétraitement des données"""
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
        
        # Initialisation sécurisée
        if not hasattr(STATE, 'preprocessing_config') or STATE.preprocessing_config is None:
            STATE.preprocessing_config = {}
        
        # Analyse des features sélectionnées
        df = STATE.data.df
        feature_list = getattr(STATE, 'feature_list', [])
        
        if feature_list:
            # Détection automatique des colonnes numériques dans les features sélectionnées
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
            
            # Imputation numérique en fonction des features sélectionnées
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
                STATE.preprocessing_config['numeric_imputation'] = 'mean'  # Valeur par défaut
            
            # Imputation catégorielle en fonction des features sélectionnées
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
                STATE.preprocessing_config['categorical_imputation'] = 'most_frequent'  # Valeur par défaut
            
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
            
            # ✅ CORRECTION CRITIQUE : Normalisation uniquement pour variables numériques
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
                    
                    # Avertissement pour variables catégorielles
                    if n_categorical > 0:
                        st.success(
                            f"✅ **{n_categorical}** variables catégorielles seront encodées "
                            f"(One-Hot ou Label Encoding) mais **PAS** normalisées"
                        )
            else:
                # Désactivation automatique si pas de variables numériques
                STATE.preprocessing_config['scale_features'] = False
                st.warning(
                    "⚠️ **Normalisation désactivée**\n\n"
                    "Aucune variable numérique dans votre sélection. "
                    "La normalisation ne s'applique qu'aux variables numériques."
                )
                
                if n_categorical > 0:
                    st.info(
                        f"ℹ️ Les **{n_categorical}** variables catégorielles seront automatiquement "
                        f"encodées (One-Hot ou Label Encoding) lors de l'entraînement."
                    )
            
            st.markdown("---")
            
            st.subheader("🔍 Réduction Dimensionnelle")
            
            # PCA uniquement si variables numériques
            if n_numeric > 10:  # Seuil recommandé
                STATE.preprocessing_config['pca_preprocessing'] = st.checkbox(
                    "🎯 Activer PCA",
                    value=STATE.preprocessing_config.get('pca_preprocessing', False),
                    help=f"Réduction dimensionnelle pour {n_numeric} variables numériques (>10)"
                )
                
                if STATE.preprocessing_config.get('pca_preprocessing', False):
                    st.success(f"✅ PCA sera appliqué sur les **{n_numeric}** variables numériques")
                    
                    # Seuil de variance expliquée
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
        
        # Analyse colonnes à nettoyer (code existant inchangé)
        if STATE.preprocessing_config.get('remove_constant_cols') or STATE.preprocessing_config.get('remove_identifier_cols'):
            st.markdown("---")
            st.subheader("🔍 Analyse des Colonnes à Nettoyer")
            
            with st.spinner("🔍 Analyse des colonnes en cours..."):
                numeric_cols = df.select_dtypes(include='number').columns
                constant_cols = [col for col in numeric_cols if df[col].std() == 0] if len(numeric_cols) > 0 else []
                identifier_cols = [col for col in df.columns if df[col].nunique() == len(df)]
                
                if constant_cols or identifier_cols:
                    col_clean1, col_clean2 = st.columns(2)
                    
                    with col_clean1:
                        if constant_cols:
                            st.warning(f"⚠️ {len(constant_cols)} colonne(s) constante(s)")
                            with st.expander("📋 Voir colonnes", expanded=False):
                                for col in constant_cols:
                                    st.markdown(f"- `{col}`")
                        else:
                            st.success("✅ Aucune colonne constante")
                    
                    with col_clean2:
                        if identifier_cols:
                            st.warning(f"⚠️ {len(identifier_cols)} colonne(s) identifiante(s)")
                            with st.expander("📋 Voir colonnes", expanded=False):
                                for col in identifier_cols:
                                    st.markdown(f"- `{col}`")
                        else:
                            st.success("✅ Aucune colonne identifiante")
                else:
                    st.success("✅ Aucune colonne problématique")
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour", use_container_width=True):
                STATE.current_step = 2
                st.rerun()
        
        with col_nav2:
            if st.button("💾 Sauvegarder et Continuer ➡️", type="primary", use_container_width=True):
                # ✅ CORRECTION : Sauvegarde du nombre de features par type
                STATE.preprocessing_config['n_numeric_features'] = n_numeric
                STATE.preprocessing_config['n_categorical_features'] = n_categorical
                STATE.preprocessing_config['numeric_features'] = numeric_features
                STATE.preprocessing_config['categorical_features'] = categorical_features
                
                STATE.current_step = 4
                st.success("✅ Configuration du prétraitement sauvegardée!")
                time.sleep(0.3)
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    
    # ============================================================================
    # ÉTAPE 5: SÉLECTION DES MODÈLES
    # ============================================================================
    
    def render_model_selection_step(self):
        """Étape 5: Sélection des algorithmes de machine learning"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🤖 Étape 5: Sélection des Modèles")
        
        task_type = STATE.task_type
        
        # Récupération des modèles disponibles pour la tâche
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
            
            # Création des colonnes pour les cartes de modèles
            cols = st.columns(3)
            col_idx = 0
            
            for model_name, config in models:
                with cols[col_idx]:
                    is_selected = model_name in selected_models
                    
                    # Couleur selon la complexité
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
                # Calcul de la complexité moyenne
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
                # Estimation du temps d'entraînement
                base_time = len(selected_models) * 30  # 30 secondes par modèle de base
                if STATE.optimize_hyperparams:
                    base_time *= 3  # ×3 pour l'optimisation
                if STATE.preprocessing_config.get('pca_preprocessing', False):
                    base_time *= 1.2  # +20% pour PCA
                
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
            
            # Liste des modèles sélectionnés
            with st.expander("📋 Détail des modèles sélectionnés", expanded=True):
                cols = st.columns(3)
                for idx, model_name in enumerate(selected_models):
                    with cols[idx % 3]:
                        config = available_models[model_name]
                        st.markdown(f"**{model_name}**")
                        st.caption(f"• {config.get('description', '')}")
                        st.caption(f"• Complexité: {config.get('complexity', 'medium')}")
                        st.caption(f"• Vitesse: {config.get('training_speed', 'medium')}")
            
            # Recommandations
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
                    value=STATE.get('test_size', 20),
                    help="Pourcentage de données réservées pour l'évaluation finale"
                )
                STATE.test_size = test_size
                st.info(f"📊 Split: {100-test_size}% train, {test_size}% test")
            else:
                st.info("🔍 Clustering: 100% des données utilisées (pas de split)")
        
        with col_adv2:
            optimize = st.checkbox(
                "🔍 Optimisation des hyperparamètres",
                value=STATE.get('optimize_hyperparams', False),
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
    
    # ============================================================================
    # ÉTAPE 6: LANCEMENT DE L'ENTRAÎNEMENT
    # ============================================================================
    
    def render_training_launch_step(self):
        
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
            
            # Accès safe à feature_list
            feature_list = STATE.feature_list if hasattr(STATE, 'feature_list') else []
            
            # Affichage debug 
            if st.checkbox("🔍 Debug feature_list", value=False, key="debug_features"):
                st.json({
                    "feature_list_from_property": STATE.feature_list if hasattr(STATE, 'feature_list') else "N/A",
                    "feature_list_from_data": STATE.data.feature_list if hasattr(STATE.data, 'feature_list') else "N/A",
                    "length": len(feature_list),
                    "first_5": feature_list[:5] if feature_list else []
                })
            
            st.markdown(f"- **Features**: `{len(feature_list)}` variables")
            
            if STATE.task_type != 'clustering':
                test_size = STATE.test_size if hasattr(STATE, 'test_size') else 20
                st.markdown(f"- **Split test**: `{test_size}%`")
        
        with col_recap2:
            st.markdown("#### 🤖 Configuration des Modèles")
            
            # Accès safe à selected_models
            selected_models = STATE.selected_models if hasattr(STATE, 'selected_models') else []
            st.markdown(f"- **Modèles sélectionnés**: `{len(selected_models)}`")
            
            optimize = STATE.optimize_hyperparams if hasattr(STATE, 'optimize_hyperparams') else False
            st.markdown(f"- **Optimisation HP**: `{'✅ Oui' if optimize else '❌ Non'}`")
            
            if STATE.task_type == 'classification':
                
                # Accès safe aux configs de déséquilibre
                preprocessing_config = STATE.preprocessing_config if hasattr(STATE, 'preprocessing_config') else {}
                imbalance_config = STATE.imbalance_config if hasattr(STATE, 'imbalance_config') else {}
                
                use_smote = preprocessing_config.get('use_smote', False) if preprocessing_config else False
                use_weights = imbalance_config.get('use_class_weights', False) if imbalance_config else False
                
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
            
            # Vérification des features
            if not feature_list or len(feature_list) == 0:
                validation_issues.append("❌ Aucune feature sélectionnée")
                
                # 🔍 Diagnostic approfondi
                if st.checkbox("🔍 Diagnostic approfondi", value=True, key="deep_debug"):
                    st.warning("🔍 **Diagnostic des features manquantes**")
                    
                    # Vérifier toutes les sources possibles
                    possible_sources = {
                        "STATE.feature_list": STATE.feature_list if hasattr(STATE, 'feature_list') else None,
                        "STATE.data.feature_list": STATE.data.feature_list if hasattr(STATE.data, 'feature_list') else None,
                        "session_state.feature_list": st.session_state.get('feature_list', None),
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
        
        if not STATE.selected_models or len(STATE.selected_models) == 0:
            validation_issues.append("❌ Aucun modèle sélectionné")
        
        # Détermination de can_launch AVANT son utilisation
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
            
            from utils.system_utils import check_system_resources
            
            try:
                resource_check = check_system_resources(
                    STATE.data.df, 
                    len(STATE.selected_models)
                )
                
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
                    
                    # Mise à jour de can_launch avec ressources
                    can_launch = can_launch and has_resources
                
                with col_res2:
                    n_models = len(STATE.selected_models)
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
                    # Estimation du temps
                    base_time = n_models * 30
                    if STATE.optimize_hyperparams:
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
                
                # Avertissements ressources
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
                # On laisse can_launch tel quel si erreur de vérification
        
        # Bouton de lancement
        st.markdown("---")
        
        col_launch1, col_launch2, col_launch3 = st.columns([1, 2, 1])
        
        with col_launch2:
            # can_launch est maintenant défini AVANT son utilisation
            if st.button(
                "🚀 Lancer l'Entraînement",
                type="primary",
                use_container_width=True,
                disabled=not can_launch,  # SAFE : Variable définie plus haut
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
        """Lance l'entraînement avec l'orchestrateur"""
        try:
            # Préparation des paramètres
            df = STATE.data.df
            target_column = STATE.target_column
            feature_list = getattr(STATE, 'feature_list', [])
            task_type = STATE.task_type
            test_size = getattr(STATE, 'test_size', 20) / 100.0
            selected_models = getattr(STATE, 'selected_models', [])
            optimize = getattr(STATE, 'optimize_hyperparams', False)
            preprocessing_config = getattr(STATE, 'preprocessing_config', {})
            use_smote = preprocessing_config.get('use_smote', False) if preprocessing_config else False
            
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
            
            # Exécution avec l'orchestrateur
            result = ml_training_orchestrator.train(context)
            
            # Mise à jour interface
            status_text.text("✅ Entraînement terminé!")
            progress_bar.progress(100)
            
            # Sauvegarde des résultats
            STATE.training_results = result
            STATE.workflow_complete = True
            
            # Affichage des résultats
            self.display_training_results(result, results_container)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de l'entraînement: {str(e)}")
            logger.error(f"Training error: {e}", exc_info=True)
            STATE.workflow_complete = False
    
    def display_training_results(self, result: MLTrainingResult, container):
        """Affiche les résultats de l'entraînement"""
        with container.container():
            st.markdown("## 📊 Résultats de l'Entraînement")
            
            # Métriques principales
            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            
            with col_res1:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <h3>🤖</h3>
                        <h4>Modèles Réussis</h4>
                        <h2>{len(result.successful_models)}/{len(result.results)}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_res2:
                st.markdown(
                    f"""
                    <div class='metric-card'>
                        <h3>⏱️</h3>
                        <h4>Temps Total</h4>
                        <h2>{result.training_time:.1f}s</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col_res3:
                best_model = result.best_model
                if best_model:
                    task_type = STATE.task_type
                    metric_key = 'accuracy' if task_type == 'classification' else 'r2' if task_type == 'regression' else 'silhouette_score'
                    best_score = best_model['metrics'].get(metric_key, 0)
                    
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
                if best_model:
                    st.markdown(
                        f"""
                        <div class='metric-card'>
                            <h3>👑</h3>
                            <h4>Meilleur Modèle</h4>
                            <h2>{best_model['model_name']}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Détails des résultats
            st.markdown("---")
            st.subheader("📋 Détail des Performances")
            
            # Tableau des résultats
            results_data = []
            for model_result in result.successful_models:
                metrics = model_result.get('metrics', {})
                results_data.append({
                    'Modèle': model_result['model_name'],
                    'Statut': '✅ Succès',
                    'Temps (s)': f"{model_result.get('training_time', 0):.1f}",
                    **{k: f"{v:.3f}" if isinstance(v, (int, float)) else str(v) 
                       for k, v in metrics.items()}
                })
            
            for model_result in result.failed_models:
                results_data.append({
                    'Modèle': model_result['model_name'],
                    'Statut': '❌ Échec',
                    'Temps (s)': f"{model_result.get('training_time', 0):.1f}",
                    'Erreur': model_result.get('metrics', {}).get('error', 'Erreur inconnue')
                })
            
            if results_data:
                st.dataframe(pd.DataFrame(results_data), use_container_width=True)
            
            # Recommandations
            if result.summary.get('recommendations'):
                st.markdown("---")
                st.subheader("💡 Recommandations")
                
                for recommendation in result.summary['recommendations']:
                    st.info(recommendation)
            
            # Bouton pour voir l'analyse détaillée
            st.markdown("---")
            if st.button("📈 Voir l'Analyse Détaillée des Résultats", type="primary", use_container_width=True):
                STATE.ml_results = result.results
                st.switch_page("pages/3_evaluation.py")
    
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
                st.switch_page("pages/3_evaluation.py")
        
        with col2:
            if st.button("🔄 Nouvel Entraînement", use_container_width=True):
                self.initialize_session_state()
                st.rerun()
        
        with col3:
            if st.button("🏠 Retour à l'Accueil", use_container_width=True):
                st.switch_page("main.py")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def run(self):
        """Exécute le workflow complet"""
        self.render_header()
        self.render_workflow_progress()
        
        if STATE.workflow_complete and STATE.training_results:
            self.render_complete_step()
        else:
            # Routing des étapes
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
    """Fonction de debug pour l'état des features"""
    if st.sidebar.checkbox("🐛 Mode Debug Features", value=False):
        st.sidebar.markdown("### 🐛 État des Features")
        
        feature_sources = {
            "STATE.feature_list": getattr(STATE, 'feature_list', "N/A"),
            "STATE.data.feature_list": getattr(STATE.data, 'feature_list', "N/A") if hasattr(STATE, 'data') else "N/A",
            "Dataset columns": STATE.data.df.columns.tolist() if hasattr(STATE, 'data') and STATE.data.df is not None else "N/A",
            "Target column": getattr(STATE, 'target_column', "N/A"),
            "Task type": getattr(STATE, 'task_type', "N/A")
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
            if hasattr(STATE, 'feature_list'):
                STATE.feature_list = []
            if hasattr(STATE.data, 'feature_list'):
                STATE.data.feature_list = []
            st.sidebar.success("Feature state reset!")
            time.sleep(1)
            st.rerun()

# Point d'entrée de l'application   
def main():
    """Fonction principale de l'application"""
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