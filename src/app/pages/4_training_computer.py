"""
🚀 ML FACTORY PRO - Interface Professionnelle d'Entraînement Computer Vision
Version fusionnée complète : Logique métier existante + Interface moderne
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import sys
import os
from src.shared.logging import get_logger
import torch # type: ignore
from typing import Any, Dict, Optional, Tuple, Union
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Configuration des chemins d'import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = get_logger(__name__)

# Imports de votre logique métier existante
try:
    from src.models.computer_vision_training import (
        ComputerVisionTrainer,
        AnomalyAwareTrainer,
        ModelConfig,
        TrainingConfig,
        ModelType,
        OptimizerType,
        SchedulerType,
        DataAugmenter,
        MLflowIntegration
    )
    from src.data.computer_vision_preprocessing import DataPreprocessor, DataValidator
    from src.shared.logging import StructuredLogger
    from utils.callbacks import LoggingCallback, StreamlitCallback
    from utils.device_manager import DeviceManager
    LOGIC_METIER_AVAILABLE = True
except ImportError as e:
    LOGIC_METIER_AVAILABLE = False
    st.warning(f"⚠️ Logique métier non disponible: {e}")

# Configuration Streamlit
st.set_page_config(
    page_title="ML Factory Pro | Entraînement Computer Vision",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS moderne professionnel
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .workflow-step-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        margin-bottom: 1.5rem;
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
    }
    .model-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .model-card.selected {
        border-color: #667eea;
        background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .badge-success { background: #28a745; color: white; }
    .badge-warning { background: #ffc107; color: black; }
    .badge-danger { background: #dc3545; color: white; }
    .badge-info { background: #17a2b8; color: white; }
    .param-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    .config-error {
        background: linear-gradient(135deg, #ff7979 0%, #eb4d4b 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class MLTrainingWorkflowPro:
    """
    🚀 Workflow professionnel d'entraînement Computer Vision
    Combine la logique métier existante avec une interface moderne
    """
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialise l'état de session complet avec toutes les configurations"""
        defaults = {
            # Navigation et état
            'current_step': 0,
            'workflow_complete': False,
            'current_experiment': None,
            'experiments': [],
            
            # Données
            'dataset_loaded': False,
            'dataset_info': {},
            'split_config': {},
            
            # Configurations
            'selected_model_type': None,
            'model_config': {},
            'training_config': None,
            'preprocessing_config': {
                "strategy": "standardize",
                "augmentation_enabled": False,
                "augmentation_factor": 2,
                "methods": ['flip', 'rotate']
            },
            'imbalance_config': {
                "use_class_weights": False,
                "use_targeted_augmentation": False,
                "augmentation_factor": 2,
                "strategy": "standardize"
            },
            
            # Résultats
            'training_history': [],
            'class_weights': {},
            'trained_model': None,
            'training_results': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
        # Initialisation spécifique à la logique métier
        if 'training_config' not in st.session_state or st.session_state.training_config is None:
            if LOGIC_METIER_AVAILABLE:
                st.session_state.training_config = TrainingConfig(
                    epochs=50,
                    batch_size=32,
                    learning_rate=1e-4,
                    early_stopping_patience=10,
                    reduce_lr_patience=5,
                    optimizer=OptimizerType.ADAMW,
                    scheduler=SchedulerType.REDUCE_ON_PLATEAU,
                    use_class_weights=False
                )
    
    def render_header(self):
        """En-tête professionnel avec navigation et métriques"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown('<div class="main-header">🚀 ML Factory Pro</div>', unsafe_allow_html=True)
            st.markdown("**Workflow Intelligent d'Entraînement Computer Vision**")
        
        with col2:
            st.metric("Étape Actuelle", f"{st.session_state.current_step + 1}/6")
            
        with col3:
            if st.session_state.current_experiment:
                st.info(f"🔄 Expérience: {st.session_state.current_experiment}")
            else:
                st.warning("⚡ Configuration en cours")
                
            # Statut GPU/CPU
            device = "CUDA 🚀" if torch.cuda.is_available() else "CPU ⚡"
            st.caption(f"Device: {device}")
    
    def render_workflow_progress(self):
        """Barre de progression intelligente avec étapes détaillées"""
        steps = [
            {"name": "📊 Données", "icon": "📊", "description": "Split et Analyse"},
            {"name": "⚖️ Déséquilibre", "icon": "⚖️", "description": "Analyse et Correction"}, 
            {"name": "🎨 Prétraitement", "icon": "🎨", "description": "Normalisation et Augmentation"},
            {"name": "🤖 Modèle", "icon": "🤖", "description": "Architecture et Paramètres"},
            {"name": "⚙️ Entraînement", "icon": "⚙️", "description": "Configuration Hyperparamètres"},
            {"name": "🚀 Lancement", "icon": "🚀", "description": "Démarrage et Monitoring"}
        ]
        
        current_step = st.session_state.current_step
        
        st.markdown("### 📋 Progression du Workflow")
        
        # Affichage des étapes en grille responsive
        cols = st.columns(len(steps))
        for idx, (col, step) in enumerate(zip(cols, steps)):
            with col:
                if idx < current_step:
                    status_icon = "✅"
                    status_color = "#28a745"
                    status_text = "Terminé"
                elif idx == current_step:
                    status_icon = "🔵" 
                    status_color = "#667eea"
                    status_text = "En cours"
                else:
                    status_icon = "⚪"
                    status_color = "#6c757d"
                    status_text = "À venir"
                
                st.markdown(
                    f"""
                    <div style="text-align: center; padding: 1rem; border-radius: 10px; 
                                background: {'#f8f9ff' if idx == current_step else 'white'}; 
                                border: 2px solid {status_color};">
                        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{step['icon']}</div>
                        <div style="font-weight: bold; color: {status_color}; margin-bottom: 0.25rem;">
                            {step['name']}
                        </div>
                        <div style="font-size: 0.8rem; color: #666;">{step['description']}</div>
                        <div style="font-size: 0.7rem; color: {status_color}; margin-top: 0.5rem;">
                            {status_icon} {status_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
    
    # ============================================================================
    # ÉTAPE 1: ANALYSE DES DONNÉES
    # ============================================================================
    
    def render_data_analysis_step(self):
        """Étape 1: Analyse et préparation des données avec validation"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("📊 Étape 1: Analyse des Données")
        
        # Vérification des données chargées
        if 'X' not in st.session_state or 'y' not in st.session_state:
            st.error("❌ Aucun dataset d'images chargé")
            st.info("Veuillez charger un dataset depuis le dashboard principal.")
            if st.button("📊 Aller au Dashboard", type="primary"):
                st.switch_page("pages/1_dashboard.py")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        X = st.session_state["X"]
        y = st.session_state["y"]
        
        # Métriques principales des données
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📷 Images Total", f"{len(X):,}")
        
        with col2:
            unique_classes = len(np.unique(y))
            st.metric("🎯 Classes", unique_classes)
        
        with col3:
            if len(X.shape) > 2:
                img_shape = f"{X.shape[1]}×{X.shape[2]}"
                if len(X.shape) > 3:
                    img_shape += f"×{X.shape[3]}"
            else:
                img_shape = "N/A"
            st.metric("📐 Taille Images", img_shape)
        
        with col4:
            data_type = "RGB" if (len(X.shape) > 3 and X.shape[-1] == 3) else "Grayscale" if (len(X.shape) > 3 and X.shape[-1] == 1) else "Unknown"
            st.metric("🎨 Type", data_type)
        
        # Validation des données avec DataValidator (logique métier)
        if LOGIC_METIER_AVAILABLE:
            with st.expander("🔍 Validation Automatique des Données"):
                validation_result = DataValidator.validate_input_data(X, y, "dataset")
                if validation_result.success:
                    st.success("✅ Données validées avec succès")
                    st.json(validation_result.metadata)
                else:
                    st.error(f"❌ Problèmes détectés: {validation_result.error}")
        
        st.markdown("---")
        
        # Configuration du split des données
        st.subheader("🔧 Configuration du Split Train/Validation/Test")
        
        col_split1, col_split2 = st.columns(2)
        
        with col_split1:
            test_size = st.slider(
                "Taille du Test Set (%)",
                min_value=10,
                max_value=40, 
                value=20,
                step=5,
                help="Pourcentage d'images réservées pour le test final"
            )
        
        with col_split2:
            val_size = st.slider(
                "Taille du Validation Set (%)", 
                min_value=10,
                max_value=30,
                value=20,
                step=5,
                help="Pourcentage des données d'entraînement pour la validation pendant l'entraînement"
            )
        
        # Calcul des tailles réelles
        test_ratio = test_size / 100
        val_ratio = val_size / 100
        
        n_test = int(len(X) * test_ratio)
        n_train_val = len(X) - n_test
        n_val = int(n_train_val * val_ratio) 
        n_train = n_train_val - n_val
        
        # Visualisation de la répartition
        st.markdown("### 📈 Répartition des Données")
        
        fig = go.Figure(data=[
            go.Pie(
                labels=['Train', 'Validation', 'Test'],
                values=[n_train, n_val, n_test],
                hole=0.4,
                marker_colors=['#28a745', '#17a2b8', '#6c757d'],
                textinfo='percent+value',
                hovertemplate='<b>%{label}</b><br>Échantillons: %{value}<br>Pourcentage: %{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Distribution Train/Validation/Test",
            showlegend=True,
            height=300,
            annotations=[dict(text=f'Total: {len(X):,}', x=0.5, y=0.5, font_size=12, showarrow=False)]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Affichage des statistiques détaillées
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
        with col_stats1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<h3>🏋️</h3>"
                f"<h4>Training Set</h4>"
                f"<h2>{n_train:,}</h2>"
                f"<p>Échantillons</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col_stats2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<h3>📊</h3>"
                f"<h4>Validation Set</h4>"
                f"<h2>{n_val:,}</h2>"
                f"<p>Échantillons</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col_stats3:
            st.markdown(
                f"<div class='metric-card'>"
                f"<h3>🧪</h3>"
                f"<h4>Test Set</h4>"
                f"<h2>{n_test:,}</h2>"
                f"<p>Échantillons</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Bouton de validation et split
        st.markdown("---")
        if st.button("🔄 Effectuer le Split et Continuer", type="primary", use_container_width=True):
            with st.spinner("Séparation des données avec stratification..."):
                try:
                    # Split des données avec stratification
                    X_train_val, X_test, y_train_val, y_test = train_test_split(
                        X, y, test_size=test_ratio, stratify=y, random_state=42
                    )
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_val, y_train_val, 
                        test_size=val_ratio / (1 - test_ratio),
                        stratify=y_train_val, 
                        random_state=42
                    )
                    
                    # Sauvegarde dans session_state
                    st.session_state.update({
                        "X_train": X_train,
                        "X_val": X_val, 
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_val": y_val,
                        "y_test": y_test,
                        "split_config": {
                            "test_size": test_size,
                            "val_size": val_size,
                            "train_samples": n_train,
                            "val_samples": n_val, 
                            "test_samples": n_test
                        },
                        "dataset_loaded": True,
                        "dataset_info": {
                            "original_samples": len(X),
                            "train_samples": n_train,
                            "val_samples": n_val,
                            "test_samples": n_test,
                            "num_classes": unique_classes,
                            "input_shape": X.shape[1:] if len(X.shape) > 2 else X.shape
                        }
                    })
                    
                    st.session_state.current_step = 1
                    st.success("✅ Split effectué avec succès!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Erreur lors du split: {str(e)}")
                    self.logger.error(f"Split error: {e}", exc_info=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================================
    # ÉTAPE 2: ANALYSE DU DÉSÉQUILIBRE
    # ============================================================================
    
    def render_imbalance_analysis_step(self):
        """Étape 2: Analyse et correction du déséquilibre des classes"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("⚖️ Étape 2: Gestion du Déséquilibre")
        
        # Vérification des données d'entraînement
        if 'y_train' not in st.session_state:
            st.error("❌ Données d'entraînement non disponibles")
            if st.button("⬅️ Retour à l'étape 1", use_container_width=True):
                st.session_state.current_step = 0
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        y_train = st.session_state.y_train
        
        # Analyse du déséquilibre des classes
        label_counts = Counter(y_train)
        total_samples = len(y_train)
        percentages = [count / total_samples * 100 for count in label_counts.values()]
        
        # Calcul du ratio de déséquilibre
        max_count = max(label_counts.values())
        min_count = min(label_counts.values()) 
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        # Détermination du niveau de déséquilibre
        if imbalance_ratio > 10:
            imbalance_level = "critique"
            imbalance_color = "#dc3545"
            imbalance_icon = "🚨"
            recommendation = "Correction impérative nécessaire"
        elif imbalance_ratio > 5:
            imbalance_level = "élevé" 
            imbalance_color = "#fd7e14"
            imbalance_icon = "⚠️"
            recommendation = "Correction fortement recommandée"
        elif imbalance_ratio > 2:
            imbalance_level = "modéré"
            imbalance_color = "#ffc107"
            imbalance_icon = "ℹ️"
            recommendation = "Correction recommandée"
        else:
            imbalance_level = "faible"
            imbalance_color = "#28a745"
            imbalance_icon = "✅"
            recommendation = "Aucune correction nécessaire"
        
        # Métriques d'analyse
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"<div style='background: {imbalance_color}; color: white; padding: 1.5rem; border-radius: 10px; text-align: center;'>"
                f"<h3 style='margin: 0; font-size: 2rem;'>{imbalance_icon}</h3>"
                f"<h4 style='margin: 0.5rem 0;'>Niveau de Déséquilibre</h4>"
                f"<h2 style='margin: 0;'>{imbalance_level.title()}</h2>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<h3>⚖️</h3>"
                f"<h4>Ratio de Déséquilibre</h4>"
                f"<h2>{imbalance_ratio:.1f}:1</h2>"
                f"<p>{recommendation}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"<div class='metric-card'>"
                f"<h3>📊</h3>"
                f"<h4>Échantillons Total</h4>"
                f"<h2>{total_samples:,}</h2>"
                f"<p>Images d'entraînement</p>"
                f"</div>",
                unsafe_allow_html=True
            )
        
        # Graphique de distribution des classes
        st.markdown("### 📈 Distribution des Classes")
        
        # Labels intelligents selon le contexte
        if len(label_counts) == 2 and set(label_counts.keys()) == {0, 1}:
            labels = ['Normal', 'Anomalie']
            colors = ['#2ecc71', '#e74c3c']
        else:
            labels = [f"Classe {cls}" for cls in sorted(label_counts.keys())]
            colors = px.colors.qualitative.Set3[:len(labels)]
        
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=list(label_counts.values()),
                text=[f"{count}<br>({perc:.1f}%)" for count, perc in zip(label_counts.values(), percentages)],
                textposition='auto',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Percentage: %{text}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Distribution des Classes dans le Training Set",
            xaxis_title="Classes",
            yaxis_title="Nombre d'images", 
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stratégies de correction du déséquilibre
        st.markdown("### 🎯 Stratégies de Correction du Déséquilibre")
        
        col_corr1, col_corr2 = st.columns(2)
        
        with col_corr1:
            st.subheader("⚖️ Poids de Classe")
            use_class_weights = st.checkbox(
                "Activer les poids de classe automatiques",
                value=imbalance_ratio > 2,
                help="Ajuste automatiquement la loss function pour compenser le déséquilibre. Recommandé pour les ratios > 2:1"
            )
            
            if use_class_weights:
                classes = np.unique(y_train)
                weights = compute_class_weight('balanced', classes=classes, y=y_train)
                weight_dict = dict(zip(classes, weights))
                
                st.info("**Poids calculés automatiquement:**")
                for cls, weight in weight_dict.items():
                    cls_name = "Normal" if cls == 0 else "Anomalie" if len(label_counts) == 2 else f"Classe {cls}"
                    st.write(f"- **{cls_name}**: `{weight:.3f}` (inverse de la fréquence)")
        
        with col_corr2:
            st.subheader("🎭 Augmentation Ciblée")
            use_targeted_augmentation = st.checkbox(
                "Augmenter les classes minoritaires",
                value=imbalance_ratio > 3,
                help="Applique plus d'augmentation de données aux classes sous-représentées"
            )
            
            if use_targeted_augmentation:
                augmentation_factor = st.slider(
                    "Facteur d'augmentation maximal",
                    min_value=2,
                    max_value=10,
                    value=min(5, int(imbalance_ratio)),
                    help="Facteur de multiplication maximal pour les classes minoritaires"
                )
                
                st.info(f"Les classes minoritaires seront augmentées jusqu'à x{augmentation_factor}")
        
        # Validation avec DataValidator (logique métier)
        if LOGIC_METIER_AVAILABLE:
            with st.expander("🔍 Analyse Détaillée du Déséquilibre"):
                imbalance_result = DataValidator.check_class_imbalance(y_train)
                if imbalance_result:
                    st.json(imbalance_result)
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.current_step = 0
                st.rerun()
        
        with col_nav2:
            if st.button("💾 Sauvegarder et Continuer ➡️", type="primary", use_container_width=True):
                # Sauvegarde de la configuration
                st.session_state.imbalance_config = {
                    "use_class_weights": use_class_weights,
                    "use_targeted_augmentation": use_targeted_augmentation,
                    "augmentation_factor": augmentation_factor if use_targeted_augmentation else 1,
                    "imbalance_ratio": imbalance_ratio,
                    "imbalance_level": imbalance_level,
                    "label_counts": label_counts
                }
                
                if use_class_weights:
                    st.session_state.class_weights = weight_dict
                    st.session_state.training_config.use_class_weights = True
                
                st.success("✅ Configuration du déséquilibre sauvegardée")
                st.session_state.current_step = 2
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================================
    # ÉTAPE 3: PRÉTRAITEMENT DES DONNÉES
    # ============================================================================
    
    def render_preprocessing_step(self):
        """Étape 3: Configuration du prétraitement et de l'augmentation"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🎨 Étape 3: Prétraitement des Images")
        
        st.markdown("""
        **Configuration du pipeline de prétraitement**
        Optimisez vos images pour l'entraînement avec des techniques standards du domaine.
        """)
        
        # Configuration de la normalisation
        st.subheader("🔧 Normalisation des Images")
        
        col_norm1, col_norm2 = st.columns(2)
        
        with col_norm1:
            normalization_method = st.selectbox(
                "Méthode de normalisation",
                options=["standardize", "normalize", "none"],
                index=0,
                help=(
                    "**Standardize**: (x - mean) / std (recommandé) - Centre à 0 et échelle unitaire\n\n"
                    "**Normalize**: min-max scaling [0, 1] - Garde les valeurs entre 0 et 1\n\n"
                    "**None**: Aucune normalisation (déconseillé)"
                )
            )
        
        with col_norm2:
            if 'X' in st.session_state:
                current_shape = st.session_state.X.shape
                if len(current_shape) > 2:
                    current_size = f"{current_shape[1]}×{current_shape[2]}"
                else:
                    current_size = "N/A"
                
                st.info(f"**Taille actuelle:** {current_size}")
            
            resize_option = st.selectbox(
                "Redimensionnement",
                options=["Conserver original", "128×128", "224×224", "256×256", "384×384"],
                index=0,
                help="Taille cible pour les images. 224×224 est standard pour la plupart des modèles."
            )
        
        st.markdown("---")
        
        # Configuration de l'augmentation de données
        st.subheader("🎭 Augmentation de Données")
        
        augmentation_enabled = st.checkbox(
            "Activer l'augmentation de données",
            value=st.session_state.preprocessing_config.get("augmentation_enabled", False),
            help="Génère des variations des images d'entraînement pour améliorer la généralisation"
        )
        
        if augmentation_enabled:
            st.markdown("#### 🔧 Méthodes d'Augmentation")
            
            col_aug1, col_aug2 = st.columns(2)
            
            with col_aug1:
                augmentation_factor = st.slider(
                    "Facteur de multiplication",
                    min_value=1,
                    max_value=5,
                    value=st.session_state.preprocessing_config.get("augmentation_factor", 2),
                    help="Nombre de variations générées par image originale"
                )
            
            with col_aug2:
                st.markdown("**Techniques sélectionnées:**")
                
                methods = []
                if st.checkbox("Flip horizontal", value=True):
                    methods.append('flip')
                if st.checkbox("Rotation (±15°)", value=True):
                    methods.append('rotate')
                if st.checkbox("Zoom aléatoire", value=False):
                    methods.append('zoom')
                if st.checkbox("Décalage de luminosité", value=False):
                    methods.append('brightness')
                if st.checkbox("Ajout de bruit gaussien", value=False):
                    methods.append('noise')
            
            # Affichage de l'impact de l'augmentation
            if 'X_train' in st.session_state:
                original_count = len(st.session_state.X_train)
                augmented_count = original_count * augmentation_factor
                
                st.info(f"""
                **Impact de l'augmentation:**
                - Images originales: {original_count:,}
                - Après augmentation: {augmented_count:,} (x{augmentation_factor})
                - Gain: +{augmented_count - original_count:,} images
                """)
        
        # Intégration avec DataAugmenter (logique métier)
        if LOGIC_METIER_AVAILABLE and augmentation_enabled:
            with st.expander("🔍 Configuration Avancée de l'Augmentation"):
                try:
                    augmenter = DataAugmenter(methods=methods)
                    st.success("✅ DataAugmenter configuré avec succès")
                    st.json({"methods": methods, "factor": augmentation_factor})
                except Exception as e:
                    st.error(f"❌ Erreur configuration DataAugmenter: {e}")
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        
        with col_nav2:
            if st.button("💾 Sauvegarder et Continuer ➡️", type="primary", use_container_width=True):
                st.session_state.preprocessing_config = {
                    "strategy": normalization_method,
                    "augmentation_enabled": augmentation_enabled,
                    "augmentation_factor": augmentation_factor if augmentation_enabled else 1,
                    "methods": methods if augmentation_enabled else [],
                    "resize": resize_option
                }
                
                st.success("✅ Configuration de prétraitement sauvegardée")
                st.session_state.current_step = 3
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================================
    # ÉTAPE 4: SÉLECTION ET CONFIGURATION DU MODÈLE
    # ============================================================================
    
    def get_model_categories(self):
        """
        Retourne les catégories de modèles organisées par use-case.
        
        Returns:
            dict: Dictionnaire structuré avec catégories et modèles
        """
        return {
            "🎯 Classification Supervisée": {
                "color": "#28a745",
                "description": "Modèles pour classification d'images avec labels",
                "models": [
                    {
                        "id": "simple_cnn",
                        "name": "CNN Simple", 
                        "description": "Réseau convolutionnel basique - Idéal pour débuter et prototyper rapidement",
                        "icon": "🖼️",
                        "complexity": "Débutant",
                        "training_time": "Rapide (~5-10 min)",
                        "use_cases": ["Prototypage rapide", "Images simples", "Apprentissage CNN"],
                        "requires_labels": True,
                        "min_samples": 500,
                        "gpu_recommended": False
                    },
                    {
                        "id": "custom_resnet", 
                        "name": "ResNet Personnalisé",
                        "description": "Architecture résiduelle profonde avec skip connections - Performances élevées",
                        "icon": "🏗️",
                        "complexity": "Intermédiaire", 
                        "training_time": "Moyen (~15-30 min)",
                        "use_cases": ["Images complexes", "Haute précision", "Datasets moyens/larges"],
                        "requires_labels": True,
                        "min_samples": 1000,
                        "gpu_recommended": True
                    },
                    {
                        "id": "transfer_learning",
                        "name": "Transfer Learning",
                        "description": "Modèles pré-entraînés ImageNet fine-tunés - State-of-the-art avec peu de données",
                        "icon": "🔄",
                        "complexity": "Avancé",
                        "training_time": "Variable (~10-20 min)",
                        "use_cases": ["Données limitées", "Production", "Précision maximale"],
                        "requires_labels": True,
                        "min_samples": 200,
                        "gpu_recommended": True
                    }
                ]
            },
            
            "🔍 Détection d'Anomalies": {
                "color": "#dc3545",
                "description": "Modèles pour détecter des anomalies sans/avec peu de labels",
                "models": [
                    {
                        "id": "conv_autoencoder",
                        "name": "AutoEncodeur Convolutif", 
                        "description": "Reconstruit les images normales - Détecte anomalies via erreur de reconstruction",
                        "icon": "🎭",
                        "complexity": "Intermédiaire",
                        "training_time": "Moyen (~10-20 min)",
                        "use_cases": ["Anomalies structurelles", "Contrôle qualité", "Images médicales"],
                        "requires_labels": False,
                        "min_samples": 500,
                        "gpu_recommended": True,
                        "note": "Entraîne uniquement sur images normales"
                    },
                    {
                        "id": "variational_autoencoder",
                        "name": "VAE (Variational)",
                        "description": "Modèle génératif probabiliste - Robuste aux variations naturelles", 
                        "icon": "🌌",
                        "complexity": "Avancé",
                        "training_time": "Long (~20-40 min)",
                        "use_cases": ["Données multimodales", "Anomalies subtiles", "Génération d'images"],
                        "requires_labels": False,
                        "min_samples": 1000,
                        "gpu_recommended": True,
                        "note": "Meilleur pour anomalies complexes"
                    },
                    {
                        "id": "denoising_autoencoder",
                        "name": "AutoEncodeur Denoiseur",
                        "description": "Apprend à débruiter les images - Très robuste en environnement réel bruité",
                        "icon": "🧹", 
                        "complexity": "Intermédiaire",
                        "training_time": "Moyen (~15-25 min)",
                        "use_cases": ["Données bruitées", "Environnements industriels", "Surveillance"],
                        "requires_labels": False,
                        "min_samples": 500,
                        "gpu_recommended": True,
                        "note": "Robuste au bruit et aux perturbations"
                    },
                    {
                        "id": "patch_core",
                        "name": "PatchCore",
                        "description": "Mémoire bank de patchs avec coreset sampling - State-of-the-art pour défauts locaux",
                        "icon": "🧩",
                        "complexity": "Expert", 
                        "training_time": "Variable (~10-30 min)",
                        "use_cases": ["Anomalies locales", "Industrie 4.0", "Défauts de surface"],
                        "requires_labels": False,
                        "min_samples": 200,
                        "gpu_recommended": True,
                        "note": "⚠️ Nécessite FAISS installé | Excellent pour localisation précise"
                    }
                ]
            },
            
            "📐 Similarité & Métrique": {
                "color": "#ffc107",
                "description": "Apprentissage métrique pour comparaison et recherche d'images",
                "models": [
                    {
                        "id": "siamese_network",
                        "name": "Réseau Siamois",
                        "description": "Apprentissage métrique par paires - Compare similarité entre images", 
                        "icon": "👯",
                        "complexity": "Avancé",
                        "training_time": "Long (~30-60 min)",
                        "use_cases": ["Re-identification", "Recherche visuelle", "One-shot learning"],
                        "requires_labels": True,
                        "min_samples": 500,
                        "gpu_recommended": True,
                        "note": "⚠️ ATTENTION: Nécessite des paires d'images (similaires/dissimilaires)",
                        "special_data_format": "pairs"
                    }
                ]
            }
        }
    
    def render_model_selection_step(self):
        """Étape 4: Sélection et configuration du modèle avec interface moderne"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🤖 Étape 4: Sélection du Modèle")
        
        st.markdown("### 🎯 Choisissez votre architecture de modèle")
        
        categories = self.get_model_categories()
        
        # Navigation par catégories avec tabs
        category_tabs = st.tabs([f"{category}" for category in categories.keys()])
        
        for i, (category, category_data) in enumerate(categories.items()):
            with category_tabs[i]:
                st.markdown(f"**{category}** - {len(category_data['models'])} modèles disponibles")
                
                # Affichage des modèles en grille responsive
                model_cols = st.columns(2)
                
                for idx, model in enumerate(category_data["models"]):
                    col = model_cols[idx % 2]
                    
                    with col:
                        # Vérification de la compatibilité avec les données
                        has_labels = 'y_train' in st.session_state and st.session_state.y_train is not None
                        is_compatible = has_labels or not model["requires_labels"]
                        is_selected = st.session_state.selected_model_type == model["id"]
                        
                        card_class = "model-card selected" if is_selected else "model-card"
                        
                        card_content = f"""
                        <div class="{card_class}" style="opacity: {'1' if is_compatible else '0.6'};">
                            <div style="display: flex; align-items: start; margin-bottom: 1rem;">
                                <span style="font-size: 2rem; margin-right: 1rem;">{model['icon']}</span>
                                <div style="flex: 1;">
                                    <h4 style="margin: 0 0 0.5rem 0; color: #333;">{model['name']}</h4>
                                    <span class="status-badge badge-{'success' if is_compatible else 'warning'}">
                                        {'✅ Compatible' if is_compatible else '⚠️ Labels requis'}
                                    </span>
                                    <span class="status-badge" style="background: {category_data['color']}; color: white;">
                                        {model['complexity']}
                                    </span>
                                </div>
                            </div>
                            <p style="color: #666; font-size: 0.9rem; margin-bottom: 1rem;">{model['description']}</p>
                            <div style="margin-bottom: 1rem;">
                        """
                        
                        for use_case in model['use_cases']:
                            card_content += f'<span class="status-badge badge-info">{use_case}</span>'
                        
                        card_content += "</div></div>"
                        
                        st.markdown(card_content, unsafe_allow_html=True)
                        
                        # Bouton de sélection
                        if is_compatible:
                            if st.button(
                                "✅ Sélectionné" if is_selected else "📝 Sélectionner",
                                key=f"select_{model['id']}",
                                use_container_width=True,
                                type="primary" if is_selected else "secondary"
                            ):
                                st.session_state.selected_model_type = model["id"]
                                st.session_state.model_config = {
                                    "model_type": model["id"],
                                    "model_params": self.get_default_model_params(model["id"])
                                }
                                st.success(f"✅ {model['name']} sélectionné")
                                st.rerun()
                        else:
                            st.button(
                                "🔒 Labels requis",
                                key=f"disabled_{model['id']}",
                                use_container_width=True,
                                disabled=True,
                                help="Ce modèle nécessite des labels d'entraînement"
                            )
        
        # Configuration avancée si modèle sélectionné
        if st.session_state.selected_model_type:
            st.markdown("---")
            st.subheader(f"⚙️ Configuration Avancée - {st.session_state.selected_model_type.upper()}")
            
            self.render_model_specific_parameters()
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
        
        with col_nav2:
            if st.button("💾 Continuer vers l'Entraînement ➡️", type="primary", use_container_width=True):
                if st.session_state.selected_model_type:
                    st.session_state.current_step = 4
                    st.rerun()
                else:
                    st.error("❌ Veuillez sélectionner un modèle")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def get_default_model_params(self, model_type):
        """
        Retourne les paramètres par défaut optimisés pour chaque modèle.
        
        Args:
            model_type (str): Type de modèle (ex: "simple_cnn", "patch_core")
            
        Returns:
            dict: Dictionnaire des paramètres par défaut
        """
        defaults = {
            # ===== CLASSIFICATION SUPERVISÉE =====
            "simple_cnn": {
                "input_channels": 3,
                "num_classes": 2,
                "base_filters": 32,
                "dropout_rate": 0.5,
                "use_batch_norm": True
            },
            
            "custom_resnet": {
                "input_channels": 3,
                "num_classes": 2,
                "base_filters": 64,
                "num_blocks": [2, 2, 2, 2],
                "dropout_rate": 0.3,
                "use_batch_norm": True
            },
            
            "transfer_learning": {
                "input_channels": 3,
                "num_classes": 2,
                "backbone_name": "resnet50",
                "pretrained": True,
                "freeze_layers": 0,
                "dropout_rate": 0.5,
                "use_custom_classifier": True
            },
            
            # ===== DÉTECTION D'ANOMALIES =====
            "conv_autoencoder": {
                "input_channels": 3,
                "latent_dim": 256,
                "base_filters": 32,
                "num_stages": 4,
                "dropout_rate": 0.2,
                "use_skip_connections": False,
                "use_vae": False
            },
            
            "variational_autoencoder": {
                "input_channels": 3,
                "latent_dim": 128,
                "base_filters": 32,
                "num_stages": 4,
                "dropout_rate": 0.2,
                "use_skip_connections": False,
                "beta": 1.0  # Poids KL divergence
            },
            
            "denoising_autoencoder": {
                "input_channels": 3,
                "latent_dim": 256,
                "base_filters": 32,
                "num_stages": 4,
                "dropout_rate": 0.2,
                "noise_factor": 0.1,
                "use_skip_connections": False
            },
            
            "patch_core": {
                "backbone_name": "wide_resnet50_2",
                "patchcore_layers": ["layer2", "layer3"],
                "faiss_index_type": "Flat",
                "coreset_ratio": 0.01,
                "num_neighbors": 1,
                "patch_size": 3,
                "stride": 1
            },
            
            # ===== SIMILARITÉ & MÉTRIQUE =====
            "siamese_network": {
                "input_channels": 3,
                "backbone_name": "resnet18",
                "embedding_dim": 128,
                "margin": 1.0,
                "distance_metric": "euclidean",
                "dropout_rate": 0.3
            }
        }
        
        # Retour sécurisé avec fallback
        if model_type not in defaults:
            logger.warning(f"Paramètres par défaut non trouvés pour {model_type}, utilisation config minimale")
            return {
                "input_channels": 3,
                "num_classes": 2
            }
        
        return defaults[model_type]
    
    def render_model_specific_parameters(self):
        """
        Affiche les paramètres spécifiques au modèle sélectionné avec interface moderne.      
        Gère les configurations pour tous les modèles avec validation et aide contextuelle.
        """
        model_type = st.session_state.selected_model_type
        model_params = st.session_state.model_config.get("model_params", {})
        
        st.markdown("#### 🔧 Paramètres du Modèle")
        
        # ========================================================================
        # CLASSIFICATION: SIMPLE CNN & CUSTOM RESNET
        # ========================================================================
        if model_type in ["simple_cnn", "custom_resnet"]:
            col1, col2 = st.columns(2)
            
            with col1:
                base_filters = st.slider(
                    "Filtres de base",
                    min_value=16,
                    max_value=128,
                    value=model_params.get("base_filters", 32 if model_type == "simple_cnn" else 64),
                    step=16,
                    help="🎯 Nombre de filtres dans la première couche. Plus = plus de capacité mais plus lent."
                )
            
            with col2:
                dropout_rate = st.slider(
                    "Taux de dropout",
                    min_value=0.0,
                    max_value=0.7,
                    value=model_params.get("dropout_rate", 0.5),
                    step=0.1,
                    help="🛡️ Régularisation contre l'overfitting. 0.3-0.5 recommandé."
                )
            
            st.session_state.model_config["model_params"].update({
                "base_filters": base_filters,
                "dropout_rate": dropout_rate
            })
            
            st.info(f"💡 **{model_type.replace('_', ' ').title()}** : {base_filters * 4} filtres max | Dropout {dropout_rate:.1%}")
        
        # ========================================================================
        # CLASSIFICATION: TRANSFER LEARNING
        # ========================================================================
        elif model_type == "transfer_learning":
            col1, col2 = st.columns(2)
            
            with col1:
                backbone_name = st.selectbox(
                    "Architecture de base",
                    ["resnet18", "resnet50", "resnet101", "efficientnet_b0", "wide_resnet50_2"],
                    index=1,
                    help="🏗️ Backbone pré-entraîné. ResNet50 = bon compromis. EfficientNet = plus léger."
                )
                
                pretrained = st.checkbox(
                    "✅ Utiliser poids ImageNet",
                    value=model_params.get("pretrained", True),
                    help="⚡ FORTEMENT RECOMMANDÉ pour de meilleures performances"
                )
            
            with col2:
                freeze_layers = st.select_slider(
                    "Stratégie de fine-tuning",
                    options=[-1, 0, 50, 100, 150],
                    value=model_params.get("freeze_layers", 0),
                    format_func=lambda x: {
                        -1: "🔒 Feature Extraction (gel complet)",
                        0: "🔄 Fine-tuning complet",
                        50: "⚡ Partiel (50 couches gelées)",
                        100: "🎯 Léger (100 couches gelées)",
                        150: "🔧 Feature extraction avancé"
                    }[x],
                    help="🎚️ Contrôle l'adaptation du modèle. 0 = apprentissage complet."
                )
            
            dropout_rate = st.slider(
                "Dropout du classifieur",
                min_value=0.0,
                max_value=0.7,
                value=model_params.get("dropout_rate", 0.5),
                step=0.1,
                help="🛡️ Dropout des couches fully-connected finales"
            )
            
            st.session_state.model_config["model_params"].update({
                "backbone_name": backbone_name,
                "pretrained": pretrained,
                "freeze_layers": freeze_layers,
                "dropout_rate": dropout_rate
            })
            
            # Estimation paramètres
            params_estimate = {
                "resnet18": "11.7M",
                "resnet50": "25.6M",
                "resnet101": "44.5M",
                "efficientnet_b0": "5.3M",
                "wide_resnet50_2": "68.9M"
            }
            
            st.success(f"✅ **{backbone_name}** sélectionné (~{params_estimate.get(backbone_name, 'N/A')} paramètres)")
        
        # ========================================================================
        # ANOMALIES: AUTOENCODERS (Conv / VAE / Denoising)
        # ========================================================================
        elif model_type in ["conv_autoencoder", "variational_autoencoder", "denoising_autoencoder"]:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                latent_dim = st.slider(
                    "Dimension latente",
                    min_value=32,
                    max_value=512,
                    value=model_params.get("latent_dim", 256 if model_type != "variational_autoencoder" else 128),
                    step=32,
                    help="🧠 Taille de l'espace compressé. Plus grand = plus de détails conservés."
                )
            
            with col2:
                base_filters = st.slider(
                    "Filtres de base",
                    min_value=16,
                    max_value=128,
                    value=model_params.get("base_filters", 32),
                    step=16,
                    help="🎯 Filtres du premier bloc. Affecte la capacité du modèle."
                )
            
            with col3:
                num_stages = st.slider(
                    "Profondeur (stages)",
                    min_value=2,
                    max_value=6,
                    value=model_params.get("num_stages", 4),
                    help="🏗️ Nombre de blocs encodeur/décodeur. 4 = bon compromis."
                )
            
            # Paramètres spécifiques
            if model_type == "denoising_autoencoder":
                noise_factor = st.slider(
                    "🌫️ Facteur de bruit",
                    min_value=0.0,
                    max_value=0.5,
                    value=model_params.get("noise_factor", 0.1),
                    step=0.05,
                    help="Intensité du bruit gaussien ajouté pendant l'entraînement"
                )
                st.session_state.model_config["model_params"]["noise_factor"] = noise_factor
            
            elif model_type == "variational_autoencoder":
                beta = st.slider(
                    "β (Beta) - Poids KL",
                    min_value=0.1,
                    max_value=10.0,
                    value=model_params.get("beta", 1.0),
                    step=0.1,
                    help="🎚️ Balance reconstruction vs régularisation. 1.0 = β-VAE standard."
                )
                st.session_state.model_config["model_params"]["beta"] = beta
            
            dropout_rate = st.slider(
                "Dropout",
                min_value=0.0,
                max_value=0.5,
                value=model_params.get("dropout_rate", 0.2),
                step=0.1,
                help="🛡️ Régularisation (généralement plus faible pour autoencoders)"
            )
            
            st.session_state.model_config["model_params"].update({
                "latent_dim": latent_dim,
                "base_filters": base_filters,
                "num_stages": num_stages,
                "dropout_rate": dropout_rate
            })
            
            # Info compression ratio
            compression_ratio = latent_dim / (64 * 64 * 3)  # Estimation pour 64x64 RGB
            st.info(f"📊 **Taux de compression estimé** : ~{compression_ratio:.2%} | Latent: {latent_dim}D")
        
        # ========================================================================
        # ANOMALIES: PATCHCORE
        # ========================================================================
        elif model_type == "patch_core":
            st.warning("⚠️ **PatchCore nécessite FAISS installé** : `pip install faiss-cpu` ou `faiss-gpu`")
            
            col1, col2 = st.columns(2)
            
            with col1:
                backbone_name = st.selectbox(
                    "Backbone d'extraction",
                    ["resnet18", "resnet50", "wide_resnet50_2"],
                    index=2,
                    help="🏗️ WideResNet50 recommandé pour précision maximale"
                )
                
                layers = st.multiselect(
                    "Couches d'extraction",
                    ["layer1", "layer2", "layer3"],
                    default=model_params.get("patchcore_layers", ["layer2", "layer3"]),
                    help="🎯 Couches du backbone pour extraire les features"
                )
            
            with col2:
                coreset_ratio = st.slider(
                    "Ratio Coreset",
                    min_value=0.001,
                    max_value=0.1,
                    value=model_params.get("coreset_ratio", 0.01),
                    step=0.001,
                    format="%.3f",
                    help="💾 % de patchs conservés. Plus petit = plus rapide mais moins précis."
                )
                
                num_neighbors = st.slider(
                    "k-NN (voisins)",
                    min_value=1,
                    max_value=9,
                    value=model_params.get("num_neighbors", 1),
                    help="🔍 Nombre de plus proches voisins. 1 = distance minimale."
                )
            
            # Options avancées
            with st.expander("⚙️ Configuration Avancée"):
                faiss_index_type = st.selectbox(
                    "Type d'index FAISS",
                    ["Flat", "IVFFlat"],
                    help="Flat = exact (lent) | IVFFlat = approximatif (rapide)"
                )
                
                patch_size = st.slider("Taille des patchs", 1, 5, 3)
                stride = st.slider("Stride d'extraction", 1, 4, 1)
            
            st.session_state.model_config["model_params"].update({
                "backbone_name": backbone_name,
                "patchcore_layers": layers,
                "coreset_ratio": coreset_ratio,
                "num_neighbors": num_neighbors,
                "faiss_index_type": faiss_index_type,
                "patch_size": patch_size,
                "stride": stride
            })
            
            # Estimation mémoire
            n_patches_estimate = int(1000 / coreset_ratio)
            st.info(f"💾 **Mémoire bank estimée** : ~{n_patches_estimate:,} patchs | Backbone: {backbone_name}")
        
        # ========================================================================
        # SIMILARITÉ: SIAMESE NETWORK
        # ========================================================================
        elif model_type == "siamese_network":
            st.error("""
            ⚠️ **ATTENTION : Format de Données Spécial Requis**
            
            Le réseau Siamois nécessite des **paires d'images** :
            - Paires **positives** : Images similaires (même classe/personne)
            - Paires **négatives** : Images dissimilaires (classes différentes)
            
            📊 Votre dataset actuel n'est probablement **PAS compatible**.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                backbone_name = st.selectbox(
                    "Architecture Backbone",
                    ["resnet18", "resnet50", "efficientnet_b0"],
                    index=0,
                    help="🏗️ ResNet18 recommandé pour démarrer"
                )
                
                embedding_dim = st.slider(
                    "Dimension embeddings",
                    min_value=64,
                    max_value=512,
                    value=model_params.get("embedding_dim", 128),
                    step=32,
                    help="🧠 Taille de l'espace de représentation. 128 = bon compromis."
                )
            
            with col2:
                margin = st.slider(
                    "Marge (Contrastive Loss)",
                    min_value=0.5,
                    max_value=2.0,
                    value=model_params.get("margin", 1.0),
                    step=0.1,
                    help="🎚️ Distance minimale entre paires négatives. 1.0 = standard."
                )
                
                distance_metric = st.selectbox(
                    "Métrique de distance",
                    ["euclidean", "cosine"],
                    help="📐 Euclidean = L2 | Cosine = angle entre vecteurs"
                )
            
            dropout_rate = st.slider(
                "Dropout",
                min_value=0.0,
                max_value=0.5,
                value=model_params.get("dropout_rate", 0.3),
                step=0.1
            )
            
            st.session_state.model_config["model_params"].update({
                "backbone_name": backbone_name,
                "embedding_dim": embedding_dim,
                "margin": margin,
                "distance_metric": distance_metric,
                "dropout_rate": dropout_rate
            })
            
            st.warning("""
            💡 **Pour utiliser ce modèle** :
            1. Organisez vos données en paires (anchor, positive, negative)
            2. Implémentez un DataLoader spécialisé
            3. Utilisez une Contrastive Loss ou Triplet Loss
            
            📚 Référez-vous à la documentation pour plus de détails.
            """)
            
            st.info(f"📊 **Embedding** : {embedding_dim}D | Marge: {margin} | Distance: {distance_metric}")
    
    # ============================================================================
    # ÉTAPE 5: CONFIGURATION DE L'ENTRAÎNEMENT
    # ============================================================================
    
    def render_training_config_step(self):
        """Étape 5: Configuration des hyperparamètres d'entraînement"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("⚙️ Étape 5: Configuration de l'Entraînement")
        
        st.markdown("**Hyperparamètres d'entraînement**")
        
        # Configuration de base
        col_hyper1, col_hyper2, col_hyper3 = st.columns(3)
        
        with col_hyper1:
            epochs = st.slider(
                "Nombre d'Époques",
                min_value=5,
                max_value=200,
                value=50,
                step=5,
                help="Nombre de passages complets sur le dataset. 50-100 recommandé pour la plupart des cas."
            )
        
        with col_hyper2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
                value=1e-4,
                format_func=lambda x: f"{x:.0e}",
                help="Taux d'apprentissage. 1e-4 recommandé pour le fine-tuning, 1e-3 pour l'entraînement from scratch."
            )
        
        with col_hyper3:
            batch_size = st.selectbox(
                "Batch Size",
                options=[8, 16, 32, 64, 128],
                index=2,
                help="Nombre d'images par batch. 32 recommandé pour la plupart des GPUs."
            )
        
        st.markdown("---")
        
        # Optimiseur et Scheduler
        st.subheader("🎯 Optimiseur et Scheduler")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            if LOGIC_METIER_AVAILABLE:
                optimizer = st.selectbox(
                    "Optimiseur",
                    options=[opt.value for opt in OptimizerType],
                    index=0,
                    help="AdamW recommandé pour la plupart des cas (meilleure généralisation que Adam)"
                )
            else:
                optimizer = st.selectbox(
                    "Optimiseur",
                    options=["adamw", "adam", "sgd", "rmsprop"],
                    index=0
                )
        
        with col_opt2:
            if LOGIC_METIER_AVAILABLE:
                scheduler = st.selectbox(
                    "Learning Rate Scheduler",
                    options=[sched.value for sched in SchedulerType],
                    index=0,
                    help="ReduceLROnPlateau réduit automatiquement le LR quand la loss stagne"
                )
            else:
                scheduler = st.selectbox(
                    "Scheduler",
                    options=["reduce_on_plateau", "cosine", "step", "none"],
                    index=0
                )
        
        st.markdown("---")
        
        # Early Stopping et Régularisation
        st.subheader("🛑 Early Stopping & Régularisation")
        
        col_callback1, col_callback2, col_callback3 = st.columns(3)
        
        with col_callback1:
            early_stopping_patience = st.slider(
                "Early Stopping Patience",
                min_value=3,
                max_value=30,
                value=10,
                help="Arrête l'entraînement si pas d'amélioration pendant N époques. 10-15 recommandé."
            )
        
        with col_callback2:
            reduce_lr_patience = st.slider(
                "Reduce LR Patience",
                min_value=2,
                max_value=15,
                value=5,
                help="Réduit le LR si pas d'amélioration pendant N époques. 5-8 recommandé."
            )
        
        with col_callback3:
            weight_decay = st.select_slider(
                "Weight Decay",
                options=[0.0, 0.001, 0.01, 0.1],
                value=0.01,
                help="Régularisation L2 pour éviter l'overfitting. 0.01 recommandé."
            )
        
        # Options avancées
        with st.expander("🔧 Options Avancées"):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                gradient_clip = st.slider(
                    "Gradient Clipping",
                    min_value=0.0,
                    max_value=5.0,
                    value=1.0,
                    step=0.5,
                    help="Limite l'amplitude des gradients pour stabiliser l'entraînement"
                )
                
                deterministic = st.checkbox(
                    "Mode Déterministe",
                    value=True,
                    help="Rend les résultats reproductibles (seed fixé)"
                )
            
            with col_adv2:
                use_mixed_precision = st.checkbox(
                    "Mixed Precision (FP16)",
                    value=torch.cuda.is_available(),
                    disabled=not torch.cuda.is_available(),
                    help="Accélère l'entraînement sur GPU récents (Volta+). Requiert CUDA."
                )
                
                num_workers = st.slider(
                    "DataLoader Workers",
                    min_value=0,
                    max_value=8,
                    value=4,
                    help="Nombre de processus pour charger les données en parallèle"
                )
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.current_step = 3
                st.rerun()
        
        with col_nav2:
            if st.button("💾 Sauvegarder et Continuer ➡️", type="primary", use_container_width=True):
                # Création de la configuration d'entraînement
                if LOGIC_METIER_AVAILABLE:
                    st.session_state.training_config = TrainingConfig(
                        epochs=epochs,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        gradient_clip=gradient_clip,
                        optimizer=OptimizerType(optimizer),
                        scheduler=SchedulerType(scheduler),
                        early_stopping_patience=early_stopping_patience,
                        reduce_lr_patience=reduce_lr_patience,
                        use_class_weights=st.session_state.imbalance_config.get('use_class_weights', False),
                        deterministic=deterministic,
                        use_mixed_precision=use_mixed_precision,
                        num_workers=num_workers,
                        seed=42
                    )
                else:
                    # Fallback si la logique métier n'est pas disponible
                    st.session_state.training_config = {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "gradient_clip": gradient_clip,
                        "optimizer": optimizer,
                        "scheduler": scheduler,
                        "early_stopping_patience": early_stopping_patience,
                        "reduce_lr_patience": reduce_lr_patience,
                        "use_class_weights": st.session_state.imbalance_config.get('use_class_weights', False),
                        "deterministic": deterministic,
                        "use_mixed_precision": use_mixed_precision,
                        "num_workers": num_workers,
                        "seed": 42
                    }
                
                st.success("✅ Configuration d'entraînement sauvegardée")
                st.session_state.current_step = 5
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ============================================================================
    # ÉTAPE 6: LANCEMENT ET MONITORING
    # ============================================================================
    
    def render_training_launch_step(self):
        """Étape 6: Lancement et monitoring de l'entraînement"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🚀 Étape 6: Lancement de l'Entraînement")
        
        st.markdown("**Récapitulatif de la Configuration**")
        
        # Affichage du récapitulatif en deux colonnes
        col_summary1, col_summary2 = st.columns(2)
        
        with col_summary1:
            st.subheader("📊 Données et Préparation")
            
            if 'split_config' in st.session_state:
                st.json(st.session_state.split_config)
            
            st.subheader("⚖️ Gestion du Déséquilibre")
            st.json(st.session_state.imbalance_config)
            
            st.subheader("🎨 Prétraitement")
            st.json(st.session_state.preprocessing_config)
        
        with col_summary2:
            st.subheader("🤖 Modèle")
            st.json(st.session_state.model_config)
            
            st.subheader("⚙️ Entraînement")
            if LOGIC_METIER_AVAILABLE and isinstance(st.session_state.training_config, TrainingConfig):
                # Conversion pour l'affichage
                training_config_dict = {
                    'epochs': st.session_state.training_config.epochs,
                    'batch_size': st.session_state.training_config.batch_size,
                    'learning_rate': st.session_state.training_config.learning_rate,
                    'weight_decay': st.session_state.training_config.weight_decay,
                    'optimizer': st.session_state.training_config.optimizer.value,
                    'scheduler': st.session_state.training_config.scheduler.value,
                    'early_stopping_patience': st.session_state.training_config.early_stopping_patience,
                    'reduce_lr_patience': st.session_state.training_config.reduce_lr_patience,
                    'use_class_weights': st.session_state.training_config.use_class_weights
                }
                st.json(training_config_dict)
            else:
                st.json(st.session_state.training_config)
        
        # Validation finale de la configuration
        st.markdown("---")
        st.subheader("🔍 Validation de la Configuration")
        
        errors, warnings = self.validate_training_configuration()
        
        if errors:
            for error in errors:
                st.markdown(f'<div class="config-error">{error}</div>', unsafe_allow_html=True)
            launch_disabled = True
        else:
            if warnings:
                for warning in warnings:
                    st.warning(warning)
            st.success("✅ Configuration valide - Prêt pour l'entraînement!")
            launch_disabled = False
        
        # Informations de lancement
        st.markdown("---")
        st.subheader("🎯 Informations de Lancement")
        
        col_launch1, col_launch2, col_launch3 = st.columns(3)
        
        with col_launch1:
            total_train_images = len(st.session_state.get('X_train', []))
            if st.session_state.preprocessing_config.get("augmentation_enabled", False):
                total_train_images *= st.session_state.preprocessing_config.get("augmentation_factor", 1)
            st.metric("📷 Images Train", f"{total_train_images:,}")
        
        with col_launch2:
            epochs = st.session_state.training_config.epochs if hasattr(st.session_state.training_config, 'epochs') else st.session_state.training_config.get('epochs', 50)
            batch_size = st.session_state.training_config.batch_size if hasattr(st.session_state.training_config, 'batch_size') else st.session_state.training_config.get('batch_size', 32)
            training_time_estimate = (total_train_images * epochs) / (batch_size * 100)  # Estimation rough
            st.metric("⏱️ Temps estimé", f"{max(1, int(training_time_estimate))} min")
        
        with col_launch3:
            use_weights = st.session_state.imbalance_config.get("use_class_weights", False)
            st.metric("⚖️ Poids de classe", "Activés" if use_weights else "Désactivés")
        
        # Informations système
        st.markdown("---")
        st.subheader("💻 Informations Système")
        
        col_sys1, col_sys2, col_sys3 = st.columns(3)
        
        with col_sys1:
            device = "CUDA 🚀" if torch.cuda.is_available() else "CPU ⚡"
            st.info(f"**Device:** {device}")
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                st.success(f"**GPU:** {gpu_name} ({gpu_memory:.1f} GB)")
        
        with col_sys2:
            mixed_precision = st.session_state.training_config.use_mixed_precision if hasattr(st.session_state.training_config, 'use_mixed_precision') else st.session_state.training_config.get('use_mixed_precision', False)
            st.info(f"**Mixed Precision:** {'Activée 🚀' if mixed_precision else 'Désactivée'}")
        
        with col_sys3:
            deterministic = st.session_state.training_config.deterministic if hasattr(st.session_state.training_config, 'deterministic') else st.session_state.training_config.get('deterministic', True)
            st.info(f"**Mode Déterministe:** {'Activé ✅' if deterministic else 'Désactivé'}")
        
        # Bouton de lancement principal
        st.markdown("---")
        
        if st.button("🚀 Démarrer l'Entraînement", type="primary", use_container_width=True, disabled=launch_disabled):
            self.launch_training()
        
        # Navigation
        st.markdown("---")
        col_back, _ = st.columns(2)
        with col_back:
            if st.button("⬅️ Retour", use_container_width=True):
                st.session_state.current_step = 4
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def validate_training_configuration(self):
        """Valide la configuration complète avant lancement"""
        errors = []
        warnings = []
        
        # Vérification des données
        required_data = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
        for data_key in required_data:
            if data_key not in st.session_state:
                errors.append(f"❌ Données manquantes: {data_key}")
        
        # Vérification modèle
        if 'model_config' not in st.session_state or not st.session_state.model_config:
            errors.append("❌ Configuration du modèle manquante")
        
        # Vérification entraînement
        if 'training_config' not in st.session_state or not st.session_state.training_config:
            errors.append("❌ Configuration d'entraînement manquante")
        
        # Validation des données avec DataValidator (logique métier)
        if LOGIC_METIER_AVAILABLE:
            for dataset_key, name in [('X_train', 'train'), ('X_val', 'val'), ('X_test', 'test')]:
                dataset = st.session_state.get(dataset_key)
                labels = st.session_state.get(f'y_{name}')
                
                if dataset is not None and labels is not None:
                    val_result = DataValidator.validate_input_data(dataset, labels, name)
                    if not val_result.success:
                        errors.append(f"❌ Validation {name}: {val_result.error}")
        
        # Vérifications spécifiques
        if st.session_state.training_config:
            epochs = st.session_state.training_config.epochs if hasattr(st.session_state.training_config, 'epochs') else st.session_state.training_config.get('epochs', 50)
            batch_size = st.session_state.training_config.batch_size if hasattr(st.session_state.training_config, 'batch_size') else st.session_state.training_config.get('batch_size', 32)
            
            if epochs > 100:
                warnings.append("⚠️ Nombre d'époques élevé - entraînement potentiellement long")
            if batch_size > 64 and not torch.cuda.is_available():
                warnings.append("⚠️ Batch size élevé - risque de mémoire insuffisante sur CPU")
        
        return errors, warnings
    
    def launch_training(self):
        """Lance l'entraînement avec la configuration complète"""
        training_container = st.container()
        with training_container:
            st.markdown("### 📈 Entraînement en Cours...")
            
            # Initialisation des composants d'interface
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            results_placeholder = st.empty()
            
            try:
                # Configuration des callbacks pour l'interface
                streamlit_components = {
                    "progress_bar": progress_bar,
                    "status_text": status_text,
                    "metrics_placeholder": metrics_placeholder
                }
                
                # Détermination du type d'anomalie pour les modèles non supervisés
                model_type = st.session_state.model_config["model_type"]
                anomaly_type = None
                if model_type in ["conv_autoencoder", "variational_autoencoder", "denoising_autoencoder", "patch_core"]:
                    anomaly_type = "structural"
                
                # Lancement de l'entraînement avec la logique métier
                if LOGIC_METIER_AVAILABLE:
                    model, history = self.train_with_metier_logic(
                        streamlit_components, 
                        anomaly_type
                    )
                else:
                    # Fallback simulation si la logique métier n'est pas disponible
                    model, history = self.train_simulation(
                        streamlit_components
                    )
                
                # Traitement des résultats
                if model is not None and history.get("success", False):
                    self.handle_training_success(model, history, results_placeholder)
                else:
                    self.handle_training_failure(history, results_placeholder)
                    
            except Exception as e:
                self.handle_training_error(e, results_placeholder)
    
    def train_with_metier_logic(self, streamlit_components, anomaly_type):
        """Utilise la logique métier existante pour l'entraînement"""
        
        try:
            # =============================================================================
            # ÉTAPE 1: VALIDATION DES DONNÉES (SANS PREPROCESSING)
            # =============================================================================
            
            # Debug: afficher les shapes originales
            logger.info(f"Données originales - X_train shape: {st.session_state.X_train.shape}")
            logger.info(f"Données originales - X_val shape: {st.session_state.X_val.shape}")
            
            # Validation basique des données
            if (st.session_state.X_train is None or st.session_state.X_val is None or
                len(st.session_state.X_train) == 0 or len(st.session_state.X_val) == 0):
                return None, {'success': False, 'error': "Données d'entraînement invalides"}
            
            # =============================================================================
            # ÉTAPE 2: CONFIGURATION DES CALLBACKS POUR L'INTERFACE
            # =============================================================================
            
            callbacks = []
            if streamlit_components:
                callbacks.append(StreamlitCallback(
                    progress_bar=streamlit_components.get('progress_bar'),
                    status_text=streamlit_components.get('status_text'),
                    total_epochs=st.session_state.training_config.epochs
                ))
            callbacks.append(LoggingCallback(log_every_n_epochs=5))
            
            # =============================================================================
            # ÉTAPE 3: CONFIGURATION DU MODÈLE
            # =============================================================================
            
            model_config = ModelConfig(
                model_type=ModelType(st.session_state.model_config["model_type"]),
                num_classes=st.session_state.model_config["model_params"].get("num_classes", 2),
                input_channels=st.session_state.model_config["model_params"].get("input_channels", 3),
                dropout_rate=st.session_state.model_config["model_params"].get("dropout_rate", 0.5),
                base_filters=st.session_state.model_config["model_params"].get("base_filters", 32),
                latent_dim=st.session_state.model_config["model_params"].get("latent_dim", 256),
                num_stages=st.session_state.model_config["model_params"].get("num_stages", 4)
            )
            
            # =============================================================================
            # ÉTAPE 4: LANCEMENT DE L'ENTRAÎNEMENT AVEC DONNÉES BRUTES
            # LE TRAINER GÈRE SON PROPRE PREPROCESSING INTERNE
            # =============================================================================
            
            if anomaly_type:
                # Cas des modèles de détection d'anomalies (Autoencoders, etc.)
                trainer = AnomalyAwareTrainer(
                    anomaly_type=anomaly_type,
                    model_config=model_config,
                    training_config=st.session_state.training_config,
                    taxonomy_config=None,
                    callbacks=callbacks
                )
                result = trainer.train(
                    st.session_state.X_train,  # ← DONNÉES BRUTES
                    st.session_state.y_train, 
                    st.session_state.X_val,    # ← DONNÉES BRUTES
                    st.session_state.y_val
                )
            else:
                # Cas des modèles de classification standard
                trainer = ComputerVisionTrainer(
                    model_config=model_config,
                    training_config=st.session_state.training_config,
                    callbacks=callbacks
                )
                result = trainer.fit(
                    st.session_state.X_train,  # ← DONNÉES BRUTES
                    st.session_state.y_train, 
                    st.session_state.X_val,    # ← DONNÉES BRUTES
                    st.session_state.y_val
                )
            
            # =============================================================================
            # ÉTAPE 5: RÉCUPÉRATION DES RÉSULTATS ET DU PREPROCESSOR
            # =============================================================================
            
            if result.success:
                # RÉCUPÉRATION CRITIQUE: le preprocessor créé par le trainer
                preprocessor = getattr(trainer, 'preprocessor', None)
                
                if preprocessor is None:
                    logger.warning("Aucun preprocessor trouvé dans le trainer")
                    # Créer un preprocessor de secours basé sur la configuration
                    preprocessor = DataPreprocessor(
                        strategy=st.session_state.preprocessing_config.get("strategy", "standardize"),
                        auto_detect_format=True
                    )
                    # Fit sur les données d'entraînement pour l'évaluation
                    preprocessor.fit(st.session_state.X_train)
                
                # SAUVEGARDE CRITIQUE: le preprocessor pour l'évaluation
                st.session_state.preprocessor = preprocessor
                
                # Préparation de l'historique avec structure garantie
                history_data = result.data['history']
                
                # Construction de l'historique final (GARANTIE SANS BOOLÉENS)
                history = {
                    'success': True,
                    'train_loss': [float(x) for x in history_data.get('train_loss', [])],
                    'val_loss': [float(x) for x in history_data.get('val_loss', [])],
                    'val_accuracy': [float(x) for x in history_data.get('val_accuracy', [])],
                    'val_f1': [float(x) for x in history_data.get('val_f1', [])],
                    'learning_rates': [float(x) for x in history_data.get('learning_rates', [])],
                    'best_epoch': int(history_data.get('best_epoch', 0)),
                    'best_val_loss': float(history_data.get('best_val_loss', 0)),
                    'training_time': float(history_data.get('training_time', 0)),
                    'total_epochs_trained': int(history_data.get('total_epochs_trained', 0)),
                    'early_stopping_triggered': bool(history_data.get('early_stopping_triggered', False)),
                    'model_type': str(history_data.get('model_type', '')),
                    'input_shape': history_data.get('input_shape'),
                    'anomaly_type': anomaly_type,
                    'preprocessor_available': preprocessor is not None,
                    'preprocessor_config': preprocessor.get_config() if preprocessor else None
                }
                
                logger.info(f"✅ Preprocessor sauvegardé: {preprocessor is not None}")
                if preprocessor:
                    logger.info(f"Config preprocessor: {preprocessor.get_config()}")
                
                # SAUVEGARDE CRITIQUE: le modèle ET le preprocessor pour l'évaluation
                return trainer.model, history
                
            else:
                # En cas d'échec, retourner l'erreur
                logger.error(f"Échec de l'entraînement: {result.error}")
                return None, {'success': False, 'error': result.error}
                
        except Exception as e:
            # Gestion robuste des erreurs avec logging détaillé
            logger.error(f"Erreur lors de l'entraînement: {e}", exc_info=True)
            
            # Informations de debug pour diagnostiquer le problème
            debug_info = {
                'X_train_shape': getattr(st.session_state, 'X_train', None).shape if hasattr(st.session_state, 'X_train') else 'N/A',
                'X_val_shape': getattr(st.session_state, 'X_val', None).shape if hasattr(st.session_state, 'X_val') else 'N/A',
                'model_type': st.session_state.model_config.get("model_type", "N/A"),
                'error_message': str(e)
            }
            
            logger.error(f"Debug info: {debug_info}")
            
            return None, {
                'success': False, 
                'error': f"Erreur lors de l'entraînement: {str(e)}",
                'debug_info': debug_info
            }
    
    def handle_training_success(self, model, history, results_placeholder):
        """Gère le succès de l'entraînement avec sauvegarde du preprocessor"""
        
        # SAUVEGARDE CRITIQUE: tous les éléments nécessaires pour l'évaluation
        st.session_state.trained_model = model
        st.session_state.training_history = history
        
        st.session_state.training_results = {
            "model": model,
            "history": history,
            "training_config": st.session_state.training_config,
            "model_config": st.session_state.model_config,
            "preprocessing_config": st.session_state.preprocessing_config,
            "imbalance_config": st.session_state.imbalance_config,
            "preprocessor": st.session_state.preprocessor,  # ← ELEMENT CRITIQUE
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Debug: vérifier que le preprocessor est bien sauvegardé
        logger.info(f"Preprocessor sauvegardé dans training_results: {st.session_state.preprocessor is not None}")
        
        with results_placeholder.container():
            st.success("✅ Entraînement terminé avec succès!")
            
            # Afficher des informations de debug
            with st.expander("🔍 Informations de Debug"):
                st.write("**Preprocessor sauvegardé:**", st.session_state.preprocessor is not None)
                if st.session_state.preprocessor:
                    st.write("**Config preprocessor:**", st.session_state.preprocessor.get_config())
                st.write("**Shape des données d'entrée:**", history.get('input_shape', 'N/A'))
            
            self.display_training_results(history)
            
    def handle_training_failure(self, history, results_placeholder):
        """Gère l'échec de l'entraînement"""
        with results_placeholder.container():
            st.error("❌ L'entraînement a échoué")
            if "error" in history:
                st.error(f"Erreur: {history['error']}")
            
            # Logs de débogage
            with st.expander("🔍 Détails de l'erreur"):
                st.json(history)
            
            if st.button("🔙 Retour à la configuration", use_container_width=True):
                st.session_state.current_step = 4
                st.rerun()
    
    def handle_training_error(self, error, results_placeholder):
        """Gère les erreurs pendant l'entraînement"""
        with results_placeholder.container():
            st.error(f"❌ Erreur lors de l'entraînement: {str(error)}")
            self.logger.error(f"Training error: {error}", exc_info=True)
            
            # Affichage de l'erreur complète pour débogage
            with st.expander("🔍 Stack trace complète"):
                st.code(str(error))
                import traceback
                st.code(traceback.format_exc())
    
    def display_training_results(self, history):
        """Affiche les résultats détaillés de l'entraînement"""
        # Métriques principales
        col_result1, col_result2, col_result3, col_result4 = st.columns(4)
        
        with col_result1:
            st.metric("Meilleure Loss Val", f"{history.get('best_val_loss', 0):.4f}")
        
        with col_result2:
            st.metric("Époques Effectuées", history.get('total_epochs_trained', 0))
        
        with col_result3:
            st.metric("Temps Total", f"{history.get('training_time', 0):.1f}s")
        
        with col_result4:
            early_stopped = "✅ Oui" if history.get('early_stopping_triggered', False) else "❌ Non"
            st.metric("Early Stopping", early_stopped)
        
        # Graphiques des courbes d'entraînement
        st.markdown("### 📊 Courbes d'Entraînement")
        
        if history.get('train_loss') and history.get('val_loss'):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=history['train_loss'],
                mode='lines',
                name='Train Loss',
                line=dict(color='#667eea', width=2)
            ))
            fig.add_trace(go.Scatter(
                y=history['val_loss'],
                mode='lines',
                name='Val Loss',
                line=dict(color='#764ba2', width=2)
            ))
            fig.update_layout(
                title="Loss au fil des Époques",
                xaxis_title="Époque",
                yaxis_title="Loss",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Métriques supplémentaires si disponibles
        if history.get('val_accuracy'):
            col_met1, col_met2 = st.columns(2)
            
            with col_met1:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(
                    y=history['val_accuracy'],
                    mode='lines+markers',
                    name='Accuracy',
                    line=dict(color='#2ecc71', width=2)
                ))
                fig_acc.update_layout(
                    title="Accuracy Validation",
                    xaxis_title="Époque",
                    yaxis_title="Accuracy",
                    template="plotly_white"
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col_met2:
                if history.get('val_f1'):
                    fig_f1 = go.Figure()
                    fig_f1.add_trace(go.Scatter(
                        y=history['val_f1'],
                        mode='lines+markers',
                        name='F1 Score',
                        line=dict(color='#e74c3c', width=2)
                    ))
                    fig_f1.update_layout(
                        title="F1 Score Validation",
                        xaxis_title="Époque",
                        yaxis_title="F1 Score",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_f1, use_container_width=True)
        
        # Actions post-entraînement
        st.markdown("---")
        col_action1, col_action2 = st.columns(2)
        
        with col_action1:
            if st.button("📊 Aller à l'Évaluation", type="primary", use_container_width=True):
                st.switch_page("pages/5_anomaly_evaluation.py")
        
        with col_action2:
            if st.button("🔄 Nouvel Entraînement", use_container_width=True):
                # Réinitialisation partielle pour un nouvel entraînement
                st.session_state.current_step = 0
                st.session_state.workflow_complete = False
                st.session_state.trained_model = None
                st.session_state.training_results = None
                st.rerun()
    
    def main(self):
        """Point d'entrée principal de l'application"""
        self.render_header()
        self.render_workflow_progress()
        
        # Routage des étapes
        if st.session_state.current_step == 0:
            self.render_data_analysis_step()
        elif st.session_state.current_step == 1:
            self.render_imbalance_analysis_step()
        elif st.session_state.current_step == 2:
            self.render_preprocessing_step()
        elif st.session_state.current_step == 3:
            self.render_model_selection_step()
        elif st.session_state.current_step == 4:
            self.render_training_config_step()
        elif st.session_state.current_step == 5:
            self.render_training_launch_step()
        
        # Footer avec informations supplémentaires
        self.render_footer()
    
    def render_footer(self):
        """Affiche le footer avec des informations utiles"""
        st.markdown("---")
        
        with st.expander("ℹ️ Informations sur la Session"):
            st.markdown("### État de la Configuration")
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown("**Données:**")
                if 'X' in st.session_state:
                    st.write(f"- Images totales: {len(st.session_state.X):,}")
                    st.write(f"- Classes: {len(np.unique(st.session_state.y))}")
                
                if 'X_train' in st.session_state:
                    st.write(f"- Train: {len(st.session_state.X_train):,}")
                    st.write(f"- Validation: {len(st.session_state.X_val):,}")
                    st.write(f"- Test: {len(st.session_state.X_test):,}")
            
            with col_info2:
                st.markdown("**Configuration:**")
                st.write(f"- Étape actuelle: {st.session_state.current_step + 1}/6")
                
                if st.session_state.selected_model_type:
                    st.write(f"- Modèle: {st.session_state.selected_model_type}")
                
                if st.session_state.training_config:
                    epochs = st.session_state.training_config.epochs if hasattr(st.session_state.training_config, 'epochs') else st.session_state.training_config.get('epochs', 'N/A')
                    batch_size = st.session_state.training_config.batch_size if hasattr(st.session_state.training_config, 'batch_size') else st.session_state.training_config.get('batch_size', 'N/A')
                    st.write(f"- Époques: {epochs}")
                    st.write(f"- Batch size: {batch_size}")
        
        # Navigation globale
        st.markdown("---")
        col_nav1, col_nav2, col_nav3 = st.columns(3)
        
        with col_nav1:
            if st.button("🏠 Retour au Dashboard", use_container_width=True):
                st.switch_page("pages/1_dashboard.py")
        
        with col_nav2:
            if st.button("🔄 Réinitialiser le Workflow", use_container_width=True):
                # Réinitialisation complète
                for key in ['current_step', 'selected_model_type', 'model_config', 'training_config']:
                    if key in st.session_state:
                        st.session_state[key] = self.initialize_session_state.__defaults__[0].get(key, None)
                st.session_state.current_step = 0
                st.rerun()
        
        with col_nav3:
            if 'trained_model' in st.session_state and st.session_state.trained_model is not None:
                if st.button("📊 Évaluation des Résultats", type="primary", use_container_width=True):
                    st.switch_page("pages/5_anomaly_evaluation.py")

# Lancement de l'application
if __name__ == "__main__":
    app = MLTrainingWorkflowPro()
    app.main()