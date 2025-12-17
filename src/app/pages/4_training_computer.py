"""
🚀 ML FACTORY PRO - Training Computer Vision 
Architecture propre avec séparation UI/logique métier
Support supervisé + non-supervisé unifié
"""

import streamlit as st
import numpy as np
import torch
import time
import plotly.graph_objects as go
from typing import Dict, Any, List
from collections import Counter

from src.shared.logging import get_logger

# === IMPORTS COMPOSANTS UI ===
from monitoring.state_managers import init, AppPage, STATE
from ui.training_vision import (
    inject_training_vision_css,
    detect_training_mode,
    perform_stratified_split,
    validate_split_quality,
    render_mode_badge,
    render_split_distribution_chart,
    render_split_stats_table,
    render_validation_warnings,
    filter_models_by_mode,
    analyze_imbalance_by_mode,
    render_imbalance_analysis
)

# === IMPORTS LOGIQUE MÉTIER ===
from src.models.computer_vision_training import (
    TrainingConfig,
    ModelType,
    OptimizerType,
    SchedulerType,
    DataAugmenter,
)
from orchestrators.visio_training_orchestrator import (
    training_orchestrator,
    TrainingContext
)
from utils.callbacks import LoggingCallback, StreamlitCallback
from sklearn.utils.class_weight import compute_class_weight

logger = get_logger(__name__)

# Configuration Streamlit
st.set_page_config(
    page_title="ML Factory Pro | Training CV",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation STATE
init()


class MLTrainingWorkflowPro:
    """
    Workflow professionnel refactorisé
    Support complet supervisé + non-supervisé
    """
    
    def __init__(self):
        self.logger = logger
        inject_training_vision_css()
    
    def render_header(self):
        """Header avec détection mode automatique"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown('<div class="main-header">🚀 ML Factory Pro</div>', unsafe_allow_html=True)
            st.markdown("**Workflow Intelligent Computer Vision**")
        
        with col2:
            st.metric("Étape", f"{STATE.current_step + 1}/6")
            
        with col3:
            device = "CUDA 🚀" if torch.cuda.is_available() else "CPU ⚡"
            st.caption(f"Device: {device}")
            
            # Affichage mode si détecté
            if STATE.loaded and STATE.data.y is not None:
                mode, _ = detect_training_mode(STATE.data.y)
                badge_color = "#4facfe" if mode == "supervised" else "#f5576c"
                st.markdown(
                    f"<div style='background:{badge_color};color:white;padding:0.3rem;border-radius:5px;text-align:center;font-size:0.7rem;'>"
                    f"{'🎯 SUPERVISÉ' if mode == 'supervised' else '🔍 ANOMALIES'}"
                    f"</div>",
                    unsafe_allow_html=True
                )
    
    def render_workflow_progress(self):
        """Barre progression"""
        steps = [
            ("📊", "Données", "Split et Analyse"),
            ("⚖️", "Déséquilibre", "Analyse et Correction"),
            ("🎨", "Prétraitement", "Normalisation"),
            ("🤖", "Modèle", "Architecture"),
            ("⚙️", "Entraînement", "Hyperparamètres"),
            ("🚀", "Lancement", "Monitoring")
        ]
        
        st.markdown("### 📋 Progression du Workflow")
        
        cols = st.columns(len(steps))
        for idx, (col, (icon, name, desc)) in enumerate(zip(cols, steps)):
            with col:
                if idx < STATE.current_step:
                    status = ("✅", "#28a745", "Terminé")
                elif idx == STATE.current_step:
                    status = ("🔵", "#667eea", "En cours")
                else:
                    status = ("⚪", "#6c757d", "À venir")
                
                st.markdown(
                    f"""<div style="text-align:center;padding:1rem;border-radius:10px;
                    background:{'#f8f9ff' if idx == STATE.current_step else 'white'};
                    border:2px solid {status[1]};">
                    <div style="font-size:1.5rem;margin-bottom:0.5rem;">{icon}</div>
                    <div style="font-weight:bold;color:{status[1]};">{name}</div>
                    <div style="font-size:0.8rem;color:#666;">{desc}</div>
                    <div style="font-size:0.7rem;color:{status[1]};margin-top:0.5rem;">
                    {status[0]} {status[2]}</div></div>""",
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
    
    # ========================================================================
    # ÉTAPE 1: SPLIT AVEC DÉTECTION MODE
    # ========================================================================
    
    def render_data_analysis_step(self):
        """Étape 1 refactorisée avec détection mode"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("📊 Étape 1: Analyse et Split des Données")
        
        # Vérification chargement
        if not STATE.loaded or STATE.data.X is None:
            st.error("❌ Aucun dataset chargé")
            st.info("Chargez un dataset depuis le dashboard")
            if st.button("📊 Dashboard", type="primary"):
                st.switch_page("pages/1_dashboard.py")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        X, y = STATE.data.X, STATE.data.y
        
        # === DÉTECTION AUTOMATIQUE DU MODE ===
        try:
            if hasattr(STATE.data, 'y_train') and STATE.data.y_train is not None:
                mode, metadata = detect_training_mode(STATE.data.y_train)
            else:
                # Fallback si pas encore splitté
                mode, metadata = detect_training_mode(STATE.data.y)
            
            logger.info(f"Mode détecté: {mode} | Metadata: {metadata}")
        except ValueError as e:
            st.error(f"❌ {e}")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Badge mode
        col_mode1, col_mode2 = st.columns([1, 2])
        with col_mode1:
            render_mode_badge(mode, metadata)
        
        with col_mode2:
            st.info(f"""
            **Caractéristiques Détectées:**
            - **Images totales:** {len(X):,}
            - **Classes:** {metadata['n_classes']}
            - **Tâche:** {metadata['task'].replace('_', ' ').title()}
            """)
        
        st.markdown("---")
        
        # === CONFIGURATION SPLIT ===
        st.subheader("🔧 Configuration du Split")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider(
                "Taille Test Set (%)",
                10, 40, 20, 5,
                help="Pourcentage réservé au test final"
            )
        
        with col2:
            val_size = st.slider(
                "Taille Validation Set (%)",
                10, 30, 20, 5,
                help="Pourcentage du train_val pour validation"
            )
        
        # Calcul tailles
        test_ratio = test_size / 100
        val_ratio = val_size / 100
        
        n_test = int(len(X) * test_ratio)
        n_train_val = len(X) - n_test
        n_val = int(n_train_val * val_ratio)
        n_train = n_train_val - n_val
        
        # Métriques
        col_met1, col_met2, col_met3 = st.columns(3)
        with col_met1:
            st.metric("🏋️ Training", f"{n_train:,}")
        with col_met2:
            st.metric("📊 Validation", f"{n_val:,}")
        with col_met3:
            st.metric("🧪 Test", f"{n_test:,}")
        
        # === BOUTON SPLIT ===
        st.markdown("---")
        if st.button("🔄 Effectuer le Split", type="primary", use_container_width=True):
            with st.spinner("Split en cours..."):
                try:
                    # Split avec fonction helper
                    split_result = perform_stratified_split(
                        X, y,
                        test_size=test_ratio,
                        val_size=val_ratio,
                        mode=mode
                    )
                    
                    # Validation
                    is_valid, warnings = validate_split_quality(split_result, mode, metadata)
                    
                    if not is_valid:
                        st.error("❌ Split invalide")
                        render_validation_warnings(warnings)
                        st.markdown('</div>', unsafe_allow_html=True)
                        return
                    
                    # Sauvegarde STATE
                    STATE.data.X_train = split_result["X_train"]
                    STATE.data.X_val = split_result["X_val"]
                    STATE.data.X_test = split_result["X_test"]
                    STATE.data.y_train = split_result["y_train"]
                    STATE.data.y_val = split_result["y_val"]
                    STATE.data.y_test = split_result["y_test"]
                    
                    STATE.data.split_config = split_result["split_info"]
                    STATE.data.split_config["mode"] = mode
                    STATE.data.split_config["metadata"] = metadata
                    
                    # Visualisation
                    st.success("✅ Split effectué avec succès")
                    render_split_distribution_chart(split_result, mode)
                    render_split_stats_table(split_result, mode, metadata)
                    render_validation_warnings(warnings)
                    
                    st.balloons()
                    STATE.current_step = 1
                    st.rerun()
                
                except Exception as e:
                    logger.error(f"Erreur split: {e}", exc_info=True)
                    st.error(f"❌ Erreur: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # ÉTAPE 2: DÉSÉQUILIBRE ADAPTATIF
    # ========================================================================
    
    def render_imbalance_analysis_step(self):
        """Étape 2 refactorisée avec logique par mode"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("⚖️ Étape 2: Gestion du Déséquilibre")
        
        if not STATE.loaded or STATE.data.y_train is None:
            st.error("❌ Données d'entraînement manquantes")
            if st.button("⬅️ Retour Étape 1"):
                STATE.current_step = 0
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        y_train = STATE.data.y_train
        
        # Récupération mode
        split_config = getattr(STATE.data, 'split_config', {})
        mode = split_config.get('mode', 'supervised')
        metadata = split_config.get('metadata', {})
        
        # Badge rappel mode
        render_mode_badge(mode, metadata)
        st.markdown("---")
        
        # === ANALYSE DÉSÉQUILIBRE ===
        imbalance_info = analyze_imbalance_by_mode(y_train, mode, metadata)
        
        render_imbalance_analysis(imbalance_info, y_train)
        
        st.markdown("---")
        
        # === OPTIONS CORRECTION (CONDITIONNELLES) ===
        st.subheader("🎯 Stratégies de Correction")
        
        if mode == "supervised":
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚖️ Poids de Classe")
                use_weights = st.checkbox(
                    "Activer poids automatiques",
                    value=imbalance_info["use_class_weights"],
                    help="Ajuste loss function selon déséquilibre"
                )
                
                if use_weights:
                    classes = np.unique(y_train)
                    weights = compute_class_weight('balanced', classes=classes, y=y_train)
                    weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
                    
                    st.info("**Poids calculés:**")
                    for cls, weight in weight_dict.items():
                        st.write(f"- Classe {cls}: `{weight:.3f}`")
                    
                    STATE.class_weights = weight_dict
            
            with col2:
                st.markdown("#### 🎭 SMOTE")
                use_smote = st.checkbox(
                    "Activer SMOTE",
                    value=imbalance_info["use_smote"],
                    disabled=not imbalance_info["use_smote"],
                    help="Génère échantillons synthétiques classes minoritaires"
                )
        
        else:  # unsupervised
            st.info("""
            **ℹ️ Mode Détection d'Anomalies**
            
            Les autoencoders apprennent à reconstruire uniquement les images **normales**.
            Le déséquilibre normal/anomalie est **attendu et souhaité**.
            
            ⚠️ **Class weights désactivés** (contre-productif pour autoencoders)
            """)
            use_weights = False
            use_smote = False
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour"):
                STATE.current_step = 0
                st.rerun()
        
        with col_nav2:
            if st.button("💾 Continuer ➡️", type="primary"):
                # Sauvegarde config
                STATE.imbalance_config = {
                    "use_class_weights": use_weights,
                    "use_smote": use_smote,
                    "imbalance_ratio": float(imbalance_info["ratio"]),
                    "mode": mode,
                    "metadata": metadata
                }
                
                # Propager aux configs training si nécessaire
                if not hasattr(STATE, 'training_config') or STATE.training_config is None:
                    STATE.training_config = {}
                
                if isinstance(STATE.training_config, dict):
                    STATE.training_config['use_class_weights'] = use_weights
                
                st.success("✅ Configuration sauvegardée")
                STATE.current_step = 2
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # ÉTAPE 3: PRÉTRAITEMENT
    # ========================================================================
    
    def render_preprocessing_step(self):
        """Étape 3 - Prétraitement"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🎨 Étape 3: Prétraitement des Images")
        
        split_config = getattr(STATE.data, 'split_config', {})
        mode = split_config.get('mode', 'supervised')
        
        st.markdown("**Configuration du pipeline de prétraitement**")
        
        # Normalisation
        st.subheader("🔧 Normalisation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            normalization = st.selectbox(
                "Méthode",
                ["standardize", "normalize", "none"],
                index=0,
                help="standardize: (x-mean)/std | normalize: [0,1] | none: aucune"
            )
        
        with col2:
            # AFFICHAGE taille actuelle
            if STATE.data.X is not None:
                sample_shape = STATE.data.X.shape
                
                # Détection format
                if sample_shape[-1] in [1, 3, 4]:  # channels_last
                    current_h, current_w = sample_shape[1], sample_shape[2]
                else:  # channels_first
                    current_h, current_w = sample_shape[2], sample_shape[3]
                
                st.info(f"📏 Taille actuelle: {current_h}×{current_w}")
            
            # Parser correctement le resize
            resize_options = ["Conserver", "128×128", "224×224", "256×256"]
            resize_choice = st.selectbox(
                "Redimensionnement",
                resize_options,
                index=0,
                help="Redimensionner toutes les images à une taille fixe"
            )
        
        st.markdown("---")
        
        # Augmentation
        st.subheader("🎭 Augmentation de Données")
        
        if not hasattr(STATE, 'preprocessing_config') or STATE.preprocessing_config is None:
            STATE.preprocessing_config = {}
        
        augmentation_enabled = st.checkbox(
            "Activer augmentation",
            value=STATE.preprocessing_config.get("augmentation_enabled", False)
        )
        
        methods = []
        augmentation_factor = 1
        
        if augmentation_enabled:
            col_aug1, col_aug2 = st.columns(2)
            
            with col_aug1:
                augmentation_factor = st.slider(
                    "Facteur multiplication",
                    1, 5,
                    STATE.preprocessing_config.get("augmentation_factor", 2)
                )
            
            with col_aug2:
                st.markdown("**Techniques:**")
                if st.checkbox("Flip horizontal", value=True):
                    methods.append('flip')
                if st.checkbox("Rotation ±15°", value=True):
                    methods.append('rotate')
                if st.checkbox("Zoom aléatoire", value=False):
                    methods.append('zoom')
                if st.checkbox("Luminosité", value=False):
                    methods.append('brightness')
            
            # Warning si mode anomalie
            if mode == "unsupervised":
                st.warning("⚠️ Mode anomalies: augmentation appliquée uniquement sur images normales")
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour"):
                STATE.current_step = 1
                st.rerun()
        
        with col_nav2:
            if st.button("💾 Continuer ➡️", type="primary"):

                # Sauvegarde config
                target_size = None
                if resize_choice != "Conserver":

                    # Extraction "224×224" → (224, 224)
                    size_str = resize_choice.replace("×", "x")  # Normalisation
                    try:
                        h_str, w_str = size_str.split("x")
                        target_size = (int(h_str), int(w_str))
                        logger.info(f"✅ Resize activé: target_size={target_size}")
                    except Exception as e:
                        logger.error(f"❌ Erreur parsing resize '{resize_choice}': {e}")
                        st.error(f"Format resize invalide: {resize_choice}")
                        return
                
                # SAUVEGARDE avec target_size
                STATE.preprocessing_config = {
                    "strategy": normalization,
                    "target_size": target_size,  
                    "augmentation_enabled": augmentation_enabled,
                    "augmentation_factor": augmentation_factor,
                    "methods": methods
                }
                
                st.success("✅ Configuration sauvegardée")
                STATE.current_step = 3
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # ÉTAPE 4: SÉLECTION MODÈLE AVEC FILTRAGE
    # ========================================================================
    
    def render_model_selection_step(self):
        """Étape 4 avec filtrage modèles par mode"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🤖 Étape 4: Sélection du Modèle")
        
        # Récupération mode
        split_config = getattr(STATE.data, 'split_config', {})
        mode = split_config.get('mode', 'supervised')
        metadata = split_config.get('metadata', {})
        
        # Rappel mode
        col_mode, _ = st.columns([1, 2])
        with col_mode:
            render_mode_badge(mode, metadata)
        
        st.markdown("---")
        
        # Catalogue complet
        all_models = self.get_model_categories()
        
        # === FILTRAGE PAR MODE ===
        available_models = filter_models_by_mode(all_models, mode, metadata)
        
        if not available_models:
            st.error(f"❌ Aucun modèle compatible avec mode {mode}")
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        n_models = sum(len(cat['models']) for cat in available_models.values())
        st.info(f"**{n_models} modèles** disponibles pour mode **{mode}**")
        
        st.markdown("### 🎯 Modèles Disponibles")
        
        # Affichage par catégorie
        for category, category_data in available_models.items():
            with st.expander(f"{category} ({len(category_data['models'])} modèles)", expanded=True):
                st.markdown(f"*{category_data['description']}*")
                
                # Grille 2 colonnes
                model_cols = st.columns(2)
                
                for idx, model in enumerate(category_data["models"]):
                    col = model_cols[idx % 2]
                    
                    with col:
                        is_selected = STATE.selected_model_type == model["id"]
                        
                        card_class = "model-card selected" if is_selected else "model-card"
                        
                        st.markdown(
                            f"""<div class="{card_class}">
                            <div style="display:flex;align-items:start;margin-bottom:1rem;">
                                <span style="font-size:2rem;margin-right:1rem;">{model['icon']}</span>
                                <div style="flex:1;">
                                    <h4 style="margin:0 0 0.5rem 0;">{model['name']}</h4>
                                    <span class="status-badge badge-info">{model['complexity']}</span>
                                </div>
                            </div>
                            <p style="color:#666;font-size:0.9rem;">{model['description']}</p>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        
                        if st.button(
                            "✅ Sélectionné" if is_selected else "📝 Sélectionner",
                            key=f"select_{model['id']}",
                            use_container_width=True,
                            type="primary" if is_selected else "secondary"
                        ):
                            STATE.selected_model_type = model["id"]
                            STATE.model_config = {
                                "model_type": model["id"],
                                "model_params": self.get_default_model_params(model["id"])
                            }
                            st.success(f"✅ {model['name']} sélectionné")
                            st.rerun()
        
        # Config avancée si modèle sélectionné
        if STATE.selected_model_type:
            st.markdown("---")
            st.subheader(f"⚙️ Configuration - {STATE.selected_model_type.upper()}")
            self.render_model_specific_parameters()
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour"):
                STATE.current_step = 2
                st.rerun()
        
        with col_nav2:
            if st.button("💾 Continuer ➡️", type="primary"):
                if STATE.selected_model_type:
                    STATE.current_step = 4
                    st.rerun()
                else:
                    st.error("❌ Sélectionnez un modèle")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def get_model_categories(self):
        """Catalogue complet des modèles"""
        return {
            "🎯 Classification Supervisée": {
                "color": "#28a745",
                "description": "Modèles pour classification avec labels",
                "models": [
                    {
                        "id": "simple_cnn",
                        "name": "CNN Simple",
                        "description": "Réseau basique - Idéal prototypage",
                        "icon": "🖼️",
                        "complexity": "Débutant"
                    },
                    {
                        "id": "custom_resnet",
                        "name": "ResNet Personnalisé",
                        "description": "Architecture résiduelle profonde",
                        "icon": "🏗️",
                        "complexity": "Intermédiaire"
                    },
                    {
                        "id": "transfer_learning",
                        "name": "Transfer Learning",
                        "description": "Modèles pré-entraînés ImageNet",
                        "icon": "🔄",
                        "complexity": "Avancé"
                    }
                ]
            },
            "🔍 Détection d'Anomalies": {
                "color": "#dc3545",
                "description": "Modèles pour anomalies sans/avec peu labels",
                "models": [
                    {
                        "id": "conv_autoencoder",
                        "name": "AutoEncodeur Convolutif",
                        "description": "Reconstruit images normales",
                        "icon": "🎭",
                        "complexity": "Intermédiaire"
                    },
                    {
                        "id": "variational_autoencoder",
                        "name": "VAE (Variational)",
                        "description": "Modèle génératif probabiliste",
                        "icon": "🌌",
                        "complexity": "Avancé"
                    },
                    {
                        "id": "denoising_autoencoder",
                        "name": "AutoEncodeur Denoiseur",
                        "description": "Robuste au bruit",
                        "icon": "🧹",
                        "complexity": "Intermédiaire"
                    },
                    {
                        "id": "patch_core",
                        "name": "PatchCore",
                        "description": "State-of-the-art défauts locaux",
                        "icon": "🧩",
                        "complexity": "Expert"
                    }
                ]
            }
        }
    
    def get_default_model_params(self, model_type: str):
        """Paramètres par défaut"""
        defaults = {
            "simple_cnn": {
                "input_channels": 3,
                "num_classes": 2,
                "base_filters": 32,
                "dropout_rate": 0.5
            },
            "custom_resnet": {
                "input_channels": 3,
                "num_classes": 2,
                "base_filters": 64,
                "dropout_rate": 0.3
            },
            "transfer_learning": {
                "input_channels": 3,
                "num_classes": 2,
                "backbone_name": "resnet50",
                "pretrained": True,
                "dropout_rate": 0.5
            },
            "conv_autoencoder": {
                "input_channels": 3,
                "latent_dim": 256,
                "base_filters": 32,
                "num_stages": 4
            },
            "variational_autoencoder": {
                "input_channels": 3,
                "latent_dim": 128,
                "base_filters": 32,
                "beta": 1.0
            },
            "denoising_autoencoder": {
                "input_channels": 3,
                "latent_dim": 256,
                "noise_factor": 0.1
            },
            "patch_core": {
                "backbone_name": "wide_resnet50_2",
                "patchcore_layers": ["layer2", "layer3"],
                "coreset_ratio": 0.01
            }
        }
        
        return defaults.get(model_type, {"input_channels": 3})
    

    def render_model_specific_parameters(self):
        """Paramètres spécifiques au modèle avec UI complète"""
        model_type = STATE.selected_model_type
        model_params = STATE.model_config.get("model_params", {})
        
        # === AUTOENCODERS: latent_dim + base_filters ===
        if model_type in ["conv_autoencoder", "variational_autoencoder", "denoising_autoencoder"]:
            st.markdown("#### 🔧 Configuration AutoEncoder")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # SLIDER latent_dim
                latent_dim = st.slider(
                    "Dimension Espace Latent",
                    min_value=32,
                    max_value=1024,
                    value=model_params.get("latent_dim", 128),
                    step=32,
                    help=(
                        "Taille du bottleneck (compression maximale). "
                        "Plus petit = compression forte (risque underfitting). "
                        "Plus grand = moins de compression (risque overfitting)."
                    )
                )
                
                # Indicateur qualité
                if latent_dim < 64:
                    st.warning("⚠️ Très petit - Risque underfitting")
                elif latent_dim > 512:
                    st.info("ℹ️ Grande dimension - Moins de compression")
            
            with col2:
                base_filters = st.slider(
                    "Filtres de base",
                    16, 128,
                    model_params.get("base_filters", 32),
                    16,
                    help="Nombre de filtres du premier bloc (doublés à chaque stage)"
                )
            
            # CALCUL taux compression (si données chargées)
            if hasattr(STATE.data, 'X') and STATE.data.X is not None:
                sample_shape = STATE.data.X.shape
                
                # Détection format
                if sample_shape[-1] in [1, 3, 4]:  # channels_last
                    h, w, c = sample_shape[1], sample_shape[2], sample_shape[3]
                else:  # channels_first
                    c, h, w = sample_shape[1], sample_shape[2], sample_shape[3]
                
                input_pixels = h * w * c
                compression_ratio = input_pixels / latent_dim
                
                st.info(
                    f"📊 Taux compression: **{compression_ratio:.1f}:1** "
                    f"({input_pixels:,} pixels → {latent_dim} dimensions latentes)"
                )
            
            # Mise à jour STATE
            STATE.model_config["model_params"].update({
                "latent_dim": latent_dim,
                "base_filters": base_filters
            })
        
        # === CLASSIFICATION CNN ===
        elif model_type in ["simple_cnn", "custom_resnet"]:
            col1, col2 = st.columns(2)
            
            with col1:
                base_filters = st.slider(
                    "Filtres de base",
                    16, 128,
                    model_params.get("base_filters", 32),
                    16
                )
            
            with col2:
                dropout_rate = st.slider(
                    "Dropout",
                    0.0, 0.7,
                    model_params.get("dropout_rate", 0.5),
                    0.1
                )
            
            STATE.model_config["model_params"].update({
                "base_filters": base_filters,
                "dropout_rate": dropout_rate
            })
        
        # === TRANSFER LEARNING ===
        elif model_type == "transfer_learning":
            col1, col2 = st.columns(2)
            
            with col1:
                backbone = st.selectbox(
                    "Backbone",
                    ["resnet18", "resnet50", "efficientnet_b0", "mobilenet_v2"],
                    index=1
                )
            
            with col2:
                dropout_rate = st.slider("Dropout", 0.0, 0.7, 0.5, 0.1)
            
            STATE.model_config["model_params"].update({
                "backbone_name": backbone,
                "dropout_rate": dropout_rate
            })
    
    # ========================================================================
    # ÉTAPE 5: CONFIGURATION ENTRAÎNEMENT
    # ========================================================================
    
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
                5, 200, 50, 5,
                help="Nombre de passages complets sur le dataset"
            )
        
        with col_hyper2:
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
                value=1e-4,
                format_func=lambda x: f"{x:.0e}",
                help="Taux d'apprentissage"
            )
        
        with col_hyper3:
            batch_size = st.selectbox(
                "Batch Size",
                options=[8, 16, 32, 64, 128],
                index=2,
                help="Nombre d'images par batch"
            )
        
        st.markdown("---")
        
        # Optimiseur et Scheduler
        st.subheader("🎯 Optimiseur et Scheduler")
        
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            optimizer = st.selectbox(
                "Optimiseur",
                options=["adamw", "adam", "sgd", "rmsprop"],
                index=0,
                help="AdamW recommandé pour la plupart des cas"
            )
        
        with col_opt2:
            scheduler = st.selectbox(
                "Scheduler",
                options=["reduce_on_plateau", "cosine", "step", "none"],
                index=0,
                help="ReduceLROnPlateau réduit automatiquement le LR"
            )
        
        st.markdown("---")
        
        # Early Stopping et Régularisation
        st.subheader("🛑 Early Stopping & Régularisation")
        
        col_callback1, col_callback2, col_callback3 = st.columns(3)
        
        with col_callback1:
            early_stopping_patience = st.slider(
                "Early Stopping Patience",
                3, 30, 10,
                help="Arrête l'entraînement si pas d'amélioration"
            )
        
        with col_callback2:
            reduce_lr_patience = st.slider(
                "Reduce LR Patience",
                2, 15, 5,
                help="Réduit le LR si pas d'amélioration"
            )
        
        with col_callback3:
            weight_decay = st.select_slider(
                "Weight Decay",
                options=[0.0, 0.001, 0.01, 0.1],
                value=0.01,
                help="Régularisation L2"
            )
        
        # Options avancées
        with st.expander("🔧 Options Avancées"):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                gradient_clip = st.slider(
                    "Gradient Clipping",
                    0.0, 5.0, 1.0, 0.5,
                    help="Limite l'amplitude des gradients"
                )
                
                deterministic = st.checkbox(
                    "Mode Déterministe",
                    value=True,
                    help="Rend les résultats reproductibles"
                )
            
            with col_adv2:
                use_mixed_precision = st.checkbox(
                    "Mixed Precision (FP16)",
                    value=torch.cuda.is_available(),
                    disabled=not torch.cuda.is_available(),
                    help="Accélère l'entraînement sur GPU"
                )
                
                num_workers = st.slider(
                    "DataLoader Workers",
                    0, 8, 4,
                    help="Processus pour charger les données"
                )
        
        # Navigation
        st.markdown("---")
        col_nav1, col_nav2 = st.columns(2)
        
        with col_nav1:
            if st.button("⬅️ Retour"):
                STATE.current_step = 3
                st.rerun()
        
        with col_nav2:
            if st.button("💾 Sauvegarder et Continuer ➡️", type="primary"):
                # Création de la configuration d'entraînement
                STATE.training_config = {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "gradient_clip": gradient_clip,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "early_stopping_patience": early_stopping_patience,
                    "reduce_lr_patience": reduce_lr_patience,
                    "use_class_weights": STATE.imbalance_config.get('use_class_weights', False),
                    "deterministic": deterministic,
                    "use_mixed_precision": use_mixed_precision,
                    "num_workers": num_workers,
                    "seed": 42
                }
                
                st.success("✅ Configuration d'entraînement sauvegardée")
                STATE.current_step = 5
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # ÉTAPE 6: LANCEMENT ET MONITORING
    # ========================================================================
    
    def render_training_launch_step(self):
        """Étape 6: Lancement de l'entraînement"""
        st.markdown('<div class="workflow-step-card">', unsafe_allow_html=True)
        st.header("🚀 Étape 6: Lancement de l'Entraînement")
        
        # Récapitulatif de la configuration
        st.subheader("📋 Récapitulatif de la Configuration")
        
        col_summary1, col_summary2 = st.columns(2)
        
        with col_summary1:
            st.subheader("📊 Données et Préparation")
            
            split_config = getattr(STATE.data, 'split_config', None)
            if split_config:
                st.json(split_config)
            else:
                st.info("Aucune configuration de split disponible")
            
            st.subheader("⚖️ Gestion du Déséquilibre")
            if hasattr(STATE, 'imbalance_config') and STATE.imbalance_config:
                st.json(STATE.imbalance_config)
            else:
                st.info("Aucune configuration de déséquilibre disponible")
            
            st.subheader("🎨 Prétraitement")
            if hasattr(STATE, 'preprocessing_config') and STATE.preprocessing_config:
                st.json(STATE.preprocessing_config)
            else:
                st.info("Aucune configuration de prétraitement disponible")

        with col_summary2:
            st.subheader("🤖 Modèle")
            if hasattr(STATE, 'model_config') and STATE.model_config:
                st.json(STATE.model_config)
            else:
                st.info("Aucune configuration de modèle disponible")
            
            st.subheader("⚙️ Entraînement")
            if hasattr(STATE, 'training_config') and STATE.training_config:
                st.json(STATE.training_config)
            else:
                st.info("Aucune configuration d'entraînement disponible")
        
        st.markdown("---")
        st.subheader("🔍 Validation de la Configuration")
        
        errors, warnings = self.validate_training_configuration()
        
        if errors:
            for error in errors:
                st.error(error)
        else:
            if warnings:
                for warning in warnings:
                    st.warning(warning)
            st.success("✅ Configuration valide - Prêt pour l'entraînement!")
        
        # Informations de lancement
        st.markdown("---")
        st.subheader("🎯 Informations de Lancement")

        col_launch1, col_launch2, col_launch3 = st.columns(3)

        with col_launch1:
            total_train_images = 0
            if STATE.loaded and hasattr(STATE.data, 'X_train') and STATE.data.X_train is not None:
                total_train_images = len(STATE.data.X_train)
                
                if (hasattr(STATE, 'preprocessing_config') and 
                    STATE.preprocessing_config and 
                    STATE.preprocessing_config.get("augmentation_enabled", False)):
                    augmentation_factor = STATE.preprocessing_config.get("augmentation_factor", 1)
                    total_train_images *= augmentation_factor
            
            st.metric("📷 Images Train", f"{total_train_images:,}")

        with col_launch2:
            epochs = STATE.training_config.get('epochs', 50) if isinstance(STATE.training_config, dict) else getattr(STATE.training_config, 'epochs', 50)
            batch_size = STATE.training_config.get('batch_size', 32) if isinstance(STATE.training_config, dict) else getattr(STATE.training_config, 'batch_size', 32)
            
            estimated_minutes = 1
            if batch_size > 0 and total_train_images > 0:
                images_per_minute = 1200 if torch.cuda.is_available() else 200
                estimated_minutes = max(1, int((total_train_images * epochs) / (batch_size * images_per_minute)))
                
            st.metric("⏱️ Temps estimé", f"{estimated_minutes} min")

        with col_launch3:
            use_weights = STATE.imbalance_config.get("use_class_weights", False) if hasattr(STATE, 'imbalance_config') and STATE.imbalance_config else False
            st.metric("⚖️ Poids de classe", "Activés" if use_weights else "Désactivés")

        # Informations système
        st.markdown("---")
        st.subheader("💻 Informations Système")

        col_sys1, col_sys2, col_sys3 = st.columns(3)

        with col_sys1:
            device = "CUDA 🚀" if torch.cuda.is_available() else "CPU ⚡"
            st.info(f"**Device:** {device}")
            if torch.cuda.is_available():
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    st.success(f"**GPU:** {gpu_name} ({gpu_memory:.1f} GB)")
                except Exception:
                    st.warning("**GPU:** Informations non disponibles")

        with col_sys2:
            mixed_precision = STATE.training_config.get('use_mixed_precision', False) if isinstance(STATE.training_config, dict) else getattr(STATE.training_config, 'use_mixed_precision', False)
            st.info(f"**Mixed Precision:** {'Activée 🚀' if mixed_precision else 'Désactivée'}")

        with col_sys3:
            deterministic = STATE.training_config.get('deterministic', True) if isinstance(STATE.training_config, dict) else getattr(STATE.training_config, 'deterministic', True)
            st.info(f"**Mode Déterministe:** {'Activé ✅' if deterministic else 'Désactivé'}")

        st.markdown("---")

        # Bouton de lancement
        launch_disabled = len(errors) > 0 if 'errors' in locals() else True
        
        if st.button(
            "🚀 Démarrer l'Entraînement", 
            type="primary", 
            use_container_width=True, 
            disabled=launch_disabled
        ):
            self.launch_training()

        # Navigation
        st.markdown("---")
        col_back, _ = st.columns(2)
        with col_back:
            if st.button("⬅️ Retour", use_container_width=True):
                STATE.current_step = 4
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    
    def validate_training_configuration(self):
        """Valide la configuration complète avant lancement"""
        errors = []
        warnings = []
        
        # Vérification des données
        required_data_attrs = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
        
        for attr in required_data_attrs:
            if not hasattr(STATE.data, attr):
                errors.append(f"❌ Attribut manquant dans STATE.data: {attr}")
            elif getattr(STATE.data, attr, None) is None:
                errors.append(f"❌ Données manquantes: {attr} est None")
        
        # Vérification des configurations essentielles
        if not hasattr(STATE, 'model_config') or not STATE.model_config:
            errors.append("❌ Configuration du modèle manquante ou vide")
        
        if not hasattr(STATE, 'training_config') or not STATE.training_config:
            errors.append("❌ Configuration d'entraînement manquante ou vide")
        
        # Vérifications des hyperparamètres
        if hasattr(STATE, 'training_config') and STATE.training_config:
            if isinstance(STATE.training_config, dict):
                epochs = STATE.training_config.get('epochs', 50)
                batch_size = STATE.training_config.get('batch_size', 32)
            else:
                epochs = getattr(STATE.training_config, 'epochs', 50)
                batch_size = getattr(STATE.training_config, 'batch_size', 32)
            
            if epochs > 100:
                warnings.append("⚠️ Nombre d'époques élevé (>100)")
            
            if batch_size > 64 and not torch.cuda.is_available():
                warnings.append("⚠️ Batch size élevé (>64) sans GPU")
        
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
                # Configuration des callbacks
                streamlit_components = {
                    "progress_bar": progress_bar,
                    "status_text": status_text,
                    "metrics_placeholder": metrics_placeholder
                }
                
                # Détermination du type d'anomalie
                model_type = STATE.model_config["model_type"]
                anomaly_type = None
                if model_type in ["conv_autoencoder", "variational_autoencoder", "denoising_autoencoder", "patch_core"]:
                    anomaly_type = "structural"
                
                # Passer split_config au contexte
                context_metadata = {
                    "dataset_name": getattr(STATE.data, 'name', 'unknown'),
                    "user_id": "anonymous"
                }
                
                # Ajout split_config si disponible
                if hasattr(STATE.data, 'split_config') and STATE.data.split_config:
                    context_metadata['split_config'] = STATE.data.split_config
                    logger.info(f"✅ split_config ajouté au contexte: {STATE.data.split_config}")
                else:
                    logger.warning("⚠️ split_config absent, mode sera déduit d'anomaly_type")
                
                # Lancement de l'entraînement
                model, history = self.train_with_metier_logic(
                    streamlit_components, 
                    anomaly_type,
                    context_metadata  # Passage metadata enrichi
                )
                # Gestion des résultats
                if model is not None and history and history.get("success", True):
                    self.handle_training_success(model, history, results_placeholder)
                else:
                    self.handle_training_failure(history, results_placeholder)
                
            except Exception as e:
                self.handle_training_error(e, results_placeholder)

    def train_with_metier_logic(self, streamlit_components, anomaly_type, context_metadata):
        """Interface vers l'orchestrateur d'entraînement"""
        try:
            # Création du contexte d'entraînement
            context = TrainingContext(
                X_train=STATE.data.X_train,
                y_train=STATE.data.y_train,
                X_val=STATE.data.X_val, 
                y_val=STATE.data.y_val,
                model_config=STATE.model_config,
                training_config=STATE.training_config,
                preprocessing_config=STATE.preprocessing_config,
                callbacks=self._create_callbacks(streamlit_components),
                anomaly_type=anomaly_type,
                metadata=context_metadata 
            )
            
            # Ajouter split_config directement au contexte
            if 'split_config' in context_metadata:
                context.split_config = context_metadata['split_config']
                logger.info("✅ split_config propagé au TrainingContext")
            
            # Délégation à l'orchestrateur
            result = training_orchestrator.train(context)
            
            if result.success:
                STATE.preprocessor = result.preprocessor
                return result.model, result.history
            else:
                return None, {'success': False, 'error': result.error}
                
        except Exception as e:
            logger.error(f"Erreur interface training: {e}", exc_info=True)
            return None, {'success': False, 'error': str(e)}

    def _create_callbacks(self, streamlit_components):
        """Crée les callbacks Streamlit"""
        callbacks = []
        
        if streamlit_components:
            callbacks.append(StreamlitCallback(
                progress_bar=streamlit_components.get('progress_bar'),
                status_text=streamlit_components.get('status_text'),
                total_epochs=STATE.training_config.get('epochs', 50) if isinstance(STATE.training_config, dict) else getattr(STATE.training_config, 'epochs', 50)
            ))
        
        callbacks.append(LoggingCallback(log_every_n_epochs=5))
        
        return callbacks
    
    def handle_training_success(self, model, history, results_placeholder):
        """Gère le succès de l'entraînement"""
        # Sauvegarde dans STATE
        STATE.trained_model = model
        STATE.training_history = history
        
        preprocessor = getattr(STATE, 'preprocessor', None)
        
        STATE.training_results = {
            "model": model,
            "history": history,
            "training_config": getattr(STATE, 'training_config', {}),
            "model_config": getattr(STATE, 'model_config', {}),
            "preprocessing_config": getattr(STATE, 'preprocessing_config', {}),
            "imbalance_config": getattr(STATE, 'imbalance_config', {}),
            "preprocessor": preprocessor,
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info("✅ Training completed successfully")
        
        with results_placeholder.container():
            st.success("✅ Entraînement terminé avec succès!")
            
            # Debug optionnel
            if st.checkbox("🔍 Afficher debug", value=False, key="show_debug"):
                with st.expander("📋 Informations Techniques"):
                    st.write("**Preprocessor:**", preprocessor is not None)
                    st.write("**Input Shape:**", history.get('input_shape', 'N/A'))
            
            self.display_training_results(history)
            
    def handle_training_failure(self, history, results_placeholder):
        """Gère l'échec de l'entraînement"""
        with results_placeholder.container():
            st.error("❌ L'entraînement a échoué")
            if "error" in history:
                st.error(f"Erreur: {history['error']}")
            
            with st.expander("🔍 Détails de l'erreur"):
                st.json(history)
            
            if st.button("🔙 Retour à la configuration", use_container_width=True):
                STATE.current_step = 4
                st.rerun()
    
    def handle_training_error(self, error, results_placeholder):
        """Gère les erreurs pendant l'entraînement"""
        with results_placeholder.container():
            st.error(f"❌ Erreur lors de l'entraînement: {str(error)}")
            self.logger.error(f"Training error: {error}", exc_info=True)
            
            with st.expander("🔍 Stack trace complète"):
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
        
        # Actions post-entraînement
        st.markdown("---")
        col_action1, col_action2 = st.columns(2)
        
        with col_action1:
            if st.button("📊 Aller à l'Évaluation", type="primary", use_container_width=True):
                st.switch_page("pages/5_anomaly_evaluation.py")
        
        with col_action2:
            if st.button("🔄 Nouvel Entraînement", use_container_width=True):
                # Réinitialisation partielle
                STATE.current_step = 0
                STATE.workflow_complete = False
                STATE.trained_model = None
                STATE.training_results = None
                st.rerun()
    
    def main(self):
        """Point d'entrée principal"""
        self.render_header()
        self.render_workflow_progress()
        
        # Routage des étapes
        if STATE.current_step == 0:
            self.render_data_analysis_step()
        elif STATE.current_step == 1:
            self.render_imbalance_analysis_step()
        elif STATE.current_step == 2:
            self.render_preprocessing_step()
        elif STATE.current_step == 3:
            self.render_model_selection_step()
        elif STATE.current_step == 4:
            self.render_training_config_step()
        elif STATE.current_step == 5:
            self.render_training_launch_step()
        
        # Footer
        self.render_footer()
    
    def render_footer(self):
        """Affiche le footer avec des informations utiles"""
        st.markdown("---")
        
        with st.expander("ℹ️ Informations sur la Session"):
            st.markdown("### État de la Configuration")
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.markdown("**Données:**")
                if STATE.loaded and STATE.data.X is not None:
                    st.write(f"- Images totales: {len(STATE.data.X):,}")
                    st.write(f"- Classes: {len(np.unique(STATE.data.y))}")

                if STATE.loaded and STATE.data.X_train is not None:
                    st.write(f"- Train: {len(STATE.data.X_train):,}")
                    st.write(f"- Validation: {len(STATE.data.X_val):,}")
                    st.write(f"- Test: {len(STATE.data.X_test):,}")
            
            with col_info2:
                st.markdown("**Configuration:**")
                st.write(f"- Étape actuelle: {STATE.current_step + 1}/6")
                
                if STATE.selected_model_type:
                    st.write(f"- Modèle: {STATE.selected_model_type}")
                
                if hasattr(STATE, 'training_config') and STATE.training_config:
                    if isinstance(STATE.training_config, dict):
                        epochs = STATE.training_config.get('epochs', 'N/A')
                        batch_size = STATE.training_config.get('batch_size', 'N/A')
                    else:
                        epochs = getattr(STATE.training_config, 'epochs', 'N/A')
                        batch_size = getattr(STATE.training_config, 'batch_size', 'N/A')
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
                STATE.current_step = 0
                STATE.selected_model_type = None
                STATE.model_config = None
                STATE.training_config = None
                st.rerun()
        
        with col_nav3:
            if hasattr(STATE, 'trained_model') and STATE.trained_model is not None:
                if st.button("📊 Évaluation des Résultats", type="primary", use_container_width=True):
                    st.switch_page("pages/5_anomaly_evaluation.py")


# Lancement de l'application
if __name__ == "__main__":
    app = MLTrainingWorkflowPro()
    app.main()