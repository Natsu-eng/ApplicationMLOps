"""
Application principale Streamlit pour DataLab Pro.
Version optimisée pour la production avec gestion robuste des erreurs et monitoring.
"""
import pkg_resources
import sys
import os
# Ajout de la racine du projet à sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import streamlit as st
import pandas as pd
import logging
import warnings
import time
import psutil
from src.data.data_loader import load_data
from src.shared.logging import setup_logging, get_logger
from typing import Dict, Any
import gc

# Import des constantes ET de la navigation
from src.config.constants import ANOMALY_CONFIG, APP_CONSTANTS, TRAINING_CONSTANTS
from helpers.navigation_manager import NavigationManager

# Configuration du logger
logger = get_logger(__name__)

def _get_production_css():
    """CSS pour masquer les éléments Streamlit en production."""
    return """
    <style>
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    .stAlert > div  {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    .main > div {
        padding-top: 1rem;
    }
    </style>
    """

# --- Configuration Production ---
def setup_production_environment():
    """Configuration pour l'environnement de production."""
    warnings.filterwarnings("ignore", category=FutureWarning, module='numpy')
    warnings.filterwarnings("ignore", category=UserWarning, module='streamlit')
    
    try:
        import mlflow
        logger.info("MLflow is available for tracking experiments")
    except ImportError:
        logger.warning("MLflow not installed, experiment tracking disabled") 
    
    setup_logging(mlflow_integration=True)
    
    if 'production_setup_done' not in st.session_state:
        st.session_state.production_setup_done = True
        if os.getenv('STREAMLIT_ENV') == 'production':
            st.markdown(_get_production_css(), unsafe_allow_html=True)

# --- Fonctions de Monitoring ---
def get_system_metrics() -> Dict[str, Any]:
    """Récupère les métriques système actuelles."""
    try:
        memory = psutil.virtual_memory()
        return {
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'timestamp': time.time()
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {'memory_percent': 0, 'memory_available_mb': 0, 'timestamp': time.time()}

def check_system_health():
    """Vérifie la santé du système et affiche des alertes si nécessaire."""
    metrics = get_system_metrics()
    if metrics['memory_percent'] > TRAINING_CONSTANTS["HIGH_MEMORY_THRESHOLD"]:
        st.warning(f"⚠️ Utilisation mémoire élevée: {metrics['memory_percent']:.1f}%")
        logger.warning(f"High memory usage detected: {metrics['memory_percent']:.1f}%")
        if metrics['memory_percent'] > 90:
            if st.button("🧹 Nettoyer la mémoire", help="Libère la mémoire et vide les caches"):
                cleanup_memory()
                st.success("Nettoyage mémoire effectué")
                st.rerun()

def cleanup_memory():
    """Nettoyage mémoire robuste avec logs."""
    try:
        collected = gc.collect()
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
        if hasattr(st, 'cache_resource'):
            st.cache_resource.clear()
        for key in list(st.session_state.keys()):
            if key.startswith("_") or key in ["df", "df_raw", "X", "y", "data_dir"]:
                continue
            if isinstance(st.session_state[key], (pd.DataFrame, dict, list)):
                del st.session_state[key]
        logger.info(f"Memory cleanup: {collected} objects collected")
        return collected
    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}", exc_info=True)
        return 0

# --- Fonctions de Gestion d'État ---
def initialize_session():
    """Initialise l'état de base de la session de façon robuste."""
    required_keys = {
        'df': None,
        'df_raw': None,
        'uploaded_file_name': None,
        'target_column_for_ml_config': None,
        'task_type': APP_CONSTANTS["DEFAULT_TASK_TYPE"],
        'config': None,
        'model_name': None,
        'model_params': {},
        'preprocessing': {},
        'n_splits': APP_CONSTANTS["DEFAULT_N_SPLITS"],
        'model': None,
        'metrics_summary': None,
        'preprocessor': None,
        'ml_results': [],
        'last_system_check': 0,
        'error_count': 0,
        # ✅ NOUVEAUX KEYS POUR NAVIGATION
        'data_type': 'none',  # 'tabular', 'images', 'none'
        'X': None,  # Pour données images
        'y': None,  # Pour labels images
        'data_dir': None,  # Pour datasets images
        'dataset_structure': None,  # Structure détectée
        'dataset_info': None,  # Infos dataset
        'current_page': 'main.py',  # Page actuelle pour navigation
        'dashboard_version': 1,  # Version pour cache
        'dataset_hash': '',  # Hash pour détection changements
        'column_types': None,  # Types de colonnes détectés
        'selected_univar_col': None,  # Sélection univariée
        'selected_bivar_col1': None,  # Sélection bivariée
        'selected_bivar_col2': None,  # Sélection bivariée
        'useless_candidates': [],  # Colonnes inutiles détectées
        'rename_list': []  # Liste de renommage
    }
    for key, default_value in required_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    current_time = time.time()
    if current_time - st.session_state.last_system_check > 300:
        check_system_health()
        st.session_state.last_system_check = current_time

def reset_app_state():
    """Réinitialise toutes les variables de session liées à un jeu de données."""
    logger.info("Réinitialisation de l'état de l'application pour un nouveau fichier")
    try:
        old_error_count = st.session_state.get('error_count', 0)
        old_current_page = st.session_state.get('current_page', 'main.py')
        
        reset_keys = [
            'df', 'df_raw', 'uploaded_file_name', 'target_column_for_ml_config',
            'task_type', 'config', 'model_name', 'model_params', 'preprocessing',
            'n_splits', 'model', 'metrics_summary', 'preprocessor', 'ml_results',
            # ✅ RÉINITIALISATION DES DONNÉES IMAGES AUSSI
            'X', 'y', 'data_dir', 'dataset_structure', 'dataset_info', 'data_type',
            'dashboard_version', 'dataset_hash', 'column_types', 'selected_univar_col',
            'selected_bivar_col1', 'selected_bivar_col2', 'useless_candidates', 'rename_list'
        ]
        for key in reset_keys:
            if key in st.session_state:
                del st.session_state[key]
        
        # Réinitialisation avec valeurs par défaut
        st.session_state.update({
            'df': None,
            'df_raw': None,
            'uploaded_file_name': None,
            'target_column_for_ml_config': None,
            'task_type': APP_CONSTANTS["DEFAULT_TASK_TYPE"],
            'config': None,
            'model_name': None,
            'model_params': {},
            'preprocessing': {},
            'n_splits': APP_CONSTANTS["DEFAULT_N_SPLITS"],
            'model': None,
            'metrics_summary': None,
            'preprocessor': None,
            'ml_results': [],
            'data_type': 'none',
            'X': None,
            'y': None,
            'data_dir': None,
            'dataset_structure': None,
            'dataset_info': None,
            'current_page': old_current_page,
            'dashboard_version': 1,
            'dataset_hash': '',
            'column_types': None,
            'selected_univar_col': None,
            'selected_bivar_col1': None,
            'selected_bivar_col2': None,
            'useless_candidates': [],
            'rename_list': [],
            'error_count': old_error_count
        })
        
        cleanup_memory()
        logger.info("État de l'application réinitialisé avec succès")
        st.toast("Application réinitialisée pour le nouveau fichier", icon="🔄")
    except Exception as e:
        logger.error(f"Erreur lors de la réinitialisation : {e}")
        st.error(f"Erreur lors de la réinitialisation : {e}")

def validate_session_state() -> bool:
    """Valide l'intégrité de l'état de la session."""
    try:
        # Vérification des données tabulaires
        if 'df' in st.session_state and st.session_state.df is not None:
            df = st.session_state.df
            if not hasattr(df, 'columns') or len(df.columns) == 0:
                logger.warning("DataFrame in session_state is corrupted")
                return False
        
        # Vérification des données images
        if 'X' in st.session_state and st.session_state.X is not None:
            X = st.session_state.X
            if len(X) == 0:
                logger.warning("Image data in session_state is corrupted")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Session state validation failed: {e}")
        return False

# --- Initialisation de l'application ---
st.set_page_config(
    page_title="DataLab Pro | Accueil",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed"
)

setup_production_environment()

try:
    initialize_session()
    if not validate_session_state():
        logger.warning("Invalid session state detected, resetting...")
        reset_app_state()
except Exception as e:
    logger.error(f"Session initialization failed: {e}")
    st.error("Erreur d'initialisation de la session. Veuillez recharger la page.")
    st.stop()

# ✅ MISE À JOUR DE LA PAGE COURANTE POUR LA NAVIGATION
st.session_state.current_page = "main.py"

# Header avec informations système
col_title, col_system = st.columns([3, 1])
with col_title:
    st.title("🧪 DataLab Pro")
    st.markdown("Plateforme d'analyse de données et de Machine Learning automatisé")
with col_system:
    metrics = get_system_metrics()
    if metrics['memory_percent'] > 0:
        color = "🔴" if metrics['memory_percent'] > TRAINING_CONSTANTS["HIGH_MEMORY_THRESHOLD"] else "🟡" if metrics['memory_percent'] > 70 else "🟢"
        st.caption(f"{color} RAM: {metrics['memory_percent']:.0f}%")

st.markdown("---")

# Section principale de chargement
st.header("📂 Importation des données")

with st.expander("ℹ️ Formats supportés et limites", expanded=False):
    st.markdown(f"""
    **Formats acceptés :** {', '.join(APP_CONSTANTS["SUPPORTED_EXTENSIONS"]).upper()}
    
    **Limites :**
    - Taille maximale : {APP_CONSTANTS["MAX_FILE_SIZE_MB"]:,} MB
    - Automatiquement optimisé selon la taille (Pandas ≤ 100MB, Dask > 100MB)
    - Validation d'intégrité avant chargement
    
    **Fonctionnalités automatiques :**
    - Détection et suppression des doublons
    - Conversion intelligente des types de données
    - Optimisation mémoire pour les gros datasets
    """)

uploaded_file = st.file_uploader(
    "Choisissez votre fichier de données",
    type=list(APP_CONSTANTS["SUPPORTED_EXTENSIONS"]),
    key="file_uploader",
    help=f"Formats supportés: {', '.join(APP_CONSTANTS['SUPPORTED_EXTENSIONS']).upper()} • Maximum {APP_CONSTANTS['MAX_FILE_SIZE_MB']}MB"
)

if uploaded_file is not None:
    try:
        file_size_mb = uploaded_file.size / (1024 * 1024) if hasattr(uploaded_file, 'size') else 0
        if file_size_mb > APP_CONSTANTS["MAX_FILE_SIZE_MB"]:
            st.error(f"❌ Fichier trop volumineux: {file_size_mb:.1f}MB > {APP_CONSTANTS['MAX_FILE_SIZE_MB']}MB")
            logger.error(f"File too large: {file_size_mb:.1f}MB")
            st.stop()
        
        if st.session_state.uploaded_file_name != uploaded_file.name:
            logger.info(f"New file detected: {uploaded_file.name}")
            reset_app_state()
            
            progress_container = st.container()
            with progress_container:
                st.info(f"📥 Chargement de **{uploaded_file.name}** ({file_size_mb:.1f}MB)...")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                progress_bar.progress(20)
                status_text.text("Validation du fichier...")
                time.sleep(0.5)
                
                progress_bar.progress(40)
                status_text.text("Chargement des données...")
                
                try:
                    df, report, df_raw = load_data(
                        file_path=uploaded_file,
                        blocksize="64MB",
                        sanitize_for_display=True
                    )
                    progress_bar.progress(80)
                    status_text.text("Finalisation...")
                except Exception as load_error:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Erreur lors du chargement: {str(load_error)}")
                    logger.error(f"Data loading failed: {load_error}")
                    st.session_state.error_count += 1
                    st.stop()
                
                progress_bar.progress(100)
                status_text.text("Terminé!")
                time.sleep(0.5)
                progress_container.empty()
            
            if df is not None:
                st.session_state.df = df
                st.session_state.df_raw = df_raw
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.data_type = "tabular"
                logger.info(f"File loaded successfully: {uploaded_file.name}")
                
                if report and report.get("actions"):
                    st.success("✅ Fichier chargé avec succès!")
                    with st.expander("📋 Rapport de chargement", expanded=False):
                        for action in report["actions"]:
                            st.write(f"• {action}")
                        if report.get("changes"):
                            st.subheader("🔧 Conversions de types automatiques")
                            changes_df = pd.DataFrame([
                                {"Colonne": col, "Conversion": change}
                                for col, change in report["changes"].items()
                            ])
                            st.dataframe(changes_df, use_container_width=True)
                        if report.get("warnings"):
                            st.subheader("⚠️ Avertissements")
                            for warning in report["warnings"]:
                                st.warning(warning)
                
                st.rerun()
            else:
                error_messages = report.get("actions", ["Erreur inconnue"]) if report else ["Erreur inconnue"]
                st.error(f"❌ Échec du chargement: {error_messages[0]}")
                logger.error(f"Data loading failed: {error_messages[0]}")
                st.session_state.error_count += 1
                st.markdown("""
                **Suggestions pour résoudre le problème:**
                - Vérifiez le format du fichier
                - Assurez-vous que le fichier n'est pas corrompu
                - Essayez avec un fichier plus petit
                - Vérifiez l'encodage (UTF-8 recommandé)
                """)
    except Exception as e:
        st.error(f"❌ Erreur inattendue: {str(e)}")
        logger.error(f"Unexpected error during file processing: {e}", exc_info=True)
        st.session_state.error_count += 1

# Section pour le dataset MVTec AD - VERSION CORRIGÉE
st.header("📷 Dataset MVTec AD - Détection d'Anomalies")

with st.expander("ℹ️ À propos du dataset MVTec AD", expanded=False):
    st.markdown("""
    **Dataset MVTec AD** : Benchmark industriel pour la détection d'anomalies visuelles
    
    **📁 Structure attendue :**
    ```
    votre_dataset/
    ├── train/
    │   └── good/          # Images normales pour l'entraînement
    │       ├── image1.png
    │       └── image2.png
    └── test/
        ├── good/          # Images normales pour le test
        └── defect_type/   # Images avec défauts spécifiques
    ```
    
    **⚙️ Configuration automatique :**
    - Redimensionnement : 256×256 pixels
    - Normalisation : Standard ImageNet
    - Format : PNG/JPG (RGB)
    - Augmentation optionnelle disponible
    """)

# Configuration en deux colonnes
col_config, col_info = st.columns([2, 1])

with col_config:
    dataset_option = st.radio(
        "Mode de chargement",
        options=["📂 Charger depuis un dossier local", "🔄 Utiliser un dataset exemple"],
        help="Choisissez comment charger vos données d'images"
    )
    
    if dataset_option == "📂 Charger depuis un dossier local":
        st.subheader("Emplacement du dataset")
        
        # Chemins par défaut intelligents
        default_paths = [
            os.path.join(project_root, "data", "mvtec_ad"),
            os.path.join(project_root, "src", "data", "mvtec_ad"),
            os.path.join(os.path.expanduser("~"), "Downloads", "mvtec_ad")
        ]
        
        existing_path = None
        for path in default_paths:
            if os.path.exists(path):
                existing_path = path
                break
        
        data_dir = st.text_input(
            "📁 Chemin absolu du dossier MVTec AD",
            value=existing_path or default_paths[0],
            placeholder=f"ex: {default_paths[0]}",
            help="Chemin complet vers le dossier racine de votre dataset MVTec AD"
        )
        
        # Validation en temps réel
        if data_dir:
            from src.data.image_processing import detect_dataset_structure
            
            structure = detect_dataset_structure(data_dir)
            
            if structure["type"] == "invalid":
                st.error("❌ Dossier introuvable - Vérifiez le chemin")
            elif structure["type"] != "mvtec_ad":
                st.warning(f"⚠️ Structure '{structure['type']}' détectée (MVTec AD attendu)")
                st.info("Le dataset sera chargé avec la structure détectée")
            else:
                st.success("✅ Structure MVTec AD détectée")
                    
    else:  # Dataset exemple
        st.subheader("Dataset d'exemple")
        example_datasets = {
            "bottle": "Bouteilles industrielles",
            "cable": "Câbles électriques", 
            "capsule": "Capsules médicaments",
            "metal_nut": "Écrous métalliques"
        }
        
        selected_example = st.selectbox(
            "Choisissez une catégorie d'exemple",
            options=list(example_datasets.keys()),
            format_func=lambda x: f"{x} - {example_datasets[x]}",
            help="Dataset MVTec AD de démonstration"
        )
        
        example_path = os.path.join(project_root, "src", "data", "mvtec_ad", selected_example)
        data_dir = example_path
        
        st.info(f"**Dataset sélectionné :** {example_datasets[selected_example]}")
        
        if not os.path.exists(example_path):
            st.warning(f"⚠️ Dataset exemple '{selected_example}' non disponible")
            st.markdown("""
            **📥 Téléchargement des données d'exemple :**
            1. Visitez [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
            2. Téléchargez la catégorie souhaitée
            3. Extrayez dans : `data/mvtec_ad/`
            """)

with col_info:
    st.subheader("📊 Informations")
    
    if 'data_dir' in locals() and data_dir and os.path.exists(data_dir):
        try:
            from src.data.image_processing import get_dataset_info
            
            info = get_dataset_info(data_dir)
            
            if "total" in info:
                st.metric("📷 Images totales", f"{info['total']:,}")
            
            if "normal" in info and "anomaly" in info:
                st.metric("🟢 Normales", f"{info['normal']:,}")
                st.metric("🔴 Anomalies", f"{info['anomaly']:,}")
            
            with st.expander("📁 Structure détaillée", expanded=False):
                st.json(info)
            
        except Exception as e:
            st.info("ℹ️ Analyse de la structure en attente...")
            logger.error(f"Info display error: {e}")

# Bouton de chargement unique
if 'data_dir' in locals() and data_dir and os.path.exists(data_dir):
    st.markdown("---")
    
    col_load, col_status = st.columns([1, 2])
    
    with col_load:
        load_button = st.button(
            "🚀 Charger le Dataset", 
            type="primary",
            key="load_mvtec_dataset",
            help="Prépare le dataset pour l'analyse et l'entraînement"
        )
    
    with col_status:
        current_dataset = st.session_state.get("data_dir")
        if current_dataset == data_dir:
            st.success("✅ Dataset déjà chargé et prêt")
        elif current_dataset:
            st.info("ℹ️ Un dataset différent est actuellement chargé")

    if load_button:
        try:
            with st.spinner("🔍 Validation et chargement du dataset..."):
                from src.data.image_processing import (
                    detect_dataset_structure,
                    load_images_flexible,
                    get_dataset_info
                )
                
                # === ÉTAPE 1 : Validation structure ===
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Validation de la structure...")
                structure = detect_dataset_structure(data_dir)
                
                if structure["type"] == "invalid":
                    st.error(f"❌ Structure invalide: {structure.get('error', 'Erreur inconnue')}")
                    st.stop()
                
                progress_bar.progress(20)
                
                # === ÉTAPE 2 : Chargement des images ===
                status_text.text("📥 Chargement des images...")
                
                try:
                    X, y = load_images_flexible(
                        data_dir,
                        target_size=(256, 256)
                    )
                    
                    if len(X) == 0:
                        st.error("❌ Aucune image trouvée dans le dataset")
                        st.stop()
                    
                    progress_bar.progress(60)
                    
                except Exception as load_error:
                    st.error(f"❌ Erreur chargement images: {str(load_error)}")
                    logger.error(f"Image loading failed: {load_error}", exc_info=True)
                    st.stop()
                
                # === ÉTAPE 3 : Normalisation ===
                status_text.text("⚙️ Normalisation des images...")
                
                # Normaliser si nécessaire
                if X.max() > 1.0:
                    X_normalized = X / 255.0
                else:
                    X_normalized = X.copy()
                
                progress_bar.progress(80)
                
                # === ÉTAPE 4 : Mise en session ===
                status_text.text("💾 Sauvegarde en session...")
                
                # Réinitialisation propre
                reset_app_state()
                
                # Calcul des infos
                info = get_dataset_info(data_dir)
                
                st.session_state.update({
                    "X": X,
                    "X_normalized": X_normalized,
                    "y": y,
                    "data_dir": data_dir,
                    "data_type": "images",
                    "task_type": "anomaly_detection" if structure["type"] == "mvtec_ad" else "classification",
                    "dataset_structure": structure,
                    "dataset_info": info,
                    "dataset_loaded_at": time.time(),
                    "image_count": len(X),
                    "image_shape": X.shape,
                    "n_classes": len(np.unique(y))
                })
                
                progress_bar.progress(100)
                status_text.text("✅ Terminé!")
                
                logger.info(f"Dataset loaded: {data_dir} | Images: {len(X)} | Classes: {len(np.unique(y))}")
                
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                # === AFFICHAGE RÉSUMÉ ===
                st.success(f"✅ Dataset chargé avec succès!")
                
                col_summary1, col_summary2, col_summary3 = st.columns(3)
                
                with col_summary1:
                    st.metric("📷 Images", f"{len(X):,}")
                
                with col_summary2:
                    st.metric("📐 Dimensions", f"{X.shape[1]}×{X.shape[2]}")
                
                with col_summary3:
                    st.metric("🎯 Classes", len(np.unique(y)))
                
                # Info additionnelle
                with st.expander("📋 Détails du chargement", expanded=False):
                    st.write(f"**Structure détectée:** {structure['type']}")
                    st.write(f"**Type de tâche:** {st.session_state.task_type}")
                    st.write(f"**Shape complète:** {X.shape}")
                    st.write(f"**Plage valeurs:** [{X.min():.2f}, {X.max():.2f}]")
                    st.write(f"**Mémoire:** {X.nbytes / (1024**2):.1f} MB")
                
                # Redirection automatique
                st.info("🎯 Redirection vers le Dashboard...")
                time.sleep(1.5)
                st.switch_page("pages/1_dashboard.py")
                
        except Exception as e:
            error_msg = f"Erreur lors du chargement: {str(e)[:200]}"
            st.error(f"❌ {error_msg}")
            logger.error(f"MVTec dataset loading failed: {error_msg}", exc_info=True)
            st.session_state.error_count = st.session_state.get('error_count', 0) + 1
            
            # Afficher les détails pour debug
            with st.expander("🔧 Détails de l'erreur (debug)", expanded=False):
                st.code(str(e))

# Section d'aide contextuelle
if not st.session_state.get("data_dir") and not st.session_state.get("X"):
    st.markdown("---")
    with st.expander("🆘 Guide de démarrage rapide", expanded=False):
        st.markdown("""
        **Pour utiliser la détection d'anomalies :**
        
        1. **📥 Téléchargez MVTec AD** depuis [le site officiel](https://www.mvtec.com/company/research/datasets/mvtec-ad)
        2. **📁 Organisez vos données** selon la structure MVTec AD
        3. **🚀 Chargez le dataset** via l'interface ci-dessus
        4. **🔍 Explorez** dans le Dashboard
        5. **🤖 Entraînez** vos modèles dans l'onglet ML
        
        **📚 Catégories disponibles :**
        - **bottle, cable, capsule** - Objets manufacturés
        - **metal_nut, pill, screw** - Composants industriels  
        - **carpet, leather, tile** - Textures et surfaces
        - **grid, transistor, wood** - Structures complexes
        
        **💡 Formats supportés :**
        - Structure MVTec AD (train/test avec good/défauts)
        - Dossiers catégoriels (un dossier = une classe)
        - Dossier plat (toutes images mélangées)
        """)

# --- Affichage de l'état actuel ---
if st.session_state.df is not None or st.session_state.X is not None:
    try:
        if st.session_state.data_type == "images":
            # Affichage pour données images
            X = st.session_state.X
            st.success(f"✅ Dataset **{st.session_state.data_dir}** prêt pour l'analyse")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📷 Images", f"{len(X):,}")
            with col2:
                st.metric("📐 Dimensions", f"{X.shape[1]}×{X.shape[2]}")
            with col3:
                st.metric("🎯 Classes", f"{st.session_state.n_classes}")
            with col4:
                memory_mb = X.nbytes / (1024**2)
                st.metric("💾 Mémoire", f"{memory_mb:.1f} MB")
            
            st.subheader("🎯 Prochaines étapes")
            st.info("Utilisez la barre latérale pour naviguer vers le Dashboard d'analyse d'images")
            
        else:
            # Affichage pour données tabulaires
            df = st.session_state.df
            st.success(f"✅ Dataset **{st.session_state.uploaded_file_name}** prêt pour l'analyse")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                n_rows = len(df) if not hasattr(df, 'npartitions') else "Dask"
                st.metric("Lignes", f"{n_rows:,}" if isinstance(n_rows, int) else n_rows)
            with col2:
                st.metric("Colonnes", f"{len(df.columns)}")
            with col3:
                if not hasattr(df, 'npartitions'):
                    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                    st.metric("Mémoire", f"{memory_mb:.1f} MB")
                else:
                    st.metric("Partitions", f"{df.npartitions}")
            with col4:
                df_type = "Dask" if hasattr(df, 'npartitions') else "Pandas"
                st.metric("Type", df_type)
            
            st.subheader("Aperçu des données")
            try:
                preview_rows = min(100, len(df) if not hasattr(df, 'npartitions') else 100)
                if hasattr(df, 'npartitions'):
                    df_preview = df.head(preview_rows).compute()
                else:
                    df_preview = df.head(preview_rows)
                st.dataframe(df_preview, use_container_width=True, height=300)
                if len(df_preview) == preview_rows:
                    st.caption(f"Affichage des {preview_rows} premières lignes")
            except Exception as preview_error:
                st.warning(f"⚠️ Erreur d'aperçu: {preview_error}")
                logger.error(f"Preview error: {preview_error}")
                try:
                    df_fallback = df.head(50).astype(str)
                    if hasattr(df_fallback, 'compute'):
                        df_fallback = df_fallback.compute()
                    st.dataframe(df_fallback, use_container_width=True)
                    st.caption("Aperçu avec conversion forcée en texte")
                except:
                    st.error("Impossible d'afficher l'aperçu des données")
        
        st.markdown("---")
        st.subheader("🚀 Étapes suivantes")
        col_nav1, col_nav2, col_nav3 = st.columns(3)
        with col_nav1:
            st.markdown("""
            **📊 Dashboard**
            - Vue d'ensemble des données
            - Analyse des valeurs manquantes
            - Distribution des variables
            """)
        with col_nav2:
            st.markdown("""
            **🤖 AutoML**
            - Configuration automatique
            - Entraînement de modèles
            - Évaluation des performances
            """)
        with col_nav3:
            st.markdown("""
            **📈 Résultats**
            - Métriques détaillées
            - Visualisations
            - Export des modèles
            """)
        st.info("💡 Utilisez la barre latérale pour naviguer entre les pages")
        
    except Exception as display_error:
        st.error(f"❌ Erreur d'affichage: {display_error}")
        logger.error(f"Display error: {display_error}", exc_info=True)
        st.session_state.error_count += 1
        if st.button("🔄 Réinitialiser l'application"):
            reset_app_state()
            st.rerun()
else:
    st.info("📁 Chargez un fichier ou un dataset d'images pour commencer l'analyse des données")
    with st.expander("💡 Conseils pour de meilleurs résultats", expanded=False):
        st.markdown("""
        **Pour données tabulaires:**
        - Nettoyez vos données avant le chargement si possible
        - Utilisez des noms de colonnes clairs et sans espaces
        - Évitez les caractères spéciaux dans les noms de colonnes
        
        **Pour données images:**
        - Structurez selon le format MVTec AD
        - Images en format PNG/JPG recommandé
        - Taille minimale recommandée: 128×128 pixels
        
        **Performance:**
        - Les fichiers > 100MB utiliseront automatiquement Dask
        - Format Parquet recommandé pour les gros volumes
        - CSV avec séparateurs standards (virgule, point-virgule)
        """)

# Footer avec informations de debug et actions utiles
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    if st.session_state.get('error_count', 0) > 0:
        st.caption(f"⚠️ Erreurs: {st.session_state.error_count}")
    else:
        st.caption("✅ Aucune erreur")
with footer_col2:
    current_time = time.strftime("%H:%M:%S")
    st.caption(f"⏰ Session: {current_time}")
with footer_col3:
    if st.button("🧹 Nettoyer cache", help="Libère la mémoire et vide les caches"):
        cleanup_memory()
        st.success("Cache nettoyé")
        st.rerun()

if 'last_error_check' not in st.session_state:
    st.session_state.last_error_check = time.time()

if time.time() - st.session_state.last_error_check > 600:
    if st.session_state.get('error_count', 0) > 10:
        st.warning("⚠️ Plusieurs erreurs détectées. Considérez recharger l'application.")
        if st.button("🔄 Recharger l'application"):
            st.session_state.clear()
            st.rerun()
    st.session_state.last_error_check = time.time()

# ✅ NETTOYAGE FINAL
gc.collect()