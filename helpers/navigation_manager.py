"""
Gestionnaire de navigation dynamique pour adapter l'interface selon le type de données.
À placer dans helpers/navigation_manager.py
"""

import streamlit as st
from typing import Dict, List, Optional
from src.shared.logging import get_logger

logger = get_logger(__name__)


class NavigationManager:
    """Gère la navigation adaptative selon le type de données chargées."""
    
    # Définition des pages disponibles
    PAGES = {
        "tabular": {
            "1_dashboard.py": {"icon": "📊", "name": "Dashboard", "required": ["df"]},
            "2_training.py": {"icon": "🤖", "name": "AutoML", "required": ["df"]},
            "3_evaluation.py": {"icon": "📈", "name": "Résultats", "required": ["model"]},
        },
        "images": {
            "1_dashboard.py": {"icon": "📊", "name": "Dashboard", "required": ["X", "y"]},
            "4_training_computer.py": {"icon": "🚀", "name": "Entraînement CV", "required": ["X", "y"]},
            "5_anomaly_evaluation.py": {"icon": "📊", "name": "Évaluation", "required": ["trained_model"]},
        }
    }
    
    @staticmethod
    def get_data_type() -> str:
        """
        Détecte le type de données chargées.
        
        Returns:
            "tabular", "images", ou "none"
        """
        has_tabular = 'df' in st.session_state and st.session_state.df is not None
        has_images = all(
            key in st.session_state and st.session_state[key] is not None 
            for key in ["X", "y"]
        )
        
        if has_images and has_tabular:
            logger.warning("Both data types loaded, prioritizing images")
            return "images"
        elif has_images:
            return "images"
        elif has_tabular:
            return "tabular"
        else:
            return "none"
    
    @staticmethod
    def check_requirements(requirements: List[str]) -> bool:
        """
        Vérifie si toutes les exigences sont satisfaites.
        
        Args:
            requirements: Liste des clés session_state requises
            
        Returns:
            True si toutes les exigences sont présentes
        """
        return all(
            key in st.session_state and st.session_state[key] is not None
            for key in requirements
        )
    
    @classmethod
    def render_sidebar(cls):
        """Affiche la sidebar avec navigation dynamique selon le type de données."""
        data_type = cls.get_data_type()
        
        if data_type == "none":
            with st.sidebar:
                st.warning("⚠️ Aucune donnée chargée")
                st.info("Chargez un dataset depuis la page d'accueil")
                if st.button("🏠 Aller à l'accueil", use_container_width=True):
                    st.switch_page("main.py")
            return
        
        # Récupération des pages appropriées
        pages = cls.PAGES.get(data_type, {})
        
        with st.sidebar:
            st.title("🧭 Navigation")
            
            # Indicateur du type de données
            data_type_emoji = "📷" if data_type == "images" else "📊"
            data_type_name = "Computer Vision" if data_type == "images" else "Données Tabulaires"
            st.info(f"{data_type_emoji} **Mode:** {data_type_name}")
            
            st.markdown("---")
            
            # Affichage des pages
            st.subheader("📑 Pages Disponibles")
            
            for page_file, page_info in pages.items():
                icon = page_info["icon"]
                name = page_info["name"]
                requirements = page_info["required"]
                
                # Vérifier si la page est accessible
                is_accessible = cls.check_requirements(requirements)
                
                # Style du bouton
                button_type = "primary" if is_accessible else "secondary"
                
                # Créer le bouton
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    if is_accessible:
                        st.markdown(f"### {icon}")
                    else:
                        st.markdown(f"### 🔒")
                
                with col2:
                    button_label = name if is_accessible else f"{name} (Verrouillé)"
                    
                    if st.button(
                        button_label,
                        key=f"nav_{page_file}",
                        use_container_width=True,
                        disabled=not is_accessible
                    ):
                        st.switch_page(f"pages/{page_file}")
                
                # Afficher les prérequis manquants
                if not is_accessible:
                    missing = [req for req in requirements if req not in st.session_state or st.session_state[req] is None]
                    st.caption(f"⚠️ Requis: {', '.join(missing)}")
            
            st.markdown("---")
            
            # Bouton d'accueil
            if st.button("🏠 Accueil", key="nav_home", use_container_width=True):
                st.switch_page("main.py")
            
            # Informations sur le dataset
            st.markdown("---")
            cls._render_dataset_info(data_type)
    
    @staticmethod
    def _render_dataset_info(data_type: str):
        """Affiche les informations sur le dataset chargé."""
        st.subheader("📋 Informations Dataset")
        
        if data_type == "images":
            X = st.session_state.get("X")
            y = st.session_state.get("y")
            
            if X is not None and y is not None:
                st.metric("📷 Images", f"{len(X):,}")
                st.metric("📐 Dimensions", f"{X.shape[1]}×{X.shape[2]}")
                st.metric("🎯 Classes", len(set(y)))
                
                memory_mb = X.nbytes / (1024**2)
                st.metric("💾 Mémoire", f"{memory_mb:.1f} MB")
                
                # Chemin du dataset
                data_dir = st.session_state.get("data_dir", "N/A")
                if data_dir != "N/A":
                    st.caption(f"📁 {data_dir.split('/')[-1]}")
        
        else:  # tabular
            df = st.session_state.get("df")
            
            if df is not None:
                from src.data.data_analysis import compute_if_dask, is_dask_dataframe
                
                n_rows = compute_if_dask(df.shape[0]) if hasattr(df, 'shape') else len(df)
                n_cols = df.shape[1] if hasattr(df, 'shape') else 0
                
                st.metric("📊 Lignes", f"{n_rows:,}")
                st.metric("📋 Colonnes", f"{n_cols}")
                
                if not is_dask_dataframe(df):
                    memory_mb = compute_if_dask(df.memory_usage(deep=True).sum()) / (1024**2)
                    st.metric("💾 Mémoire", f"{memory_mb:.1f} MB")
                else:
                    st.metric("💾 Partitions", f"{df.npartitions}")
    
    @staticmethod
    def show_workflow_progress():
        """Affiche la progression du workflow selon le type de données."""
        data_type = NavigationManager.get_data_type()
        
        if data_type == "none":
            return
        
        st.markdown("---")
        st.subheader("📈 Progression du Workflow")
        
        if data_type == "images":
            # Workflow Computer Vision
            steps = {
                "1. Chargement": "X" in st.session_state and st.session_state.X is not None,
                "2. Prétraitement": "preprocessing_config" in st.session_state and bool(st.session_state.preprocessing_config),
                "3. Entraînement": "trained_model" in st.session_state and st.session_state.trained_model is not None,
                "4. Évaluation": "evaluation_metrics" in st.session_state and bool(st.session_state.evaluation_metrics),
            }
        else:
            # Workflow Tabulaire
            steps = {
                "1. Chargement": "df" in st.session_state and st.session_state.df is not None,
                "2. Configuration": "config" in st.session_state and st.session_state.config is not None,
                "3. Entraînement": "model" in st.session_state and st.session_state.model is not None,
                "4. Résultats": "metrics_summary" in st.session_state and st.session_state.metrics_summary is not None,
            }
        
        # Affichage des étapes
        for step_name, is_complete in steps.items():
            icon = "✅" if is_complete else "⬜"
            color = "green" if is_complete else "gray"
            st.markdown(f"<span style='color:{color}'>{icon} {step_name}</span>", unsafe_allow_html=True)


# === FONCTION UTILITAIRE POUR INTÉGRATION FACILE ===

def setup_navigation():
    """
    Fonction à appeler au début de chaque page pour configurer la navigation.
    
    Usage dans chaque page:
    ```python
    from src.helpers.navigation_manager import setup_navigation
    
    setup_navigation()
    ```
    """
    # Vérifier si on est sur la page d'accueil
    try:
        current_page = st.session_state.get("_current_page", "main.py")
    except:
        current_page = "main.py"
    
    # Ne pas afficher la sidebar sur la page d'accueil
    if current_page == "main.py" or "main.py" in str(st.session_state.get("_main_script_path", "")):
        return
    
    # Afficher la navigation
    NavigationManager.render_sidebar()


# === DÉCORATEUR POUR PROTECTION DES PAGES ===
def require_data(*required_keys, redirect_page: str = "pages/1_dashboard.py"):
    """
    Décorateur OU fonction directe pour protéger les pages.
    """

    # --- Cas 1 : Utilisation directe ---
    if len(required_keys) > 0 and not callable(required_keys[0]):
        def decorator(func):
            def wrapper(*args, **kwargs):
                missing = [key for key in required_keys if key not in st.session_state or st.session_state[key] is None]
                if missing:
                    st.error(f"❌ Données manquantes: {', '.join(missing)}")
                    st.info("Retournez aux étapes précédentes pour charger les données nécessaires.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🏠 Accueil", use_container_width=True):
                            st.switch_page("main.py")
                    with col2:
                        if st.button("📊 Dashboard", use_container_width=True):
                            st.switch_page(redirect_page)
                    
                    st.stop()
                return func(*args, **kwargs)
            return wrapper
        return decorator

    # --- Cas 2 : Utilisation sans argument ---
    else:
        func = required_keys[0]
        def wrapper(*args, **kwargs):
            missing = [key for key in st.session_state if key not in st.session_state or st.session_state[key] is None]
            if missing:
                st.error(f"❌ Données manquantes: {', '.join(missing)}")
                st.stop()
            return func(*args, **kwargs)
        return wrapper
