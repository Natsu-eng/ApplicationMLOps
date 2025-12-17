"""
🏠 DataLab Pro - Page d'Accueil CORRIGÉE
Fix MVTec AD → y_train correctement transmis pour détection unsupervised
"""

import streamlit as st # type: ignore
import sys
import os
import numpy as np

# Configuration paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from helpers.ui_components.cards import ModernComponents
from monitoring.state_managers import AppPage, STATE
from src.data.data_loader import load_data
from src.explorations.image_exploration_plots import (
    detect_dataset_structure,
    load_images_flexible,
    get_dataset_info
)
from src.shared.logging import get_logger, PerformanceLogger

logger = get_logger(__name__)
perf_logger = PerformanceLogger("home_page_performance")


class ModernHomePage:
    """Page d'accueil avec upload MVTec AD fonctionnel"""
    
    def __init__(self):
        logger.info("🏠 Initialisation de la page d'accueil")
        self.components = ModernComponents()
        self.state = STATE
    
    def render_hero_section(self):
        """Hero section"""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 4rem 2rem;
            border-radius: 24px;
            text-align: center;
            color: white;
            margin-bottom: 3rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        ">
            <h1 style="font-size: 3.5rem; font-weight: 900; margin-bottom: 1rem;">
                DataLab Pro
            </h1>
            <p style="font-size: 1.4rem; opacity: 0.95; max-width: 700px; margin: 0 auto 2rem;">
                Plateforme tout-en-un d'analyse de données et d'IA — Tabulaire, Images et MVTec AD
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_value_proposition(self):
        """Proposition de valeur"""
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0 3rem;">
            <h2 style="font-size: 2.5rem; font-weight: 800; color: #2c3e50;">
                Une plateforme unifiée pour tous vos besoins data
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(3)
        
        with cols[0]:
            ModernComponents.feature_card(
                icon="📊",
                title="Analyse Tabulaire Avancée",
                description="AutoML, Feature Engineering, SHAP",
                features=["AutoML", "Nettoyage", "Corrélations", "Prédictions"]
            )
        
        with cols[1]:
            ModernComponents.feature_card(
                icon="🖼️",
                title="Computer Vision Complète",
                description="Classification, Anomalies (Supervised & Unsupervised)",
                features=["MVTec AD", "YOLO", "U-Net", "ViT", "Transfer Learning"]
            )
        
        with cols[2]:
            ModernComponents.feature_card(
                icon="🚀",
                title="Production Ready",
                description="MLflow, Monitoring, Déploiement",
                features=["Tracking", "Versioning", "CI/CD", "Rapports"]
            )
    
    def render_upload_section(self):
        """Section upload avec onglets"""
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0 2rem;">
            <h2 style="font-size: 2.5rem; font-weight: 800; color: #2c3e50;">
                Commencez Maintenant
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📊 Données Tabulaires", "🖼️ Données Images", "📦 Exemples MVTec"])
        
        with tab1:
            self._render_tabular_upload()
        
        with tab2:
            self._render_image_upload()
        
        with tab3:
            self._render_mvtec_examples()
    
    def _render_tabular_upload(self):
        """Upload tabulaire"""
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem;">📤</div>
            <h3>Glissez-déposez vos fichiers tabulaires</h3>
            <p>CSV, Excel, Parquet, JSON — jusqu'à 500 Mo</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choisir un fichier",
            type=['csv', 'xlsx', 'xls', 'parquet', 'feather', 'json'],
            label_visibility="collapsed",
            key="tabular_upload"
        )
        
        if uploaded_file and st.button("🚀 Analyser", type="primary", use_container_width=True):
            with st.spinner("Chargement en cours..."):
                df, report, df_raw = load_data(uploaded_file, sanitize_for_display=True)
                if df is not None and not df.empty:
                    if self.state.set_tabular(df, df_raw, uploaded_file.name):
                        st.success(f"✅ {len(df):,} lignes chargées !")
                        st.balloons()
                        if self.state.switch(AppPage.DASHBOARD):
                            st.rerun()
    
    def _render_image_upload(self):
        """
        ✅ PRODUCTION READY : Upload images avec support complet
        - Upload de dossier via chemin
        - Upload multiple de fichiers images
        - Support MVTec AD et datasets personnalisés
        """
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem;">📁</div>
            <h3>Chargement d'Images</h3>
            <p>MVTec AD, dossiers par classe, dossier plat ou fichiers multiples</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Onglets pour différentes méthodes d'upload
        upload_tab1, upload_tab2 = st.tabs(["📁 Dossier (Chemin)", "📤 Fichiers Multiples"])
        
        with upload_tab1:
            self._render_folder_upload()
        
        with upload_tab2:
            self._render_multiple_files_upload()
    
    def _render_folder_upload(self):
        """Upload via chemin de dossier"""
        col1, col2 = st.columns([4, 1])
        
        with col1:
            data_dir = st.text_input(
                "Chemin du dossier", 
                placeholder="/chemin/vers/votre/dataset ou C:\\chemin\\dataset", 
                key="image_dir_input"
            )
            st.caption("💡 Formats supportés: MVTec AD, dossiers par classe, dossier plat avec images")
        
        with col2:
            st.write("<br>", unsafe_allow_html=True)
            load_btn = st.button("📁 Charger Dossier", type="primary", use_container_width=True)
        
        if load_btn and data_dir:
            if not os.path.exists(data_dir):
                st.error("❌ Dossier introuvable. Vérifiez le chemin.")
                return
            
            if not os.path.isdir(data_dir):
                st.error("❌ Le chemin spécifié n'est pas un dossier.")
                return
            
            perf_logger.start_operation("image_loading_folder")
            with st.spinner("Analyse et chargement des images..."):
                try:
                    structure = detect_dataset_structure(data_dir)
                    
                    if structure["type"] == "invalid":
                        st.error(f"❌ Structure invalide : {structure.get('error')}")
                        st.info("💡 Formats attendus:\n- MVTec AD: train/good, test/good, test/defect\n- Par classe: dossier1/, dossier2/, etc.\n- Plat: toutes images dans un seul dossier")
                        return
                    
                    # Récupération de y_train
                    X, X_norm, y, y_train = load_images_flexible(data_dir, target_size=(256, 256))
                    
                    if len(X) == 0:
                        st.error("❌ Aucune image trouvée dans le dossier")
                        st.info("💡 Vérifiez que le dossier contient des images (jpg, png, bmp, etc.)")
                        return
                    
                    info = get_dataset_info(data_dir)
                    
                    # y_train au state manager
                    if self.state.set_images(X, X_norm, y, data_dir, structure, info, y_train=y_train):
                        
                        # Détection du mode
                        mode_icon = "🔍" if y_train is not None and len(np.unique(y_train)) == 1 else "🎯"
                        mode_label = "Unsupervised (MVTec AD)" if mode_icon == "🔍" else "Supervised"
                        
                        st.success(f"✅ {len(X):,} images chargées | Mode: {mode_icon} {mode_label}")
                        st.balloons()
                        perf_logger.end_operation("image_loading_folder", f"{len(X)} images")
                        
                        if self.state.switch(AppPage.DASHBOARD):
                            st.rerun()
                    else:
                        st.error("❌ Échec du chargement dans l'état")
                
                except Exception as e:
                    logger.error(f"❌ Erreur chargement images: {e}", exc_info=True)
                    st.error(f"❌ Erreur : {str(e)}")
                    st.info("💡 Consultez les logs pour plus de détails")
    
    def _render_multiple_files_upload(self):
        """Upload multiple de fichiers images"""
        uploaded_files = st.file_uploader(
            "Choisissez des fichiers images",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'],
            accept_multiple_files=True,
            key="image_files_upload"
        )
        
        if uploaded_files and len(uploaded_files) > 0:
            st.info(f"📤 {len(uploaded_files)} fichier(s) sélectionné(s)")
            
            if st.button("🚀 Charger et Analyser", type="primary", use_container_width=True):
                perf_logger.start_operation("image_loading_files")
                with st.spinner(f"Chargement de {len(uploaded_files)} images..."):
                    try:
                        from PIL import Image
                        import tempfile
                        from pathlib import Path
                        
                        # Créer un dossier temporaire pour stocker les images
                        with tempfile.TemporaryDirectory() as temp_dir:
                            temp_path = Path(temp_dir)
                            
                            # Sauvegarder toutes les images dans le dossier temporaire
                            for idx, uploaded_file in enumerate(uploaded_files):
                                file_path = temp_path / uploaded_file.name
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                            
                            # Charger depuis le dossier temporaire comme un dossier plat
                            data_dir = str(temp_path)
                            structure = detect_dataset_structure(data_dir)
                            
                            if structure["type"] == "invalid":
                                # Forcer structure FLAT pour fichiers uploadés
                                structure = {"type": "flat"}
                            
                            # Charger les images
                            X, X_norm, y, y_train = load_images_flexible(data_dir, target_size=(256, 256))
                            
                            if len(X) == 0:
                                st.error("❌ Aucune image valide chargée")
                                return
                            
                            # Créer info basique
                            info = {
                                "total_images": len(X),
                                "structure_type": structure["type"],
                                "upload_method": "multiple_files"
                            }
                            
                            # ✅ TRANSMISSION au state manager
                            if self.state.set_images(X, X_norm, y, data_dir, structure, info, y_train=y_train):
                                st.success(f"✅ {len(X):,} images chargées depuis {len(uploaded_files)} fichier(s)")
                                st.balloons()
                                perf_logger.end_operation("image_loading_files", f"{len(X)} images")
                                
                                if self.state.switch(AppPage.DASHBOARD):
                                    st.rerun()
                            else:
                                st.error("❌ Échec du chargement dans l'état")
                    
                    except Exception as e:
                        logger.error(f"❌ Erreur chargement fichiers: {e}", exc_info=True)
                        st.error(f"❌ Erreur : {str(e)}")
                        st.info("💡 Vérifiez que les fichiers sont des images valides")
    
    def _render_mvtec_examples(self):
        """
        Exemples MVTec avec y_train
        """
        st.markdown("""
        <h3 style='text-align:center; padding:2rem'>
            🎯 Datasets MVTec Anomaly Detection (Unsupervised)
        </h3>
        """, unsafe_allow_html=True)
        
        examples = {
            "bottle": "🍾 Bouteilles",
            "cable": "🔌 Câbles",
            "capsule": "💊 Capsules",
            "metal_nut": "🔩 Écrous",
            "pill": "💊 Pilules",
            "screw": "🔧 Vis",
            "toothbrush": "🪥 Brosses à dents",
            "transistor": "⚡ Transistors",
            "zipper": "🔗 Fermetures éclair",
            "bottle_supervised": "Bouteilles (Supervised)"
        }
        
        # Grille 3x3
        for i in range(0, len(examples), 3):
            cols = st.columns(3)
            items = list(examples.items())[i:i+3]
            
            for col, (key, label) in zip(cols, items):
                with col:
                    if st.button(label, use_container_width=True, key=f"mvtec_{key}"):
                        path = os.path.join(project_root, "src", "data", "mvtec_ad", key)
                        
                        if not os.path.exists(path):
                            st.warning(f"⚠️ Dataset {key} manquant dans:\n`{path}`")
                            continue
                        
                        perf_logger.start_operation(f"mvtec_{key}_loading")
                        
                        with st.spinner(f"Chargement {label}..."):
                            try:
                                structure = detect_dataset_structure(path)
                                
                                if structure["type"] == "invalid":
                                    st.error(f"❌ Structure invalide pour {key}")
                                    continue
                                
                                # ✅ RÉCUPÉRATION y_train pour MVTec
                                X, X_norm, y, y_train = load_images_flexible(path, target_size=(256, 256))
                                
                                info = get_dataset_info(path)
                                
                                # ✅ TRANSMISSION y_train
                                if self.state.set_images(X, X_norm, y, path, structure, info, y_train=y_train):
                                    
                                    # Vérification mode détecté
                                    if y_train is not None and len(np.unique(y_train)) == 1:
                                        mode_msg = "🔍 Mode Unsupervised détecté (train = only normal)"
                                    else:
                                        mode_msg = "⚠️ Mode Supervised détecté (attention !)"
                                    
                                    st.success(f"✅ {key.title()} chargé ! | {mode_msg}")
                                    st.balloons()
                                    perf_logger.end_operation(f"mvtec_{key}_loading", f"{len(X)} images")
                                    
                                    if self.state.switch(AppPage.DASHBOARD):
                                        st.rerun()
                                else:
                                    st.error("❌ Échec chargement state")
                            
                            except Exception as e:
                                logger.error(f"❌ Erreur MVTec {key}: {e}", exc_info=True)
                                st.error(f"❌ Erreur : {str(e)}")
                                perf_logger.end_operation(f"mvtec_{key}_loading", "Échec")
    
    def render_footer(self):
        """Footer"""
        st.markdown("---")
        cols = st.columns(4)
        
        with cols[0]:
            st.markdown("**DataLab Pro**")
            st.caption("Plateforme IA complète")
        
        with cols[1]:
            st.markdown("**Fonctionnalités**")
            st.caption("• Analyse Tabulaire\n• Computer Vision\n• MLOps")
        
        with cols[2]:
            st.markdown("**Ressources**")
            st.caption("• Documentation\n• Tutoriels\n• Support")
        
        with cols[3]:
            st.markdown("**Contact**")
            st.caption("• À propos\n• Contact")
        
        st.caption("© 2025 DataLab Pro • Tous droits réservés")
    
    def render(self):
        """Rendu complet"""
        logger.info("🎨 Rendu page d'accueil v1")
        perf_logger.start_operation("home_render")
        
        try:
            ModernComponents.inject_custom_css()
            
            self.render_hero_section()
            st.markdown("---")
            self.render_value_proposition()
            st.markdown("---")
            self.render_upload_section()
            st.markdown("---")
            self.render_footer()
            
            perf_logger.end_operation("home_render", "OK")
            logger.info("✅ Page d'accueil rendue avec succès")
        
        except Exception as e:
            logger.error(f"❌ Erreur rendu page accueil: {e}", exc_info=True)
            st.error("❌ Erreur lors du chargement de la page")


# Instanciation
if __name__ == "__main__":
    ModernHomePage().render()