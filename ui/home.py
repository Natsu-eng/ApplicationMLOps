"""
🏠 DataLab Pro - Page d'Accueil Moderne
Version 1.0 | Production Ready | Upload MVTec Fonctionnel
"""

import streamlit as st
import sys
import os

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

# Logging - Configuration centralisée
from src.shared.logging import get_logger, PerformanceLogger
logger = get_logger(__name__)
perf_logger = PerformanceLogger("home_page_performance")


class ModernHomePage:
    """Page d'accueil moderne avec upload fonctionnel"""
    
    def __init__(self):
        logger.info("🏠 Initialisation de la page d'accueil")
        self.components = ModernComponents()
        self.state = STATE
    
    def render_hero_section(self):
        """Hero section impactante"""
        logger.debug("Rendu hero section")
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
                Plateforme tout-en-un d'analyse de données et d'IA — Tabulaire, Images et bien plus
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_value_proposition(self):
        """Proposition de valeur"""
        logger.debug("Rendu value proposition")
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0 3rem;">
            <h2 style="font-size: 2.5rem; font-weight: 800; color: #2c3e50; margin-bottom: 1rem;">
                Une plateforme unifiée pour tous vos besoins data
            </h2>
            <p style="font-size: 1.2rem; color: #6c757d; max-width: 800px; margin: 0 auto;">
                DataLab Pro va bien au-delà du MVTec — analysez n'importe quel type de données avec une suite complète d'outils
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(3)
        
        with cols[0]:
            ModernComponents.feature_card(
                icon="📊",
                title="Analyse Tabulaire Avancée",
                description="Explorez, nettoyez et modélisez vos données structurées avec des outils puissants",
                features=[
                    "ML Automatique & AutoML",
                    "Prétraitement intelligent",
                    "Feature Engineering",
                    "Visualisations interactives"
                ]
            )
        
        with cols[1]:
            ModernComponents.feature_card(
                icon="🖼️",
                title="Computer Vision Complète",
                description="Bien plus que MVTec — traitez n'importe quel dataset d'images avec des modèles state-of-the-art",
                features=[
                    "Classification & Détection",
                    "Segmentation d'images",
                    "Détection d'anomalies",
                    "Transfer Learning"
                ]
            )
        
        with cols[2]:
            ModernComponents.feature_card(
                icon="🚀",
                title="Production Ready",
                description="Des workflows industriels avec monitoring, versioning et déploiement",
                features=[
                    "MLflow intégré",
                    "Tracking d'expériences",
                    "Optimisation hyperparamètres",
                    "Rapports automatiques"
                ]
            )
    
    def render_features_grid(self):
        """Grille de fonctionnalités"""
        logger.debug("Rendu features grid")
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0 3rem;">
            <h2 style="font-size: 2.5rem; font-weight: 800; color: #2c3e50; margin-bottom: 1rem;">
                Fonctionnalités Clés
            </h2>
            <p style="font-size: 1.2rem; color: #6c757d;">
                Découvrez la puissance d'une plateforme data complète
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Première ligne
        cols = st.columns(4)
        features_row1 = [
            ("🤖", "AutoML", "Machine Learning Automatisé"),
            ("📁", "Multi-Format", "Tous types de données"),
            ("📊", "Visualisations", "Graphiques interactifs"),
            ("⚙️", "CI/CD", "Pipelines MLOps")
        ]
        
        for col, (icon, title, desc) in zip(cols, features_row1):
            with col:
                ModernComponents.metric_card(title, desc, icon)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Deuxième ligne détaillée
        cols = st.columns(2)
        
        with cols[0]:
            st.markdown("""
            <div class="modern-card">
                <div class="modern-card-icon">🔍</div>
                <div class="modern-card-title">Exploration Intelligente</div>
                <div class="modern-card-content">
                    <p><strong>Découvrez automatiquement les insights cachés</strong></p>
                    <ul>
                        <li>Analyse de corrélations avancée</li>
                        <li>Détection automatique d'anomalies</li>
                        <li>Visualisations interactives</li>
                        <li>Rapports d'exploration automatisés</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with cols[1]:
            st.markdown("""
            <div class="modern-card">
                <div class="modern-card-icon">🧠</div>
                <div class="modern-card-title">IA Explicable</div>
                <div class="modern-card-content">
                    <p><strong>Comprenez les décisions de vos modèles</strong></p>
                    <ul>
                        <li>SHAP & Feature Importance</li>
                        <li>Analyse de biais et fairness</li>
                        <li>Visualisation des prédictions</li>
                        <li>Rapports d'audit complets</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def render_how_it_works(self):
        """Comment ça marche"""
        logger.debug("Rendu how it works")
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0 3rem;">
            <h2 style="font-size: 2.5rem; font-weight: 800; color: #2c3e50; margin-bottom: 1rem;">
                Comment ça marche ?
            </h2>
            <p style="font-size: 1.2rem; color: #6c757d;">
                3 étapes simples pour transformer vos données en insights
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(3)
        
        steps = [
            ("1️⃣", "Importez vos données", "CSV, Excel, images, bases de données — nous supportons tous les formats"),
            ("2️⃣", "Explorez et modélisez", "Anlyse et preparation des données"),
            ("3️⃣", "Déployez et monitor", "Surveillez vos modèles en production")
        ]
        
        for col, (emoji, title, desc) in zip(cols, steps):
            with col:
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">{emoji}</div>
                    <h3 style="color: #2c3e50;">{title}</h3>
                    <p style="color: #666;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_upload_section(self):
        """Section upload avec onglets"""
        logger.debug("Rendu upload section")
        st.markdown("""
        <div style="text-align: center; margin: 4rem 0 2rem;">
            <h2 style="font-size: 2.5rem; font-weight: 800; color: #2c3e50; margin-bottom: 1rem;">
                Commencez Maintenant
            </h2>
            <p style="font-size: 1.2rem; color: #6c757d;">
                Importez vos données et découvrez la puissance de l'IA
            </p>
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
        """Upload tabulaire fonctionnel"""
        logger.debug("Rendu upload tabulaire")
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
            <h3>Glissez-déposez vos fichiers tabulaires</h3>
            <p style="color: #666; margin-bottom: 2rem;">
                CSV, Excel, Parquet, JSON, Feather — jusqu'à 500MB
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choisir un fichier",
            type=['csv', 'xlsx', 'xls', 'parquet', 'feather', 'json'],
            label_visibility="collapsed",
            key="tabular_upload"
        )
        
        if uploaded_file:
            logger.info(f"📁 Fichier détecté: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.2f} MB)")
            st.success(f"✅ Fichier **{uploaded_file.name}** prêt")
            
            if st.button("🚀 Analyser", type="primary", use_container_width=True, key="analyze_tabular"):
                perf_logger.start_operation("tabular_data_loading")
                try:
                    with st.spinner("Chargement des données..."):
                        logger.info(f"🔄 Début du chargement de {uploaded_file.name}")
                        df, report, df_raw = load_data(uploaded_file, sanitize_for_display=True)
                        
                        if df is not None and not df.empty:
                            logger.info(f"✅ Données chargées avec succès: {len(df):,} lignes x {len(df.columns)} colonnes")
                            
                            if self.state.set_tabular(df, df_raw, uploaded_file.name):
                                st.success(f"✅ {len(df):,} lignes chargées !")
                                st.balloons()
                                perf_logger.end_operation("tabular_data_loading", f"{len(df):,} lignes")
                                
                                if self.state.switch(AppPage.DASHBOARD):
                                    logger.info("🎯 Navigation vers le dashboard")
                                    st.rerun()
                        else:
                            logger.warning("⚠️ Fichier vide ou illisible")
                            st.error("❌ Fichier vide ou illisible")
                            
                except Exception as e:
                    logger.error(f"❌ Erreur upload tabulaire: {e}", exc_info=True)
                    perf_logger.end_operation("tabular_data_loading", "Échec")
                    st.error(f"❌ Erreur: {str(e)}")
    
    def _render_image_upload(self):
        """Upload images fonctionnel"""
        logger.debug("Rendu upload images")
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📁</div>
            <h3>Sélectionnez un dossier d'images</h3>
            <p style="color: #666; margin-bottom: 2rem;">
                Dossiers organisés par classes ou structure MVTec
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            data_dir = st.text_input(
                "Chemin du dossier",
                placeholder="/chemin/vers/images",
                help="Chemin absolu vers le dossier d'images",
                key="image_dir_input"
            )
        
        with col2:
            st.write("<br>", unsafe_allow_html=True)
            load_btn = st.button("📁 Charger", use_container_width=True, type="primary", key="load_images")
        
        if load_btn and data_dir:
            if not os.path.exists(data_dir):
                logger.warning(f"⚠️ Dossier introuvable: {data_dir}")
                st.error("❌ Dossier introuvable")
            else:
                perf_logger.start_operation("image_data_loading")
                try:
                    logger.info(f"🔄 Analyse du dossier: {data_dir}")
                    with st.spinner("Analyse du dossier..."):
                        structure = detect_dataset_structure(data_dir)
                        
                        if structure["type"] == "invalid":
                            error_msg = structure.get('error', 'Structure invalide')
                            logger.warning(f"⚠️ Structure invalide: {error_msg}")
                            st.error(f"❌ {error_msg}")
                        else:
                            logger.info(f"✅ Structure détectée: {structure['type']}")
                            with st.spinner("Chargement des images..."):
                                X, y = load_images_flexible(data_dir, target_size=(256, 256))
                                
                                if len(X) > 0:
                                    logger.info(f"✅ {len(X):,} images chargées")
                                    X_norm = X / 255.0 if X.max() > 1 else X.copy()
                                    info = get_dataset_info(data_dir)
                                    
                                    if self.state.set_images(X, X_norm, y, data_dir, structure, info):
                                        st.success(f"✅ {len(X):,} images chargées !")
                                        st.balloons()
                                        perf_logger.end_operation("image_data_loading", f"{len(X):,} images")
                                        
                                        if self.state.switch(AppPage.DASHBOARD):
                                            logger.info("🎯 Navigation vers le dashboard")
                                            st.rerun()
                                else:
                                    logger.warning("⚠️ Aucune image trouvée")
                                    st.error("❌ Aucune image trouvée")
                                    
                except Exception as e:
                    logger.error(f"❌ Erreur upload images: {e}", exc_info=True)
                    perf_logger.end_operation("image_data_loading", "Échec")
                    st.error(f"❌ Erreur: {str(e)}")
    
    def _render_mvtec_examples(self):
        """Exemples MVTec fonctionnels"""
        logger.debug("Rendu exemples MVTec")
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h3>🎯 Datasets MVTec Anomaly Detection</h3>
            <p style="color: #666; margin-bottom: 2rem;">
                Essayez la détection d'anomalies sur des cas d'usage industriels
            </p>
        </div>
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
            "zipper": "🔗 Fermetures éclair"
        }
        
        # Grille 3x3
        for i in range(0, len(examples), 3):
            cols = st.columns(3)
            items = list(examples.items())[i:i+3]
            
            for col, (key, label) in zip(cols, items):
                with col:
                    if st.button(label, use_container_width=True, key=f"mvtec_{key}"):
                        logger.info(f"🎯 Sélection exemple MVTec: {key}")
                        example_path = os.path.join(project_root, "src", "data", "mvtec_ad", key)
                        
                        if os.path.exists(example_path):
                            perf_logger.start_operation(f"mvtec_{key}_loading")
                            try:
                                with st.spinner(f"Chargement {label}..."):
                                    logger.info(f"🔄 Chargement dataset MVTec: {key} depuis {example_path}")
                                    structure = detect_dataset_structure(example_path)
                                    
                                    if structure["type"] != "invalid":
                                        X, y = load_images_flexible(example_path, target_size=(256, 256))
                                        X_norm = X / 255.0 if X.max() > 1 else X.copy()
                                        info = get_dataset_info(example_path)
                                        
                                        if self.state.set_images(X, X_norm, y, example_path, structure, info):
                                            logger.info(f"✅ MVTec {key} chargé: {len(X)} images")
                                            st.success(f"✅ {key.title()} chargé !")
                                            st.balloons()
                                            perf_logger.end_operation(f"mvtec_{key}_loading", f"{len(X)} images")
                                            
                                            if self.state.switch(AppPage.DASHBOARD):
                                                st.rerun()
                                        else:
                                            logger.error(f"❌ Échec chargement state pour {key}")
                                            st.error("❌ Erreur de chargement")
                                    else:
                                        logger.warning(f"⚠️ Structure invalide pour {key}")
                                        st.error("❌ Structure invalide")
                            except Exception as e:
                                logger.error(f"❌ Erreur MVTec {key}: {e}", exc_info=True)
                                perf_logger.end_operation(f"mvtec_{key}_loading", "Échec")
                                st.error(f"❌ Erreur: {str(e)}")
                        else:
                            logger.warning(f"⚠️ Dataset MVTec {key} non disponible: {example_path}")
                            st.warning(f"⚠️ Dataset {key} non disponible dans:\n`{example_path}`")
    
    def render_footer(self):
        """Footer"""
        logger.debug("Rendu footer")
        st.markdown("---")
        
        cols = st.columns(4)
        
        with cols[0]:
            st.markdown("**DataLab Pro**")
            st.caption("Plateforme IA complète")
        
        with cols[1]:
            st.markdown("**Fonctionnalités**")
            st.caption("• Analyse Tabulaire")
            st.caption("• Computer Vision")
            st.caption("• MLOps")
        
        with cols[2]:
            st.markdown("**Ressources**")
            st.caption("• Documentation")
            st.caption("• Tutoriels")
            st.caption("• API")
        
        with cols[3]:
            st.markdown("**Contact**")
            st.caption("• Support")
            st.caption("• À propos")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("© 2025 DataLab Pro. Plateforme d'analyse IA tout-en-un.")
    
    def render(self):
        """Rendu complet"""
        logger.info("🎨 Début du rendu de la page d'accueil")
        perf_logger.start_operation("home_page_render")
        
        try:
            # CSS
            ModernComponents.inject_custom_css()
            
            # Sections
            self.render_hero_section()
            st.markdown("---")
            self.render_value_proposition()
            st.markdown("---")
            self.render_features_grid()
            st.markdown("---")
            self.render_how_it_works()
            st.markdown("---")
            self.render_upload_section()
            st.markdown("---")
            self.render_footer()
            
            perf_logger.end_operation("home_page_render", "Succès")
            logger.info("✅ Rendu de la page d'accueil terminé avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du rendu de la page d'accueil: {e}", exc_info=True)
            perf_logger.end_operation("home_page_render", "Échec")
            st.error("Une erreur est survenue lors du chargement de la page")