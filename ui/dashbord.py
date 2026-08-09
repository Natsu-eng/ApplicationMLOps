"""
📊 DataLab Pro - Dashboard Complet
Version 1.0 | Production Ready | Design Moderne
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import sys
import os
import io

# Configuration paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, Dict, Any, Tuple
from helpers.ui_components.cards import ModernComponents
from helpers.ui_components.layaout import LayoutComponents, DataDisplayComponents
from helpers.ui_components.sections import SectionComponents
from monitoring.state_managers import STATE, AppPage
from src.data.data_analysis import (
    auto_detect_column_types,
    compute_global_metrics,
    detect_useless_columns,
    get_all_problematic_columns,
    get_data_profile
)
from src.explorations.exploratory_plots import (
    plot_missing_values_overview,
    plot_correlation_heatmap,
    create_simple_correlation_heatmap
)
from src.explorations.image_exploration_plots import (
    analyze_image_quality,
    get_dataset_stats
)

# Logging
from src.shared.logging import get_logger, PerformanceLogger
logger = get_logger(__name__)
perf_logger = PerformanceLogger("dashboard_performance")


class ModernDashboard:
    """Dashboard complet avec design moderne"""
    
    def __init__(self):
        logger.info("📊 Initialisation du dashboard")
        self.components = ModernComponents()
        self.layout = LayoutComponents()
        self.data_display = DataDisplayComponents()
        self.sections = SectionComponents()
        self.state = STATE
    
    def render_header(self):
        """En-tête du dashboard"""
        logger.debug("Rendu header dashboard")
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown('<div class="modern-header">📊 Dashboard Analytique</div>', unsafe_allow_html=True)
            dataset_type = "Tabulaires" if self.state.tabular else "Images" if self.state.images else "Aucun"
            st.markdown(f'<p style="font-size: 1.1rem; color: #666;">**{dataset_type}** • `{self.state.data.name or "Dataset"}`</p>', unsafe_allow_html=True)
        
        with col2:
            self._render_status_badge()
        
        with col3:
            self._render_quick_actions()
    
    def _render_status_badge(self):
        """Badge de statut"""
        logger.debug("Rendu status badge")
        if self.state.images:
            st.markdown('<div class="status-badge success">🖼️ IMAGES</div>', unsafe_allow_html=True)
        elif self.state.tabular:
            st.markdown('<div class="status-badge info">📊 TABULAIRE</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge warning">⏳ EN ATTENTE</div>', unsafe_allow_html=True)
    
    def _render_quick_actions(self):
        """Actions rapides - Correction use_container_width"""
        logger.debug("Rendu quick actions")
        if self.state.tabular:
            if st.button("🤖 ML", key="nav_ml", type="primary"):
                logger.info("🎯 Navigation vers ML Training")
                self.state.switch(AppPage.ML_TRAINING)
        elif self.state.images:
            if st.button("👁️ Vision", key="nav_cv", type="primary"):
                logger.info("🎯 Navigation vers CV Training")
                self.state.switch(AppPage.CV_TRAINING)
        
        if st.button("🏠 Accueil", key="nav_home"):
            logger.info("🏠 Retour à l'accueil")
            self.state.switch(AppPage.HOME)
    
    def render_overview(self):
        """Vue d'ensemble"""
        logger.debug("Rendu overview")
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2 style="font-size: 2rem; font-weight: 700; color: #2c3e50;">
                Vue d'Ensemble
            </h2>
            <p style="color: #666;">Statistiques globales du dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
        if self.state.tabular:
            self._render_tabular_overview()
        elif self.state.images:
            self._render_images_overview()
        else:
            logger.warning("⚠️ Aucune donnée chargée")
            st.info("📥 Chargez des données pour voir les métriques")
    
    def _render_tabular_overview(self):
        """Overview tabulaire avec métriques modernes"""
        logger.debug("Rendu tabular overview")
        perf_logger.start_operation("tabular_overview")
        
        try:
            df = self.state.data.df
            logger.info(f"📊 Calcul métriques pour dataset: {len(df)} lignes x {len(df.columns)} colonnes")
            
            metrics = compute_global_metrics(df)
            mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
            
            # Utiliser le nouveau composant de métriques
            self.layout.metric_row([
                {
                    "label": "Lignes",
                    "value": f"{metrics['n_rows']:,}",
                    "icon": "📈",
                    "color": "#667eea"
                },
                {
                    "label": "Colonnes",
                    "value": metrics['n_cols'],
                    "icon": "📊",
                    "color": "#f093fb"
                },
                {
                    "label": "Manquants",
                    "value": f"{metrics['missing_percentage']:.1f}%",
                    "icon": "⚠️",
                    "color": "#feca57"
                },
                {
                    "label": "Doublons",
                    "value": f"{metrics['duplicate_rows']:,}",
                    "icon": "🔍",
                    "color": "#ee5a6f"
                },
                {
                    "label": "Mémoire",
                    "value": f"{mem_mb:.1f} MB",
                    "icon": "💾",
                    "color": "#4facfe"
                }
            ])
            
            perf_logger.end_operation("tabular_overview", f"{metrics['n_rows']:,} lignes analysées")
            
        except Exception as e:
            logger.error(f"❌ Erreur tabular overview: {e}", exc_info=True)
            st.error("Erreur lors du calcul des métriques")
    
    def _render_images_overview(self):
        """Overview images avec métriques modernes"""
        logger.debug("Rendu images overview")
        perf_logger.start_operation("images_overview")
        
        try:
            d = self.state.data
            logger.info(f"🖼️ Affichage métriques images: {d.img_count} images, {d.n_classes} classes")
            
            mem_mb = d.X.nbytes/(1024**2) if d.X is not None else 0
            shape_str = f"{d.img_shape[1]}×{d.img_shape[2]}" if d.img_shape else "N/A"
            if d.task == "anomaly_detection_unsupervised":
                task = "Anomalies (Unsupervised)"
            elif "classification" in d.task:
                task = "Classification"
            else:
                task = d.task.replace("_", " ").title()
            
            self.layout.metric_row([
                {
                    "label": "Images",
                    "value": f"{d.img_count:,}",
                    "icon": "🖼️",
                    "color": "#667eea"
                },
                {
                    "label": "Résolution",
                    "value": shape_str,
                    "icon": "📐",
                    "color": "#f093fb"
                },
                {
                    "label": "Classes",
                    "value": d.n_classes,
                    "icon": "🏷️",
                    "color": "#4facfe"
                },
                {
                    "label": "Mémoire",
                    "value": f"{mem_mb:.1f} MB",
                    "icon": "💾",
                    "color": "#43e97b"
                },
                {
                    "label": "Tâche",
                    "value": task,
                    "icon": "🎯",
                    "color": "#feca57"
                }
            ])
            
            perf_logger.end_operation("images_overview", f"{d.img_count} images")
            
        except Exception as e:
            logger.error(f"❌ Erreur images overview: {e}", exc_info=True)
            st.error("Erreur lors du calcul des métriques images")
    
    def render_tabular_tabs(self):
        """Onglets pour données tabulaires - Design moderne"""
        logger.debug("Rendu tabular tabs")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "👀 Aperçu",
            "🏷️ Types & Variables",
            "🔗 Relations",
            "🧹 Nettoyage",
            "📈 Corrélations",
            "📊 Profil Complet"
        ])
        
        with tab1:
            self._render_data_preview()
        
        with tab2:
            self._render_types_and_variables()
        
        with tab3:
            self._render_relationships()
        
        with tab4:
            self._render_data_cleaning()
        
        with tab5:
            self._render_correlations()
        
        with tab6:
            self._render_full_profile()
    
    def _render_data_preview(self):
        """Onglet Aperçu - Design moderne"""
        logger.debug("Rendu data preview")
        st.markdown("""
        <div style='
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-left: 1rem;
            border-left: 4px solid #667eea;
        '>👀 Aperçu des Données</div>
        """, unsafe_allow_html=True)
        
        try:
            df = self.state.data.df

            # Détection des colonnes datetime
            problematic_cols = []
            if problematic_cols:
                logger.info(f"Colonnes datetime détectées: {problematic_cols}")

            # Options d'affichage
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                n_rows = st.slider("Nombre de lignes", 10, 100, 50, key="preview_rows")
            with col2:
                show_info = st.checkbox("Info détaillée", value=False)
            with col3:
                show_stats = st.checkbox("Statistiques", value=True)
            
            # Aperçu avec style moderne
            self.data_display.styled_dataframe(
                df.head(n_rows),
                title="📊 Échantillon de Données",
                height=400
            )
            
            if show_info:
                st.markdown("""
                <div style='
                    background: #f8f9fa;
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin-top: 1rem;
                '>
                    <div style='font-size: 1.2rem; font-weight: 700; margin-bottom: 1rem;'>
                        📊 Informations Détaillées
                    </div>
                """, unsafe_allow_html=True)
                
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.code(buffer.getvalue())
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Statistiques descriptives
            if show_stats:
                st.markdown("<br>", unsafe_allow_html=True)
                self.data_display.styled_dataframe(
                    df.describe(),
                    title="📈 Statistiques Descriptives",
                    height=300
                )
            
            logger.info(f"✅ Aperçu affiché: {n_rows} lignes sur {len(df)}")
            
        except Exception as e:
            logger.error(f"❌ Erreur data preview: {e}", exc_info=True)
            st.error("Erreur lors de l'affichage de l'aperçu")
    
    def _render_types_and_variables(self):
        """Onglet Types & Variables - Design moderne"""
        logger.debug("Rendu types and variables")
        st.markdown("""
        <div style='
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-left: 1rem;
            border-left: 4px solid #f093fb;
        '>🏷️ Types de Colonnes & Variables</div>
        """, unsafe_allow_html=True)
        
        try:
            df = self.state.data.df
            logger.info("🔍 Détection des types de colonnes")
            column_types = auto_detect_column_types(df)
            
            # Statistiques par type avec nouveau composant
            st.markdown("#### 📊 Répartition des Types")
            
            type_stats = [
                {
                    "value": len(column_types.get('numeric', [])),
                    "label": "Numériques",
                    "icon": "🔢",
                    "color": "#667eea"
                },
                {
                    "value": len(column_types.get('categorical', [])),
                    "label": "Catégorielles",
                    "icon": "🏷️",
                    "color": "#f093fb"
                },
                {
                    "value": len(column_types.get('text_or_high_cardinality', [])),
                    "label": "Texte",
                    "icon": "📝",
                    "color": "#4facfe"
                },
                {
                    "value": len(column_types.get('datetime', [])),
                    "label": "Dates",
                    "icon": "📅",
                    "color": "#43e97b"
                }
            ]
            
            self.sections.stats_section(type_stats, layout="horizontal")
            
            # Détail par type
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📋 Détail des Colonnes")
            
            type_configs = [
                ('numeric', 'Numériques', '🔢'),
                ('categorical', 'Catégorielles', '🏷️'),
                ('text_or_high_cardinality', 'Texte', '📝'),
                ('datetime', 'Dates', '📅')
            ]
            
            accordion_items = []
            for key, label, icon in type_configs:
                cols_list = column_types.get(key, [])
                if cols_list:
                    # Préparer les données pour le tableau
                    col_data = []
                    for col in cols_list:
                        col_series = df[col]
                        null_pct = (col_series.isnull().sum() / len(col_series) * 100)
                        unique_count = col_series.nunique()
                        
                        col_data.append({
                            "Colonne": col,
                            "Uniques": f"{unique_count:,}",
                            "Manquants": f"{null_pct:.1f}%",
                            "Type": str(col_series.dtype)
                        })
                    
                    def create_content(data=col_data):
                        self.sections.comparison_table(
                            data,
                            headers=["Colonne", "Uniques", "Manquants", "Type"]
                        )
                    
                    accordion_items.append({
                        "title": f"{label} ({len(cols_list)} colonnes)",
                        "icon": icon,
                        "content": create_content
                    })
            
            self.sections.accordion_section(accordion_items)
            
            logger.info(f"✅ Types détectés: {sum(len(v) for v in column_types.values())} colonnes classifiées")
            
        except Exception as e:
            logger.error(f"❌ Erreur types and variables: {e}", exc_info=True)
            st.error("Erreur lors de la détection des types")
    
    def _render_relationships(self):
        """Onglet Relations - Design moderne"""
        logger.debug("Rendu relationships")
        st.markdown("""
        <div style='
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-left: 1rem;
            border-left: 4px solid #4facfe;
        '>🔗 Relations entre Variables</div>
        """, unsafe_allow_html=True)
        
        try:
            df = self.state.data.df
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if len(numeric_cols) >= 2:
                st.markdown("#### 📊 Analyse Bivariée")
                
                col1, col2 = st.columns(2)
                with col1:
                    var1 = st.selectbox("Variable 1", numeric_cols, key="rel_var1")
                with col2:
                    var2 = st.selectbox("Variable 2", [c for c in numeric_cols if c != var1], key="rel_var2")
                
                if var1 and var2:
                    logger.info(f"📊 Analyse relation: {var1} vs {var2}")
                    
                    # Scatter plot moderne
                    fig = px.scatter(
                        df,
                        x=var1,
                        y=var2,
                        title=f"Relation: {var1} vs {var2}",
                        trendline="ols",
                        color_discrete_sequence=["#667eea"],
                        template="plotly_white"
                    )
                    
                    fig.update_layout(
                        title_font_size=18,
                        title_font_color="#2c3e50",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    
                    st.plotly_chart(fig, width="stretch")
                    
                    # Corrélation avec badge moderne
                    corr = df[[var1, var2]].corr().iloc[0, 1]
                    
                    st.markdown(f"""
                    <div style='
                        background: linear-gradient(135deg, #667eea, #764ba2);
                        color: white;
                        padding: 1.5rem;
                        border-radius: 12px;
                        text-align: center;
                        margin-top: 1rem;
                    '>
                        <div style='font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem;'>
                            Corrélation de Pearson
                        </div>
                        <div style='font-size: 2.5rem; font-weight: 800;'>
                            {corr:.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    logger.info(f"✅ Corrélation calculée: {corr:.3f}")
            else:
                logger.warning("⚠️ Colonnes numériques insuffisantes pour analyse bivariée")
                self.layout.info_badge(
                    "Au moins 2 colonnes numériques requises pour l'analyse bivariée",
                    badge_type="warning",
                    icon="⚠️"
                )
                
        except Exception as e:
            logger.error(f"❌ Erreur relationships: {e}", exc_info=True)
            st.error("Erreur lors de l'analyse des relations")
    
    def _render_data_cleaning(self):
        """Onglet Nettoyage - Design moderne"""
        logger.debug("Rendu data cleaning")
        st.markdown("""
        <div style='
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-left: 1rem;
            border-left: 4px solid #43e97b;
        '>🧹 Nettoyage des Données</div>
        """, unsafe_allow_html=True)
        
        try:
            df = self.state.data.df
            
            # Valeurs manquantes
            st.markdown("#### ❓ Valeurs Manquantes")
            logger.info("🔍 Analyse des valeurs manquantes")
            
            missing_fig = plot_missing_values_overview(df)
            if missing_fig:
                st.plotly_chart(missing_fig, width="stretch")
                logger.info("✅ Graphique valeurs manquantes affiché")
            else:
                self.layout.info_badge(
                    "Aucune valeur manquante détectée dans le dataset",
                    badge_type="success",
                    icon="✅"
                )
            
            # Colonnes problématiques
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🔍 Colonnes Problématiques")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                analyze_btn = st.button("🔬 Analyser les Colonnes", type="primary")
            
            if analyze_btn:
                perf_logger.start_operation("analyze_problematic_columns")
                with st.spinner("Analyse en cours..."):
                    logger.info("🔍 Début analyse colonnes problématiques")
                    problematic = detect_useless_columns(df)
                    
                    if problematic:
                        total_issues = sum(len(v) for v in problematic.values())
                        logger.warning(f"⚠️ {total_issues} colonnes problématiques détectées")
                        
                        self.layout.info_badge(
                            f"{total_issues} colonnes problématiques détectées dans {len(problematic)} catégories",
                            badge_type="warning",
                            icon="⚠️"
                        )
                        
                        # Afficher avec accordion moderne
                        accordion_items = []
                        for category, cols in problematic.items():
                            if cols:
                                category_label = category.replace('_', ' ').title()
                                
                                def create_content(columns=cols):
                                    for col in columns:
                                        st.markdown(f"- `{col}`")
                                
                                accordion_items.append({
                                    "title": f"{category_label} ({len(cols)} colonnes)",
                                    "icon": "🔍",
                                    "content": create_content
                                })
                        
                        self.sections.accordion_section(accordion_items)
                        
                        all_problematic = get_all_problematic_columns(problematic)
                        self.layout.info_badge(
                            f"Recommandation: Supprimer ou traiter ces {len(all_problematic)} colonnes avant le ML",
                            badge_type="info",
                            icon="💡"
                        )
                    else:
                        logger.info("✅ Aucune colonne problématique détectée")
                        self.layout.info_badge(
                            "Aucune colonne problématique détectée - Dataset prêt pour le ML",
                            badge_type="success",
                            icon="✅"
                        )
                
                perf_logger.end_operation("analyze_problematic_columns", f"{total_issues if problematic else 0} colonnes")
                
        except Exception as e:
            logger.error(f"❌ Erreur data cleaning: {e}", exc_info=True)
            st.error("Erreur lors de l'analyse de nettoyage")
    
    def _render_correlations(self):
        """Onglet Corrélations - Design moderne"""
        logger.debug("Rendu correlations")
        st.markdown("""
        <div style='
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-left: 1rem;
            border-left: 4px solid #feca57;
        '>📈 Matrice de Corrélation</div>
        """, unsafe_allow_html=True)
        
        try:
            df = self.state.data.df
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if len(numeric_cols) >= 2:
                logger.info(f"📊 Calcul corrélations pour {len(numeric_cols)} colonnes numériques")
                perf_logger.start_operation("correlation_matrix")
                
                with st.spinner("Calcul des corrélations..."):
                    try:
                        result = plot_correlation_heatmap(df)
                        
                        if result and isinstance(result, tuple):
                            fig, used_cols = result
                            if fig:
                                st.plotly_chart(fig, width="stretch")
                                self.layout.info_badge(
                                    f"{len(used_cols)} variables analysées dans la matrice de corrélation",
                                    badge_type="info",
                                    icon="📊"
                                )
                                logger.info(f"✅ Matrice de corrélation affichée pour {len(used_cols)} variables")
                            else:
                                st.error("❌ Erreur lors du calcul")
                        else:
                            # Fallback simple
                            fig, used_cols = create_simple_correlation_heatmap(df)
                            if fig:
                                st.plotly_chart(fig, width="stretch")
                                logger.info("✅ Matrice simple affichée (fallback)")
                            else:
                                st.error("❌ Impossible de calculer les corrélations")
                                
                    except Exception as e:
                        logger.error(f"❌ Erreur calcul corrélation: {e}", exc_info=True)
                        st.error(f"❌ Erreur: {str(e)}")
                
                perf_logger.end_operation("correlation_matrix", f"{len(numeric_cols)} colonnes")
            else:
                logger.warning("⚠️ Colonnes numériques insuffisantes pour matrice corrélation")
                self.layout.info_badge(
                    "Au moins 2 colonnes numériques requises pour la matrice de corrélation",
                    badge_type="warning",
                    icon="⚠️"
                )
                
        except Exception as e:
            logger.error(f"❌ Erreur correlations: {e}", exc_info=True)
            st.error("Erreur lors du calcul des corrélations")
    
    def _render_full_profile(self):
        """Onglet Profil Complet - Design moderne sans JSON"""
        logger.debug("Rendu full profile")
        st.markdown("""
        <div style='
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-left: 1rem;
            border-left: 4px solid #ee5a6f;
        '>📊 Profil Complet des Données</div>
        """, unsafe_allow_html=True)
        
        if st.button("🔬 Générer le Profil", type="primary"):
            perf_logger.start_operation("generate_full_profile")
            with st.spinner("Génération en cours..."):
                try:
                    df = self.state.data.df
                    logger.info(f"🔬 Génération profil complet pour {len(df.columns)} colonnes")
                    
                    profile = get_data_profile(df)
                    
                    if profile:
                        self.layout.info_badge(
                            f"Profil généré avec succès pour {len(profile)} colonnes",
                            badge_type="success",
                            icon="✅"
                        )
                        logger.info(f"✅ Profil généré avec succès pour {len(profile)} colonnes")
                        
                        # Afficher par colonne avec design moderne
                        accordion_items = []
                        for col, stats in profile.items():
                            def create_content(column=col, statistics=stats):
                                # Afficher les stats de manière élégante
                                self.data_display.key_value_pairs(
                                    statistics,
                                    title=None,
                                    columns=2
                                )
                            
                            accordion_items.append({
                                "title": col,
                                "icon": "📊",
                                "content": create_content
                            })
                        
                        self.sections.accordion_section(
                            accordion_items,
                            title="Profil Détaillé par Colonne"
                        )
                    else:
                        logger.error("❌ Profil vide retourné")
                        st.error("❌ Erreur lors de la génération du profil")

                    perf_logger.end_operation("generate_full_profile", f"{len(profile)} colonnes")
                    
                except Exception as e:
                    logger.error(f"❌ Erreur génération profil: {e}", exc_info=True)
                    st.error(f"❌ Erreur lors de la génération: {str(e)}")

    def render_image_tabs(self):
        """Onglets pour images - Design moderne avec 5 onglets"""
        logger.debug("Rendu image tabs")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎨 Galerie",
            "📊 Distribution",
            "🔍 Qualité",
            "📈 Statistiques",
            "📋 Détails"
        ])
        
        with tab1:
            self._render_image_gallery()
        
        with tab2:
            self._render_image_distribution()
        
        with tab3:
            self._render_image_quality()
        
        with tab4:
            self._render_image_statistics()
        
        with tab5:
            self._render_image_details()

    def _get_class_names(self) -> List[str]:
        """
        Retourne les VRAIS noms dans l'ordre des labels
        Gère MVTec, Supervised Anomaly, Classification
        """
        d = self.state.data
        
        # CAS 1: Anomaly Detection (supervisée ou MVTec)
        if d.task and 'anomaly' in d.task.lower():
            # Toujours: label 0 = Normal, label 1 = Anomalie/Défectueuse
            if d.n_classes == 2:
                return ["Normal", "Défectueuse"]
            # Fallback si plus de 2 classes (rare)
            return [f"Classe {i}" for i in range(d.n_classes or 0)]
        
        # CAS 2: Classification avec class_names fournis (prioritaire)
        if hasattr(d, 'info') and d.info and 'class_names' in d.info:
            class_names = d.info['class_names']
            if len(class_names) == d.n_classes:
                logger.debug(f"✅ Utilisation class_names depuis info: {class_names}")
                return class_names
        
        # CAS 3: Structure avec class_to_idx (ImageFolder)
        if d.structure and 'class_to_idx' in d.structure:
            class_to_idx = d.structure['class_to_idx']
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            names = [idx_to_class.get(i, f"Classe {i}") for i in range(d.n_classes)]
            logger.debug(f"✅ Utilisation class_to_idx: {names}")
            return names
        
        # CAS 4: Liste des classes explicite dans structure
        if d.structure and 'categories' in d.structure:
            categories = d.structure['categories']
            if len(categories) == d.n_classes:
                logger.debug(f"✅ Utilisation categories: {categories}")
                return categories
        
        # FALLBACK: Noms génériques
        logger.warning(f"⚠️ Pas de métadonnées classes, fallback générique")
        return [f"Classe {i}" for i in range(d.n_classes or 0)]

    def _render_image_gallery(self):
        """
        Galerie avec noms de classes
        """
        logger.debug("Rendu image gallery")
        st.markdown("""
        <div style='
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-left: 1rem;
            border-left: 4px solid #667eea;
        '>🎨 Galerie d'Images</div>
        """, unsafe_allow_html=True)
        
        try:
            d = self.state.data
            class_names = self._get_class_names()  
            
            # Vérification cohérence
            if len(class_names) != d.n_classes:
                logger.error(
                    f"Incohérence: {len(class_names)} class_names "
                    f"pour {d.n_classes} classes"
                )
                class_names = [f"Classe {i}" for i in range(d.n_classes)]
            
            # Contrôles
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                n_samples = st.slider("Nombre d'images", 4, 16, 8, step=2)
            with col2:
                grid_cols = st.selectbox("Colonnes", [2, 3, 4, 6], index=1)
            with col3:
                if st.button("🔄 Aléatoire", width="stretch"):
                    st.rerun()
            
            # Sélection images
            indices = np.random.choice(len(d.X), min(n_samples, len(d.X)), replace=False)
            
            cols = st.columns(grid_cols)
            for i, idx in enumerate(indices):
                with cols[i % grid_cols]:
                    img = d.X[idx]
                    if img.max() > 1.0:
                        img = img / 255.0
                    
                    st.markdown("""
                    <div style='
                        background: white;
                        border-radius: 12px;
                        padding: 1rem;
                        margin-bottom: 1rem;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                        border: 1px solid #e0e0e0;
                    '>
                    """, unsafe_allow_html=True)
                    
                    st.image(img, width='stretch')
                    
                    # AFFICHAGE CORRECT
                    label_idx = d.y[idx]
                    # VALIDATION INTELLIGENTE: vérifier que class_names existe ET a assez d'éléments
                    if class_names and len(class_names) > 0:
                        # Cas MVTec AD: toujours 2 classes possibles (Normal=0, Anomaly=1)
                        is_mvtec = (d.task or '').lower() in ['unsupervised', 'anomaly_detection_unsupervised']
                        
                        if is_mvtec and label_idx in [0, 1]:
                            # Pour MVTec AD: forcer les noms corrects même si n_classes=1
                            if label_idx == 0:
                                label = "Normal"
                            else:
                                label = "Anomaly"
                            color = "#4facfe" if label_idx == 0 else "#ee5a6f"
                            icon = "✅" if label_idx == 0 else "⚠️"
                        
                        elif label_idx < len(class_names):
                            # Cas standard: utiliser class_names
                            label = class_names[label_idx]
                            
                            # Couleur selon type
                            if 'anomaly' in (d.task or '').lower():
                                color = "#4facfe" if label_idx == 0 else "#ee5a6f"
                                icon = "✅" if label_idx == 0 else "⚠️"
                            else:
                                color = "#667eea"
                                icon = "🏷️"
                        
                        else:
                            # Hors range: fallback
                            label = f"Classe {label_idx}"
                            color = "#6c757d"
                            icon = "❓"
                            logger.debug(f"Label {label_idx} hors range class_names (len={len(class_names)})")

                    else:
                        # Aucun class_names disponible: fallback générique
                        label = f"Classe {label_idx}"
                        color = "#6c757d"
                        icon = "❓"
                        logger.warning(f"⚠️ class_names vide ou None, impossible de mapper label {label_idx}")

                    # Résolution image
                    shape_str = f"{img.shape[0]}×{img.shape[1]}" if len(img.shape) >= 2 else "N/A"
                    
                    # Badge couleur selon type
                    if 'anomaly' in (d.task or '').lower():
                        color = "#4facfe" if label_idx == 0 else "#ee5a6f"
                        icon = "✅" if label_idx == 0 else "⚠️"
                    else:
                        color = "#667eea"
                        icon = "🏷️"
                    
                    st.markdown(f"""
                    <div style='
                        margin-top: 0.5rem;
                        padding: 0.5rem;
                        background: #f8f9fa;
                        border-radius: 8px;
                    '>
                        <div style='font-weight: 600; color: #2c3e50;'>#{idx}</div>
                        <div style='font-size: 0.9rem; color: {color}; font-weight: 700;'>
                            {icon} <strong>{label}</strong>
                        </div>
                        <div style='font-size: 0.8rem; color: #666;'>
                            📐 {shape_str}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            logger.info(f"✅ Galerie: {n_samples} images avec labels corrects")
            
        except Exception as e:
            logger.error(f"❌ Erreur image gallery: {e}", exc_info=True)
            self.layout.error_section("Erreur galerie", str(e))

    def _render_image_distribution(self):
        """
        Distribution avec noms CORRECTS
        """
        logger.debug("Rendu image distribution")
        st.markdown("""
        <div style='
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-left: 1rem;
            border-left: 4px solid #f093fb;
        '>📊 Distribution des Classes</div>
        """, unsafe_allow_html=True)
        
        try:
            d = self.state.data
            class_names = self._get_class_names() 
            counts = Counter(d.y)
            
            # Métriques
            total_images = sum(counts.values())
            self.layout.metric_row([
                {
                    "label": "Images Total",
                    "value": f"{total_images:,}",
                    "icon": "🖼️",
                    "color": "#667eea"
                },
                {
                    "label": "Classes",
                    "value": len(counts),
                    "icon": "🏷️",
                    "color": "#f093fb"
                },
                {
                    "label": "Classe Majoritaire",
                    "value": f"{max(counts.values()):,}",
                    "icon": "📈",
                    "color": "#4facfe"
                },
                {
                    "label": "Classe Minoritaire",
                    "value": f"{min(counts.values()):,}",
                    "icon": "📉",
                    "color": "#43e97b"
                }
            ])
            
            # LABELS CORRECTS pour le pie chart
            labels = []
            values = []
            for label_idx in sorted(counts.keys()):
                if label_idx < len(class_names):
                    labels.append(class_names[label_idx])
                else:
                    labels.append(f"Classe {label_idx}")
                values.append(counts[label_idx])
            
            # Couleurs adaptées
            if len(labels) == 2 and 'anomaly' in (d.task or '').lower():
                colors = ['#4facfe', '#ee5a6f']  # Bleu/Rouge pour normal/anomaly
            else:
                colors = px.colors.qualitative.Set3
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(colors=colors),
                textinfo='label+percent',
                insidetextorientation='radial'
            )])
            
            fig.update_layout(
                title={
                    'text': "Répartition des Classes",
                    'font': {'size': 20, 'color': '#2c3e50'}
                },
                height=500,
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Tableau avec noms corrects
            st.markdown("#### 📋 Détail par Classe")
            stats_data = []
            for label, value, count in zip(labels, values, values):
                pct = (count / total_images) * 100
                stats_data.append({
                    "Classe": label,
                    "Nombre": f"{count:,}",
                    "Pourcentage": f"{pct:.1f}%",
                    "Ratio": f"1:{total_images/count:.2f}"
                })
            
            self.data_display.styled_dataframe(
                pd.DataFrame(stats_data),
                title=None,
                height=200
            )
            
            logger.info(f"✅ Distribution: {len(counts)} classes avec noms corrects")
            
        except Exception as e:
            logger.error(f"❌ Erreur distribution: {e}", exc_info=True)
            self.layout.error_section("Erreur lors de la distribution", str(e))

    def _render_image_quality(self):
        """Analyse qualité - Design moderne"""
        logger.debug("Rendu image quality")
        st.markdown("""
        <div style='
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-left: 1rem;
            border-left: 4px solid #feca57;
        '>🔍 Analyse de Qualité</div>
        """, unsafe_allow_html=True)
        
        if st.button("🔬 Analyser la Qualité", type="primary"):
            perf_logger.start_operation("image_quality_analysis")
            with st.spinner("Analyse en cours..."):
                try:
                    logger.info("🔍 Début analyse qualité images")
                    report = analyze_image_quality(self.state.data.X, sample_size=200)
                    
                    if 'error' not in report:
                        st.success("✅ Analyse terminée")
                        logger.info("✅ Analyse qualité terminée avec succès")

                        # Métriques principales avec design moderne
                        quality_metrics = [
                            {
                                "label": "Luminosité Moyenne",
                                "value": f"{report.get('brightness', {}).get('mean', 0):.1f}",
                                "icon": "💡",
                                "color": "#667eea"
                            },
                            {
                                "label": "Contraste Moyen",
                                "value": f"{report.get('contrast', {}).get('mean', 0):.1f}",
                                "icon": "🎨",
                                "color": "#f093fb"
                            },
                            {
                                "label": "Images Problématiques",
                                "value": f"{report.get('problematic_summary', {}).get('percentage_problematic', 0):.1f}%",
                                "icon": "⚠️",
                                "color": "#ee5a6f"
                            },
                            {
                                "label": "Échantillon Analysé",
                                "value": f"{len(report.get('brightness', {}).get('values', []))}",
                                "icon": "🔍",
                                "color": "#4facfe"
                            }
                        ]
                        
                        self.layout.metric_row(quality_metrics)
                        
                        # Graphiques côte à côte
                        st.markdown("#### 📊 Distributions de Qualité")
                        col1, col2 = st.columns(2)
                        
                        brightness_values = report.get('brightness', {}).get('values', [])
                        contrast_values = report.get('contrast', {}).get('values', [])
                        
                        with col1:
                            if brightness_values:
                                fig = px.histogram(
                                    x=brightness_values,
                                    nbins=30,
                                    title="Distribution de la Luminosité",
                                    color_discrete_sequence=['#667eea']
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                )
                                st.plotly_chart(fig, width="stretch")
                        
                        with col2:
                            if contrast_values:
                                fig = px.histogram(
                                    x=contrast_values,
                                    nbins=30,
                                    title="Distribution du Contraste",
                                    color_discrete_sequence=['#f093fb']
                                )
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                )
                                st.plotly_chart(fig, width="stretch")
                        
                        # ✅ CORRECTION : Vérification sécurisée de problematic_count
                        problematic_summary = report.get('problematic_summary', {})
                        problematic_count = problematic_summary.get('total_problematic', 0)
                        
                        if problematic_count > 0:
                            st.markdown("#### ⚠️ Images Problématiques Détectées")
                            
                            problematic_details = []
                            problematic_by_type = problematic_summary.get('problematic_by_type', {})
                            
                            problematic_by_type = {
                                'dark': problematic_summary.get('total_dark', 0),
                                'bright': problematic_summary.get('total_bright', 0),
                                'low_contrast': problematic_summary.get('total_low_contrast', 0)
                            }
                            for issue_type, count in problematic_by_type.items():
                                if count > 0:
                                    problematic_details.append({
                                        "Type de Problème": issue_type.replace('_', ' ').title(),
                                        "Nombre": count,
                                        "Pourcentage": f"{(count/len(brightness_values)*100):.1f}%" if brightness_values else "N/A"
                                    })
                            
                            if problematic_details:
                                self.data_display.styled_dataframe(
                                    pd.DataFrame(problematic_details),
                                    title="Détail des Problèmes",
                                    height=150
                                )
                    else:
                        logger.error(f"❌ Erreur analyse qualité: {report.get('error', 'Erreur inconnue')}")
                        self.layout.error_section(
                            "Erreur lors de l'analyse de qualité",
                            report.get('error', 'Erreur inconnue')
                        )
                        
                    perf_logger.end_operation("image_quality_analysis", f"{len(brightness_values)} images")
                    
                except Exception as e:
                    logger.error(f"❌ Erreur qualité: {e}", exc_info=True)
                    self.layout.error_section(
                        "Erreur lors de l'analyse de qualité",
                        str(e)
                    )

    def _render_image_statistics(self):
        """Statistiques images - Design moderne"""
        logger.debug("Rendu image statistics")
        st.markdown("""
        <div style='
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-left: 1rem;
            border-left: 4px solid #43e97b;
        '>📈 Statistiques Globales</div>
        """, unsafe_allow_html=True)
        
        try:
            d = self.state.data
            
            # Calcul des statistiques avancées
            stats_data = {
                "Luminosité Moyenne": f"{np.mean(d.X):.3f}",
                "Écart-Type Global": f"{np.std(d.X):.3f}",
                "Valeur Minimale": f"{np.min(d.X):.3f}",
                "Valeur Maximale": f"{np.max(d.X):.3f}",
                "Médiane": f"{np.median(d.X):.3f}",
                "Dynamique": f"{np.ptp(d.X):.3f}",
                "Shape des Images": str(d.img_shape),
                "Type de Données": str(d.X.dtype),
                "Plage Dynamique": f"[{np.min(d.X):.1f}, {np.max(d.X):.1f}]"
            }
            
            # Affichage avec design moderne
            self.data_display.key_value_pairs(
                stats_data,
                title="📊 Caractéristiques du Dataset",
                columns=2
            )
            
            # Histogramme des valeurs de pixels
            st.markdown("#### 📈 Distribution des Valeurs de Pixel")
            
            # Échantillonnage pour les performances
            sample_size = min(10000, d.X.size)
            if d.X.size > sample_size:
                flat_sample = np.random.choice(d.X.flatten(), sample_size, replace=False)
            else:
                flat_sample = d.X.flatten()
            
            fig = px.histogram(
                x=flat_sample,
                nbins=50,
                title="Histogramme des Valeurs de Pixel",
                color_discrete_sequence=['#667eea']
            )
            
            fig.update_layout(
                xaxis_title="Valeur de Pixel",
                yaxis_title="Fréquence",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            
            st.plotly_chart(fig, width="stretch")
            
            logger.info("✅ Statistiques images affichées")
            
        except Exception as e:
            logger.error(f"❌ Erreur image statistics: {e}", exc_info=True)
            self.layout.error_section(
                "Erreur lors du calcul des statistiques",
                str(e)
            )

    def _render_image_details(self):
        """Détails du dataset - Design moderne"""
        logger.debug("Rendu image details")
        st.markdown("""
        <div style='
            font-size: 1.8rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            padding-left: 1rem;
            border-left: 4px solid #ee5a6f;
        '>📋 Détails du Dataset</div>
        """, unsafe_allow_html=True)
        
        try:
            stats_df = get_dataset_stats(self.state.data.dir)
            
            if stats_df is not None and not stats_df.empty:
                self.layout.info_badge(
                    f"Dataset analysé avec {len(stats_df)} catégories",
                    badge_type="info",
                    icon="📊"
                )
                
                self.data_display.styled_dataframe(
                    stats_df,
                    title="📁 Structure du Dataset",
                    height=400
                )
                
                # ✅ CORRECTION : Vérification sécurisée des colonnes
                st.markdown("#### 📊 Résumé des Statistiques")
                
                summary_stats = {}
                
                # Vérifier si la colonne 'count' existe
                if 'count' in stats_df.columns:
                    summary_stats["Total Images"] = f"{stats_df['count'].sum():,}"
                    summary_stats["Plus Grande Catégorie"] = f"{stats_df['count'].max():,} images"
                else:
                    summary_stats["Total Images"] = "N/A"
                    summary_stats["Plus Grande Catégorie"] = "N/A"
                
                summary_stats["Catégories"] = f"{len(stats_df):,}"
                
                # Vérifier si la colonne 'size_mb' existe
                if 'size_mb' in stats_df.columns:
                    summary_stats["Taille Moyenne"] = f"{stats_df['size_mb'].mean():.1f} MB"
                else:
                    summary_stats["Taille Moyenne"] = "N/A"
                
                self.data_display.key_value_pairs(summary_stats, columns=2)
                
                logger.info("✅ Détails dataset images affichés")
            else:
                logger.warning("⚠️ Statistiques non disponibles pour dataset images")
                self.layout.info_badge(
                    "Statistiques détaillées non disponibles pour ce dataset",
                    badge_type="warning",
                    icon="⚠️"
                )
                
        except Exception as e:
            logger.error(f"❌ Erreur image details: {e}", exc_info=True)
            self.layout.info_badge(
                f"Informations limitées disponibles: {str(e)}",
                badge_type="warning",
                icon="ℹ️"
            )

    def _render_images_overview(self):
        """Overview images avec badge mode"""
        logger.debug("Rendu images overview")
        perf_logger.start_operation("images_overview")
        
        try:
            d = self.state.data
            
            # ✅ DÉTECTION DU MODE
            if hasattr(d, 'task_metadata') and d.task_metadata:
                task_desc = d.task_metadata.get('description', d.task)
            else:
                task_desc = d.task or 'unknown'
            
            # Badge mode
            if 'unsupervised' in task_desc.lower():
                mode_badge = '<span class="status-badge badge-info">🔍 Unsupervised</span>'
            elif 'anomaly' in task_desc.lower():
                mode_badge = '<span class="status-badge badge-warning">⚠️ Anomaly (Supervised)</span>'
            else:
                mode_badge = '<span class="status-badge badge-success">🎯 Classification</span>'
            
            st.markdown(f"### Mode Détecté\n{mode_badge}", unsafe_allow_html=True)
            
            # Métriques
            mem_mb = d.X.nbytes/(1024**2) if d.X is not None else 0
            shape_str = f"{d.img_shape[1]}×{d.img_shape[2]}" if d.img_shape else "N/A"
            
            self.layout.metric_row([
                {
                    "label": "Images",
                    "value": f"{d.img_count:,}",
                    "icon": "🖼️",
                    "color": "#667eea"
                },
                {
                    "label": "Résolution",
                    "value": shape_str,
                    "icon": "📐",
                    "color": "#f093fb"
                },
                {
                    "label": "Classes",
                    "value": d.n_classes,
                    "icon": "🏷️",
                    "color": "#4facfe"
                },
                {
                    "label": "Mémoire",
                    "value": f"{mem_mb:.1f} MB",
                    "icon": "💾",
                    "color": "#43e97b"
                },
                {
                    "label": "Tâche",
                    "value": task_desc.split('-')[0].strip(),
                    "icon": "🎯",
                    "color": "#feca57"
                }
            ])
            
            perf_logger.end_operation("images_overview", f"{d.img_count} images")
            
        except Exception as e:
            logger.error(f"❌ Erreur images overview: {e}", exc_info=True)
            st.error("Erreur lors du calcul des métriques images")

    def render(self):
        """Rendu complet du dashboard - Version moderne"""
        logger.info("🎬 Début rendu dashboard v1.0")
        perf_logger.start_operation("full_dashboard_render")
        
        # Injection CSS moderne
        self.components.inject_custom_css()
        
        # Vérifier données chargées
        if not self.state.loaded:
            logger.error("❌ Aucun dataset chargé pour le dashboard")
            
            st.markdown("""
            <div style='
                text-align: center;
                padding: 4rem 2rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 20px;
                margin: 2rem 0;
            '>
                <h2 style='color: white; margin-bottom: 1rem;'>📊 Dashboard DataLab Pro</h2>
                <p style='font-size: 1.2rem; opacity: 0.9;'>
                    Aucun dataset n'est actuellement chargé
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🏠 Retour à l'Accueil", type="primary", width="stretch"):
                    logger.info("🏠 Navigation vers accueil depuis dashboard vide")
                    self.state.switch(AppPage.HOME)
            with col2:
                if st.button("🔄 Recharger", type="secondary", width="stretch"):
                    logger.info("🔄 Tentative de rechargement")
                    st.rerun()
            
            return
        
        # Rendu principal
        try:
            self.render_header()
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            self.render_overview()
            st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
            
            # Contenu spécifique au type de données
            if self.state.tabular:
                logger.info("📊 Affichage onglets tabulaires")
                self.render_tabular_tabs()
            elif self.state.images:
                logger.info("🖼️ Affichage onglets images")
                self.render_image_tabs()
            
            # Footer moderne
            self._render_footer()
            
            perf_logger.end_operation("full_dashboard_render", "Dashboard v2.0 complet rendu")
            logger.info("✅ Dashboard v1.0 rendu avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur critique rendu dashboard: {e}", exc_info=True)
            self.layout.error_section(
                "Erreur critique lors du rendu du dashboard",
                str(e),
                suggestion="Veuillez recharger la page ou retourner à l'accueil"
            )
            
            if st.button("🏠 Retour à l'Accueil d'Urgence", type="primary"):
                self.state.switch(AppPage.HOME)

    def _render_footer(self):
        """Footer moderne du dashboard"""
        st.markdown("""
        <div style='
            margin-top: 4rem;
            padding: 2rem 0;
            text-align: center;
            color: #666;
            border-top: 1px solid #e0e0e0;
        '>
            <div style='font-size: 0.9rem;'>
                <strong>DataLab Pro v2.0</strong> • Dashboard Analytique Modern
            </div>
            <div style='font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.7;'>
                Performances optimisées • Design responsive • Production Ready
            </div>
        </div>
        """, unsafe_allow_html=True)