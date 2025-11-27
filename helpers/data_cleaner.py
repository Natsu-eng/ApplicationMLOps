"""
🧹 Data Cleaner - Gestionnaire de Nettoyage Interactif
Version: 1.0.0 | Production Ready
Auteur: DataLab Team

Fonctionnalités:
- Détection automatique de colonnes inutiles
- Suppression interactive avec prévisualisation
- Historique des modifications (undo/redo)
- Synchronisation avec STATE
- Export de rapport de nettoyage
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from src.data.data_analysis import detect_useless_columns, get_all_problematic_columns
from monitoring.state_managers import STATE
from src.shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningAction:
    """Représente une action de nettoyage"""
    timestamp: str
    action_type: str  # 'remove_columns', 'rename_column', 'fill_missing', etc.
    details: Dict
    df_shape_before: Tuple[int, int]
    df_shape_after: Tuple[int, int]


class DataCleaner:
    """Gestionnaire de nettoyage de données avec historique"""
    
    def __init__(self):
        """Initialise le cleaner"""
        if 'cleaning_history' not in st.session_state:
            st.session_state.cleaning_history = []
        
        if 'cleaning_undo_stack' not in st.session_state:
            st.session_state.cleaning_undo_stack = []
    
    def detect_problems(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Détecte les problèmes dans le DataFrame.
        
        Args:
            df: DataFrame à analyser
            
        Returns:
            Dict avec catégories de problèmes
        """
        try:
            problems = detect_useless_columns(df)
            logger.info(f"Problèmes détectés: {sum(len(v) for v in problems.values())} colonnes")
            return problems
        
        except Exception as e:
            logger.error(f"Erreur détection problèmes: {e}")
            return {}
    
    def render_problem_summary(self, problems: Dict[str, List[str]]) -> None:
        """
        Affiche un résumé des problèmes détectés.
        
        Args:
            problems: Dict des problèmes
        """
        if not problems:
            st.success("✅ Aucun problème détecté dans ce dataset!")
            return
        
        total = sum(len(cols) for cols in problems.values())
        
        st.warning(f"⚠️ {total} colonnes problématiques détectées")
        
        # Expansion par catégorie
        for category, columns in problems.items():
            if not columns:
                continue
            
            category_label = self._get_category_label(category)
            icon = self._get_category_icon(category)
            
            with st.expander(f"{icon} {category_label} ({len(columns)} colonnes)", expanded=True):
                st.write(f"**Colonnes:** {', '.join([f'`{c}`' for c in columns[:10]])}")
                
                if len(columns) > 10:
                    st.caption(f"... et {len(columns) - 10} autres")
                
                # Description du problème
                st.info(self._get_category_description(category))
    
    def render_removal_interface(self, df: pd.DataFrame, problems: Dict[str, List[str]]) -> None:
        """
        Interface interactive pour supprimer des colonnes.
        
        Args:
            df: DataFrame actuel
            problems: Problèmes détectés
        """
        st.markdown("### 🗑️ Suppression de Colonnes")
        
        all_problematic = get_all_problematic_columns(problems)
        
        if not all_problematic:
            st.info("Aucune colonne problématique à supprimer")
            return
        
        # Sélection des colonnes
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selection_mode = st.radio(
                "Mode de sélection",
                ["Toutes les colonnes problématiques", "Sélection par catégorie", "Sélection manuelle"],
                horizontal=True
            )
        
        with col2:
            preview_mode = st.checkbox("Prévisualisation avant suppression", value=True)
        
        # Définir les colonnes à supprimer selon le mode
        columns_to_remove = []
        
        if selection_mode == "Toutes les colonnes problématiques":
            columns_to_remove = all_problematic
            st.info(f"📊 {len(columns_to_remove)} colonnes sélectionnées")
        
        elif selection_mode == "Sélection par catégorie":
            st.markdown("#### Sélectionnez les catégories à supprimer:")
            
            selected_categories = []
            for category, columns in problems.items():
                if columns:
                    icon = self._get_category_icon(category)
                    label = self._get_category_label(category)
                    
                    if st.checkbox(f"{icon} {label} ({len(columns)} colonnes)", key=f"cat_{category}"):
                        selected_categories.append(category)
            
            # Collecter toutes les colonnes des catégories sélectionnées
            for category in selected_categories:
                columns_to_remove.extend(problems.get(category, []))
            
            if columns_to_remove:
                st.info(f"📊 {len(columns_to_remove)} colonnes sélectionnées")
        
        else:  # Sélection manuelle
            columns_to_remove = st.multiselect(
                "Sélectionnez les colonnes à supprimer",
                options=df.columns.tolist(),
                default=all_problematic[:5],  # 5 premières par défaut
                help="Vous pouvez sélectionner n'importe quelles colonnes"
            )
        
        # Prévisualisation
        if preview_mode and columns_to_remove:
            st.markdown("---")
            st.markdown("#### 👁️ Prévisualisation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Colonnes actuelles", len(df.columns))
            
            with col2:
                st.metric("Colonnes à supprimer", len(columns_to_remove))
            
            with col3:
                st.metric("Colonnes restantes", len(df.columns) - len(columns_to_remove))
            
            # Tableau des colonnes
            with st.expander("📋 Détails des colonnes à supprimer", expanded=False):
                removal_info = []
                
                for col in columns_to_remove:
                    info = {
                        'Colonne': col,
                        'Type': str(df[col].dtype),
                        'Valeurs Uniques': df[col].nunique(),
                        'Manquant (%)': f"{df[col].isnull().mean() * 100:.1f}%"
                    }
                    removal_info.append(info)
                
                st.dataframe(pd.DataFrame(removal_info), use_container_width=True)
        
        # Bouton de suppression
        if columns_to_remove:
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button(
                    f"🗑️ Supprimer {len(columns_to_remove)} colonne(s)",
                    type="primary",
                    use_container_width=True
                ):
                    self._execute_removal(df, columns_to_remove)
    
    def _execute_removal(self, df: pd.DataFrame, columns: List[str]) -> None:
        """
        Exécute la suppression de colonnes.
        
        Args:
            df: DataFrame actuel
            columns: Colonnes à supprimer
        """
        try:
            # Sauvegarde pour undo
            st.session_state.cleaning_undo_stack.append(df.copy())
            
            # Suppression
            shape_before = df.shape
            new_df = df.drop(columns=columns)
            shape_after = new_df.shape
            
            # Enregistrer l'action
            action = CleaningAction(
                timestamp=datetime.now().isoformat(),
                action_type='remove_columns',
                details={'columns': columns},
                df_shape_before=shape_before,
                df_shape_after=shape_after
            )
            
            st.session_state.cleaning_history.append(action)
            
            # Mettre à jour STATE
            STATE.set_tabular(new_df, STATE.data.df_raw, STATE.data.name)
            
            # Feedback
            st.success(f"✅ {len(columns)} colonne(s) supprimée(s) avec succès!")
            st.info(f"📊 Dataset mis à jour: {shape_before} → {shape_after}")
            
            logger.info(f"Colonnes supprimées: {columns}")
            
            # Rerun pour mise à jour
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Erreur lors de la suppression: {str(e)}")
            logger.error(f"Erreur suppression colonnes: {e}", exc_info=True)
    
    def render_undo_interface(self) -> None:
        """Interface pour annuler les modifications"""
        if not st.session_state.cleaning_undo_stack:
            st.info("Aucune modification à annuler")
            return
        
        st.markdown("### ↩️ Annuler les Modifications")
        
        n_undo = len(st.session_state.cleaning_undo_stack)
        st.write(f"**{n_undo} version(s) précédente(s) disponible(s)**")
        
        if st.button("↩️ Annuler la dernière modification", type="secondary"):
            try:
                # Restaurer la version précédente
                previous_df = st.session_state.cleaning_undo_stack.pop()
                
                STATE.set_tabular(previous_df, STATE.data.df_raw, STATE.data.name)
                
                st.success("✅ Modification annulée!")
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Erreur lors de l'annulation: {str(e)}")
    
    def render_history(self) -> None:
        """Affiche l'historique des modifications"""
        if not st.session_state.cleaning_history:
            st.info("Aucune modification effectuée")
            return
        
        st.markdown("### 📜 Historique des Modifications")
        
        for i, action in enumerate(reversed(st.session_state.cleaning_history), 1):
            timestamp = datetime.fromisoformat(action.timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            with st.expander(f"#{len(st.session_state.cleaning_history) - i + 1} - {action.action_type} - {timestamp}"):
                st.write(f"**Type:** {action.action_type}")
                st.write(f"**Shape avant:** {action.df_shape_before}")
                st.write(f"**Shape après:** {action.df_shape_after}")
                st.write(f"**Détails:** {action.details}")
    
    def export_cleaning_report(self) -> str:
        """
        Génère un rapport de nettoyage en JSON.
        
        Returns:
            Rapport JSON
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_actions': len(st.session_state.cleaning_history),
            'actions': [
                {
                    'timestamp': action.timestamp,
                    'type': action.action_type,
                    'details': action.details,
                    'shape_before': action.df_shape_before,
                    'shape_after': action.df_shape_after
                }
                for action in st.session_state.cleaning_history
            ]
        }
        
        return json.dumps(report, indent=2)
    
    # Helpers pour catégories
    
    @staticmethod
    def _get_category_label(category: str) -> str:
        """Retourne le label lisible d'une catégorie"""
        labels = {
            'high_missing': 'Trop de Valeurs Manquantes',
            'constant': 'Colonnes Constantes',
            'quasi_constant': 'Colonnes Quasi-Constantes',
            'low_variance': 'Faible Variance',
            'high_cardinality': 'Cardinalité Trop Élevée',
            'single_value': 'Une Seule Valeur Non-Nulle'
        }
        return labels.get(category, category.replace('_', ' ').title())
    
    @staticmethod
    def _get_category_icon(category: str) -> str:
        """Retourne l'icône d'une catégorie"""
        icons = {
            'high_missing': '🕳️',
            'constant': '⚪',
            'quasi_constant': '🔘',
            'low_variance': '📉',
            'high_cardinality': '📊',
            'single_value': '1️⃣'
        }
        return icons.get(category, '❓')
    
    @staticmethod
    def _get_category_description(category: str) -> str:
        """Retourne la description d'une catégorie"""
        descriptions = {
            'high_missing': "Ces colonnes contiennent trop de valeurs manquantes (>80%), ce qui les rend peu utiles pour l'analyse.",
            'constant': "Ces colonnes ont toujours la même valeur, elles n'apportent aucune information.",
            'quasi_constant': "Ces colonnes ont une valeur dominante (>80%), peu de variabilité.",
            'low_variance': "Ces colonnes numériques ont une variance très faible, peu d'information.",
            'high_cardinality': "Ces colonnes catégorielles ont trop de valeurs uniques, difficiles à utiliser.",
            'single_value': "Ces colonnes n'ont qu'une seule valeur non-nulle malgré beaucoup de NaN."
        }
        return descriptions.get(category, "Catégorie non documentée.")


# Export
__all__ = ['DataCleaner', 'CleaningAction']