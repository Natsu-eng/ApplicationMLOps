"""
📐 Composants de layout responsive
"""

import streamlit as st
from typing import Dict, List, Any, Callable, Optional

class LayoutComponents:
    """Composants de layout avancés"""
    
    @staticmethod
    def responsive_grid(items: List[Any], items_per_row: int = 3, gap: str = "medium"):
        """
        Grille responsive pour cartes
        
        Args:
            items: Liste d'éléments à afficher
            items_per_row: Nombre d'items par ligne
            gap: Espacement ("small", "medium", "large")
        """
        for i in range(0, len(items), items_per_row):
            cols = st.columns(items_per_row, gap=gap)
            for j, col in enumerate(cols):
                if i + j < len(items):
                    with col:
                        if callable(items[i + j]):
                            items[i + j]()
                        else:
                            st.write(items[i + j])
    
    @staticmethod
    def split_section(
        left_content: Callable, 
        right_content: Callable, 
        ratio: List[int] = [1, 1],
        vertical_alignment: str = "top"
    ):
        """
        Section splitée verticalement
        
        Args:
            left_content: Fonction pour contenu gauche
            right_content: Fonction pour contenu droite
            ratio: Ratio de colonnes [gauche, droite]
            vertical_alignment: Alignement vertical
        """
        left_col, right_col = st.columns(ratio)
        with left_col:
            left_content()
        with right_col:
            right_content()
    
    @staticmethod
    def tabbed_interface(tabs_config: List[Dict[str, Any]]):
        """
        Interface à onglets moderne
        
        Args:
            tabs_config: Liste de configs avec {title, icon, content}
        """
        tab_titles = [f"{tab.get('icon', '')} {tab['title']}" for tab in tabs_config]
        tabs = st.tabs(tab_titles)
        
        for i, tab_container in enumerate(tabs):
            with tab_container:
                tabs_config[i]["content"]()
    
    @staticmethod
    def metric_row(metrics: List[Dict[str, Any]]):
        """
        Ligne de métriques modernes
        
        Args:
            metrics: Liste de {label, value, icon, delta, color}
        """
        cols = st.columns(len(metrics))
        
        for i, (col, metric) in enumerate(zip(cols, metrics)):
            with col:
                icon = metric.get('icon', '📊')
                color = metric.get('color', '#667eea')
                
                st.markdown(f"""
                <div style='
                    background: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
                    text-align: center;
                    border-left: 4px solid {color};
                '>
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>{icon}</div>
                    <div style='font-size: 0.85rem; color: #666; margin-bottom: 0.25rem;'>{metric['label']}</div>
                    <div style='font-size: 1.8rem; font-weight: 800; color: {color};'>{metric['value']}</div>
                    {f"<div style='font-size: 0.75rem; color: #28a745; margin-top: 0.25rem;'>▲ {metric.get('delta', '')}</div>" if metric.get('delta') else ""}
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def card_container(
        title: str, 
        content: Callable,
        icon: str = "📊",
        color: str = "#667eea",
        collapsible: bool = False
    ):
        """
        Container de carte moderne
        
        Args:
            title: Titre de la carte
            content: Fonction de contenu
            icon: Icône
            color: Couleur d'accent
            collapsible: Si pliable
        """
        st.markdown(f"""
        <div style='
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border-top: 4px solid {color};
        '>
            <div style='
                display: flex;
                align-items: center;
                margin-bottom: 1.5rem;
            '>
                <div style='font-size: 2rem; margin-right: 1rem;'>{icon}</div>
                <div style='font-size: 1.5rem; font-weight: 700; color: #2c3e50;'>{title}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if collapsible:
            with st.expander("Voir les détails", expanded=True):
                content()
        else:
            content()
    
    @staticmethod
    def info_badge(
        text: str, 
        badge_type: str = "info",
        icon: str = "ℹ️"
    ):
        """
        Badge d'information moderne
        
        Args:
            text: Texte du badge
            badge_type: Type (info, success, warning, error)
            icon: Icône
        """
        colors = {
            "info": {"bg": "#d1ecf1", "text": "#0c5460", "border": "#bee5eb"},
            "success": {"bg": "#d4edda", "text": "#155724", "border": "#c3e6cb"},
            "warning": {"bg": "#fff3cd", "text": "#856404", "border": "#ffeaa7"},
            "error": {"bg": "#f8d7da", "text": "#721c24", "border": "#f5c6cb"}
        }
        
        style = colors.get(badge_type, colors["info"])
        
        st.markdown(f"""
        <div style='
            background: {style["bg"]};
            color: {style["text"]};
            border: 1px solid {style["border"]};
            border-radius: 8px;
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            display: flex;
            align-items: center;
        '>
            <div style='font-size: 1.5rem; margin-right: 1rem;'>{icon}</div>
            <div>{text}</div>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def error_section(title: str, description: str, suggestion: str = ""):
        """
        Section d'erreur moderne
        
        Args:
            title: Titre de l'erreur
            description: Description de l'erreur  
            suggestion: Suggestion pour résoudre (optionnel)
        """
        st.markdown(f"""
        <div style='
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            border-radius: 12px;
            padding: 2rem;
            margin: 1.5rem 0;
        '>
            <div style='
                display: flex;
                align-items: center;
                margin-bottom: 1rem;
            '>
                <div style='font-size: 2rem; margin-right: 1rem;'>❌</div>
                <div style='font-size: 1.5rem; font-weight: 700;'>{title}</div>
            </div>
            <div style='margin-bottom: 1rem;'>{description}</div>
            {f"<div><strong>Suggestion :</strong> {suggestion}</div>" if suggestion else ""}
        </div>
        """, unsafe_allow_html=True)


class DataDisplayComponents:
    """Composants d'affichage de données modernes"""
    
    @staticmethod
    def styled_dataframe(
        df, 
        title: Optional[str] = None,
        height: int = 400,
        highlight_cols: List[str] = []
    ):
        """
        DataFrame avec style moderne
        
        Args:
            df: DataFrame à afficher
            title: Titre optionnel
            height: Hauteur
            highlight_cols: Colonnes à surligner
        """
        if title:
            st.markdown(f"""
            <div style='
                font-size: 1.2rem;
                font-weight: 700;
                color: #2c3e50;
                margin-bottom: 1rem;
                padding-left: 0.5rem;
                border-left: 4px solid #667eea;
            '>{title}</div>
            """, unsafe_allow_html=True)
        
        # Utiliser width au lieu de use_container_width
        st.dataframe(
            df,
            height=height,
            use_container_width=True
        )
    
    @staticmethod
    def key_value_pairs(
        data: Dict[str, Any],
        title: Optional[str] = None,
        columns: int = 2
    ):
        """
        Affichage clé-valeur moderne
        
        Args:
            data: Dictionnaire de données
            title: Titre optionnel
            columns: Nombre de colonnes
        """
        if title:
            st.markdown(f"### {title}")
        
        items = list(data.items())
        for i in range(0, len(items), columns):
            cols = st.columns(columns)
            for j, col in enumerate(cols):
                if i + j < len(items):
                    key, value = items[i + j]
                    with col:
                        st.markdown(f"""
                        <div style='
                            background: #f8f9fa;
                            padding: 1rem;
                            border-radius: 8px;
                            margin-bottom: 0.5rem;
                        '>
                            <div style='
                                font-size: 0.85rem;
                                color: #666;
                                margin-bottom: 0.25rem;
                            '>{key}</div>
                            <div style='
                                font-size: 1.1rem;
                                font-weight: 600;
                                color: #2c3e50;
                            '>{value}</div>
                        </div>
                        """, unsafe_allow_html=True)
    
    @staticmethod
    def progress_indicator(
        label: str,
        current: int,
        total: int,
        color: str = "#667eea"
    ):
        """
        Indicateur de progression moderne
        
        Args:
            label: Label
            current: Valeur actuelle
            total: Valeur totale
            color: Couleur
        """
        percentage = (current / total * 100) if total > 0 else 0
        
        st.markdown(f"""
        <div style='margin-bottom: 1rem;'>
            <div style='
                display: flex;
                justify-content: space-between;
                margin-bottom: 0.5rem;
            '>
                <span style='font-weight: 600;'>{label}</span>
                <span style='color: {color}; font-weight: 700;'>{current}/{total}</span>
            </div>
            <div style='
                width: 100%;
                height: 8px;
                background: #e9ecef;
                border-radius: 4px;
                overflow: hidden;
            '>
                <div style='
                    width: {percentage}%;
                    height: 100%;
                    background: {color};
                    transition: width 0.3s ease;
                '></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


__all__ = [
    'LayoutComponents',
    'DataDisplayComponents'
]