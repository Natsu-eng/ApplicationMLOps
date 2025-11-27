"""
📦 Composants de sections réutilisables
"""

import streamlit as st
from typing import List, Dict, Any, Optional


class SectionComponents:
    """Composants de sections modernes"""
    
    @staticmethod
    def feature_grid(
        features: List[Dict[str, Any]],
        title: str = "",
        subtitle: str = ""
    ):
        """
        Grille de fonctionnalités moderne
        
        Args:
            features: Liste de {icon, title, description, features, color}
            title: Titre de la section
            subtitle: Sous-titre
        """
        if title:
            st.markdown(f"""
            <div style='text-align: center; margin-bottom: 3rem;'>
                <h2 style='font-size: 2.5rem; font-weight: 700; color: #2c3e50; margin-bottom: 1rem;'>
                    {title}
                </h2>
                {f"<p style='font-size: 1.2rem; color: #666;'>{subtitle}</p>" if subtitle else ""}
            </div>
            """, unsafe_allow_html=True)
        
        # Grille de fonctionnalités
        cols = st.columns(len(features))
        
        for i, (col, feature) in enumerate(zip(cols, features)):
            with col:
                color = feature.get('color', '#667eea')
                icon = feature.get('icon', '📊')
                
                # Liste des features
                features_html = ""
                if 'features' in feature:
                    features_html = "<ul style='padding-left: 1.2rem; margin-top: 1rem;'>"
                    for f in feature['features']:
                        features_html += f"<li style='margin-bottom: 0.5rem; color: #666;'>{f}</li>"
                    features_html += "</ul>"
                
                st.markdown(f"""
                <div style='
                    background: white;
                    padding: 2rem;
                    border-radius: 16px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                    height: 100%;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                    border-top: 4px solid {color};
                ' onmouseover="this.style.transform='translateY(-8px)'; this.style.boxShadow='0 12px 40px rgba(0,0,0,0.15)';"
                  onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 20px rgba(0,0,0,0.08)';">
                    <div style='font-size: 3rem; margin-bottom: 1rem; text-align: center;'>{icon}</div>
                    <div style='
                        font-size: 1.3rem;
                        font-weight: 700;
                        color: #2c3e50;
                        margin-bottom: 1rem;
                        text-align: center;
                    '>{feature['title']}</div>
                    <div style='
                        font-size: 1rem;
                        color: #666;
                        line-height: 1.6;
                        text-align: center;
                        margin-bottom: 1rem;
                    '>{feature['description']}</div>
                    {features_html}
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def stats_section(
        stats: List[Dict[str, Any]],
        title: str = "",
        layout: str = "horizontal"
    ):
        """
        Section de statistiques
        
        Args:
            stats: Liste de {value, label, icon, color}
            title: Titre de section
            layout: "horizontal" ou "grid"
        """
        if title:
            st.markdown(f"""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h3 style='font-size: 2rem; font-weight: 700; color: #2c3e50;'>{title}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        if layout == "horizontal":
            cols = st.columns(len(stats))
        else:
            cols = st.columns(2)
        
        for i, stat in enumerate(stats):
            col_idx = i if layout == "horizontal" else i % 2
            with cols[col_idx]:
                color = stat.get('color', '#667eea')
                icon = stat.get('icon', '📊')
                
                st.markdown(f"""
                <div style='
                    background: linear-gradient(135deg, {color}, {color}dd);
                    color: white;
                    padding: 2rem;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
                    margin-bottom: 1rem;
                '>
                    <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>{icon}</div>
                    <div style='font-size: 2.5rem; font-weight: 800; margin-bottom: 0.5rem;'>{stat['value']}</div>
                    <div style='font-size: 1rem; opacity: 0.9;'>{stat['label']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    @staticmethod
    def comparison_table(
        data: List[Dict[str, Any]],
        title: str = "",
        headers: List[str] = []
    ):
        """
        Tableau de comparaison moderne
        
        Args:
            data: Liste de dictionnaires
            title: Titre
            headers: En-têtes personnalisées
        """
        if title:
            st.markdown(f"""
            <div style='
                font-size: 1.5rem;
                font-weight: 700;
                color: #2c3e50;
                margin-bottom: 1.5rem;
                text-align: center;
            '>{title}</div>
            """, unsafe_allow_html=True)
        
        if not data:
            st.info("Aucune donnée à afficher")
            return
        
        # Auto-detect headers si non fournis
        if not headers and data:
            headers = list(data[0].keys())
        
        # En-tête du tableau
        header_html = "<tr style='background: linear-gradient(135deg, #667eea, #764ba2); color: white;'>"
        for header in headers:
            header_html += f"<th style='padding: 1rem; text-align: left; font-weight: 600;'>{header}</th>"
        header_html += "</tr>"
        
        # Lignes du tableau
        rows_html = ""
        for i, row in enumerate(data):
            bg_color = "#f8f9fa" if i % 2 == 0 else "white"
            rows_html += f"<tr style='background: {bg_color};'>"
            for header in headers:
                value = row.get(header, "—")
                rows_html += f"<td style='padding: 1rem; border-bottom: 1px solid #dee2e6;'>{value}</td>"
            rows_html += "</tr>"
        
        st.markdown(f"""
        <div style='overflow-x: auto; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08);'>
            <table style='width: 100%; border-collapse: collapse;'>
                <thead>{header_html}</thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def accordion_section(
        items: List[Dict[str, Any]],
        title: str = ""
    ):
        """
        Section accordéon moderne
        
        Args:
            items: Liste de {title, content, icon}
            title: Titre de section
        """
        if title:
            st.markdown(f"### {title}")
        
        for item in items:
            icon = item.get('icon', '▶️')
            with st.expander(f"{icon} {item['title']}", expanded=False):
                if callable(item['content']):
                    item['content']()
                else:
                    st.markdown(item['content'])
    
    @staticmethod
    def timeline(
        events: List[Dict[str, Any]],
        title: str = ""
    ):
        """
        Timeline visuelle
        
        Args:
            events: Liste de {title, description, icon, color}
            title: Titre
        """
        if title:
            st.markdown(f"""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h3 style='font-size: 2rem; font-weight: 700;'>{title}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        for i, event in enumerate(events):
            color = event.get('color', '#667eea')
            icon = event.get('icon', '●')
            
            st.markdown(f"""
            <div style='
                display: flex;
                align-items: flex-start;
                margin-bottom: 2rem;
            '>
                <div style='
                    min-width: 60px;
                    height: 60px;
                    background: {color};
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.5rem;
                    color: white;
                    margin-right: 1.5rem;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                '>{icon}</div>
                <div style='flex: 1;'>
                    <div style='
                        font-size: 1.2rem;
                        font-weight: 700;
                        color: #2c3e50;
                        margin-bottom: 0.5rem;
                    '>{event['title']}</div>
                    <div style='
                        font-size: 1rem;
                        color: #666;
                        line-height: 1.6;
                    '>{event['description']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Ligne de connexion (sauf pour le dernier)
            if i < len(events) - 1:
                st.markdown(f"""
                <div style='
                    width: 3px;
                    height: 30px;
                    background: {color};
                    margin-left: 28px;
                    margin-bottom: 0.5rem;
                    opacity: 0.3;
                '></div>
                """, unsafe_allow_html=True)


__all__ = ['SectionComponents']