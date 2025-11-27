"""
🎨 Composants UI réutilisables - DataLab Pro
Design system moderne et cohérent
"""

import streamlit as st
from typing import Optional, List, Dict, Any
import streamlit.components.v1 as components

class UIConfig:
    """Configuration du design system"""
    PRIMARY_COLOR = "#667eea"
    SECONDARY_COLOR = "#764ba2"
    SUCCESS_COLOR = "#4facfe"
    WARNING_COLOR = "#43e97b"
    DANGER_COLOR = "#fa709a"
    DARK_COLOR = "#2c3e50"
    LIGHT_COLOR = "#f8f9fa"
    BORDER_RADIUS = "16px"
    SHADOW = "0 8px 30px rgba(0,0,0,0.12)"
    TRANSITION = "all 0.3s ease"

class ModernComponents:
    """Composants UI modernes réutilisables"""
    
    @staticmethod
    def inject_custom_css():
        """Injecte le CSS moderne global"""
        st.markdown(f"""
        <style>
            :root {{
                --primary: {UIConfig.PRIMARY_COLOR};
                --secondary: {UIConfig.SECONDARY_COLOR};
                --success: {UIConfig.SUCCESS_COLOR};
                --warning: {UIConfig.WARNING_COLOR};
                --danger: {UIConfig.DANGER_COLOR};
                --dark: {UIConfig.DARK_COLOR};
                --light: {UIConfig.LIGHT_COLOR};
                --radius: {UIConfig.BORDER_RADIUS};
                --shadow: {UIConfig.SHADOW};
                --transition: {UIConfig.TRANSITION};
            }}
            
            /* Reset et base */
            .main .block-container {{
                padding-top: 2rem;
                max-width: 1200px;
            }}
            
            /* Headers modernes */
            .modern-header {{
                font-size: 3.5rem;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
                text-align: center;
                margin-bottom: 1rem;
                line-height: 1.1;
            }}
            
            .modern-subheader {{
                font-size: 1.4rem;
                color: var(--dark);
                text-align: center;
                margin-bottom: 3rem;
                opacity: 0.8;
                font-weight: 400;
            }}
            
            /* Cards modernes */
            .modern-card {{
                background: white;
                padding: 2.5rem;
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                border: 1px solid rgba(255,255,255,0.2);
                backdrop-filter: blur(10px);
                transition: var(--transition);
                height: 100%;
                display: flex;
                flex-direction: column;
            }}
            
            .modern-card:hover {{
                transform: translateY(-8px);
                box-shadow: 0 20px 40px rgba(0,0,0,0.15);
            }}
            
            .modern-card-icon {{
                font-size: 3rem;
                margin-bottom: 1.5rem;
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            
            .modern-card-title {{
                font-size: 1.4rem;
                font-weight: 700;
                margin-bottom: 1rem;
                color: var(--dark);
            }}
            
            .modern-card-content {{
                color: #666;
                line-height: 1.6;
                flex-grow: 1;
            }}
            
            /* Boutons modernes */
            .modern-btn-primary {{
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
                border: none;
                padding: 1rem 2.5rem;
                border-radius: 50px;
                font-weight: 600;
                font-size: 1.1rem;
                transition: var(--transition);
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                text-align: center;
            }}
            
            .modern-btn-primary:hover {{
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
                color: white;
            }}
            
            .modern-btn-secondary {{
                background: transparent;
                color: var(--primary);
                border: 2px solid var(--primary);
                padding: 1rem 2.5rem;
                border-radius: 50px;
                font-weight: 600;
                font-size: 1.1rem;
                transition: var(--transition);
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                text-align: center;
            }}
            
            .modern-btn-secondary:hover {{
                background: var(--primary);
                color: white;
                transform: translateY(-2px);
            }}
            
            /* Sections */
            .section-header {{
                font-size: 2.5rem;
                font-weight: 700;
                text-align: center;
                margin-bottom: 1rem;
                color: var(--dark);
            }}
            
            .section-subheader {{
                font-size: 1.2rem;
                text-align: center;
                margin-bottom: 4rem;
                color: #666;
                max-width: 600px;
                margin-left: auto;
                margin-right: auto;
            }}
            
            /* Metrics et stats */
            .metric-card {{
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
                padding: 2rem;
                border-radius: var(--radius);
                text-align: center;
                box-shadow: var(--shadow);
            }}
            
            .metric-value {{
                font-size: 2.5rem;
                font-weight: 800;
                margin-bottom: 0.5rem;
            }}
            
            .metric-label {{
                font-size: 1rem;
                opacity: 0.9;
            }}
            
            /* Navigation */
            .nav-pill {{
                background: white;
                padding: 0.5rem 1.5rem;
                border-radius: 50px;
                box-shadow: var(--shadow);
                display: inline-flex;
                gap: 1rem;
                margin-bottom: 2rem;
            }}
            
            .nav-pill-item {{
                padding: 0.5rem 1.5rem;
                border-radius: 50px;
                cursor: pointer;
                transition: var(--transition);
                font-weight: 500;
            }}
            
            .nav-pill-item.active {{
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
            }}
            
            /* Upload zones */
            .upload-zone {{
                border: 3px dashed var(--primary);
                border-radius: var(--radius);
                padding: 4rem 2rem;
                text-align: center;
                background: rgba(102, 126, 234, 0.05);
                transition: var(--transition);
                margin: 2rem 0;
            }}
            
            .upload-zone:hover {{
                background: rgba(102, 126, 234, 0.1);
                border-color: var(--secondary);
            }}
            
            /* Status badges */
            .status-badge {{
                padding: 0.5rem 1.5rem;
                border-radius: 50px;
                font-size: 0.9rem;
                font-weight: 700;
                display: inline-block;
                background: var(--light);
                color: var(--dark);
            }}
            
            .status-badge.success {{
                background: #d4edda;
                color: #155724;
            }}
            
            .status-badge.info {{
                background: #d1ecf1;
                color: #0c5460;
            }}
            
            .status-badge.warning {{
                background: #fff3cd;
                color: #856404;
            }}
            
            /* Hide Streamlit default elements */
            #MainMenu, footer, .stDeployButton {{
                visibility: hidden;
            }}
            
            /* Responsive */
            @media (max-width: 768px) {{
                .modern-header {{
                    font-size: 2.5rem;
                }}
                
                .modern-card {{
                    padding: 2rem;
                }}
            }}
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def hero_section(title: str, subtitle: str, cta_text: str = "Commencer l'analyse"):
        """Section hero moderne"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f'<div class="modern-header">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="modern-subheader">{subtitle}</div>', unsafe_allow_html=True)
            
            # CTA
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button(cta_text, use_container_width=True, type="primary"):
                    st.session_state.nav_action = "dashboard"
        
        with col2:
            # Placeholder pour illustration
            st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
                <div style="font-size: 1.2rem; color: #666;">Data Intelligence Platform</div>
            </div>
            """, unsafe_allow_html=True)
    
    @staticmethod
    def feature_card(icon: str, title: str, description: str, features: List[str]):
        """Carte de fonctionnalité moderne"""
        features_html = "".join([f'<li>{feature}</li>' for feature in features])
        
        st.markdown(f"""
        <div class="modern-card">
            <div class="modern-card-icon">{icon}</div>
            <div class="modern-card-title">{title}</div>
            <div class="modern-card-content">
                <p>{description}</p>
                <ul style="margin-top: 1rem; padding-left: 1.2rem;">
                    {features_html}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def metric_card(value: Any, label: str, icon: str = ""):
        """Carte de métrique moderne"""
        icon_html = f'<div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>' if icon else ""
        
        st.markdown(f"""
        <div class="metric-card">
            {icon_html}
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def section_header(title: str, subtitle: str = ""):
        """En-tête de section"""
        st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<div class="section-subheader">{subtitle}</div>', unsafe_allow_html=True)
    
    @staticmethod
    def cta_section(title: str, subtitle: str, button_text: str, button_type: str = "primary"):
        """Section CTA"""
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f'<div style="text-align: center; margin-bottom: 1rem;"><h3>{title}</h3></div>', unsafe_allow_html=True)
            st.markdown(f'<div style="text-align: center; margin-bottom: 2rem; color: #666;">{subtitle}</div>', unsafe_allow_html=True)
            
            if button_type == "primary":
                if st.button(button_text, use_container_width=True, type="primary"):
                    st.session_state.nav_action = "upload"
            else:
                if st.button(button_text, use_container_width=True):
                    st.session_state.nav_action = "upload"
    
    @staticmethod
    def create_navigation_pill(options: List[Dict[str, str]], default_active: str = ""):
        """Crée une navigation par onglets moderne"""
        pills_html = ""
        for option in options:
            active_class = "active" if option["value"] == default_active else ""
            pills_html += f'<div class="nav-pill-item {active_class}" onclick="window.streamlitNativeAPI.setComponentValue(\'{option["value"]}\')">{option["label"]}</div>'
        
        st.markdown(f'<div class="nav-pill">{pills_html}</div>', unsafe_allow_html=True)
        
        # Gérer l'interaction
        if "nav_selection" in st.session_state:
            return st.session_state.nav_selection
        return default_active

# Export des composants
__all__ = ['ModernComponents', 'UIConfig']