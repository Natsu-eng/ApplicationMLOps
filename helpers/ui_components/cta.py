"""
🎯 Hero Component - Section héro moderne pour landing pages
Version: 1.0.0
"""

import streamlit as st
from typing import Optional, Callable


def render_hero(
    title: str,
    subtitle: str,
    cta_text: str = "Commencer",
    cta_callback: Optional[Callable] = None,
    image_url: Optional[str] = None,
    background_gradient: tuple = ("#667eea", "#764ba2")
):
    """
    Affiche une section héro moderne avec titre, sous-titre, CTA
    
    Args:
        title: Titre principal
        subtitle: Sous-titre descriptif
        cta_text: Texte du bouton CTA
        cta_callback: Fonction à appeler au clic
        image_url: URL image ou None
        background_gradient: Tuple de couleurs pour gradient
    """
    
    st.markdown(f"""
    <style>
        .hero-container {{
            background: linear-gradient(135deg, {background_gradient[0]}, {background_gradient[1]});
            padding: 4rem 2rem;
            border-radius: 24px;
            margin-bottom: 3rem;
            color: white;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0,0,0,0.15);
        }}
        .hero-title {{
            font-size: 3.5rem;
            font-weight: 900;
            margin-bottom: 1rem;
            line-height: 1.2;
            text-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .hero-subtitle {{
            font-size: 1.4rem;
            font-weight: 400;
            opacity: 0.95;
            margin-bottom: 2rem;
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
        }}
        .hero-cta {{
            background: white;
            color: {background_gradient[0]};
            padding: 1rem 3rem;
            border-radius: 50px;
            font-size: 1.2rem;
            font-weight: 700;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}
        .hero-cta:hover {{
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(0,0,0,0.3);
        }}
    </style>
    
    <div class="hero-container">
        <h1 class="hero-title">{title}</h1>
        <p class="hero-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # CTA Button (Streamlit native pour callback)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button(cta_text, key="hero_cta", use_container_width=True, type="primary"):
            if cta_callback:
                cta_callback()


def render_feature_grid(features: list):
    """
    Affiche une grille de features modernes
    
    Args:
        features: Liste de dicts avec {icon, title, description}
    """
    
    st.markdown("""
    <style>
        .feature-card {{
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            text-align: center;
            transition: all 0.3s ease;
            height: 100%;
        }}
        .feature-card:hover {{
            transform: translateY(-8px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        }}
        .feature-icon {{
            font-size: 3rem;
            margin-bottom: 1rem;
        }}
        .feature-title {{
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            color: #2c3e50;
        }}
        .feature-description {{
            font-size: 1rem;
            color: #6c757d;
            line-height: 1.6;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Grille responsive
    cols_per_row = 3 if len(features) % 3 == 0 else 4
    
    for i in range(0, len(features), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(features):
                feature = features[idx]
                with col:
                    st.markdown(f"""
                    <div class="feature-card">
                        <div class="feature-icon">{feature['icon']}</div>
                        <div class="feature-title">{feature['title']}</div>
                        <div class="feature-description">{feature['description']}</div>
                    </div>
                    """, unsafe_allow_html=True)


def render_value_proposition(values: list):
    """
    Affiche les valeurs ajoutées sous forme de cartes larges
    
    Args:
        values: Liste de dicts avec {icon, title, description, color}
    """
    
    st.markdown("""
    <style>
        .value-card {{
            background: white;
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            border-left: 6px solid;
            transition: all 0.3s ease;
        }}
        .value-card:hover {{
            transform: translateX(10px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        }}
        .value-header {{
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }}
        .value-icon {{
            font-size: 2.5rem;
            margin-right: 1rem;
        }}
        .value-title {{
            font-size: 1.8rem;
            font-weight: 800;
            color: #2c3e50;
        }}
        .value-description {{
            font-size: 1.1rem;
            color: #6c757d;
            line-height: 1.7;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    for value in values:
        color = value.get('color', '#667eea')
        st.markdown(f"""
        <div class="value-card" style="border-left-color: {color};">
            <div class="value-header">
                <div class="value-icon">{value['icon']}</div>
                <div class="value-title">{value['title']}</div>
            </div>
            <div class="value-description">{value['description']}</div>
        </div>
        """, unsafe_allow_html=True)


def render_workflow_steps(steps: list):
    """
    Affiche un workflow en étapes avec connecteurs visuels
    
    Args:
        steps: Liste de dicts avec {number, title, description, icon}
    """
    
    st.markdown("""
    <style>
        .workflow-container {{
            display: flex;
            justify-content: space-between;
            align-items: stretch;
            margin: 2rem 0;
            gap: 1rem;
        }}
        .workflow-step {{
            flex: 1;
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            text-align: center;
            position: relative;
        }}
        .step-number {{
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            font-weight: 800;
            margin: 0 auto 1rem;
        }}
        .step-icon {{
            font-size: 2rem;
            margin-bottom: 0.75rem;
        }}
        .step-title {{
            font-size: 1.2rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 0.5rem;
        }}
        .step-description {{
            font-size: 0.95rem;
            color: #6c757d;
            line-height: 1.5;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns(len(steps))
    
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            st.markdown(f"""
            <div class="workflow-step">
                <div class="step-number">{step['number']}</div>
                <div class="step-icon">{step['icon']}</div>
                <div class="step-title">{step['title']}</div>
                <div class="step-description">{step['description']}</div>
            </div>
            """, unsafe_allow_html=True)


__all__ = [
    'render_hero',
    'render_feature_grid',
    'render_value_proposition',
    'render_workflow_steps'
]