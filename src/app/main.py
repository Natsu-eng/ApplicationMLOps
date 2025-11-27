"""
🏠 DataLab Pro - Page d'Accueil Moderne
Version 1.0 | Design Produit Complet
"""

import streamlit as st
import sys
import os

# Configuration des paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importer le système de logging avant tout autre import
from src.shared.logging import setup_logging, get_logger, log_system_info

# Initialisation du système de logging
setup_logging(
    console_logging=True,  
    mlflow_integration=False  # Désactiver MLflow pour l'instant
)

# Logger les infos système au démarrage
log_system_info()

# Logger pour cette page
logger = get_logger(__name__)
logger.info("=" * 80)
logger.info("🚀 DÉMARRAGE DE L'APPLICATION DATALAB PRO")
logger.info("=" * 80)

# Maintenant importer le reste
from ui.home import ModernHomePage
from monitoring.state_managers import init, AppPage, STATE

# Initialisation
st.set_page_config(
    page_title="DataLab Pro - Plateforme IA Complète",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialiser le state manager
init()

def main():
    """Application principale"""
    logger.info("📄 Rendu de la page d'accueil")
    try:
        # Créer et afficher la page d'accueil moderne
        home_page = ModernHomePage()
        home_page.render()
        logger.info("✅ Page d'accueil rendue avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur critique dans main(): {e}", exc_info=True)
        st.error("Une erreur est survenue. Veuillez recharger la page.")
        st.code(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()