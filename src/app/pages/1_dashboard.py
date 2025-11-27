"""
📊 DataLab Pro - Dashboard Moderne
Version 1.0 | Analytics Complète
"""

import streamlit as st
import sys
import os

# Configuration des paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.shared.logging import setup_logging, get_logger

# Initialiser le système de logging
setup_logging(console_logging=True, mlflow_integration=False)

# Logger pour cette page
logger = get_logger(__name__)
logger.info("=" * 80)
logger.info("📊 DÉMARRAGE DU DASHBOARD")
logger.info("=" * 80)

# Maintenant importer le reste
from ui.dashbord import ModernDashboard
from monitoring.state_managers import init, AppPage, STATE

# Configuration de la page
st.set_page_config(
    page_title="DataLab Pro | Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisation
init()

def main():
    """Dashboard principal"""
    logger.info("📊 Rendu du dashboard")
    try:
        # Créer et afficher le dashboard moderne
        dashboard = ModernDashboard()
        dashboard.render()
        logger.info("✅ Dashboard rendu avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur critique dans dashboard: {e}", exc_info=True)
        st.error("Erreur dans le dashboard. Retour à l'accueil.")
        if st.button("🏠 Accueil"):
            STATE.switch(AppPage.HOME)
        st.code(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()