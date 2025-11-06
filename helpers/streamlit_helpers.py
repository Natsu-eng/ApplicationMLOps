"""
Helpers spécifiques à Streamlit pour le caching et la gestion d'état.
"""
import functools
from typing import Any, Callable
from src.shared.logging import get_logger

logger = get_logger(__name__)

# Variable globale pour la disponibilité de Streamlit
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit non disponible, caching désactivé")

def conditional_cache(use_cache: bool = True, ttl: int = 300, show_spinner: bool = False) -> Callable:
    """
    Décorateur de cache conditionnel pour Streamlit.
    
    Args:
        use_cache: Si True, active le caching
        ttl: Time to live en secondes
        show_spinner: Afficher un spinner pendant le chargement
        
    Returns:
        Décorateur
    """
    def decorator(func: Callable) -> Callable:
        if STREAMLIT_AVAILABLE and use_cache:
            return st.cache_data(ttl=ttl, show_spinner=show_spinner)(func)
        return func
    return decorator

def safe_store_in_session(key: str, value: Any) -> bool:
    """
    Stocke une valeur dans session_state de manière sécurisée.
    
    Args:
        key: Clé de stockage
        value: Valeur à stocker
        
    Returns:
        True si succès, False sinon
    """
    if not STREAMLIT_AVAILABLE:
        return False
        
    try:
        st.session_state[key] = value
        logger.debug(f"✅ Valeur stockée dans session_state: {key}")
        return True
    except Exception as e:
        logger.error(f"❌ Erreur stockage session_state {key}: {e}")
        return False

def get_from_session(key: str, default: Any = None) -> Any:
    """
    Récupère une valeur de session_state de manière sécurisée.
    
    Args:
        key: Clé de récupération
        default: Valeur par défaut si non trouvée
        
    Returns:
        Valeur stockée ou default
    """
    if not STREAMLIT_AVAILABLE:
        return default
        
    try:
        return st.session_state.get(key, default)
    except Exception as e:
        logger.error(f"❌ Erreur récupération session_state {key}: {e}")
        return default