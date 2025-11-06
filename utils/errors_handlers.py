"""
Module de gestion des erreurs - Version consolidée.

Ce module fournit des fonctions utilitaires utilisant les décorateurs standardisés
de monitoring/decorators.py. Les décorateurs sont maintenant centralisés dans
monitoring/decorators.py pour éviter la duplication.

Pour la compatibilité ascendante, ce module reste disponible mais utilise
les décorateurs standardisés en interne.
"""

from typing import Any, Dict, List, Optional

# Import des décorateurs standardisés
from monitoring.decorators import safe_execute, handle_mlflow_errors
from src.shared.logging import get_logger

logger = get_logger(__name__)

# Déprécié: Utiliser directement safe_execute de monitoring/decorators.py
# Conservé pour compatibilité ascendante
class ErrorHandler:
    """
    Classe de gestion d'erreurs - DÉPRÉCIÉE.
    
    Utiliser directement les décorateurs de monitoring/decorators.py:
    - safe_execute() pour exécution sécurisée
    - handle_mlflow_errors() pour gestion MLflow
    
    Cette classe est conservée uniquement pour compatibilité ascendante.
    """
    
    @staticmethod
    def safe_execute(default_return=None, max_retries: int = 1):
        """
        Décorateur pour exécution sécurisée avec reprise.
        
        DÉPRÉCIÉ: Utiliser directement safe_execute() de monitoring/decorators.py
        
        Args:
            default_return: Valeur de retour en cas d'erreur
            max_retries: Nombre de tentatives supplémentaires
            
        Returns:
            Décorateur
        """
        logger.warning(
            "ErrorHandler.safe_execute est déprécié. "
            "Utiliser directement safe_execute() de monitoring/decorators.py"
        )
        return safe_execute(fallback_value=default_return, max_retries=max_retries)
    
    @staticmethod
    def handle_mlflow_errors(func):
        """
        Gestion spécifique des erreurs MLflow.
        
        DÉPRÉCIÉ: Utiliser directement handle_mlflow_errors() de monitoring/decorators.py
        
        Args:
            func: Fonction à décorer
            
        Returns:
            Fonction wrappée
        """
        logger.warning(
            "ErrorHandler.handle_mlflow_errors est déprécié. "
            "Utiliser directement handle_mlflow_errors() de monitoring/decorators.py"
        )
        return handle_mlflow_errors(func)


@safe_execute(fallback_value=None, max_retries=1)
def safe_train_models(**kwargs):
    """
    Exécution sécurisée de l'entraînement des modèles.
    
    Cette fonction utilise le décorateur safe_execute standardisé pour
    gérer les erreurs et permettre les retries.
    
    Args:
        **kwargs: Arguments passés à train_models()
        
    Returns:
        Résultats de l'entraînement ou None en cas d'erreur
    """
    from src.models.training import train_models
    return train_models(**kwargs)


__all__ = ['ErrorHandler', 'safe_train_models']