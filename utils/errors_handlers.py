"""
Module de gestion des erreurs robuste pour l'application.
Version optimisée pour production avec logging et reprise intelligente.
"""

import logging
from typing import Callable, Any
from functools import wraps

from src.shared.logging import get_logger

logger = get_logger(__name__)

class ErrorHandler:
    """Gestion centralisée des erreurs avec reprise intelligente"""
    
    @staticmethod
    def safe_execute(default_return=None, max_retries: int = 1):
        """Décorateur pour exécution sécurisée avec reprise"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        logger.warning(f"Tentative {attempt + 1} échouée pour {func.__name__}: {str(e)[:100]}")
                        
                        if attempt < max_retries:
                            continue
                
                logger.error(f"Échec définitif de {func.__name__}: {str(last_exception)}", exc_info=True)
                return default_return
            return wrapper
        return decorator
    
    @staticmethod
    def handle_mlflow_errors(func: Callable) -> Callable:
        """Gestion spécifique des erreurs MLflow"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except ImportError:
                logger.warning("MLflow non disponible - poursuite sans tracking")
                return None
            except Exception as e:
                logger.error(f"Erreur MLflow dans {func.__name__}: {str(e)[:100]}", exc_info=True)
                return None
        return wrapper

@ErrorHandler.safe_execute(default_return=None, max_retries=1)
def safe_train_models(**kwargs):
    """Exécution sécurisée de l'entraînement des modèles."""
    from src.models.training import train_models
    return train_models(**kwargs)

__all__ = ['ErrorHandler', 'safe_train_models']