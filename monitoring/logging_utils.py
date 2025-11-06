"""
Utilitaires de logging structuré pour l'application.
Version consolidée - utilise le système de logging centralisé.
"""
import json
import threading
from datetime import datetime
from typing import Dict, Any

# Import du système de logging centralisé
from src.shared.logging import get_logger

# Instance globale pour les métriques (utilise get_logger)
metrics_logger = get_logger('metrics')

# Lock pour thread-safety
_log_lock = threading.Lock()


def log_metrics(level: str, message: str, extra: Dict = None):
    """
    Interface de logging simplifiée pour les métriques avec format structuré.
    
    Utilise le système de logging centralisé (get_logger) avec formatage JSON.
    
    Args:
        level: Niveau de log (INFO, WARNING, ERROR, DEBUG)
        message: Message à logger
        extra: Dictionnaire supplémentaire pour contexte
    """
    with _log_lock:
        try:
            log_dict = {
                "timestamp": datetime.now().isoformat(),
                "level": level.upper(),
                "message": message,
                "module": "metrics"
            }
            if extra:
                log_dict.update(extra)
            
            log_message = json.dumps(log_dict, ensure_ascii=False, default=str)
            log_method = getattr(metrics_logger, level.lower(), metrics_logger.info)
            log_method(log_message)
        except Exception as e:
            # Fallback ultra-robuste
            metrics_logger.error(f"LOGGING_ERROR: {message} - {str(e)}", exc_info=True)