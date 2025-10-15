"""
Utilitaires de logging structuré pour l'application.
"""
import logging
import json
import threading
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    """Logger structuré et thread-safe pour l'application."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._lock = threading.Lock()
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure le logging avec format structuré."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_structured(self, level: str, message: str, extra: Dict = None):
        """Log structuré en JSON thread-safe."""
        with self._lock:
            try:
                log_dict = {
                    "timestamp": datetime.now().isoformat(),
                    "level": level.upper(),
                    "message": message,
                    "module": self.logger.name
                }
                if extra:
                    log_dict.update(extra)
                
                log_message = json.dumps(log_dict, ensure_ascii=False, default=str)
                getattr(self.logger, level.lower())(log_message)
            except Exception as e:
                # Fallback ultra-robuste
                print(f"LOGGING_ERROR: {message} - {str(e)}")

# Instance globale pour les métriques
metrics_logger = StructuredLogger('metrics')

def log_metrics(level: str, message: str, extra: Dict = None):
    """Interface de logging simplifiée pour les métriques."""
    metrics_logger.log_structured(level, message, extra)