"""
Module de monitoring système centralisé, optimisé pour production.
Utilise psutil pour des métriques fiables sans dépendances externes excessives.
"""
import time
from typing import Dict, Any

import psutil
from src.shared.logging import get_logger

logger = get_logger(__name__)

class SystemMonitor:
    """
    Classe principale pour monitoring système.
    Utilisation en prod : Instanciez une fois et réutilisez pour éviter overhead.
    """
    def __init__(self, memory_threshold: float = 80.0, cpu_threshold: float = 90.0):
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold

    def get_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques système actuelles."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            return {
                'memory_percent': memory.percent,
                'memory_available_mb': memory.available / (1024 ** 2),
                'cpu_percent': cpu_percent,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métriques: {e}")
            return {
                'memory_percent': 0.0,
                'memory_available_mb': 0.0,
                'cpu_percent': 0.0,
                'timestamp': time.time()
            }

    def check_resources(self) -> str:
        """Vérifie les ressources et retourne un statut (normal/warning/critical/error)."""
        try:
            metrics = self.get_metrics()
            memory_percent = metrics['memory_percent']
            if memory_percent > 95.0:  # Seuil critique prod
                logger.critical(f"Mémoire critique: {memory_percent}%")
                return "critical"
            elif memory_percent > self.memory_threshold:
                logger.warning(f"Mémoire élevée: {memory_percent}%")
                return "warning"
            return "normal"
        except Exception as e:
            logger.error(f"Erreur lors de la vérification des ressources: {e}")
            return "error"

def _get_memory_usage() -> float:
    """Fonction privée pour usage mémoire du process actuel (en MB)."""
    try:
        process = psutil.Process()
        return process.memory_info().rss / (1024 ** 2)
    except Exception as e:
        logger.error(f"Erreur _get_memory_usage: {e}")
        return 0.0

def get_system_metrics() -> Dict[str, Any]:
    """Fonction publique wrapper pour compatibilité."""
    return SystemMonitor().get_metrics()

def check_system_resources(df, n_models: int) -> Dict[str, Any]:
    """Vérifie les ressources système pour l'entraînement ML"""
    try:
        metrics = get_system_metrics()
        available_memory_mb = metrics['memory_available_mb']
        
        # Estimation de la mémoire nécessaire
        df_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024) if hasattr(df, 'memory_usage') else 0
        estimated_needed_mb = df_size_mb * n_models * 2  # Facteur conservateur
        
        has_enough = available_memory_mb > estimated_needed_mb * 1.5  # Marge de 50%
        
        result = {
            "has_enough_resources": has_enough,
            "issues": [],
            "warnings": [],
            "available_memory_mb": available_memory_mb,
            "estimated_needed_mb": estimated_needed_mb
        }
        
        if not has_enough:
            result["issues"].append(f"Mémoire insuffisante: {available_memory_mb:.1f}MB disponible vs {estimated_needed_mb:.1f}MB estimé")
        
        if metrics['memory_percent'] > 80:
            result["warnings"].append(f"Utilisation mémoire élevée: {metrics['memory_percent']:.1f}%")
        
        logger.info(f"Ressources vérifiées: {available_memory_mb:.1f}MB disponible, {estimated_needed_mb:.1f}MB estimé")
        
        return result
    except Exception as e:
        logger.error(f"Erreur check_system_resources: {e}")
        return {
            "has_enough_resources": False,
            "issues": [str(e)],
            "warnings": [],
            "available_memory_mb": 0,
            "estimated_needed_mb": 0
        }

__all__ = ['SystemMonitor', 'get_system_metrics', '_get_memory_usage', 'check_system_resources']
