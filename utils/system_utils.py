import gc
import psutil # type: ignore
import time
import logging
from typing import Dict, Any, List
from src.shared.logging import get_logger

logger = get_logger(__name__)

def get_system_metrics() -> Dict[str, Any]:
    """Récupère les métriques système"""
    try:
        memory = psutil.virtual_memory()
        metrics = {
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'timestamp': time.time()
        }
        logger.info("Métriques système collectées", extra=metrics)
        return metrics
    except Exception as e:
        logger.error(f"Échec métriques système: {str(e)[:100]}")
        return {'memory_percent': 0, 'memory_available_mb': 0, 'cpu_percent': 0, 'timestamp': time.time()}

def check_system_resources(df, n_models: int) -> Dict[str, Any]:
    """Vérifie les ressources système"""
    check_result = {
        "has_enough_resources": True,
        "issues": [],
        "warnings": [],
        "available_memory_mb": 0,
        "estimated_needed_mb": 0
    }
    
    try:
        df_memory = df.memory_usage(deep=True).sum() / (1024**2)
        estimated_needed = df_memory * n_models * 2  # Multiplicateur arbitraire
        available_memory = psutil.virtual_memory().available / (1024**2)
        check_result["available_memory_mb"] = available_memory
        check_result["estimated_needed_mb"] = estimated_needed
        
        if estimated_needed > available_memory:
            check_result["has_enough_resources"] = False
            check_result["issues"].append(f"Mémoire insuffisante: {estimated_needed:.0f}MB requis, {available_memory:.0f}MB disponible")
        elif estimated_needed > available_memory * 0.8:
            check_result["warnings"].append(f"Mémoire limite: {estimated_needed:.0f}MB requis, {available_memory:.0f}MB disponible")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            check_result["warnings"].append(f"CPU élevé: {cpu_percent:.1f}%")
            
        logger.info("Vérification ressources", extra=check_result)
    except Exception as e:
        check_result["warnings"].append("Échec vérification ressources")
        logger.warning(f"Erreur vérification ressources: {str(e)[:100]}")
    
    return check_result

def get_memory_usage() -> float:
    """Retourne l'utilisation mémoire actuelle en MB"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        return memory_mb
    except Exception as e:
        logger.error(f"Erreur mémoire: {str(e)[:100]}")
        return 0.0

def is_system_healthy() -> Dict[str, Any]:
    """Vérifie la santé globale du système"""
    health = {
        "healthy": True,
        "issues": [],
        "metrics": {}
    }
    
    try:
        # Mémoire
        memory = psutil.virtual_memory()
        health["metrics"]["memory_percent"] = memory.percent
        if memory.percent > 90:
            health["healthy"] = False
            health["issues"].append(f"Mémoire critique: {memory.percent:.1f}%")
        elif memory.percent > 80:
            health["issues"].append(f"Mémoire élevée: {memory.percent:.1f}%")
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        health["metrics"]["cpu_percent"] = cpu_percent
        if cpu_percent > 90:
            health["healthy"] = False
            health["issues"].append(f"CPU critique: {cpu_percent:.1f}%")
        elif cpu_percent > 80:
            health["issues"].append(f"CPU élevé: {cpu_percent:.1f}%")
        
        # Disk (racine)
        disk = psutil.disk_usage('/')
        health["metrics"]["disk_percent"] = disk.percent
        if disk.percent > 95:
            health["healthy"] = False
            health["issues"].append(f"Disk critique: {disk.percent:.1f}%")
        elif disk.percent > 85:
            health["issues"].append(f"Disk élevé: {disk.percent:.1f}%")
            
    except Exception as e:
        health["healthy"] = False
        health["issues"].append(f"Erreur vérification santé: {str(e)[:100]}")
        logger.error(f"Erreur santé système: {str(e)[:100]}")
    
    return health

# Définition locale de cleanup_memory (pour compatibilité et nettoyage mémoire)
def cleanup_memory():
    """Nettoyage mémoire simple et robuste."""
    gc.collect()