"""
Décorateurs de monitoring et gestion d'erreurs pour l'application.
"""
import concurrent
from datetime import datetime
import json
import time
import logging
import functools
from typing import Any, Callable, Optional, Union
import psutil

logger = logging.getLogger(__name__)

# Variable globale pour la disponibilité de psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil non disponible, monitoring mémoire limité")

def monitor_performance(operation_name: Optional[str] = None) -> Callable:
    """
    Décorateur pour monitorer les performances des fonctions critiques.
    Supporte @monitor_performance et @monitor_performance("nom_operation")
    
    Args:
        operation_name: Nom optionnel de l'opération pour le logging
        
    Returns:
        Fonction décorée
    """
    def actual_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not PSUTIL_AVAILABLE:
                return func(*args, **kwargs)
            
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                duration = end_time - start_time
                memory_delta = end_memory - start_memory
                
                func_name = operation_name if operation_name else func.__name__
                logger.debug(f"{func_name} - Duration: {duration:.2f}s, Memory: {memory_delta:+.1f}MB")
                
                if duration > 30:
                    logger.warning(f"⏰ {func_name} took {duration:.2f}s - performance issue")
                if memory_delta > 500:
                    logger.warning(f"💾 {func_name} used {memory_delta:.1f}MB - memory issue")
                
                return result
                
            except TypeError as e:
                # Cas où kwargs ne correspondent pas à la fonction décorée
                if "unexpected keyword argument" in str(e):
                    logger.error(f"❌ {func.__name__} received unexpected kwargs: {list(kwargs.keys())}")
                raise
            except Exception as e:
                func_name = operation_name if operation_name else func.__name__
                logger.error(f"❌ Error in {func_name}: {str(e)}", exc_info=True)
                raise
        
        return wrapper
    
    # Gérer usage avec ou sans parenthèses
    if callable(operation_name):
        # Cas: @monitor_performance sans parenthèses
        return actual_decorator(operation_name)
    return actual_decorator

def monitor_ml_operation(func: Callable) -> Callable:
    """Décorateur de monitoring pour les opérations ML"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2) if PSUTIL_AVAILABLE else 0
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            memory_delta = (psutil.virtual_memory().used / (1024**2) - start_memory) if PSUTIL_AVAILABLE else 0
            
            # Seuils configurables
            if elapsed > 30:  # 30 secondes
                logger.warning(f"Opération lente: {func.__name__} a pris {elapsed:.2f}s")
            if memory_delta > 500:  # 500 MB
                logger.warning(f"Usage mémoire élevé: {func.__name__} a utilisé {memory_delta:.1f}MB")
                
            return result
        except Exception as e:
            logger.error(f"Échec {func.__name__}: {str(e)[:100]}")
            raise
    return wrapper

def safe_execute(fallback_value: Any = None, log_errors: bool = True) -> Callable:
    """
    Décorateur pour l'exécution sécurisée avec fallback.
    
    Args:
        fallback_value: Valeur de retour en cas d'erreur
        log_errors: Si True, log les erreurs
        
    Returns:
        Décorateur
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"❌ Safe execution failed in {func.__name__}: {str(e)}")
                return fallback_value
        return wrapper
    return decorator

def timeout(seconds: int = 300) -> Callable:
    """Décorateur de timeout pour les opérations longues."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                try:
                    future = executor.submit(func, *args, **kwargs)
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    logger.error(f"⏰ Timeout: {func.__name__} > {seconds}s")
                    return None
                except Exception as e:
                    logger.error(f"❌ Exception in {func.__name__}: {str(e)}")
                    return None
        return wrapper
    return decorator

def safe_metric_calculation(fallback_value: Any = None, max_retries: int = 1) -> Callable:
    """Décorateur robuste pour calculs de métriques avec retry."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"⚠️ Tentative {attempt + 1} échouée pour {func.__name__}, retry...")
                        time.sleep(0.1)  # Backoff minimal
                    else:
                        logger.error(f"❌ Échec après {max_retries + 1} tentatives pour {func.__name__}")
            
            return fallback_value
        return wrapper
    return decorator

def monitor_operation(func: Callable) -> Callable:
    """
    Décorateur pour monitorer les opérations ML avec logs clairs et lisibles.
    
    Args:
        func: Fonction à monitorer
        
    Returns:
        Fonction wrappée avec monitoring
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        operation_name = func.__name__
        start_time = time.time()
        
        # Log de démarrage
        logger.info(f"🔄 Démarrage: {operation_name}")
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Log de succès avec format clair
            logger.info(
                f"✅ Succès: {operation_name} | "
                f"Durée: {duration:.2f}s | "
                f"Module: {func.__module__}"
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)[:200]
            
            # Log d'erreur avec format clair
            logger.error(
                f"❌ Échec: {operation_name} | "
                f"Durée: {duration:.2f}s | "
                f"Erreur: {error_msg}"
            )
            
            raise
    
    return wrapper