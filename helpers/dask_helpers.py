"""
Helpers pour la gestion des DataFrames Dask.
"""
import time
from typing import Any
from src.shared.logging import get_logger

logger = get_logger(__name__)

# Variable globale pour la disponibilité de Dask
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask non disponible, utilisation de Pandas uniquement")

def is_dask_dataframe(df: Any) -> bool:
    """
    Vérifie si l'objet est un DataFrame Dask.
    
    Args:
        df: Objet à vérifier
        
    Returns:
        Booléen indiquant si c'est un DataFrame Dask
    """
    return DASK_AVAILABLE and isinstance(df, dd.DataFrame)

def compute_if_dask(data: Any) -> Any:
    """
    Exécute .compute() si l'objet est un DataFrame, Series ou Scalar Dask.
    
    Args:
        data: Objet à évaluer (DataFrame, Series ou Scalar)
        
    Returns:
        Objet calculé (si Dask) ou inchangé (si Pandas)
    """
    if is_dask_dataframe(data) or (DASK_AVAILABLE and isinstance(data, (dd.Series, dd.core.Scalar))):
        start = time.time()
        try:
            result = data.compute()
            elapsed = time.time() - start
            logger.debug(f"Dask compute() terminé en {elapsed:.2f} sec")
            return result
        except Exception as e:
            logger.error(f"❌ Dask compute() failed: {e}")
            raise
    return data