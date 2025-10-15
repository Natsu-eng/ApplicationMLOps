
"""
Fonctions d'échantillonnage sécurisé pour DataFrames.
"""
import pandas as pd
import numpy as np
import dask.dataframe as dd
import logging
from typing import Union
from helpers.dask_helpers import is_dask_dataframe, compute_if_dask
from helpers.data_transformers import optimize_dataframe
from monitoring.decorators import monitor_performance


logger = logging.getLogger(__name__)

def safe_sample(
    df: Union[pd.DataFrame, 'dd.DataFrame'], 
    sample_frac: float = 0.01, 
    max_rows: int = 10000,
    min_rows: int = 100,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Échantillonnage sécurisé d'un DataFrame avec gestion d'erreurs.
    
    Args:
        df: DataFrame à échantillonner
        sample_frac: Fraction d'échantillonnage
        max_rows: Nombre maximum de lignes
        min_rows: Nombre minimum de lignes requises
        random_state: Seed pour la reproductibilité
        
    Returns:
        DataFrame échantillonné
    """
    try:
        is_dask = is_dask_dataframe(df)
        n_rows = len(df) if not is_dask else compute_if_dask(df.shape[0])
        
        if n_rows < min_rows:
            logger.warning(f"⚠️ Dataset trop petit ({n_rows} rows), utilisation complète")
            return compute_if_dask(df) if is_dask else df
            
        # Calcul de la taille d'échantillon optimale
        target_size = min(max_rows, max(min_rows, int(n_rows * sample_frac)))
        
        if target_size >= n_rows:
            sample_df = df
        else:
            if is_dask:
                # Pour Dask, utiliser un échantillonnage par fraction
                actual_frac = min(0.1, target_size / n_rows)
                sample_df = df.sample(frac=actual_frac, random_state=random_state)
                sample_df = sample_df.head(target_size, compute=False)
            else:
                # Pour Pandas, échantillonnage direct
                sample_df = df.sample(n=target_size, random_state=random_state)
                
        result_df = compute_if_dask(sample_df)
        result_df = optimize_dataframe(result_df)
        
        logger.debug(f"📊 Échantillonnage: {len(result_df)} lignes sur {n_rows} total")
        return result_df
        
    except Exception as e:
        logger.error(f"❌ Sampling failed: {e}")
        # Fallback: retourner les premières lignes
        try:
            fallback_size = min(min_rows, len(df) if not is_dask else compute_if_dask(df.shape[0]))
            fallback_df = compute_if_dask(df.head(fallback_size))
            return optimize_dataframe(fallback_df)
        except Exception as fallback_error:
            logger.error(f"❌ Complete sampling fallback failed: {fallback_error}")
            raise


class Config:
    MAX_PREVIEW_ROWS = 100
    MAX_SAMPLE_SIZE = 15000
    MAX_BIVARIATE_SAMPLE = 10000
    MEMORY_CHECK_INTERVAL = 180  # 3 minutes
    CACHE_TTL = 600  # 10 minutes
    TIMEOUT_THRESHOLD = 30
    MEMORY_WARNING = 85
    MEMORY_CRITICAL = 90

class DataSampler:
    """Gestion optimisée de l'échantillonnage"""  
    @staticmethod
    @monitor_performance("data_sampling")
    def get_sample(df, max_rows: int = Config.MAX_SAMPLE_SIZE, random_state: int = 42) -> pd.DataFrame:
        """Retourne un échantillon optimisé"""
        try:
            total_rows = compute_if_dask(df.shape[0])
            
            if total_rows <= max_rows:
                if is_dask_dataframe(df):
                    return compute_if_dask(df.head(max_rows))
                else:
                    return df.copy()
            
            sample_fraction = min(0.1, max_rows / total_rows)
            
            if is_dask_dataframe(df):
                sample = df.sample(frac=sample_fraction, random_state=random_state).head(max_rows)
                return compute_if_dask(sample)
            else:
                return df.sample(n=max_rows, random_state=random_state, replace=False)
                
        except Exception as e:
            logger.error(f"Sampling error: {e}")
            fallback_size = min(1000, total_rows) if 'total_rows' in locals() else 1000
            if is_dask_dataframe(df):
                return compute_if_dask(df.head(fallback_size))
            else:
                return df.head(fallback_size)
