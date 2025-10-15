"""
Transformations de données pour l'optimisation mémoire et le prétraitement.
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Union
import gc
import dask.dataframe as dd
from monitoring.decorators import safe_metric_calculation

logger = logging.getLogger(__name__)

def optimize_dataframe(
    df: pd.DataFrame, 
    memory_threshold_mb: float = 100.0,
    downcast_int: bool = True,
    downcast_float: bool = True
) -> pd.DataFrame:
    """
    Optimise un DataFrame Pandas en réduisant l'utilisation mémoire.
    
    Args:
        df: DataFrame Pandas
        memory_threshold_mb: Seuil de mémoire pour déclencher l'optimisation
        downcast_int: Downcaster les entiers
        downcast_float: Downcaster les floats
        
    Returns:
        DataFrame optimisé
    """
    if df.empty:
        return df
        
    try:
        # Vérifier l'utilisation mémoire actuelle
        current_memory = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        
        if current_memory < memory_threshold_mb:
            return df  # Pas d'optimisation nécessaire
            
        logger.info(f"🔧 Optimisation du DataFrame ({current_memory:.1f}MB)")
        
        df_copy = df.copy()
        memory_saved = 0
        
        # Downcasting numérique
        if downcast_int or downcast_float:
            numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_type = df_copy[col].dtype
                
                # Downcast entiers
                if downcast_int and np.issubdtype(col_type, np.integer):
                    df_copy[col] = pd.to_numeric(df_copy[col], downcast='integer')
                
                # Downcast floats
                elif downcast_float and np.issubdtype(col_type, np.floating):
                    df_copy[col] = pd.to_numeric(df_copy[col], downcast='float')
        
        # Conversion object en category
        object_cols = df_copy.select_dtypes(include=['object']).columns
        optimized_columns = 0
        
        for col in object_cols:
            try:
                unique_ratio = df_copy[col].nunique() / len(df_copy[col].dropna())
                if unique_ratio < 0.5 and df_copy[col].nunique() < 10000:
                    df_copy[col] = df_copy[col].astype("category")
                    optimized_columns += 1
            except Exception as e:
                logger.debug(f"Failed to optimize column {col}: {e}")
                continue
        
        # Calcul mémoire économisée
        if optimized_columns > 0:
            new_memory = df_copy.memory_usage(deep=True).sum() / 1024 / 1024
            memory_saved = current_memory - new_memory
            logger.info(f"✅ DataFrame optimisé: {optimized_columns} colonnes, {memory_saved:.1f}MB économisés")
            
        return df_copy
        
    except Exception as e:
        logger.warning(f"⚠️ DataFrame optimization failed: {e}")
        return df

def intelligent_type_coercion(
    df: Union[pd.DataFrame, 'dd.DataFrame'], 
    use_dask: bool, 
    max_sample_size: int = 100000
) -> Tuple[Union[pd.DataFrame, 'dd.DataFrame'], Dict[str, str]]:
    """
    Applique une coercion de type intelligente sur les colonnes pour éviter les erreurs de type mixte.
    
    Args:
        df: DataFrame Pandas ou Dask
        use_dask: Booléen indiquant si Dask est utilisé
        max_sample_size: Taille maximale de l'échantillon pour l'analyse
    
    Returns:
        Tuple contenant le DataFrame modifié et un dictionnaire des changements de type
    """
    changes = {}
    
    if use_dask:
        logger.info("Coercion de type limitée pour Dask. Inspectez les types manuellement.")
        return df, changes

    df_copy = df.copy()
    
    # Échantillonnage intelligent pour gros datasets
    if len(df_copy) > max_sample_size:
        sample_df = df_copy.sample(n=max_sample_size, random_state=42)
        logger.info(f"Échantillonnage de {max_sample_size} lignes pour l'analyse des types")
    else:
        sample_df = df_copy

    # Formats de date optimisés
    date_formats = [
        '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d/%m/%Y', '%m/%d/%Y',
        '%Y/%m/%d', '%d-%m-%Y', '%Y%m%d', '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S'
    ]

    object_columns = df_copy.select_dtypes(include=['object']).columns
    total_columns = len(object_columns)
    
    for idx, col in enumerate(object_columns, 1):
        try:
            # Logging du progrès pour les gros datasets
            if total_columns > 10 and idx % 10 == 0:
                logger.info(f"Type coercion progress: {idx}/{total_columns} columns processed")
            
            sample_col = sample_df[col].dropna()
            if sample_col.empty:
                continue
            
            # Test numérique
            try:
                numeric_col = pd.to_numeric(sample_col, errors='coerce')
                numeric_success_rate = numeric_col.notna().mean()
                
                if numeric_success_rate > 0.95:
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                    changes[col] = f"object -> numeric (success rate: {numeric_success_rate:.1%})"
                    logger.debug(f"Colonne '{col}' convertie en numérique.")
                    continue
                    
            except (ValueError, TypeError, OverflowError) as e:
                logger.debug(f"Numeric conversion failed for {col}: {e}")

            # Test de date
            first_values = sample_col.head(100).astype(str)
            date_like_ratio = sum(1 for val in first_values 
                                if any(char.isdigit() for char in val) and 
                                   any(sep in val for sep in ['-', '/', 'T', ' '])) / len(first_values)

            if date_like_ratio > 0.8:
                datetime_converted = False
                
                for fmt in date_formats[:6]:  # Top 6 formats
                    try:
                        datetime_col = pd.to_datetime(sample_col, format=fmt, errors='coerce')
                        datetime_success_rate = datetime_col.notna().mean()
                        
                        if datetime_success_rate > 0.9:
                            df_copy[col] = pd.to_datetime(df_copy[col], format=fmt, errors='coerce')
                            changes[col] = f"object -> datetime64[ns] (format: {fmt}, success rate: {datetime_success_rate:.1%})"
                            logger.debug(f"Colonne '{col}' convertie en datetime avec format {fmt}.")
                            datetime_converted = True
                            break
                            
                    except (ValueError, TypeError) as e:
                        logger.debug(f"DateTime conversion failed for {col} with format {fmt}: {e}")
                        continue

                # Fallback avec infer_datetime_format
                if not datetime_converted:
                    try:
                        datetime_col = pd.to_datetime(sample_col, errors='coerce', infer_datetime_format=True)
                        datetime_success_rate = datetime_col.notna().mean()
                        
                        if datetime_success_rate > 0.9:
                            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce', infer_datetime_format=True)
                            changes[col] = f"object -> datetime64[ns] (inferred, success rate: {datetime_success_rate:.1%})"
                            logger.debug(f"Colonne '{col}' convertie en datetime via infer_datetime_format.")
                            datetime_converted = True
                            
                    except (ValueError, TypeError) as e:
                        logger.debug(f"DateTime infer conversion failed for {col}: {e}")

                if datetime_converted:
                    continue

            # Optimisation catégorie vs string
            unique_count = sample_col.nunique()
            total_count = len(sample_col)
            unique_ratio = unique_count / total_count if total_count > 0 else 0
            
            if unique_ratio < 0.5 and unique_count < 1000:
                try:
                    df_copy[col] = df_copy[col].astype('category')
                    changes[col] = f"object -> category (unique ratio: {unique_ratio:.1%})"
                    logger.debug(f"Colonne '{col}' convertie en 'category'.")
                except (ValueError, TypeError) as e:
                    logger.debug(f"Category conversion failed for {col}: {e}")
                    changes[col] = f"object -> object (category conversion failed)"
            else:
                # Garder comme object mais s'assurer de l'homogénéité
                try:
                    df_copy[col] = df_copy[col].astype(str)
                    changes[col] = f"object -> str (homogenization, unique ratio: {unique_ratio:.1%})"
                except Exception as e:
                    logger.debug(f"String conversion failed for {col}: {e}")
                    changes[col] = f"object -> object (no change, conversion failed)"

        except Exception as e:
            logger.warning(f"Type coercion error for column '{col}': {str(e)}")
            changes[col] = f"object -> object (error: {type(e).__name__})"
            continue

    # Nettoyage mémoire
    del sample_df
    gc.collect()
    
    logger.info(f"Type coercion completed: {len(changes)} columns processed")
    return df_copy, changes


@safe_metric_calculation(fallback_value=np.array([]))
def safe_array_conversion(data: Any, max_samples: int = 100000, sample: bool = True) -> np.ndarray:
    """
    Convertit les données en array numpy de façon ultra-robuste.
    """
    try:
        if data is None:
            logger.warning("Données None fournies à safe_array_conversion")
            return np.array([])
        
        # Conversion robuste selon le type
        if isinstance(data, pd.Series):
            result = data.values
        elif isinstance(data, pd.DataFrame):
            result = data.values.flatten() if data.shape[1] == 1 else data.values
        elif isinstance(data, list):
            result = np.array(data, dtype=object)
        elif isinstance(data, np.ndarray):
            result = data.copy()
        else:
            # Tentative générique
            result = np.array(data, dtype=object)
        
        # Nettoyage des données
        if hasattr(result, 'size') and result.size == 0:
            logger.warning("Tableau vide après conversion")
            return np.array([])
        
        # Échantillonnage intelligent
        if sample and hasattr(result, 'shape') and len(result) > max_samples:
            logger.info(f"Application échantillonnage: {len(result)} → {max_samples}")
            
            random_state = 42
            rng = np.random.RandomState(random_state)
            
            try:
                indices = rng.choice(len(result), size=max_samples, replace=False)
                result = result[indices]
            except Exception as e:
                logger.warning(f"Échec échantillonnage, prise des premiers échantillons: {e}")
                result = result[:max_samples]
        
        # Aplatissement si nécessaire
        if hasattr(result, 'ndim') and result.ndim > 1 and result.shape[1] == 1:
            result = result.flatten()
            
        return result
        
    except Exception as e:
        logger.error(f"Échec critique conversion tableau: {e}")
        return np.array([])