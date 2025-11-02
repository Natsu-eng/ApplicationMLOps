"""
Module d'analyse de données robuste pour le machine learning.
Optimisé pour la production avec gestion mémoire avancée et monitoring.
src/data/data_analysis.py
"""

import pandas as pd
import numpy as np
import time
from typing import Union, Tuple, Dict, Any, List, Optional
import gc

# Import des modules déplacés
from monitoring.decorators import monitor_performance, safe_execute
from monitoring.system_monitor import check_system_resources as check_resources
from helpers.dask_helpers import is_dask_dataframe, compute_if_dask
from helpers.data_samplers import safe_sample
from helpers.data_transformers import optimize_dataframe
from helpers.streamlit_helpers import conditional_cache

# Configuration du logging
from src.shared.logging import get_logger
logger = get_logger(__name__)

# Tentative d'import de dépendances optionnelles
try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask non disponible, utilisation de Pandas uniquement")

try:
    from scipy.stats import pointbiserialr, f_oneway
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy non disponible, certaines analyses statistiques limitées")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


# Définition locale de cleanup_memory (pour compatibilité et nettoyage mémoire)
def cleanup_memory():
    """Nettoyage mémoire simple et robuste."""
    gc.collect()


# =============================
# Fonctions principales
# =============================

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={"numeric": [], "categorical": [], "text_or_high_cardinality": [], "datetime": []})
@monitor_performance
def auto_detect_column_types(
    df: Union[pd.DataFrame, 'dd.DataFrame'], 
    sample_frac: float = 0.01, 
    max_rows: int = 10000,
    high_cardinality_threshold: int = 100
) -> Dict[str, List[str]]:
    """
    Détecte automatiquement les types de colonnes (numérique, catégorielle, datetime, texte).
    Version robuste avec gestion d'erreurs améliorée.
    
    Args:
        df: DataFrame Pandas ou Dask
        sample_frac: Fraction de l'échantillon pour Dask ou gros datasets
        max_rows: Nombre maximum de lignes à analyser
        high_cardinality_threshold: Seuil pour la cardinalité élevée
        
    Returns:
        Dictionnaire avec les listes de colonnes par type
    """
    try:
        if df is None or df.empty or len(df.columns) == 0:
            logger.warning("⚠️ DataFrame vide ou sans colonnes")
            return {"numeric": [], "categorical": [], "text_or_high_cardinality": [], "datetime": []}
        
        # Échantillonnage sécurisé
        sample_df = safe_sample(df, sample_frac, max_rows)
        
        if sample_df.empty:
            logger.warning("⚠️ Échantillon DataFrame vide")
            return {"numeric": [], "categorical": [], "text_or_high_cardinality": [], "datetime": []}

        # Initialisation du résultat
        result = {
            "numeric": [],
            "datetime": [],
            "categorical": [],
            "text_or_high_cardinality": []
        }

        # Détection des colonnes numériques (types natifs)
        try:
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
            result["numeric"] = [col for col in numeric_cols if col in df.columns]
            logger.debug(f"🔢 Colonnes numériques détectées: {len(result['numeric'])}")
        except Exception as e:
            logger.warning(f"⚠️ Détection colonnes numériques échouée: {e}")

        # Détection des colonnes datetime (types natifs et conversion)
        try:
            # Types datetime natifs
            datetime_cols = sample_df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
            
            # Tentative de conversion des colonnes objet qui ressemblent à des dates
            object_cols = sample_df.select_dtypes(include=["object"]).columns
            for col in object_cols:
                if col not in datetime_cols and col not in result["numeric"]:
                    try:
                        # Essayer de convertir en datetime
                        converted = pd.to_datetime(sample_df[col], errors='coerce')
                        if converted.notna().mean() > 0.8:  # Si >80% de conversion réussie
                            datetime_cols.append(col)
                    except:
                        pass
            
            result["datetime"] = [col for col in datetime_cols if col in df.columns]
            logger.debug(f"📅 Colonnes datetime détectées: {len(result['datetime'])}")
        except Exception as e:
            logger.warning(f"⚠️ Détection colonnes datetime échouée: {e}")

        # Analyse des colonnes object/category
        try:
            object_cols = sample_df.select_dtypes(include=["object", "category"]).columns.tolist()
            # Exclure les colonnes déjà classées comme datetime
            object_cols = [col for col in object_cols if col not in result["datetime"]]
            
            for col in object_cols:
                try:
                    if col not in df.columns:
                        continue
                        
                    col_series = sample_df[col].dropna()
                    if len(col_series) == 0:
                        result["text_or_high_cardinality"].append(col)
                        continue
                        
                    unique_count = col_series.nunique()
                    total_count = len(col_series)
                    unique_ratio = unique_count / total_count if total_count > 0 else 1
                    
                    # Logique de classification améliorée
                    if unique_ratio < 0.5 and unique_count <= high_cardinality_threshold:
                        result["categorical"].append(col)
                    else:
                        result["text_or_high_cardinality"].append(col)
                        
                except Exception as e:
                    logger.debug(f"❌ Erreur analyse colonne {col}: {e}")
                    result["text_or_high_cardinality"].append(col)
                    
            logger.debug(f"🏷️ Colonnes catégorielles détectées: {len(result['categorical'])}")
            logger.debug(f"📝 Colonnes texte/haute cardinalité: {len(result['text_or_high_cardinality'])}")
            
        except Exception as e:
            logger.warning(f"⚠️ Analyse colonnes object échouée: {e}")

        # Nettoyage mémoire
        del sample_df
        gc.collect()

        total_detected = sum(len(cols) for cols in result.values())
        logger.info(f"✅ Détection types colonnes terminée: {total_detected} colonnes classifiées")
        return result

    except Exception as e:
        logger.error(f"❌ Erreur critique dans auto_detect_column_types: {e}", exc_info=True)
        return {"numeric": [], "categorical": [], "text_or_high_cardinality": [], "datetime": []}

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={"count": 0, "missing_values": 0, "missing_percentage": "100.00%"})
@monitor_performance
def get_column_profile(
    series: Union[pd.Series, 'dd.Series'], 
    sample_frac: float = 0.01, 
    max_rows: int = 10000
) -> Dict[str, Any]:
    """
    Génère un profil statistique pour une colonne donnée.
    Version robuste avec gestion d'erreurs.
    
    Args:
        series: Série Pandas ou Dask
        sample_frac: Fraction de l'échantillon pour Dask ou gros datasets
        max_rows: Nombre maximum de lignes à analyser
        
    Returns:
        Dictionnaire avec les statistiques de la colonne
    """
    try:
        if series is None or len(series) == 0:
            return {"count": 0, "missing_values": 0, "missing_percentage": "100.00%"}
        
        # Échantillonnage sécurisé
        is_dask = is_dask_dataframe(series) if hasattr(series, 'dtype') else False
        n_rows = len(series) if not is_dask else compute_if_dask(series.shape[0])
        
        if is_dask or n_rows > max_rows:
            sample_size = min(max_rows, max(100, int(n_rows * sample_frac)))
            if is_dask:
                actual_frac = min(0.1, sample_size / n_rows)
                sample_series = series.sample(frac=actual_frac).head(sample_size)
            else:
                sample_series = series.sample(n=sample_size, random_state=42)
        else:
            sample_series = series
            
        sample_series = compute_if_dask(sample_series)

        # Statistiques de base
        total_count = len(sample_series)
        valid_count = sample_series.count()
        missing_count = total_count - valid_count
        missing_percentage = (missing_count / total_count * 100) if total_count > 0 else 0

        profile = {
            "count": valid_count,
            "missing_values": missing_count,
            "missing_percentage": f"{missing_percentage:.2f}%",
            "total_rows_analyzed": total_count,
            "dtype": str(sample_series.dtype)
        }

        # Statistiques spécifiques au type
        if valid_count > 0:
            try:
                if pd.api.types.is_numeric_dtype(sample_series.dtype):
                    valid_series = sample_series.dropna()
                    stats = {
                        "mean": float(valid_series.mean()),
                        "std_dev": float(valid_series.std()),
                        "min": float(valid_series.min()),
                        "25%": float(valid_series.quantile(0.25)),
                        "median": float(valid_series.median()),
                        "75%": float(valid_series.quantile(0.75)),
                        "max": float(valid_series.max()),
                    }
                    
                    # Skewness seulement si assez de données
                    if len(valid_series) > 1:
                        stats["skewness"] = float(valid_series.skew())
                    
                    profile.update(stats)
                    
                elif pd.api.types.is_datetime64_any_dtype(sample_series.dtype):
                    valid_series = sample_series.dropna()
                    profile.update({
                        "min_date": str(valid_series.min()),
                        "max_date": str(valid_series.max()),
                        "unique_dates": int(valid_series.nunique()),
                        "date_range_days": (valid_series.max() - valid_series.min()).days
                    })
                else:
                    # Colonnes catégorielles ou texte
                    valid_series = sample_series.dropna()
                    unique_count = valid_series.nunique()
                    profile.update({
                        "unique_values": int(unique_count),
                        "unique_ratio": float(unique_count / len(valid_series)) if len(valid_series) > 0 else 0
                    })
                    
                    # Top valeurs pour les colonnes catégorielles
                    if unique_count <= 20:
                        try:
                            top_values = valid_series.value_counts().head(10).to_dict()
                            profile["top_values"] = {str(k): int(v) for k, v in top_values.items()}
                        except Exception as e:
                            logger.debug(f"❌ Calcul top valeurs échoué: {e}")
                            
            except Exception as e:
                logger.warning(f"⚠️ Calcul statistiques avancées échoué pour {series.name}: {e}")
                profile["computation_error"] = str(e)

        logger.debug(f"✅ Profil calculé pour colonne {series.name}: {profile.get('count', 0)} valeurs valides")
        return profile

    except Exception as e:
        logger.error(f"❌ Erreur critique dans get_column_profile for {getattr(series, 'name', 'unknown')}: {e}")
        return {"error": str(e), "count": 0, "missing_values": 0, "missing_percentage": "100.00%"}

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={})
@monitor_performance
def get_data_profile(
    df: Union[pd.DataFrame, 'dd.DataFrame'], 
    sample_frac: float = 0.01, 
    max_rows: int = 10000
) -> Dict[str, Dict[str, Any]]:
    """
    Génère un profil global du dataset par colonne.
    Version optimisée avec traitement par batch.
    
    Args:
        df: DataFrame Pandas ou Dask
        sample_frac: Fraction de l'échantillon pour Dask ou gros datasets
        max_rows: Nombre maximum de lignes à analyser
        
    Returns:
        Dictionnaire avec les profils par colonne
    """
    try:
        if df is None or df.empty or len(df.columns) == 0:
            logger.warning("⚠️ DataFrame vide ou sans colonnes")
            return {}

        profiles = {}
        total_columns = len(df.columns)
        
        logger.info(f"📊 Calcul profil données pour {total_columns} colonnes")
        
        # Traitement par batch pour éviter la surcharge mémoire
        batch_size = min(10, total_columns)
        
        for i in range(0, total_columns, batch_size):
            batch_cols = df.columns[i:i + batch_size]
            logger.debug(f"🔧 Traitement batch {i//batch_size + 1}/{(total_columns + batch_size - 1)//batch_size}")
            
            for col in batch_cols:
                try:
                    profiles[col] = get_column_profile(df[col], sample_frac, max_rows)
                except Exception as e:
                    logger.warning(f"⚠️ Profilage colonne {col} échoué: {e}")
                    profiles[col] = {"error": str(e), "count": 0}
                    
            # Nettoyage mémoire périodique
            if i % (batch_size * 3) == 0:
                gc.collect()

        logger.info(f"✅ Profil données terminé pour {len(profiles)} colonnes")
        return profiles

    except Exception as e:
        logger.error(f"❌ Erreur critique dans get_data_profile: {e}")
        return {}

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={"constant": [], "id_like": []})
@monitor_performance
def analyze_columns(
    df: Union[pd.DataFrame, 'dd.DataFrame'], 
    sample_frac: float = 0.01, 
    max_rows: int = 10000
) -> Dict[str, List[str]]:
    """
    Détecte les colonnes constantes ou de type ID.
    Version optimisée avec gestion d'erreurs.
    
    Args:
        df: DataFrame Pandas ou Dask
        sample_frac: Fraction de l'échantillon pour Dask ou gros datasets
        max_rows: Nombre maximum de lignes à analyser
        
    Returns:
        Dictionnaire avec les colonnes constantes et ID-like
    """
    try:
        if df is None or df.empty or len(df.columns) == 0:
            return {"constant": [], "id_like": []}

        # Échantillonnage sécurisé
        sample_df = safe_sample(df, sample_frac, max_rows)
        
        if sample_df.empty:
            return {"constant": [], "id_like": []}

        constant_cols = []
        id_like_cols = []
        
        try:
            nunique = sample_df.nunique()
            n_rows = len(sample_df)
            
            for col in sample_df.columns:
                try:
                    unique_count = nunique.get(col, 0)
                    
                    # Colonnes constantes
                    if unique_count <= 1:
                        constant_cols.append(col)
                        
                    # Colonnes de type ID (unique pour chaque ligne)
                    elif unique_count == n_rows and n_rows > 10:
                        id_like_cols.append(col)
                        
                except Exception as e:
                    logger.debug(f"❌ Erreur analyse colonne {col}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"⚠️ Analyse colonnes échouée: {e}")
            return {"constant": [], "id_like": []}

        # Nettoyage mémoire
        del sample_df
        gc.collect()

        logger.info(f"✅ Analyse colonnes terminée: {len(constant_cols)} constantes, {len(id_like_cols)} ID-like")
        return {"constant": constant_cols, "id_like": id_like_cols}

    except Exception as e:
        logger.error(f"❌ Erreur critique dans analyze_columns: {e}")
        return {"constant": [], "id_like": []}

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={"is_imbalanced": False, "imbalance_ratio": 1.0, "message": "Error in detection"})
@monitor_performance
def detect_imbalance(
    df: Union[pd.DataFrame, 'dd.DataFrame'], 
    target_column: str, 
    threshold: float = 0.8,
    sample_frac: float = 0.1,
    max_rows: int = 10000
) -> Dict[str, Any]:
    """
    Détecte le déséquilibre des classes pour les problèmes de classification.
    Version robuste avec gestion d'erreurs.
    
    Args:
        df: DataFrame Pandas ou Dask
        target_column: Nom de la colonne cible
        threshold: Seuil de déséquilibre (0.8 = 80% dans une classe)
        sample_frac: Fraction d'échantillonnage
        max_rows: Nombre maximum de lignes à analyser
        
    Returns:
        Dictionnaire avec les résultats de détection
    """
    try:
        # Validation des entrées
        if target_column not in df.columns:
            return {
                "is_imbalanced": False,
                "imbalance_ratio": 1.0,
                "message": f"Colonne cible '{target_column}' non trouvée",
                "class_distribution": {}
            }
        
        # Échantillonnage sécurisé
        sample_df = safe_sample(df, sample_frac, max_rows)
        
        if sample_df.empty:
            return {
                "is_imbalanced": False,
                "imbalance_ratio": 1.0,
                "message": "Échantillon vide",
                "class_distribution": {}
            }
        
        target_series = sample_df[target_column].dropna()
        
        if len(target_series) == 0:
            return {
                "is_imbalanced": False,
                "imbalance_ratio": 1.0,
                "message": "Aucune valeur valide dans la colonne cible",
                "class_distribution": {}
            }
        
        # Calcul de la distribution des classes
        class_distribution = target_series.value_counts().to_dict()
        total_samples = len(target_series)
        
        if len(class_distribution) <= 1:
            return {
                "is_imbalanced": False,
                "imbalance_ratio": 1.0,
                "message": "Une seule classe détectée",
                "class_distribution": class_distribution
            }
        
        # Calcul du ratio de déséquilibre
        majority_class_count = max(class_distribution.values())
        imbalance_ratio = majority_class_count / total_samples
        
        # Détection du déséquilibre
        is_imbalanced = imbalance_ratio > threshold
        
        # Calcul de métriques supplémentaires
        minority_class_count = min(class_distribution.values())
        balance_ratio = minority_class_count / majority_class_count if majority_class_count > 0 else 0
        
        result = {
            "is_imbalanced": is_imbalanced,
            "imbalance_ratio": float(imbalance_ratio),
            "balance_ratio": float(balance_ratio),
            "threshold_used": float(threshold),
            "total_samples": total_samples,
            "total_classes": len(class_distribution),
            "class_distribution": class_distribution,
            "majority_class": {
                "class": max(class_distribution, key=class_distribution.get),
                "count": majority_class_count,
                "percentage": float(majority_class_count / total_samples * 100)
            },
            "minority_class": {
                "class": min(class_distribution, key=class_distribution.get),
                "count": minority_class_count,
                "percentage": float(minority_class_count / total_samples * 100)
            }
        }
        
        # Messages explicatifs
        if is_imbalanced:
            result["message"] = f"🚨 Déséquilibre détecté : {result['majority_class']['percentage']:.1f}% dans la classe majoritaire"
            result["recommendation"] = "Envisagez d'activer SMOTE ou d'utiliser l'échantillonnage"
        else:
            result["message"] = "✅ Classes équilibrées"
            result["recommendation"] = "Aucune action nécessaire"
        
        logger.info(f"✅ Analyse déséquilibre terminée : {result['message']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur dans detect_imbalance : {e}")
        return {
            "is_imbalanced": False,
            "imbalance_ratio": 1.0,
            "message": f"Erreur d'analyse : {str(e)}",
            "class_distribution": {}
        }

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={"target_type": "unknown", "task": "unknown"})
def get_target_and_task(
    df: Union[pd.DataFrame, 'dd.DataFrame'],
    target: Optional[str]
) -> Dict[str, str]:
    """
    Détecte le type de tâche ML selon la colonne cible.
    Version robuste avec validation.
    
    Args:
        df: DataFrame Pandas ou Dask
        target: Nom de la colonne cible ou None pour non supervisé
        
    Returns:
        Dictionnaire avec le type de cible et la tâche ML
    """
    try:
        # Cas non supervisé
        if target is None:
            return {"target_type": "unsupervised", "task": "clustering"}

        if target not in df.columns:
            logger.warning(f"⚠️ Colonne cible '{target}' non trouvée dans le DataFrame")
            return {"target_type": "unknown", "task": "unknown"}
            
        # Échantillonnage pour l'analyse
        sample_df = safe_sample(df, sample_frac=0.05, max_rows=20000)
        
        if sample_df.empty or target not in sample_df.columns:
            return {"target_type": "unknown", "task": "unknown"}
            
        target_series = sample_df[target].dropna()
        
        if len(target_series) == 0:
            logger.warning(f"⚠️ Colonne cible '{target}' sans valeurs valides")
            return {"target_type": "unknown", "task": "unknown"}
            
        unique_vals = target_series.nunique()
        total_vals = len(target_series)

        # Logique de détection améliorée
        if pd.api.types.is_numeric_dtype(target_series):
            # Pour les variables numériques
            unique_ratio = unique_vals / total_vals
            
            if unique_vals <= 20 or unique_ratio < 0.05:
                # Peu de valeurs uniques -> classification
                return {"target_type": "classification", "task": "classification"}
            else:
                # Beaucoup de valeurs uniques -> régression
                return {"target_type": "regression", "task": "regression"}
        else:
            # Pour les variables non-numériques -> toujours classification
            return {"target_type": "classification", "task": "classification"}
            
    except Exception as e:
        logger.error(f"❌ Erreur dans get_target_and_task pour target '{target}': {e}")
        return {"target_type": "unknown", "task": "unknown"}

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={"numeric": [], "categorical": []})
@monitor_performance
def get_relevant_features(
    df: Union[pd.DataFrame, 'dd.DataFrame'],
    target: str,
    sample_frac: float = 0.05,
    max_rows: int = 20000,
    correlation_threshold: float = 0.1,
    p_value_threshold: float = 0.05
) -> Dict[str, List[str]]:
    """
    Sélectionne les features pertinentes via corrélation/ANOVA.
    Version robuste avec validation améliorée.
    
    Args:
        df: DataFrame Pandas ou Dask
        target: Nom de la colonne cible
        sample_frac: Fraction de l'échantillon pour Dask ou gros datasets
        max_rows: Nombre maximum de lignes à analyser
        correlation_threshold: Seuil de corrélation minimum
        p_value_threshold: Seuil de p-value maximum
        
    Returns:
        Dictionnaire avec les features numériques et catégorielles pertinentes
    """
    try:
        if target not in df.columns:
            logger.warning(f"⚠️ Colonne cible '{target}' non trouvée")
            return {"numeric": [], "categorical": []}

        # Échantillonnage sécurisé
        sample_df = safe_sample(df, sample_frac, max_rows)
        
        if sample_df.empty or target not in sample_df.columns:
            return {"numeric": [], "categorical": []}

        target_series = sample_df[target].dropna()
        
        if len(target_series) < 10:  # Minimum pour les tests statistiques
            logger.warning(f"⚠️ Données insuffisantes pour analyse features ({len(target_series)} valeurs target valides)")
            return {"numeric": [], "categorical": []}

        features = {"numeric": [], "categorical": []}
        
        # Aligner les indices pour éviter les erreurs de correspondance
        valid_indices = target_series.index
        sample_df_aligned = sample_df.loc[valid_indices]

        # Analyse des features numériques
        numeric_cols = sample_df_aligned.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target]
        
        for col in numeric_cols:
            try:
                feature_series = sample_df_aligned[col].fillna(0)  # Imputation simple
                
                if feature_series.nunique() > 1:  # Éviter les colonnes constantes
                    try:
                        # Corrélation de Pearson pour variables continues
                        if pd.api.types.is_numeric_dtype(target_series):
                            corr_coef = np.corrcoef(target_series, feature_series)[0, 1]
                            if not np.isnan(corr_coef) and abs(corr_coef) > correlation_threshold:
                                features["numeric"].append(col)
                        else:
                            # Point-biserial pour target catégorielle
                            if SCIPY_AVAILABLE:
                                corr, p_val = pointbiserialr(target_series.astype('category').cat.codes, 
                                                        feature_series)
                                if not np.isnan(corr) and abs(corr) > correlation_threshold and p_val < p_value_threshold:
                                    features["numeric"].append(col)
                            else:
                                # Fallback sans scipy
                                corr_coef = np.corrcoef(target_series.astype('category').cat.codes, feature_series)[0, 1]
                                if not np.isnan(corr_coef) and abs(corr_coef) > correlation_threshold:
                                    features["numeric"].append(col)
                                
                    except Exception as e:
                        logger.debug(f"❌ Analyse corrélation échouée pour {col}: {e}")
                        
            except Exception as e:
                logger.debug(f"❌ Erreur traitement feature numérique {col}: {e}")
                continue

        # Analyse des features catégorielles
        categorical_cols = sample_df_aligned.select_dtypes(include=["object", "category"]).columns
        categorical_cols = [col for col in categorical_cols if col != target]
        
        for col in categorical_cols:
            try:
                feature_series = sample_df_aligned[col].dropna()
                
                if len(feature_series) < 10 or feature_series.nunique() <= 1:
                    continue
                    
                # Éviter les colonnes avec trop de catégories
                if feature_series.nunique() > min(50, len(feature_series) * 0.5):
                    continue
                
                try:
                    # ANOVA pour tester la différence de moyennes entre groupes
                    if SCIPY_AVAILABLE and pd.api.types.is_numeric_dtype(target_series):
                        groups = []
                        for value in feature_series.unique():
                            if pd.notna(value):
                                group_data = target_series[sample_df_aligned[col] == value].dropna()
                                if len(group_data) > 0:
                                    groups.append(group_data)
                        
                        if len(groups) > 1 and all(len(g) > 0 for g in groups):
                            _, p_val = f_oneway(*groups)
                            if not np.isnan(p_val) and p_val < p_value_threshold:
                                features["categorical"].append(col)
                    else:
                        # Pour target catégorielle ou sans scipy, utiliser une heuristique simple
                        if feature_series.nunique() >= 2:
                            features["categorical"].append(col)
                                
                except Exception as e:
                    logger.debug(f"❌ ANOVA échouée pour {col}: {e}")
                    
            except Exception as e:
                logger.debug(f"❌ Erreur traitement feature catégorielle {col}: {e}")
                continue

        # Nettoyage mémoire
        del sample_df, sample_df_aligned
        gc.collect()

        total_features = len(features["numeric"]) + len(features["categorical"])
        logger.info(f"✅ Analyse pertinence features terminée: {total_features} features pertinentes trouvées")
        
        return features

    except Exception as e:
        logger.error(f"❌ Erreur critique dans get_relevant_features: {e}")
        return {"numeric": [], "categorical": []}

@conditional_cache(use_cache=True)
@safe_execute(fallback_value=[])
@monitor_performance
def detect_useless_columns(
    df: Union[pd.DataFrame, 'dd.DataFrame'],
    threshold_missing: float = 0.6,
    min_unique_ratio: float = 0.0001,
    max_sample_size: int = 50000
) -> List[str]:
    """
    Détecte les colonnes inutiles - version robuste.
    """
    try:
        # Validation initiale
        if df is None or len(df.columns) == 0:
            logger.warning("⚠️ DataFrame vide ou sans colonnes")
            return []
        
        if hasattr(df, 'empty') and df.empty:
            return []

        # Échantillonnage
        sample_df = safe_sample(df, sample_frac=0.05, max_rows=max_sample_size)
        if sample_df.empty or len(sample_df.columns) == 0:
            return []

        useless_columns = set()

        # 1. Colonnes avec trop de NaN
        try:
            missing_ratios = sample_df.isna().mean()
            high_missing = missing_ratios[missing_ratios > threshold_missing].index
            useless_columns.update(high_missing)
            logger.debug(f"Colonnes avec NaN excessifs: {len(high_missing)}")
        except Exception as e:
            logger.warning(f"⚠️ Vérification NaN échouée: {e}")

        # 2. Colonnes constantes
        try:
            nunique_counts = sample_df.nunique(dropna=True)
            constant_cols = nunique_counts[nunique_counts <= 1].index
            useless_columns.update(constant_cols)
            logger.debug(f"Colonnes constantes: {len(constant_cols)}")
        except Exception as e:
            logger.warning(f"⚠️ Vérification constantes échouée: {e}")

        # 3. Colonnes à faible variance (version robuste)
        try:
            numeric_cols = sample_df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                if col in useless_columns:  # Déjà détecté
                    continue
                    
                std_val = sample_df[col].std()
                if pd.isna(std_val):
                    continue
                    
                # Variance nulle
                if std_val == 0:
                    useless_columns.add(col)
                    continue
                    
                # Faible coefficient de variation
                mean_val = sample_df[col].mean()
                if mean_val != 0 and abs(std_val / mean_val) < min_unique_ratio:
                    useless_columns.add(col)
                    
            logger.debug(f"Colonnes numériques vérifiées: {len(numeric_cols)}")
        except Exception as e:
            logger.warning(f"⚠️ Vérification variance échouée: {e}")

        # 4. Validation finale
        valid_columns = [col for col in useless_columns if col in df.columns]
        
        logger.info(f"✅ Colonnes inutiles détectées: {len(valid_columns)}/{len(df.columns)}")
        if valid_columns:
            logger.debug(f"Colonnes: {valid_columns}")
            
        return valid_columns

    except Exception as e:
        logger.error(f"❌ Erreur critique: {e}")
        return []

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={})
@monitor_performance
def compute_global_metrics(df: Union[pd.DataFrame, 'dd.DataFrame']) -> Dict[str, Any]:
    """
    Calcule les métriques globales du dataset.
    """
    try:
        if df is None or df.empty:
            return {}

        # Si c'est un Dask DataFrame, on calcule les métriques de manière distribuée
        if is_dask_dataframe(df):
            n_rows = compute_if_dask(df.shape[0])
            n_cols = compute_if_dask(df.shape[1])
            missing_count = compute_if_dask(df.isnull().sum().sum())
            duplicate_rows = compute_if_dask(df.duplicated().sum())
        else:
            n_rows = len(df)
            n_cols = len(df.columns)
            missing_count = df.isnull().sum().sum()
            duplicate_rows = df.duplicated().sum()

        total_cells = n_rows * n_cols
        missing_percentage = (missing_count / total_cells * 100) if total_cells > 0 else 0

        return {
            'n_rows': n_rows,
            'n_cols': n_cols,
            'missing_count': missing_count,
            'missing_percentage': missing_percentage,
            'duplicate_rows': duplicate_rows,
            'total_cells': total_cells
        }

    except Exception as e:
        logger.error(f"Erreur dans compute_global_metrics: {e}")
        return {}
    
@conditional_cache(use_cache=True)
@safe_execute(fallback_value={})
@monitor_performance
def detect_outliers_iqr(data: pd.Series) -> Dict:
    """Détection des outliers avec méthode IQR - Logique métier"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    return {
        'outliers': outliers,
        'bounds': {'lower': lower_bound, 'upper': upper_bound},
        'stats': {'Q1': Q1, 'Q3': Q3, 'IQR': IQR},
        'count': len(outliers),
        'percentage': (len(outliers) / len(data)) * 100
    }

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={})
@monitor_performance
def calculate_correlation_significance(df: pd.DataFrame, var1: str, var2: str) -> Dict:
    """Calcule la significativité des corrélations - Logique métier"""
    try:
        from scipy.stats import pearsonr, spearmanr
        
        data = df[[var1, var2]].dropna()
        
        if len(data) < 3:
            return {'error': 'Données insuffisantes'}
        
        # Pearson pour relation linéaire
        pearson_corr, pearson_p = pearsonr(data[var1], data[var2])
        
        # Spearman pour relation monotone
        spearman_corr, spearman_p = spearmanr(data[var1], data[var2])
        
        return {
            'pearson': {'correlation': pearson_corr, 'p_value': pearson_p},
            'spearman': {'correlation': spearman_corr, 'p_value': spearman_p},
            'sample_size': len(data)
        }
    except ImportError:
        # Fallback sans scipy
        correlation = data[var1].corr(data[var2])
        return {
            'pearson': {'correlation': correlation, 'p_value': None},
            'spearman': {'correlation': correlation, 'p_value': None},
            'sample_size': len(data),
            'warning': 'SciPy non disponible, calculs limités'
        }

@conditional_cache(use_cache=True)
@safe_execute(fallback_value={})
@monitor_performance  
def chi_square_test(contingency_table: pd.DataFrame) -> Dict:
    """Test du chi-carré pour tables de contingence - Logique métier"""
    try:
        from scipy.stats import chi2_contingency
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        return {
            'chi2': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'significant': p_value < 0.05
        }
    except Exception as e:
        return {'error': str(e)}

# Export des fonctions principales
__all__ = [
    'auto_detect_column_types', # Pour détection types colonnes
    'get_column_profile', # Pour le profil d'une colonne
    'get_data_profile', # Pour le profil global du dataset
    'analyze_columns', # Pour détection colonnes constantes/ID
    'detect_imbalance', # Pour détection déséquilibre des classes
    'get_target_and_task', # Pour détection type tâche ML
    'get_relevant_features', # Pour sélection features pertinentes
    'detect_useless_columns', # Pour détection colonnes inutiles
    'cleanup_memory', # Nettoyage mémoire
    'safe_sample', # Échantillonnage sécurisé
    'optimize_dataframe', # Optimisation DataFrame
    'compute_global_metrics', # Nouvelle fonction pour métriques globales
    'detect_outliers_iqr', # Détection outliers IQR
    'calculate_correlation_significance', # Pour corrélation et significativité
    'chi_square_test' # Test du chi-carré
]