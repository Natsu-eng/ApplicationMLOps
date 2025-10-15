"""
Module de validation des données pour garantir l'intégrité des DataFrames.
Optimisé pour la production avec gestion d'erreurs robuste et logging.
"""
import pandas as pd
import numpy as np
import re
from typing import Union, Optional, Dict, Any, List
import logging
logger = logging.getLogger(__name__)

try:
    import dask.dataframe as dd
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    logger.warning("Dask non disponible, validation limitée à Pandas")

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    logger.warning("Streamlit non disponible, validation sans interface utilisateur")

from src.shared.logging import get_logger
from monitoring.decorators import monitor_performance
from src.config.constants import TRAINING_CONSTANTS, VALIDATION_CONSTANTS, APP_CONSTANTS

logger = get_logger(__name__)

class DataValidator:
    """
    Classe pour valider les DataFrames et leurs colonnes.
    Utilisation en production : validation robuste avec gestion d'erreurs.
    """
    @staticmethod
    @monitor_performance("validate_dataframe")
    def validate_dataframe(df: Union[pd.DataFrame, 'dd.DataFrame'] = None) -> Union[pd.DataFrame, 'dd.DataFrame']:
        """
        Valide un DataFrame pour garantir qu'il est utilisable pour l'analyse.
        
        Args:
            df: DataFrame Pandas ou Dask à valider. Si None et Streamlit disponible, utilise st.session_state.df.
            
        Returns:
            DataFrame validé.
            
        Raises:
            ValueError: Si le DataFrame est invalide ou vide.
        """
        try:
            # Si df est None et Streamlit disponible, tenter de récupérer depuis st.session_state
            if df is None and STREAMLIT_AVAILABLE:
                if 'df' not in st.session_state or st.session_state.df is None:
                    if STREAMLIT_AVAILABLE:
                        st.error("📊 Aucun dataset chargé")
                        st.info("Chargez un dataset depuis la page d'accueil pour commencer l'analyse.")
                        if st.button("🏠 Retour à l'accueil"):
                            st.switch_page("app.py")
                        st.stop()
                    raise ValueError("Aucun DataFrame trouvé dans st.session_state.df")
                df = st.session_state.df
            elif df is None:
                raise ValueError("DataFrame non fourni")

            # Vérifications de base
            if df is None:
                raise ValueError("DataFrame est None")
            
            if DASK_AVAILABLE and isinstance(df, dd.DataFrame):
                # Vérification minimale pour Dask
                if len(df.columns) == 0:
                    raise ValueError("DataFrame Dask vide (aucune colonne)")
                n_rows = df.shape[0].compute()
                if n_rows == 0:
                    raise ValueError("DataFrame Dask vide (aucune ligne)")
                logger.info(f"Validation DataFrame Dask réussie: {n_rows} lignes, {len(df.columns)} colonnes")
                return df
            elif isinstance(df, pd.DataFrame):
                # Vérifications pour Pandas
                if df.empty:
                    raise ValueError("DataFrame Pandas vide")
                if len(df.columns) == 0:
                    raise ValueError("DataFrame sans colonnes")
                
                # Vérification des noms de colonnes
                for col in df.columns:
                    if not DataValidator.is_valid_column_name(col):
                        logger.warning(f"Nom de colonne invalide détecté: {col}")
                
                logger.info(f"Validation DataFrame Pandas réussie: {df.shape[0]} lignes, {df.shape[1]} colonnes")
                return df
            else:
                raise ValueError(f"Type de DataFrame non supporté: {type(df)}")
                
        except Exception as e:
            logger.error(f"Erreur lors de la validation du DataFrame: {e}", exc_info=True)
            if STREAMLIT_AVAILABLE:
                st.error(f"Erreur de validation: {str(e)[:200]}")
                st.stop()
            raise ValueError(f"Validation échouée: {str(e)}")

    @staticmethod
    def is_valid_column_name(name: str) -> bool:
        """
        Vérifie si un nom de colonne est valide (non vide, alphanumérique, caractères autorisés).
        
        Args:
            name: Nom de la colonne à valider.
            
        Returns:
            bool: True si le nom est valide, False sinon.
        """
        try:
            if not isinstance(name, str) or not name.strip():
                logger.warning("Nom de colonne vide ou non-string")
                return False
            
            name = name.strip()
            if len(name) > 128 or len(name) < 1:
                logger.warning(f"Nom de colonne invalide (longueur): {name}")
                return False
            
            if not re.match(r'^[a-zA-Z0-9_\[\]][a-zA-Z0-9_\[\] ]*[a-zA-Z0-9_\[\]]$|^[a-zA-Z0-9_\[\]]$', name):
                logger.warning(f"Caractères invalides dans le nom de colonne: {name}")
                return False
             
            return True
        except Exception as e:
            logger.error(f"Erreur lors de la validation du nom de colonne {name}: {e}")
            return False

    @staticmethod
    @monitor_performance("validate_training_data")
    def validate_training_data(
        X: Union[pd.DataFrame, 'dd.DataFrame'],
        y: Optional[Union[pd.Series, 'dd.Series', np.ndarray]] = None,
        task_type: str = APP_CONSTANTS["DEFAULT_TASK_TYPE"],
        min_samples: int = TRAINING_CONSTANTS["MIN_SAMPLES_REQUIRED"],
        max_missing_ratio: float = TRAINING_CONSTANTS["MAX_MISSING_RATIO"]
    ) -> Dict[str, Any]:
        """
        Valide les données d'entraînement de manière robuste pour le machine learning.
        
        Args:
            X: DataFrame Pandas ou Dask contenant les features.
            y: Série Pandas, Dask ou tableau NumPy contenant la cible (optionnel pour clustering).
            task_type: Type de tâche ML ("classification" ou "clustering").
            min_samples: Nombre minimum d'échantillons requis.
            max_missing_ratio: Ratio maximum de valeurs manquantes autorisé.
            
        Returns:
            Dictionnaire contenant :
            - is_valid: Booléen indiquant si les données sont valides.
            - issues: Liste des erreurs critiques.
            - warnings: Liste des avertissements.
            - samples_count: Nombre d'échantillons.
            - features_count: Nombre de features.
            - data_quality: Métriques de qualité (valeurs manquantes, classes, etc.).
            
        Raises:
            ValueError: Si la validation échoue.
        """
        try:
            validation = {
                "is_valid": True,
                "issues": [],
                "warnings": [],
                "samples_count": 0,
                "features_count": 0,
                "data_quality": {}
            }

            # Validation de base de X
            X = DataValidator.validate_dataframe(X)
            validation["samples_count"] = X.shape[0].compute() if DASK_AVAILABLE and isinstance(X, dd.DataFrame) else len(X)
            validation["features_count"] = len(X.columns)

            # Vérification des dimensions de base
            if validation["samples_count"] < min_samples:
                validation["is_valid"] = False
                validation["issues"].append(f"Trop peu d'échantillons ({validation['samples_count']} < {min_samples})")
                logger.error(validation["issues"][-1])

            if validation["features_count"] < VALIDATION_CONSTANTS["MIN_COLS_REQUIRED"]:
                validation["is_valid"] = False
                validation["issues"].append(f"Nombre de features insuffisant ({validation['features_count']} < {VALIDATION_CONSTANTS['MIN_COLS_REQUIRED']})")
                logger.error(validation["issues"][-1])

            # Vérification des valeurs manquantes
            missing_stats = X.isna().sum()
            if DASK_AVAILABLE and isinstance(X, dd.DataFrame):
                missing_stats = missing_stats.compute()
            total_missing = missing_stats.sum()
            total_elements = validation["samples_count"] * validation["features_count"]
            missing_ratio = total_missing / total_elements if total_elements > 0 else 0
            
            validation["data_quality"]["total_missing"] = int(total_missing)
            validation["data_quality"]["missing_ratio"] = float(missing_ratio)

            if missing_ratio > max_missing_ratio:
                validation["warnings"].append(f"Ratio de valeurs manquantes élevé: {missing_ratio:.1%} (max: {max_missing_ratio:.1%})")
                logger.warning(validation["warnings"][-1])

            # Vérification des noms de colonnes
            for col in X.columns:
                if not DataValidator.is_valid_column_name(col):
                    validation["warnings"].append(f"Nom de colonne invalide: {col}")
                    logger.warning(validation["warnings"][-1])

            # Vérification spécifique au clustering
            if task_type.lower() == 'clustering':
                if y is not None:
                    validation["warnings"].append("Target ignorée pour le clustering")
                    logger.warning(validation["warnings"][-1])
            
            # Vérification de la cible pour supervisé
            elif y is not None:
                # Convertir y en Series si c'est un tableau NumPy
                if isinstance(y, np.ndarray):
                    y = pd.Series(y, index=X.index if isinstance(X, pd.DataFrame) else None)
                
                # Vérification des dimensions
                y_len = y.shape[0].compute() if DASK_AVAILABLE and isinstance(y, dd.Series) else len(y)
                if y_len != validation["samples_count"]:
                    validation["is_valid"] = False
                    validation["issues"].append(f"Dimensions X et y incohérentes: {validation['samples_count']} vs {y_len}")
                    logger.error(validation["issues"][-1])

                # Vérification des valeurs manquantes dans y
                valid_target_count = y.notna().sum().compute() if DASK_AVAILABLE and isinstance(y, dd.Series) else y.notna().sum() if hasattr(y, 'notna') else np.sum(~np.isnan(y))
                validation["data_quality"]["valid_target_count"] = int(valid_target_count)

                if valid_target_count < min_samples:
                    validation["is_valid"] = False
                    validation["issues"].append(f"Trop peu de targets valides ({valid_target_count} < {min_samples})")
                    logger.error(validation["issues"][-1])

                # Vérification des classes pour classification
                if task_type.lower() == 'classification':
                    unique_classes = y.dropna().unique().compute() if DASK_AVAILABLE and isinstance(y, dd.Series) else np.unique(y.dropna()) if hasattr(y, 'dropna') else np.unique(y[~np.isnan(y)])
                    n_classes = len(unique_classes)
                    validation["data_quality"]["n_classes"] = n_classes

                    if n_classes < 2:
                        validation["is_valid"] = False
                        validation["issues"].append("Moins de 2 classes distinctes")
                        logger.error(validation["issues"][-1])
                    elif n_classes > TRAINING_CONSTANTS["MAX_CLASSES"]:
                        validation["warnings"].append(f"Trop de classes ({n_classes} > {TRAINING_CONSTANTS['MAX_CLASSES']})")
                        logger.warning(validation["warnings"][-1])

                # Avertissement pour outliers dans y (si numérique)
                if pd.api.types.is_numeric_dtype(y.dtype):
                    y_clean = y.dropna()
                    if DASK_AVAILABLE and isinstance(y, dd.Series):
                        y_clean = y_clean.compute()
                    if not y_clean.empty:
                        std = y_clean.std()
                        mean = y_clean.mean()
                        if std > 0 and (y_clean.abs() > mean + 5 * std).any():
                            validation["warnings"].append("Valeurs aberrantes potentielles dans la cible")
                            logger.warning(validation["warnings"][-1])

            # Avertissement pour cardinalité élevée dans X
            for col in X.columns:
                try:
                    unique_ratio = (X[col].nunique() / validation["samples_count"]).compute() if DASK_AVAILABLE and isinstance(X, dd.DataFrame) else X[col].nunique() / validation["samples_count"]
                    if unique_ratio > 0.9:
                        validation["warnings"].append(f"Colonne '{col}' a une cardinalité élevée ({unique_ratio:.1%})")
                        logger.warning(validation["warnings"][-1])
                except Exception as e:
                    logger.debug(f"Erreur lors de l'analyse de cardinalité pour {col}: {e}")

            # Avertissement pour outliers dans les colonnes numériques de X
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    col_data = X[col].dropna()
                    if DASK_AVAILABLE and isinstance(X, dd.DataFrame):
                        col_data = col_data.compute()
                    if not col_data.empty:
                        std = col_data.std()
                        mean = col_data.mean()
                        if std > 0 and (col_data.abs() > mean + 5 * std).any():
                            validation["warnings"].append(f"Colonne '{col}' contient des valeurs aberrantes potentielles")
                            logger.warning(validation["warnings"][-1])
                except Exception as e:
                    logger.debug(f"Erreur lors de l'analyse des outliers pour {col}: {e}")

            if validation["issues"]:
                validation["is_valid"] = False
                if STREAMLIT_AVAILABLE:
                    for issue in validation["issues"]:
                        st.error(issue)
                    st.stop()

            logger.info(f"✅ Validation données: {validation['samples_count']} échantillons, "
                        f"{validation['features_count']} features, {len(validation['issues'])} issues")
            return validation

        except Exception as e:
            validation["is_valid"] = False
            validation["issues"].append(f"Erreur critique lors de la validation: {str(e)}")
            logger.error(validation["issues"][-1], exc_info=True)
            if STREAMLIT_AVAILABLE:
                st.error(validation["issues"][-1])
                st.stop()
            raise ValueError(validation["issues"][-1])

    @staticmethod
    @monitor_performance("validate_dataframe_for_ml")
    def validate_dataframe_for_ml(
        df: Union[pd.DataFrame, 'dd.DataFrame']
    ) -> Dict[str, Any]:
        """
        Valide un DataFrame pour l'entraînement ML, en vérifiant les critères minimaux.
        
        Args:
            df: DataFrame Pandas ou Dask à valider.
            
        Returns:
            Dictionnaire contenant :
            - is_valid: Booléen indiquant si le DataFrame est valide.
            - issues: Liste des erreurs critiques.
            - warnings: Liste des avertissements.
            - stats: Statistiques du DataFrame (n_rows, n_cols, memory_mb, missing_ratio).
        """
        try:
            validation = {
                "is_valid": True,
                "issues": [],
                "warnings": [],
                "stats": {
                    "n_rows": 0,
                    "n_cols": 0,
                    "memory_mb": 0.0,
                    "missing_ratio": 0.0
                }
            }

            # Validation de base du DataFrame
            df = DataValidator.validate_dataframe(df)
            validation["stats"]["n_rows"] = df.shape[0].compute() if DASK_AVAILABLE and isinstance(df, dd.DataFrame) else len(df)
            validation["stats"]["n_cols"] = len(df.columns)

            # Vérification des dimensions minimales
            if validation["stats"]["n_rows"] < VALIDATION_CONSTANTS["MIN_ROWS_REQUIRED"]:
                validation["is_valid"] = False
                validation["issues"].append(f"Trop peu de lignes ({validation['stats']['n_rows']} < {VALIDATION_CONSTANTS['MIN_ROWS_REQUIRED']})")
                logger.error(validation["issues"][-1])

            if validation["stats"]["n_cols"] < VALIDATION_CONSTANTS["MIN_COLS_REQUIRED"]:
                validation["is_valid"] = False
                validation["issues"].append(f"Trop peu de colonnes ({validation['stats']['n_cols']} < {VALIDATION_CONSTANTS['MIN_COLS_REQUIRED']})")
                logger.error(validation["issues"][-1])

            # Vérification des valeurs manquantes
            missing_stats = df.isna().sum()
            if DASK_AVAILABLE and isinstance(df, dd.DataFrame):
                missing_stats = missing_stats.compute()
            total_missing = missing_stats.sum()
            total_elements = validation["stats"]["n_rows"] * validation["stats"]["n_cols"]
            missing_ratio = total_missing / total_elements if total_elements > 0 else 0
            validation["stats"]["missing_ratio"] = float(missing_ratio)

            if missing_ratio > VALIDATION_CONSTANTS["MAX_MISSING_RATIO"]:
                validation["is_valid"] = False
                validation["issues"].append(f"Ratio de valeurs manquantes trop élevé: {missing_ratio:.1%} (max: {VALIDATION_CONSTANTS['MAX_MISSING_RATIO']:.1%})")
                logger.error(validation["issues"][-1])
            elif missing_ratio > VALIDATION_CONSTANTS["MISSING_WARNING_THRESHOLD"]:
                validation["warnings"].append(f"Ratio de valeurs manquantes élevé: {missing_ratio:.1%} (seuil d'avertissement: {VALIDATION_CONSTANTS['MISSING_WARNING_THRESHOLD']:.1%})")
                logger.warning(validation["warnings"][-1])

            # Calcul de l'utilisation mémoire
            if not DASK_AVAILABLE or not isinstance(df, dd.DataFrame):
                try:
                    memory_bytes = df.memory_usage(deep=True).sum()
                    validation["stats"]["memory_mb"] = memory_bytes / (1024 ** 2)
                except Exception as e:
                    logger.debug(f"Erreur calcul mémoire: {e}")
                    validation["stats"]["memory_mb"] = 0.0
            else:
                validation["stats"]["memory_mb"] = 0.0  # Non calculé pour Dask

            # Vérification des colonnes constantes ou identifiantes
            for col in df.columns:
                try:
                    n_unique = df[col].nunique().compute() if DASK_AVAILABLE and isinstance(df, dd.DataFrame) else df[col].nunique()
                    if n_unique == 1:
                        validation["warnings"].append(f"Colonne '{col}' est constante (1 valeur unique)")
                        logger.warning(validation["warnings"][-1])
                    elif n_unique == validation["stats"]["n_rows"]:
                        validation["warnings"].append(f"Colonne '{col}' est un identifiant (toutes valeurs uniques)")
                        logger.warning(validation["warnings"][-1])
                except Exception as e:
                    logger.debug(f"Erreur analyse cardinalité pour {col}: {e}")

            if validation["issues"]:
                validation["is_valid"] = False
                if STREAMLIT_AVAILABLE:
                    for issue in validation["issues"]:
                        st.error(issue)
                    st.stop()

            logger.info(f"✅ Validation DataFrame pour ML: {validation['stats']['n_rows']} lignes, "
                        f"{validation['stats']['n_cols']} colonnes, {len(validation['issues'])} issues")
            return validation

        except Exception as e:
            validation["is_valid"] = False
            validation["issues"].append(f"Erreur critique lors de la validation: {str(e)}")
            logger.error(validation["issues"][-1], exc_info=True)
            if STREAMLIT_AVAILABLE:
                st.error(validation["issues"][-1])
                st.stop()
            raise ValueError(validation["issues"][-1])

    @staticmethod
    @monitor_performance("validate_clustering_features")
    def validate_clustering_features(
        df: Union[pd.DataFrame, 'dd.DataFrame'],
        features: List[str]
    ) -> Dict[str, Any]:
        """
        Valide les features pour le clustering, en vérifiant qu'elles sont numériques et adaptées.
        
        Args:
            df: DataFrame Pandas ou Dask contenant les données.
            features: Liste des colonnes à valider pour le clustering.
            
        Returns:
            Dictionnaire contenant :
            - is_valid: Booléen indiquant si les features sont valides.
            - valid_features: Liste des features valides.
            - issues: Liste des erreurs critiques.
            - warnings: Liste des avertissements.
            - suggested_features: Liste des features recommandées si certaines sont invalides.
        """
        try:
            validation = {
                "is_valid": True,
                "valid_features": [],
                "issues": [],
                "warnings": [],
                "suggested_features": []
            }

            # Validation de base du DataFrame
            df = DataValidator.validate_dataframe(df)

            # Vérification des features fournies
            if not features:
                validation["is_valid"] = False
                validation["issues"].append("Aucune feature fournie pour le clustering")
                logger.error(validation["issues"][-1])

            # Vérifier que les features existent dans le DataFrame
            valid_features = [col for col in features if col in df.columns]
            if not valid_features:
                validation["is_valid"] = False
                validation["issues"].append("Aucune feature valide fournie (noms incorrects)")
                logger.error(validation["issues"][-1])
            else:
                validation["valid_features"] = valid_features

            # Vérifier que les features sont numériques
            for col in valid_features:
                if not pd.api.types.is_numeric_dtype(df[col].dtype):
                    validation["is_valid"] = False
                    validation["issues"].append(f"Colonne '{col}' n'est pas numérique")
                    logger.error(validation["issues"][-1])
                    validation["valid_features"].remove(col)

            # Vérification des colonnes constantes
            for col in validation["valid_features"]:
                try:
                    n_unique = df[col].nunique().compute() if DASK_AVAILABLE and isinstance(df, dd.DataFrame) else df[col].nunique()
                    if n_unique == 1:
                        validation["warnings"].append(f"Colonne '{col}' est constante (1 valeur unique)")
                        logger.warning(validation["warnings"][-1])
                        validation["valid_features"].remove(col)
                except Exception as e:
                    logger.debug(f"Erreur analyse cardinalité pour {col}: {e}")

            # Recommandation de features numériques si nécessaire
            if not validation["valid_features"] and validation["is_valid"]:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                validation["suggested_features"] = numeric_cols[:TRAINING_CONSTANTS["MAX_FEATURES"]]
                if numeric_cols:
                    validation["valid_features"] = numeric_cols[:TRAINING_CONSTANTS["MAX_FEATURES"]]
                    validation["warnings"].append(f"Features non valides, suggestion: {', '.join(validation['suggested_features'][:5])}")
                    logger.warning(validation["warnings"][-1])
                else:
                    validation["is_valid"] = False
                    validation["issues"].append("Aucune colonne numérique disponible pour le clustering")
                    logger.error(validation["issues"][-1])

            # Vérification du nombre minimum de features
            if len(validation["valid_features"]) < VALIDATION_CONSTANTS["MIN_COLS_REQUIRED"]:
                validation["is_valid"] = False
                validation["issues"].append(f"Nombre de features insuffisant ({len(validation['valid_features'])} < {VALIDATION_CONSTANTS['MIN_COLS_REQUIRED']})")
                logger.error(validation["issues"][-1])

            # Vérification des valeurs manquantes
            if validation["valid_features"]:
                missing_stats = df[validation["valid_features"]].isna().sum()
                if DASK_AVAILABLE and isinstance(df, dd.DataFrame):
                    missing_stats = missing_stats.compute()
                total_missing = missing_stats.sum()
                total_elements = len(validation["valid_features"]) * (df.shape[0].compute() if DASK_AVAILABLE and isinstance(df, dd.DataFrame) else len(df))
                missing_ratio = total_missing / total_elements if total_elements > 0 else 0
                if missing_ratio > VALIDATION_CONSTANTS["MAX_MISSING_RATIO"]:
                    validation["warnings"].append(f"Ratio de valeurs manquantes élevé dans les features: {missing_ratio:.1%}")
                    logger.warning(validation["warnings"][-1])

            if validation["issues"]:
                validation["is_valid"] = False
                if STREAMLIT_AVAILABLE:
                    for issue in validation["issues"]:
                        st.error(issue)
                    st.stop()

            logger.info(f"✅ Validation features clustering: {len(validation['valid_features'])} features valides, {len(validation['issues'])} issues")
            return validation

        except Exception as e:
            validation["is_valid"] = False
            validation["issues"].append(f"Erreur critique lors de la validation des features: {str(e)}")
            logger.error(validation["issues"][-1], exc_info=True)
            if STREAMLIT_AVAILABLE:
                st.error(validation["issues"][-1])
                st.stop()
            raise ValueError(validation["issues"][-1])

__all__ = ['DataValidator']