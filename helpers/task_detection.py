import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
from src.shared.logging import get_logger
from src.data.data_analysis import get_target_and_task, detect_imbalance

logger = get_logger(__name__)

def safe_get_task_type(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """Détection sécurisée du type de tâche ML avec gestion d'erreurs"""
    try:
        if not target_column or target_column not in df.columns:
            return {
                "task_type": "unknown", 
                "n_classes": 0, 
                "error": "Colonne cible invalide", 
                "warnings": []
            }
        
        if df[target_column].nunique() == len(df):
            return {
                "task_type": "unknown", 
                "n_classes": 0, 
                "error": "Variable cible est un identifiant", 
                "warnings": []
            }
        
        # Utiliser la fonction existante de data_analysis
        result_dict = get_target_and_task(df, target_column)
        task_type = result_dict.get("task", "unknown")
        target_type = result_dict.get("target_type", "unknown")
        
        warnings = []
        n_classes = 0
        
        if task_type == "classification":
            n_classes = df[target_column].nunique()
            if n_classes > 50:  # Seuil pour trop de classes
                return {
                    "task_type": "unknown", 
                    "n_classes": n_classes, 
                    "error": f"Trop de classes ({n_classes}) pour la classification", 
                    "warnings": []
                }
            if n_classes < 2:
                warnings.append("Variable cible a moins de 2 classes")
        
        logger.info(f"Type de tâche détecté: {task_type} avec {n_classes} classes")
        
        return {
            "task_type": task_type, 
            "target_type": target_type, 
            "n_classes": n_classes, 
            "error": None, 
            "warnings": warnings
        }
        
    except Exception as e:
        logger.error(f"Échec détection type tâche: {str(e)[:100]}")
        return {
            "task_type": "unknown", 
            "n_classes": 0, 
            "error": str(e)[:100], 
            "warnings": []
        }

def detect_task_from_dataframe(df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
    """Détection automatique du type de tâche avec analyse des données"""
    analysis = {
        "suggested_task": "unknown",
        "confidence": 0.0,
        "reasoning": [],
        "target_suggestions": [],
        "warnings": []
    }
    
    try:
        # Si une colonne cible est fournie
        if target_column and target_column in df.columns:
            task_info = safe_get_task_type(df, target_column)
            analysis["suggested_task"] = task_info["task_type"]
            analysis["confidence"] = 0.9 if task_info["error"] is None else 0.3
            analysis["reasoning"].append(f"Colonne cible '{target_column}' analysée")
            
            if task_info["error"]:
                analysis["warnings"].append(task_info["error"])
            analysis["warnings"].extend(task_info["warnings"])
            
            return analysis
        
        # Analyse sans colonne cible (clustering)
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) >= 2:
            analysis["suggested_task"] = "clustering"
            analysis["confidence"] = 0.7
            analysis["reasoning"].append(f"{len(numeric_cols)} variables numériques détectées - adapté au clustering")
        else:
            analysis["suggested_task"] = "unknown"
            analysis["confidence"] = 0.1
            analysis["warnings"].append("Pas assez de variables numériques pour le clustering")
        
        # Suggestions de colonnes cibles potentielles
        potential_targets = []
        for col in df.columns:
            if df[col].dtype in ['object', 'category'] and df[col].nunique() <= 10:
                potential_targets.append((col, df[col].nunique(), "classification"))
            elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
                potential_targets.append((col, df[col].nunique(), "regression"))
        
        # Trier par pertinence
        potential_targets.sort(key=lambda x: x[1] if x[2] == "classification" else 1/x[1])
        analysis["target_suggestions"] = [target[0] for target in potential_targets[:5]]
        
    except Exception as e:
        logger.error(f"Erreur détection tâche: {str(e)[:100]}")
        analysis["warnings"].append(f"Erreur analyse: {str(e)[:100]}")
    
    return analysis

def validate_task_configuration(df: pd.DataFrame, task_type: str, target_column: str = None) -> Dict[str, Any]:
    """Valide la configuration de tâche ML proposée"""
    validation = {
        "is_valid": True,
        "issues": [],
        "warnings": [],
        "suggestions": []
    }
    
    try:
        if task_type in ["classification", "regression"]:
            if not target_column:
                validation["is_valid"] = False
                validation["issues"].append("Colonne cible requise pour l'apprentissage supervisé")
                return validation
            
            if target_column not in df.columns:
                validation["is_valid"] = False
                validation["issues"].append(f"Colonne cible '{target_column}' introuvable")
                return validation
            
            # Validation spécifique selon le type de tâche
            if task_type == "classification":
                n_classes = df[target_column].nunique()
                if n_classes == 1:
                    validation["is_valid"] = False
                    validation["issues"].append("Variable cible n'a qu'une seule classe")
                elif n_classes > 50:
                    validation["warnings"].append(f"Nombre élevé de classes ({n_classes}) - considérez la régression")
                
                # Vérifier le déséquilibre
                imbalance_info = detect_imbalance(df, target_column)
                if imbalance_info.get("is_imbalanced", False):
                    validation["warnings"].append(
                        f"Déséquilibre détecté (ratio: {imbalance_info.get('imbalance_ratio', 'N/A'):.2f})"
                    )
            
            elif task_type == "regression":
                if not pd.api.types.is_numeric_dtype(df[target_column]):
                    validation["is_valid"] = False
                    validation["issues"].append("Variable cible doit être numérique pour la régression")
                elif df[target_column].std() == 0:
                    validation["warnings"].append("Variable cible constante")
        
        elif task_type == "clustering":
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                validation["is_valid"] = False
                validation["issues"].append("Au moins 2 variables numériques requises pour le clustering")
            elif len(numeric_cols) > 100:
                validation["warnings"].append("Nombre élevé de variables - considérez la réduction de dimension")
        
        else:
            validation["is_valid"] = False
            validation["issues"].append(f"Type de tâche non supporté: {task_type}")
        
    except Exception as e:
        validation["is_valid"] = False
        validation["issues"].append(f"Erreur validation: {str(e)[:100]}")
        logger.error(f"Erreur validation tâche: {str(e)[:100]}")
    
    return validation

def get_task_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Génère des recommandations de tâches ML basées sur les données"""
    recommendations = []
    
    try:
        # Analyser les colonnes
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Recommandation 1: Clustering
        if len(numeric_cols) >= 2:
            recommendations.append({
                "task_type": "clustering",
                "confidence": "high",
                "reason": f"{len(numeric_cols)} variables numériques disponibles",
                "required_columns": list(numeric_cols)[:5]
            })
        
        # Recommandation 2: Classification
        potential_class_targets = [
            col for col in categorical_cols 
            if 2 <= df[col].nunique() <= 20
        ]
        if potential_class_targets:
            best_target = min(potential_class_targets, key=lambda x: df[x].nunique())
            recommendations.append({
                "task_type": "classification",
                "confidence": "medium",
                "reason": f"Variable catégorielle '{best_target}' avec {df[best_target].nunique()} classes",
                "target_suggestion": best_target
            })
        
        # Recommandation 3: Régression
        potential_reg_targets = [
            col for col in numeric_cols 
            if df[col].nunique() > 10 and df[col].std() > 0
        ]
        if potential_reg_targets:
            best_target = max(potential_reg_targets, key=lambda x: df[x].nunique())
            recommendations.append({
                "task_type": "regression", 
                "confidence": "medium",
                "reason": f"Variable numérique '{best_target}' avec bonne variance",
                "target_suggestion": best_target
            })
        
    except Exception as e:
        logger.error(f"Erreur génération recommandations: {str(e)[:100]}")
    
    return recommendations