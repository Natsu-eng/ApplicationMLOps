"""
Module d'analyse des performances par type d'anomalie.
Fournit des métriques granulaires et recommandations spécifiques.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

from src.config.anomaly_taxonomy import ANOMALY_TAXONOMY, DETECTION_DIFFICULTY_SCORES, BUSINESS_IMPACT_WEIGHTS

class AnomalyTypeAnalyzer:
    """
    Analyseur de performances par type d'anomalie spécifique.
    """
    
    def __init__(self, anomaly_metadata: Optional[Dict] = None):
        self.anomaly_metadata = anomaly_metadata or {}
        self.metrics_by_type = {}
        
    def compute_metrics_by_anomaly_type(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      anomaly_types: List[str], threshold: float = 0.5) -> Dict[str, Any]:
        """
        Calcule les métriques pour chaque type d'anomalie spécifique.
        
        Args:
            y_true: Labels réels (1 = anomalie, 0 = normal)
            y_pred: Scores de prédiction
            anomaly_types: Liste des types d'anomalie pour chaque échantillon
            threshold: Seuil de classification
            
        Returns:
            Dict avec métriques par type d'anomalie
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        y_pred_binary = (y_pred > threshold).astype(int)
        results = {}
        
        # Métriques globales d'abord
        results["global"] = {
            "precision": precision_score(y_true, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true, y_pred_binary, zero_division=0),
            "f1_score": f1_score(y_true, y_pred_binary, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred_binary),
            "sample_count": len(y_true),
            "anomaly_count": np.sum(y_true)
        }
        
        # Métriques par type d'anomalie
        unique_types = set(anomaly_types)
        
        for anomaly_type in unique_types:
            if anomaly_type == "normal":
                continue
                
            # Masque pour ce type d'anomalie spécifique
            type_mask = np.array([t == anomaly_type for t in anomaly_types])
            anomaly_mask = y_true == 1
            
            # Échantillons avec cette anomalie spécifique
            type_anomaly_mask = type_mask & anomaly_mask
            
            if np.sum(type_anomaly_mask) > 0:  # Au moins une occurrence
                type_metrics = {
                    "precision": precision_score(y_true[type_anomaly_mask], y_pred_binary[type_anomaly_mask], zero_division=0),
                    "recall": recall_score(y_true[type_anomaly_mask], y_pred_binary[type_anomaly_mask], zero_division=0),
                    "f1_score": f1_score(y_true[type_anomaly_mask], y_pred_binary[type_anomaly_mask], zero_division=0),
                    "sample_count": np.sum(type_anomaly_mask),
                    "avg_confidence": np.mean(y_pred[type_anomaly_mask]) if np.sum(type_anomaly_mask) > 0 else 0,
                    "detection_rate": np.mean(y_pred_binary[type_anomaly_mask]) if np.sum(type_anomaly_mask) > 0 else 0
                }
                
                # Ajouter la catégorie
                type_metrics["category"] = self._get_anomaly_category(anomaly_type)
                type_metrics["display_name"] = self._get_display_name(anomaly_type)
                
                results[anomaly_type] = type_metrics
        
        self.metrics_by_type = results
        return results
    
    def generate_type_specific_recommendations(self, metrics_by_type: Dict) -> List[Dict]:
        """
        Génère des recommandations spécifiques pour chaque type d'anomalie.
        
        Args:
            metrics_by_type: Métriques calculées par type d'anomalie
            
        Returns:
            Liste de recommandations structurées
        """
        recommendations = []
        
        for anomaly_type, metrics in metrics_by_type.items():
            if anomaly_type == "global":
                continue
                
            recall = metrics.get("recall", 0)
            precision = metrics.get("precision", 0)
            sample_count = metrics.get("sample_count", 0)
            display_name = metrics.get("display_name", anomaly_type)
            
            # Recommandations basées sur le recall
            if recall < 0.3:
                rec = self._get_low_recall_recommendation(anomaly_type, display_name, recall, sample_count)
                recommendations.append(rec)
            elif recall < 0.6:
                rec = self._get_medium_recall_recommendation(anomaly_type, display_name, recall, sample_count)
                recommendations.append(rec)
                
            # Recommandations basées sur la précision
            if precision < 0.4:
                rec = self._get_low_precision_recommendation(anomaly_type, display_name, precision, sample_count)
                recommendations.append(rec)
        
        # Recommandations globales basées sur les patterns
        global_recs = self._get_global_recommendations(metrics_by_type)
        recommendations.extend(global_recs)
        
        return recommendations
    
    def _get_low_recall_recommendation(self, anomaly_type: str, display_name: str, recall: float, sample_count: int) -> Dict:
        """Recommandations pour recall très faible."""
        base_msg = f"**{display_name}** : Recall très faible ({recall:.1%}) - Le modèle rate la majorité de ces anomalies"
        
        if anomaly_type == "scratch":
            return {
                "type": "critical",
                "category": "structural",
                "anomaly_type": anomaly_type,
                "message": f"{base_msg}. Augmentez le contraste et utilisez des filtres de détection de contours.",
                "action": "Preprocessing amélioré",
                "priority": "high"
            }
        elif anomaly_type == "crack":
            return {
                "type": "critical", 
                "category": "structural",
                "anomaly_type": anomaly_type,
                "message": f"{base_msg}. Les microfissures nécessitent un réseau plus profond avec attention aux textures.",
                "action": "Architecture modèle",
                "priority": "high"
            }
        elif anomaly_type == "discoloration":
            return {
                "type": "warning",
                "category": "visual", 
                "anomaly_type": anomaly_type,
                "message": f"{base_msg}. Enrichissez l'augmentation avec variations de couleur et balance des blancs.",
                "action": "Augmentation données",
                "priority": "medium"
            }
        else:
            return {
                "type": "warning",
                "category": self._get_anomaly_category(anomaly_type),
                "anomaly_type": anomaly_type,
                "message": f"{base_msg}. Considérez plus d'échantillons d'entraînement pour ce type spécifique.",
                "action": "Collecte données",
                "priority": "medium" if sample_count < 50 else "low"
            }
    
    def _get_medium_recall_recommendation(self, anomaly_type: str, display_name: str, recall: float, sample_count: int) -> Dict:
        """Recommandations pour recall moyen."""
        base_msg = f"**{display_name}** : Recall modéré ({recall:.1%}) - Amélioration possible"
        
        if anomaly_type in ["stain", "contamination"]:
            return {
                "type": "info",
                "category": "visual",
                "anomaly_type": anomaly_type,
                "message": f"{base_msg}. Ajoutez des variations d'éclairage et d'angles de vue en augmentation.",
                "action": "Augmentation données",
                "priority": "medium"
            }
        else:
            return {
                "type": "info",
                "category": self._get_anomaly_category(anomaly_type),
                "anomaly_type": anomaly_type,
                "message": f"{base_msg}. Ajustez le seuil de classification ou les poids de classe.",
                "action": "Optimisation seuil",
                "priority": "low"
            }
    
    def _get_low_precision_recommendation(self, anomaly_type: str, display_name: str, precision: float, sample_count: int) -> Dict:
        """Recommandations pour précision faible."""
        base_msg = f"**{display_name}** : Précision faible ({precision:.1%}) - Trop de faux positifs"
        
        if anomaly_type == "blur":
            return {
                "type": "warning",
                "category": "visual",
                "anomaly_type": anomaly_type,
                "message": f"{base_msg}. Les reflets et ombres sont confondus avec du flou. Améliorez l'éclairage des prises de vue.",
                "action": "Conditions acquisition",
                "priority": "medium"
            }
        else:
            return {
                "type": "info",
                "category": self._get_anomaly_category(anomaly_type),
                "anomaly_type": anomaly_type,
                "message": f"{base_msg}. Augmentez le seuil de classification pour ce type spécifique.",
                "action": "Seuil adaptatif",
                "priority": "low"
            }
    
    def _get_global_recommendations(self, metrics_by_type: Dict) -> List[Dict]:
        """Recommandations globales basées sur les patterns."""
        recommendations = []
        
        # Analyser les performances par catégorie
        category_performance = defaultdict(list)
        for anomaly_type, metrics in metrics_by_type.items():
            if anomaly_type != "global" and "category" in metrics:
                category_performance[metrics["category"]].append(metrics.get("recall", 0))
        
        # Recommandations par catégorie
        for category, recalls in category_performance.items():
            avg_recall = np.mean(recalls) if recalls else 0
            category_name = ANOMALY_TAXONOMY[category]["name"] if category in ANOMALY_TAXONOMY else category
            
            if avg_recall < 0.5:
                recommendations.append({
                    "type": "critical",
                    "category": category,
                    "anomaly_type": "all",
                    "message": f"**{category_name}** : Performance globale faible (recall moyen: {avg_recall:.1%}). Revoyez la stratégie pour cette catégorie.",
                    "action": "Stratégie catégorie",
                    "priority": "high"
                })
        
        return recommendations
    
    def _get_anomaly_category(self, anomaly_type: str) -> str:
        """Retourne la catégorie d'une anomalie."""
        for category_id, category in ANOMALY_TAXONOMY.items():
            if anomaly_type in category["types"]:
                return category_id
        return "unknown"
    
    def _get_display_name(self, anomaly_type: str) -> str:
        """Retourne le nom d'affichage d'une anomalie."""
        for category in ANOMALY_TAXONOMY.values():
            if anomaly_type in category["types"]:
                return category["types"][anomaly_type]["name"]
        return anomaly_type
    
    def create_performance_heatmap(self, metrics_by_type: Dict) -> go.Figure:
        """
        Crée une heatmap des performances par type d'anomalie.
        
        Args:
            metrics_by_type: Métriques calculées par type d'anomalie
            
        Returns:
            Figure Plotly de la heatmap
        """
        categories = []
        anomaly_names = []
        recall_scores = []
        precision_scores = []
        
        for anomaly_type, metrics in metrics_by_type.items():
            if anomaly_type == "global":
                continue
                
            category = metrics.get("category", "unknown")
            display_name = metrics.get("display_name", anomaly_type)
            
            categories.append(ANOMALY_TAXONOMY[category]["name"] if category in ANOMALY_TAXONOMY else category)
            anomaly_names.append(display_name)
            recall_scores.append(metrics.get("recall", 0))
            precision_scores.append(metrics.get("precision", 0))
        
        # Créer la heatmap pour le recall
        fig = go.Figure(data=go.Heatmap(
            z=[recall_scores],
            x=anomaly_names,
            y=["Recall"],
            text=[[f"{v:.1%}" for v in recall_scores]],
            texttemplate="%{text}",
            textfont={"size": 12},
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            hoverinfo="x+z",
            showscale=True
        ))
        
        fig.update_layout(
            title="📊 Performance de Recall par Type d'Anomalie",
            xaxis_title="Type d'Anomalie",
            yaxis_title="Métrique",
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_category_summary(self, metrics_by_type: Dict) -> pd.DataFrame:
        """
        Crée un résumé des performances par catégorie.
        
        Args:
            metrics_by_type: Métriques calculées par type d'anomalie
            
        Returns:
            DataFrame avec le résumé par catégorie
        """
        category_data = defaultdict(lambda: {"recalls": [], "precisions": [], "count": 0})
        
        for anomaly_type, metrics in metrics_by_type.items():
            if anomaly_type == "global":
                continue
                
            category = metrics.get("category", "unknown")
            category_data[category]["recalls"].append(metrics.get("recall", 0))
            category_data[category]["precisions"].append(metrics.get("precision", 0))
            category_data[category]["count"] += 1
        
        summary_rows = []
        for category, data in category_data.items():
            if data["recalls"]:
                category_name = ANOMALY_TAXONOMY[category]["name"] if category in ANOMALY_TAXONOMY else category
                summary_rows.append({
                    "Catégorie": category_name,
                    "Types d'Anomalies": data["count"],
                    "Recall Moyen": f"{np.mean(data['recalls']):.1%}",
                    "Précision Moyenne": f"{np.mean(data['precisions']):.1%}",
                    "Recall Min": f"{np.min(data['recalls']):.1%}",
                    "Recall Max": f"{np.max(data['recalls']):.1%}"
                })
        
        return pd.DataFrame(summary_rows)

# Fonctions utilitaires
def load_anomaly_metadata(file_path: str) -> Dict:
    """
    Charge les métadonnées d'anomalie depuis un fichier.
    
    Args:
        file_path: Chemin vers le fichier de métadonnées
        
    Returns:
        Dictionnaire des métadonnées
    """
    import json
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Erreur chargement métadonnées: {e}")
        return {}

def validate_anomaly_types(anomaly_types: List[str]) -> List[str]:
    """
    Valide et filtre les types d'anomalies selon la taxonomie.
    
    Args:
        anomaly_types: Liste des types d'anomalie
        
    Returns:
        Liste des types valides
    """
    valid_types = []
    for anomaly_type in anomaly_types:
        if any(anomaly_type in category["types"] for category in ANOMALY_TAXONOMY.values()):
            valid_types.append(anomaly_type)
        elif anomaly_type == "normal":
            valid_types.append(anomaly_type)
        else:
            print(f"Type d'anomalie non reconnu: {anomaly_type}")
    
    return valid_types