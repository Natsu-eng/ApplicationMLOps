"""
Composants UI réutilisables pour l'interface ML Training.
Fonctions helpers pour le design moderne
Séparation logique/présentation
Version: 1.0
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from typing import Dict, Any, List


class UIComponents:
    """Composants UI réutilisables"""
    
    @staticmethod
    def create_modern_donut_chart(
        class_counts: pd.Series, 
        imbalance_level: Dict[str, str]
    ) -> go.Figure:
        """
        Crée un donut chart moderne pour visualiser la distribution des classes.
        
        Args:
            class_counts: Series avec les comptages par classe
            imbalance_level: Dict avec info sur le niveau de déséquilibre
            
        Returns:
            Figure Plotly
        """
        colors = pc.qualitative.Pastel
        labels = [f"Classe {cls}" for cls in class_counts.index]
        values = class_counts.values
        
        fig = go.Figure()
        
        # Donut chart principal
        fig.add_trace(go.Pie(
            labels=labels,
            values=values,
            hole=0.6,
            marker=dict(colors=colors),
            textinfo='percent+label',
            textposition='inside',
            insidetextorientation='radial',
            hovertemplate='<b>%{label}</b><br>Échantillons: %{value}<br>Pourcentage: %{percent}<extra></extra>',
            showlegend=False
        ))
        
        # Annotation centrale
        total = sum(values)
        fig.add_annotation(
            text=f"<b>{total:,}</b><br>Échantillons<br>{len(class_counts)} Classes<br>{imbalance_level['label']}",
            x=0.5, y=0.5,
            font=dict(size=14, color=imbalance_level['color']),
            showarrow=False
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Arial, sans-serif")
        )
        
        return fig
    
    @staticmethod
    def create_modern_histogram(
        data: pd.Series, 
        color: str
    ) -> go.Figure:
        """
        Crée un histogramme moderne pour la régression.
        
        Args:
            data: Série de données à visualiser
            color: Couleur principale
            
        Returns:
            Figure Plotly
        """
        fig = go.Figure()
        
        # Histogramme
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=50,
            marker=dict(
                color=color,
                line=dict(color='white', width=1)
            ),
            opacity=0.8,
            hovertemplate='Valeur: %{x:.2f}<br>Fréquence: %{y}<extra></extra>'
        ))
        
        # Ligne de moyenne
        mean_value = data.mean()
        max_freq = np.histogram(data, bins=50)[0].max()
        
        fig.add_trace(go.Scatter(
            x=[mean_value, mean_value],
            y=[0, max_freq * 0.8],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f'Moyenne: {mean_value:.2f}',
            hovertemplate='Moyenne: %{x:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': "Distribution de la Variable Cible",
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Valeur",
            yaxis_title="Fréquence",
            template="plotly_white",
            height=400,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    @staticmethod
    def get_imbalance_level(ratio: float) -> Dict[str, str]:
        """
        Détermine le niveau de déséquilibre et retourne les infos associées.
        
        Args:
            ratio: Ratio de déséquilibre
            
        Returns:
            Dict avec label, color, icon, class
        """
        if ratio > 10:
            return {
                "label": "Critique", 
                "color": "#dc3545", 
                "icon": "🚨", 
                "class": "critical"
            }
        elif ratio > 5:
            return {
                "label": "Élevé", 
                "color": "#fd7e14", 
                "icon": "⚠️", 
                "class": "high"
            }
        elif ratio > 2:
            return {
                "label": "Modéré", 
                "color": "#ffc107", 
                "icon": "📊", 
                "class": "moderate"
            }
        else:
            return {
                "label": "Faible", 
                "color": "#28a745", 
                "icon": "✅", 
                "class": "low"
            }
    
    @staticmethod
    def get_class_color(class_name: str, class_counts: pd.Series) -> str:
        """
        Retourne une couleur cohérente pour une classe.
        
        Args:
            class_name: Nom de la classe
            class_counts: Series des comptages
            
        Returns:
            Code couleur hexadécimal
        """
        colors = [
            '#667eea', '#764ba2', '#f093fb', '#f5576c', 
            '#4facfe', '#43e97b', '#fa709a', '#fee140'
        ]
        
        try:
            class_index = list(class_counts.index).index(class_name)
            return colors[class_index % len(colors)]
        except:
            return colors[0]
    
    @staticmethod
    def calculate_avg_correlation(df: pd.DataFrame) -> float:
        """
        Calcule la corrélation moyenne entre les features numériques.
        
        Args:
            df: DataFrame avec features numériques
            
        Returns:
            Corrélation moyenne
        """
        if len(df.columns) < 2:
            return 0.0
        
        try:
            corr_matrix = df.corr().abs()
            # Exclure la diagonale (auto-corrélation = 1)
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            correlations = corr_matrix.where(mask).stack()
            return float(correlations.mean()) if len(correlations) > 0 else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def generate_imbalance_recommendations(
        ratio: float, 
        class_counts: pd.Series
    ) -> List[str]:
        """
        Génère des recommandations pour gérer le déséquilibre.
        
        Args:
            ratio: Ratio de déséquilibre
            class_counts: Series des comptages
            
        Returns:
            Liste de recommandations
        """
        recommendations = []
        
        if ratio > 10:
            recommendations.extend([
                "🚨 **Déséquilibre critique** : Activez SMOTE ET les poids de classe",
                "📉 **Collecte de données** : Essayez de collecter plus d'échantillons pour les classes minoritaires", 
                "🎯 **Métriques** : Utilisez F1-score et AUC-ROC au lieu de l'accuracy",
                "🧪 **Modèles** : Privilégiez Random Forest et XGBoost avec class_weights"
            ])
        elif ratio > 5:
            recommendations.extend([
                "⚠️ **Déséquilibre élevé** : Activez au moins SMOTE ou les poids de classe",
                "📊 **Évaluation** : Utilisez la validation croisée stratifiée",
                "🔧 **Techniques** : Essayez l'undersampling combiné avec SMOTE",
                "📈 **Modèles** : Les méthodes d'ensemble avec class_weights sont recommandées"
            ])
        elif ratio > 2:
            recommendations.extend([
                "📊 **Déséquilibre modéré** : Les poids de classe devraient suffire",
                "🎯 **Focus** : Surveillez le recall pour les classes minoritaires",
                "⚖️ **Option** : SMOTE peut aider si les performances sont insuffisantes"
            ])
        else:
            recommendations.extend([
                "✅ **Équilibre bon** : Aucune correction nécessaire",
                "📈 **Optimisation** : Concentrez-vous sur l'optimisation des hyperparamètres",
                "🔍 **Monitoring** : Surveillez quand même les métriques par classe"
            ])
        
        return recommendations


class TargetAnalysisHelpers:
    """Helpers pour l'analyse de la variable cible"""
    
    @staticmethod
    def get_classification_targets(df: pd.DataFrame) -> List[str]:
        """
        Retourne les colonnes appropriées pour la classification.
        
        Args:
            df: DataFrame
            
        Returns:
            Liste de noms de colonnes
        """
        targets = []
        for col in df.columns:
            n_unique = df[col].nunique()
            # Classification: colonnes avec 2-50 valeurs uniques
            if 2 <= n_unique <= 50:
                targets.append(col)
            # Ou colonnes catégorielles avec plus de valeurs mais considérées comme catégorielles
            elif not pd.api.types.is_numeric_dtype(df[col]) and n_unique <= 100:
                targets.append(col)
        return targets
    
    @staticmethod
    def get_regression_targets(df: pd.DataFrame) -> List[str]:
        """
        Retourne les colonnes appropriées pour la régression.
        
        Args:
            df: DataFrame
            
        Returns:
            Liste de noms de colonnes
        """
        targets = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                n_unique = df[col].nunique()
                # Régression: colonnes numériques avec suffisamment de variance
                if n_unique > 10 and df[col].std() > 0:
                    targets.append(col)
        return targets
    
    @staticmethod
    def analyze_target_quality(
        df: pd.DataFrame, 
        target_column: str, 
        task_type: str
    ) -> Dict[str, Any]:
        """
        Analyse la qualité de la variable cible.
        
        Args:
            df: DataFrame
            target_column: Nom de la colonne cible
            task_type: Type de tâche
            
        Returns:
            Dict avec analyse de qualité
        """
        analysis = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "statistics": {}
        }
        
        target_data = df[target_column]
        
        # Vérification valeurs manquantes
        missing_count = target_data.isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        
        if missing_count > 0:
            analysis["warnings"].append(
                f"{missing_count} valeurs manquantes ({missing_pct:.1f}%)"
            )
            if missing_pct > 10:
                analysis["issues"].append("Trop de valeurs manquantes (>10%)")
                analysis["is_valid"] = False
        
        # Analyse spécifique par type
        if task_type == 'classification':
            n_classes = target_data.nunique()
            analysis["statistics"]["n_classes"] = n_classes
            
            if n_classes < 2:
                analysis["issues"].append("Moins de 2 classes détectées")
                analysis["is_valid"] = False
            elif n_classes > 50:
                analysis["warnings"].append(
                    f"Nombre élevé de classes ({n_classes}) - temps d'entraînement long"
                )
            
            # Vérification déséquilibre
            class_dist = target_data.value_counts()
            ratio = class_dist.max() / class_dist.min() if class_dist.min() > 0 else float('inf')
            analysis["statistics"]["imbalance_ratio"] = ratio
            
            if ratio > 10:
                analysis["warnings"].append(
                    f"Déséquilibre critique détecté (ratio: {ratio:.1f}:1)"
                )
        
        elif task_type == 'regression':
            stats = target_data.describe()
            analysis["statistics"]["mean"] = stats['mean']
            analysis["statistics"]["std"] = stats['std']
            analysis["statistics"]["range"] = (stats['min'], stats['max'])
            
            # Vérification variance
            if stats['std'] == 0:
                analysis["issues"].append("Variable cible constante (variance nulle)")
                analysis["is_valid"] = False
            elif stats['std'] < 0.01:
                analysis["warnings"].append("Très faible variance détectée")
            
            # Vérification outliers
            Q1 = target_data.quantile(0.25)
            Q3 = target_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = target_data[
                (target_data < (Q1 - 1.5 * IQR)) | 
                (target_data > (Q3 + 1.5 * IQR))
            ]
            
            if len(outliers) > 0:
                outlier_pct = (len(outliers) / len(target_data)) * 100
                analysis["statistics"]["outliers_pct"] = outlier_pct
                if outlier_pct > 10:
                    analysis["warnings"].append(
                        f"Nombreuses valeurs extrêmes ({outlier_pct:.1f}%)"
                    )
        
        return analysis


# Export
__all__ = [
    'UIComponents',
    'TargetAnalysisHelpers'
]