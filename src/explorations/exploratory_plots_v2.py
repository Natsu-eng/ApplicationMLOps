"""
📊 Exploratory Plots V2 - Visualisations Avancées
Version: 2.0.0 | Production Ready
Auteur: DataLab Team

Nouvelles visualisations ajoutées:
- Pairplots interactifs
- Violin plots
- Sunburst pour catégories
- Scatter matrix
- Time series décomposition
- Box plots comparatifs avancés
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import List, Optional, Dict, Tuple
import warnings

from src.shared.logging import get_logger

logger = get_logger(__name__)
warnings.filterwarnings('ignore')


# ============================================================================
# NOUVELLES VISUALISATIONS TABULAIRES
# ============================================================================

@st.cache_data(ttl=1800, max_entries=10)
def create_interactive_pairplot(
    df: pd.DataFrame,
    columns: List[str],
    color_column: Optional[str] = None,
    max_vars: int = 5
) -> Optional[go.Figure]:
    """
    Crée un pairplot interactif avec Plotly.
    
    Args:
        df: DataFrame
        columns: Colonnes à inclure
        color_column: Colonne pour coloration
        max_vars: Nombre max de variables
        
    Returns:
        Figure Plotly ou None
    """
    try:
        if len(columns) < 2:
            logger.warning("Au moins 2 colonnes requises pour pairplot")
            return None
        
        # Limiter le nombre de variables
        columns = columns[:max_vars]
        
        # Échantillonnage si trop de données
        if len(df) > 1000:
            df_sample = df[columns + ([color_column] if color_column else [])].sample(1000, random_state=42)
        else:
            df_sample = df[columns + ([color_column] if color_column else [])]
        
        # Créer le scatter matrix
        fig = px.scatter_matrix(
            df_sample,
            dimensions=columns,
            color=color_column if color_column else None,
            title=f"Pairplot Interactif ({len(columns)} variables)"
        )
        
        fig.update_traces(diagonal_visible=False, showupperhalf=False)
        fig.update_layout(height=700, width=900)
        
        logger.info(f"Pairplot créé avec {len(columns)} variables")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création pairplot: {e}")
        return None


@st.cache_data(ttl=1800, max_entries=20)
def create_violin_plot(
    df: pd.DataFrame,
    numeric_col: str,
    category_col: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Crée un violin plot pour visualiser la distribution.
    
    Args:
        df: DataFrame
        numeric_col: Colonne numérique
        category_col: Colonne catégorielle pour grouper
        
    Returns:
        Figure Plotly ou None
    """
    try:
        if category_col and category_col in df.columns:
            # Violin par catégorie
            fig = px.violin(
                df,
                y=numeric_col,
                x=category_col,
                box=True,
                points="outliers",
                title=f"Distribution de {numeric_col} par {category_col}"
            )
        else:
            # Violin simple
            fig = go.Figure(data=go.Violin(
                y=df[numeric_col],
                box_visible=True,
                meanline_visible=True,
                name=numeric_col
            ))
            fig.update_layout(title=f"Distribution de {numeric_col}")
        
        fig.update_layout(height=500)
        
        logger.info(f"Violin plot créé pour {numeric_col}")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création violin plot: {e}")
        return None


@st.cache_data(ttl=1800, max_entries=10)
def create_sunburst_chart(
    df: pd.DataFrame,
    path_columns: List[str],
    value_column: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Crée un sunburst pour visualiser des hiérarchies.
    
    Args:
        df: DataFrame
        path_columns: Colonnes définissant la hiérarchie
        value_column: Colonne pour la taille des segments
        
    Returns:
        Figure Plotly ou None
    """
    try:
        if len(path_columns) < 2:
            logger.warning("Au moins 2 colonnes requises pour sunburst")
            return None
        
        # Agrégation des données
        if value_column and value_column in df.columns:
            values = df.groupby(path_columns)[value_column].sum().reset_index()
        else:
            values = df.groupby(path_columns).size().reset_index(name='count')
            value_column = 'count'
        
        fig = px.sunburst(
            values,
            path=path_columns,
            values=value_column,
            title=f"Hiérarchie: {' → '.join(path_columns)}"
        )
        
        fig.update_layout(height=600)
        
        logger.info(f"Sunburst créé avec {len(path_columns)} niveaux")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création sunburst: {e}")
        return None


@st.cache_data(ttl=1800, max_entries=20)
def create_advanced_box_comparison(
    df: pd.DataFrame,
    numeric_cols: List[str],
    max_cols: int = 6
) -> Optional[go.Figure]:
    """
    Crée une comparaison de box plots pour plusieurs variables.
    
    Args:
        df: DataFrame
        numeric_cols: Colonnes numériques à comparer
        max_cols: Nombre max de colonnes
        
    Returns:
        Figure Plotly ou None
    """
    try:
        if not numeric_cols:
            return None
        
        numeric_cols = numeric_cols[:max_cols]
        
        fig = go.Figure()
        
        for col in numeric_cols:
            fig.add_trace(go.Box(
                y=df[col],
                name=col,
                boxmean='sd'
            ))
        
        fig.update_layout(
            title=f"Comparaison Box Plots ({len(numeric_cols)} variables)",
            yaxis_title="Valeur",
            showlegend=True,
            height=500
        )
        
        logger.info(f"Box comparison créé pour {len(numeric_cols)} variables")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création box comparison: {e}")
        return None


@st.cache_data(ttl=1800, max_entries=10)
def create_time_series_plot(
    df: pd.DataFrame,
    date_column: str,
    value_columns: List[str],
    resample_freq: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Crée un graphique de série temporelle.
    
    Args:
        df: DataFrame avec colonne datetime
        date_column: Nom de la colonne datetime
        value_columns: Colonnes de valeurs à tracer
        resample_freq: Fréquence de rééchantillonnage ('D', 'W', 'M')
        
    Returns:
        Figure Plotly ou None
    """
    try:
        if date_column not in df.columns:
            logger.error(f"Colonne datetime {date_column} non trouvée")
            return None
        
        # Conversion en datetime si nécessaire
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Trier par date
        df_sorted = df.sort_values(date_column)
        
        # Rééchantillonnage si demandé
        if resample_freq:
            df_sorted = df_sorted.set_index(date_column)
            df_sorted = df_sorted[value_columns].resample(resample_freq).mean().reset_index()
        
        fig = go.Figure()
        
        for col in value_columns:
            if col in df_sorted.columns:
                fig.add_trace(go.Scatter(
                    x=df_sorted[date_column],
                    y=df_sorted[col],
                    mode='lines+markers',
                    name=col
                ))
        
        fig.update_layout(
            title=f"Série Temporelle: {', '.join(value_columns)}",
            xaxis_title="Date",
            yaxis_title="Valeur",
            hovermode='x unified',
            height=500
        )
        
        logger.info(f"Time series plot créé pour {len(value_columns)} variables")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création time series: {e}")
        return None


@st.cache_data(ttl=1800, max_entries=10)
def create_parallel_coordinates(
    df: pd.DataFrame,
    columns: List[str],
    color_column: Optional[str] = None,
    max_rows: int = 500
) -> Optional[go.Figure]:
    """
    Crée un graphique de coordonnées parallèles.
    
    Args:
        df: DataFrame
        columns: Colonnes à inclure
        color_column: Colonne pour coloration
        max_rows: Nombre max de lignes
        
    Returns:
        Figure Plotly ou None
    """
    try:
        if len(columns) < 3:
            logger.warning("Au moins 3 colonnes requises pour parallel coordinates")
            return None
        
        # Échantillonnage
        if len(df) > max_rows:
            df_sample = df[columns + ([color_column] if color_column else [])].sample(max_rows, random_state=42)
        else:
            df_sample = df[columns + ([color_column] if color_column else [])]
        
        fig = px.parallel_coordinates(
            df_sample,
            dimensions=columns,
            color=color_column if color_column else None,
            title="Coordonnées Parallèles"
        )
        
        fig.update_layout(height=600)
        
        logger.info(f"Parallel coordinates créé avec {len(columns)} dimensions")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création parallel coordinates: {e}")
        return None


# ============================================================================
# NOUVELLES VISUALISATIONS IMAGES
# ============================================================================

@st.cache_data(ttl=1800, max_entries=5)
def create_image_similarity_heatmap(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    max_images: int = 50
) -> Optional[go.Figure]:
    """
    Crée une heatmap de similarité entre images.
    
    Args:
        embeddings: Vecteurs d'embeddings (N, D)
        labels: Labels optionnels
        max_images: Nombre max d'images
        
    Returns:
        Figure Plotly ou None
    """
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Limiter le nombre d'images
        if len(embeddings) > max_images:
            indices = np.random.choice(len(embeddings), max_images, replace=False)
            embeddings = embeddings[indices]
            if labels is not None:
                labels = labels[indices]
        
        # Calculer similarité cosinus
        similarity_matrix = cosine_similarity(embeddings)
        
        # Créer heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title=f"Matrice de Similarité ({len(embeddings)} images)",
            xaxis_title="Image Index",
            yaxis_title="Image Index",
            height=600,
            width=600
        )
        
        logger.info(f"Similarity heatmap créée pour {len(embeddings)} images")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création similarity heatmap: {e}")
        return None


@st.cache_data(ttl=1800, max_entries=5)
def create_tsne_visualization(
    images: np.ndarray,
    labels: np.ndarray,
    n_components: int = 2,
    perplexity: int = 30,
    max_images: int = 1000
) -> Optional[go.Figure]:
    """
    Crée une visualisation t-SNE des images.
    
    Args:
        images: Images (N, H, W, C) ou (N, features)
        labels: Labels
        n_components: Dimensions de sortie
        perplexity: Paramètre t-SNE
        max_images: Nombre max d'images
        
    Returns:
        Figure Plotly ou None
    """
    try:
        from sklearn.manifold import TSNE
        
        # Échantillonnage
        if len(images) > max_images:
            indices = np.random.choice(len(images), max_images, replace=False)
            images_sample = images[indices]
            labels_sample = labels[indices]
        else:
            images_sample = images
            labels_sample = labels
        
        # Flatten si images 3D/4D
        if len(images_sample.shape) > 2:
            n_samples = len(images_sample)
            images_flat = images_sample.reshape(n_samples, -1)
        else:
            images_flat = images_sample
        
        # t-SNE
        logger.info("Calcul t-SNE en cours...")
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        embeddings = tsne.fit_transform(images_flat)
        
        # Visualisation
        fig = px.scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            color=labels_sample.astype(str),
            labels={'x': 't-SNE 1', 'y': 't-SNE 2', 'color': 'Classe'},
            title=f"Visualisation t-SNE ({len(images_sample)} images)"
        )
        
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=600)
        
        logger.info(f"t-SNE visualization créée pour {len(images_sample)} images")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création t-SNE: {e}")
        return None


@st.cache_data(ttl=1800, max_entries=10)
def create_image_grid_comparison(
    images: np.ndarray,
    labels: np.ndarray,
    n_per_class: int = 4,
    figsize: Tuple[int, int] = (12, 8)
) -> Optional[go.Figure]:
    """
    Crée une grille de comparaison d'images par classe.
    
    Args:
        images: Images
        labels: Labels
        n_per_class: Nombre d'images par classe
        figsize: Taille de la figure
        
    Returns:
        Figure Plotly ou None
    """
    try:
        import matplotlib.pyplot as plt
        
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        
        fig, axes = plt.subplots(n_classes, n_per_class, figsize=figsize)
        
        if n_classes == 1:
            axes = axes.reshape(1, -1)
        
        for i, label in enumerate(unique_labels):
            # Trouver images de cette classe
            indices = np.where(labels == label)[0]
            
            # Échantillonner
            if len(indices) > n_per_class:
                indices = np.random.choice(indices, n_per_class, replace=False)
            
            for j, idx in enumerate(indices[:n_per_class]):
                img = images[idx]
                
                # Normaliser pour affichage
                if img.max() > 1.0:
                    img = img / 255.0
                
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                
                if j == 0:
                    axes[i, j].set_ylabel(f"Classe {label}", fontsize=12)
        
        plt.suptitle("Grille de Comparaison par Classe", fontsize=16)
        plt.tight_layout()
        
        logger.info(f"Image grid créée: {n_classes} classes × {n_per_class} images")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur création image grid: {e}")
        return None


# Export
__all__ = [
    'create_interactive_pairplot',
    'create_violin_plot',
    'create_sunburst_chart',
    'create_advanced_box_comparison',
    'create_time_series_plot',
    'create_parallel_coordinates',
    'create_image_similarity_heatmap',
    'create_tsne_visualization',
    'create_image_grid_comparison'
]