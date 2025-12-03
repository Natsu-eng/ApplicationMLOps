"""
Fonctions de visualisation pour l'évaluation des modèles de détection d'anomalies en vision par ordinateur.
Utilise Plotly pour des graphiques interactifs, optimisés pour la production avec intégration MLflow.
"""
import numpy as np
import plotly.graph_objects as go # type: ignore
import plotly.express as px # type: ignore
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix # type: ignore
from typing import Optional, Dict, List, Tuple
from src.shared.logging import get_logger
from src.config.constants import ANOMALY_CONFIG
import mlflow # type: ignore
import tempfile
import os

logger = get_logger(__name__)

def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, multi_class: str = "ovr", mlflow_run_id: Optional[str] = None) -> Optional[go.Figure]:
    """
    Génère une courbe ROC pour évaluer la performance du modèle (binaire ou multi-classes).

    Args:
        y_true (np.ndarray): Labels réels.
        y_score (np.ndarray): Scores de prédiction (probabilités ou scores d'anomalie).
        multi_class (str): Stratégie pour multi-classes ("ovr" pour One-vs-Rest, "ovo" pour One-vs-One).
        mlflow_run_id (Optional[str]): ID du run MLflow pour logger la figure.

    Returns:
        Optional[go.Figure]: Figure Plotly de la courbe ROC, ou None si erreur.
    """
    try:
        if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
            raise ValueError("y_true et y_score doivent être des numpy arrays")
        if len(y_true) != len(y_score):
            raise ValueError(f"Incohérence entre longueurs: y_true={len(y_true)}, y_score={len(y_score)}")
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_score)):
            raise ValueError("NaN détecté dans y_true ou y_score")
        if len(np.unique(y_true)) < 2:
            logger.warning("Pas assez de classes pour ROC")
            return None

        fig = go.Figure()
        if y_score.ndim > 1 and y_score.shape[1] > 1:  # Multi-classes
            classes = np.unique(y_true)
            for i, cls in enumerate(classes):
                y_true_bin = (y_true == cls).astype(int)
                y_score_cls = y_score[:, i]
                fpr, tpr, _ = roc_curve(y_true_bin, y_score_cls)
                fig.add_trace(go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"Classe {cls} (OvR)",
                    line=dict(width=2)
                ))
        else:  # Binaire
            fpr, tpr, _ = roc_curve(y_true, y_score)
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name="ROC Curve",
                line=dict(color="#3498db", width=2)
            ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Baseline",
            line=dict(color="#e74c3c", dash="dash")
        ))
        fig.update_layout(
            title="Courbe ROC",
            xaxis_title="Taux de faux positifs (FPR)",
            yaxis_title="Taux de vrais positifs (TPR)",
            template="plotly_white",
            height=400,
            showlegend=True
        )

        # Log dans MLflow
        if mlflow_run_id and ANOMALY_CONFIG.get("MLFLOW_ENABLED", False):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    file_path = os.path.join(tmpdir, "roc_curve.html")
                    fig.write_html(file_path)
                    mlflow.log_artifact(file_path, artifact_path="plots")
                    logger.info(f"Courbe ROC loguée dans MLflow pour run {mlflow_run_id}")
            except Exception as e:
                logger.error(f"MLflow logging failed pour ROC: {e}")

        logger.info("Courbe ROC générée avec succès")
        return fig
    except Exception as e:
        logger.error(f"Erreur dans plot_roc_curve: {e}")
        return None

def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, multi_class: str = "weighted", mlflow_run_id: Optional[str] = None) -> Optional[go.Figure]:
    """
    Génère une courbe précision-rappel pour évaluer la performance du modèle.

    Args:
        y_true (np.ndarray): Labels réels.
        y_score (np.ndarray): Scores de prédiction.
        multi_class (str): Stratégie pour multi-classes ("weighted" pour moyenne pondérée).
        mlflow_run_id (Optional[str]): ID du run MLflow pour logger la figure.

    Returns:
        Optional[go.Figure]: Figure Plotly de la courbe PR, ou None si erreur.
    """
    try:
        if not isinstance(y_true, np.ndarray) or not isinstance(y_score, np.ndarray):
            raise ValueError("y_true et y_score doivent être des numpy arrays")
        if len(y_true) != len(y_score):
            raise ValueError(f"Incohérence entre longueurs: y_true={len(y_true)}, y_score={len(y_score)}")
        if np.any(np.isnan(y_true)) or np.any(np.isnan(y_score)):
            raise ValueError("NaN détecté dans y_true ou y_score")
        if len(np.unique(y_true)) < 2:
            logger.warning("Pas assez de classes pour PR")
            return None

        fig = go.Figure()
        if y_score.ndim > 1 and y_score.shape[1] > 1:  # Multi-classes
            classes = np.unique(y_true)
            for i, cls in enumerate(classes):
                y_true_bin = (y_true == cls).astype(int)
                y_score_cls = y_score[:, i]
                precision, recall, _ = precision_recall_curve(y_true_bin, y_score_cls)
                fig.add_trace(go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"Classe {cls}",
                    line=dict(width=2)
                ))
        else:  # Binaire
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            fig.add_trace(go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name="Precision-Recall",
                line=dict(color="#2ecc71", width=2)
            ))

        fig.update_layout(
            title="Courbe Précision-Rappel",
            xaxis_title="Rappel (Recall)",
            yaxis_title="Précision (Precision)",
            template="plotly_white",
            height=400,
            showlegend=True
        )

        # Log dans MLflow
        if mlflow_run_id and ANOMALY_CONFIG.get("MLFLOW_ENABLED", False):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    file_path = os.path.join(tmpdir, "pr_curve.html")
                    fig.write_html(file_path)
                    mlflow.log_artifact(file_path, artifact_path="plots")
                    logger.info(f"Courbe PR loguée dans MLflow pour run {mlflow_run_id}")
            except Exception as e:
                logger.error(f"MLflow logging failed pour PR: {e}")

        logger.info("Courbe précision-rappel générée avec succès")
        return fig
    except Exception as e:
        logger.error(f"Erreur dans plot_pr_curve: {e}")
        return None

def plot_confusion_matrix(conf_matrix: np.ndarray, labels: Optional[List[str]] = None, mlflow_run_id: Optional[str] = None) -> Optional[go.Figure]:
    """
    Génère une matrice de confusion visualisée.

    Args:
        conf_matrix (np.ndarray): Matrice de confusion.
        labels (Optional[List[str]]): Noms des classes, sinon généré automatiquement.
        mlflow_run_id (Optional[str]): ID du run MLflow pour logger la figure.

    Returns:
        Optional[go.Figure]: Figure Plotly de la matrice, ou None si erreur.
    """
    try:
        if not isinstance(conf_matrix, np.ndarray):
            raise ValueError("conf_matrix doit être un numpy array")
        if conf_matrix.ndim != 2 or conf_matrix.shape[0] != conf_matrix.shape[1]:
            raise ValueError("conf_matrix doit être carrée")

        if labels is None:
            labels = [f"Classe {i}" for i in range(conf_matrix.shape[0])]
        if len(labels) != conf_matrix.shape[0]:
            raise ValueError("Nombre de labels incohérent avec la matrice")

        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=conf_matrix,
            texttemplate="%{text}",
            textfont={"size": 14}
        ))
        fig.update_layout(
            title="Matrice de Confusion",
            xaxis_title="Prédit",
            yaxis_title="Réel",
            template="plotly_white",
            height=400
        )

        # Log dans MLflow
        if mlflow_run_id and ANOMALY_CONFIG.get("MLFLOW_ENABLED", False):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    file_path = os.path.join(tmpdir, "confusion_matrix.html")
                    fig.write_html(file_path)
                    mlflow.log_artifact(file_path, artifact_path="plots")
                    logger.info(f"Matrice de confusion loguée dans MLflow pour run {mlflow_run_id}")
            except Exception as e:
                logger.error(f"MLflow logging failed pour confusion matrix: {e}")

        logger.info("Matrice de confusion générée avec succès")
        return fig
    except Exception as e:
        logger.error(f"Erreur dans plot_confusion_matrix: {e}")
        return None

def plot_anomaly_heatmap(
    image: np.ndarray, 
    anomaly_score: np.ndarray, 
    mlflow_run_id: Optional[str] = None,
    original_size: Optional[Tuple[int, int]] = None
) -> Optional[go.Figure]:
    """
    Génère une heatmap des anomalies superposée à l'image d'entrée avec Plotly.
    
    ✅ CORRECTION #10, #17: Alignement automatique des dimensions heatmap/image

    Args:
        image (np.ndarray): Image originale (H, W, C).
        anomaly_score (np.ndarray): Carte de scores d'anomalie (H, W) ou (B, H, W).
        mlflow_run_id (Optional[str]): ID du run MLflow pour logger la figure.
        original_size (Optional[Tuple[int, int]]): Taille originale de l'image (H_orig, W_orig)
            pour aligner la heatmap si nécessaire.

    Returns:
        Optional[go.Figure]: Figure Plotly avec heatmap superposée, ou None si erreur.
    """
    try:
        if not isinstance(image, np.ndarray) or not isinstance(anomaly_score, np.ndarray):
            raise ValueError("image et anomaly_score doivent être des numpy arrays")
        
        # ✅ CORRECTION #18: Gestion format channels
        # Extraire la heatmap si batch
        if anomaly_score.ndim == 3:
            # (B, H, W) → prendre la première
            heatmap_raw = anomaly_score[0]
        elif anomaly_score.ndim == 4:
            # (B, 1, H, W) → prendre la première
            heatmap_raw = anomaly_score[0, 0]
        else:
            heatmap_raw = anomaly_score
        
        # ✅ CORRECTION #10, #17: Alignement dimensions si nécessaire
        if original_size is not None:
            from src.evaluation.localization_utils import align_heatmap_to_image
            
            current_size = heatmap_raw.shape[:2]
            if current_size != original_size:
                logger.info(
                    f"🔧 Alignement heatmap: {current_size} → {original_size}"
                )
                heatmap_raw = align_heatmap_to_image(
                    heatmap_raw,
                    original_size,
                    current_size
                )
        
        # Validation shapes après alignement
        if image.shape[:2] != heatmap_raw.shape[:2]:
            logger.warning(
                f"⚠️ Shapes non alignées après correction: "
                f"image={image.shape[:2]}, anomaly_score={heatmap_raw.shape[:2]}. "
                f"Tentative resize automatique."
            )
            # Tentative resize automatique
            from src.evaluation.localization_utils import resize_error_map
            heatmap_raw = resize_error_map(
                heatmap_raw,
                image.shape[:2],
                method="bilinear"
            )
        
        if image.shape[:2] != heatmap_raw.shape[:2]:
            raise ValueError(
                f"❌ Impossible d'aligner les dimensions: "
                f"image={image.shape[:2]}, anomaly_score={heatmap_raw.shape[:2]}"
            )
        
        if np.any(np.isnan(image)) or np.any(np.isnan(heatmap_raw)):
            raise ValueError("NaN détecté dans image ou anomaly_score")

        # Normaliser l'image
        img = image.copy()
        if img.max() > 1:
            img = img / 255.0

        # Normaliser la heatmap
        heatmap = (heatmap_raw - heatmap_raw.min()) / (heatmap_raw.max() - heatmap_raw.min() + 1e-8)

        # Convertir l'image en RGB si nécessaire
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif len(img.shape) == 2:
            # Grayscale 2D → RGB
            img = np.stack([img, img, img], axis=-1)

        # Créer la figure Plotly
        fig = go.Figure()
        fig.add_trace(go.Image(z=(img * 255).astype(np.uint8)))
        fig.add_trace(go.Heatmap(
            z=heatmap,
            colorscale="Jet",
            opacity=0.4,
            showscale=True
        ))
        fig.update_layout(
            title="Heatmap des Anomalies",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400,
            width=400
        )

        # Log dans MLflow
        if mlflow_run_id and ANOMALY_CONFIG.get("MLFLOW_ENABLED", False):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    file_path = os.path.join(tmpdir, "anomaly_heatmap.html")
                    fig.write_html(file_path)
                    mlflow.log_artifact(file_path, artifact_path="plots")
                    logger.info(f"Heatmap d'anomalie loguée dans MLflow pour run {mlflow_run_id}")
            except Exception as e:
                logger.error(f"MLflow logging failed pour heatmap: {e}")

        logger.info("✅ Heatmap d'anomalie générée avec succès")
        return fig
    except Exception as e:
        logger.error(f"Erreur dans plot_anomaly_heatmap: {e}", exc_info=True)
        return None

def plot_reconstruction_error_histogram(errors: np.ndarray, threshold: Optional[float] = None, mlflow_run_id: Optional[str] = None) -> Optional[go.Figure]:
    """
    Génère un histogramme des erreurs de reconstruction pour AutoEncoders.

    Args:
        errors (np.ndarray): Erreurs de reconstruction (ex. MSE par image).
        threshold (Optional[float]): Seuil d'anomalie pour visualisation.
        mlflow_run_id (Optional[str]): ID du run MLflow pour logger la figure.

    Returns:
        Optional[go.Figure]: Figure Plotly de l'histogramme, ou None si erreur.
    """
    try:
        if not isinstance(errors, np.ndarray):
            raise ValueError("errors doit être un numpy array")
        if np.any(np.isnan(errors)):
            raise ValueError("NaN détecté dans errors")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=errors,
            nbinsx=50,
            name="Erreurs de Reconstruction",
            marker_color="#3498db"
        ))
        if threshold is not None:
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text="Seuil",
                annotation_position="top right"
            )
        fig.update_layout(
            title="Histogramme des Erreurs de Reconstruction",
            xaxis_title="Erreur (ex. MSE)",
            yaxis_title="Nombre d'images",
            template="plotly_white",
            height=400,
            showlegend=True
        )

        # Log dans MLflow
        if mlflow_run_id and ANOMALY_CONFIG.get("MLFLOW_ENABLED", False):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    file_path = os.path.join(tmpdir, "reconstruction_error_histogram.html")
                    fig.write_html(file_path)
                    mlflow.log_artifact(file_path, artifact_path="plots")
                    logger.info(f"Histogramme des erreurs logué dans MLflow pour run {mlflow_run_id}")
            except Exception as e:
                logger.error(f"MLflow logging failed pour histogramme: {e}")

        logger.info("Histogramme des erreurs généré avec succès")
        return fig
    except Exception as e:
        logger.error(f"Erreur dans plot_reconstruction_error_histogram: {e}")
        return None


def plot_loss_history(history: Dict[str, List[float]], mlflow_run_id: Optional[str] = None) -> Optional[go.Figure]:
    try:
        if not isinstance(history, dict) or not history:
            raise ValueError("Historique doit être un dictionnaire non vide")
        
        # ✅ FILTRE: Exclure les clés non-list (scalaires, booléens, etc.)
        valid_keys = {
            k: v for k, v in history.items() 
            if isinstance(v, (list, np.ndarray)) and len(v) > 0
        }
        
        if not valid_keys:
            logger.error("Aucune série temporelle valide dans l'historique")
            return None
        
        logger.info(f"Clés valides trouvées: {list(valid_keys.keys())}")
        
        # Recherche des clés de loss
        train_loss_key = None
        val_loss_key = None
        
        for key in ['train_loss', 'loss', 'training_loss']:
            if key in valid_keys:
                train_loss_key = key
                break
        
        for key in ['val_loss', 'validation_loss']:
            if key in valid_keys:
                val_loss_key = key
                break
        
        if not train_loss_key:
            logger.error(f"Aucune clé de loss trouvée. Clés valides: {list(valid_keys.keys())}")
            return None
        
        # Conversion en float
        train_loss = np.array(valid_keys[train_loss_key], dtype=float)
        val_loss = np.array(valid_keys[val_loss_key], dtype=float) if val_loss_key else None
        
        epochs = list(range(1, len(train_loss) + 1))
        
        # Création du graphique
        fig = go.Figure()
        
        # Train loss
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_loss,
            mode="lines",
            name="Training Loss",
            line=dict(color="#3498db", width=2)
        ))
        
        # Validation loss
        if val_loss is not None:
            fig.add_trace(go.Scatter(
                x=epochs,
                y=val_loss,
                mode="lines",
                name="Validation Loss",
                line=dict(color="#e74c3c", width=2)
            ))
        
        # Autres métriques (accuracy, f1, etc.)
        metric_colors = ["#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
        color_idx = 0
        
        # ✅ FILTRE: Uniquement les métriques de validation
        for key in valid_keys:
            if key.startswith('val_') and key != val_loss_key:
                try:
                    metric_values = np.array(valid_keys[key], dtype=float)
                    fig.add_trace(go.Scatter(
                        x=epochs[:len(metric_values)],  # Ajuster la longueur
                        y=metric_values,
                        mode="lines",
                        name=key.replace("_", " ").title(),
                        line=dict(color=metric_colors[color_idx % len(metric_colors)], width=2),
                        yaxis="y2"
                    ))
                    color_idx += 1
                except Exception as e:
                    logger.warning(f"Impossible de tracer {key}: {e}")
        
        # Layout
        layout_config = {
            "title": "Courbes d'Entraînement",
            "xaxis_title": "Époques",
            "yaxis_title": "Loss",
            "template": "plotly_white",
            "height": 500,
            "showlegend": True
        }
        
        if color_idx > 0:
            layout_config["yaxis2"] = {
                "title": "Métriques",
                "overlaying": "y",
                "side": "right"
            }
        
        fig.update_layout(**layout_config)
        
        # Log MLflow
        if mlflow_run_id and ANOMALY_CONFIG.get("MLFLOW_ENABLED", False):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    file_path = os.path.join(tmpdir, "training_history.html")
                    fig.write_html(file_path)
                    mlflow.log_artifact(file_path, artifact_path="plots")
            except Exception as e:
                logger.error(f"MLflow logging failed: {e}")
        
        logger.info("Courbe d'historique générée avec succès")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur dans plot_loss_history: {e}")
        return None


def plot_anomaly_distribution_by_type(
    anomaly_scores: np.ndarray, 
    anomaly_types: np.ndarray,
    mlflow_run_id: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Génère un boxplot des scores d'anomalie par type d'anomalie.
    Utile pour analyser la séparabilité des types d'anomalies.
    """
    try:
        if len(anomaly_scores) != len(anomaly_types):
            raise ValueError("Incohérence entre scores et types d'anomalies")
        
        # Préparation des données pour Plotly
        unique_types = np.unique(anomaly_types)
        data = []
        
        for anomaly_type in unique_types:
            mask = anomaly_types == anomaly_type
            scores_for_type = anomaly_scores[mask]
            data.append(go.Box(
                y=scores_for_type,
                name=str(anomaly_type),
                boxpoints='outliers',
                marker_color="#3498db"
            ))
        
        fig = go.Figure(data=data)
        fig.update_layout(
            title="Distribution des Scores d'Anomalie par Type",
            xaxis_title="Type d'Anomalie",
            yaxis_title="Score d'Anomalie",
            template="plotly_white",
            height=500
        )
        
        # Log MLflow
        if mlflow_run_id and ANOMALY_CONFIG.get("MLFLOW_ENABLED", False):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    file_path = os.path.join(tmpdir, "anomaly_distribution_by_type.html")
                    fig.write_html(file_path)
                    mlflow.log_artifact(file_path, artifact_path="plots")
            except Exception as e:
                logger.error(f"MLflow logging failed pour anomaly distribution: {e}")
        
        logger.info("Boxplot des anomalies par type généré avec succès")
        return fig
        
    except Exception as e:
        logger.error(f"Erreur dans plot_anomaly_distribution_by_type: {e}")
        return None