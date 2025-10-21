"""
Sauvegarde et chargement des modèles.
Support pour modèles scikit-learn et TensorFlow/Keras.
"""
import joblib
import os
from typing import Any, Dict, Optional
import tensorflow as tf
from tensorflow.keras.models import Model, save_model as keras_save_model, load_model as keras_load_model # type: ignore
import json
from pathlib import Path

from src.shared.logging import get_logger

logger = get_logger(__name__)

# === FONCTIONS GÉNÉRIQUES ===

def save_model(model: Any, path: str):
    """Sauvegarde un modèle sur le disque."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}")
        raise

def load_model(path: str) -> Any:
    """Charge un modèle depuis le disque."""
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found at {path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        raise

# === FONCTIONS SPÉCIALISÉES POUR COMPUTER VISION ===

def save_computer_vision_model(
    model: Any, 
    path: str, 
    model_type: str = "autoencoder",
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Sauvegarde un modèle de vision par ordinateur avec métadonnées.
    
    Args:
        model: Modèle Keras/TensorFlow
        path: Chemin de sauvegarde
        model_type: Type de modèle ("autoencoder", "cnn_classifier", "transfer_learning")
        metadata: Métadonnées supplémentaires (architecture, hyperparamètres, etc.)
    """
    try:
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        
        # Déterminer l'extension
        if path.suffix in ['.h5', '.keras']:
            model_path = path
        else:
            model_path = path.with_suffix('.h5')
        
        # Sauvegarder le modèle Keras
        if isinstance(model, tf.keras.Model):
            model.save(model_path, save_format='h5')
            logger.info(f"Keras model saved to {model_path}")
        else:
            # Fallback pour les autres types
            joblib.dump(model, model_path)
            logger.info(f"Model saved with joblib to {model_path}")
        
        # Sauvegarder les métadonnées si fournies
        if metadata:
            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Model metadata saved to {metadata_path}")
        
        logger.info(f"Computer vision model successfully saved: {model_path}")
        
    except Exception as e:
        logger.error(f"Failed to save computer vision model to {path}: {e}")
        raise

def load_computer_vision_model(
    path: str, 
    custom_objects: Optional[Dict[str, Any]] = None
) -> tuple[Any, Optional[Dict[str, Any]]]:
    """
    Charge un modèle de vision par ordinateur avec ses métadonnées.
    
    Args:
        path: Chemin du modèle
        custom_objects: Objets personnalisés pour le chargement Keras
        
    Returns:
        Tuple (model, metadata) - Modèle chargé et métadonnées
    """
    try:
        path = Path(path)
        
        # Essayer d'abord avec Keras
        model = None
        metadata = None
        
        # Chercher le fichier modèle
        model_paths = [
            path,
            path.with_suffix('.h5'),
            path.with_suffix('.keras')
        ]
        
        model_path = None
        for mp in model_paths:
            if mp.exists():
                model_path = mp
                break
        
        if not model_path:
            raise FileNotFoundError(f"No model file found at {path}")
        
        # Charger le modèle
        if model_path.suffix in ['.h5', '.keras']:
            try:
                if custom_objects:
                    model = keras_load_model(model_path, custom_objects=custom_objects)
                else:
                    model = keras_load_model(model_path)
                logger.info(f"Keras model loaded from {model_path}")
            except Exception as keras_error:
                logger.warning(f"Keras loading failed, trying joblib: {keras_error}")
                model = joblib.load(model_path)
                logger.info(f"Model loaded with joblib from {model_path}")
        else:
            model = joblib.load(model_path)
            logger.info(f"Model loaded with joblib from {model_path}")
        
        # Charger les métadonnées si disponibles
        metadata_path = path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Model metadata loaded from {metadata_path}")
        
        logger.info(f"Computer vision model successfully loaded: {model_path}")
        return model, metadata
        
    except FileNotFoundError:
        logger.error(f"Model file not found at {path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load computer vision model from {path}: {e}")
        raise

def save_training_session(
    model: Any,
    history: Dict[str, Any],
    config: Dict[str, Any],
    save_dir: str,
    session_name: str = None
):
    """
    Sauvegarde une session d'entraînement complète.
    
    Args:
        model: Modèle entraîné
        history: Historique d'entraînement
        config: Configuration utilisée
        save_dir: Dossier de sauvegarde
        session_name: Nom de la session (optionnel)
    """
    try:
        save_dir = Path(save_dir)
        if session_name:
            save_dir = save_dir / session_name
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = tf.timestamp().numpy().astype(int)
        base_name = f"training_session_{timestamp}"
        
        # Sauvegarder le modèle
        model_path = save_dir / f"{base_name}_model.h5"
        save_computer_vision_model(model, str(model_path), metadata=config)
        
        # Sauvegarder l'historique
        history_path = save_dir / f"{base_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Sauvegarder la configuration
        config_path = save_dir / f"{base_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training session saved to {save_dir}")
        return str(save_dir)
        
    except Exception as e:
        logger.error(f"Failed to save training session: {e}")
        raise

def load_training_session(session_dir: str) -> tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Charge une session d'entraînement complète.
    
    Args:
        session_dir: Dossier de la session
        
    Returns:
        Tuple (model, history, config) - Session complète
    """
    try:
        session_dir = Path(session_dir)
        
        # Trouver les fichiers
        model_files = list(session_dir.glob("*_model.h5"))
        history_files = list(session_dir.glob("*_history.json"))
        config_files = list(session_dir.glob("*_config.json"))
        
        if not model_files:
            raise FileNotFoundError(f"No model file found in {session_dir}")
        
        # Charger le modèle
        model, metadata = load_computer_vision_model(str(model_files[0]))
        
        # Charger l'historique
        history = {}
        if history_files:
            with open(history_files[0], 'r') as f:
                history = json.load(f)
        
        # Charger la configuration
        config = metadata if metadata else {}
        if config_files:
            with open(config_files[0], 'r') as f:
                config = json.load(f)
        
        logger.info(f"Training session loaded from {session_dir}")
        return model, history, config
        
    except Exception as e:
        logger.error(f"Failed to load training session from {session_dir}: {e}")
        raise

# === FONCTIONS UTILITAIRES ===

def get_model_info(model: Any) -> Dict[str, Any]:
    """
    Récupère les informations d'un modèle.
    
    Args:
        model: Modèle à analyser
        
    Returns:
        Informations du modèle
    """
    info = {}
    
    try:
        if isinstance(model, tf.keras.Model):
            info.update({
                "type": "keras_model",
                "layers": len(model.layers),
                "trainable_params": model.count_params(),
                "input_shape": model.input_shape,
                "output_shape": model.output_shape
            })
            
            # Ajouter le résumé
            string_list = []
            model.summary(print_fn=lambda x: string_list.append(x))
            info["summary"] = "\n".join(string_list)
            
        else:
            info.update({
                "type": "generic_model",
                "class": model.__class__.__name__,
                "module": model.__class__.__module__
            })
        
    except Exception as e:
        logger.warning(f"Could not get complete model info: {e}")
        info["error"] = str(e)
    
    return info

def export_model_for_production(
    model: Any,
    export_path: str,
    format: str = "saved_model"
):
    """
    Exporte un modèle pour la production.
    
    Args:
        model: Modèle à exporter
        export_path: Chemin d'export
        format: Format d'export ("saved_model", "h5", "tflite")
    """
    try:
        export_path = Path(export_path)
        os.makedirs(export_path.parent, exist_ok=True)
        
        if isinstance(model, tf.keras.Model):
            if format == "saved_model":
                tf.saved_model.save(model, str(export_path))
                logger.info(f"Model exported as SavedModel to {export_path}")
            elif format == "h5":
                model.save(str(export_path), save_format='h5')
                logger.info(f"Model exported as H5 to {export_path}")
            elif format == "tflite":
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                tflite_model = converter.convert()
                with open(export_path, 'wb') as f:
                    f.write(tflite_model)
                logger.info(f"Model exported as TensorFlow Lite to {export_path}")
            else:
                raise ValueError(f"Unsupported export format: {format}")
        else:
            logger.warning("Export for production only supports Keras models")
            save_model(model, str(export_path))
        
    except Exception as e:
        logger.error(f"Failed to export model for production: {e}")
        raise

# === ALIAS POUR COMPATIBILITÉ ===

# Alias pour la fonction principale de sauvegarde CV
save_cv_model = save_computer_vision_model
load_cv_model = load_computer_vision_model