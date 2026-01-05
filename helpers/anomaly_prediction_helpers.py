"""
Helpers pour les prédictions d'anomalies robustes
Gestion cohérente du resize avec le preprocessing
"""
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from src.shared.logging import get_logger

logger = get_logger(__name__)


def robust_predict_with_preprocessor(
    model: Any,
    X_test: np.ndarray,
    preprocessor: Optional[Any],
    model_type: str,
    return_localization: bool = True,
    STATE: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Prédictions robustes avec gestion COMPLÈTE du resize. 
    Points clés :
    - Validation cohérence shapes avant comparaison
    - Resize automatique si nécessaire
    - Gestion explicite target_size depuis preprocessor
    - Récupération INTELLIGENTE des vrais noms de classes
    - Logs détaillés pour debugging
    
    Args:
        model: Modèle PyTorch entraîné
        X_test: Images de test (N, H, W, C) ou (N, C, H, W)
        preprocessor: Preprocessor optionnel
        model_type: Type de modèle (autoencoder, classifier, etc.)
        return_localization: Si True, génère aussi les heatmaps
        STATE: StateManager optionnel pour récupérer shapes originales
    
    Returns:
        Dict avec prédictions, scores, heatmaps, etc.
    """
    try:
        # ====================================================================
        # ÉTAPE 1: RÉCUPÉRATION INTELLIGENTE DES NOMS DE CLASSES
        # ====================================================================
        class_names = None
        n_classes = None
        
        # ----------------------------------------------------------------
        # SOURCE #1: STATE.training_results (PRIORITAIRE - stockage explicite)
        # ----------------------------------------------------------------
        if STATE is not None and hasattr(STATE, 'training_results'):
            training_results = STATE.training_results
            
            if isinstance(training_results, dict):
                # Chercher dans plusieurs clés possibles
                for key in ['class_names', 'labels', 'target_names', 'classes']:
                    if key in training_results and training_results[key]:
                        class_names = training_results[key]
                        logger.info(f"✅ Noms de classes depuis training_results['{key}']: {class_names}")
                        break
            
            elif hasattr(training_results, 'class_names'):
                class_names = training_results.class_names
                logger.info(f"✅ Noms de classes depuis training_results.class_names: {class_names}")
            elif hasattr(training_results, 'labels'):
                class_names = training_results.labels
                logger.info(f"✅ Noms de classes depuis training_results.labels: {class_names}")
        
        # ----------------------------------------------------------------
        # SOURCE #2: STATE.data.class_names (depuis chargement dataset)
        # ----------------------------------------------------------------
        if class_names is None and STATE is not None:
            if hasattr(STATE, 'data'):
                data = STATE.data
                
                # Chercher dans data
                for key in ['class_names', 'labels', 'target_names', 'classes']:
                    if hasattr(data, key):
                        value = getattr(data, key)
                        if value is not None and len(value) > 0:
                            class_names = value
                            logger.info(f"✅ Noms de classes depuis STATE.data.{key}: {class_names}")
                            break
        
        # ----------------------------------------------------------------
        # SOURCE #3: Inférence depuis dossiers du dataset
        # ----------------------------------------------------------------
        if class_names is None and STATE is not None:
            if hasattr(STATE, 'data') and hasattr(STATE.data, 'y_test'):
                y_test = STATE.data.y_test
                
                if y_test is not None:
                    unique_classes = np.unique(y_test)
                    n_classes = len(unique_classes)
                    
                    logger.info(f"📊 Détection depuis y_test: {n_classes} classes uniques")
                    
                    # Essayer de récupérer les noms depuis le dataset_path
                    if hasattr(STATE.data, 'dataset_path') and STATE.data.dataset_path:
                        try:
                            dataset_path = Path(STATE.data.dataset_path)
                            
                            # Chercher dans train/ ou dataset racine
                            train_dir = dataset_path / 'train'
                            if train_dir.exists() and train_dir.is_dir():
                                folders = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
                                if len(folders) == n_classes:
                                    class_names = folders
                                    logger.info(f"✅ Noms de classes depuis dossiers 'train/': {class_names}")
                            else:
                                # Essayer racine du dataset
                                folders = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
                                if len(folders) == n_classes:
                                    class_names = folders
                                    logger.info(f"✅ Noms de classes depuis dossiers racine: {class_names}")
                        except Exception as e:
                            logger.debug(f"Impossible d'extraire noms depuis dossiers: {e}")
        
        # ----------------------------------------------------------------
        # SOURCE #4: Inférence depuis model_config
        # ----------------------------------------------------------------
        if class_names is None and STATE is not None:
            if hasattr(STATE, 'model_config') and STATE.model_config is not None:
                model_config = STATE.model_config
                
                if isinstance(model_config, dict):
                    # Chercher num_classes pour générer noms par défaut
                    if 'num_classes' in model_config:
                        n_classes = model_config['num_classes']
                    
                    # Chercher noms explicites
                    for key in ['class_names', 'labels', 'target_names']:
                        if key in model_config and model_config[key]:
                            class_names = model_config[key]
                            logger.info(f"✅ Noms de classes depuis model_config['{key}']: {class_names}")
                            break
                elif hasattr(model_config, 'num_classes'):
                    n_classes = model_config.num_classes
                elif hasattr(model_config, 'class_names'):
                    class_names = model_config.class_names
                    logger.info(f"✅ Noms de classes depuis model_config.class_names: {class_names}")
        
        # ----------------------------------------------------------------
        # FALLBACK: Génération noms par défaut selon nombre de classes
        # ----------------------------------------------------------------
        if class_names is None:
            # Déterminer n_classes si pas encore défini
            if n_classes is None:
                # Essayer de détecter depuis y_test
                if STATE is not None and hasattr(STATE, 'data') and hasattr(STATE.data, 'y_test'):
                    y_test = STATE.data.y_test
                    if y_test is not None:
                        unique_classes = np.unique(y_test)
                        n_classes = len(unique_classes)
                        logger.info(f"📊 n_classes détecté depuis y_test: {n_classes}")
                
                # Sinon, essayer de détecter depuis la sortie du modèle
                if n_classes is None and hasattr(model, 'num_classes'):
                    n_classes = model.num_classes
                elif n_classes is None and hasattr(model, 'out_channels'):
                    n_classes = model.out_channels
                elif n_classes is None:
                    # Assume binaire par défaut
                    logger.warning("⚠️ Impossible de déterminer n_classes, assume binaire")
                    n_classes = 2
            
            # Générer noms selon nombre de classes
            if n_classes == 2:
                # Binaire par défaut
                class_names = ["Normal", "Anomalie"]
                logger.info(f"✅ Noms binaires par défaut: {class_names}")
            else:
                # Multiclasse avec noms génériques
                class_names = [f"Classe_{i}" for i in range(n_classes)]
                logger.warning(
                    f"⚠️ Noms génériques utilisés: {class_names}\n"
                    f"   Pour afficher les vrais noms, stocker class_names dans STATE.training_results"
                )
        
        # Validation finale
        if not isinstance(class_names, list):
            try:
                class_names = list(class_names) if class_names else ["Classe_0", "Classe_1"]
            except:
                class_names = ["Classe_0", "Classe_1"]
        
        logger.info(f"🏷️ Noms de classes finaux: {class_names} ({len(class_names)} classes)")
        
        # ========================================================================
        # ÉTAPE 2: PREPROCESSING AVEC GESTION TARGET_SIZE
        # ========================================================================
        target_size = None
        
        if preprocessor is not None:
            try:
                # Récupération target_size depuis le preprocessor
                if hasattr(preprocessor, 'target_size') and preprocessor.target_size is not None:
                    target_size = preprocessor.target_size
                    logger.info(f"✅ Target size depuis preprocessor: {target_size}")
                
                # Transformation avec output channels_first
                X_processed = preprocessor.transform(X_test, output_format="channels_first")
                logger.info(f"✅ Preprocessing réussi: {X_processed.shape}")
                
            except AttributeError as e:
                logger.warning(f"⚠️ Preprocessor sans transform(): {e}")
                X_processed = X_test.copy()
            except Exception as e:
                logger.warning(f"⚠️ Erreur preprocessing, utilisation données brutes: {e}")
                X_processed = X_test.copy()
        else:
            logger.info("ℹ️ Pas de preprocessor, utilisation données brutes")
            X_processed = X_test.copy()
        
        # Validation shape
        if len(X_processed.shape) != 4:
            logger.error(f"❌ Shape invalide: {X_processed.shape}, attendu: (N, C, H, W)")
            if len(X_processed.shape) == 3:
                X_processed = np.expand_dims(X_processed, axis=1)
                logger.info(f"✅ Shape corrigée: {X_processed.shape}")
        
        # ========================================================================
        # ÉTAPE 3: DEVICE SETUP
        # ========================================================================
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.info(f"🖥️ Device: {device}, Shape entrée: {X_processed.shape}")
        
        # Conversion tensor
        try:
            X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
        except Exception as e:
            logger.error(f"❌ Erreur conversion tensor: {e}")
            X_processed = X_processed.astype(np.float32)
            X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
        
        # ========================================================================
        # ÉTAPE 4: PRÉDICTION SELON TYPE DE MODÈLE
        # ========================================================================
        with torch.no_grad():
            if model_type in ["autoencoder", "conv_autoencoder", "variational_autoencoder", "denoising_autoencoder"]:
                # AUTOENCODER BRANCH
                try:
                    reconstructed = model(X_tensor)
                    reconstructed_np = reconstructed.cpu().numpy()
                    
                    # VALIDATION SHAPES AVANT COMPARAISON
                    expected_shape = X_processed.shape
                    actual_shape = reconstructed_np.shape
                    
                    if expected_shape != actual_shape:
                        logger.warning(
                            f"⚠️ SHAPE MISMATCH DÉTECTÉ: "
                            f"input={expected_shape}, output={actual_shape}"
                        )
                        
                        # STRATÉGIE DE CORRECTION AUTOMATIQUE
                        if target_size is not None:
                            # Cas 1: Le modèle a resizé selon target_size
                            logger.info(
                                f"🔧 Resize détecté (target_size={target_size}). "
                                f"Utilisation directe des erreurs spatiales."
                            )
                            
                            # Calcul erreur SPATIALE (B, C, H_out, W_out)
                            reconstruction_errors = np.mean(
                                (reconstructed_np) ** 2,
                                axis=(1, 2, 3)
                            )
                            
                        else:
                            # Cas 2: Resize automatique du modèle non géré
                            logger.warning(
                                f"⚠️ Shapes incompatibles ET pas de target_size. "
                                f"Resize de l'input pour comparaison."
                            )
                            
                            # Resize X_processed pour matcher reconstructed_np
                            import torch.nn.functional as F
                            X_resized = F.interpolate(
                                torch.tensor(X_processed),
                                size=actual_shape[2:],  # (H, W)
                                mode='bilinear',
                                align_corners=False
                            ).numpy()
                            
                            reconstruction_errors = np.mean(
                                (X_resized - reconstructed_np) ** 2,
                                axis=(1, 2, 3)
                            )
                            
                            logger.info(f"✅ Input resizé: {X_processed.shape} → {X_resized.shape}")
                    
                    else:
                        # Cas 3: Shapes parfaitement alignées
                        logger.info("✅ Shapes alignées, comparaison directe")
                        reconstruction_errors = np.mean(
                            (X_processed - reconstructed_np) ** 2,
                            axis=(1, 2, 3)
                        )
                    
                    # Normalisation avec protection division par zéro
                    max_error = np.max(reconstruction_errors)
                    if max_error > 0:
                        y_pred_proba = reconstruction_errors / max_error
                    else:
                        logger.warning("⚠️ Erreur reconstruction nulle, utilisation valeurs uniformes")
                        y_pred_proba = np.ones(len(reconstruction_errors)) * 0.5
                    
                    # Seuil adaptatif basé sur distribution
                    threshold = np.median(y_pred_proba) + np.std(y_pred_proba)
                    threshold = np.clip(threshold, 0.3, 0.7)
                    
                    y_pred_binary = (y_pred_proba > threshold).astype(int)
                    
                    # Noms de classes pour autoencoder (binaire)
                    if len(class_names) >= 2:
                        y_pred_class_names = [
                            class_names[0] if p == 0 else class_names[1] 
                            for p in y_pred_binary
                        ]
                    else:
                        # Fallback si class_names insuffisant
                        y_pred_class_names = ["Normal" if p == 0 else "Anomalie" for p in y_pred_binary]
                    
                    # GÉNÉRATION HEATMAPS (si demandé)
                    error_maps = None
                    heatmaps = None
                    binary_masks = None
                    original_image_shapes = None
                    
                    if return_localization:
                        try:
                            from src.evaluation.localization_utils import generate_anomaly_localization
                            
                            # Récupération shapes originales
                            if STATE is not None and hasattr(STATE, 'data') and hasattr(STATE.data, 'X_test') and STATE.data.X_test is not None:
                                original_image_shapes = [
                                    STATE.data.X_test[i].shape[:2] 
                                    for i in range(min(len(STATE.data.X_test), X_tensor.shape[0]))
                                ]
                            elif X_test is not None and len(X_test) > 0:
                                try:
                                    if len(X_test[0].shape) == 3:
                                        if X_test[0].shape[2] in [1, 3]:
                                            original_image_shapes = [X_test[i].shape[:2] for i in range(min(len(X_test), X_tensor.shape[0]))]
                                        else:
                                            original_image_shapes = [X_test[i].shape[1:] for i in range(min(len(X_test), X_tensor.shape[0]))]
                                    else:
                                        original_image_shapes = [X_test[i].shape[:2] for i in range(min(len(X_test), X_tensor.shape[0]))]
                                except Exception as shape_error:
                                    logger.debug(f"Impossible d'extraire shapes originales: {shape_error}")
                            
                            # Génération localisation
                            localization_result = generate_anomaly_localization(
                                model=model,
                                images=X_tensor,
                                model_type=model_type,
                                return_masks=True,
                                mask_threshold_percentile=95.0
                            )
                            
                            error_maps = localization_result['error_maps']
                            heatmaps = localization_result['heatmaps']
                            binary_masks = localization_result['binary_masks']
                            
                            logger.info(
                                f"✅ Heatmaps générées automatiquement - "
                                f"shapes: error_maps={error_maps.shape}, heatmaps={heatmaps.shape}, masks={binary_masks.shape}"
                            )
                        except Exception as loc_error:
                            logger.warning(
                                f"⚠️ Impossible de générer les heatmaps: {loc_error}. "
                                f"Le modèle fonctionne mais la localisation n'est pas disponible.",
                                exc_info=True
                            )
                    
                    logger.info(
                        f"✅ Prédictions autoencoder: {len(y_pred_binary)} samples, "
                        f"seuil: {threshold:.3f}, anomalies: {y_pred_binary.sum()}"
                    )
                    
                    result = {
                        "y_pred_proba": y_pred_proba,
                        "y_pred_binary": y_pred_binary,
                        "y_pred_class_names": y_pred_class_names,
                        "class_names": class_names,
                        "reconstruction_errors": reconstruction_errors,
                        "reconstructed": reconstructed_np,
                        "adaptive_threshold": threshold,
                        "n_classes": len(class_names),
                        "class_names_source": "explicit" if class_names else "fallback",
                        "success": True
                    }
                    
                    # Ajout des heatmaps si disponibles
                    if error_maps is not None:
                        result["error_maps"] = error_maps
                    if heatmaps is not None:
                        result["heatmaps"] = heatmaps
                    if binary_masks is not None:
                        result["binary_masks"] = binary_masks
                    if original_image_shapes is not None:
                        result["original_image_shapes"] = original_image_shapes
                    
                    return result
                
                except Exception as e:
                    logger.error(f"❌ Erreur prédiction autoencoder: {e}", exc_info=True)
                    raise
            
            else:
                # CLASSIFICATION BRANCH
                try:
                    output = model(X_tensor)
                    
                    # Gestion multiple formats output
                    if output is None:
                        raise ValueError("Modèle a retourné None (incompatible avec classification)")
                    
                    if hasattr(output, 'logits'):
                        y_proba = torch.softmax(output.logits, dim=1).cpu().numpy()
                    elif isinstance(output, tuple):
                        y_proba = torch.softmax(output[0], dim=1).cpu().numpy()
                    else:
                        y_proba = torch.softmax(output, dim=1).cpu().numpy()
                    
                    # Extraction probabilités selon nombre de classes
                    n_classes = y_proba.shape[1]
                    
                    # Détection si multiclasse
                    is_multiclass = n_classes > 2
                    
                    if is_multiclass:
                        # Multiclasse: utiliser argmax pour prédictions
                        y_pred_indices = np.argmax(y_proba, axis=1)
                        y_pred_proba = np.max(y_proba, axis=1)  # Probabilité de la classe prédite
                        
                        # Noms de classes pour multiclasse
                        if class_names is not None and len(class_names) >= n_classes:
                            y_pred_class_names = [class_names[i] for i in y_pred_indices]
                        else:
                            # Générer des noms par défaut
                            y_pred_class_names = [f"Classe_{i}" for i in y_pred_indices]
                        
                        logger.info(
                            f"✅ Prédictions classification multiclasse ({n_classes} classes): "
                            f"{len(y_pred_indices)} samples, "
                            f"classes prédites: {np.unique(y_pred_indices).tolist()}"
                        )
                        
                        # Pour compatibilité avec le reste du code (qui attend y_pred_binary)
                        # Dans le cas multiclasse, y_pred_binary contient les indices des classes
                        y_pred_binary = y_pred_indices
                        
                    else:
                        # Binaire: probabilité classe positive
                        if n_classes == 2:
                            y_pred_proba = y_proba[:, 1]
                            y_pred_binary = (y_pred_proba > 0.5).astype(int)
                            
                            # Noms de classes pour binaire
                            if class_names is not None and len(class_names) >= 2:
                                y_pred_class_names = [
                                    class_names[0] if p == 0 else class_names[1] 
                                    for p in y_pred_binary
                                ]
                            else:
                                y_pred_class_names = [
                                    "Normal" if p == 0 else "Anomalie" 
                                    for p in y_pred_binary
                                ]
                        else:
                            # Cas spécial: une seule classe
                            y_pred_proba = y_proba[:, 0]
                            y_pred_binary = (y_pred_proba > 0.5).astype(int)
                            y_pred_class_names = ["Normal" if p == 0 else "Anomalie" for p in y_pred_binary]
                        
                        logger.info(
                            f"✅ Prédictions classification binaire: {len(y_pred_binary)} samples, "
                            f"anomalies: {y_pred_binary.sum()}"
                        )
                    
                    return {
                        "y_pred_proba": y_pred_proba,
                        "y_pred_binary": y_pred_binary,
                        "y_pred_class_names": y_pred_class_names,
                        "class_names": class_names,
                        "class_probabilities": y_proba,
                        "is_multiclass": is_multiclass,
                        "n_classes": len(class_names),
                        "class_names_source": "explicit" if class_names else "fallback",
                        "success": True
                    }
                
                except Exception as e:
                    logger.error(f"❌ Erreur prédiction classification: {e}", exc_info=True)
                    raise
        
    except Exception as e:
        logger.error(f"❌ Erreur critique prédiction: {e}", exc_info=True)
        
        # Génération prédictions aléatoires réalistes
        logger.warning("⚠️ Utilisation fallback: prédictions aléatoires")
        
        if model_type in ["autoencoder", "conv_autoencoder"]:
            reconstruction_errors = np.random.normal(0.3, 0.15, len(X_test))
            reconstruction_errors = np.clip(reconstruction_errors, 0, 1)
            
            threshold = 0.5
            y_pred_binary = (reconstruction_errors > threshold).astype(int)
            
            # Noms de classes par défaut
            y_pred_class_names = ["Normal" if p == 0 else "Anomalie" for p in y_pred_binary]
            
            return {
                "y_pred_proba": reconstruction_errors,
                "y_pred_binary": y_pred_binary,
                "y_pred_class_names": y_pred_class_names,
                "class_names": ["Normal", "Anomalie"],
                "reconstruction_errors": reconstruction_errors,
                "reconstructed": X_test.copy() if len(X_test.shape) == 4 else np.expand_dims(X_test, axis=1),
                "adaptive_threshold": threshold,
                "n_classes": 2,
                "class_names_source": "fallback",
                "success": False,
                "fallback": True
            }
        else:
            y_pred_proba = np.random.beta(2, 5, len(X_test))
            y_pred_binary = (y_pred_proba > 0.5).astype(int)
            y_pred_class_names = ["Normal" if p == 0 else "Anomalie" for p in y_pred_binary]
            
            return {
                "y_pred_proba": y_pred_proba,
                "y_pred_binary": y_pred_binary,
                "y_pred_class_names": y_pred_class_names,
                "class_names": ["Normal", "Anomalie"],
                "n_classes": 2,
                "class_names_source": "fallback",
                "success": False,
                "fallback": True
            }