"""
Utilitaires de localisation pour la détection d'anomalies.
Génération de heatmaps, masks binaires et alignement des dimensions.
"""
import numpy as np
from typing import Tuple, Optional, Dict, Any
from src.shared.logging import get_logger

logger = get_logger(__name__)


def generate_binary_mask(
    error_map: np.ndarray,
    threshold: Optional[float] = None,
    method: str = "percentile",
    percentile: float = 95.0
) -> np.ndarray:
    """
    Génère un mask binaire à partir d'une carte d'erreur.
    
    Args:
        error_map: Carte d'erreur (H, W) ou (B, H, W) ou (B, 1, H, W)
        threshold: Seuil absolu (si method="absolute") ou percentile (si method="percentile")
        method: "percentile" ou "absolute"
        percentile: Percentile à utiliser si method="percentile" (0-100)
    
    Returns:
        Mask binaire (H, W) ou (B, H, W) avec valeurs 0 ou 1
    
    Raises:
        ValueError: Si format invalide
    """
    try:
        # Normalisation shape: gérer (B, 1, H, W) → (B, H, W) ou (H, W)
        original_shape = error_map.shape
        
        if error_map.ndim == 4:
            # (B, 1, H, W) → (B, H, W)
            if error_map.shape[1] == 1:
                error_map = error_map[:, 0, :, :]
            else:
                raise ValueError(f"Shape 4D inattendue: {original_shape}. Attendu (B, 1, H, W)")
        elif error_map.ndim == 3:
            # (B, H, W) - OK
            pass
        elif error_map.ndim == 2:
            # (H, W) - OK
            pass
        else:
            raise ValueError(f"Shape invalide: {original_shape}. Attendu 2D, 3D ou 4D")
        
        # Calcul du seuil
        if method == "percentile":
            if threshold is None:
                threshold = percentile
            actual_threshold = np.percentile(error_map, threshold)
        elif method == "absolute":
            if threshold is None:
                raise ValueError("threshold doit être fourni si method='absolute'")
            actual_threshold = threshold
        else:
            raise ValueError(f"Méthode invalide: {method}. Utilisez 'percentile' ou 'absolute'")
        
        # Génération du mask binaire
        mask = (error_map > actual_threshold).astype(np.uint8)
        
        logger.debug(
            f"Mask binaire généré - shape: {mask.shape}, "
            f"threshold: {actual_threshold:.4f}, "
            f"method: {method}, "
            f"anomaly_pixels: {mask.sum()} / {mask.size} ({mask.sum() / mask.size * 100:.1f}%)"
        )
        
        return mask
        
    except Exception as e:
        logger.error(f"Erreur génération mask binaire: {e}", exc_info=True)
        raise


def resize_error_map(
    error_map: np.ndarray,
    target_size: Tuple[int, int],
    method: str = "bilinear"
) -> np.ndarray:
    """
    Redimensionne une carte d'erreur vers une taille cible.
    
    Args:
        error_map: Carte d'erreur (H, W) ou (B, H, W) ou (B, 1, H, W)
        target_size: Taille cible (target_h, target_w)
        method: Méthode de resize ("bilinear", "nearest", "cubic")
    
    Returns:
        Carte d'erreur redimensionnée avec la même structure dimensionnelle
    
    Raises:
        ValueError: Si format invalide
    """
    try:
        from scipy.ndimage import zoom
        
        # Normalisation shape
        original_shape = error_map.shape
        is_batch = False
        is_4d = False
        
        if error_map.ndim == 4:
            # (B, 1, H, W)
            is_4d = True
            is_batch = True
            B, C, H, W = error_map.shape
            error_map_2d = error_map[:, 0, :, :]  # (B, H, W)
        elif error_map.ndim == 3:
            # (B, H, W)
            is_batch = True
            error_map_2d = error_map
            H, W = error_map.shape[1], error_map.shape[2]
        elif error_map.ndim == 2:
            # (H, W)
            error_map_2d = error_map[np.newaxis, :, :]  # (1, H, W) pour traitement
            H, W = error_map.shape
        else:
            raise ValueError(f"Shape invalide: {original_shape}")
        
        target_h, target_w = target_size
        
        # Calcul facteurs de zoom
        zoom_factors = (target_h / H, target_w / W)
        
        # Resize pour chaque image du batch
        resized_maps = []
        for img_map in error_map_2d:
            if method == "nearest":
                order = 0
            elif method == "bilinear":
                order = 1
            elif method == "cubic":
                order = 3
            else:
                order = 1
            
            resized = zoom(img_map, zoom_factors, order=order, mode='constant', cval=0.0)
            resized_maps.append(resized)
        
        result = np.array(resized_maps)
        
        # Restauration de la structure originale
        if not is_batch and error_map.ndim == 2:
            result = result[0]  # (H, W)
        elif is_4d:
            result = result[:, np.newaxis, :, :]  # (B, 1, H, W)
        # Sinon déjà (B, H, W)
        
        logger.debug(
            f"Error map resized: {original_shape} → {result.shape}, "
            f"target: {target_size}, method: {method}"
        )
        
        return result
        
    except ImportError:
        # Fallback: resize simple avec numpy (moins précis)
        logger.warning("⚠️ scipy.ndimage non disponible, utilisation resize numpy (moins précis)")
        
        if error_map.ndim == 4:
            B, C, H, W = error_map.shape
            error_map_2d = error_map[:, 0, :, :]
        elif error_map.ndim == 3:
            error_map_2d = error_map
        else:
            error_map_2d = error_map[np.newaxis, :, :]
        
        target_h, target_w = target_size
        
        # Resize simple par répétition/interpolation
        resized = []
        for img in error_map_2d:
            from PIL import Image
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_resized = img_pil.resize((target_w, target_h), Image.BILINEAR)
            resized.append(np.array(img_resized).astype(np.float32) / 255.0)
        
        result = np.array(resized)
        
        if error_map.ndim == 4:
            result = result[:, np.newaxis, :, :]
        elif error_map.ndim == 2:
            result = result[0]
        
        return result
    
    except Exception as e:
        logger.error(f"Erreur resize error_map: {e}", exc_info=True)
        raise


def align_heatmap_to_image(
    heatmap: np.ndarray,
    original_image_shape: Tuple[int, int],
    processed_image_shape: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Aligne une heatmap à la taille de l'image originale.
    
    Gère les cas où l'image a été resizée pendant le preprocessing.
    
    Args:
        heatmap: Carte de chaleur (H, W) ou (B, H, W)
        original_image_shape: Taille de l'image originale (H_orig, W_orig)
        processed_image_shape: Taille de l'image après preprocessing (H_proc, W_proc)
            Si None, utilise la shape actuelle de heatmap
    
    Returns:
        Heatmap alignée à la taille originale (H_orig, W_orig) ou (B, H_orig, W_orig)
    """
    try:
        if processed_image_shape is None:
            if heatmap.ndim == 2:
                processed_image_shape = heatmap.shape
            elif heatmap.ndim == 3:
                processed_image_shape = heatmap.shape[1:]
            else:
                raise ValueError(f"Shape heatmap invalide: {heatmap.shape}")
        
        # Vérifier si resize nécessaire
        if processed_image_shape == original_image_shape:
            logger.debug("Heatmap déjà alignée, pas de resize nécessaire")
            return heatmap
        
        # Resize
        aligned_heatmap = resize_error_map(heatmap, original_image_shape, method="bilinear")
        
        logger.info(
            f"✅ Heatmap alignée: {processed_image_shape} → {original_image_shape}"
        )
        
        return aligned_heatmap
        
    except Exception as e:
        logger.error(f"Erreur alignement heatmap: {e}", exc_info=True)
        raise


def generate_anomaly_localization(
    model: Any,
    images: np.ndarray,
    model_type: str,
    return_masks: bool = True,
    mask_threshold_percentile: float = 95.0
) -> Dict[str, np.ndarray]:
    """
    Génère la localisation complète des anomalies pour un batch d'images.
    
    Args:
        model: Modèle PyTorch (autoencoder ou classifier)
        images: Images d'entrée (B, C, H, W) en format PyTorch
        model_type: Type de modèle ("autoencoder", "conv_autoencoder", etc.)
        return_masks: Si True, génère aussi les masks binaires
        mask_threshold_percentile: Percentile pour le seuil du mask
    
    Returns:
        Dict avec:
            - 'error_maps': (B, H, W) Cartes d'erreur spatiales
            - 'heatmaps': (B, H, W) Heatmaps normalisées [0, 1]
            - 'binary_masks': (B, H, W) Masks binaires si return_masks=True
            - 'max_errors': (B,) Scores d'anomalie par image
    """
    import torch
    
    try:
        model.eval()
        device = next(model.parameters()).device
        
        # Conversion si nécessaire
        if not isinstance(images, torch.Tensor):
            images_tensor = torch.tensor(images, dtype=torch.float32).to(device)
        else:
            images_tensor = images.to(device)
        
        with torch.no_grad():
            if model_type in ["autoencoder", "conv_autoencoder", "variational_autoencoder", "denoising_autoencoder"]:
                # Autoencoder: utiliser get_reconstruction_error_map
                if hasattr(model, 'get_reconstruction_error_map'):
                    # Génération des cartes d'erreur
                    error_maps = model.get_reconstruction_error_map(images_tensor)
                    # Conversion (B, 1, H, W) → (B, H, W)
                    error_maps_np = error_maps[:, 0, :, :].cpu().numpy()
                else:
                    # Fallback: calcul manuel
                    reconstructed = model(images_tensor)
                    error_maps = torch.mean((images_tensor - reconstructed) ** 2, dim=1, keepdim=True)
                    error_maps_np = error_maps[:, 0, :, :].cpu().numpy()
                
            else:
                # Classification: utiliser gradients ou attention maps
                images_tensor.requires_grad = True
                output = model(images_tensor)
                
                # Score par classe positive
                if output.shape[1] == 2:
                    score = output[:, 1]
                else:
                    score = output.max(dim=1)[0]
                
                # Backward pour obtenir gradients
                score.sum().backward()
                gradients = images_tensor.grad.abs()
                
                # Heatmap = magnitude des gradients moyennée sur les canaux
                error_maps = torch.mean(gradients, dim=1, keepdim=True)
                error_maps_np = error_maps[:, 0, :, :].cpu().numpy()
        
        # Normalisation des heatmaps [0, 1]
        heatmaps = []
        max_errors = []
        
        for i in range(len(error_maps_np)):
            error_map = error_maps_np[i]
            max_error = error_map.max()
            min_error = error_map.min()
            
            if max_error > min_error:
                normalized = (error_map - min_error) / (max_error - min_error + 1e-8)
            else:
                normalized = np.zeros_like(error_map)
            
            heatmaps.append(normalized)
            max_errors.append(float(max_error))
        
        heatmaps = np.array(heatmaps)
        max_errors = np.array(max_errors)
        
        result = {
            'error_maps': error_maps_np,
            'heatmaps': heatmaps,
            'max_errors': max_errors
        }
        
        # Génération des masks binaires si demandé
        if return_masks:
            binary_masks = []
            for error_map in error_maps_np:
                mask = generate_binary_mask(
                    error_map,
                    method="percentile",
                    percentile=mask_threshold_percentile
                )
                binary_masks.append(mask)
            
            result['binary_masks'] = np.array(binary_masks)
        
        logger.info(
            f"✅ Localisation générée pour {len(images_tensor)} images - "
            f"shapes: error_maps={error_maps_np.shape}, heatmaps={heatmaps.shape}"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Erreur génération localisation: {e}", exc_info=True)
        raise


