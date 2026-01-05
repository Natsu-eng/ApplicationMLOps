"""
src/evaluation/grad_cam.py - GRAD-CAM PRODUCTION

Features:
- Support ResNet, VGG, EfficientNet, MobileNet, DenseNet
- Détection automatique de la target layer
- Visualisations multiples (heatmap, superposition, side-by-side)
- Batch processing
- Gestion erreurs robuste
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List, Tuple, Dict, Any, Union
from src.shared.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# GRAD-CAM CORE
# ============================================================================

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping
    
    Visualise les régions importantes pour la prédiction du modèle.
    
    Compatible avec:
    - ResNet (tous)
    - VGG (tous)
    - EfficientNet
    - MobileNet
    - DenseNet
    - CNNs custom
    
    Reference: https://arxiv.org/abs/1610.02391
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        use_cuda: bool = True
    ):
        """
        Args:
            model: Modèle PyTorch
            target_layer: Couche à analyser (auto-détectée si None)
            use_cuda: Utiliser GPU si disponible
        """
        self.model = model
        self.model.eval()
        
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        
        # Auto-détection target layer si non fournie
        if target_layer is None:
            self.target_layer = self._find_target_layer()
        else:
            self.target_layer = target_layer
        
        # Hooks pour capturer gradients et activations
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
        
        logger.info(
            f"✅ GradCAM initialisé:\n"
            f"   • Modèle: {model.__class__.__name__}\n"
            f"   • Target layer: {self.target_layer}\n"
            f"   • Device: {self.device}"
        )
    
    def _find_target_layer(self) -> nn.Module:
        """
        Auto-détection de la dernière couche convolutionnelle.
        
        Recherche dans l'ordre:
        1. layer4 (ResNet)
        2. features[-1] (VGG, EfficientNet, MobileNet)
        3. Dernière Conv2d trouvée
        """
        # ResNet, DenseNet
        if hasattr(self.model, 'layer4'):
            logger.info("🔍 Target layer détectée: layer4 (ResNet/DenseNet)")
            return self.model.layer4
        
        # VGG, EfficientNet, MobileNet
        if hasattr(self.model, 'features'):
            # Trouver dernière couche conv
            for i in reversed(range(len(self.model.features))):
                layer = self.model.features[i]
                if isinstance(layer, nn.Conv2d):
                    logger.info(f"🔍 Target layer détectée: features[{i}] (VGG/EfficientNet)")
                    return layer
            
            # Fallback: dernière couche
            logger.info("🔍 Target layer détectée: features[-1] (fallback)")
            return self.model.features[-1]
        
        # Transfer Learning avec backbone
        if hasattr(self.model, 'backbone'):
            return self._find_target_layer_in_module(self.model.backbone)
        
        # Recherche exhaustive
        return self._find_target_layer_in_module(self.model)
    
    def _find_target_layer_in_module(self, module: nn.Module) -> nn.Module:
        """Recherche récursive de la dernière Conv2d."""
        last_conv = None
        
        for child in module.modules():
            if isinstance(child, nn.Conv2d):
                last_conv = child
        
        if last_conv is None:
            raise ValueError(
                "❌ Aucune couche Conv2d trouvée dans le modèle.\n"
                "Grad-CAM nécessite un modèle CNN avec couches convolutionnelles."
            )
        
        logger.info(f"🔍 Target layer détectée: {last_conv} (recherche exhaustive)")
        return last_conv
    
    def _register_hooks(self):
        """Enregistre les hooks pour capturer activations et gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Génère la Grad-CAM pour une image.
        
        Args:
            input_tensor: Image (1, C, H, W)
            target_class: Classe cible (None = classe prédite)
        
        Returns:
            CAM normalisée (H, W) dans [0, 1]
        """
        # Forward pass
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True
        
        output = self.model(input_tensor)
        
        # Classe cible
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        
        # One-hot encoding de la classe cible
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Calcul Grad-CAM
        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)
        
        # Pondérations (Global Average Pooling des gradients)
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Weighted sum des activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU (garder valeurs positives)
        cam = np.maximum(cam, 0)
        
        # Normalisation [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_batch(
        self,
        input_tensor: torch.Tensor,
        target_classes: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Génère Grad-CAM pour un batch d'images.
        
        Args:
            input_tensor: Images (B, C, H, W)
            target_classes: Classes cibles (None = classes prédites)
        
        Returns:
            Liste de CAMs normalisées
        """
        batch_size = input_tensor.size(0)
        cams = []
        
        for i in range(batch_size):
            img = input_tensor[i:i+1]
            target = target_classes[i] if target_classes else None
            
            try:
                cam = self.generate_cam(img, target)
                cams.append(cam)
            except Exception as e:
                logger.error(f"❌ Erreur Grad-CAM image {i}: {e}")
                # Fallback: CAM vide
                cams.append(np.zeros((224, 224), dtype=np.float32))
        
        return cams
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """Alias pour generate_cam()"""
        return self.generate_cam(input_tensor, target_class)


# ============================================================================
# VISUALISATION
# ============================================================================

class GradCAMVisualizer:
    """
    Visualiseur Grad-CAM avec options multiples.
    
    Modes:
    - heatmap: Heatmap seule
    - overlay: Superposition image + heatmap
    - side_by_side: Image | Heatmap | Overlay
    """
    
    @staticmethod
    def apply_colormap(
        cam: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Applique une colormap à la CAM.
        
        Args:
            cam: CAM (H, W) dans [0, 1]
            colormap: Colormap OpenCV
        
        Returns:
            Heatmap RGB (H, W, 3) dans [0, 255]
        """
        # Conversion [0, 1] → [0, 255]
        cam_uint8 = np.uint8(255 * cam)
        
        # Application colormap
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        
        # BGR → RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    @staticmethod
    def overlay(
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Superpose la CAM sur l'image.
        
        Args:
            image: Image (H, W, 3) dans [0, 255] uint8
            cam: CAM (H', W') dans [0, 1]
            alpha: Transparence heatmap (0=invisible, 1=opaque)
            colormap: Colormap OpenCV
        
        Returns:
            Image avec overlay (H, W, 3) uint8
        """
        # Resize CAM vers taille image
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Génération heatmap
        heatmap = GradCAMVisualizer.apply_colormap(cam_resized, colormap)
        
        # Superposition
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay.astype(np.uint8)
    
    @staticmethod
    def side_by_side(
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Affichage côte à côte: Image | Heatmap | Overlay
        
        Returns:
            Image concaténée (H, W*3, 3)
        """
        # Resize CAM
        h, w = image.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # Heatmap
        heatmap = GradCAMVisualizer.apply_colormap(cam_resized, colormap)
        
        # Overlay
        overlay_img = GradCAMVisualizer.overlay(image, cam, alpha, colormap)
        
        # Concaténation horizontale
        result = np.concatenate([image, heatmap, overlay_img], axis=1)
        
        return result
    
    @staticmethod
    def create_figure(
        images: List[np.ndarray],
        cams: List[np.ndarray],
        predictions: Optional[List[str]] = None,
        ground_truths: Optional[List[str]] = None,
        mode: str = "overlay",
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Crée une figure avec plusieurs images.
        
        Args:
            images: Liste d'images (H, W, 3) uint8
            cams: Liste de CAMs (H, W) float32
            predictions: Prédictions (optionnel)
            ground_truths: Vérités terrain (optionnel)
            mode: "heatmap", "overlay", "side_by_side"
            alpha: Transparence
        
        Returns:
            Figure complète (grid)
        """
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        
        n_images = len(images)
        
        if mode == "side_by_side":
            # Une ligne par image
            fig, axes = plt.subplots(n_images, 1, figsize=(15, 5 * n_images))
            if n_images == 1:
                axes = [axes]
            
            for i, (img, cam) in enumerate(zip(images, cams)):
                vis = GradCAMVisualizer.side_by_side(img, cam, alpha)
                axes[i].imshow(vis)
                axes[i].axis('off')
                
                # Titre
                title_parts = []
                if predictions:
                    title_parts.append(f"Pred: {predictions[i]}")
                if ground_truths:
                    title_parts.append(f"GT: {ground_truths[i]}")
                
                if title_parts:
                    axes[i].set_title(" | ".join(title_parts), fontsize=12)
        
        else:
            # Grid 2 colonnes
            n_cols = 2
            n_rows = (n_images + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
            axes = axes.flatten() if n_images > 1 else [axes]
            
            for i, (img, cam) in enumerate(zip(images, cams)):
                if mode == "overlay":
                    vis = GradCAMVisualizer.overlay(img, cam, alpha)
                elif mode == "heatmap":
                    h, w = img.shape[:2]
                    cam_resized = cv2.resize(cam, (w, h))
                    vis = GradCAMVisualizer.apply_colormap(cam_resized)
                else:
                    vis = img
                
                axes[i].imshow(vis)
                axes[i].axis('off')
                
                # Titre
                title_parts = []
                if predictions:
                    title_parts.append(f"Pred: {predictions[i]}")
                if ground_truths:
                    title_parts.append(f"GT: {ground_truths[i]}")
                
                if title_parts:
                    axes[i].set_title(" | ".join(title_parts), fontsize=10)
            
            # Cacher axes vides
            for j in range(i + 1, len(axes)):
                axes[j].axis('off')
        
        plt.tight_layout()
        
        return fig


# ============================================================================
# HELPER FUNCTION PRODUCTION
# ============================================================================

def generate_gradcam_visualization(
    model: nn.Module,
    images: Union[torch.Tensor, np.ndarray],
    predictions: Optional[np.ndarray] = None,
    ground_truths: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    target_layer: Optional[nn.Module] = None,
    mode: str = "overlay",
    alpha: float = 0.5,
    use_cuda: bool = True
) -> Dict[str, Any]:
    """
    Génère des visualisations Grad-CAM prêtes pour affichage.
    
    Args:
        model: Modèle PyTorch
        images: Images (B, C, H, W) tensor ou (B, H, W, C) numpy
        predictions: Prédictions (B,) indices de classes
        ground_truths: Vérités terrain (B,)
        class_names: Noms des classes
        target_layer: Couche cible (None = auto)
        mode: "overlay", "heatmap", "side_by_side"
        alpha: Transparence
        use_cuda: Utiliser GPU
    
    Returns:
        Dict avec:
        - cams: Liste de CAMs (H, W)
        - overlays: Liste d'images avec overlay
        - figure: Matplotlib figure
        - success: bool
    """
    try:
        # Conversion images si numpy
        if isinstance(images, np.ndarray):
            # Assume (B, H, W, C) → (B, C, H, W)
            if images.shape[-1] in [1, 3]:
                images = np.transpose(images, (0, 3, 1, 2))
            
            images_tensor = torch.from_numpy(images).float()
        else:
            images_tensor = images
        
        # Normalisation [0, 1] si nécessaire
        if images_tensor.max() > 1:
            images_tensor = images_tensor / 255.0
        
        # Initialisation Grad-CAM
        grad_cam = GradCAM(model, target_layer, use_cuda)
        
        # Génération CAMs
        target_classes = predictions.tolist() if predictions is not None else None
        cams = grad_cam.generate_batch(images_tensor, target_classes)
        
        # Conversion images pour visualisation
        images_vis = []
        for img_tensor in images_tensor:
            # (C, H, W) → (H, W, C)
            img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
            
            # Normalisation [0, 255] uint8
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            
            # Grayscale → RGB
            if img_np.shape[2] == 1:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            
            images_vis.append(img_np)
        
        # Génération overlays
        overlays = []
        for img, cam in zip(images_vis, cams):
            overlay = GradCAMVisualizer.overlay(img, cam, alpha)
            overlays.append(overlay)
        
        # Noms de classes pour affichage
        pred_names = None
        gt_names = None
        
        if class_names:
            if predictions is not None:
                pred_names = [class_names[p] for p in predictions]
            if ground_truths is not None:
                gt_names = [class_names[gt] for gt in ground_truths]
        
        # Création figure
        figure = GradCAMVisualizer.create_figure(
            images_vis,
            cams,
            pred_names,
            gt_names,
            mode,
            alpha
        )
        
        logger.info(
            f"✅ Grad-CAM générées: {len(cams)} images, mode={mode}"
        )
        
        return {
            "cams": cams,
            "overlays": overlays,
            "figure": figure,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur génération Grad-CAM: {e}", exc_info=True)
        return {
            "cams": None,
            "overlays": None,
            "figure": None,
            "success": False,
            "error": str(e)
        }