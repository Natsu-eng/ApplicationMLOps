"""
Module d'augmentation de données pour images.
Optimisé pour la détection d'anomalies avec configurations prédéfinies.
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Any
import albumentations as A
from albumentations.core.composition import Compose
import logging
from src.shared.logging import get_logger

logger = get_logger(__name__)

# === CONFIGURATIONS PRÉDÉFINIES POUR L'AUGMENTATION ===

class AugmentationConfigs:
    """Configurations prédéfinies pour l'augmentation de données."""
    
    # Configuration légère - transformations subtiles
    LIGHT = {
        "rotation_range": 10,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "brightness_range": [0.9, 1.1],
        "contrast_range": [0.9, 1.1],
        "horizontal_flip": True,
        "vertical_flip": False,
        "fill_mode": "reflect"
    }
    
    # Configuration par défaut - équilibrée
    DEFAULT = {
        "rotation_range": 15,
        "width_shift_range": 0.15,
        "height_shift_range": 0.15,
        "brightness_range": [0.8, 1.2],
        "contrast_range": [0.8, 1.2],
        "horizontal_flip": True,
        "vertical_flip": True,
        "fill_mode": "reflect"
    }
    
    # Configuration forte - transformations agressives
    HEAVY = {
        "rotation_range": 25,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "brightness_range": [0.7, 1.3],
        "contrast_range": [0.7, 1.3],
        "horizontal_flip": True,
        "vertical_flip": True,
        "fill_mode": "reflect"
    }
    
    # Configuration spéciale pour détection d'anomalies
    ANOMALY = {
        "rotation_range": 20,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "brightness_range": [0.6, 1.4],  # Large range pour anomalies lumineuses/sombres
        "contrast_range": [0.6, 1.4],
        "horizontal_flip": True,
        "vertical_flip": False,
        "fill_mode": "constant",
        "fill_value": 0
    }
    
    # Configuration pour équilibrage des classes (augmentation ciblée)
    BALANCED = {
        "rotation_range": 30,
        "width_shift_range": 0.3,
        "height_shift_range": 0.3,
        "brightness_range": [0.5, 1.5],
        "contrast_range": [0.5, 1.5],
        "horizontal_flip": True,
        "vertical_flip": True,
        "fill_mode": "constant",
        "fill_value": 0
    }

# === CLASSES POUR L'AUGMENTATION ===

class ImageAugmentor:
    """
    Classe pour l'augmentation d'images avec différentes stratégies.
    Utilise Albumentations pour des transformations rapides et optimisées.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise l'augmentateur avec une configuration.
        
        Args:
            config: Configuration des transformations
        """
        self.config = config
        self.transform = self._build_transform()
        logger.info(f"ImageAugmentor initialisé avec config: {config.get('name', 'custom')}")
    
    def _build_transform(self) -> Compose:
        """Construit le pipeline de transformations Albumentations."""
        transforms = []
        
        # Rotation
        if self.config.get("rotation_range", 0) > 0:
            transforms.append(
                A.Rotate(
                    limit=self.config["rotation_range"],
                    p=0.7,
                    border_mode=cv2.BORDER_REFLECT if self.config.get("fill_mode") == "reflect" else cv2.BORDER_CONSTANT,
                    value=self.config.get("fill_value", 0)
                )
            )
        
        # Shifts
        if self.config.get("width_shift_range", 0) > 0 or self.config.get("height_shift_range", 0) > 0:
            transforms.append(
                A.ShiftScaleRotate(
                    shift_limit_x=self.config.get("width_shift_range", 0),
                    shift_limit_y=self.config.get("height_shift_range", 0),
                    rotate_limit=0,  # Rotation gérée séparément
                    p=0.7,
                    border_mode=cv2.BORDER_REFLECT if self.config.get("fill_mode") == "reflect" else cv2.BORDER_CONSTANT,
                    value=self.config.get("fill_value", 0)
                )
            )
        
        # Brightness et Contrast
        brightness_range = self.config.get("brightness_range")
        contrast_range = self.config.get("contrast_range")
        
        if brightness_range or contrast_range:
            transforms.append(
                A.RandomBrightnessContrast(
                    brightness_limit=(
                        (brightness_range[1] - 1.0) if brightness_range else 0.2,
                        (1.0 - brightness_range[0]) if brightness_range else 0.2
                    ) if brightness_range else 0.2,
                    contrast_limit=(
                        (contrast_range[1] - 1.0) if contrast_range else 0.2,
                        (1.0 - contrast_range[0]) if contrast_range else 0.2
                    ) if contrast_range else 0.2,
                    p=0.5
                )
            )
        
        # Flips
        if self.config.get("horizontal_flip", False):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        if self.config.get("vertical_flip", False):
            transforms.append(A.VerticalFlip(p=0.5))
        
        # Ajouter du bruit gaussien pour les configurations robustes
        if self.config.get("name") in ["heavy", "anomaly", "balanced"]:
            transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.3))
        
        # Flou gaussien pour simuler des défauts de mise au point
        if self.config.get("name") in ["anomaly", "balanced"]:
            transforms.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))
        
        # Compression JPEG pour simuler des artefacts
        if self.config.get("name") == "anomaly":
            transforms.append(A.ImageCompression(quality_lower=70, quality_upper=95, p=0.2))
        
        # Vérifier qu'on a au moins une transformation
        if not transforms:
            transforms.append(A.NoOp(p=1.0))
        
        return Compose(transforms)
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Applique l'augmentation à une image.
        
        Args:
            image: Image d'entrée (H, W, C)
            
        Returns:
            Image augmentée
        """
        try:
            # Vérifier les dimensions
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            
            # Appliquer les transformations
            augmented = self.transform(image=image)
            
            return augmented["image"]
            
        except Exception as e:
            logger.error(f"Erreur lors de l'augmentation: {e}")
            return image  # Retourner l'image originale en cas d'erreur
    
    def augment_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Applique l'augmentation à un batch d'images.
        
        Args:
            images: Batch d'images (N, H, W, C)
            
        Returns:
            Batch d'images augmentées
        """
        augmented_images = []
        
        for i in range(len(images)):
            augmented_img = self.augment(images[i])
            augmented_images.append(augmented_img)
        
        return np.array(augmented_images)

# === FONCTIONS PRINCIPALES ===

def get_augmentor_config(config_name: str = "default") -> ImageAugmentor:
    """
    Retourne un augmentateur configuré selon le nom de configuration.
    
    Args:
        config_name: Nom de la configuration ("light", "default", "heavy", "anomaly", "balanced")
        
    Returns:
        ImageAugmentor configuré
        
    Raises:
        ValueError: Si le nom de configuration n'est pas reconnu
    """
    config_mapping = {
        "light": AugmentationConfigs.LIGHT,
        "default": AugmentationConfigs.DEFAULT,
        "heavy": AugmentationConfigs.HEAVY,
        "anomaly": AugmentationConfigs.ANOMALY,
        "balanced": AugmentationConfigs.BALANCED
    }
    
    if config_name not in config_mapping:
        raise ValueError(f"Configuration non reconnue: {config_name}. Options: {list(config_mapping.keys())}")
    
    config = config_mapping[config_name].copy()
    config["name"] = config_name  # Ajouter le nom pour référence
    
    logger.info(f"Configuration d'augmentation chargée: {config_name}")
    return ImageAugmentor(config)

def create_custom_augmentor(**kwargs) -> ImageAugmentor:
    """
    Crée un augmentateur personnalisé avec des paramètres spécifiques.
    
    Args:
        **kwargs: Paramètres de configuration
        
    Returns:
        ImageAugmentor personnalisé
    """
    config = {
        "rotation_range": kwargs.get("rotation_range", 15),
        "width_shift_range": kwargs.get("width_shift_range", 0.15),
        "height_shift_range": kwargs.get("height_shift_range", 0.15),
        "brightness_range": kwargs.get("brightness_range", [0.8, 1.2]),
        "contrast_range": kwargs.get("contrast_range", [0.8, 1.2]),
        "horizontal_flip": kwargs.get("horizontal_flip", True),
        "vertical_flip": kwargs.get("vertical_flip", False),
        "fill_mode": kwargs.get("fill_mode", "reflect"),
        "fill_value": kwargs.get("fill_value", 0),
        "name": "custom"
    }
    
    logger.info("Augmentateur personnalisé créé")
    return ImageAugmentor(config)

def augment_dataset_for_training(
    X: np.ndarray, 
    y: np.ndarray, 
    augmentor: ImageAugmentor, 
    augmentation_factor: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augmente un dataset complet pour l'entraînement.
    
    Args:
        X: Images d'entrée (N, H, W, C)
        y: Labels correspondants (N,)
        augmentor: Augmentateur à utiliser
        augmentation_factor: Nombre de variantes à générer par image
        
    Returns:
        Tuple (X_augmented, y_augmented) - Dataset augmenté
    """
    if augmentation_factor < 1:
        raise ValueError("Le facteur d'augmentation doit être >= 1")
    
    if len(X) != len(y):
        raise ValueError("X et y doivent avoir la même longueur")
    
    logger.info(f"Début de l'augmentation du dataset: {len(X)} images → {len(X) * augmentation_factor} images")
    
    X_augmented = []
    y_augmented = []
    
    # Ajouter les images originales
    X_augmented.extend(X)
    y_augmented.extend(y)
    
    # Générer les images augmentées
    for i in range(augmentation_factor):
        for j in range(len(X)):
            augmented_image = augmentor.augment(X[j])
            X_augmented.append(augmented_image)
            y_augmented.append(y[j])
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    logger.info(f"Augmentation terminée: {len(X_augmented)} images totales")
    
    return X_augmented, y_augmented

def augment_dataset_balanced(
    X: np.ndarray,
    y: np.ndarray,
    target_samples_per_class: int,
    max_augmentation_factor: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Augmente le dataset de manière équilibrée pour résoudre le déséquilibre des classes.
    
    Args:
        X: Images d'entrée
        y: Labels
        target_samples_per_class: Nombre cible d'échantillons par classe
        max_augmentation_factor: Facteur d'augmentation maximum par classe
        
    Returns:
        Dataset équilibré
    """
    from collections import Counter
    
    label_counts = Counter(y)
    unique_classes = np.unique(y)
    
    X_balanced = []
    y_balanced = []
    
    logger.info(f"Équilibrage des classes. Target: {target_samples_per_class} par classe")
    
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        current_count = len(class_indices)
        
        # Ajouter les échantillons originaux
        X_balanced.extend(X[class_indices])
        y_balanced.extend([class_label] * current_count)
        
        # Calculer le nombre d'échantillons à générer
        needed_samples = max(0, target_samples_per_class - current_count)
        
        if needed_samples > 0:
            # Limiter le facteur d'augmentation
            augmentation_factor = min(
                max_augmentation_factor,
                (needed_samples // current_count) + 1
            )
            
            # Créer un augmentateur pour cette classe
            augmentor = get_augmentor_config("balanced")
            
            # Générer des échantillons supplémentaires
            generated_count = 0
            while generated_count < needed_samples:
                for idx in class_indices:
                    if generated_count >= needed_samples:
                        break
                    
                    augmented_image = augmentor.augment(X[idx])
                    X_balanced.append(augmented_image)
                    y_balanced.append(class_label)
                    generated_count += 1
            
            logger.info(f"Classe {class_label}: {current_count} → {current_count + generated_count} échantillons")
    
    X_balanced = np.array(X_balanced)
    y_balanced = np.array(y_balanced)
    
    final_counts = Counter(y_balanced)
    logger.info(f"Équilibrage terminé. Distribution finale: {dict(final_counts)}")
    
    return X_balanced, y_balanced

# === FONCTIONS UTILITAIRES ===

def preview_augmentations(
    image: np.ndarray, 
    augmentor: ImageAugmentor, 
    n_previews: int = 5
) -> List[np.ndarray]:
    """
    Génère des aperçus d'augmentation pour une image.
    
    Args:
        image: Image originale
        augmentor: Augmentateur à utiliser
        n_previews: Nombre d'aperçus à générer
        
    Returns:
        Liste des images augmentées
    """
    previews = []
    
    for i in range(n_previews):
        augmented = augmentor.augment(image)
        previews.append(augmented)
    
    return previews

def get_augmentation_config_info(config_name: str) -> Dict[str, Any]:
    """
    Retourne les informations détaillées d'une configuration d'augmentation.
    
    Args:
        config_name: Nom de la configuration
        
    Returns:
        Informations détaillées
    """
    config_mapping = {
        "light": {
            "name": "Légère",
            "description": "Transformations subtiles pour préserver les caractéristiques originales",
            "use_cases": ["Données déjà variées", "Préservation des détails fins", "Entraînement initial"]
        },
        "default": {
            "name": "Standard",
            "description": "Équilibre entre variété et préservation des caractéristiques",
            "use_cases": ["Usage général", "Bon compromis performance/stabilité"]
        },
        "heavy": {
            "name": "Forte",
            "description": "Transformations agressives pour une grande variété",
            "use_cases": ["Données limitées", "Robustesse aux variations", "Overfitting prevention"]
        },
        "anomaly": {
            "name": "Spéciale Anomalies",
            "description": "Optimisée pour la détection d'anomalies avec variations lumineuses et artefacts",
            "use_cases": ["Détection d'anomalies", "Variations lumineuses", "Artefacts d'images"]
        },
        "balanced": {
            "name": "Équilibrage",
            "description": "Augmentation agressive pour résoudre le déséquilibre des classes",
            "use_cases": ["Classes déséquilibrées", "Augmentation ciblée", "Balancement de dataset"]
        }
    }
    
    if config_name not in config_mapping:
        raise ValueError(f"Configuration non reconnue: {config_name}")
    
    return config_mapping[config_name]

# === FONCTION DE TEST ===

def test_augmentor():
    """Fonction de test pour vérifier le bon fonctionnement de l'augmentation."""
    try:
        # Créer une image de test
        test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        
        # Tester chaque configuration
        for config_name in ["light", "default", "heavy", "anomaly", "balanced"]:
            augmentor = get_augmentor_config(config_name)
            augmented = augmentor.augment(test_image)
            
            print(f"✅ Configuration {config_name}: OK - Shape: {augmented.shape}, Range: [{augmented.min()}, {augmented.max()}]")
        
        # Tester l'augmentation de batch
        test_batch = np.random.randint(0, 255, (10, 64, 64, 3), dtype=np.uint8)
        augmentor = get_augmentor_config("default")
        augmented_batch = augmentor.augment_batch(test_batch)
        
        print(f"✅ Augmentation de batch: OK - Shape: {augmented_batch.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test échoué: {e}")
        return False

if __name__ == "__main__":
    # Exécuter les tests si le fichier est exécuté directement
    test_augmentor()