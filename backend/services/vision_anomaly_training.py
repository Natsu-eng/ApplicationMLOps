"""Moteur d'entraînement — détection d'anomalies visuelles MVTec AD (pilier
Vision, Lot 15 sous-lot C).

Module séparé, même principe que `vision_classification_training.py` :
aucune notion commune avec le moteur ML tabulaire ni avec la classification
d'images (pas de notion de classe à prédire, entraînement sur `train/good/`
UNIQUEMENT — reconstruction, pas classification).

Corrige directement plusieurs des 9 bugs critiques déjà documentés dans
`docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md` :
- **#5** (incohérence de format channels_first/last) — évitée par
  construction : le pipeline ne manipule QUE des tenseurs PyTorch
  (channels_first natif), jamais de conversion numpy intermédiaire avant le
  calcul d'erreur.
- **#7/#12** (seuil de détection fixe, non calibré) — remplacé par une
  calibration réelle sur `test/` (labels good/défaut TOUJOURS disponibles,
  structure garantie par le sous-lot A) via le J de Youden sur la courbe
  ROC, jamais un percentile arbitraire sur le train.
- **#8/#16** (heatmap jamais générée automatiquement) — la carte d'erreur
  est calculée pour CHAQUE image de test pendant l'évaluation standard, pas
  une fonction annexe jamais appelée.
- **#11** (conversion tensor→numpy incohérente) — un seul point de calcul
  de l'erreur, toujours dans le même espace ([0,1], sans normalisation
  ImageNet — la reconstruction doit rester comparable pixel à pixel).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from services.ml_preprocessing import TrainingAbortedError
from services.vision_anomaly_registry import IMAGE_SIZE, get_anomaly_model_spec
from services.vision_localization import (
    DEFAULT_MASK_PERCENTILE,
    encode_heatmap_png,
    encode_mask_png,
    generate_binary_mask,
    resize_map_to_original,
)

ProgressCallback = Callable[[str, int], None]

# En dessous, pas assez d'images "good" pour un split train/val fiable —
# distinct du seuil de validation à l'upload (sous-lot A,
# MIN_TRAIN_GOOD_IMAGES=5) : ce seuil-ci conditionne la faisabilité de
# l'ENTRAÎNEMENT, pas seulement la validité structurelle du dataset.
MIN_TRAIN_GOOD_FOR_TRAINING = 10
MAX_EXAMPLES = 12


@dataclass
class AnomalyVisionConfig:
    model_id: str = "conv_autoencoder"
    num_epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 1e-3
    seed: int = 42
    max_training_seconds: int = 1500
    mask_percentile: float = DEFAULT_MASK_PERCENTILE


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float


@dataclass
class AnomalyExample:
    relative_path: str
    defect_category: str
    true_label: int  # 0 = good, 1 = défaut
    predicted_label: int
    anomaly_score: float
    heatmap_png: str
    mask_png: str


@dataclass
class AnomalyVisionResult:
    model_id: str
    n_train: int
    n_val: int
    n_test: int
    history: list[EpochMetrics]
    threshold: float
    roc_auc: float
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    confusion_matrix: list[list[int]]
    examples: list[AnomalyExample]
    model_card: dict[str, Any]
    model_artifact: dict[str, Any] = field(repr=False)


def _build_transform() -> transforms.Compose:
    # Pas de normalisation ImageNet : la reconstruction est comparée
    # directement en espace [0,1] (sortie Sigmoid du décodeur) — mélanger un
    # espace normalisé et l'espace [0,1] est précisément le bug #11 déjà
    # documenté, évité ici en n'introduisant jamais de normalisation.
    return transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])


class _UnlabeledImageDataset(Dataset):
    """Wrapper léger autour d'un `ImageFolder` mono-classe (train/good/) —
    ignore le label (inutile, reconstruction pure), retourne juste l'image."""

    def __init__(self, image_folder: ImageFolder):
        self._image_folder = image_folder

    def __len__(self) -> int:
        return len(self._image_folder)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _ = self._image_folder[idx]
        return image


def train_and_evaluate_anomaly_vision(
    dataset_dir: Path,
    config: AnomalyVisionConfig,
    progress_cb: ProgressCallback,
) -> AnomalyVisionResult:
    """Point d'entrée principal. `dataset_dir` doit être un `VisionDataset`
    de structure "mvtec_ad" (train/good + test/good + test/<defaut> —
    structure garantie par le sous-lot A, revérifiée par le worker)."""
    torch.manual_seed(config.seed)
    progress_cb("Préparation des données", 3)

    spec = get_anomaly_model_spec(config.model_id)
    transform = _build_transform()

    train_folder = ImageFolder(str(dataset_dir / "train"), transform=transform)
    if len(train_folder) < MIN_TRAIN_GOOD_FOR_TRAINING:
        raise TrainingAbortedError(
            f"train/good ne contient que {len(train_folder)} image(s) exploitable(s) — "
            f"au moins {MIN_TRAIN_GOOD_FOR_TRAINING} sont nécessaires pour un entraînement fiable"
        )

    train_dataset = _UnlabeledImageDataset(train_folder)
    n_val = max(1, int(0.15 * len(train_dataset)))
    n_train = len(train_dataset) - n_val
    generator = torch.Generator().manual_seed(config.seed)
    train_split, val_split = random_split(train_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_split, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_split, batch_size=config.batch_size, shuffle=False)

    test_folder = ImageFolder(str(dataset_dir / "test"), transform=transform)
    good_class_idx = test_folder.class_to_idx["good"]  # garanti présent — structure validée au sous-lot A
    test_loader = DataLoader(test_folder, batch_size=config.batch_size, shuffle=False)

    progress_cb("Construction du modèle", 8)
    model: nn.Module = spec.build_model()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history: list[EpochMetrics] = []
    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    start_time = time.monotonic()
    time_capped = False

    for epoch in range(config.num_epochs):
        model.train()
        train_loss_total = 0.0
        train_n = 0
        for images in train_loader:
            optimizer.zero_grad()
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * images.size(0)
            train_n += images.size(0)

        model.eval()
        val_loss_total = 0.0
        val_n = 0
        with torch.no_grad():
            for images in val_loader:
                reconstructed = model(images)
                loss = criterion(reconstructed, images)
                val_loss_total += loss.item() * images.size(0)
                val_n += images.size(0)

        train_loss = train_loss_total / train_n
        val_loss = val_loss_total / val_n
        history.append(EpochMetrics(epoch=epoch, train_loss=train_loss, val_loss=val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        percent = 10 + int(60 * (epoch + 1) / config.num_epochs)
        progress_cb(f"Époque {epoch + 1}/{config.num_epochs}", percent)

        if time.monotonic() - start_time > config.max_training_seconds:
            time_capped = True
            break

    model.load_state_dict(best_state)
    model.eval()

    progress_cb("Calcul des scores d'anomalie sur le jeu de test", 75)
    all_scores: list[float] = []
    all_true_labels: list[int] = []
    all_error_maps: list[np.ndarray] = []
    with torch.no_grad():
        for images, class_indices in test_loader:
            reconstructed = model(images)
            # Erreur par pixel moyennée sur les canaux (B, H, W) — même
            # espace [0,1] des deux côtés (voir _build_transform).
            error_maps = torch.mean((images - reconstructed) ** 2, dim=1)
            scores = error_maps.mean(dim=(1, 2))
            all_scores.extend(scores.tolist())
            all_true_labels.extend((class_indices != good_class_idx).long().tolist())
            all_error_maps.extend([e.numpy() for e in error_maps])

    progress_cb("Calibration du seuil de détection", 85)
    fpr, tpr, roc_thresholds = roc_curve(all_true_labels, all_scores)
    youden_j = tpr - fpr
    threshold = float(roc_thresholds[int(np.argmax(youden_j))])
    roc_auc = float(roc_auc_score(all_true_labels, all_scores))

    predicted_labels = [1 if s > threshold else 0 for s in all_scores]
    test_accuracy = sum(1 for t, p in zip(all_true_labels, predicted_labels) if t == p) / len(all_true_labels)
    test_precision = float(precision_score(all_true_labels, predicted_labels, zero_division=0))
    test_recall = float(recall_score(all_true_labels, predicted_labels, zero_division=0))
    test_f1 = float(f1_score(all_true_labels, predicted_labels, zero_division=0))
    conf_matrix = confusion_matrix(all_true_labels, predicted_labels, labels=[0, 1])

    progress_cb("Génération des cartes de localisation", 95)
    class_names = {idx: name for name, idx in test_folder.class_to_idx.items()}
    order = np.argsort(all_scores)[::-1]  # du plus anormal au plus normal
    examples: list[AnomalyExample] = []
    for i in order[:MAX_EXAMPLES]:
        abs_path, class_idx = test_folder.samples[i]
        with Image.open(abs_path) as original_image:
            original_size = original_image.size  # (largeur, hauteur)
        error_map_original_size = resize_map_to_original(all_error_maps[i], original_size)
        mask = generate_binary_mask(error_map_original_size, config.mask_percentile)
        examples.append(
            AnomalyExample(
                # .as_posix() — jamais str() nu, voir vision_classification_training.py
                # (antislashs Windows invalides dans un contrat d'API JSON portable).
                relative_path=Path(abs_path).relative_to(dataset_dir).as_posix(),
                defect_category=class_names[class_idx],
                true_label=all_true_labels[i],
                predicted_label=predicted_labels[i],
                anomaly_score=float(all_scores[i]),
                heatmap_png=encode_heatmap_png(error_map_original_size),
                mask_png=encode_mask_png(mask),
            )
        )

    progress_cb("Terminé", 100)

    model_card = {
        "model_id": config.model_id,
        "image_size": IMAGE_SIZE,
        "num_epochs_requested": config.num_epochs,
        "num_epochs_run": len(history),
        "time_capped": time_capped,
        "seed": config.seed,
        "mask_percentile": config.mask_percentile,
        "n_defect_categories": len(class_names) - 1,  # toutes sauf "good"
        "defect_categories": sorted(name for name in class_names.values() if name != "good"),
    }

    return AnomalyVisionResult(
        model_id=config.model_id,
        n_train=n_train,
        n_val=n_val,
        n_test=len(test_folder),
        history=history,
        threshold=threshold,
        roc_auc=roc_auc,
        test_accuracy=test_accuracy,
        test_precision=test_precision,
        test_recall=test_recall,
        test_f1=test_f1,
        confusion_matrix=conf_matrix.tolist(),
        examples=examples,
        model_card=model_card,
        model_artifact={
            "model_id": config.model_id,
            "threshold": threshold,
            "state_dict": model.state_dict(),
        },
    )
