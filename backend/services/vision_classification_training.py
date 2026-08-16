"""Moteur d'entraînement — classification d'images par transfer learning
(pilier Vision, Lot 15 sous-lot B).

Module séparé, jamais une branche dans le moteur ML tabulaire existant
(même principe que `services/anomaly_training.py`) : aucune notion commune
avec `ml_training.py` (Optuna, `ColumnTransformer`, SHAP n'ont aucun sens
pour des images/PyTorch — diagnostic du chantier Vision).

Contrainte CPU (aucun GPU dans `docker-compose.yml`, un seul worker RQ
partagé) : un budget de temps interne (`max_training_seconds`, largement
sous le timeout RQ de 1800s) arrête proprement l'entraînement entre deux
époques plutôt que de risquer un timeout RQ brutal sur un gros dataset —
le modèle obtenu reste utilisable (poids de la meilleure époque conservés),
`model_card["time_capped"]` le signale honnêtement plutôt que de prétendre
que le nombre d'époques demandé a été respecté.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from services.ml_preprocessing import TrainingAbortedError
from services.vision_classification_registry import (
    DEFAULT_BACKBONE_ID,
    freeze_backbone,
    get_backbone_spec,
    unfreeze_backbone,
)

ProgressCallback = Callable[[str, int], None]

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# En dessous, un split stratifié train/val/test 70/15/15 n'est pas fiable
# (risque de classe absente d'un des trois splits) — vérifié explicitement
# avant tout calcul coûteux, jamais une ValueError sklearn brute remontée
# à l'utilisateur.
MIN_IMAGES_PER_CLASS_FOR_TRAINING = 6
MAX_EXAMPLES_PER_KIND = 12  # exemples corrects/erronés conservés pour l'UI


@dataclass
class ClassificationConfig:
    backbone_id: str = DEFAULT_BACKBONE_ID
    num_epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 1e-3
    dropout_rate: float = 0.3
    freeze_backbone: bool = True
    # Époque (0-indexée) à partir de laquelle le backbone est entièrement
    # dégelé (fine-tuning complet) — None = jamais, comportement par défaut
    # documenté à l'utilisateur (skill "Transfer learning") : le backbone
    # pré-entraîné reste gelé tant que ce n'est pas explicitement demandé.
    unfreeze_after_epoch: Optional[int] = None
    seed: int = 42
    # Garde-fou de temps interne — voir docstring du module.
    max_training_seconds: int = 1500


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


@dataclass
class PredictionExample:
    relative_path: str
    true_label: str
    predicted_label: str
    confidence: float
    correct: bool


@dataclass
class ClassificationResult:
    backbone_id: str
    class_names: list[str]
    n_train: int
    n_val: int
    n_test: int
    history: list[EpochMetrics]
    test_accuracy: float
    test_precision_macro: float
    test_recall_macro: float
    test_f1_macro: float
    confusion_matrix: list[list[int]]
    examples: list[PredictionExample]
    model_card: dict[str, Any]
    model_artifact: dict[str, Any] = field(repr=False)  # à torch.save tel quel


def build_eval_transform() -> transforms.Compose:
    """Transform d'évaluation (pas d'augmentation) — publique et réutilisée
    telle quelle par `services/vision_gradcam.py` (sous-lot D) : Grad-CAM
    doit voir exactement la même normalisation que l'entraînement, jamais
    une transformation reconstruite indépendamment (source d'incohérence)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, build_eval_transform()


def _stratified_split(targets: list[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(len(targets)))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=targets, random_state=seed)
    temp_targets = [targets[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, stratify=temp_targets, random_state=seed)
    return train_idx, val_idx, test_idx


def _run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None) -> tuple[float, float]:
    """Une époque train (si `optimizer` fourni) ou éval — factorisé pour ne
    jamais dupliquer la logique de calcul de perte/accuracy entre les deux."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            if is_training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total_n += images.size(0)
    return total_loss / total_n, total_correct / total_n


def train_and_evaluate_classification(
    dataset_dir: Path,
    config: ClassificationConfig,
    progress_cb: ProgressCallback,
) -> ClassificationResult:
    """Point d'entrée principal. `dataset_dir` doit être un `VisionDataset`
    de structure "classification" (dossiers de classes — voir
    `services/vision_datasets.py`), jamais MVTec AD."""
    torch.manual_seed(config.seed)
    progress_cb("Préparation des données", 3)

    spec = get_backbone_spec(config.backbone_id)
    train_transform, eval_transform = _build_transforms()

    probe = ImageFolder(str(dataset_dir))
    class_names = probe.classes
    if len(class_names) < 2:
        raise TrainingAbortedError("Au moins 2 classes sont nécessaires pour entraîner un classifieur")

    counts = Counter(probe.targets)
    under_min = sorted(
        class_names[c] for c, n in counts.items() if n < MIN_IMAGES_PER_CLASS_FOR_TRAINING
    )
    if under_min:
        raise TrainingAbortedError(
            f"Classe(s) avec moins de {MIN_IMAGES_PER_CLASS_FOR_TRAINING} images (minimum pour un "
            f"split train/validation/test fiable) : {', '.join(under_min)}"
        )

    train_idx, val_idx, test_idx = _stratified_split(probe.targets, config.seed)

    train_dataset = Subset(ImageFolder(str(dataset_dir), transform=train_transform), train_idx)
    eval_base = ImageFolder(str(dataset_dir), transform=eval_transform)
    val_dataset = Subset(eval_base, val_idx)
    test_dataset = Subset(eval_base, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    progress_cb("Construction du modèle", 8)
    model = spec.build_model(len(class_names), config.dropout_rate)
    if config.freeze_backbone:
        freeze_backbone(model, spec)

    criterion = nn.CrossEntropyLoss()

    def _make_optimizer() -> torch.optim.Optimizer:
        return torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)

    optimizer = _make_optimizer()

    history: list[EpochMetrics] = []
    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    time_capped = False
    start_time = time.monotonic()

    for epoch in range(config.num_epochs):
        if config.unfreeze_after_epoch is not None and epoch == config.unfreeze_after_epoch:
            unfreeze_backbone(model)
            optimizer = _make_optimizer()  # nouveaux paramètres entraînables à suivre

        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer=None)
        history.append(EpochMetrics(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc,
                                     val_loss=val_loss, val_accuracy=val_acc))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        percent = 10 + int(75 * (epoch + 1) / config.num_epochs)
        progress_cb(f"Époque {epoch + 1}/{config.num_epochs}", percent)

        if time.monotonic() - start_time > config.max_training_seconds:
            time_capped = True
            break

    model.load_state_dict(best_state)  # poids de la meilleure époque (val_loss mini), pas la dernière

    progress_cb("Évaluation sur le jeu de test", 90)
    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_confidence: list[float] = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)
            all_true.extend(labels.tolist())
            all_pred.extend(preds.tolist())
            all_confidence.extend(confidences.tolist())

    test_accuracy = sum(1 for t, p in zip(all_true, all_pred) if t == p) / len(all_true)
    test_precision = precision_score(all_true, all_pred, average="macro", zero_division=0)
    test_recall = recall_score(all_true, all_pred, average="macro", zero_division=0)
    test_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(all_true, all_pred, labels=list(range(len(class_names))))

    progress_cb("Sélection des exemples", 95)
    examples: list[PredictionExample] = []
    correct_examples: list[PredictionExample] = []
    incorrect_examples: list[PredictionExample] = []
    for position, sample_idx in enumerate(test_idx):
        abs_path, true_idx = eval_base.samples[sample_idx]
        # .as_posix() explicite — jamais str() nu : sur Windows, str() produit
        # des antislashs (ex. "bleu\6.png"), invalides dans un contrat d'API
        # JSON censé être portable (le frontend reconstruit une URL avec ce
        # chemin). Bug réel trouvé en testant en local sous Windows — la
        # cible de déploiement (Docker/Linux) ne l'aurait jamais révélé.
        relative_path = Path(abs_path).relative_to(dataset_dir).as_posix()
        pred_idx = all_pred[position]
        example = PredictionExample(
            relative_path=relative_path,
            true_label=class_names[true_idx],
            predicted_label=class_names[pred_idx],
            confidence=all_confidence[position],
            correct=(true_idx == pred_idx),
        )
        (correct_examples if example.correct else incorrect_examples).append(example)
    # Priorité aux erreurs (skill : "erreurs de classification, pas
    # seulement les succès") — toujours représentées si elles existent,
    # jamais noyées dans les succès par un échantillonnage aveugle.
    examples = incorrect_examples[:MAX_EXAMPLES_PER_KIND] + correct_examples[:MAX_EXAMPLES_PER_KIND]

    progress_cb("Terminé", 100)

    model_card = {
        "backbone_id": config.backbone_id,
        "num_epochs_requested": config.num_epochs,
        "num_epochs_run": len(history),
        "time_capped": time_capped,
        "freeze_backbone": config.freeze_backbone,
        "unfreeze_after_epoch": config.unfreeze_after_epoch,
        "seed": config.seed,
        "n_classes": len(class_names),
        "n_incorrect_examples": len(incorrect_examples),
        "n_correct_examples": len(correct_examples),
    }

    return ClassificationResult(
        backbone_id=config.backbone_id,
        class_names=class_names,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
        history=history,
        test_accuracy=test_accuracy,
        test_precision_macro=float(test_precision),
        test_recall_macro=float(test_recall),
        test_f1_macro=float(test_f1),
        confusion_matrix=conf_matrix.tolist(),
        examples=examples,
        model_card=model_card,
        model_artifact={
            "backbone_id": config.backbone_id,
            "class_names": class_names,
            "dropout_rate": config.dropout_rate,
            "state_dict": model.state_dict(),
        },
    )
