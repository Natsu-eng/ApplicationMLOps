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

# Lot 6A (correctif I9, AUDIT_DATALAB_2026-08-16.md §I9) — jusqu'ici une
# seule augmentation FIGÉE (flip + rotation 10° + jitter léger),
# l'utilisateur ne contrôlait rien. 4 presets, du plus faible au plus fort
# — jamais de valeurs choisies au hasard : chaque niveau ajoute une
# transformation à celles du niveau précédent, jamais une combinaison
# disjointe (progression cohérente, prévisible pour l'utilisateur).
AUGMENTATION_PRESET_IDS = ("aucune", "legere", "standard", "forte")
DEFAULT_AUGMENTATION_PRESET = "standard"  # comportement historique, inchangé par défaut


def augmentation_transforms(preset: str) -> list:
    if preset == "aucune":
        return []
    if preset == "legere":
        return [transforms.RandomHorizontalFlip()]
    if preset == "standard":
        return [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    if preset == "forte":
        return [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    raise ValueError(f"Preset d'augmentation inconnu : {preset!r} (attendu parmi {AUGMENTATION_PRESET_IDS})")


# Seuils empiriques (pas une science exacte) fondés sur la taille de la
# classe la plus PETITE — le goulot d'étranglement réel pour le risque de
# sur-apprentissage, jamais le total d'images (masquerait un déséquilibre
# sévère : 1000 images dont une classe à 5 reste une classe à 5). Peu
# d'images par classe → sur-apprentissage plus probable → augmentation
# plus forte pour diversifier artificiellement le peu de données
# disponibles ; beaucoup d'images → la variété réelle suffit déjà, une
# augmentation trop forte distordrait inutilement la distribution et
# ralentirait la convergence sans bénéfice.
_RECOMMENDATION_THRESHOLDS = ((20, "forte"), (50, "standard"), (150, "legere"))


def recommend_augmentation_preset(min_class_size: int) -> str:
    """Recommandation fondée sur la taille du dataset (I9) — indicative,
    jamais appliquée automatiquement : l'utilisateur choisit toujours
    explicitement le preset final (voir ClassificationConfig.augmentation_preset)."""
    for threshold, preset in _RECOMMENDATION_THRESHOLDS:
        if min_class_size < threshold:
            return preset
    return "aucune"


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
    # Lot 6A (correctif I8, AUDIT_DATALAB_2026-08-16.md §I8) — jusqu'ici
    # CrossEntropyLoss() sans pondération : sur un dataset déséquilibré (une
    # classe très majoritaire), le modèle apprenait à toujours prédire la
    # classe majoritaire tout en affichant une accuracy trompeusement haute.
    # Pondération par fréquence inverse des classes DU SPLIT D'ENTRAÎNEMENT
    # (jamais validation/test, qui ne doivent influencer ni la perte ni
    # l'optimisation) — voir _class_weights. Généralise à N classes (pas
    # seulement binaire) : un poids par classe, quel que soit leur nombre.
    class_weighting: bool = True
    # Arrête l'entraînement si val_loss ne s'améliore plus depuis ce nombre
    # d'époques consécutives (poids de la meilleure époque déjà conservés,
    # voir best_state ci-dessous — l'early stopping économise seulement du
    # calcul, ne change jamais QUEL modèle est retenu). None = désactivé
    # (comportement historique : toujours num_epochs époques, sous réserve
    # de max_training_seconds).
    early_stopping_patience: Optional[int] = 3
    # ReduceLROnPlateau sur val_loss — réduit le taux d'apprentissage quand
    # la progression stagne, plutôt qu'un taux fixe du début à la fin.
    use_lr_scheduler: bool = True
    # Lot 6A (correctif I9) — voir AUGMENTATION_PRESET_IDS ci-dessus.
    # "standard" reproduit exactement l'augmentation historique (seule
    # option avant ce lot) — défaut inchangé, comportement identique pour
    # quiconque ne choisit pas explicitement un autre preset.
    augmentation_preset: str = DEFAULT_AUGMENTATION_PRESET
    # Répartition personnalisée train/validation/test — 0.15/0.15
    # reproduisent exactement le 70/15/15 historique (seules valeurs
    # possibles avant ce correctif). Validé par `_stratified_split` (doivent
    # rester dans (0, 1) et laisser au moins 10 % pour l'entraînement).
    val_ratio: float = 0.15
    test_ratio: float = 0.15


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


def _build_transforms(augmentation_preset: str) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        *augmentation_transforms(augmentation_preset),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, build_eval_transform()


def _should_stop_early(epochs_without_improvement: int, patience: Optional[int]) -> bool:
    """Décision d'arrêt anticipé — extraite en fonction pure (patience=None
    désactive) pour rester testable sans dépendre de la dynamique réelle
    d'un entraînement (bruit de convergence sur un dataset synthétique
    minuscule, non déterministe d'une exécution à l'autre)."""
    return patience is not None and epochs_without_improvement >= patience


def _class_weights(targets: list[int], train_idx: list[int], n_classes: int) -> torch.Tensor:
    """Poids par fréquence inverse (`total / (n_classes * n_c)`), calculés
    UNIQUEMENT sur le split d'entraînement — jamais validation/test, qui ne
    doivent influencer ni la perte ni l'optimisation. Généralise à N
    classes : un poids par classe, jamais un traitement spécial "binaire
    vs multiclasse" (`CrossEntropyLoss(weight=...)` accepte nativement un
    poids par classe, quel que soit leur nombre)."""
    train_targets = [targets[i] for i in train_idx]
    counts = Counter(train_targets)
    total = len(train_targets)
    weights = [total / (n_classes * counts.get(c, 1)) for c in range(n_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def _stratified_split(
    targets: list[int], seed: int, val_ratio: float = 0.15, test_ratio: float = 0.15
) -> tuple[list[int], list[int], list[int]]:
    """Découpage stratifié train/val/test — `val_ratio`/`test_ratio`
    personnalisables (répartition, Lot 6A) : 0.15/0.15 reproduit exactement
    le 70/15/15 historique (seules valeurs possibles avant ce correctif).
    Toujours 2 appels `train_test_split` en cascade (train d'abord, puis
    val/test au sein du reste) — jamais 3 découpages indépendants, qui ne
    garantiraient pas des ensembles disjoints."""
    holdout_ratio = val_ratio + test_ratio
    indices = list(range(len(targets)))
    train_idx, temp_idx = train_test_split(indices, test_size=holdout_ratio, stratify=targets, random_state=seed)
    temp_targets = [targets[i] for i in temp_idx]
    # Part de test/ AU SEIN du reste (temp_idx) — ex. val=0.15/test=0.15 sur
    # le total donne holdout=0.30, dont la moitié (0.15/0.30=0.5) va aux
    # deux, exactement le repli historique 50/50 dans temp_idx.
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio / holdout_ratio, stratify=temp_targets, random_state=seed
    )
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
    if config.augmentation_preset not in AUGMENTATION_PRESET_IDS:
        raise TrainingAbortedError(
            f"Preset d'augmentation inconnu : {config.augmentation_preset!r} "
            f"(attendu parmi {', '.join(AUGMENTATION_PRESET_IDS)})"
        )
    # Répartition (Lot 6A) — validée ici, jamais au niveau API seul : ce
    # module reste directement testable/appelable sans passer par FastAPI.
    # >= 10 % pour train : en dessous, le modèle n'a plus assez d'exemples
    # pour apprendre quoi que ce soit d'utile, quel que soit le dataset.
    if config.val_ratio <= 0 or config.test_ratio <= 0 or config.val_ratio + config.test_ratio >= 0.9:
        raise TrainingAbortedError(
            "Répartition invalide : validation et test doivent être > 0 % chacun, et laisser au moins "
            "10 % des images pour l'entraînement"
        )
    torch.manual_seed(config.seed)
    progress_cb("Préparation des données", 3)

    spec = get_backbone_spec(config.backbone_id)
    train_transform, eval_transform = _build_transforms(config.augmentation_preset)

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

    train_idx, val_idx, test_idx = _stratified_split(
        probe.targets, config.seed, val_ratio=config.val_ratio, test_ratio=config.test_ratio
    )

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

    if config.class_weighting:
        criterion = nn.CrossEntropyLoss(weight=_class_weights(probe.targets, train_idx, len(class_names)))
    else:
        criterion = nn.CrossEntropyLoss()

    def _make_optimizer() -> torch.optim.Optimizer:
        return torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)

    def _make_scheduler(opt: torch.optim.Optimizer):
        # patience=2 : plus courte que early_stopping_patience (défaut 3) —
        # le taux d'apprentissage doit avoir une chance de se réduire et de
        # relancer la progression AVANT que l'arrêt anticipé n'intervienne,
        # sans quoi le scheduler ne servirait jamais à rien en pratique.
        if not config.use_lr_scheduler:
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2, factor=0.5)

    optimizer = _make_optimizer()
    scheduler = _make_scheduler(optimizer)

    history: list[EpochMetrics] = []
    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    time_capped = False
    early_stopped = False
    epochs_without_improvement = 0
    start_time = time.monotonic()

    for epoch in range(config.num_epochs):
        if config.unfreeze_after_epoch is not None and epoch == config.unfreeze_after_epoch:
            unfreeze_backbone(model)
            optimizer = _make_optimizer()  # nouveaux paramètres entraînables à suivre
            scheduler = _make_scheduler(optimizer)
            # Dégeler change substantiellement la dynamique d'entraînement
            # (beaucoup plus de paramètres optimisables d'un coup) — repartir
            # avec un budget de patience frais évite un arrêt anticipé
            # déclenché par la hausse transitoire de val_loss qui suit
            # souvent un dégel, pas par une réelle stagnation.
            epochs_without_improvement = 0

        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer=None)
        history.append(EpochMetrics(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc,
                                     val_loss=val_loss, val_accuracy=val_acc))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(val_loss)

        percent = 10 + int(75 * (epoch + 1) / config.num_epochs)
        progress_cb(f"Époque {epoch + 1}/{config.num_epochs}", percent)

        if time.monotonic() - start_time > config.max_training_seconds:
            time_capped = True
            break

        if _should_stop_early(epochs_without_improvement, config.early_stopping_patience):
            early_stopped = True
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
        # Lot 6A (correctif I8) — honnêteté sur CE qui a réellement déterminé
        # la fin de l'entraînement, jamais supposé implicitement égal à
        # num_epochs_requested (même principe que time_capped, déjà en
        # place) : early_stopped=True et num_epochs_run < num_epochs_requested
        # signifie "arrêté par manque de progression", pas par le budget de
        # temps (time_capped) ni par épuisement du budget d'époques demandé.
        "early_stopped": early_stopped,
        "class_weighting_applied": config.class_weighting,
        "lr_scheduler_used": config.use_lr_scheduler,
        "augmentation_preset": config.augmentation_preset,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
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
