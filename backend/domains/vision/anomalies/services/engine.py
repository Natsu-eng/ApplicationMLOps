"""Moteur d'entraînement — détection d'anomalies visuelles (structure normal/défaut, pilier
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
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from domains.shared.ml_preprocessing import TrainingAbortedError
from domains.vision.anomalies.services.registry import ANOMALY_MODEL_REGISTRY, IMAGE_SIZE, get_anomaly_model_spec
from domains.vision.shared import AUGMENTATION_PRESET_IDS, augmentation_transforms
from domains.vision.localization import (
    DEFAULT_MASK_PERCENTILE,
    encode_mask_png,
    generate_binary_mask,
    overlay_heatmap_on_image,
    resize_map_to_original,
)

ProgressCallback = Callable[[str, int], None]

# En dessous, pas assez d'images "good" pour un split train/val fiable —
# distinct du seuil de validation à l'upload (sous-lot A,
# MIN_TRAIN_GOOD_IMAGES=5) : ce seuil-ci conditionne la faisabilité de
# l'ENTRAÎNEMENT, pas seulement la validité structurelle du dataset.
MIN_TRAIN_GOOD_FOR_TRAINING = 10
MAX_EXAMPLES = 12

# Correctif C2 (AUDIT_DATALAB_2026-08-16.md) — en dessous de ce nombre
# d'images PAR CATÉGORIE (good ou un défaut nommé), un découpage stratifié
# calibration/évaluation à 50/50 n'a plus de sens statistique (moins de 3
# points de chaque côté). 6 = 2 x 3 : 3 est déjà un plancher bas, mais c'est
# la limite en dessous de laquelle un seuil ou une métrique par catégorie
# devient un artefact de hasard plutôt qu'une mesure. Repli explicite sur
# l'ancien comportement (biaisé, signalé dans model_card) en dessous de ce
# seuil — voir DECISIONS.md D0.4. Ce repli n'est PAS un cas rare : plusieurs
# catégories MVTec AD officielles ont des sous-types de défaut à moins de
# 10 images de test, donc en dessous de ce plancher une fois divisées par 2.
MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT = 6

# Mode expert (retour utilisateur direct : parité avec le comparatif
# multi-modèles du ML tabulaire et le comparatif multi-backbones de la
# classification) — borne au nombre d'architectures réellement disponibles
# dans le registre (3 aujourd'hui) : comparer un id en double n'aurait aucun
# sens, jamais une constante arbitraire déconnectée du registre.
MAX_MODELS_PER_COMPARISON = len(ANOMALY_MODEL_REGISTRY)


@dataclass
class AnomalyVisionConfig:
    model_id: str = "conv_autoencoder"
    num_epochs: int = 15
    batch_size: int = 16
    learning_rate: float = 1e-3
    # Régularisation L2 (mode expert) — voir
    # vision/classification/services/engine.py::ClassificationConfig.weight_decay,
    # même levier, même raisonnement. 0.0 = comportement historique inchangé.
    weight_decay: float = 0.0
    seed: int = 42
    max_training_seconds: int = 1500
    mask_percentile: float = DEFAULT_MASK_PERCENTILE
    # Même système de presets que la classification (voir
    # vision_classification_training.py::AUGMENTATION_PRESET_IDS) —
    # appliqué UNIQUEMENT à train/good/, jamais à test/ (même règle que la
    # classification : l'évaluation doit toujours voir les images réelles).
    # "aucune" par défaut (comportement historique, inchangé) : contrairement
    # à la classification, l'augmentation ici est nouvelle, pas déjà en
    # usage — un défaut inchangé n'a de sens que pour un réglage préexistant.
    augmentation_preset: str = "aucune"
    # Part de train/good/ réservée à la validation (arrêt sur la meilleure
    # époque) — 0.15 reproduit exactement le comportement historique (valeur
    # fixe avant ce correctif).
    val_ratio: float = 0.15


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
    # Correctif C2 — n_test reste la taille totale de test/ (inchangé,
    # rétrocompatibilité par absence pour les enregistrements existants).
    # n_calibration/n_evaluation précisent comment ce total a été utilisé :
    # deux sous-ensembles disjoints (découpage honnête) ou le même ensemble
    # réutilisé deux fois (repli biaisé, voir model_card["threshold_calibration_status"]).
    n_calibration: int
    n_evaluation: int
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
    # Retour utilisateur : "d'autres fonctionnalités modernes que les autres
    # plateformes n'offrent pas" — parité EXACTE de forme avec la
    # classification (`ClassificationResult.roc_curves`/`pr_curves`, une
    # seule clé "Défaut" ici — classe positive binaire) pour réutiliser
    # `EvaluationCharts.tsx` tel quel côté frontend, jamais un second
    # composant de graphique ROC/PR à maintenir en parallèle. + 2
    # diagnostics propres aux anomalies (séparabilité des scores, détection
    # par catégorie de défaut). Calculés sur l'ÉVALUATION uniquement, comme
    # roc_auc/accuracy ci-dessus.
    roc_curves: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    pr_curves: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    score_histogram: dict[str, Any] = field(default_factory=dict)
    category_breakdown: list[dict[str, Any]] = field(default_factory=list)


def _build_transform(augmentation_preset: str = "aucune") -> transforms.Compose:
    # Pas de normalisation ImageNet : la reconstruction est comparée
    # directement en espace [0,1] (sortie Sigmoid du décodeur) — mélanger un
    # espace normalisé et l'espace [0,1] est précisément le bug #11 déjà
    # documenté, évité ici en n'introduisant jamais de normalisation.
    # `augmentation_preset` réutilise domains.vision.shared
    # (même presets, mêmes noms) — n'a de sens que sur train/good/, jamais
    # sur test/ (voir appels ci-dessous).
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        *augmentation_transforms(augmentation_preset),
        transforms.ToTensor(),
    ])


def _split_calibration_evaluation(
    categories: list[str], seed: int
) -> tuple[list[int], list[int], bool]:
    """Découpe les indices de test/ en calibration (sert à choisir le seuil
    par J de Youden) et évaluation (sert à calculer les métriques
    rapportées) — correctif C2 (AUDIT_DATALAB_2026-08-16.md) : avant, le
    seuil était calibré ET évalué sur le même jeu, biais optimiste
    systématique sur toutes les métriques ponctuelles (accuracy, precision,
    recall, f1 — `roc_auc` restait valide, indépendant du seuil, mais est
    désormais lui aussi calculé sur l'évaluation seule pour que tous les
    chiffres rapportés viennent du même sous-ensemble non vu).

    Stratifié par catégorie ("good" compris) à 50/50 pour que chaque
    catégorie soit représentée des deux côtés avec au moins
    MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT // 2 images — un 50/50
    évite toute ambiguïté d'arrondi (contrairement à un split asymétrique,
    où une petite catégorie pourrait tomber sous le plancher par arrondi
    même après le filtre ci-dessous).

    Repli explicite si une catégorie a moins de
    MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT images : retourne les
    MÊMES indices des deux côtés (`biased=True`) plutôt qu'un découpage
    déséquilibré silencieux — un seuil calibré sur 1 ou 2 images n'aurait
    aucun sens. Le repli doit être irréprochable, pas un cas dégradé
    secondaire : plusieurs catégories MVTec AD officielles ont moins de 10
    images de test par sous-type de défaut (voir DECISIONS.md D0.4)."""
    counts = Counter(categories)
    if min(counts.values()) < MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT:
        all_idx = list(range(len(categories)))
        return all_idx, all_idx, True

    indices = list(range(len(categories)))
    calibration_idx, evaluation_idx = train_test_split(
        indices, test_size=0.5, stratify=categories, random_state=seed
    )
    return sorted(calibration_idx), sorted(evaluation_idx), False


def _downsample_curve(x, y, max_points: int = 100) -> tuple[list[float], list[float]]:
    """Même fonction que `ml_training.py`/`vision_classification_training.py::
    _downsample_curve` (dupliquée plutôt qu'importée — modules d'entraînement
    volontairement sans dépendance croisée) : la courbe ROC de sklearn a
    autant de points que d'échantillons de test, inutile d'en envoyer des
    milliers pour un graphe qui en affiche une centaine."""
    if len(x) <= max_points:
        return [float(v) for v in x], [float(v) for v in y]
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return [float(x[i]) for i in idx], [float(y[i]) for i in idx]


def _compute_score_histogram(scores: list[float], labels: list[int], n_bins: int = 20) -> dict[str, Any]:
    """Distribution des scores d'anomalie, séparée par classe réelle (retour
    utilisateur : "d'autres fonctionnalités modernes que les autres
    plateformes n'offrent pas") — montre VISUELLEMENT si les images
    normales et défectueuses forment deux populations bien séparées, et où
    se situe le seuil retenu par rapport à elles. Toujours calculé sur
    l'ÉVALUATION (même sous-ensemble que roc_auc/accuracy ci-dessus),
    jamais la calibration.

    Mêmes bornes (min/max) pour les deux histogrammes — sinon les barres de
    `normal_counts`/`defect_counts` ne seraient pas comparables sur le même
    axe côté frontend."""
    scores_arr = np.asarray(scores)
    lo, hi = float(scores_arr.min()), float(scores_arr.max())
    if hi <= lo:  # tous les scores identiques (cas dégénéré) — pas d'histogramme, pas de division par zéro
        return {"bin_edges": [], "normal_counts": [], "defect_counts": []}
    edges = np.linspace(lo, hi, n_bins + 1)
    normal_scores = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    defect_scores = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    normal_counts, _ = np.histogram(normal_scores, bins=edges)
    defect_counts, _ = np.histogram(defect_scores, bins=edges)
    return {
        "bin_edges": edges.tolist(),
        "normal_counts": normal_counts.tolist(),
        "defect_counts": defect_counts.tolist(),
    }


def _compute_category_breakdown(
    categories: list[str], true_labels: list[int], predicted_labels: list[int]
) -> list[dict[str, Any]]:
    """Taux de détection PAR CATÉGORIE (retour utilisateur : rendre l'onglet
    anomalies aussi riche/transparent que la classification) — calculé sur
    la TOTALITÉ de l'évaluation, pas seulement les `MAX_EXAMPLES` exemples
    affichés dans l'onglet "Exemples". Un dataset multi-défauts (MVTec AD)
    peut très bien afficher une bonne exactitude globale tout en ratant
    systématiquement UN type de défaut précis — invisible dans la seule
    métrique agrégée déjà affichée.

    La catégorie "good" (label toujours 0) donne la spécificité du modèle ;
    chaque catégorie de défaut (label toujours 1) donne son rappel propre —
    même calcul dans les deux cas (part des prédictions correctes), le
    label étant constant au sein d'une catégorie."""
    by_category: dict[str, list[bool]] = {}
    for category, true_label, predicted_label in zip(categories, true_labels, predicted_labels, strict=True):
        by_category.setdefault(category, []).append(true_label == predicted_label)
    return [
        {"category": category, "n": len(outcomes), "detection_rate": sum(outcomes) / len(outcomes)}
        for category, outcomes in sorted(by_category.items())
    ]


class _UnlabeledImageDataset(Dataset):
    """Wrapper léger autour d'un `ImageFolder`/`Subset` mono-classe
    (train/good/) — ignore le label (inutile, reconstruction pure), retourne
    juste l'image."""

    def __init__(self, image_dataset: ImageFolder | Subset):
        self._image_dataset = image_dataset

    def __len__(self) -> int:
        return len(self._image_dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image, _ = self._image_dataset[idx]
        return image


def train_and_evaluate_anomaly_vision(
    dataset_dir: Path,
    config: AnomalyVisionConfig,
    progress_cb: ProgressCallback,
) -> AnomalyVisionResult:
    """Point d'entrée principal. `dataset_dir` doit être un `VisionDataset`
    de structure "mvtec_ad" (train/good + test/good + test/<defaut> —
    structure garantie par le sous-lot A, revérifiée par le worker)."""
    if config.augmentation_preset not in AUGMENTATION_PRESET_IDS:
        raise TrainingAbortedError(
            f"Preset d'augmentation inconnu : {config.augmentation_preset!r} "
            f"(attendu parmi {', '.join(AUGMENTATION_PRESET_IDS)})"
        )
    if not 0.05 <= config.val_ratio <= 0.5:
        raise TrainingAbortedError("La part de validation doit être comprise entre 5 % et 50 % de train/good/")

    torch.manual_seed(config.seed)
    progress_cb("Préparation des données", 3)

    spec = get_anomaly_model_spec(config.model_id)
    train_transform = _build_transform(config.augmentation_preset)
    eval_transform = _build_transform("aucune")

    probe = ImageFolder(str(dataset_dir / "train"), transform=eval_transform)
    if len(probe) < MIN_TRAIN_GOOD_FOR_TRAINING:
        raise TrainingAbortedError(
            f"train/good ne contient que {len(probe)} image(s) exploitable(s) — "
            f"au moins {MIN_TRAIN_GOOD_FOR_TRAINING} sont nécessaires pour un entraînement fiable"
        )

    # Deux instances ImageFolder distinctes sur le MÊME dossier (transforms
    # différents) plutôt qu'un seul dataset partagé — l'augmentation ne doit
    # jamais s'appliquer à la validation (même règle que la classification,
    # voir vision_classification_training.py). Indices calculés UNE fois sur
    # `probe`, réutilisés pour les deux `Subset` : mêmes images des deux
    # côtés, seule la transformation diffère.
    indices = list(range(len(probe)))
    train_idx, val_idx = train_test_split(indices, test_size=config.val_ratio, random_state=config.seed)
    n_train, n_val = len(train_idx), len(val_idx)

    train_folder_aug = ImageFolder(str(dataset_dir / "train"), transform=train_transform)
    val_folder_eval = ImageFolder(str(dataset_dir / "train"), transform=eval_transform)
    train_dataset = _UnlabeledImageDataset(Subset(train_folder_aug, train_idx))
    val_dataset = _UnlabeledImageDataset(Subset(val_folder_eval, val_idx))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    test_folder = ImageFolder(str(dataset_dir / "test"), transform=eval_transform)
    good_class_idx = test_folder.class_to_idx["good"]  # garanti présent — structure validée au sous-lot A
    test_loader = DataLoader(test_folder, batch_size=config.batch_size, shuffle=False)

    progress_cb("Construction du modèle", 8)
    model: nn.Module = spec.build_model()
    criterion = nn.MSELoss()
    # VAE (voir vision_anomaly_registry.py::ConvVAE) : la reconstruction
    # seule ne suffit pas à entraîner mu/logvar, `model.compute_loss` ajoute
    # le terme KL. Les autres architectures (dont le débruiteur, qui ne
    # change QUE le forward, pas la loss) restent sur MSELoss standard.
    use_custom_loss = spec.loss_kind == "vae"
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

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
            loss = model.compute_loss(images, reconstructed) if use_custom_loss else criterion(reconstructed, images)
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
                # Éval toujours en MSE pur, même pour le VAE : `val_loss`
                # sert à choisir la meilleure époque par fidélité de
                # reconstruction, jamais par le terme KL (régularisation,
                # pas un signal de qualité de reconstruction).
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

    class_names = {idx: name for name, idx in test_folder.class_to_idx.items()}
    # Aligné index à index avec all_scores/all_true_labels/all_error_maps —
    # test_loader itère dans l'ordre du dataset (shuffle=False), donc
    # test_folder.samples[i] correspond bien au i-ème score.
    all_categories = [class_names[class_idx] for _, class_idx in test_folder.samples]

    progress_cb("Calibration du seuil de détection", 85)
    # Correctif C2 — le seuil est calibré sur la calibration UNIQUEMENT ;
    # toutes les métriques rapportées (roc_auc compris) viennent de
    # l'évaluation UNIQUEMENT, jamais du même sous-ensemble que la
    # calibration (sauf repli explicite, signalé ci-dessous).
    calibration_idx, evaluation_idx, threshold_calibration_biased = _split_calibration_evaluation(
        all_categories, config.seed
    )

    calibration_scores = [all_scores[i] for i in calibration_idx]
    calibration_labels = [all_true_labels[i] for i in calibration_idx]
    fpr, tpr, roc_thresholds = roc_curve(calibration_labels, calibration_scores)
    youden_j = tpr - fpr
    threshold = float(roc_thresholds[int(np.argmax(youden_j))])

    evaluation_scores = [all_scores[i] for i in evaluation_idx]
    evaluation_labels = [all_true_labels[i] for i in evaluation_idx]
    evaluation_categories = [all_categories[i] for i in evaluation_idx]
    roc_auc = float(roc_auc_score(evaluation_labels, evaluation_scores))
    evaluation_predicted_labels = [1 if s > threshold else 0 for s in evaluation_scores]
    test_accuracy = sum(1 for t, p in zip(evaluation_labels, evaluation_predicted_labels) if t == p) / len(
        evaluation_labels
    )
    test_precision = float(precision_score(evaluation_labels, evaluation_predicted_labels, zero_division=0))
    test_recall = float(recall_score(evaluation_labels, evaluation_predicted_labels, zero_division=0))
    test_f1 = float(f1_score(evaluation_labels, evaluation_predicted_labels, zero_division=0))

    # 4 diagnostics supplémentaires (retour utilisateur : "d'autres
    # fonctionnalités modernes...") — courbes ROC/PR sur l'ÉVALUATION
    # (distinctes de fpr/tpr ci-dessus, calculées sur la CALIBRATION pour
    # choisir le seuil — jamais mélangées) + distribution des scores +
    # détection par catégorie, tous calculés sur l'évaluation uniquement.
    # "Défaut" = clé unique (classe positive) — même convention que le
    # binaire côté classification (`class_names[1]`), pour réutiliser
    # `EvaluationCharts.tsx` sans adaptation.
    eval_fpr, eval_tpr, _ = roc_curve(evaluation_labels, evaluation_scores)
    eval_fpr_s, eval_tpr_s = _downsample_curve(eval_fpr, eval_tpr)
    eval_precision, eval_recall, _ = precision_recall_curve(evaluation_labels, evaluation_scores)
    eval_precision_s, eval_recall_s = _downsample_curve(eval_precision, eval_recall)
    result_roc_curves = {"Défaut": {"fpr": eval_fpr_s, "tpr": eval_tpr_s}}
    result_pr_curves = {"Défaut": {"precision": eval_precision_s, "recall": eval_recall_s}}
    score_histogram = _compute_score_histogram(evaluation_scores, evaluation_labels)
    category_breakdown = _compute_category_breakdown(
        evaluation_categories, evaluation_labels, evaluation_predicted_labels
    )
    conf_matrix = confusion_matrix(evaluation_labels, evaluation_predicted_labels, labels=[0, 1])

    progress_cb("Génération des cartes de localisation", 95)
    # Les exemples présentés dans l'UI viennent TOUJOURS de l'évaluation,
    # jamais de la calibration — une image ayant servi à fixer le seuil ne
    # doit jamais être présentée comme une prédiction sur donnée non vue.
    # En repli biaisé, evaluation_idx == tous les indices (voir
    # _split_calibration_evaluation) : l'honnêteté vient alors du drapeau
    # model_card ci-dessous, pas d'une restriction supplémentaire ici.
    order = sorted(evaluation_idx, key=lambda i: all_scores[i], reverse=True)
    examples: list[AnomalyExample] = []
    for i in order[:MAX_EXAMPLES]:
        abs_path, class_idx = test_folder.samples[i]
        with Image.open(abs_path) as raw_image:
            # .convert("RGB") décode entièrement l'image en mémoire — reste
            # valide après la fermeture du fichier, contrairement à
            # `raw_image` (nécessaire pour la superposition ci-dessous, pas
            # seulement `.size` comme avant le Lot 16A).
            original_image = raw_image.convert("RGB")
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
                predicted_label=1 if all_scores[i] > threshold else 0,
                anomaly_score=float(all_scores[i]),
                # Superposition (Lot 16A) — remplace la heatmap seule,
                # directement lisible sur l'image source (zones rouges =
                # plus forte contribution à l'anomalie détectée).
                heatmap_png=overlay_heatmap_on_image(original_image, error_map_original_size),
                mask_png=encode_mask_png(mask),
            )
        )

    progress_cb("Terminé", 100)

    model_card: dict[str, Any] = {
        "model_id": config.model_id,
        "image_size": IMAGE_SIZE,
        "num_epochs_requested": config.num_epochs,
        "num_epochs_run": len(history),
        "time_capped": time_capped,
        "seed": config.seed,
        "mask_percentile": config.mask_percentile,
        "weight_decay": config.weight_decay,
        "augmentation_preset": config.augmentation_preset,
        "val_ratio": config.val_ratio,
        "n_defect_categories": len(class_names) - 1,  # toutes sauf "good"
        "defect_categories": sorted(name for name in class_names.values() if name != "good"),
        # Correctif C2 — motif dégradation honnête déjà en usage ailleurs
        # dans le projet (explainability_status, calibration_status...).
        "threshold_calibration_status": "degraded" if threshold_calibration_biased else "ok",
        "threshold_calibration_message": (
            "Le jeu de test est trop petit pour un découpage calibration/évaluation fiable par "
            "catégorie (au moins "
            f"{MIN_IMAGES_PER_CATEGORY_FOR_CALIBRATION_SPLIT} images par catégorie sont nécessaires) : "
            "le seuil de détection a été calibré sur les mêmes images que celles utilisées pour les "
            "métriques ci-dessus, qui sont donc optimistes. Importez davantage d'images de test pour "
            "un diagnostic non biaisé."
        )
        if threshold_calibration_biased
        else None,
        "calibration_category_counts": dict(Counter(all_categories[i] for i in calibration_idx)),
        "evaluation_category_counts": dict(Counter(all_categories[i] for i in evaluation_idx)),
    }

    return AnomalyVisionResult(
        model_id=config.model_id,
        n_train=n_train,
        n_val=n_val,
        n_test=len(test_folder),
        n_calibration=len(calibration_idx),
        n_evaluation=len(evaluation_idx),
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
        roc_curves=result_roc_curves,
        pr_curves=result_pr_curves,
        score_histogram=score_histogram,
        category_breakdown=category_breakdown,
    )


def train_and_compare_anomaly_models(
    dataset_dir: Path,
    base_config: AnomalyVisionConfig,
    model_ids: list[str],
    progress_cb: ProgressCallback,
) -> AnomalyVisionResult:
    """Mode expert — comparatif automatique de plusieurs architectures
    (autoencodeur convolutif / débruiteur / variationnel), même principe que
    `vision/classification/services/engine.py::train_and_compare_backbones`
    (voir sa docstring pour le raisonnement complet, identique ici).

    Sélection par min(val_loss) sur l'historique de chaque candidat — jamais
    le ROC-AUC ni les métriques de test, qui dépendent du SEUIL calibré sur
    `test/` (fuite si utilisé pour choisir ENTRE modèles, pas seulement pour
    évaluer le modèle déjà choisi). `val_loss` est toujours une MSE de
    reconstruction pure, y compris pour le VAE (voir plus haut, la boucle
    d'entraînement) : les 3 architectures restent directement comparables sur
    ce même critère, aucun besoin de normaliser entre elles."""
    if len(model_ids) < 2:
        raise TrainingAbortedError("La comparaison de modèles nécessite au moins 2 architectures")
    if len(model_ids) > MAX_MODELS_PER_COMPARISON:
        raise TrainingAbortedError(f"Au plus {MAX_MODELS_PER_COMPARISON} architectures comparables par entraînement")
    if len(set(model_ids)) != len(model_ids):
        raise TrainingAbortedError("Architectures en double dans le comparatif")

    n = len(model_ids)
    results: dict[str, AnomalyVisionResult] = {}
    candidates: list[dict[str, Any]] = []

    for i, model_id in enumerate(model_ids):
        spec = get_anomaly_model_spec(model_id)  # lève si id inconnu — jamais découvert à mi-job

        def scoped_progress_cb(step: str, percent: int, _i: int = i, _label: str = spec.label) -> None:
            overall = int((_i * 100 + percent) / n)
            progress_cb(f"[{_i + 1}/{n}] {_label} — {step}", min(overall, 99))

        candidate_config = replace(base_config, model_id=model_id)
        t0 = time.monotonic()
        result = train_and_evaluate_anomaly_vision(dataset_dir, candidate_config, scoped_progress_cb)
        elapsed_seconds = time.monotonic() - t0
        results[model_id] = result

        best_val_loss = min((h.val_loss for h in result.history), default=float("inf"))
        candidates.append({
            "model_id": model_id,
            "model_label": spec.label,
            "best_val_loss": best_val_loss,
            "roc_auc": result.roc_auc,
            "test_accuracy": result.test_accuracy,
            "num_epochs_run": len(result.history),
            "time_capped": bool(result.model_card.get("time_capped")),
            "training_seconds": round(elapsed_seconds, 1),
        })

    winner = min(candidates, key=lambda c: c["best_val_loss"])
    winner_id = winner["model_id"]
    for c in candidates:
        c["selected"] = c["model_id"] == winner_id

    progress_cb("Terminé", 100)

    winner_result = results[winner_id]
    winner_result.model_card["candidates"] = candidates
    winner_result.model_card["comparison_mode"] = True
    return winner_result
