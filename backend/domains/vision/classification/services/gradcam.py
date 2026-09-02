"""Grad-CAM — explicabilité visuelle pour la classification d'images
(pilier Vision, Lot 15 sous-lot D).

Équivalent visuel de SHAP côté ML tabulaire (`services/ml_explainability.py`)
: pourquoi CE modèle a prédit CETTE classe pour CETTE image précise. Le
legacy prévoyait Grad-CAM (mentionné dans la documentation historique) mais
ne l'a jamais branché nulle part — aucune trace dans les fichiers audités
(`docs/legacy/`).

Réutilise intégralement `services/vision_localization.py` (sous-lot C) pour
le réalignement à la taille d'image originale et l'encodage PNG — un seul
mécanisme de superposition heatmap/image dans tout le module vision, jamais
deux implémentations séparées."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from domains.vision.classification.services.engine import IMAGE_SIZE, build_eval_transform
from domains.vision.classification.services.registry import get_backbone_spec
from domains.vision.localization import overlay_heatmap_on_image, resize_map_to_original

logger = logging.getLogger("datalab.vision.gradcam")


class GradCamError(ValueError):
    """Requête d'explication invalide (classe cible inconnue, etc.)."""


@dataclass
class GradCamResult:
    predicted_label: str
    probabilities: dict[str, float]
    target_label: str  # classe réellement expliquée (= predicted_label si non précisée)
    heatmap_png: str
    # Part de l'attention Grad-CAM tombant dans la bordure de l'image, et
    # forme de la carte BRUTE (résolution du dernier bloc convolutif, ex.
    # 7×7) qui l'a produite — voir `compute_border_attention_fraction`
    # ci-dessous. Jamais la carte redimensionnée à la taille de l'image
    # d'origine (coût inutile, la position RELATIVE suffit). Sert
    # uniquement à `synthesize_attention_pattern` (jamais affichée seule,
    # un seul chiffre par image n'a pas de sens isolément) — jamais None :
    # `_run_gradcam` les calcule toujours.
    border_attention_fraction: float
    cam_map_shape: tuple[int, ...]


@dataclass
class GradCamBatchItemResult:
    """Un élément du batch (retour utilisateur direct : "Grad-CAM devrait
    supporter le batch, pas une image à la fois") — `key` identifie l'image
    pour l'appelant (ex. son `relative_path` dans le dataset), `error` porte
    un message actionnable SI cette image précise n'a pas pu être expliquée
    (jamais silencieux) sans faire échouer le reste du batch."""

    key: str
    result: GradCamResult | None
    error: str | None


# ── Synthèse agrégée (constat transversal sur PLUSIEURS images, pas une
# heatmap de plus) — retour d'évaluation d'une maquette externe : "pas juste
# une heatmap par image, une observation transversale sur ce qui cloche"
# (ex. "3 fois sur 4, l'attention porte sur le bord plutôt que la zone
# usinée"). Jamais un LLM qui commenterait les images — un calcul
# géométrique exact sur les cartes Grad-CAM déjà produites (même principe
# que `services/verdict.py` : uniquement des règles déterministes sur des
# nombres déjà calculés, rien d'inventé).
#
# Choix du split "bordure vs centre" plutôt qu'une notion métier ("zone
# usinée", "objet d'intérêt") : générique à N'IMPORTE QUEL dataset d'images
# sans connaissance a priori du sujet (contrairement à une segmentation ou
# une détection d'objet, hors périmètre) — une attention systématiquement
# concentrée sur la bordure plutôt que le sujet central est un signe de
# biais connu et documenté en interprétabilité (le modèle capte le cadrage/
# arrière-plan plutôt que l'objet), pas une garantie absolue : présenté
# comme une OBSERVATION statistique avec ses chiffres, jamais un diagnostic
# certain.
BORDER_RATIO = 0.2
# Épaisseur de la bordure, en fraction de chaque côté de l'image — ex. 0,2
# = les 20 % extérieurs sur chaque axe comptent comme "bordure". Valeur
# fixe documentée (comme `PSI_BINS`/`MIN_CURRENT_ROWS_FOR_DRIFT` dans
# `domains/shared/drift.py`), jamais recalculée par image.

MIN_IMAGES_FOR_SYNTHESIS = 4
# En dessous, "X fois sur Y" n'a pas de sens statistique (voir même
# raisonnement que `MIN_CURRENT_ROWS_FOR_DRIFT`) — `synthesize_attention_
# pattern` renvoie `None` plutôt qu'un constat sur un échantillon trop
# petit pour être autre chose que du bruit.

_BORDER_MASK_CACHE: dict[tuple[int, ...], np.ndarray] = {}
# La forme de `cam_map` est FIXE pour un backbone donné (résolution du
# dernier bloc convolutif, ex. toujours 7×7 pour un backbone à stride 32
# sur une entrée 224×224) — recalculer le même masque booléen à chaque
# image d'un batch de 12 serait un travail identique répété 12 fois pour
# rien ; le cache est mémoire-négligeable (un booléen par cellule de carte,
# jamais par pixel d'image).


def _border_mask(shape: tuple[int, ...], border_ratio: float) -> np.ndarray:
    """Masque booléen (True = cellule de bordure) sur une grille `shape`,
    par POSITION RELATIVE (0..1 sur chaque axe) — indépendant de la
    résolution réelle de `cam_map`, qui diffère selon le backbone."""
    cache_key = shape
    cached = _BORDER_MASK_CACHE.get(cache_key)
    if cached is not None:
        return cached
    h, w = shape
    row_pos = np.linspace(0.0, 1.0, h) if h > 1 else np.array([0.5])
    col_pos = np.linspace(0.0, 1.0, w) if w > 1 else np.array([0.5])
    row_is_border = (row_pos < border_ratio) | (row_pos > 1 - border_ratio)
    col_is_border = (col_pos < border_ratio) | (col_pos > 1 - border_ratio)
    mask = row_is_border[:, None] | col_is_border[None, :]
    _BORDER_MASK_CACHE[cache_key] = mask
    return mask


def compute_border_attention_fraction(cam_map: np.ndarray, border_ratio: float = BORDER_RATIO) -> float:
    """Part de la masse d'attention Grad-CAM (carte déjà post-ReLU, donc
    ≥ 0 partout) tombant dans la bordure de l'image, sur [0, 1]. `cam_map`
    est la carte BRUTE (résolution du dernier bloc convolutif), jamais la
    version redimensionnée — la position relative suffit, redimensionner
    d'abord ne changerait pas le résultat mais coûterait plus cher.

    0.0 si la carte est entièrement nulle (cas dégénéré : gradient nul,
    jamais observé en pratique mais géré explicitement plutôt qu'une
    division par zéro qui remonterait comme une erreur 500 opaque)."""
    total = float(cam_map.sum())
    if total <= 0:
        return 0.0
    mask = _border_mask(cam_map.shape, border_ratio)
    return float(cam_map[mask].sum() / total)


def _area_fraction_border(shape: tuple[int, ...], border_ratio: float = BORDER_RATIO) -> float:
    """Part de la carte occupée par la bordure PAR PURE GÉOMÉTRIE (sans
    aucune attention) — la référence "au hasard" à laquelle comparer
    `compute_border_attention_fraction` : sur une grille grossière (ex.
    7×7), la bordure à 20 % occupe déjà une bonne partie de la carte, une
    comparaison à un seuil fixe (0,5) serait donc trompeuse à résolution
    grossière. Comparer à CETTE valeur exacte, recalculée pour la forme
    réelle de la carte, s'auto-calibre quelle que soit la résolution du
    backbone."""
    mask = _border_mask(shape, border_ratio)
    return float(mask.mean())


@dataclass
class GradCamAttentionSynthesis:
    """Constat agrégé sur un lot d'images déjà expliquées — voir le
    commentaire de section ci-dessus. `observation` est un gabarit rempli
    par les chiffres calculés, jamais du texte libre."""

    n_images: int
    n_border_biased: int
    border_biased_fraction: float
    area_fraction_border: float  # ce que donnerait une attention purement au hasard, pour comparaison
    has_notable_pattern: bool
    observation: str


def synthesize_attention_pattern(items: list[GradCamBatchItemResult]) -> GradCamAttentionSynthesis | None:
    """Agrège `border_attention_fraction` sur les images expliquées AVEC
    SUCCÈS d'un lot (les échecs individuels, déjà signalés par `error`,
    n'ont pas de carte à agréger). `None` si moins de
    `MIN_IMAGES_FOR_SYNTHESIS` images exploitables — jamais un constat sur
    un échantillon trop petit.

    Une image est "en biais de bordure" quand sa part d'attention en
    bordure DÉPASSE ce que donnerait une attention purement uniforme sur
    cette même carte (`_area_fraction_border`) — over-représentée par
    rapport au hasard, pas juste "plus de la moitié" (trompeur sur une
    grille grossière, voir `_area_fraction_border`).

    `has_notable_pattern` = majorité stricte des images en biais de
    bordure — en dessous, l'observation le dit explicitement plutôt que de
    forcer un constat sur un pattern qui n'en est pas un."""
    successful = [item.result for item in items if item.result is not None]
    n_images = len(successful)
    if n_images < MIN_IMAGES_FOR_SYNTHESIS:
        return None

    # `area_fraction_border` est identique pour toutes les images du lot
    # (même backbone pour tout un batch, donc même forme de carte) —
    # calculée une seule fois à partir de la forme réellement observée sur
    # le premier résultat.
    area_fraction = _area_fraction_border(successful[0].cam_map_shape)
    n_border_biased = sum(1 for r in successful if r.border_attention_fraction > area_fraction)
    border_biased_fraction = n_border_biased / n_images
    has_notable_pattern = border_biased_fraction > 0.5

    if has_notable_pattern:
        observation = (
            f"Sur {n_images} images expliquées, {n_border_biased} ({border_biased_fraction * 100:.0f} %) "
            f"montrent une attention concentrée sur la bordure de l'image plutôt que sur le centre — "
            f"davantage que si l'attention était uniformément répartie ({area_fraction * 100:.0f} % "
            f"attendus par pur hasard sur cette carte). Signe possible que le modèle capte le cadrage "
            f"ou l'arrière-plan plutôt que le sujet lui-même."
        )
    else:
        observation = (
            f"Sur {n_images} images expliquées, {n_border_biased} ({border_biased_fraction * 100:.0f} %) "
            f"montrent une attention concentrée sur la bordure plutôt que le centre — pas de biais "
            f"systématique notable sur cet échantillon."
        )

    return GradCamAttentionSynthesis(
        n_images=n_images,
        n_border_biased=n_border_biased,
        border_biased_fraction=border_biased_fraction,
        area_fraction_border=area_fraction,
        has_notable_pattern=has_notable_pattern,
        observation=observation,
    )


class _ActivationGradientCapture:
    """Capture l'activation et le gradient d'UNE couche cible via hooks —
    classe dédiée plutôt que des variables `nonlocal` éparpillées, pour
    rester lisible et testable isolément."""

    def __init__(self, target_layer: nn.Module):
        self.activation: torch.Tensor | None = None
        self.gradient: torch.Tensor | None = None
        target_layer.register_forward_hook(self._on_forward)
        target_layer.register_full_backward_hook(self._on_backward)

    def _on_forward(self, module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
        self.activation = output

    def _on_backward(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        self.gradient = grad_output[0]


def _rebuild_model(artifact: dict[str, Any]) -> nn.Module:
    spec = get_backbone_spec(artifact["backbone_id"])
    model = spec.build_model(len(artifact["class_names"]), artifact["dropout_rate"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model


def _prepare_model_and_capture(artifact: dict[str, Any]) -> tuple[nn.Module, _ActivationGradientCapture, list[str]]:
    """Reconstruction du modèle + enregistrement des hooks Grad-CAM — coût
    dominant de cet endpoint ("le plus coûteux du backend", voir
    `api/routers/vision_classification.py::_explain_rate_limit`). Isolé pour
    n'être payé QU'UNE FOIS par requête, y compris en mode batch (retour
    utilisateur direct : jusqu'ici une image à la fois, chaque appel
    rechargeait le modèle depuis zéro)."""
    class_names: list[str] = artifact["class_names"]
    spec = get_backbone_spec(artifact["backbone_id"])
    model = _rebuild_model(artifact)
    capture = _ActivationGradientCapture(spec.gradcam_target_layer(model))
    return model, capture, class_names


def _run_gradcam(
    model: nn.Module,
    capture: _ActivationGradientCapture,
    class_names: list[str],
    image: Image.Image,
    target_label: str | None,
    image_size: int,
) -> GradCamResult:
    """Un seul forward+backward pass — partagé par `explain_classification_
    prediction` (1 image) et `explain_classification_predictions_batch`
    (N images, modèle/hooks déjà prêts via `_prepare_model_and_capture`).
    `image_size` = résolution RÉELLEMENT utilisée à l'entraînement de CE
    modèle (`artifact["image_size"]`, mode expert) — jamais 224 en dur,
    sous peine de faire voir au modèle une image à une résolution différente
    de celle sur laquelle il a appris (dégrade silencieusement la qualité
    de la carte Grad-CAM, voire fausse la prédiction elle-même)."""
    transform = build_eval_transform(image_size)
    original_size = image.size  # (largeur, hauteur)
    input_tensor = transform(image.convert("RGB")).unsqueeze(0)
    # Le backbone est gelé par défaut (sous-lot B) : sans ceci, AUCUN tenseur
    # en amont de la couche cible n'exige de gradient, le graphe autograd
    # n'est jamais construit jusqu'à elle et le hook backward capte un
    # gradient vide/incorrect. Ne sert qu'à connecter le graphe — les poids
    # gelés ne sont jamais mis à jour (pas d'optimizer.step() ici).
    input_tensor.requires_grad_(True)

    model.zero_grad()
    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1)[0].detach()
    predicted_idx = int(probabilities.argmax().item())

    if target_label is None:
        target_idx = predicted_idx
    else:
        if target_label not in class_names:
            raise GradCamError(f"Classe inconnue : {target_label}. Options : {', '.join(class_names)}")
        target_idx = class_names.index(target_label)

    logits[0, target_idx].backward()

    # Poids par canal = moyenne spatiale du gradient (Grad-CAM original,
    # Selvaraju et al. 2017) — combinaison pondérée des activations, ReLU
    # pour ne garder que l'influence POSITIVE sur la classe cible.
    weights = capture.gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * capture.activation).sum(dim=1, keepdim=True))
    cam_map = cam[0, 0].detach().numpy()

    # Même réalignement testé au sous-lot C (bugs #10/#17) : la carte Grad-CAM
    # est calculée à la résolution du dernier bloc convolutif (ex. 7x7 pour
    # un backbone à stride 32 sur une entrée 224x224), jamais superposée
    # telle quelle à l'image d'origine.
    cam_original_size = resize_map_to_original(cam_map, original_size)

    return GradCamResult(
        predicted_label=class_names[predicted_idx],
        probabilities={name: float(probabilities[i]) for i, name in enumerate(class_names)},
        target_label=class_names[target_idx],
        # Superposition (Lot 16A) — remplace la heatmap seule : les zones
        # rouges de l'image superposée sont celles qui ont le plus influencé
        # la prédiction, directement lisibles sur la photo elle-même.
        heatmap_png=overlay_heatmap_on_image(image, cam_original_size),
        # Sur la carte BRUTE (pas `cam_original_size`) — voir
        # `compute_border_attention_fraction`, calcul quasi gratuit
        # (quelques dizaines de cellules), jamais sur l'image pleine
        # résolution.
        border_attention_fraction=compute_border_attention_fraction(cam_map),
        cam_map_shape=cam_map.shape,
    )


def explain_classification_prediction(
    artifact: dict[str, Any],
    image: Image.Image,
    target_label: str | None = None,
) -> GradCamResult:
    """Point d'entrée — une seule image (ex. une image externe apportée par
    l'utilisateur, hors du dataset d'entraînement). `artifact` est le
    contenu de `VisionClassificationModel.file_path` (`torch.load`, voir
    `workers/vision_classification_worker.py`). `target_label` optionnel :
    explique la classe prédite par défaut, ou une classe précise choisie par
    l'utilisateur (ex. "pourquoi PAS la classe X ?")."""
    model, capture, class_names = _prepare_model_and_capture(artifact)
    # `.get(..., IMAGE_SIZE)` — rétrocompatibilité par absence (modèles
    # entraînés avant le mode expert résolution, tous à 224, l'ancien
    # `IMAGE_SIZE` en dur).
    image_size = artifact.get("image_size", IMAGE_SIZE)
    return _run_gradcam(model, capture, class_names, image, target_label, image_size)


def explain_classification_predictions_batch(
    artifact: dict[str, Any],
    images: list[tuple[str, Image.Image]],
) -> list[GradCamBatchItemResult]:
    """Point d'entrée — PLUSIEURS images en un seul appel (retour utilisateur
    direct : "Grad-CAM devrait supporter le batch, pas une image à la
    fois"). Typiquement les exemples mal classés déjà affichés dans l'onglet
    "Exemples" — expliquer plusieurs erreurs d'un coup, sans ré-uploader
    chacune une par une.

    Modèle reconstruit et hooks enregistrés UNE SEULE FOIS pour tout le
    batch (voir `_prepare_model_and_capture`) — le coût dominant de
    l'opération n'est payé qu'une fois, pas N fois. Toujours la classe
    PRÉDITE qui est expliquée pour chaque image (pas de `target_label` par
    image en mode batch — un batch mélange typiquement plusieurs classes
    prédites différentes, choisir une cible commune n'aurait pas de sens).

    Dégradation par image, jamais par lot entier : une image individuellement
    illisible (cas limite non filtré en amont) n'interrompt pas les autres —
    voir `GradCamBatchItemResult.error`, jamais silencieux."""
    model, capture, class_names = _prepare_model_and_capture(artifact)
    image_size = artifact.get("image_size", IMAGE_SIZE)
    results: list[GradCamBatchItemResult] = []
    for key, image in images:
        try:
            result = _run_gradcam(model, capture, class_names, image, None, image_size)
            results.append(GradCamBatchItemResult(key=key, result=result, error=None))
        except Exception as exc:
            logger.warning(
                "[GradCam] Échec sur l'image '%s' du batch, poursuite avec les suivantes", key, exc_info=True
            )
            results.append(GradCamBatchItemResult(key=key, result=None, error=str(exc)))
    return results
