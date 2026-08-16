"""Localisation des défauts — cartes d'erreur, masques binaires,
encodage en image affichable (pilier Vision, Lot 15 sous-lot C).

Module séparé, réutilisé tel quel par Grad-CAM (sous-lot D, planifié) : un
seul mécanisme de superposition heatmap/image dans tout le module vision,
jamais deux.

Corrige directement 3 des 9 bugs critiques déjà documentés dans
`docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md` :
- **#14** — aucune fonction de génération de masque binaire n'existait dans
  le legacy (`generate_binary_mask` ci-dessous, absente avant ce lot).
- **#10/#17** — la carte d'erreur était calculée à la résolution du modèle
  (ex. 128x128) puis superposée telle quelle à l'image originale (ex.
  512x512) sans redimensionnement → décalage. `resize_map_to_original`
  refait ce resize explicitement, systématiquement, jamais laissé à la
  charge de l'appelant.
- **#11** — la comparaison image/reconstruction pour le calcul d'erreur
  n'est jamais mélangée entre espace normalisé et non-normalisé : ce module
  ne fait QUE de la mise en forme (resize, normalisation [0,1] pour
  l'affichage, encodage PNG) sur des cartes déjà calculées de façon
  cohérente par `vision_anomaly_training.py` — jamais de recalcul ici.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

# Méthode par défaut du masque binaire : percentile plutôt qu'un seuil
# absolu — une carte d'erreur n'a pas d'échelle universelle (dépend du
# modèle/dataset), le percentile reste interprétable sur n'importe quelle
# carte ("les 5% de pixels les plus atypiques de CETTE image").
DEFAULT_MASK_PERCENTILE = 0.95


def generate_binary_mask(error_map: np.ndarray, percentile: float = DEFAULT_MASK_PERCENTILE) -> np.ndarray:
    """Convertit une carte d'erreur continue (H, W) en masque binaire (H, W)
    — correctif du bug #14 (fonction absente du legacy). `percentile` définit
    la fraction de pixels considérés comme faisant partie du défaut (0.95 =
    les 5% de pixels les plus erronés de cette image précise)."""
    if not 0.0 < percentile < 1.0:
        raise ValueError(f"percentile doit être entre 0 et 1 (exclus), reçu {percentile}")
    threshold = np.percentile(error_map, percentile * 100)
    return (error_map > threshold).astype(np.uint8)


def resize_map_to_original(error_map: np.ndarray, original_size: tuple[int, int]) -> np.ndarray:
    """Redimensionne une carte (H, W) — calculée à la résolution du modèle —
    vers `original_size` (largeur, hauteur) de l'image source AVANT toute
    superposition. Correctif des bugs #10/#17 (legacy : la carte n'était
    jamais réalignée, décalage systématique dès que l'image source différait
    de la résolution du modèle)."""
    img = Image.fromarray(error_map.astype(np.float32), mode="F")
    resized = img.resize(original_size, resample=Image.BILINEAR)
    return np.array(resized, dtype=np.float32)


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Normalisation min-max propre à CETTE carte — jamais un seuil global
    partagé entre images (chaque carte d'erreur a sa propre échelle)."""
    min_val, max_val = float(array.min()), float(array.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(array, dtype=np.uint8)
    normalized = (array - min_val) / (max_val - min_val)
    return (normalized * 255).astype(np.uint8)


def encode_heatmap_png(error_map: np.ndarray) -> str:
    """Encode une carte d'erreur (H, W) en PNG niveaux de gris, data URI
    base64 directement affichable (`<img src="...">`), sans dépendance
    frontend à l'encodage. Une des sorties concrètes du correctif #8/#16
    (heatmap désormais toujours produite, jamais une fonction orpheline)."""
    grayscale = _normalize_to_uint8(error_map)
    buf = io.BytesIO()
    Image.fromarray(grayscale, mode="L").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def encode_mask_png(mask: np.ndarray) -> str:
    """Encode un masque binaire (H, W, valeurs 0/1) en PNG noir/blanc, data
    URI base64 — même format d'encodage que `encode_heatmap_png` pour rester
    cohérent côté consommateur."""
    black_white = (mask.astype(np.uint8) * 255)
    buf = io.BytesIO()
    Image.fromarray(black_white, mode="L").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
