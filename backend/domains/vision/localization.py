"""Localisation des défauts — cartes d'erreur, masques binaires,
encodage en image affichable (pilier Vision, Lot 15 sous-lot C ; colorisation
et superposition ajoutées au Lot 16A).

Module séparé, réutilisé tel quel par Grad-CAM (sous-lot D) : un seul
mécanisme de superposition heatmap/image dans tout le module vision, jamais
deux.

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

**Lot 16A** — correctif d'un vrai défaut trouvé en testant l'app réellement
(retour utilisateur direct) : `encode_heatmap_png` produisait un PNG en
NIVEAUX DE GRIS, jamais la carte de chaleur rouge/bleu standard attendue
pour du Grad-CAM (Selvaraju et al. 2017), et les deux usages (Grad-CAM,
anomalies visuelles, structure normal/défaut) affichaient la heatmap et l'image source côte
à côte plutôt que superposées. `_apply_colormap` (palette "jet" — bleu =
faible, rouge = fort) et `overlay_heatmap_on_image` corrigent les deux à la
fois, une seule fois, puisque les deux usages partagent ce module.
`generate_binary_mask`/`resize_map_to_original` restent INCHANGÉES : elles
opèrent sur la carte brute (float32, avant toute colorisation), le calibrage
du seuil déjà testé ne doit jamais dépendre du rendu visuel.
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


def _apply_colormap(gray_uint8: np.ndarray) -> np.ndarray:
    """Palette "jet" (bleu = faible, rouge = fort) — approximation standard
    par fonctions triangulaires sur chaque canal, sans dépendance
    supplémentaire (pas de matplotlib pour une seule palette). Entrée (H, W)
    uint8 [0,255], sortie (H, W, 3) uint8."""
    x = gray_uint8.astype(np.float32) / 255.0
    r = np.clip(np.minimum(4 * x - 1.5, -4 * x + 4.5), 0, 1)
    g = np.clip(np.minimum(4 * x - 0.5, -4 * x + 3.5), 0, 1)
    b = np.clip(np.minimum(4 * x + 0.5, -4 * x + 2.5), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def encode_heatmap_png(error_map: np.ndarray) -> str:
    """Encode une carte d'erreur (H, W) en PNG COULEUR (palette "jet" — bleu
    = faible contribution/erreur, rouge = forte), data URI base64
    directement affichable (`<img src="...">`), sans dépendance frontend à
    l'encodage. Une des sorties concrètes du correctif #8/#16 (heatmap
    désormais toujours produite, jamais une fonction orpheline) — colorisée
    depuis le Lot 16A, jamais plus en niveaux de gris."""
    grayscale = _normalize_to_uint8(error_map)
    colored = _apply_colormap(grayscale)
    buf = io.BytesIO()
    Image.fromarray(colored, mode="RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def overlay_heatmap_on_image(original_image: Image.Image, error_map: np.ndarray, alpha: float = 0.45) -> str:
    """Superpose la carte d'erreur (colormap "jet") sur l'image source en
    alpha-blend — rendu Grad-CAM standard, remplace l'affichage côte à côte
    utilisé jusqu'ici pour les deux usages (Grad-CAM, exemples d'anomalie
    visuelle). `error_map` DOIT déjà être réalignée à la taille de
    `original_image` (voir `resize_map_to_original`) — cette fonction ne
    fait aucun resize, uniquement la mise en couleur et le mélange."""
    rgb_original = np.array(original_image.convert("RGB"), dtype=np.float32)
    if error_map.shape != rgb_original.shape[:2]:
        raise ValueError(
            "La carte d'erreur doit déjà être réalignée à la taille de l'image d'origine "
            f"(reçu {error_map.shape}, attendu {rgb_original.shape[:2]}) — voir resize_map_to_original."
        )
    colored = _apply_colormap(_normalize_to_uint8(error_map)).astype(np.float32)
    blended = np.clip((1 - alpha) * rgb_original + alpha * colored, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended, mode="RGB").save(buf, format="PNG")
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


def encode_image_png(image: Image.Image) -> str:
    """Encode une image PIL RGB en PNG, data URI base64 — même format que
    les autres encodeurs de ce module (aperçu d'augmentation, Lot 6A)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
