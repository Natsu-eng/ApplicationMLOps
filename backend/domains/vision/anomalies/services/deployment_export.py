"""Export de déploiement autonome pour la détection d'anomalies visuelles —
même principe que `domains/vision/classification/services/deployment_export.py`.
Génère un script Python qui recharge le bundle `state_dict` (`GET
.../model/export`) et note de nouvelles images, SANS jamais importer un
module `domains.*` de ce projet.

Différence avec la classification : les 3 architectures du registre
(`services/registry.py::ANOMALY_MODEL_REGISTRY`) ne sont pas des backbones
torchvision standard (aucun constructeur `models.xxx()` à appeler) mais des
`nn.Module` PROPRES à ce projet (`ConvAutoEncoder`/`DenoisingConvAutoEncoder`/
`ConvVAE`) — ce script embarque donc une copie littérale de leurs
DÉFINITIONS DE CLASSE, pas seulement un appel de constructeur. Reproduction
fidèle, jamais une réimplémentation approximative : prouvée par un test en
sous-processus réel comparant l'erreur de reconstruction à la référence.

Portée volontairement limitée au score numérique (erreur de reconstruction +
comparaison au seuil) — jamais la carte de chaleur visuelle (superposition
image, voir `services/localization.py`), qui resterait un gros morceau de
code supplémentaire (colormap, réalignement) pour une valeur secondaire dans
un contexte de script hors ligne ; disponible dans l'application elle-même
(`POST .../predict`) pour qui a besoin de la visualisation."""
from __future__ import annotations

from datetime import datetime, timezone
from string import Template

_MODEL_CLASSES = '''
class ConvAutoEncoder(nn.Module):
    """Copie fidèle de services/registry.py::ConvAutoEncoder — entièrement
    convolutif, reçoit toujours des images IMAGE_SIZE x IMAGE_SIZE."""

    def __init__(self, base_filters: int = 32, num_stages: int = 3):
        super().__init__()
        encoder_layers = []
        in_channels = 3
        for stage in range(num_stages):
            out_channels = base_filters * (2**stage)
            encoder_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for stage in reversed(range(num_stages)):
            out_channels = base_filters * (2 ** (stage - 1)) if stage > 0 else 3
            decoder_layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
            if stage > 0:
                decoder_layers += [nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
            in_channels = out_channels
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class DenoisingConvAutoEncoder(ConvAutoEncoder):
    """Copie fidèle de services/registry.py::DenoisingConvAutoEncoder — le
    bruit n'est ajouté qu'en entraînement (self.training), sans effet en
    inférence (.eval(), toujours le cas dans ce script)."""

    def __init__(self, base_filters: int = 32, num_stages: int = 3, noise_factor: float = 0.1):
        super().__init__(base_filters=base_filters, num_stages=num_stages)
        self.noise_factor = noise_factor

    def forward(self, x):
        if self.training and self.noise_factor > 0:
            x = torch.clamp(x + torch.randn_like(x) * self.noise_factor, 0.0, 1.0)
        return super().forward(x)


class ConvVAE(nn.Module):
    """Copie fidèle de services/registry.py::ConvVAE — en inférence
    (.eval()), z = mu (déterministe, pas d'échantillonnage) : le terme KL
    d'entraînement n'intervient jamais ici."""

    def __init__(self, base_filters: int = 32, num_stages: int = 3):
        super().__init__()
        self._base = ConvAutoEncoder(base_filters=base_filters, num_stages=num_stages)
        bottleneck_channels = base_filters * (2 ** (num_stages - 1))
        self.mu_head = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=1)

    def forward(self, x):
        h = self._base.encoder(x)
        mu = self.mu_head(h)
        return self._base.decoder(mu)


_MODEL_BUILDERS = {
    "conv_autoencoder": ConvAutoEncoder,
    "denoising_autoencoder": DenoisingConvAutoEncoder,
    "conv_vae": ConvVAE,
}
'''

_SCRIPT_TEMPLATE = Template('''#!/usr/bin/env python3
"""Script de déploiement autonome — détection d'anomalies visuelles —
généré par DataLab Pro le $generated_at.

Modèle : $model_label
Résolution d'entrée : $image_size x $image_size
Seuil de détection (calibré à l'entraînement, point de Youden sur ROC) : $threshold_display

INSTALLATION
    pip install torch torchvision pillow

UTILISATION
    1) Placez ce script à côté du fichier modèle exporté ($artifact_filename).
    2) Une seule image :
         python $script_filename --predict chemin/vers/image.jpg
    3) Un dossier entier d'images :
         python $script_filename --batch dossier/ sortie.csv

Ce script ne dépend d'AUCUN module de la plateforme DataLab Pro — seules les
bibliothèques listées ci-dessus sont nécessaires. Le fichier modèle
($artifact_filename) contient les poids entraînés ET le seuil de détection
déjà calibré (réutilisé tel quel, jamais recalculé sur une seule image — un
seuil n'a de sens que calibré sur un jeu de données de référence).

Portée : score numérique uniquement (erreur de reconstruction + comparaison
au seuil), pas la carte de chaleur visuelle — voir la docstring du module
`deployment_export.py` de DataLab Pro pour le raisonnement."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
$model_classes

ARTIFACT_PATH = Path(__file__).parent / "$artifact_filename"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_model():
    artifact = torch.load(ARTIFACT_PATH, weights_only=True, map_location="cpu")
    model_id = artifact["model_id"]
    if model_id not in _MODEL_BUILDERS:
        raise ValueError(f"Modèle inconnu : {model_id}")
    model = _MODEL_BUILDERS[model_id]()
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    threshold = float(artifact["threshold"])
    # 128 = ANOMALY_IMAGE_SIZE historique de DataLab Pro — rétrocompatibilité
    # par absence pour un artefact entraîné avant le mode expert résolution,
    # JAMAIS la résolution de ce modèle précis (déjà lue depuis la clé
    # ci-dessus si elle existe).
    image_size = artifact.get("image_size", 128)
    return model, threshold, image_size


def score(image_path: Path, model=None, threshold=None, image_size=None) -> dict:
    """Note UNE image. Recharge le modèle si non fourni (usage CLI) —
    réutilisez `_load_model()` une seule fois vous-même pour un traitement
    en lot dans votre propre code Python (voir --batch pour l'équivalent en
    ligne de commande)."""
    if model is None:
        model, threshold, image_size = _load_model()
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        reconstructed = model(input_tensor)
        error_map = torch.mean((input_tensor - reconstructed) ** 2, dim=1)
        anomaly_score = float(error_map.mean().item())
    return {
        "fichier": str(image_path),
        "score_anomalie": anomaly_score,
        "seuil": threshold,
        "est_anomalie": anomaly_score > threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Notation hors ligne (modèle de détection exporté de DataLab Pro).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", metavar="IMAGE", help="Chemin vers une image.")
    group.add_argument("--batch", nargs=2, metavar=("DOSSIER", "SORTIE.csv"), help="Notation sur un dossier.")
    args = parser.parse_args()

    if args.predict:
        result = score(Path(args.predict))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        folder, output_path = args.batch
        model, threshold, image_size = _load_model()
        image_paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not image_paths:
            raise SystemExit(f"Aucune image trouvée dans {folder} (extensions supportées : {sorted(IMAGE_EXTENSIONS)})")
        rows = [score(p, model, threshold, image_size) for p in image_paths]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["fichier", "score_anomalie", "seuil", "est_anomalie"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"{len(rows)} image(s) notée(s) -> {output_path}")


if __name__ == "__main__":
    main()
''')


def generate_vision_anomaly_deployment_script(
    model_id: str,
    model_label: str,
    image_size: int,
    threshold: float,
    artifact_filename: str,
    script_filename: str,
) -> str:
    """Construit le script `.py` autonome — voir docstring du module."""
    if model_id not in {"conv_autoencoder", "denoising_autoencoder", "conv_vae"}:
        raise ValueError(f"Modèle inconnu pour le déploiement : {model_id}")
    return _SCRIPT_TEMPLATE.substitute(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        model_label=model_label,
        image_size=image_size,
        threshold_display=f"{threshold:.6f}",
        artifact_filename=artifact_filename,
        script_filename=script_filename,
        model_classes=_MODEL_CLASSES,
    )
