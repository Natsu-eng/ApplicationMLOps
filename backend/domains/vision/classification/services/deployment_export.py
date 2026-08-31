"""Export de déploiement autonome pour la classification d'images — même
principe que les piliers non supervisés (`domains/clustering/services/
deployment_export.py`), adapté à Vision : l'artefact (`GET .../model/export`)
n'est PAS un pipeline scikit-learn autonome (contrairement au tabulaire/
clustering/anomalies/réduction de dimension) mais un dict `{"backbone_id",
"class_names", "dropout_rate", "image_size", "state_dict"}` — recharger un
`state_dict` PyTorch exige de RECONSTRUIRE l'architecture du réseau AVANT de
pouvoir y injecter les poids (`nn.Module.load_state_dict`), une étape que les
autres piliers n'ont jamais eue (leur bundle joblib est directement
utilisable après désérialisation).

Ce module embarque donc, de façon littérale et autonome (jamais un import
`domains.*`), une copie fidèle de la construction d'architecture des 7
backbones du registre (`services/registry.py::CLASSIFICATION_BACKBONE_REGISTRY`)
et du prétraitement d'inférence (`services/engine.py::build_eval_transform`)
— reproduction exacte, jamais une approximation, prouvée par un test en
sous-processus réel comparant les probabilités prédites à la référence.

Différence volontaire avec `build_model` (registry.py) : `weights=None` au
lieu du pré-entraînement ImageNet — le script autonome charge de toute façon
le `state_dict` entraîné juste après, télécharger les poids ImageNet
(potentiellement indisponible/lent sur la machine de déploiement finale)
n'aurait servi à rien."""
from __future__ import annotations

from datetime import datetime, timezone
from string import Template

# Les 3 façons dont un backbone du registre remplace sa tête de
# classification (voir services/registry.py) — reproduites ici à
# l'identique, seule la source des poids diffère (weights=None, voir
# docstring du module).
_FC_BACKBONES = {"resnet18", "resnet34", "shufflenet_v2"}
_CLASSIFIER_LAST_BACKBONES = {"mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"}
_CLASSIFIER_FULL_BACKBONES = {"densenet121"}

_TORCHVISION_CONSTRUCTOR_BY_ID = {
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "shufflenet_v2": "shufflenet_v2_x1_0",
    "mobilenet_v3_small": "mobilenet_v3_small",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "efficientnet_b0": "efficientnet_b0",
    "densenet121": "densenet121",
}

_SCRIPT_TEMPLATE = Template('''#!/usr/bin/env python3
"""Script de déploiement autonome — classification d'images — généré par
DataLab Pro le $generated_at.

Backbone : $backbone_label
Classes ($n_classes) : $class_names_list
Résolution d'entrée : $image_size x $image_size

INSTALLATION
    pip install torch torchvision pillow

UTILISATION
    1) Placez ce script à côté du fichier modèle exporté ($artifact_filename).
    2) Une seule image :
         python $script_filename --predict chemin/vers/image.jpg
    3) Un dossier entier d'images :
         python $script_filename --batch dossier/ sortie.csv

Ce script ne dépend d'AUCUN module de la plateforme DataLab Pro — seules les
bibliothèques listées ci-dessus sont nécessaires. Contrairement aux exports
scikit-learn (ML tabulaire/clustering/...), un modèle PyTorch ne peut pas se
recharger d'un bloc : ce script reconstruit d'abord l'ARCHITECTURE du réseau
($backbone_label), puis y injecte les poids entraînés ($artifact_filename)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ARTIFACT_PATH = Path(__file__).parent / "$artifact_filename"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_backbone(backbone_id: str, num_classes: int, dropout_rate: float) -> nn.Module:
    """Reconstruction FIDÈLE de l'architecture — copie de
    services/registry.py de DataLab Pro, à la différence près que les poids
    ImageNet ne sont pas téléchargés (weights=None, voir en-tête du script) :
    le state_dict entraîné ci-dessous les remplace de toute façon en entier."""
    if backbone_id == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    elif backbone_id == "resnet34":
        model = models.resnet34(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    elif backbone_id == "shufflenet_v2":
        model = models.shufflenet_v2_x1_0(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    elif backbone_id == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    elif backbone_id == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    elif backbone_id == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    elif backbone_id == "densenet121":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(in_features, num_classes))
    else:
        raise ValueError(f"Backbone inconnu : {backbone_id}")
    return model


def _load_model():
    artifact = torch.load(ARTIFACT_PATH, weights_only=True, map_location="cpu")
    class_names = artifact["class_names"]
    # 224 = ancien IMAGE_SIZE en dur de DataLab Pro — rétrocompatibilité par
    # absence pour un artefact entraîné avant le mode expert résolution,
    # JAMAIS la résolution de ce modèle précis (déjà lue depuis la clé
    # ci-dessus si elle existe).
    image_size = artifact.get("image_size", 224)
    model = _build_backbone(artifact["backbone_id"], len(class_names), artifact["dropout_rate"])
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    return model, class_names, image_size


def predict(image_path: Path, model=None, class_names=None, image_size=None) -> dict:
    """Prédit UNE image. Recharge le modèle si non fourni (usage CLI) —
    réutilisez `_load_model()` une seule fois vous-même pour un traitement
    en lot dans votre propre code Python (voir --batch pour l'équivalent
    en ligne de commande)."""
    if model is None:
        model, class_names, image_size = _load_model()
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    predicted_idx = int(probabilities.argmax().item())
    return {
        "fichier": str(image_path),
        "prediction": class_names[predicted_idx],
        "confiance": float(probabilities[predicted_idx]),
        "probabilites": {name: float(probabilities[i]) for i, name in enumerate(class_names)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prédiction hors ligne (modèle de classification exporté).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predict", metavar="IMAGE", help="Chemin vers une image.")
    group.add_argument("--batch", nargs=2, metavar=("DOSSIER", "SORTIE.csv"), help="Prédiction sur un dossier.")
    args = parser.parse_args()

    if args.predict:
        result = predict(Path(args.predict))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        folder, output_path = args.batch
        model, class_names, image_size = _load_model()
        image_paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not image_paths:
            raise SystemExit(f"Aucune image trouvée dans {folder} (extensions supportées : {sorted(IMAGE_EXTENSIONS)})")
        rows = [predict(p, model, class_names, image_size) for p in image_paths]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["fichier", "prediction", "confiance"] + [f"probabilite_{c}" for c in class_names]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "fichier": row["fichier"],
                        "prediction": row["prediction"],
                        "confiance": row["confiance"],
                        **{f"probabilite_{c}": row["probabilites"][c] for c in class_names},
                    }
                )
        print(f"{len(rows)} image(s) prédite(s) -> {output_path}")


if __name__ == "__main__":
    main()
''')


def generate_vision_classification_deployment_script(
    backbone_id: str,
    backbone_label: str,
    class_names: list[str],
    image_size: int,
    artifact_filename: str,
    script_filename: str,
) -> str:
    """Construit le script `.py` autonome — voir docstring du module."""
    if backbone_id not in _TORCHVISION_CONSTRUCTOR_BY_ID:
        raise ValueError(f"Backbone inconnu pour le déploiement : {backbone_id}")
    return _SCRIPT_TEMPLATE.substitute(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        backbone_label=backbone_label,
        n_classes=len(class_names),
        class_names_list=", ".join(class_names),
        image_size=image_size,
        artifact_filename=artifact_filename,
        script_filename=script_filename,
    )
