"""
Modèles de détection d'anomalies pour Computer Vision.

Classes disponibles:
- ConvAutoEncoder: Autoencodeur convolutif standard
- VariationalAutoEncoder: Autoencodeur variationnel (VAE)
- DenoisingAutoEncoder: Autoencodeur avec débruitage
- ProfessionalPatchCore: PatchCore pour détection d'anomalies
- ProfessionalSiameseNetwork: Réseau Siamese pour détection d'anomalies
- AnomalyTypeClassifier: Classificateur des types d'anomalies (crack, scratch, etc.)
"""

from .autoencoders import (
    ConvAutoEncoder,
    VariationalAutoEncoder,
    DenoisingAutoEncoder
)
from .patch_core import ProfessionalPatchCore
from .siamese_networks import ProfessionalSiameseNetwork
from .anomaly_type_classifier import (
    AnomalyTypeClassifier,
    load_anomaly_type_labels_from_mvtec
)

__all__ = [
    'ConvAutoEncoder',
    'VariationalAutoEncoder',
    'DenoisingAutoEncoder',
    'ProfessionalPatchCore',
    'ProfessionalSiameseNetwork',
    'AnomalyTypeClassifier',
    'load_anomaly_type_labels_from_mvtec'
]

