"""
Modèles de classification pour Computer Vision.

Classes disponibles:
- SimpleCNN: CNN simple avec Global Average Pooling
- CustomResNet: ResNet personnalisé
- TransferLearningModel: Modèles pré-entraînés (ResNet, VGG, EfficientNet, etc.)
- FineTuningScheduler: Scheduler pour fine-tuning progressif
"""

from .cnn_models import SimpleCNN, CustomResNet
from .transfer_learning import TransferLearningModel, FineTuningScheduler

__all__ = [
    'SimpleCNN',
    'CustomResNet',
    'TransferLearningModel',
    'FineTuningScheduler'
]

