from datetime import datetime
from pathlib import Path
from typing import Callable, Dict
import torch
import torch.nn as nn

from src.data.computer_vision_preprocessing import DataPreprocessor, Result
from src.models.computer_vision_training import ModelConfig, TrainingConfig

from src.shared.logging import StructuredLogger

logger = StructuredLogger(__name__)

# ========================
# SAUVEGARDE ET CHARGEMENT
# ========================

class ModelPersistence:
    """Gestion de la persistance des modèles"""
    
    @staticmethod
    def save(
        model: nn.Module,
        filepath: Path,
        preprocessor: DataPreprocessor = None,
        model_config: ModelConfig = None,
        training_config: TrainingConfig = None,
        history: Dict = None,
        metadata: Dict = None
    ) -> Result:
        """
        Sauvegarde complète du modèle et de son contexte.
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'model_config': model_config.__dict__ if model_config else None,
                'training_config': training_config.__dict__ if training_config else None,
                'preprocessor': {
                    'strategy': preprocessor.strategy if preprocessor else None,
                    'mean': preprocessor.mean_ if preprocessor else None,
                    'std': preprocessor.std_ if preprocessor else None,
                    'min': preprocessor.min_ if preprocessor else None,
                    'max': preprocessor.max_ if preprocessor else None
                },
                'history': history,
                'metadata': metadata or {},
                'saved_at': datetime.now().isoformat(),
                'pytorch_version': torch.__version__
            }
            
            torch.save(checkpoint, filepath)
            
            logger.info(f"Modèle sauvegardé: {filepath}")
            
            return Result.ok(str(filepath))
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
            return Result.err(f"Sauvegarde échouée: {str(e)}")
    
    @staticmethod
    def load(
        filepath: Path,
        model_builder: Callable = None
    ) -> Result:
        """
        Charge un modèle complet.
        
        Returns:
            Result avec dict contenant model, preprocessor, config
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return Result.err(f"Fichier non trouvé: {filepath}")
            
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Reconstruction du preprocessor
            preprocessor = None
            if checkpoint.get('preprocessor'):
                prep_data = checkpoint['preprocessor']
                if prep_data['strategy']:
                    preprocessor = DataPreprocessor(strategy=prep_data['strategy'])
                    preprocessor.mean_ = prep_data.get('mean')
                    preprocessor.std_ = prep_data.get('std')
                    preprocessor.min_ = prep_data.get('min')
                    preprocessor.max_ = prep_data.get('max')
                    preprocessor.fitted = True
            
            logger.info(f"Modèle chargé: {filepath}")
            
            return Result.ok({
                'model_state_dict': checkpoint['model_state_dict'],
                'model_class': checkpoint.get('model_class'),
                'model_config': checkpoint.get('model_config'),
                'training_config': checkpoint.get('training_config'),
                'preprocessor': preprocessor,
                'history': checkpoint.get('history'),
                'metadata': checkpoint.get('metadata', {})
            })
            
        except Exception as e:
            logger.error(f"Erreur chargement: {e}")
            return Result.err(f"Chargement échoué: {str(e)}")