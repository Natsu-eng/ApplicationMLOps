

# ============================
# PIPELINE COMPLET HAUT NIVEAU
# ============================

from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from src.data.computer_vision_preprocessing import Result
from src.models.computer_vision.persistence import ModelPersistence
from src.models.computer_vision_training import ComputerVisionTrainer, ModelConfig, TrainingConfig
from utils.callbacks import LoggingCallback, TrainingCallback

from src.shared.logging import get_logger

logger = get_logger(__name__)


class ProductionPipeline:
    """
    Pipeline complet haut niveau pour production.
    
    Usage:
        pipeline = ProductionPipeline(model_config, training_config)
        result = pipeline.train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test)
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        callbacks: List[TrainingCallback] = None,
        checkpoint_dir: Path = None
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.callbacks = callbacks or [LoggingCallback()]
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        
        self.trainer = None
        self.test_metrics = None
    
    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_model: bool = True
    ) -> Result:
        """
        Pipeline complet: train -> validate -> test.
        
        GARANTIT: Test set jamais vu pendant training/validation.
        """
        try:
            logger.info("=== DÉBUT DU PIPELINE COMPLET ===")
            
            # 1. Entraînement
            self.trainer = ComputerVisionTrainer(
                model_config=self.model_config,
                training_config=self.training_config,
                callbacks=self.callbacks
            )
            
            train_result = self.trainer.fit(X_train, y_train, X_val, y_val)
            
            if not train_result.success:
                return train_result
            
            logger.info("Entraînement terminé avec succès")
            
            # 2. Évaluation sur TEST SET (jamais vu)
            eval_result = self.trainer.evaluate(X_test, y_test)
            
            if not eval_result.success:
                return eval_result
            
            self.test_metrics = eval_result.data
            logger.info(
                "Évaluation test complétée",
                test_accuracy=self.test_metrics['accuracy'],
                test_f1=self.test_metrics['f1']
            )
            
            # 3. Sauvegarde si demandé
            if save_model:
                model_path = self.checkpoint_dir / f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                
                save_result = ModelPersistence.save(
                    model=self.trainer.model,
                    filepath=model_path,
                    preprocessor=self.trainer.preprocessor,
                    model_config=self.model_config,
                    training_config=self.training_config,
                    history=self.trainer.history,
                    metadata={'test_metrics': self.test_metrics}
                )
                
                if not save_result.success:
                    logger.warning(f"Sauvegarde échouée: {save_result.error}")
            
            # 4. Retour du résultat complet
            return Result.ok({
                'model': self.trainer.model,
                'preprocessor': self.trainer.preprocessor,
                'training_history': self.trainer.history,
                'test_metrics': self.test_metrics,
                'model_path': str(model_path) if save_model else None
            })
            
        except Exception as e:
            logger.error(f"Erreur pipeline: {e}", exc_info=True)
            return Result.err(f"Pipeline échoué: {str(e)}")