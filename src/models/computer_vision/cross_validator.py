from typing import List

import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.data.computer_vision_preprocessing import Result
from src.models.computer_vision_training import ComputerVisionTrainer, ModelConfig, TrainingConfig
from src.shared.logging import StructuredLogger
from utils.callbacks import TrainingCallback
logger = StructuredLogger(__name__)

# ================
# CROSS-VALIDATION
# ================

class CrossValidator:
    """
    Cross-validation pour estimer la variance du modèle.
    """   
    def __init__(
        self,
        n_splits: int = 5,
        model_config: ModelConfig = None,
        training_config: TrainingConfig = None
    ):
        self.n_splits = n_splits
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        
        logger.info(f"CrossValidator initialisé: {n_splits} folds")
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        callbacks: List[TrainingCallback] = None
    ) -> Result:
        """
        Effectue une validation croisée stratifiée.
        
        Returns:
            Result avec scores de chaque fold
        """
        try:
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.training_config.seed
            )
            
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                logger.info(f"=== Fold {fold + 1}/{self.n_splits} ===")
                
                # Split des données
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_val_fold = X[val_idx]
                y_val_fold = y[val_idx]
                
                # Entraînement sur ce fold
                trainer = ComputerVisionTrainer(
                    model_config=self.model_config,
                    training_config=self.training_config,
                    callbacks=callbacks
                )
                
                result = trainer.fit(
                    X_train_fold, y_train_fold,
                    X_val_fold, y_val_fold
                )
                
                if not result.success:
                    logger.warning(f"Fold {fold+1} échoué: {result.error}")
                    continue
                
                # Évaluation sur validation
                eval_result = trainer.evaluate(X_val_fold, y_val_fold)
                
                if eval_result.success:
                    fold_results.append({
                        'fold': fold + 1,
                        'metrics': eval_result.data,
                        'best_epoch': result.metadata.get('best_epoch')
                    })
            
            # Calcul statistiques agrégées
            if not fold_results:
                return Result.err("Aucun fold réussi")
            
            metrics_names = ['accuracy', 'precision', 'recall', 'f1']
            aggregated = {}
            
            for metric in metrics_names:
                values = [f['metrics'][metric] for f in fold_results]
                aggregated[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            
            logger.info(
                "Cross-validation terminée",
                n_successful_folds=len(fold_results),
                avg_accuracy=aggregated['accuracy']['mean'],
                std_accuracy=aggregated['accuracy']['std']
            )
            
            return Result.ok({
                'fold_results': fold_results,
                'aggregated_metrics': aggregated,
                'n_folds': len(fold_results)
            })
            
        except Exception as e:
            logger.error(f"Erreur cross-validation: {e}", exc_info=True)
            return Result.err(f"Cross-validation échouée: {str(e)}")
