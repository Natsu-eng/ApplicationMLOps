# =====================
# HYPERPARAMETER TUNING
# =====================

from typing import Any, Dict, List

import numpy as np
from src.data.computer_vision_preprocessing import Result
from src.models.computer_vision.cross_validator import CrossValidator
from src.models.computer_vision_training import ModelConfig, TrainingConfig

from src.shared.logging import get_logger

logger = get_logger(__name__)


class HyperparameterTuner:
    """
    Optimisation des hyperparamètres avec recherche en grille ou aléatoire.
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        base_training_config: TrainingConfig,
        param_grid: Dict[str, List[Any]],
        n_trials: int = 10,
        cv_splits: int = 3
    ):
        self.model_config = model_config
        self.base_training_config = base_training_config
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.cv_splits = cv_splits
        
        self.best_params = None
        self.best_score = 0.0
        self.trials_history = []
        
        logger.info(
            f"HyperparameterTuner initialisé - "
            f"n_trials: {n_trials}, "
            f"param_grid: {param_grid}"
        )
    
    def tune(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = 'f1'
    ) -> Result:
        """
        Optimise les hyperparamètres.
        
        Args:
            X: Features
            y: Labels
            metric: Métrique à optimiser ('f1', 'accuracy', etc.)
        """
        try:
            logger.info(f"Début du tuning sur {self.n_trials} essais")
            
            for trial in range(self.n_trials):
                # Échantillonnage aléatoire des hyperparamètres
                trial_params = self._sample_params()
                
                logger.info(f"Trial {trial + 1}/{self.n_trials} - params: {trial_params}")
                
                # Configuration pour ce trial
                trial_config = self._create_trial_config(trial_params)
                
                # Cross-validation
                cv = CrossValidator(
                    n_splits=self.cv_splits,
                    model_config=self.model_config,
                    training_config=trial_config
                )
                
                cv_result = cv.cross_validate(X, y)
                
                if not cv_result.success:
                    logger.warning(f"Trial {trial + 1} échoué: {cv_result.error}")
                    continue
                
                # Score moyen
                score = cv_result.data['aggregated_metrics'][metric]['mean']
                
                # Stockage
                self.trials_history.append({
                    'trial': trial + 1,
                    'params': trial_params,
                    'score': score,
                    'metric': metric
                })
                
                # Mise à jour du meilleur
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = trial_params
                    logger.info(
                        f"Nouveau meilleur score! - "
                        f"score: {score:.4f}, "
                        f"params: {trial_params}"
                    )
            
            if self.best_params is None:
                return Result.err("Aucun trial réussi")
            
            logger.info(
                f"Tuning terminé - "
                f"best_score: {self.best_score:.4f}, "
                f"best_params: {self.best_params}"
            )
            
            return Result.ok({
                'best_params': self.best_params,
                'best_score': self.best_score,
                'trials_history': self.trials_history
            })
            
        except Exception as e:
            logger.error(f"Erreur tuning: {e}", exc_info=True)
            return Result.err(f"Tuning échoué: {str(e)}")
    
    def _sample_params(self) -> Dict[str, Any]:
        """Échantillonne aléatoirement des hyperparamètres"""
        sampled = {}
        for param, values in self.param_grid.items():
            sampled[param] = np.random.choice(values)
        return sampled
    
    def _create_trial_config(self, trial_params: Dict[str, Any]) -> TrainingConfig:
        """Crée une config pour un trial"""
        config_dict = self.base_training_config.__dict__.copy()
        config_dict.update(trial_params)
        return TrainingConfig(**config_dict)