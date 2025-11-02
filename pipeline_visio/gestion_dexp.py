from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
from src.data.computer_vision_preprocessing import Result
from src.models.computer_vision_training import ModelConfig, ProductionPipeline, TrainingConfig
from src.shared.logging import StructuredLogger
from utils.callbacks import CheckpointCallback, LoggingCallback


logger = StructuredLogger(__name__)

# ==========================
# GESTIONNAIRE D'EXPÉRIENCES
# ==========================

class ExperimentManager:
    """
    Gestionnaire pour organiser plusieurs expériences.
    """
    
    def __init__(self, experiments_dir: Path = Path("./experiments")):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []
        
        logger.info(f"ExperimentManager initialisé: {self.experiments_dir}")
    
    def run_experiment(
        self,
        name: str,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        description: str = ""
    ) -> Result:
        """
        Exécute une expérience complète et sauvegarde les résultats.
        """
        
        logger.info(f"Démarrage expérience: {name}")
        
        experiment_dir = self.experiments_dir / name
        experiment_dir.mkdir(exist_ok=True)
        
        try:
            # Pipeline
            pipeline = ProductionPipeline(
                model_config=model_config,
                training_config=training_config,
                callbacks=[
                    LoggingCallback(log_every_n_epochs=5),
                    CheckpointCallback(checkpoint_dir=experiment_dir / "checkpoints")
                ],
                checkpoint_dir=experiment_dir
            )
            
            result = pipeline.train_and_evaluate(
                X_train, y_train,
                X_val, y_val,
                X_test, y_test,
                save_model=True
            )
            
            if not result.success:
                return result
            
            # Sauvegarde métadonnées expérience
            experiment_metadata = {
                'name': name,
                'description': description,
                'timestamp': datetime.now().isoformat(),
                'model_config': {k: str(v) for k, v in model_config.__dict__.items()},
                'training_config': {k: str(v) for k, v in training_config.__dict__.items()},
                'test_metrics': result.data['test_metrics'],
                'training_time': result.metadata.get('training_time'),
                'model_path': result.data.get('model_path')
            }
            
            metadata_path = experiment_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                import json
                json.dump(experiment_metadata, f, indent=2)
            
            self.experiments.append(experiment_metadata)
            
            logger.info(
                f"Expérience {name} terminée",
                test_accuracy=result.data['test_metrics']['accuracy']
            )
            
            return Result.ok(experiment_metadata)
            
        except Exception as e:
            logger.error(f"Erreur expérience {name}: {e}", exc_info=True)
            return Result.err(f"Expérience échouée: {str(e)}")
    
    def compare_experiments(self) -> Dict[str, Any]:
        """Compare toutes les expériences exécutées"""
        
        if not self.experiments:
            return {"error": "Aucune expérience à comparer"}
        
        comparison = {
            'n_experiments': len(self.experiments),
            'experiments': []
        }
        
        for exp in self.experiments:
            comparison['experiments'].append({
                'name': exp['name'],
                'test_accuracy': exp['test_metrics']['accuracy'],
                'test_f1': exp['test_metrics']['f1'],
                'model_type': exp['model_config'].get('model_type')
            })
        
        # Meilleure expérience
        best_exp = max(
            self.experiments,
            key=lambda x: x['test_metrics']['f1']
        )
        
        comparison['best_experiment'] = {
            'name': best_exp['name'],
            'test_f1': best_exp['test_metrics']['f1'],
            'test_accuracy': best_exp['test_metrics']['accuracy']
        }
        
        return comparison
    
    def generate_report(self, output_path: Path = None) -> str:
        """Génère un rapport comparatif"""
        
        comparison = self.compare_experiments()
        
        report = []
        report.append("="*70)
        report.append("RAPPORT DE COMPARAISON D'EXPÉRIENCES")
        report.append("="*70)
        report.append(f"\nNombre d'expériences: {comparison['n_experiments']}\n")
        
        report.append("RÉSULTATS PAR EXPÉRIENCE:")
        report.append("-"*70)
        
        for exp in comparison['experiments']:
            report.append(f"\n{exp['name']}")
            report.append(f"  Model: {exp['model_type']}")
            report.append(f"  Test Accuracy: {exp['test_accuracy']:.4f}")
            report.append(f"  Test F1: {exp['test_f1']:.4f}")
        
        report.append("\n" + "-"*70)
        report.append("MEILLEURE EXPÉRIENCE:")
        best = comparison['best_experiment']
        report.append(f"  {best['name']}")
        report.append(f"  Test F1: {best['test_f1']:.4f}")
        report.append(f"  Test Accuracy: {best['test_accuracy']:.4f}")
        report.append("="*70)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Rapport sauvegardé: {output_path}")
        
        return report_text