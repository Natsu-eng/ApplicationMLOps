# ============================================================================
# EXEMPLES D'UTILISATION AVANCÉS
# ============================================================================

from pathlib import Path
import numpy as np

from src.models.computer_vision_training import AnomalyAwareTrainer, ComputerVisionTrainer, ConfigFactory, HyperparameterTuner, ModelConfig, ModelType, TrainingConfig
from src.models.pipeline_visio.gestion_dexp import ExperimentManager
from utils.callbacks import LoggingCallback


def example_experiment_comparison():
    """Exemple: Comparaison de plusieurs configurations"""
    
    print("\n" + "="*70)
    print("EXEMPLE: COMPARAISON D'EXPÉRIENCES")
    print("="*70 + "\n")
    
    # Données
    X_train = np.random.randn(200, 32, 32, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, 200)
    X_val = np.random.randn(40, 32, 32, 3).astype(np.float32)
    y_val = np.random.randint(0, 2, 40)
    X_test = np.random.randn(50, 32, 32, 3).astype(np.float32)
    y_test = np.random.randint(0, 2, 50)
    
    # Manager
    manager = ExperimentManager(experiments_dir=Path("./experiments_demo"))
    
    # Expérience 1: Quick test
    print("Expérience 1: Quick Test...")
    model_cfg, train_cfg = ConfigFactory.get_config('quick_test')
    result1 = manager.run_experiment(
        name="quick_test",
        model_config=model_cfg,
        training_config=train_cfg,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        description="Test rapide avec configuration légère"
    )
    
    # Expérience 2: Balanced
    print("\nExpérience 2: Balanced...")
    model_cfg, train_cfg = ConfigFactory.get_config('balanced')
    train_cfg.epochs = 10  # Réduction pour demo
    result2 = manager.run_experiment(
        name="balanced",
        model_config=model_cfg,
        training_config=train_cfg,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        description="Configuration équilibrée"
    )
    
    # Comparaison
    print("\n" + "-"*70)
    print("COMPARAISON:")
    print(manager.generate_report())


def example_hyperparameter_tuning():
    """Exemple: Optimisation des hyperparamètres"""
    
    print("\n" + "="*70)
    print("EXEMPLE: OPTIMISATION HYPERPARAMÈTRES")
    print("="*70 + "\n")
    
    # Configuration de base
    model_config = ModelConfig(
        model_type=ModelType.SIMPLE_CNN,
        num_classes=2
    )
    
    base_training_config = TrainingConfig(
        epochs=5,  # Court pour demo
        batch_size=16
    )
    
    # Grille de paramètres à tester
    param_grid = {
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'dropout_rate': [0.3, 0.5, 0.7],
        'batch_size': [16, 32]
    }
    
    # Données
    X = np.random.randn(150, 32, 32, 3).astype(np.float32)
    y = np.random.randint(0, 2, 150)
    
    # Tuning
    tuner = HyperparameterTuner(
        model_config=model_config,
        base_training_config=base_training_config,
        param_grid=param_grid,
        n_trials=5,
        cv_splits=2
    )
    
    result = tuner.tune(X, y, metric='f1')
    
    if result.success:
        print(f"\n✅ Meilleurs hyperparamètres:")
        for param, value in result.data['best_params'].items():
            print(f"  {param}: {value}")
        print(f"\nMeilleur score F1: {result.data['best_score']:.4f}")


def example_anomaly_detection():
    """Exemple: Détection d'anomalies spécifiques"""
    
    print("\n" + "="*70)
    print("EXEMPLE: DÉTECTION D'ANOMALIES")
    print("="*70 + "\n")
    
    # Trainer pour anomalie spécifique
    anomaly_trainer = AnomalyAwareTrainer(
        anomaly_type="scratch",
        taxonomy_config=None  # Utilise taxonomie par défaut
    )
    
    # Données
    X_train = np.random.randn(200, 64, 64, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, 200)  # 0: normal, 1: anomalie
    X_val = np.random.randn(40, 64, 64, 3).astype(np.float32)
    y_val = np.random.randint(0, 2, 40)
    X_test = np.random.randn(50, 64, 64, 3).astype(np.float32)
    y_test = np.random.randint(0, 2, 50)
    
    # Entraînement
    result = anomaly_trainer.train(
        X_train, y_train,
        X_val, y_val,
        callbacks=[LoggingCallback(log_every_n_epochs=5)]
    )
    
    if result.success:
        print("✅ Modèle entraîné pour détection de 'scratch'")
        
        # Création trainer standard pour évaluation
        trainer = ComputerVisionTrainer(
            model_config=anomaly_trainer.model_config,
            training_config=anomaly_trainer.training_config
        )
        trainer.model = result.data['model']
        trainer.preprocessor = result.data['preprocessor']
        
        # Évaluation avec métriques anomalie
        eval_result = anomaly_trainer.evaluate_with_anomaly_metrics(
            trainer, X_test, y_test
        )
        
        if eval_result.success:
            metrics = eval_result.data
            anomaly_metrics = metrics['anomaly_specific']
            
            print(f"\nMétriques standard:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
            
            print(f"\nMétriques spécifiques anomalie:")
            print(f"  F1 ajusté (difficulté): {anomaly_metrics['adjusted_f1_score']:.4f}")
            print(f"  Taux faux positifs: {anomaly_metrics['false_positive_rate']:.4f}")
            print(f"  Taux faux négatifs: {anomaly_metrics['false_negative_rate']:.4f}")
            
            print(f"\nRecommandations:")
            for rec in anomaly_metrics['recommendations']:
                print(f"  - {rec}")