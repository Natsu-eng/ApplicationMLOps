# =======================
# EXEMPLES D'UTILISATION
# =======================

import numpy as np
from src.models.computer_vision.cross_validator import CrossValidator
from src.models.computer_vision_training import ComputerVisionTrainer, ModelConfig, ModelType, ProductionPipeline, TrainingConfig
from utils.callbacks import LoggingCallback


def example_basic_training():
    """Exemple d'utilisation basique"""
    
    print("=== EXEMPLE: ENTRAÎNEMENT BASIQUE ===\n")
    
    # Configuration
    model_config = ModelConfig(
        model_type=ModelType.SIMPLE_CNN,
        num_classes=2,
        input_channels=3
    )
    
    training_config = TrainingConfig(
        epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        use_class_weights=True,
        deterministic=True
    )
    
    # Données synthétiques
    X_train = np.random.randn(100, 64, 64, 3).astype(np.float32)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(20, 64, 64, 3).astype(np.float32)
    y_val = np.random.randint(0, 2, 20)
    
    # Entraînement
    trainer = ComputerVisionTrainer(
        model_config=model_config,
        training_config=training_config,
        callbacks=[LoggingCallback(log_every_n_epochs=2)]
    )
    
    result = trainer.fit(X_train, y_train, X_val, y_val)
    
    if result.success:
        print(f"\n✅ Entraînement réussi!")
        print(f"Best F1: {max(trainer.history['val_f1']):.4f}")
    else:
        print(f"\n❌ Échec: {result.error}")


def example_full_pipeline():
    """Exemple avec pipeline complet"""
    
    print("\n=== EXEMPLE: PIPELINE COMPLET ===\n")
    
    # Configuration
    model_config = ModelConfig(
        model_type=ModelType.SIMPLE_CNN,
        num_classes=3,
        input_channels=3
    )
    
    training_config = TrainingConfig(
        epochs=15,
        batch_size=32,
        learning_rate=1e-3,
        early_stopping_patience=5
    )
    
    # Données (train/val/test séparés)
    X_train = np.random.randn(200, 64, 64, 3).astype(np.float32)
    y_train = np.random.randint(0, 3, 200)
    X_val = np.random.randn(40, 64, 64, 3).astype(np.float32)
    y_val = np.random.randint(0, 3, 40)
    X_test = np.random.randn(50, 64, 64, 3).astype(np.float32)
    y_test = np.random.randint(0, 3, 50)
    
    # Pipeline
    pipeline = ProductionPipeline(
        model_config=model_config,
        training_config=training_config,
        callbacks=[LoggingCallback()]
    )
    
    result = pipeline.train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        save_model=True
    )
    
    if result.success:
        print(f"\n✅ Pipeline terminé!")
        print(f"Test Accuracy: {result.data['test_metrics']['accuracy']:.4f}")
        print(f"Test F1: {result.data['test_metrics']['f1']:.4f}")
        print(f"Modèle sauvegardé: {result.data['model_path']}")
    else:
        print(f"\n❌ Échec: {result.error}")


def example_cross_validation():
    """Exemple avec cross-validation"""
    
    print("\n=== EXEMPLE: CROSS-VALIDATION ===\n")
    
    model_config = ModelConfig(
        model_type=ModelType.SIMPLE_CNN,
        num_classes=2
    )
    
    training_config = TrainingConfig(
        epochs=5,
        batch_size=16
    )
    
    # Données
    X = np.random.randn(150, 64, 64, 3).astype(np.float32)
    y = np.random.randint(0, 2, 150)
    
    # Cross-validation
    cv = CrossValidator(
        n_splits=3,
        model_config=model_config,
        training_config=training_config
    )
    
    result = cv.cross_validate(X, y)
    
    if result.success:
        print(f"\n✅ Cross-validation terminée!")
        agg = result.data['aggregated_metrics']
        print(f"Accuracy: {agg['accuracy']['mean']:.4f} ± {agg['accuracy']['std']:.4f}")
        print(f"F1: {agg['f1']['mean']:.4f} ± {agg['f1']['std']:.4f}")
    else:
        print(f"\n❌ Échec: {result.error}")


if __name__ == "__main__":
    # Exécuter les exemples
    example_basic_training()
    example_full_pipeline()
    example_cross_validation()

