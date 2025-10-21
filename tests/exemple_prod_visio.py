# ==========================
# EXEMPLE COMPLET PRODUCTION
# ==========================

from pathlib import Path
import numpy as np
from src.models.computer_vision_training import ModelConfig, ModelPersistence, ModelType, ProductionMonitor, ProductionPipeline, TrainingConfig
from utils.callbacks import CheckpointCallback, LoggingCallback


def production_example_complete():
    """
    Exemple complet montrant toutes les features en production.
    """
    
    print("\n" + "="*70)
    print("EXEMPLE COMPLET - PIPELINE PRODUCTION CV")
    print("="*70 + "\n")
    
    # 1. Configuration
    print("1️⃣ Configuration...")
    model_config = ModelConfig(
        model_type=ModelType.SIMPLE_CNN,
        num_classes=3,
        input_channels=3,
        dropout_rate=0.5
    )
    
    training_config = TrainingConfig(
        epochs=20,
        batch_size=32,
        learning_rate=1e-3,
        use_class_weights=True,
        early_stopping_patience=5,
        deterministic=True,
        seed=42
    )
    
    # 2. Génération données (simulation)
    print("2️⃣ Génération des données...")
    np.random.seed(42)
    
    X_train = np.random.randn(300, 64, 64, 3).astype(np.float32)
    y_train = np.random.randint(0, 3, 300)
    
    X_val = np.random.randn(60, 64, 64, 3).astype(np.float32)
    y_val = np.random.randint(0, 3, 60)
    
    X_test = np.random.randn(100, 64, 64, 3).astype(np.float32)
    y_test = np.random.randint(0, 3, 100)
    
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 3. Pipeline complet
    print("3️⃣ Lancement du pipeline...")
    
    callbacks = [
        LoggingCallback(log_every_n_epochs=5),
        CheckpointCallback(checkpoint_dir=Path("./checkpoints"), save_best_only=True)
    ]
    
    pipeline = ProductionPipeline(
        model_config=model_config,
        training_config=training_config,
        callbacks=callbacks
    )
    
    result = pipeline.train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        save_model=True
    )
    
    if not result.success:
        print(f"❌ Pipeline échoué: {result.error}")
        return
    
    # 4. Résultats
    print("\n4️⃣ Résultats:")
    test_metrics = result.data['test_metrics']
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test F1: {test_metrics['f1']:.4f}")
    print(f"   Test Precision: {test_metrics['precision']:.4f}")
    print(f"   Test Recall: {test_metrics['recall']:.4f}")
    
    # 5. Monitoring en production
    print("\n5️⃣ Setup monitoring production...")
    
    monitor = ProductionMonitor(
        model=result.data['model'],
        preprocessor=result.data['preprocessor'],
        baseline_metrics={'accuracy': test_metrics['accuracy'], 'f1': test_metrics['f1']}
    )
    
    # Simulation de prédictions en production
    X_prod = np.random.randn(50, 64, 64, 3).astype(np.float32)
    y_prod = np.random.randint(0, 3, 50)
    
    monitor_result = monitor.predict_and_monitor(X_prod, y_prod)
    
    if monitor_result.success:
        if monitor_result.data.get('alerts'):
            print(f"   ⚠️ Alertes: {monitor_result.data['alerts']}")
        else:
            print("   ✅ Aucune alerte, modèle performant")
        
        report = monitor.get_monitoring_report()
        print(f"   Total prédictions: {report['total_predictions']}")
    
    print("\n6️⃣ Sauvegarde et chargement...")
    model_path = result.data.get('model_path')
    if model_path:
        print(f"   Modèle sauvegardé: {model_path}")
        
        # Test de chargement
        load_result = ModelPersistence.load(Path(model_path))
        if load_result.success:
            print("   ✅ Modèle rechargé avec succès")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLET TERMINÉ AVEC SUCCÈS")
    print("="*70 + "\n")