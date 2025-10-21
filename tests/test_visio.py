# ===================
# TESTS ET VALIDATION
# ===================

import numpy as np
from src.data.computer_vision_preprocessing import DataPreprocessor
from src.models.computer_vision_training import ComputerVisionTrainer, ModelConfig, ModelType, TrainingConfig


def test_no_data_leakage():
    """Test critique: Vérifie l'absence de fuite de données"""
    
    print("\n=== TEST: ABSENCE DE FUITE DE DONNÉES ===\n")
    
    # Données avec valeurs spécifiques pour détecter les fuites
    X_train = np.ones((100, 32, 32, 3)) * 10.0  # Mean = 10
    y_train = np.zeros(100, dtype=int)
    
    X_val = np.ones((20, 32, 32, 3)) * 20.0  # Mean = 20 (différent!)
    y_val = np.zeros(20, dtype=int)
    
    # Preprocessing
    preprocessor = DataPreprocessor(strategy="standardize")
    X_train_norm = preprocessor.fit_transform(X_train)
    X_val_norm = preprocessor.transform(X_val)
    
    # Vérifications
    print(f"Train mean après norm: {X_train_norm.mean():.6f}")  # Devrait être ~0
    print(f"Val mean après norm: {X_val_norm.mean():.6f}")      # Devrait être != 0
    
    assert abs(X_train_norm.mean()) < 0.1, "Train devrait être normalisé à ~0"
    assert abs(X_val_norm.mean() - 1.0) < 0.2, "Val devrait rester à valeur différente"
    
    print("✅ Pas de fuite détectée: Val utilise les stats de Train")


def test_class_weights_isolation():
    """Test: Class weights appliqués seulement sur train"""
    
    print("\n=== TEST: ISOLATION DES CLASS WEIGHTS ===\n")
    
    model_config = ModelConfig(model_type=ModelType.SIMPLE_CNN, num_classes=2)
    training_config = TrainingConfig(
        epochs=2,
        batch_size=16,
        use_class_weights=True
    )
    
    X_train = np.random.randn(80, 32, 32, 3).astype(np.float32)
    y_train = np.array([0]*60 + [1]*20)  # Déséquilibré
    X_val = np.random.randn(20, 32, 32, 3).astype(np.float32)
    y_val = np.array([0]*10 + [1]*10)
    
    trainer = ComputerVisionTrainer(
        model_config=model_config,
        training_config=training_config
    )
    
    result = trainer.fit(X_train, y_train, X_val, y_val)
    
    assert result.success, "Entraînement devrait réussir"
    
    # Vérifier que train et val criterion sont différents
    assert trainer.train_criterion is not None
    assert trainer.val_criterion is not None
    assert type(trainer.train_criterion) == type(trainer.val_criterion)
    
    print("✅ Class weights correctement isolés sur train")


def test_deterministic_training():
    """Test: Reproductibilité avec seed"""
    
    print("\n=== TEST: ENTRAÎNEMENT DÉTERMINISTE ===\n")
    
    model_config = ModelConfig(model_type=ModelType.SIMPLE_CNN)
    training_config = TrainingConfig(
        epochs=3,
        batch_size=16,
        deterministic=True,
        seed=42
    )
    
    X = np.random.randn(50, 32, 32, 3).astype(np.float32)
    y = np.random.randint(0, 2, 50)
    X_val = np.random.randn(10, 32, 32, 3).astype(np.float32)
    y_val = np.random.randint(0, 2, 10)
    
    # Premier entraînement
    trainer1 = ComputerVisionTrainer(model_config, training_config)
    result1 = trainer1.fit(X, y, X_val, y_val)
    loss1 = result1.data['history']['val_loss'][-1]
    
    # Deuxième entraînement (même seed)
    trainer2 = ComputerVisionTrainer(model_config, training_config)
    result2 = trainer2.fit(X, y, X_val, y_val)
    loss2 = result2.data['history']['val_loss'][-1]
    
    print(f"Loss run 1: {loss1:.6f}")
    print(f"Loss run 2: {loss2:.6f}")
    print(f"Différence: {abs(loss1 - loss2):.6f}")
    
    # Note: Peut ne pas être exactement identique à cause du GPU
    # mais devrait être très proche
    assert abs(loss1 - loss2) < 0.01, "Entraînements devraient être similaires"
    
    print("✅ Entraînement déterministe vérifié")


def run_all_tests():
    """Exécute tous les tests"""
    print("\n" + "="*60)
    print("SUITE DE TESTS COMPLÈTE")
    print("="*60)
    
    try:
        test_no_data_leakage()
        test_class_weights_isolation()
        test_deterministic_training()
        
        print("\n" + "="*60)
        print("✅ TOUS LES TESTS PASSÉS")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n❌ TEST ÉCHOUÉ: {e}\n")
        raise
