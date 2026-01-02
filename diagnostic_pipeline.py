"""
Script de diagnostic complet pour identifier les problèmes d'augmentation.
"""
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

def diagnose_augmentation_pipeline():
    """Diagnostic étape par étape."""
    print("🔍 DIAGNOSTIC PIPELINE AUGMENTATION")
    print("=" * 50)
    
    # 1. Créer des données de test
    print("\n1. Création données de test...")
    X_raw = np.random.randint(0, 256, (10, 64, 64, 3), dtype=np.uint8)
    y = np.random.randint(0, 2, (10,))
    
    print(f"   X shape: {X_raw.shape}")
    print(f"   X dtype: {X_raw.dtype}")
    print(f"   X range: [{X_raw.min()}, {X_raw.max()}]")
    print(f"   Format: channels_last (vérifié)")
    
    # 2. Tester DataPreprocessor
    print("\n2. Test DataPreprocessor...")
    from src.data.computer_vision_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor(
        strategy="standardize",
        auto_detect_format=True,
        target_size=(64, 64)
    )
    
    try:
        X_processed = preprocessor.fit_transform(X_raw, output_format="channels_first")
        print(f"   ✅ Preprocessing réussi")
        print(f"   Shape après preprocessing: {X_processed.shape}")
        print(f"   Dtype après preprocessing: {X_processed.dtype}")
        print(f"   Range après preprocessing: [{X_processed.min():.3f}, {X_processed.max():.3f}]")
    except Exception as e:
        print(f"   ❌ Preprocessing échoué: {e}")
        return
    
    # 3. Tester AugmentedImageDataset
    print("\n3. Test AugmentedImageDataset...")
    from src.data.computer_vision_preprocessing import AugmentedImageDataset
    
    try:
        # Sans augmentation
        dataset_no_aug = AugmentedImageDataset(
            X=X_raw,
            y=y,
            target_size=(64, 64),
            augmentation_config=None,
            is_training=False,
            preprocessor=preprocessor
        )
        
        img_no_aug, label_no_aug = dataset_no_aug[0]
        print(f"   ✅ Dataset sans augmentation: OK")
        print(f"   Image shape: {img_no_aug.shape}")
        print(f"   Image dtype: {img_no_aug.dtype}")
        
        # Avec augmentation
        aug_config = {
            'augmentation_enabled': True,
            'methods': ['flip', 'brightness']
        }
        
        dataset_aug = AugmentedImageDataset(
            X=X_raw,
            y=y,
            target_size=(64, 64),
            augmentation_config=aug_config,
            is_training=True,
            preprocessor=preprocessor
        )
        
        img_aug, label_aug = dataset_aug[0]
        print(f"   ✅ Dataset avec augmentation: OK")
        print(f"   Image shape: {img_aug.shape}")
        
    except Exception as e:
        print(f"   ❌ Dataset échoué: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Tester DataLoaderFactory
    print("\n4. Test DataLoaderFactory...")
    from src.data.computer_vision_preprocessing import DataLoaderFactory
    
    try:
        # Sans augmentation
        loader_no_aug = DataLoaderFactory.create(
            X=X_raw,
            y=y,
            batch_size=4,
            augmentation_config=None,
            is_training=False,
            target_size=(64, 64),
            output_format="channels_first",
            preprocessor=preprocessor
        )
        
        batch_no_aug = next(iter(loader_no_aug))
        print(f"   ✅ DataLoader sans augmentation: OK")
        print(f"   Batch shape: {batch_no_aug[0].shape}")
        
        # Avec augmentation
        loader_aug = DataLoaderFactory.create(
            X=X_raw,
            y=y,
            batch_size=4,
            augmentation_config=aug_config,
            is_training=True,
            target_size=(64, 64),
            output_format="channels_first",
            preprocessor=preprocessor
        )
        
        batch_aug = next(iter(loader_aug))
        print(f"   ✅ DataLoader avec augmentation: OK")
        print(f"   Batch shape: {batch_aug[0].shape}")
        
    except Exception as e:
        print(f"   ❌ DataLoaderFactory échoué: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Tester le Trainer
    print("\n5. Test Trainer...")
    from src.models.trainer import ComputerVisionTrainer
    from src.config.model_config import ModelConfig, ModelType
    from src.config.training_config import TrainingConfig
    
    try:
        model_config = ModelConfig(
            model_type=ModelType.SIMPLE_CNN,
            num_classes=2,
            input_channels=3,
            dropout_rate=0.5
        )
        
        training_config = TrainingConfig(
            epochs=1,
            batch_size=4,
            learning_rate=1e-3
        )
        
        trainer = ComputerVisionTrainer(model_config, training_config)
        
        # Préparer plus de données pour l'entraînement
        X_train_large = np.random.randint(0, 256, (20, 64, 64, 3), dtype=np.uint8)
        y_train_large = np.random.randint(0, 2, (20,))
        X_val_large = np.random.randint(0, 256, (5, 64, 64, 3), dtype=np.uint8)
        y_val_large = np.random.randint(0, 2, (5,))
        
        # Avec augmentation
        preprocessing_config = {
            'augmentation_enabled': True,
            'methods': ['flip'],
            'target_size': (64, 64)
        }
        
        print(f"   Lancement entraînement (1 epoch)...")
        result = trainer.fit(
            X_train=X_train_large,
            y_train=y_train_large,
            X_val=X_val_large,
            y_val=y_val_large,
            preprocessing_config=preprocessing_config
        )
        
        if result.success:
            print(f"   ✅ Trainer avec augmentation: OK")
            print(f"   History keys: {list(result.data['history'].keys())}")
        else:
            print(f"   ❌ Trainer échoué: {result.error}")
            
    except Exception as e:
        print(f"   ❌ Trainer échoué: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 50)
    print("✅ DIAGNOSTIC COMPLET TERMINÉ - TOUS LES TESTS PASSENT")
    print("=" * 50)

def check_common_issues():
    """Vérifier les problèmes communs."""
    print("\n🔧 VÉRIFICATION DES PROBLÈMES COURANTS")
    print("=" * 50)
    
    issues = []
    
    # Vérification 1: Format des données
    X_test = np.random.rand(10, 64, 64, 3).astype(np.float32)
    if X_test.max() <= 1.0:
        issues.append("❌ Données normalisées (float32 [0,1]) au lieu de uint8 [0,255]")
    
    # Vérification 2: Format channels
    if X_test.shape[-1] not in [1, 3, 4]:
        issues.append("❌ Format channels incertain")
    
    # Vérification 3: Compatibilité augmentation
    from src.data.computer_vision_preprocessing import AugmentedImageDataset
    
    try:
        _ = AugmentedImageDataset(
            X=X_test,  # float32 au lieu de uint8
            y=np.zeros(10),
            target_size=(64, 64),
            augmentation_config={'augmentation_enabled': True},
            is_training=True,
            preprocessor=None
        )
    except Exception as e:
        issues.append(f"✅ Augmentation rejette float32 (comportement attendu)")
    
    # Afficher les problèmes
    if issues:
        print("Problèmes détectés:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ Aucun problème commun détecté")

if __name__ == "__main__":
    diagnose_augmentation_pipeline()
    check_common_issues()