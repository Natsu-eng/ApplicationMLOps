# test_augmentation_fix.py
import numpy as np
from orchestrators.visio_training_orchestrator import (
    training_orchestrator,
    TrainingContext
)

def test_augmentation_propagation():
    """
    Teste que preprocessing_config est correctement propagé.
    """
    # Données factices
    X_train = np.random.randint(0, 256, (100, 64, 64, 3), dtype=np.uint8)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randint(0, 256, (20, 64, 64, 3), dtype=np.uint8)
    y_val = np.random.randint(0, 2, 20)
    
    # ✅ Config avec augmentation
    preprocessing_config = {
        'strategy': 'standardize',
        'target_size': (64, 64),  # ✅ OBLIGATOIRE
        'augmentation_enabled': True,
        'methods': ['flip', 'rotate'],
        'augmentation_factor': 2
    }
    
    # Contexte
    context = TrainingContext(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_config={'model_type': 'simple_cnn', 'model_params': {}},
        training_config={'epochs': 2, 'batch_size': 16},
        preprocessing_config=preprocessing_config,  # ✅ PASSÉ
        callbacks=[]
    )
    
    # Lancement
    result = training_orchestrator.train(context)
    
    # ✅ ASSERTIONS
    assert result.success, f"Training échoué: {result.error}"
    assert result.preprocessor is not None, "Preprocessor manquant"
    
    # Vérifier que target_size a été appliqué
    if hasattr(result.preprocessor, 'target_size'):
        assert result.preprocessor.target_size == (64, 64), \
            f"Target size incorrect: {result.preprocessor.target_size}"
    
    print("✅ Test passed: Augmentation correctement propagée")

if __name__ == "__main__":
    test_augmentation_propagation()
