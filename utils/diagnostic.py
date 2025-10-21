from datetime import datetime
from typing import Any, Dict

import numpy as np

from src.models.computer_vision.anomaly_detection.autoencoders import ConvAutoEncoder, DenoisingAutoEncoder, VariationalAutoEncoder
from src.models.computer_vision.classification.cnn_models import CustomResNet, SimpleCNN
from src.models.computer_vision.classification.transfer_learning import TransferLearningModel
from src.models.computer_vision_training import ComputerVisionTrainer, ModelConfig, ModelType, TrainingConfig, example_basic_training, test_no_data_leakage
from utils.callbacks import LoggingCallback

# ==========================
# DIAGNOSTIC ET VÉRIFICATION
# ==========================

def check_models_availability() -> Dict[str, Any]:
    """
    Vérifie quels modèles sont disponibles.
    
    Returns:
        Dict avec statut de chaque type de modèle
    """
    status = {
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }
    
    # Test SimpleCNN
    try:
        if SimpleCNN is not None:
            test_model = SimpleCNN(input_channels=3, num_classes=2)
            status['models']['SimpleCNN'] = {
                'available': True,
                'params': sum(p.numel() for p in test_model.parameters())
            }
        else:
            status['models']['SimpleCNN'] = {'available': False, 'reason': 'Not imported'}
    except Exception as e:
        status['models']['SimpleCNN'] = {'available': False, 'reason': str(e)}
    
    # Test CustomResNet
    try:
        if CustomResNet is not None:
            test_model = CustomResNet(input_channels=3, num_classes=2)
            status['models']['CustomResNet'] = {
                'available': True,
                'params': sum(p.numel() for p in test_model.parameters())
            }
        else:
            status['models']['CustomResNet'] = {'available': False, 'reason': 'Not imported'}
    except Exception as e:
        status['models']['CustomResNet'] = {'available': False, 'reason': str(e)}
    
    # Test TransferLearning
    try:
        if TransferLearningModel is not None:
            status['models']['TransferLearningModel'] = {'available': True}
        else:
            status['models']['TransferLearningModel'] = {'available': False, 'reason': 'Not imported'}
    except Exception as e:
        status['models']['TransferLearningModel'] = {'available': False, 'reason': str(e)}
    
    # Test Autoencoders
    for ae_name, ae_class in [
        ('ConvAutoEncoder', ConvAutoEncoder),
        ('VariationalAutoEncoder', VariationalAutoEncoder),
        ('DenoisingAutoEncoder', DenoisingAutoEncoder)
    ]:
        try:
            if ae_class is not None:
                test_model = ae_class(input_channels=3, latent_dim=128)
                status['models'][ae_name] = {
                    'available': True,
                    'params': sum(p.numel() for p in test_model.parameters())
                }
            else:
                status['models'][ae_name] = {'available': False, 'reason': 'Not imported'}
        except Exception as e:
            status['models'][ae_name] = {'available': False, 'reason': str(e)}
    
    # Résumé
    available_count = sum(1 for m in status['models'].values() if m.get('available', False))
    total_count = len(status['models'])
    
    status['summary'] = {
        'available': available_count,
        'total': total_count,
        'percentage': (available_count / total_count * 100) if total_count > 0 else 0
    }
    
    return status


def print_models_status():
    """Affiche le statut des modèles de manière lisible"""
    status = check_models_availability()
    
    print("\n" + "="*70)
    print("STATUT DES MODÈLES DISPONIBLES")
    print("="*70)
    print(f"\nTimestamp: {status['timestamp']}")
    print(f"\nRésumé: {status['summary']['available']}/{status['summary']['total']} "
          f"modèles disponibles ({status['summary']['percentage']:.1f}%)\n")
    
    print("-"*70)
    for model_name, model_info in status['models'].items():
        if model_info.get('available'):
            params = model_info.get('params', 'N/A')
            if isinstance(params, int):
                params_str = f"{params:,} paramètres"
            else:
                params_str = "paramètres inconnus"
            print(f"✅ {model_name:<30} {params_str}")
        else:
            reason = model_info.get('reason', 'Unknown')
            print(f"❌ {model_name:<30} Raison: {reason}")
    
    print("="*70 + "\n")
    
    # Recommandations
    if status['summary']['available'] == 0:
        print("⚠️  AUCUN modèle réel disponible - Le pipeline utilisera des placeholders")
        print("   Pour utiliser les vrais modèles, vérifiez que les modules sont importables:")
        print("   - src.models.computer_vision.classification.cnn_models")
        print("   - src.models.computer_vision.classification.transfer_learning")
        print("   - src.models.computer_vision.anomaly_detection.autoencoders")
    elif status['summary']['available'] < status['summary']['total']:
        print(f"⚠️  Seulement {status['summary']['available']}/{status['summary']['total']} "
              f"modèles disponibles")
        print("   Certains modèles utiliseront des placeholders")
    else:
        print("✅ Tous les modèles sont disponibles!")
    
    print()


# ===========================
# EXEMPLES AVEC VRAIS MODÈLES
# ===========================

def example_with_real_models():
    """Exemple utilisant les vrais modèles si disponibles"""
    
    print("\n" + "="*70)
    print("EXEMPLE: UTILISATION DES VRAIS MODÈLES")
    print("="*70 + "\n")
    
    # Vérification des modèles
    print_models_status()
    
    # Configuration selon disponibilité
    available_models = check_models_availability()
    
    if available_models['models'].get('SimpleCNN', {}).get('available'):
        model_type = ModelType.SIMPLE_CNN
        print(f"✅ Utilisation de SimpleCNN (modèle réel)\n")
    else:
        model_type = ModelType.SIMPLE_CNN  # Utilisera le placeholder
        print(f"⚠️  SimpleCNN non disponible, utilisation placeholder\n")
    
    # Configuration
    model_config = ModelConfig(
        model_type=model_type,
        num_classes=2,
        input_channels=3,
        dropout_rate=0.5
    )
    
    training_config = TrainingConfig(
        epochs=10,
        batch_size=16,
        learning_rate=1e-3,
        deterministic=True
    )
    
    # Données
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
        print(f"Best F1: {max(trainer.history.get('val_f1', [0])):.4f}")
        print(f"Modèle utilisé: {trainer.model.__class__.__name__}")
    else:
        print(f"\n❌ Échec: {result.error}")


def run_all_tests():
    """Version simplifiée pour la démo"""
    print("🧪 Running basic tests...")
    test_no_data_leakage()
    print("✅ Basic tests passed")

def production_example_complete():
    """Version simplifiée"""
    print("🚀 Running production example...")
    example_basic_training()