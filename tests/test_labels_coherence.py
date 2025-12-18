# test_labels_coherence.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explorations.image_exploration_plots import (
    detect_dataset_structure, 
    load_images_flexible
)

def test_bottle_supervised():
    """Test du dataset bottle_supervised créé"""
    dataset_path = Path(__file__).parent.parent / "src/data/mvtec_ad/bottle_supervised"
    
    # 1. Détection structure
    structure = detect_dataset_structure(str(dataset_path))
    print(f"\n📋 Structure détectée:")
    print(f"   Type: {structure['type']}")
    print(f"   Categories: {structure.get('categories')}")
    print(f"   class_to_idx: {structure.get('class_to_idx')}")
    
    # 2. Chargement images
    X, X_norm, y, y_train = load_images_flexible(str(dataset_path))
    
    print(f"\n📊 Résultats chargement:")
    print(f"   X.shape: {X.shape}")
    print(f"   y unique: {np.unique(y)}")
    print(f"   y_train: {y_train if y_train is not None else 'None'}")
    
    # 3. Vérification ordre classes
    from collections import Counter
    counts = Counter(y)
    print(f"\n🏷️ Distribution labels:")
    for label, count in sorted(counts.items()):
        class_name = structure['class_to_idx']
        # Inverser pour trouver le nom
        name = [k for k, v in class_name.items() if v == label][0]
        print(f"   Label {label} ({name}): {count} images")
    
    # 4. Validation
    assert structure['is_anomaly_supervised'], "❌ Pas détecté comme anomaly supervised"
    assert structure['class_to_idx']['normal'] == 0, "❌ Normal n'est pas label 0"
    assert structure['class_to_idx']['defect'] == 1, "❌ Defect n'est pas label 1"
    
    print("\n✅ TOUS LES TESTS PASSÉS !")

if __name__ == "__main__":
    import numpy as np
    test_bottle_supervised()