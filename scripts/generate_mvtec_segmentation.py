"""
🎯 Générateur Dataset Segmentation MVTec AD
Paires (image, masque) pour U-Net / Semantic Segmentation

Usage:
    python generate_mvtec_segmentation.py --category leather
    python generate_mvtec_segmentation.py --category capsule --split 0.8 0.2

Datasets avec ground-truth disponibles:
- bottle, cable, capsule, carpet, grid, hazelnut, leather,
  metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

Structure requise:
mvtec_ad/{category}/
    test/
        {defect_type}/
            {img}.png
    ground_truth/
        {defect_type}/
            {img}_mask.png
"""

import shutil
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import random
from PIL import Image
import numpy as np

def validate_segmentation_data(mvtec_dir: Path) -> Tuple[bool, str, List[Tuple[Path, Path]]]:
    """
    Valide la présence de test/ et ground_truth/, retourne les paires valides
    """
    if not mvtec_dir.exists():
        return False, f"Dossier introuvable: {mvtec_dir}", []
    
    test_dir = mvtec_dir / "test"
    gt_dir = mvtec_dir / "ground_truth"
    
    if not test_dir.exists():
        return False, "Dossier test/ manquant", []
    
    if not gt_dir.exists():
        return False, "Dossier ground_truth/ manquant (segmentation non disponible)", []
    
    # Collecte des paires (image, masque)
    pairs = []
    extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    for gt_defect_dir in gt_dir.iterdir():
        if not gt_defect_dir.is_dir():
            continue
        
        defect_type = gt_defect_dir.name
        
        # Masques disponibles
        mask_files = [f for f in gt_defect_dir.glob('*_mask.*') if f.suffix.lower() in extensions]
        
        for mask_path in mask_files:
            # Trouver l'image correspondante
            img_name = mask_path.name.replace('_mask', '')
            img_path = test_dir / defect_type / img_name
            
            if img_path.exists():
                pairs.append((img_path, mask_path))
    
    if len(pairs) == 0:
        return False, "Aucune paire (image, masque) valide trouvée", []
    
    return True, f"{len(pairs)} paires trouvées", pairs

def validate_mask_quality(mask_path: Path) -> Tuple[bool, Dict]:
    """
    Vérifie qu'un masque est binaire et non vide
    """
    try:
        with Image.open(mask_path) as img:
            mask_array = np.array(img.convert('L'))
            
            unique_values = np.unique(mask_array)
            n_white_pixels = np.sum(mask_array > 128)
            
            stats = {
                "unique_values": len(unique_values),
                "white_pixels": int(n_white_pixels),
                "white_ratio": float(n_white_pixels / mask_array.size),
                "shape": mask_array.shape
            }
            
            # Validation
            if n_white_pixels == 0:
                return False, stats  # Masque vide
            
            if n_white_pixels == mask_array.size:
                return False, stats  # Tout blanc (invalide)
            
            return True, stats
            
    except Exception as e:
        return False, {"error": str(e)}

def split_segmentation_data(
    pairs: List[Tuple[Path, Path]], 
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List, List]:
    """
    Split train/val pour segmentation
    """
    random.seed(seed)
    
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    n_train = int(len(shuffled) * train_ratio)
    
    train_set = shuffled[:n_train]
    val_set = shuffled[n_train:]
    
    return train_set, val_set

def create_segmentation_dataset(
    mvtec_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    seed: int = 42,
    validate_masks: bool = True
):
    """
    Crée le dataset de segmentation
    
    Structure de sortie:
    output_dir/
        train/
            images/
                defect_type_img001.png
                ...
            masks/
                defect_type_img001_mask.png
                ...
        val/
            images/
            masks/
        metadata.json
        README.txt
    """
    print(f"\n🎯 Création dataset segmentation depuis: {mvtec_dir.name}")
    
    # Validation
    is_valid, msg, pairs = validate_segmentation_data(mvtec_dir)
    if not is_valid:
        print(f"❌ {msg}")
        return
    
    print(f"✅ {msg}")
    
    # Validation qualité masques
    if validate_masks:
        print(f"\n🔍 Validation qualité des masques...")
        valid_pairs = []
        invalid_count = 0
        
        for img_path, mask_path in pairs:
            is_valid_mask, stats = validate_mask_quality(mask_path)
            
            if is_valid_mask:
                valid_pairs.append((img_path, mask_path))
            else:
                invalid_count += 1
                print(f"   ⚠️  Masque invalide: {mask_path.name} - {stats}")
        
        print(f"✅ Masques valides: {len(valid_pairs)} / {len(pairs)}")
        if invalid_count > 0:
            print(f"⚠️  Masques rejetés: {invalid_count}")
        
        pairs = valid_pairs
    
    if len(pairs) == 0:
        print("❌ Aucune paire valide après validation")
        return
    
    # Nettoyage
    if output_dir.exists():
        print(f"🗑️  Suppression ancien dataset: {output_dir}")
        shutil.rmtree(output_dir)
    
    # Statistiques par type de défaut
    defect_types = Counter([img_path.parent.name for img_path, _ in pairs])
    
    print(f"\n📊 Répartition par type de défaut:")
    for defect_type, count in sorted(defect_types.items()):
        print(f"   {defect_type:20s}: {count:4d} paires")
    
    # Split
    print(f"\n🔀 Split données (train={train_ratio}, val={1-train_ratio})...")
    train_set, val_set = split_segmentation_data(pairs, train_ratio, seed)
    
    print(f"   Train: {len(train_set)} paires")
    print(f"   Val  : {len(val_set)} paires")
    
    # Création structure
    for split_name in ["train", "val"]:
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "masks").mkdir(parents=True, exist_ok=True)
    
    # Copie des fichiers
    print(f"\n📁 Copie des fichiers vers {output_dir}...")
    
    copied_train = 0
    copied_val = 0
    
    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        for img_path, mask_path in split_data:
            # Nommage cohérent: defect_type_imgname
            defect_type = img_path.parent.name
            img_name = img_path.stem
            
            new_img_name = f"{defect_type}_{img_name}.png"
            new_mask_name = f"{defect_type}_{img_name}_mask.png"
            
            # Copier image
            img_dest = output_dir / split_name / "images" / new_img_name
            shutil.copy2(img_path, img_dest)
            
            # Copier masque
            mask_dest = output_dir / split_name / "masks" / new_mask_name
            shutil.copy2(mask_path, mask_dest)
            
            if split_name == "train":
                copied_train += 1
            else:
                copied_val += 1
    
    print(f"✅ Copie terminée: {copied_train} train + {copied_val} val")
    
    # Métadonnées
    metadata = {
        "source_dataset": mvtec_dir.name,
        "task_type": "semantic_segmentation",
        "n_classes": 2,  # Background (0) + Défaut (1)
        "class_names": ["background", "defect"],
        "defect_types": list(defect_types.keys()),
        "splits": {
            "train": {
                "n_pairs": len(train_set),
                "defect_distribution": dict(Counter([img.parent.name for img, _ in train_set]))
            },
            "val": {
                "n_pairs": len(val_set),
                "defect_distribution": dict(Counter([img.parent.name for img, _ in val_set]))
            }
        },
        "split_ratio": {
            "train": train_ratio,
            "val": 1 - train_ratio
        },
        "random_seed": seed,
        "total_pairs": len(pairs),
        "validation": {
            "mask_quality_check": validate_masks,
            "rejected_masks": len(pairs) - len(valid_pairs) if validate_masks else 0
        }
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Métadonnées: {metadata_path}")
    
    # README
    readme_content = f"""
# {mvtec_dir.name.upper()} - Segmentation Sémantique des Défauts

## 📋 Description
Dataset de segmentation sémantique binaire (background vs défaut).
Paires (image, masque) pour entraînement U-Net / SegNet.

## 🏷️ Classes
0. **Background** (pixels noirs dans le masque)
1. **Défaut** (pixels blancs dans le masque)

## 📊 Statistiques
- **Total paires**: {len(pairs)}
- **Train**: {len(train_set)} paires ({train_ratio*100:.0f}%)
- **Val**: {len(val_set)} paires ({(1-train_ratio)*100:.0f}%)

### Types de défauts présents:
"""
    for defect_type, count in sorted(defect_types.items()):
        readme_content += f"- {defect_type}: {count} paires\n"
    
    readme_content += f"""

## 📁 Structure
```
{output_dir.name}/
    train/
        images/
            defect_img001.png
            defect_img002.png
            ...
        masks/
            defect_img001_mask.png  (binaire: 0=fond, 255=défaut)
            defect_img002_mask.png
            ...
    val/
        images/
        masks/
    metadata.json
    README.txt
```

## 🎯 Utilisation avec DataLab Pro

### Chargement manuel (PyTorch)
```python
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.img_dir = Path(root_dir) / split / 'images'
        self.mask_dir = Path(root_dir) / split / 'masks'
        
        self.images = sorted(self.img_dir.glob('*.png'))
        self.masks = sorted(self.mask_dir.glob('*_mask.png'))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')
        
        # Transforms...
        return img, mask

# Usage
train_dataset = SegmentationDataset('{output_dir}', split='train')
val_dataset = SegmentationDataset('{output_dir}', split='val')
```

### Entraînement U-Net
1. Charger les paires (image, masque)
2. Resize à 256×256 ou 512×512
3. Normaliser images [0, 1]
4. Masques binaires {{0, 1}}
5. Loss: BCE ou Dice
6. Métriques: IoU, Dice Score

## 🔧 Génération
- Source: MVTec AD {mvtec_dir.name}
- Script: generate_mvtec_segmentation.py
- Random seed: {seed}
- Validation masques: {validate_masks}
"""
    
    readme_path = output_dir / "README.txt"
    readme_path.write_text(readme_content, encoding='utf-8')
    
    print(f"✅ README: {readme_path}")
    
    # Résumé final
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║        DATASET SEGMENTATION CRÉÉ AVEC SUCCÈS ! 🎉           ║
╚══════════════════════════════════════════════════════════════╝

📂 Chemin        : {output_dir}
🏷️  Classes       : 2 (background, defect)
📦 Total paires  : {len(pairs)}

📊 Splits:
   • Train: {len(train_set):4d} paires ({train_ratio*100:5.1f}%)
   • Val  : {len(val_set):4d} paires ({(1-train_ratio)*100:5.1f}%)

📁 Structure:
   • train/images/ : {len(train_set)} images
   • train/masks/  : {len(train_set)} masques binaires
   • val/images/   : {len(val_set)} images
   • val/masks/    : {len(val_set)} masques binaires

⚠️  NOTE: Segmentation nécessite un loader custom
   Voir README.txt pour exemple PyTorch Dataset

✅ Prêt pour entraînement U-Net !
""")

def main():
    parser = argparse.ArgumentParser(
        description="Générateur dataset segmentation MVTec AD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Leather
  python generate_mvtec_segmentation.py --category leather
  
  # Capsule avec split personnalisé
  python generate_mvtec_segmentation.py --category capsule --split 0.8 0.2
  
  # Sans validation masques (plus rapide)
  python generate_mvtec_segmentation.py --category bottle --no-validate

Datasets avec ground-truth:
  bottle, cable, capsule, carpet, grid, hazelnut, leather,
  metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper
        """
    )
    
    parser.add_argument(
        '--category',
        type=str,
        required=True,
        help='Catégorie MVTec AD (ex: leather, capsule)'
    )
    
    parser.add_argument(
        '--split',
        type=float,
        nargs=2,
        default=[0.8, 0.2],
        metavar=('TRAIN', 'VAL'),
        help='Ratios train/val (défaut: 0.8 0.2)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (défaut: 42)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Dossier de sortie (défaut: {category}_segmentation)'
    )
    
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Désactiver validation qualité masques'
    )
    
    args = parser.parse_args()
    
    # Validation splits
    train_ratio, val_ratio = args.split
    if not (0.99 <= sum(args.split) <= 1.01):
        print(f"❌ Erreur: Les ratios doivent sommer à 1.0 (actuel: {sum(args.split)})")
        return 1
    
    # Chemins
    project_root = Path(__file__).parent.parent
    mvtec_dir = project_root / "src" / "data" / "mvtec_ad" / args.category
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "src" / "data" / "mvtec_ad" / f"{args.category}_segmentation"
    
    # Exécution
    try:
        create_segmentation_dataset(
            mvtec_dir=mvtec_dir,
            output_dir=output_dir,
            train_ratio=train_ratio,
            seed=args.seed,
            validate_masks=not args.no_validate
        )
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())