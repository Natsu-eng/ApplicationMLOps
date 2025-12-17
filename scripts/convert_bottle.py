# test_bottle_supervised.py
from pathlib import Path
import shutil
import os
import time

def create_bottle_supervised_dataset():
    project_root = Path(__file__).parent.parent
    mvtec_dir = project_root / "src" / "data" / "mvtec_ad" / "bottle"
    output_dir = project_root / "src" / "data" / "mvtec_ad" / "bottle_supervised"
    
    if not mvtec_dir.exists():
        print(f"❌ Dossier MVTec non trouvé : {mvtec_dir}")
        return
    
    # Supprimer l'ancien si existe
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Créer dans l'ordre garanti : normal avant defect
    (output_dir / "normal").mkdir(parents=True, exist_ok=True)
    (output_dir / "defect").mkdir(parents=True, exist_ok=True)
    
    normal_count = 0
    defect_count = 0
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.tif')
    
    # Normales
    for split in ["train", "test"]:
        good_path = mvtec_dir / split / "good"
        if good_path.exists():
            for img in good_path.glob('*'):
                if img.suffix.lower() in extensions:
                    shutil.copy(img, output_dir / "normal" / f"{split}_{img.name}")
                    normal_count += 1
    
    # Défectueuses
    test_path = mvtec_dir / "test"
    if test_path.exists():
        for defect_type in test_path.iterdir():
            if defect_type.is_dir() and defect_type.name != "good":
                for img in defect_type.glob('*'):
                    if img.suffix.lower() in extensions:
                        shutil.copy(img, output_dir / "defect" / f"{defect_type.name}_{img.name}")
                        defect_count += 1
    
    # README
    readme = f"""
# MVTec AD Bottle - Mode Supervisé (Test Dashboard)

Structure garantie :
- normal/  → label 0
- defect/  → label 1

Images :
- Normal     : {normal_count}
- Défectueuse: {defect_count}
- Total      : {normal_count + defect_count}

Ce dataset testera parfaitement :
→ Détection ANOMALY_DETECTION
→ Affichage "Normal" / "Défectueuse" dans le dashboard
→ Robustesse à l'ordre des classes
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")
    
    print(f"✅ Dataset créé avec succès !")
    print(f"   Chemin : {output_dir}")
    print(f"   Normal  : {normal_count} images (label 0)")
    print(f"   Defect  : {defect_count} images (label 1)")

if __name__ == "__main__":
    create_bottle_supervised_dataset()