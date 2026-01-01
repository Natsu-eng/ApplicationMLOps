"""
🎯 Générateur Dataset Multi-classe MVTec AD
Classification des TYPES DE DÉFAUTS (sans images "good")
"""

import shutil
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
import random


# ============================================================
# VALIDATION
# ============================================================

def validate_mvtec_structure(mvtec_dir: Path) -> Tuple[bool, str]:
    if not mvtec_dir.exists():
        return False, f"Dossier introuvable: {mvtec_dir}"

    test_dir = mvtec_dir / "test"
    if not test_dir.exists():
        return False, "Pas de dossier test/"

    defect_types = [
        d.name for d in test_dir.iterdir()
        if d.is_dir() and d.name != "good"
    ]

    if len(defect_types) < 2:
        return False, f"Minimum 2 types requis (trouvé: {len(defect_types)})"

    return True, f"{len(defect_types)} types de défauts trouvés: {defect_types}"


# ============================================================
# STATS
# ============================================================

def get_defect_statistics(test_dir: Path) -> Dict[str, int]:
    stats = {}
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    for defect_type in test_dir.iterdir():
        if not defect_type.is_dir() or defect_type.name == "good":
            continue

        count = len([
            f for f in defect_type.glob('*')
            if f.suffix.lower() in extensions
        ])

        if count > 0:
            stats[defect_type.name] = count

    return stats


# ============================================================
# SPLIT STRATIFIÉ
# ============================================================

def split_data(
    items: List[Tuple[Path, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42
) -> Tuple[List, List, List]:

    random.seed(seed)

    by_label: Dict[str, List[Path]] = {}
    for path, label in items:
        by_label.setdefault(label, []).append(path)

    train, val, test = [], [], []

    for label, paths in by_label.items():
        random.shuffle(paths)

        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_paths = paths[:n_train]
        val_paths = paths[n_train:n_train + n_val]
        test_paths = paths[n_train + n_val:]

        train.extend([(p, label) for p in train_paths])
        val.extend([(p, label) for p in val_paths])
        test.extend([(p, label) for p in test_paths])

    return train, val, test


# ============================================================
# DATASET CREATION
# ============================================================

def create_multiclass_dataset(
    mvtec_dir: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42
):
    print(f"\n🎯 Création dataset multiclass depuis: {mvtec_dir.name}")

    is_valid, msg = validate_mvtec_structure(mvtec_dir)
    if not is_valid:
        print(f"❌ {msg}")
        return
    print(f"✅ {msg}")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    test_dir = mvtec_dir / "test"
    defect_stats = get_defect_statistics(test_dir)

    print("\n📊 Statistiques par type de défaut:")
    for k, v in sorted(defect_stats.items()):
        print(f"   {k:20s}: {v:4d}")

    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    all_items = []

    defect_types = sorted(defect_stats.keys())
    label2idx = {d: i for i, d in enumerate(defect_types)}
    idx2label = {i: d for d, i in label2idx.items()}

    print("\n🏷️ Mapping labels:")
    for i, d in idx2label.items():
        print(f"   {i} → {d}")

    for defect in defect_types:
        for img in (test_dir / defect).glob('*'):
            if img.suffix.lower() in extensions:
                all_items.append((img, defect))

    print(f"\n✅ {len(all_items)} images collectées")

    train_set, val_set, test_set = split_data(
        all_items, train_ratio, val_ratio, seed
    )

    print("\n📊 Distribution par split:")
    for name, split_items in [
        ("Train", train_set),
        ("Val", val_set),
        ("Test", test_set)
    ]:
        counter = Counter(label for _, label in split_items)
        print(f"\n   {name}:")
        for defect in defect_types:
            count = counter.get(defect, 0)
            pct = (count / len(split_items) * 100) if split_items else 0
            print(f"      {defect:20s}: {count:4d} ({pct:5.1f}%)")

    print("\n📁 Copie des fichiers...")
    for split_name, split_items in [
        ("train", train_set),
        ("val", val_set),
        ("test", test_set)
    ]:
        for img_path, label in split_items:
            dest_dir = output_dir / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_dir / img_path.name)

    metadata = {
        "source_dataset": mvtec_dir.name,
        "n_classes": len(defect_types),
        "class_names": defect_types,
        "label2idx": label2idx,
        "idx2label": {str(k): v for k, v in idx2label.items()},
        "splits": {
            "train": dict(Counter(l for _, l in train_set)),
            "val": dict(Counter(l for _, l in val_set)),
            "test": dict(Counter(l for _, l in test_set)),
        },
        "seed": seed
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\n🎉 DATASET MULTICLASS CRÉÉ AVEC SUCCÈS")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()
    train_ratio, val_ratio, _ = args.split

    project_root = Path(__file__).parent.parent
    mvtec_dir = project_root / "src" / "data" / "mvtec_ad" / args.category
    output_dir = Path(args.output) if args.output else \
        project_root / "src" / "data" / "mvtec_ad" / f"{args.category}_multiclass"

    create_multiclass_dataset(
        mvtec_dir, output_dir, train_ratio, val_ratio, args.seed
    )


if __name__ == "__main__":
    main()
