"""Tests de `services/vision_datasets.py` (pilier Vision, Lot 15 sous-lot A)
— logique pure d'ingestion ZIP, sans DB ni HTTP."""
from __future__ import annotations

import io
import tarfile
import zipfile

import pytest
from PIL import Image

from services.vision_datasets import (
    MIN_IMAGES_PER_CLASS,
    MIN_TRAIN_GOOD_IMAGES,
    UnsupportedFileType,
    VisionDatasetError,
    analyze_and_extract_vision_archive,
    analyze_and_extract_vision_folder,
    validate_archive_extension,
)


def _png_bytes(color=(255, 0, 0), size=(32, 32), variant: int = 0) -> bytes:
    """`variant` rend deux images de même couleur bit-à-bit distinctes —
    nécessaire depuis la déduplication (Lot 0.1, correctif C1) : deux
    appels avec la même couleur et sans variant produiraient un PNG
    strictement identique, désormais détecté (à raison) comme un vrai
    doublon plutôt que comme deux images distinctes du même jeu de test."""
    img = Image.new("RGB", size, color)
    if variant:
        img.putpixel((0, 0), (variant % 256, (variant * 7) % 256, (variant * 13) % 256))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _build_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _classification_zip(n_per_class=4, n_classes=2) -> bytes:
    files = {}
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for c in range(n_classes):
        for i in range(n_per_class):
            files[f"classe_{c}/img_{i}.png"] = _png_bytes(colors[c % len(colors)], variant=i + 1)
    return _build_zip(files)


def _build_tar(files: dict[str, bytes], compression: str = "") -> bytes:
    buf = io.BytesIO()
    mode = f"w:{compression}" if compression else "w"
    with tarfile.open(fileobj=buf, mode=mode) as tf:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _mvtec_zip(n_train_good=6, n_test_good=3, n_test_defect=3) -> bytes:
    files = {}
    for i in range(n_train_good):
        files[f"train/good/{i}.png"] = _png_bytes((10, 10, 10), variant=i + 1)
    for i in range(n_test_good):
        files[f"test/good/{i}.png"] = _png_bytes((20, 20, 20), variant=i + 1)
    for i in range(n_test_defect):
        files[f"test/scratch/{i}.png"] = _png_bytes((200, 0, 0), variant=i + 1)
    return _build_zip(files)


def test_valid_mvtec_ad_structure_with_category_wrapper_detected(tmp_path):
    """Un téléchargement MVTec AD officiel réel est zippé PAR CATÉGORIE
    (ex. bottle/train/good/..., bottle/test/...) — bug réel trouvé en testant
    avec un vrai dataset, pas seulement des fixtures synthétiques sans
    dossier englobant."""
    files = {f"bottle/train/good/{i}.png": _png_bytes((10, 10, 10), variant=i + 1) for i in range(MIN_TRAIN_GOOD_IMAGES)}
    files.update({f"bottle/test/good/{i}.png": _png_bytes((20, 20, 20), variant=i + 1) for i in range(3)})
    files.update({f"bottle/test/broken_large/{i}.png": _png_bytes((200, 0, 0), variant=i + 1) for i in range(3)})
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "mvtec_ad"
    assert report.class_distribution["train/good"] == MIN_TRAIN_GOOD_IMAGES
    assert report.class_distribution["test/broken_large"] == 3
    # La catégorie englobante ("bottle") ne doit pas se retrouver dans
    # l'arborescence extraite — normalisée vers train/good, test/<x>.
    assert (tmp_path / "train" / "good").exists()
    assert not (tmp_path / "bottle").exists()


def test_ground_truth_folder_ignored_silently(tmp_path):
    """Le dossier `ground_truth/` (masques de segmentation pixel par pixel)
    d'un téléchargement MVTec AD officiel doit être ignoré, jamais une
    erreur de structure ni copié dans le dataset extrait."""
    files = {f"bottle/train/good/{i}.png": _png_bytes((10, 10, 10), variant=i + 1) for i in range(MIN_TRAIN_GOOD_IMAGES)}
    files.update({f"bottle/test/good/{i}.png": _png_bytes((20, 20, 20), variant=i + 1) for i in range(3)})
    files.update({f"bottle/test/broken_large/{i}.png": _png_bytes((200, 0, 0), variant=i + 1) for i in range(3)})
    files.update({f"bottle/ground_truth/broken_large/{i}_mask.png": _png_bytes((255, 255, 255), variant=i + 1) for i in range(3)})
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "mvtec_ad"
    assert report.n_images == MIN_TRAIN_GOOD_IMAGES + 6  # ground_truth exclu du compte
    assert not (tmp_path / "ground_truth").exists()
    assert not (tmp_path / "bottle" / "ground_truth").exists()


def test_multi_category_mvtec_collection_rejected_with_clear_message(tmp_path):
    """Le dossier MVTec AD officiel complet (bottle/, cable/, capsule/, ...)
    zippé tel quel doit être refusé avec un message explicite — pas
    l'erreur générique de structure de classification, trompeuse ici."""
    files = {}
    for category in ("bottle", "cable"):
        files.update({f"{category}/train/good/{i}.png": _png_bytes() for i in range(MIN_TRAIN_GOOD_IMAGES)})
        files.update({f"{category}/test/good/{i}.png": _png_bytes() for i in range(2)})
        files.update({f"{category}/test/scratch/{i}.png": _png_bytes((200, 0, 0)) for i in range(2)})
    content = _build_zip(files)
    with pytest.raises(VisionDatasetError, match="plusieurs catégories"):
        analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)


def test_valid_classification_structure_detected(tmp_path):
    content = _classification_zip(n_per_class=4, n_classes=2)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "classification"
    assert report.n_classes == 2
    assert report.n_images == 8
    assert set(report.class_distribution) == {"classe_0", "classe_1"}
    assert (tmp_path / "classe_0" / "img_0.png").exists()


def test_valid_classification_structure_with_wrapper_folder_detected(tmp_path):
    """Sélectionner le dossier parent contenant les classes puis
    "Compresser" (Explorateur Windows/Finder) inclut naturellement ce
    dossier comme racine du ZIP — bug réel trouvé en testant avec un
    dataset zippé ainsi (pas seulement des fixtures sans dossier
    englobant)."""
    files = {}
    for c in range(2):
        for i in range(4):
            files[f"mon_dataset/classe_{c}/img_{i}.png"] = _png_bytes((50 * c, 50 * c, 50 * c), variant=i + 1)
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "classification"
    assert set(report.class_distribution) == {"classe_0", "classe_1"}
    assert (tmp_path / "classe_0" / "img_0.png").exists()
    assert not (tmp_path / "mon_dataset").exists()


def test_valid_mvtec_ad_structure_detected(tmp_path):
    content = _mvtec_zip()
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "mvtec_ad"
    assert report.n_classes is None
    assert report.class_distribution["train/good"] == 6
    assert report.class_distribution["test/good"] == 3
    assert report.class_distribution["test/scratch"] == 3
    assert (tmp_path / "train/good").exists()


def test_unrecognized_structure_raises(tmp_path):
    content = _build_zip({"photo1.png": _png_bytes(), "photo2.png": _png_bytes()})
    with pytest.raises(VisionDatasetError):
        analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)


def test_zip_slip_path_rejected(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../evil.png", _png_bytes())
    with pytest.raises(VisionDatasetError):
        analyze_and_extract_vision_archive(buf.getvalue(), tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)


def test_bad_zip_file_rejected(tmp_path):
    with pytest.raises(VisionDatasetError):
        analyze_and_extract_vision_archive(b"not a zip", tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)


def test_corrupted_image_excluded_and_reported(tmp_path):
    files = {f"classe_0/img_{i}.png": _png_bytes(variant=i + 1) for i in range(MIN_IMAGES_PER_CLASS + 1)}
    files["classe_0/broken.png"] = b"this is not a real image"
    files.update({f"classe_1/img_{i}.png": _png_bytes((0, 255, 0), variant=i + 1) for i in range(MIN_IMAGES_PER_CLASS)})
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.n_corrupted == 1
    assert "classe_0/broken.png" in report.corrupted_files
    assert any("corrompue" in w for w in report.warnings)


def test_duplicate_images_excluded_not_just_flagged(tmp_path):
    """Correctif C1 (AUDIT_DATALAB_2026-08-16.md) — avant, un doublon était
    compté mais TOUJOURS copié sur disque ("conservé, à revoir
    manuellement"). Maintenant une seule copie survit réellement : le
    fichier exclu n'existe pas sur disque, `class_distribution` (qui pilote
    ensuite le split d'entraînement) ne le compte plus."""
    same = _png_bytes((123, 45, 67))
    files = {
        "classe_0/a.png": same,
        "classe_0/b.png": same,
        "classe_0/c.png": _png_bytes((1, 2, 3)),
        "classe_1/x.png": _png_bytes((4, 5, 6)),
        "classe_1/y.png": _png_bytes((7, 8, 9)),
    }
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.n_duplicates_removed == 1
    assert report.duplicate_removed_files == ["classe_0/b.png"]  # "a" < "b" : la première triée survit
    assert (tmp_path / "classe_0" / "a.png").exists()
    assert not (tmp_path / "classe_0" / "b.png").exists()
    # class_distribution reflète le disque réel : 2 images (a, c), pas 3.
    assert report.class_distribution["classe_0"] == 2
    assert report.n_images == 4  # 2 (classe_0) + 2 (classe_1), pas 5
    assert report.duplicate_detection_note  # limite SHA-256 toujours documentée


def test_dedup_can_push_class_below_minimum_with_explicit_message(tmp_path):
    """La déduplication peut faire passer une classe sous le seuil minimum
    alors qu'elle le respectait avant exclusion des doublons — c'est le bon
    comportement, mais le message doit dire explicitement que c'est la
    déduplication qui en est responsable, sinon l'utilisateur ne comprend
    pas pourquoi un import qui passait avant est refusé."""
    same = _png_bytes((9, 9, 9))
    # Exactement MIN_IMAGES_PER_CLASS fichiers, mais deux sont des doublons
    # bit-à-bit : après déduplication, il n'en reste que MIN_IMAGES_PER_CLASS - 1.
    files = {"classe_0/a.png": same, "classe_0/b.png": same}
    files.update({f"classe_0/img_{i}.png": _png_bytes((i, i, i)) for i in range(MIN_IMAGES_PER_CLASS - 2)})
    files.update({f"classe_1/img_{i}.png": _png_bytes((100 + i, 0, 0)) for i in range(MIN_IMAGES_PER_CLASS)})
    content = _build_zip(files)
    with pytest.raises(VisionDatasetError, match="déduplication|doublons"):
        analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)


def test_mvtec_train_test_duplicate_keeps_train_excludes_test(tmp_path):
    """LE test central du correctif C1 : une image bit-à-bit identique
    présente à la fois dans train/good et test/good est exactement la fuite
    corrigée. La copie de train/ doit survivre, celle de test/ doit être
    exclue — jamais l'inverse, jamais laissé au hasard de l'ordre
    d'itération des buckets. Après ce correctif, aucun hash ne peut plus
    apparaître des deux côtés du split train/test."""
    leaked = _png_bytes((77, 77, 77))
    files = {"train/good/leaked.png": leaked, "test/good/leaked.png": leaked}
    files.update({f"train/good/{i}.png": _png_bytes((i, i, i)) for i in range(MIN_TRAIN_GOOD_IMAGES)})
    files.update({f"test/good/{i}.png": _png_bytes((50 + i, 0, 0)) for i in range(2)})
    files.update({f"test/scratch/{i}.png": _png_bytes((200, 0, 0)) for i in range(2)})
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)

    assert (tmp_path / "train" / "good" / "leaked.png").exists()
    assert not (tmp_path / "test" / "good" / "leaked.png").exists()
    assert "test/good/leaked.png" in report.duplicate_removed_files
    assert "train/good/leaked.png" not in report.duplicate_removed_files

    # Preuve directe qu'aucun hash n'apparaît des deux côtés : les empreintes
    # des fichiers réellement extraits sous train/ et test/ sont disjointes.
    import hashlib as _hashlib

    train_hashes = {
        _hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path / "train").rglob("*.png")
    }
    test_hashes = {
        _hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path / "test").rglob("*.png")
    }
    assert train_hashes.isdisjoint(test_hashes)


def test_cross_class_duplicate_is_label_conflict_not_dedup(tmp_path):
    """La même image bit-à-bit présente dans DEUX classes différentes n'est
    pas un doublon à trancher (garder une copie au hasard fausserait la
    classe survivante) : c'est un conflit d'étiquette, les deux copies
    doivent être exclues."""
    ambiguous = _png_bytes((5, 5, 5))
    files = {"classe_0/a.png": ambiguous, "classe_1/a.png": ambiguous}
    files.update({f"classe_0/extra_{i}.png": _png_bytes((10 + i, 0, 0)) for i in range(MIN_IMAGES_PER_CLASS)})
    files.update({f"classe_1/extra_{i}.png": _png_bytes((0, 10 + i, 0)) for i in range(MIN_IMAGES_PER_CLASS)})
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)

    assert not (tmp_path / "classe_0" / "a.png").exists()
    assert not (tmp_path / "classe_1" / "a.png").exists()
    assert len(report.label_conflicts) == 1
    assert report.label_conflicts[0]["categories"] == ["classe_0", "classe_1"]
    assert set(report.label_conflicts[0]["paths"]) == {"classe_0/a.png", "classe_1/a.png"}
    assert report.n_duplicates_removed == 0  # ce n'est pas comptabilisé comme un doublon bénin
    assert any("classes différentes" in w for w in report.warnings)


def test_mvtec_same_category_duplicate_within_test_is_benign_dedup(tmp_path):
    """Un doublon entre test/good et test/scratch (catégories différentes,
    toutes deux dans test/) est un conflit d'étiquette, pas une fuite
    train/test — la règle "on garde train/" ne s'applique qu'à la MÊME
    catégorie présente des deux côtés du split."""
    ambiguous = _png_bytes((88, 88, 88))
    files = {"test/good/x.png": ambiguous, "test/scratch/x.png": ambiguous}
    files.update({f"train/good/{i}.png": _png_bytes((i, i, i)) for i in range(MIN_TRAIN_GOOD_IMAGES)})
    files.update({f"test/good/{i}.png": _png_bytes((50 + i, 0, 0)) for i in range(2)})
    files.update({f"test/scratch/{i}.png": _png_bytes((200, 0, 0)) for i in range(2)})
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)

    assert not (tmp_path / "test" / "good" / "x.png").exists()
    assert not (tmp_path / "test" / "scratch" / "x.png").exists()
    assert len(report.label_conflicts) == 1
    assert report.label_conflicts[0]["categories"] == ["good", "scratch"]


def test_undersized_image_excluded(tmp_path):
    files = {f"classe_0/img_{i}.png": _png_bytes(size=(32, 32), variant=i + 1) for i in range(MIN_IMAGES_PER_CLASS)}
    files["classe_0/tiny.png"] = _png_bytes(size=(5, 5))
    files.update(
        {f"classe_1/img_{i}.png": _png_bytes((0, 0, 255), size=(32, 32), variant=i + 1) for i in range(MIN_IMAGES_PER_CLASS)}
    )
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.n_undersized == 1
    assert "classe_0/tiny.png" in report.undersized_files


def test_class_with_too_few_images_rejected(tmp_path):
    content = _classification_zip(n_per_class=MIN_IMAGES_PER_CLASS - 1, n_classes=2)
    with pytest.raises(VisionDatasetError):
        analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)


def test_mvtec_missing_test_good_rejected(tmp_path):
    files = {f"train/good/{i}.png": _png_bytes() for i in range(MIN_TRAIN_GOOD_IMAGES)}
    files.update({f"test/scratch/{i}.png": _png_bytes((200, 0, 0)) for i in range(2)})
    content = _build_zip(files)
    with pytest.raises(VisionDatasetError):
        analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)


def test_train_with_a_non_good_folder_is_reinterpreted_as_classification(tmp_path):
    """Avant le Lot 6A : train/ contenant autre chose que 'good' était
    REJETÉ (fuite de labels d'anomalie dans l'entraînement non supervisé,
    correctif du bug #1). Depuis le Lot 6A : train_categories != {"good"}
    ne signifie plus automatiquement une erreur — c'est aussi le signal
    d'une classification pré-découpée légitime (ex. train/good +
    train/scratch = 2 classes distinctes, aussi valide que n'importe quel
    autre jeu de classes). Le dataset est exploité plutôt que rejeté :
    comportement strictement meilleur, jamais un jeu de données perdu
    pour l'utilisateur. La structure normal/défaut STRICTE reste rejetée
    dans ce cas précis (voir test_mvtec_structure_still_detected_when_train_test_val_all_present
    pour la confirmation qu'elle continue de fonctionner quand train/ ne
    contient bien QUE 'good')."""
    files = {f"train/good/{i}.png": _png_bytes(variant=i + 1) for i in range(MIN_TRAIN_GOOD_IMAGES)}
    files.update({f"train/scratch/{i}.png": _png_bytes((200, 0, 0), variant=i + 1) for i in range(2)})
    files.update({f"test/good/{i}.png": _png_bytes(variant=i + 1) for i in range(2)})
    files.update({f"test/scratch/{i}.png": _png_bytes((200, 0, 0), variant=i + 1) for i in range(2)})
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "classification"
    assert set(report.class_distribution) == {"good", "scratch"}


def test_too_many_images_rejected(tmp_path):
    content = _classification_zip(n_per_class=5, n_classes=2)
    with pytest.raises(VisionDatasetError):
        analyze_and_extract_vision_archive(content, tmp_path, max_images=3, max_uncompressed_bytes=10_000_000)


def test_class_imbalance_warns_but_does_not_block(tmp_path):
    files = {f"classe_0/img_{i}.png": _png_bytes(variant=i + 1) for i in range(30)}
    files.update({f"classe_1/img_{i}.png": _png_bytes((0, 255, 0), variant=i + 1) for i in range(2)})
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "classification"
    assert any("déséquilibrées" in w for w in report.warnings)


def test_non_image_files_ignored(tmp_path):
    files = {f"classe_0/img_{i}.png": _png_bytes(variant=i + 1) for i in range(MIN_IMAGES_PER_CLASS)}
    files.update({f"classe_1/img_{i}.png": _png_bytes((0, 255, 0), variant=i + 1) for i in range(MIN_IMAGES_PER_CLASS)})
    files["classe_0/notes.txt"] = b"pas une image"
    files["__MACOSX/classe_0/._img_0.png"] = b"metadata macos"
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.n_images == MIN_IMAGES_PER_CLASS * 2


# ── Lot 6A — formats d'archive étendus (tar/tar.gz) + import de dossier ─────


def test_validate_archive_extension_accepts_zip_and_tar_family():
    assert validate_archive_extension("d.zip") == ".zip"
    assert validate_archive_extension("d.tar") == ".tar"
    assert validate_archive_extension("d.tar.gz") == ".tar.gz"
    assert validate_archive_extension("d.tgz") == ".tgz"


def test_validate_archive_extension_rejects_unknown_extension():
    with pytest.raises(UnsupportedFileType):
        validate_archive_extension("d.rar")


@pytest.mark.parametrize("compression", ["", "gz"])
def test_tar_archive_detected_and_extracted(tmp_path, compression):
    """Le format officiel de téléchargement MVTec AD est .tar.xz — jamais
    .zip — d'où l'exigence "supporte aussi tar/tar.gz" (pas seulement zip)."""
    files = {
        f"classe_{c}/img_{i}.png": _png_bytes([(255, 0, 0), (0, 255, 0)][c], variant=i + 1)
        for c in range(2) for i in range(4)
    }
    content = _build_tar(files, compression=compression)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "classification"
    assert report.n_images == 8
    assert (tmp_path / "classe_0" / "img_0.png").exists()


def test_tar_archive_sniffed_by_content_not_by_a_trusted_extension(tmp_path):
    """`analyze_and_extract_vision_archive` ne reçoit jamais le nom de
    fichier déclaré — seul le contenu réel (signature binaire) détermine
    le format, jamais une extension qui pourrait mentir."""
    content = _build_tar({"classe_0/a.png": _png_bytes(variant=1), "classe_0/b.png": _png_bytes(variant=2),
                           "classe_1/a.png": _png_bytes((0, 255, 0), variant=1),
                           "classe_1/b.png": _png_bytes((0, 255, 0), variant=2)})
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "classification"


def test_unrecognized_archive_format_rejected(tmp_path):
    with pytest.raises(VisionDatasetError):
        analyze_and_extract_vision_archive(b"ni un zip ni un tar", tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)


def test_folder_upload_detects_classification_structure(tmp_path):
    """Import de dossier (Lot 6A) — chaque fichier porte son chemin relatif
    complet (webkitRelativePath côté navigateur), pas d'archive à ouvrir."""
    files = [
        (f"mon_dataset/classe_{c}/img_{i}.png", _png_bytes([(255, 0, 0), (0, 255, 0)][c], variant=i + 1))
        for c in range(2) for i in range(4)
    ]
    report = analyze_and_extract_vision_folder(files, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "classification"
    assert report.n_images == 8
    assert (tmp_path / "classe_0" / "img_0.png").exists()
    assert not (tmp_path / "mon_dataset").exists()  # dossier englobant normalisé, comme pour un zip


def test_folder_upload_detects_mvtec_structure(tmp_path):
    files = [(f"train/good/{i}.png", _png_bytes((10, 10, 10), variant=i + 1)) for i in range(MIN_TRAIN_GOOD_IMAGES)]
    files += [(f"test/good/{i}.png", _png_bytes((20, 20, 20), variant=i + 1)) for i in range(3)]
    files += [(f"test/scratch/{i}.png", _png_bytes((200, 0, 0), variant=i + 1)) for i in range(3)]
    report = analyze_and_extract_vision_folder(files, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "mvtec_ad"


def test_pre_split_classification_merges_train_test_val_by_class(tmp_path):
    """Lot 6A — classification déjà découpée en train/test(/val), un
    sous-dossier par classe sous chaque split, SANS dossier "good" (donc
    pas normal/défaut malgré la même profondeur à 3 niveaux) — bug réel
    trouvé en testant avec un vrai dataset utilisateur multi-classes
    (pas seulement des fixtures synthétiques)."""
    files = {}
    variant = 0
    for split in ("train", "test", "val"):
        for cls in ("chat", "chien"):
            for i in range(3):
                variant += 1
                files[f"{split}/{cls}/{i}.png"] = _png_bytes((255, 0, 0) if cls == "chat" else (0, 255, 0), variant=variant)
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "classification"
    assert set(report.class_distribution) == {"chat", "chien"}
    # 3 splits x 3 images x 2 classes = 18, fusionnées (pas de dossier train/test/val sur disque).
    assert report.n_images == 18
    assert report.class_distribution["chat"] == 9
    assert not (tmp_path / "train").exists()
    assert not (tmp_path / "test").exists()
    assert (tmp_path / "chat").exists()


def test_pre_split_classification_without_val_also_recognized(tmp_path):
    files = {}
    variant = 0
    for split in ("train", "test"):
        for cls in ("a", "b", "c"):
            for i in range(3):
                variant += 1
                files[f"{split}/{cls}/{i}.png"] = _png_bytes(variant=variant)
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "classification"
    assert set(report.class_distribution) == {"a", "b", "c"}


def test_pre_split_classification_resolves_filename_collisions_across_splits(tmp_path):
    """Deux fichiers SOURCE distincts (train/chat/0.png et test/chat/0.png)
    partagent le même nom une fois fusionnés dans la classe "chat" — ne
    doivent JAMAIS s'écraser silencieusement sur disque."""
    files = {
        "train/chat/0.png": _png_bytes((255, 0, 0), variant=1),
        "test/chat/0.png": _png_bytes((255, 0, 0), variant=2),  # même nom, contenu DIFFÉRENT
        "train/chien/0.png": _png_bytes((0, 255, 0), variant=3),
        "test/chien/0.png": _png_bytes((0, 255, 0), variant=4),
    }
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.n_images == 4  # aucune perte par écrasement
    assert report.class_distribution["chat"] == 2
    assert len(list((tmp_path / "chat").glob("*.png"))) == 2


def test_mvtec_structure_still_detected_when_train_test_val_all_present(tmp_path):
    """Régression : train/good + test/good + test/<defaut> reste détecté
    comme normal/défaut même si un dossier val/good est aussi présent
    (généralisation à train/test/val, Lot 6A) — ne bascule jamais par
    erreur vers la classification pré-découpée."""
    files = {f"train/good/{i}.png": _png_bytes(variant=i + 1) for i in range(MIN_TRAIN_GOOD_IMAGES)}
    files.update({f"test/good/{i}.png": _png_bytes((20, 20, 20), variant=i + 1) for i in range(2)})
    files.update({f"test/scratch/{i}.png": _png_bytes((200, 0, 0), variant=i + 1) for i in range(2)})
    files.update({f"val/good/{i}.png": _png_bytes((30, 30, 30), variant=i + 1) for i in range(2)})
    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)
    assert report.structure_type == "mvtec_ad"
    assert report.class_distribution["val/good"] == 2


def test_folder_upload_rejects_path_traversal(tmp_path):
    """Même garde-fou zip-slip que les archives — un navigateur ne
    produirait normalement jamais un chemin `..`, mais la requête HTTP
    elle-même n'a aucune garantie d'origine légitime."""
    with pytest.raises(VisionDatasetError):
        analyze_and_extract_vision_folder(
            [("../../evil.png", _png_bytes())], tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000
        )


# ── EDA d'images (Lot 6A, §G.3/§G.4/§G.5) ──────────────────────────────────


def test_image_eda_reports_resolution_format_and_color_mode(tmp_path):
    files = {}
    # Deux tailles distinctes, deux formats distincts, un mode RGBA.
    for i in range(3):
        img = Image.new("RGB", (32, 32), (255, 0, 0))
        if i:
            img.putpixel((0, 0), (i % 256, (i * 7) % 256, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        files[f"classe_0/small_{i}.png"] = buf.getvalue()
    for i in range(2):
        img = Image.new("RGB", (300, 300), (0, 255, 0))
        img.putpixel((0, 0), (i + 1, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        files[f"classe_0/large_{i}.jpg"] = buf.getvalue()
    rgba_img = Image.new("RGBA", (64, 64), (0, 0, 255, 128))
    buf = io.BytesIO()
    rgba_img.save(buf, format="PNG")
    files["classe_1/rgba.png"] = buf.getvalue()
    files["classe_1/other.png"] = _png_bytes((10, 10, 10), size=(64, 64), variant=99)

    content = _build_zip(files)
    report = analyze_and_extract_vision_archive(content, tmp_path, max_images=1000, max_uncompressed_bytes=10_000_000)

    eda = report.image_eda
    assert eda["width"]["min"] == 32
    assert eda["width"]["max"] == 300
    assert eda["height"]["min"] == 32
    assert eda["height"]["max"] == 300
    assert sum(eda["resolution_buckets"].values()) == report.n_images
    assert eda["format_distribution"].get("PNG", 0) >= 4  # small_* (3) + rgba + other
    assert eda["format_distribution"].get("JPEG", 0) == 2
    assert eda["color_mode_distribution"].get("RGB", 0) >= 5
    assert eda["color_mode_distribution"].get("RGBA", 0) == 1


def test_image_eda_excludes_deduplicated_images():
    """L'EDA ne doit compter QUE les images réellement conservées après
    déduplication — sinon l'histogramme affiché à l'utilisateur ne
    correspond pas à ce qui sera vraiment utilisé à l'entraînement."""
    from services.vision_datasets import _ValidImage, _compute_image_eda
    from pathlib import PurePosixPath

    kept = [
        _ValidImage(
            rel_path=PurePosixPath("classe_0/a.png"), content=b"x", bucket_name="classe_0",
            digest="d1", width=100, height=100, format="PNG", mode="RGB",
        ),
    ]
    eda = _compute_image_eda(kept)
    assert sum(eda["resolution_buckets"].values()) == 1
    assert eda["format_distribution"] == {"PNG": 1}


def test_image_eda_empty_dataset_degrades_honestly():
    from services.vision_datasets import _compute_image_eda

    eda = _compute_image_eda([])
    assert eda["resolution_buckets"] == {}
    assert eda["width"]["min"] is None
    assert eda["format_distribution"] == {}


def test_resolution_bucket_labels_are_contiguous():
    from services.vision_datasets import _resolution_bucket_label

    assert _resolution_bucket_label(50) == "< 128px"
    assert _resolution_bucket_label(128) == "128-224px"
    assert _resolution_bucket_label(223) == "128-224px"
    assert _resolution_bucket_label(224) == "224-512px"
    assert _resolution_bucket_label(2000) == ">= 1024px"
