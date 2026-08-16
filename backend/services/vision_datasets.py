"""Logique pure d'ingestion des datasets d'images (pilier Vision, Lot 15
sous-lot A) — sans dépendance HTTP, directement testable.

Fondation partagée par la classification d'images (sous-lot B) et la
détection d'anomalies visuelles MVTec AD (sous-lot C) : les deux ont besoin
d'ingérer un ZIP d'images, de détecter sa structure et de le valider. Écrit
une seule fois ici plutôt que dupliqué dans deux endpoints (même
raisonnement que `services/job_quota.py` pour les jobs).

Corrige le bug #1 déjà documenté dans
`docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md` (labels `y_train` d'un
dataset MVTec AD jamais validés avant training, risque de partir en mode
supervisé par accident) : ici la structure est validée AU MOMENT DE
L'UPLOAD, de façon stricte — jamais une supposition silencieuse.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_IGNORED_NAME_PREFIXES = ("__MACOSX/", ".")
_IGNORED_FILENAMES = {".ds_store", "thumbs.db"}

MIN_IMAGE_DIMENSION_PX = 20
MIN_IMAGES_PER_CLASS = 2
MIN_TRAIN_GOOD_IMAGES = 5
CLASS_IMBALANCE_WARNING_RATIO = 10.0


class VisionDatasetError(ValueError):
    """Le ZIP a été reçu mais ne peut pas être ingéré comme dataset d'images
    (structure non reconnue, contenu dangereux, aucune image exploitable)."""


class UnsupportedFileType(ValueError):
    """Extension de fichier non supportée pour un dataset vision."""


def validate_zip_extension(filename: str) -> None:
    if Path(filename).suffix.lower() != ".zip":
        raise UnsupportedFileType("Le dataset d'images doit être fourni sous forme d'archive .zip")


@dataclass
class VisionDatasetReport:
    structure_type: str  # "classification" | "mvtec_ad"
    n_images: int
    n_classes: int | None
    class_distribution: dict[str, int]
    n_corrupted: int
    corrupted_files: list[str]
    # Correctif C1 (AUDIT_DATALAB_2026-08-16.md) — avant, ces doublons
    # étaient comptés mais TOUS conservés sur disque ("à revoir
    # manuellement"), ce qui permettait à deux copies bit-à-bit de la même
    # image de se retrouver de part et d'autre d'un split. Maintenant une
    # seule copie survit par doublon (voir _validate_and_copy_images).
    n_duplicates_removed: int
    duplicate_removed_files: list[str]
    # Cas distinct des doublons : la même image bit-à-bit présente dans des
    # classes/catégories DIFFÉRENTES n'est pas un doublon à trancher mais un
    # conflit d'étiquette — toutes les copies sont exclues, jamais un choix
    # arbitraire (voir _validate_and_copy_images).
    label_conflicts: list[dict[str, list[str]]]
    duplicate_detection_note: str
    n_undersized: int
    undersized_files: list[str]
    warnings: list[str] = field(default_factory=list)


def _safe_member_path(name: str) -> PurePosixPath | None:
    """Normalise une entrée d'archive et rejette tout ce qui sort du dossier
    cible (zip-slip) — retourne None pour les entrées à ignorer silencieusement."""
    if name.endswith("/"):
        return None  # dossier
    if any(name.startswith(prefix) for prefix in _IGNORED_NAME_PREFIXES):
        return None
    posix = PurePosixPath(name)
    if posix.name.lower() in _IGNORED_FILENAMES:
        return None
    # Un téléchargement MVTec AD officiel inclut un dossier `ground_truth/`
    # (masques de segmentation pixel par pixel) à côté de train/test — ce
    # sous-lot ne calibre le seuil que sur des labels image entière
    # (bug #7/#12, J de Youden), aucune métrique pixel (IoU) n'est calculée :
    # ignoré silencieusement, jamais une erreur de structure sur ces fichiers.
    if "ground_truth" in (p.lower() for p in posix.parts):
        return None
    if posix.is_absolute() or ".." in posix.parts:
        raise VisionDatasetError("Archive invalide : chemin d'entrée non autorisé détecté")
    return posix


def _open_zip_members(content: bytes, max_images: int, max_uncompressed_bytes: int) -> list[tuple[PurePosixPath, zipfile.ZipInfo]]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise VisionDatasetError(f"Archive ZIP illisible : {exc}") from exc

    members: list[tuple[PurePosixPath, zipfile.ZipInfo]] = []
    total_uncompressed = 0
    for info in zf.infolist():
        rel_path = _safe_member_path(info.filename)
        if rel_path is None:
            continue
        if rel_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        total_uncompressed += info.file_size
        if total_uncompressed > max_uncompressed_bytes:
            raise VisionDatasetError(
                f"Archive trop volumineuse une fois décompressée (max {max_uncompressed_bytes // (1024 * 1024)} Mo)"
            )
        members.append((rel_path, info))
        if len(members) > max_images:
            raise VisionDatasetError(f"Trop d'images dans l'archive (max {max_images})")

    if not members:
        raise VisionDatasetError("Aucune image exploitable trouvée dans l'archive (formats acceptés : "
                                  f"{', '.join(sorted(IMAGE_EXTENSIONS))})")
    return members


def _detect_structure(members: list[tuple[PurePosixPath, zipfile.ZipInfo]]) -> tuple[str, dict[str, list[tuple[PurePosixPath, zipfile.ZipInfo]]]]:
    """Détecte "classification" (dossiers de classes) ou "mvtec_ad"
    (train/good + test/good + test/<defaut>) à partir des chemins présents
    dans l'archive. Ne devine jamais silencieusement : lève une erreur
    explicite si aucune structure reconnue ne correspond (correctif du bug
    #1 — plus de mode détecté par accident).

    Retourne le type détecté et les membres groupés par "bucket" (nom de
    classe pour la classification, "train/good"/"test/good"/"test/<x>" pour
    MVTec AD)."""
    top_level_dirs = {m[0].parts[0] for m in members if len(m[0].parts) >= 2}

    # Un téléchargement MVTec AD officiel est zippé PAR CATÉGORIE
    # (ex. bottle.tar.gz → bottle/train/good/..., bottle/test/...) : un seul
    # dossier de plus au-dessus de train/test, jamais l'archive elle-même.
    # Détecté ici plutôt que d'obliger l'utilisateur à re-zipper manuellement
    # le contenu du dossier catégorie (bug réel trouvé en testant avec un
    # vrai dataset MVTec AD, pas seulement des fixtures synthétiques).
    mvtec_offset = 0
    if top_level_dirs != {"train", "test"} and len(top_level_dirs) == 1:
        wrapper_dir = next(iter(top_level_dirs))
        second_level_dirs = {
            m[0].parts[1] for m in members if len(m[0].parts) >= 3 and m[0].parts[0] == wrapper_dir
        }
        if second_level_dirs == {"train", "test"}:
            mvtec_offset = 1

    if mvtec_offset == 1 or top_level_dirs == {"train", "test"}:
        buckets: dict[str, list[tuple[PurePosixPath, zipfile.ZipInfo]]] = {}
        for rel_path, info in members:
            parts = rel_path.parts[mvtec_offset:]
            if len(parts) != 3:
                raise VisionDatasetError(
                    f"Structure MVTec AD invalide : '{rel_path}' doit être directement sous "
                    "train/<categorie>/ ou test/<categorie>/ (pas de sous-dossier supplémentaire)"
                )
            split, category = parts[0], parts[1]
            bucket = f"{split}/{category}"
            buckets.setdefault(bucket, []).append((rel_path, info))

        if len(buckets.get("train/good", [])) < MIN_TRAIN_GOOD_IMAGES:
            raise VisionDatasetError(
                f"Structure MVTec AD invalide : train/good/ doit contenir au moins "
                f"{MIN_TRAIN_GOOD_IMAGES} images normales (trouvé {len(buckets.get('train/good', []))})"
            )
        train_categories = {b.split("/", 1)[1] for b in buckets if b.startswith("train/")}
        if train_categories != {"good"}:
            raise VisionDatasetError(
                "Structure MVTec AD invalide : train/ ne doit contenir que des images normales "
                f"(dossier 'good'), trouvé : {', '.join(sorted(train_categories))}"
            )
        test_categories = {b.split("/", 1)[1] for b in buckets if b.startswith("test/")}
        if "good" not in test_categories:
            raise VisionDatasetError(
                "Structure MVTec AD invalide : test/good/ est requis (nécessaire pour calibrer le seuil "
                "de détection, voir docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md #12)"
            )
        if len(test_categories) < 2:
            raise VisionDatasetError(
                "Structure MVTec AD invalide : test/ doit contenir au moins une catégorie de défaut "
                "en plus de good/ (ex. test/scratch/, test/crack/)"
            )
        return "mvtec_ad", buckets

    # Diagnostic dédié : le téléchargement officiel MVTec AD complet
    # (ex. dossier "MVTec AD/" avec bottle/, cable/, capsule/, ... — 15
    # catégories) zippé tel quel a PLUSIEURS dossiers de premier niveau,
    # contenant chacun train/test — jamais un seul dataset exploitable (le
    # produit ne représente qu'UNE catégorie par dataset : mélanger des
    # bouteilles et des câbles comme s'ils étaient tous "normaux" n'a pas de
    # sens pour un autoencodeur). Message explicite plutôt que l'erreur
    # générique de classification, trompeuse ici (bug réel trouvé en testant
    # avec le vrai dossier MVTec AD complet, pas seulement une catégorie).
    if len(top_level_dirs) >= 2:
        categories_with_train_test = [
            d
            for d in top_level_dirs
            if {m[0].parts[1] for m in members if len(m[0].parts) >= 3 and m[0].parts[0] == d} >= {"train", "test"}
        ]
        if len(categories_with_train_test) >= 2:
            raise VisionDatasetError(
                "Cette archive contient plusieurs catégories MVTec AD à la fois "
                f"({', '.join(sorted(categories_with_train_test))}) — un dataset ne peut représenter "
                "qu'une seule catégorie. Zippez le dossier d'une seule catégorie (ex. juste 'bottle/', "
                "ou son contenu train/+test/) et importez chaque catégorie séparément."
            )

    # Même tolérance que MVTec AD ci-dessus, pour la même raison réelle :
    # sélectionner un dossier parent contenant les dossiers de classes puis
    # "Compresser" (Explorateur Windows/Finder) inclut naturellement ce
    # dossier parent comme racine du ZIP — comportement le plus probable
    # d'un utilisateur, pas une erreur à rejeter (bug réel trouvé en testant
    # avec un dataset classification zippé ainsi, pas seulement des
    # fixtures sans dossier englobant).
    classification_offset = 0
    if len(top_level_dirs) == 1:
        wrapper_dir = next(iter(top_level_dirs))
        candidate_classes = {
            m[0].parts[1] for m in members if len(m[0].parts) >= 3 and m[0].parts[0] == wrapper_dir
        }
        if len(candidate_classes) >= 2:
            classification_offset = 1

    buckets = {}
    for rel_path, info in members:
        parts = rel_path.parts[classification_offset:]
        if len(parts) != 2:
            raise VisionDatasetError(
                f"Structure de classification invalide : '{rel_path}' doit être directement sous "
                "<classe>/ (pas de sous-dossier supplémentaire, pas de fichier à la racine)"
            )
        class_name = parts[0]
        buckets.setdefault(class_name, []).append((rel_path, info))

    if len(buckets) < 2:
        raise VisionDatasetError(
            "Structure non reconnue : fournissez soit au moins 2 dossiers de classes "
            "(classification), soit une structure train/good + test/good + test/<defaut> (MVTec AD)"
        )
    return "classification", buckets


@dataclass
class _ValidImage:
    """Une image ayant passé les contrôles d'intégrité (pas corrompue, pas
    sous-dimensionnée) — étape intermédiaire nécessaire avant de décider
    quelle copie garder en cas de doublon : cette décision doit voir TOUTES
    les occurrences du même hash dans le ZIP entier, y compris dans un
    bucket pas encore parcouru, donc impossible en un seul passage
    streaming comme avant ce correctif (copie immédiate sur disque)."""
    rel_path: PurePosixPath
    info: zipfile.ZipInfo
    bucket_name: str
    digest: str


@dataclass
class _ValidationOutcome:
    n_valid: int
    corrupted_files: list[str]
    undersized_files: list[str]
    class_distribution: dict[str, int]
    # Comptes AVANT exclusion des doublons/conflits d'étiquette (mais après
    # corrompues/sous-dimensionnées) — sert uniquement à distinguer, dans le
    # message d'erreur, une classe qui était déjà sous le seuil d'une classe
    # que la déduplication y a fait passer.
    class_distribution_before_dedup: dict[str, int]
    n_duplicates_removed: int
    duplicate_removed_files: list[str]
    label_conflicts: list[dict[str, list[str]]]


def _bucket_category(bucket_name: str, structure_type: str) -> str:
    """La "vérité terrain" portée par un bucket — la classe en
    classification, la catégorie de défaut (good/scratch/...) en MVTec AD.
    train/good et test/good portent la MÊME catégorie ("good") : ce n'est
    pas un conflit d'étiquette, contrairement à train/good vs test/scratch."""
    if structure_type == "mvtec_ad":
        return bucket_name.split("/", 1)[1]
    return bucket_name


def _bucket_split(bucket_name: str, structure_type: str) -> str | None:
    """"train"/"test" pour MVTec AD, None sinon — la classification n'a pas
    de notion de split à l'ingestion (le split train/val/test est fait plus
    tard, aléatoirement, par vision_classification_training.py ; comme la
    déduplication ci-dessous ne laisse jamais deux copies bit-à-bit du même
    contenu nulle part dans le dataset, ce split aléatoire ne peut alors
    plus les répartir des deux côtés — correctif C1)."""
    if structure_type == "mvtec_ad":
        return bucket_name.split("/", 1)[0]
    return None


def _validate_and_copy_images(
    zip_content: bytes,
    buckets: dict[str, list[tuple[PurePosixPath, zipfile.ZipInfo]]],
    target_dir: Path,
    structure_type: str,
) -> _ValidationOutcome:
    """Ouvre chaque image (détection des fichiers corrompus/tronqués),
    calcule un hash, décide quelle(s) copie(s) survivent et copie
    UNIQUEMENT les images retenues vers `target_dir` — jamais bloquant sur
    une image individuelle corrompue, elle est simplement exclue et
    reportée.

    Trois traitements distincts des images en double (correctif C1,
    AUDIT_DATALAB_2026-08-16.md — avant ce correctif, TOUTES les copies
    étaient conservées "à revoir manuellement", ce qui permettait à des
    doublons bit-à-bit de se retrouver de part et d'autre d'un split) :

    1. Doublon bénin (même catégorie, même split, ou classification) : une
       seule copie survit (la première par ordre alphabétique du chemin,
       choix déterministe) — c'est un doublon ordinaire, pas une fuite.
    2. Fuite train/test (MVTec AD uniquement, même catégorie présente à la
       fois dans train/ et test/) : la copie de train/ survit TOUJOURS, la
       ou les copies de test/ sont exclues — le jeu d'évaluation doit
       rester non vu, jamais l'inverse. Décidé explicitement ici, jamais
       laissé à l'ordre d'itération des buckets.
    3. Conflit d'étiquette (la même image bit-à-bit présente dans des
       catégories DIFFÉRENTES, ex. classe "chat" et classe "chien", ou
       "good" et "scratch") : ce ne sont pas des doublons à trancher mais
       deux vérités contradictoires sur la même image — TOUTES les copies
       sont exclues, aucun arbitrage arbitraire qui fausserait une classe.

    Limite assumée, documentée dans le rapport et affichée à l'utilisateur
    (`VisionDatasetReport.duplicate_detection_note`) : la détection est par
    empreinte SHA-256, donc strictement bit-à-bit. Une image recadrée,
    redimensionnée, recompressée ou réenregistrée dans un autre format
    n'est pas détectée — hachage perceptuel hors périmètre de ce correctif."""
    zf = zipfile.ZipFile(io.BytesIO(zip_content))
    corrupted_files: list[str] = []
    undersized_files: list[str] = []
    valid_images: list[_ValidImage] = []

    for bucket_name, entries in buckets.items():
        for rel_path, info in entries:
            raw = zf.read(info)
            try:
                img = Image.open(io.BytesIO(raw))
                img.verify()
                # verify() invalide l'objet pour un usage ultérieur — réouverture
                # nécessaire pour lire les dimensions réelles (docs Pillow).
                img = Image.open(io.BytesIO(raw))
                width, height = img.size
                img.load()
            except (UnidentifiedImageError, OSError, ValueError):
                corrupted_files.append(str(rel_path))
                continue

            if width < MIN_IMAGE_DIMENSION_PX or height < MIN_IMAGE_DIMENSION_PX:
                undersized_files.append(str(rel_path))
                continue

            digest = hashlib.sha256(raw).hexdigest()
            valid_images.append(_ValidImage(rel_path=rel_path, info=info, bucket_name=bucket_name, digest=digest))

    by_digest: dict[str, list[_ValidImage]] = {}
    for vi in valid_images:
        by_digest.setdefault(vi.digest, []).append(vi)

    excluded_as_duplicate: set[str] = set()
    excluded_as_conflict: set[str] = set()
    label_conflicts: list[dict[str, list[str]]] = []

    for group in by_digest.values():
        if len(group) < 2:
            continue
        categories = sorted({_bucket_category(vi.bucket_name, structure_type) for vi in group})
        if len(categories) > 1:
            paths = sorted(str(vi.rel_path) for vi in group)
            label_conflicts.append({"categories": categories, "paths": paths})
            excluded_as_conflict.update(paths)
            continue

        train_copies = [vi for vi in group if _bucket_split(vi.bucket_name, structure_type) == "train"]
        test_copies = [vi for vi in group if _bucket_split(vi.bucket_name, structure_type) == "test"]
        if train_copies and test_copies:
            survivor = min(train_copies, key=lambda vi: str(vi.rel_path))  # jamais une copie de test/
        else:
            survivor = min(group, key=lambda vi: str(vi.rel_path))
        excluded_as_duplicate.update(str(vi.rel_path) for vi in group if vi is not survivor)

    all_excluded = excluded_as_duplicate | excluded_as_conflict

    class_distribution_before_dedup: dict[str, int] = {bucket_name: 0 for bucket_name in buckets}
    class_distribution: dict[str, int] = {bucket_name: 0 for bucket_name in buckets}
    for vi in valid_images:
        class_distribution_before_dedup[vi.bucket_name] += 1
        if str(vi.rel_path) not in all_excluded:
            class_distribution[vi.bucket_name] += 1

    for bucket_name in buckets:
        (target_dir / bucket_name).mkdir(parents=True, exist_ok=True)

    n_valid = 0
    for vi in valid_images:
        if str(vi.rel_path) in all_excluded:
            continue
        raw = zf.read(vi.info)
        (target_dir / vi.bucket_name / vi.rel_path.name).write_bytes(raw)
        n_valid += 1

    return _ValidationOutcome(
        n_valid=n_valid,
        corrupted_files=corrupted_files,
        undersized_files=undersized_files,
        class_distribution=class_distribution,
        class_distribution_before_dedup=class_distribution_before_dedup,
        n_duplicates_removed=len(excluded_as_duplicate),
        duplicate_removed_files=sorted(excluded_as_duplicate),
        label_conflicts=label_conflicts,
    )


def analyze_and_extract_vision_zip(
    content: bytes,
    target_dir: Path,
    max_images: int,
    max_uncompressed_bytes: int,
) -> VisionDatasetReport:
    """Point d'entrée principal — valide le ZIP, détecte la structure,
    valide, déduplique et copie les images retenues vers `target_dir`. Lève
    `VisionDatasetError` pour tout problème structurel (jamais pour une
    image individuelle corrompue ou un doublon, qui sont simplement exclus
    et reportés)."""
    members = _open_zip_members(content, max_images, max_uncompressed_bytes)
    structure_type, buckets = _detect_structure(members)

    outcome = _validate_and_copy_images(content, buckets, target_dir, structure_type)

    if structure_type == "classification":
        empty_classes = sorted(c for c, n in outcome.class_distribution.items() if n == 0)
        if empty_classes:
            raise VisionDatasetError(
                f"Classe(s) sans image exploitable après validation : {', '.join(empty_classes)}"
            )
        under_min = sorted(c for c, n in outcome.class_distribution.items() if n < MIN_IMAGES_PER_CLASS)
        if under_min:
            message = f"Classe(s) avec moins de {MIN_IMAGES_PER_CLASS} images exploitables : {', '.join(under_min)}"
            # La déduplication (correctif C1) peut faire passer une classe sous
            # le seuil alors qu'elle le respectait avant exclusion des
            # doublons/conflits d'étiquette — message honnête plutôt qu'un
            # refus incompréhensible pour l'utilisateur.
            caused_by_dedup = sorted(
                c for c in under_min if outcome.class_distribution_before_dedup.get(c, 0) >= MIN_IMAGES_PER_CLASS
            )
            if caused_by_dedup:
                message += (
                    f". La suppression des doublons/conflits d'étiquette a fait passer "
                    f"{', '.join(caused_by_dedup)} sous ce seuil (elle(s) en avaient assez avant) — "
                    "importez des images supplémentaires et distinctes pour cette/ces classe(s)."
                )
            raise VisionDatasetError(message)
        n_classes = len(outcome.class_distribution)
    else:
        train_good_after = outcome.class_distribution.get("train/good", 0)
        if train_good_after < MIN_TRAIN_GOOD_IMAGES:
            message = f"Après validation, train/good/ contient moins de {MIN_TRAIN_GOOD_IMAGES} images exploitables"
            if outcome.class_distribution_before_dedup.get("train/good", 0) >= MIN_TRAIN_GOOD_IMAGES:
                message += (
                    " — la suppression des doublons/conflits d'étiquette est responsable "
                    "(train/good/ en avait assez avant) : importez des images supplémentaires et distinctes"
                )
            raise VisionDatasetError(message)
        n_classes = None

    warnings: list[str] = []
    if n_classes:
        counts = list(outcome.class_distribution.values())
        if min(counts) > 0 and max(counts) / min(counts) > CLASS_IMBALANCE_WARNING_RATIO:
            warnings.append(
                f"Classes déséquilibrées (ratio {max(counts) / min(counts):.1f}x entre la plus grande et la plus petite classe)"
            )
    if outcome.corrupted_files:
        warnings.append(f"{len(outcome.corrupted_files)} image(s) corrompue(s) ou illisible(s) exclue(s)")
    if outcome.n_duplicates_removed:
        warnings.append(
            f"{outcome.n_duplicates_removed} image(s) en double (empreinte SHA-256 identique) exclue(s) — "
            "une seule copie conservée par doublon, jamais des deux côtés d'un split"
        )
    if outcome.label_conflicts:
        n_conflict_images = sum(len(c["paths"]) for c in outcome.label_conflicts)
        details = "; ".join(" / ".join(c["categories"]) for c in outcome.label_conflicts)
        warnings.append(
            f"{n_conflict_images} image(s) identique(s) trouvée(s) dans des classes différentes "
            f"({details}) — toutes les copies exclues (conflit d'étiquette, pas un doublon)"
        )
    if outcome.undersized_files:
        warnings.append(
            f"{len(outcome.undersized_files)} image(s) trop petite(s) (< {MIN_IMAGE_DIMENSION_PX}px) exclue(s)"
        )

    return VisionDatasetReport(
        structure_type=structure_type,
        n_images=outcome.n_valid,
        n_classes=n_classes,
        class_distribution=outcome.class_distribution,
        n_corrupted=len(outcome.corrupted_files),
        corrupted_files=outcome.corrupted_files,
        n_duplicates_removed=outcome.n_duplicates_removed,
        duplicate_removed_files=outcome.duplicate_removed_files,
        label_conflicts=outcome.label_conflicts,
        duplicate_detection_note=(
            "La détection de doublons repère uniquement les fichiers strictement identiques "
            "(même empreinte SHA-256) : une image recadrée, redimensionnée, recompressée ou "
            "réenregistrée dans un autre format n'est pas détectée comme doublon."
        ),
        n_undersized=len(outcome.undersized_files),
        undersized_files=outcome.undersized_files,
        warnings=warnings,
    )
