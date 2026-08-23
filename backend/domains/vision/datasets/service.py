"""Logique pure d'ingestion des datasets d'images (pilier Vision, Lot 15
sous-lot A) — sans dépendance HTTP, directement testable.

Fondation partagée par la classification d'images (sous-lot B) et la
détection d'anomalies visuelles MVTec AD (sous-lot C) : les deux ont besoin
d'ingérer une archive d'images, de détecter sa structure et de la valider.
Écrit une seule fois ici plutôt que dupliqué dans deux endpoints (même
raisonnement que `services/job_quota.py` pour les jobs).

Corrige le bug #1 déjà documenté dans
`docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md` (labels `y_train` d'un
dataset MVTec AD jamais validés avant training, risque de partir en mode
supervisé par accident) : ici la structure est validée AU MOMENT DE
L'UPLOAD, de façon stricte — jamais une supposition silencieuse.

Lot 6A — jusqu'ici .zip strictement obligatoire (contrairement au
téléchargement officiel MVTec AD, distribué en .tar.xz) et aucun moyen
d'importer un dossier déjà déplié sans le re-compresser à la main. Toute
la logique de détection de structure/validation/déduplication
(`_detect_structure`/`_validate_and_copy_images`) opère désormais sur une
liste de `_ExtractedMember` (chemin relatif + contenu déjà en mémoire) —
format-agnostique par construction. Trois sources alimentent cette liste
uniforme : `_extract_zip_members`, `_extract_tar_members`,
`_members_from_uploaded_files` (dossier) — chacune la SEULE responsable de
son format, jamais mélangée au reste du pipeline.
"""
from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_IGNORED_NAME_PREFIXES = ("__MACOSX/", ".")
_IGNORED_FILENAMES = {".ds_store", "thumbs.db"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tar.gz", ".tgz"}

MIN_IMAGE_DIMENSION_PX = 20
MIN_IMAGES_PER_CLASS = 2
MIN_TRAIN_GOOD_IMAGES = 5
CLASS_IMBALANCE_WARNING_RATIO = 10.0


class VisionDatasetError(ValueError):
    """L'archive/le dossier a été reçu(e) mais ne peut pas être ingéré(e)
    comme dataset d'images (structure non reconnue, contenu dangereux,
    aucune image exploitable)."""


class UnsupportedFileType(ValueError):
    """Extension de fichier non supportée pour un dataset vision."""


def validate_archive_extension(filename: str) -> str:
    """Vérification RAPIDE côté nom de fichier (retour immédiat à
    l'utilisateur avant tout upload) — jamais la seule ligne de défense :
    `_extract_archive_members` sniffe le contenu réel (signature
    binaire), une extension mensongère est rejetée là, pas ici."""
    name = filename.lower()
    for ext in sorted(ARCHIVE_EXTENSIONS, key=len, reverse=True):  # ".tar.gz" avant ".gz" implicite
        if name.endswith(ext):
            return ext
    raise UnsupportedFileType(
        f"Le dataset d'images doit être fourni sous forme d'archive ({', '.join(sorted(ARCHIVE_EXTENSIONS))})"
    )


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
    # Lot 6A (AUDIT_DATALAB_2026-08-16.md §G.3/§G.4/§G.5) — distribution des
    # résolutions/formats/modes colorimétriques des images RÉELLEMENT
    # conservées (voir _compute_image_eda).
    image_eda: dict = field(default_factory=dict)


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


@dataclass(frozen=True)
class _ExtractedMember:
    """Une image déjà extraite en mémoire (chemin relatif + contenu) —
    représentation UNIQUE et format-agnostique consommée par
    `_detect_structure`/`_validate_and_copy_images`, quelle que soit la
    source (zip, tar, dossier). Le contenu est matérialisé ICI, à
    l'extraction, jamais relu plus tard depuis un objet zip/tar rouvert :
    `max_uncompressed_bytes` borne déjà la taille totale, matérialiser
    n'aggrave pas le pic mémoire par rapport à l'ancien flux (qui relisait
    de toute façon chaque image entière pour la hasher/valider)."""
    rel_path: PurePosixPath
    content: bytes


def _accumulate_member(
    rel_path: PurePosixPath | None,
    content: bytes,
    members: list[_ExtractedMember],
    total_uncompressed: int,
    max_images: int,
    max_uncompressed_bytes: int,
) -> int:
    """Filtre + garde-fous communs aux 3 sources (extension image, taille
    cumulée, nombre max) — extrait pour ne jamais dupliquer ces 3 vérifications
    entre zip/tar/dossier. Retourne le nouveau total cumulé."""
    if rel_path is None or rel_path.suffix.lower() not in IMAGE_EXTENSIONS:
        return total_uncompressed
    total_uncompressed += len(content)
    if total_uncompressed > max_uncompressed_bytes:
        raise VisionDatasetError(
            f"Archive trop volumineuse une fois décompressée (max {max_uncompressed_bytes // (1024 * 1024)} Mo)"
        )
    members.append(_ExtractedMember(rel_path=rel_path, content=content))
    if len(members) > max_images:
        raise VisionDatasetError(f"Trop d'images dans l'archive (max {max_images})")
    return total_uncompressed


def _finalize_members(members: list[_ExtractedMember]) -> list[_ExtractedMember]:
    if not members:
        raise VisionDatasetError("Aucune image exploitable trouvée dans l'archive (formats acceptés : "
                                  f"{', '.join(sorted(IMAGE_EXTENSIONS))})")
    return members


def _extract_zip_members(content: bytes, max_images: int, max_uncompressed_bytes: int) -> list[_ExtractedMember]:
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise VisionDatasetError(f"Archive ZIP illisible : {exc}") from exc

    members: list[_ExtractedMember] = []
    total_uncompressed = 0
    for info in zf.infolist():
        rel_path = _safe_member_path(info.filename)
        if rel_path is None or rel_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Correctif Phase 1 (AUDIT_BACKEND_2026-08-23.md §C.3) — `info.file_size`
        # (taille décompressée déclarée dans le répertoire central, lue SANS
        # décompresser) est vérifié AVANT `zf.read(info)` : la garde
        # `_accumulate_member` s'appliquait seulement APRÈS avoir déjà
        # matérialisé l'entrée entière en mémoire, donc une archive à UNE
        # seule entrée à ratio de compression extrême pouvait faire exploser
        # la mémoire avant que le rejet n'ait l'occasion de s'appliquer.
        if info.file_size > max_uncompressed_bytes:
            raise VisionDatasetError(
                f"Archive trop volumineuse une fois décompressée (max {max_uncompressed_bytes // (1024 * 1024)} Mo)"
            )
        total_uncompressed = _accumulate_member(
            rel_path, zf.read(info), members, total_uncompressed, max_images, max_uncompressed_bytes
        )
    return _finalize_members(members)


def _extract_tar_members(content: bytes, max_images: int, max_uncompressed_bytes: int) -> list[_ExtractedMember]:
    try:
        # mode="r:*" détecte automatiquement gzip/bzip2/xz ou l'absence de
        # compression depuis le contenu réel — jamais depuis l'extension
        # fournie par le client (défense en profondeur, même principe que
        # le sniff zip vs tar dans _extract_archive_members).
        tf = tarfile.open(fileobj=io.BytesIO(content), mode="r:*")
    except tarfile.TarError as exc:
        raise VisionDatasetError(f"Archive TAR illisible : {exc}") from exc

    members: list[_ExtractedMember] = []
    total_uncompressed = 0
    with tf:
        for info in tf.getmembers():
            # Seuls les fichiers réguliers — jamais un lien symbolique/dur,
            # exploitable pour pointer hors de l'archive (classe de faille
            # distincte du zip-slip, propre au format tar).
            if not info.isfile():
                continue
            rel_path = _safe_member_path(info.name)
            if rel_path is None or rel_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            # Même correctif que _extract_zip_members ci-dessus — `info.size`
            # (taille décompressée déclarée dans l'en-tête tar) est connu
            # sans extraire.
            if info.size > max_uncompressed_bytes:
                raise VisionDatasetError(
                    f"Archive trop volumineuse une fois décompressée (max {max_uncompressed_bytes // (1024 * 1024)} Mo)"
                )
            extracted = tf.extractfile(info)
            if extracted is None:
                continue
            total_uncompressed = _accumulate_member(
                rel_path, extracted.read(), members, total_uncompressed, max_images, max_uncompressed_bytes
            )
    return _finalize_members(members)


def _members_from_uploaded_files(
    files: list[tuple[str, bytes]], max_images: int, max_uncompressed_bytes: int
) -> list[_ExtractedMember]:
    """Import d'un DOSSIER (Lot 6A) — chaque fichier vient du navigateur
    avec son chemin relatif déjà porté par son nom (`webkitRelativePath`,
    voir `POST /vision/datasets` côté routeur) : pas d'archive à ouvrir,
    seulement le même filtrage/mêmes garde-fous que zip/tar."""
    members: list[_ExtractedMember] = []
    total_uncompressed = 0
    for filename, content in files:
        rel_path = _safe_member_path(filename)
        total_uncompressed = _accumulate_member(
            rel_path, content, members, total_uncompressed, max_images, max_uncompressed_bytes
        )
    return _finalize_members(members)


_ZIP_MAGIC = b"PK\x03\x04"
_ZIP_EMPTY_MAGIC = b"PK\x05\x06"  # archive zip vide (aucun fichier) — signature distincte


def _extract_archive_members(content: bytes, max_images: int, max_uncompressed_bytes: int) -> list[_ExtractedMember]:
    """Point d'entrée archive (zip/tar/tar.gz/tgz) — détecte le format
    RÉEL par signature binaire (jamais par l'extension déclarée, qui peut
    mentir) puis délègue à l'extracteur correspondant."""
    if content.startswith(_ZIP_MAGIC) or content.startswith(_ZIP_EMPTY_MAGIC):
        return _extract_zip_members(content, max_images, max_uncompressed_bytes)
    if tarfile.is_tarfile(io.BytesIO(content)):
        return _extract_tar_members(content, max_images, max_uncompressed_bytes)
    raise VisionDatasetError(
        f"Format d'archive non reconnu (formats acceptés : {', '.join(sorted(ARCHIVE_EXTENSIONS))})"
    )


_SPLIT_NAMES = {"train", "test", "val"}


def _detect_structure(members: list[_ExtractedMember]) -> tuple[str, dict[str, list[_ExtractedMember]]]:
    """Détecte "classification" (dossiers de classes) ou "mvtec_ad"
    (train/good + test/good + test/<defaut>) à partir des chemins présents
    dans l'archive/le dossier. Ne devine jamais silencieusement : lève une
    erreur explicite si aucune structure reconnue ne correspond (correctif
    du bug #1 — plus de mode détecté par accident).

    Lot 6A — troisième cas reconnu, testé contre un vrai dataset
    utilisateur (pas seulement des fixtures synthétiques) :
    classification PRÉ-DÉCOUPÉE en train/test(/val), un sous-dossier par
    classe sous chaque split (ex. train/bent_wire/, test/bent_wire/...),
    SANS dossier "good" — donc pas un dataset normal/défaut, malgré la
    même profondeur de 3 niveaux. Distingué de la structure normal/défaut
    par le contenu de train/ : `{"good"}` exactement ⇒ normal/défaut,
    n'importe quoi d'autre ⇒ classification pré-découpée. Les trois
    splits sont FUSIONNÉS par nom de classe (même sac que la
    classification simple) : ce produit n'a pas de notion de split
    figé — `services/vision_classification_training.py` refait de toute
    façon son propre split aléatoire stratifié, respecter le split
    d'origine exigerait une fonctionnalité séparée, hors périmètre ici.

    Retourne le type détecté et les membres groupés par "bucket" (nom de
    classe pour la classification, "train/good"/"test/good"/"test/<x>" pour
    normal/défaut)."""
    top_level_dirs = {m.rel_path.parts[0] for m in members if len(m.rel_path.parts) >= 2}

    # Un téléchargement MVTec AD officiel est zippé PAR CATÉGORIE
    # (ex. bottle.tar.gz → bottle/train/good/..., bottle/test/...) : un seul
    # dossier de plus au-dessus de train/test(/val), jamais l'archive
    # elle-même. Détecté ici plutôt que d'obliger l'utilisateur à re-zipper
    # manuellement le contenu du dossier catégorie (bug réel trouvé en
    # testant avec un vrai dataset MVTec AD, pas seulement des fixtures
    # synthétiques). Généralisé à train/test/val (Lot 6A, pas seulement
    # train/test) pour couvrir aussi la classification pré-découpée.
    split_offset = 0
    effective_split_dirs = top_level_dirs
    if not ({"train", "test"} <= top_level_dirs) and len(top_level_dirs) == 1:
        wrapper_dir = next(iter(top_level_dirs))
        second_level_dirs = {
            m.rel_path.parts[1] for m in members if len(m.rel_path.parts) >= 3 and m.rel_path.parts[0] == wrapper_dir
        }
        if second_level_dirs <= _SPLIT_NAMES and {"train", "test"} <= second_level_dirs:
            split_offset = 1
            effective_split_dirs = second_level_dirs

    is_split_layout = effective_split_dirs <= _SPLIT_NAMES and {"train", "test"} <= effective_split_dirs

    if is_split_layout:
        raw_buckets: dict[tuple[str, str], list[_ExtractedMember]] = {}
        for member in members:
            parts = member.rel_path.parts[split_offset:]
            if len(parts) != 3:
                raise VisionDatasetError(
                    f"Structure invalide : '{member.rel_path}' doit être directement sous "
                    "train/<categorie>/, test/<categorie>/ ou val/<categorie>/ (pas de sous-dossier supplémentaire)"
                )
            split, category = parts[0], parts[1]
            if split not in _SPLIT_NAMES:
                raise VisionDatasetError(
                    f"Dossier de premier niveau inattendu : '{split}' (attendu parmi train/test/val)"
                )
            raw_buckets.setdefault((split, category), []).append(member)

        train_categories = {cat for (split, cat) in raw_buckets if split == "train"}

        if train_categories == {"good"}:
            # ─── Structure normal/défaut (validation stricte inchangée) ───
            buckets = {f"{split}/{cat}": items for (split, cat), items in raw_buckets.items()}

            if len(buckets.get("train/good", [])) < MIN_TRAIN_GOOD_IMAGES:
                raise VisionDatasetError(
                    f"Structure normal/défaut invalide : train/good/ doit contenir au moins "
                    f"{MIN_TRAIN_GOOD_IMAGES} images normales (trouvé {len(buckets.get('train/good', []))})"
                )
            test_categories = {b.split("/", 1)[1] for b in buckets if b.startswith("test/")}
            if "good" not in test_categories:
                raise VisionDatasetError(
                    "Structure normal/défaut invalide : test/good/ est requis (nécessaire pour calibrer le seuil "
                    "de détection, voir docs/legacy/ANALYSE_COMPLETE_COMPUTER_VISION.md #12)"
                )
            if len(test_categories) < 2:
                raise VisionDatasetError(
                    "Structure normal/défaut invalide : test/ doit contenir au moins une catégorie de défaut "
                    "en plus de good/ (ex. test/scratch/, test/crack/)"
                )
            return "mvtec_ad", buckets

        # ─── Classification pré-découpée en splits — fusionnée par classe ───
        buckets = {}
        for (split, category), items in raw_buckets.items():
            buckets.setdefault(category, []).extend(items)
        if len(buckets) < 2:
            raise VisionDatasetError(
                "Structure non reconnue : les dossiers train/test(/val) ne contiennent pas assez de "
                "classes distinctes (minimum 2) pour une classification pré-découpée"
            )
        return "classification", buckets

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
            if {
                m.rel_path.parts[1] for m in members if len(m.rel_path.parts) >= 3 and m.rel_path.parts[0] == d
            } >= {"train", "test"}
        ]
        if len(categories_with_train_test) >= 2:
            raise VisionDatasetError(
                "Cette archive contient plusieurs catégories normal/défaut à la fois "
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
            m.rel_path.parts[1] for m in members if len(m.rel_path.parts) >= 3 and m.rel_path.parts[0] == wrapper_dir
        }
        if len(candidate_classes) >= 2:
            classification_offset = 1

    buckets = {}
    for member in members:
        parts = member.rel_path.parts[classification_offset:]
        if len(parts) != 2:
            raise VisionDatasetError(
                f"Structure de classification invalide : '{member.rel_path}' doit être directement sous "
                "<classe>/ (pas de sous-dossier supplémentaire, pas de fichier à la racine)"
            )
        class_name = parts[0]
        buckets.setdefault(class_name, []).append(member)

    if len(buckets) < 2:
        raise VisionDatasetError(
            "Structure non reconnue : fournissez soit au moins 2 dossiers de classes "
            "(classification), soit une structure train/good + test/good + test/<defaut> (normal/défaut)"
        )
    return "classification", buckets


@dataclass
class _ValidImage:
    """Une image ayant passé les contrôles d'intégrité (pas corrompue, pas
    sous-dimensionnée) — étape intermédiaire nécessaire avant de décider
    quelle copie garder en cas de doublon : cette décision doit voir TOUTES
    les occurrences du même hash dans l'archive/le dossier entier, y compris
    dans un bucket pas encore parcouru, donc impossible en un seul passage
    streaming comme avant ce correctif (copie immédiate sur disque)."""
    rel_path: PurePosixPath
    content: bytes
    bucket_name: str
    digest: str
    # Capturés gratuitement pendant l'ouverture déjà nécessaire à la
    # validation d'intégrité ci-dessous (Lot 6A, EDA d'images, §G.3/G.4) —
    # jamais une seconde passe de lecture disque dédiée.
    width: int
    height: int
    format: str | None
    mode: str


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
    image_eda: dict


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
    buckets: dict[str, list[_ExtractedMember]],
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
    corrupted_files: list[str] = []
    undersized_files: list[str] = []
    valid_images: list[_ValidImage] = []

    for bucket_name, entries in buckets.items():
        for member in entries:
            raw = member.content
            try:
                img = Image.open(io.BytesIO(raw))
                img.verify()
                # verify() invalide l'objet pour un usage ultérieur — réouverture
                # nécessaire pour lire les dimensions réelles (docs Pillow).
                img = Image.open(io.BytesIO(raw))
                width, height = img.size
                img.load()
            except (UnidentifiedImageError, OSError, ValueError):
                corrupted_files.append(str(member.rel_path))
                continue

            if width < MIN_IMAGE_DIMENSION_PX or height < MIN_IMAGE_DIMENSION_PX:
                undersized_files.append(str(member.rel_path))
                continue

            digest = hashlib.sha256(raw).hexdigest()
            valid_images.append(
                _ValidImage(
                    rel_path=member.rel_path,
                    content=raw,
                    bucket_name=bucket_name,
                    digest=digest,
                    width=width,
                    height=height,
                    format=img.format,
                    mode=img.mode,
                )
            )

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
        dest = target_dir / vi.bucket_name / vi.rel_path.name
        if dest.exists():
            # Collision entre deux chemins SOURCE distincts partageant le
            # même nom de fichier final — possible depuis la classification
            # pré-découpée (Lot 6A) qui fusionne train/test/val dans la
            # même classe (ex. train/chat/003.png et test/chat/003.png) :
            # jamais un écrasement silencieux, désambiguïsé par un suffixe.
            stem, suffix = vi.rel_path.stem, vi.rel_path.suffix
            counter = 1
            while dest.exists():
                dest = target_dir / vi.bucket_name / f"{stem}__{counter}{suffix}"
                counter += 1
        dest.write_bytes(vi.content)
        n_valid += 1

    kept_images = [vi for vi in valid_images if str(vi.rel_path) not in all_excluded]

    return _ValidationOutcome(
        n_valid=n_valid,
        corrupted_files=corrupted_files,
        undersized_files=undersized_files,
        class_distribution=class_distribution,
        class_distribution_before_dedup=class_distribution_before_dedup,
        n_duplicates_removed=len(excluded_as_duplicate),
        duplicate_removed_files=sorted(excluded_as_duplicate),
        label_conflicts=label_conflicts,
        image_eda=_compute_image_eda(kept_images),
    )


# Bornes en pixels (dimension la plus grande de l'image) — mêmes seuils que
# les tailles d'entrée réelles des deux pipelines d'entraînement (128 pour
# l'autoencodeur d'anomalies, 224 pour la classification, voir IMAGE_SIZE
# dans vision_anomaly_registry.py/vision_classification_training.py) : les
# bornes intermédiaires (128/224/512) parlent directement à la question que
# l'utilisateur se pose ("mes images sont-elles assez grandes pour ce que
# le modèle va en faire, ou juste agrandies pour rien ?"), pas des paliers
# arbitraires.
_RESOLUTION_BUCKET_BOUNDS = (128, 224, 512, 1024)


def _resolution_bucket_label(max_dimension: int) -> str:
    if max_dimension < _RESOLUTION_BUCKET_BOUNDS[0]:
        return f"< {_RESOLUTION_BUCKET_BOUNDS[0]}px"
    for lo, hi in zip(_RESOLUTION_BUCKET_BOUNDS, _RESOLUTION_BUCKET_BOUNDS[1:]):
        if max_dimension < hi:
            return f"{lo}-{hi}px"
    return f">= {_RESOLUTION_BUCKET_BOUNDS[-1]}px"


def _compute_image_eda(kept_images: list[_ValidImage]) -> dict:
    """EDA d'images (Lot 6A, AUDIT_DATALAB_2026-08-16.md §G.3/§G.4/§G.5) —
    distribution des résolutions/formats/modes colorimétriques, jusqu'ici
    totalement absente (seul un minimum de `MIN_IMAGE_DIMENSION_PX` était
    vérifié, jamais montré). Calculée sur les images RÉELLEMENT conservées
    (après exclusion doublons/conflits) — un histogramme qui inclurait des
    images finalement exclues du dataset induirait l'utilisateur en erreur
    sur ce qui sera vraiment utilisé à l'entraînement. `s'appuie sur
    width/height/format/mode déjà capturés dans `_ValidImage`, jamais une
    seconde lecture disque des images."""
    if not kept_images:
        return {
            "resolution_buckets": {},
            "width": {"min": None, "max": None, "mean": None},
            "height": {"min": None, "max": None, "mean": None},
            "format_distribution": {},
            "color_mode_distribution": {},
        }

    widths = [vi.width for vi in kept_images]
    heights = [vi.height for vi in kept_images]

    resolution_buckets: dict[str, int] = {}
    for vi in kept_images:
        label = _resolution_bucket_label(max(vi.width, vi.height))
        resolution_buckets[label] = resolution_buckets.get(label, 0) + 1

    format_distribution: dict[str, int] = {}
    for vi in kept_images:
        fmt = vi.format or "inconnu"
        format_distribution[fmt] = format_distribution.get(fmt, 0) + 1

    color_mode_distribution: dict[str, int] = {}
    for vi in kept_images:
        color_mode_distribution[vi.mode] = color_mode_distribution.get(vi.mode, 0) + 1

    return {
        "resolution_buckets": resolution_buckets,
        "width": {"min": min(widths), "max": max(widths), "mean": round(sum(widths) / len(widths), 1)},
        "height": {"min": min(heights), "max": max(heights), "mean": round(sum(heights) / len(heights), 1)},
        "format_distribution": format_distribution,
        "color_mode_distribution": color_mode_distribution,
    }


def analyze_and_extract_vision_archive(
    content: bytes,
    target_dir: Path,
    max_images: int,
    max_uncompressed_bytes: int,
) -> VisionDatasetReport:
    """Point d'entrée archive (zip/tar/tar.gz/tgz, Lot 6A — voir
    `_extract_archive_members` pour le sniff de format) — valide, détecte
    la structure, déduplique et copie les images retenues vers
    `target_dir`. Lève `VisionDatasetError` pour tout problème structurel
    (jamais pour une image individuelle corrompue ou un doublon, qui sont
    simplement exclus et reportés)."""
    members = _extract_archive_members(content, max_images, max_uncompressed_bytes)
    return _analyze_and_extract(members, target_dir)


def analyze_and_extract_vision_folder(
    files: list[tuple[str, bytes]],
    target_dir: Path,
    max_images: int,
    max_uncompressed_bytes: int,
) -> VisionDatasetReport:
    """Point d'entrée dossier (Lot 6A) — `files` : liste (chemin relatif,
    contenu) telle qu'envoyée par le navigateur pour un import de dossier
    (`webkitRelativePath` porté par chaque fichier, voir
    `POST /vision/datasets` côté routeur). Mêmes garanties que
    `analyze_and_extract_vision_archive`, seule la source diffère."""
    members = _members_from_uploaded_files(files, max_images, max_uncompressed_bytes)
    return _analyze_and_extract(members, target_dir)


def _analyze_and_extract(members: list[_ExtractedMember], target_dir: Path) -> VisionDatasetReport:
    """Cœur commun aux deux points d'entrée ci-dessus — détecte la
    structure, valide, déduplique, copie. Format-agnostique par
    construction (`_ExtractedMember` ne porte plus aucune trace de sa
    source zip/tar/dossier à ce stade)."""
    structure_type, buckets = _detect_structure(members)

    outcome = _validate_and_copy_images(buckets, target_dir, structure_type)

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
        image_eda=outcome.image_eda,
    )
