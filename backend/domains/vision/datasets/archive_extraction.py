"""Extraction sûre d'images depuis une archive ou un dossier (pilier Vision).

Couche la PLUS BASSE de l'ingestion d'un dataset d'images : transformer des
octets reçus du client en une liste uniforme de `_ExtractedMember` (chemin
relatif + contenu en mémoire), format-agnostique, que la suite du pipeline
(`service.py` : détection de structure, validation, EDA) consomme sans
jamais savoir d'où elle vient.

Extrait de `service.py` lors du découpage du fichier (822 lignes) : ce
bloc est cohérent et autonome — il ne dépend d'aucune notion de
"structure de dataset", seulement de "lire des octets sans se faire
piéger". C'est aussi le bloc qui concentre les défenses de sécurité de
l'ingestion, et les regrouper les rend auditables d'un seul tenant :

- zip-slip / chemins absolus ou `..` (`_safe_member_path`) ;
- liens symboliques et durs dans un tar, qui peuvent pointer hors de
  l'archive — classe de faille distincte du zip-slip, propre au format
  tar (`_extract_tar_members`, filtre `info.isfile()`) ;
- bombe de décompression, vérifiée sur la taille DÉCLARÉE avant toute
  matérialisation en mémoire, puis sur le cumul réel (correctif Phase 1,
  AUDIT_BACKEND_2026-08-23.md §C.3) ;
- format réel déterminé par signature binaire, jamais par l'extension
  fournie par le client, qui peut mentir (`_extract_archive_members`).

Trois sources alimentent la liste uniforme — `_extract_zip_members`,
`_extract_tar_members`, `_members_from_uploaded_files` (dossier) — chacune
SEULE responsable de son format, jamais mélangée au reste du pipeline.
"""
from __future__ import annotations

import io
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_IGNORED_NAME_PREFIXES = ("__MACOSX/", ".")
_IGNORED_FILENAMES = {".ds_store", "thumbs.db"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tar.gz", ".tgz"}


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
