"""Catalogue central des codes d'erreur — Phase 3 (AUDIT_BACKEND_2026-08-23.md,
Axe I, §5), mis à jour lors du chantier de migration (2026-09-02, voir
`_backend/RAPPORT-FINAL.md`, "ce qui a été laissé de côté").

Avant la Phase 3 : des dizaines de codes distincts (`"code": "..."`) semés
comme des littéraux indépendants dans les fichiers routeurs, sans aucun
point de vérité — certains recopiés à l'identique jusqu'à 19 fois
(`RESULTAT_INDISPONIBLE`), avec le risque qu'une future faute de frappe
dans une seule des copies fasse silencieusement diverger deux domaines
qui étaient censés renvoyer le même code pour la même situation. Aucun
endroit où lister "tous les codes d'erreur que cette API peut renvoyer"
— ni pour un développeur frontend, ni pour la doc OpenAPI (`/openapi.json`
ne portait aucune trace de ces codes, entièrement absents du schéma
malgré `responses={"detail": {"code","message"}}` systématique sur toute
erreur, voir `api/main.py`).

Ce module établit le POINT DE VÉRITÉ : la liste ci-dessous est exhaustive
(revérifiée le 2026-09-02, 17 codes manquants ajoutés — apparus depuis la
Phase 3 dans des domaines développés ensuite, comme `SEUIL_INCONNU`) par
`grep -rhoE '"code":\\s*"[A-Z_0-9]+"' api domains`, recoupée avec les 6
codes synthétisés par les gestionnaires d'erreur globaux d'`api/main.py`
(jamais trouvés par ce grep puisqu'ils référencent déjà `ErrorCode.XXX`,
pas un littéral). `api/main.py::custom_openapi` l'expose dans
`/openapi.json` (extension `x-error-codes`) — consultable par n'importe
quel client sans avoir à parcourir le code source.

Migration TERMINÉE (2026-09-03) des littéraux existants vers
`ErrorCode.XXX` : plus un seul `"code": "..."` dupliqué dans `api` ou
`domains` (vérifié par la même commande grep que ci-dessus, filtrée aux
codes apparaissant ≥ 2 fois — résultat vide). 5 lots, chacun vérifié
individuellement (ruff + mypy comparés ligne à ligne contre le HEAD
précédent + suite de tests complète rejouée avant chaque commit), du
code le plus dupliqué au moins dupliqué :
- `RESULTAT_INDISPONIBLE` (19 sites, 6 fichiers) — migré 2026-09-02
  (1er lot).
- `DATASET_LECTURE_ECHEC` (19), `MODELE_NON_DISPONIBLE` (17),
  `ARTEFACT_INTROUVABLE` (12), `DATASET_NON_PRET` (11),
  `DATASET_INTROUVABLE` (10) — migrés 2026-09-02 (2ᵉ lot, 69 sites, 7
  fichiers).
- `JOB_NON_ANNULABLE`, `COLONNES_INCONNUES`, `COLONNE_INTROUVABLE`,
  `ARTEFACT_ILLISIBLE` — migrés 2026-09-03 (3ᵉ lot, 21 sites).
- `VISION_DATASET_NON_PRET`, `VISION_DATASET_INTROUVABLE`,
  `TRAINING_JOB_INTROUVABLE`, `TAILLE_IMAGE_INCONNUE`,
  `NOTATION_IMPOSSIBLE`, `MODELE_INCONNU`, `COLONNES_MANQUANTES`,
  `AUGMENTATION_PRESET_INCONNU` — migrés 2026-09-03 (4ᵉ lot, 24 sites,
  7 fichiers : anomalies, clustering, dimensionality, training,
  vision/anomalies, vision/classification, vision/datasets).
- `VISION_DATASET_STRUCTURE_INVALIDE`, `VISION_CLASSIFICATION_JOB_INTROUVABLE`,
  `VISION_ANOMALY_JOB_INTROUVABLE`, `RESULTAT_INTROUVABLE`,
  `PROJECTION_IMPOSSIBLE`, `PREDICTION_LOT_INTROUVABLE`,
  `IMAGE_INVALIDE`, `HYPERPARAMETRE_INVALIDE`,
  `DIMENSIONALITY_JOB_INTROUVABLE`, `DATASET_TROP_VOLUMINEUX`,
  `DATASET_FORMAT_NON_SUPPORTE`, `DATASET_FICHIER_VIDE`,
  `COMPARATIF_MODELES_INVALIDE`, `COMPARATIF_BACKBONES_INVALIDE`,
  `CLUSTERING_JOB_INTROUVABLE`, `BACKBONE_INCONNU`,
  `AUTH_UTILISATEUR_INTROUVABLE_OU_DESACTIVE`,
  `AUTH_RESET_TOKEN_INVALIDE`, `AUTH_MDP_TROP_FAIBLE`,
  `AUTH_EMAIL_DEJA_UTILISE`, `ASSIGNATION_IMPOSSIBLE`,
  `ANOMALY_JOB_INTROUVABLE` — migrés 2026-09-03 (5ᵉ et dernier lot,
  44 sites, 8 fichiers : les 7 du lot précédent + auth et datasets).
- Au total : 40 codes migrés sur ces 5 lots, ~177 sites au moment de
  chaque migration (le compte vivant continue d'augmenter ensuite avec
  les fonctionnalités ajoutées après-coup, ex. `RESULTAT_INDISPONIBLE`
  compte 22 sites aujourd'hui).

Les codes à un seul site (jamais dupliqués) restent volontairement des
littéraux `"code": "XXX"` — les migrer n'éliminerait aucun risque de
divergence puisqu'il n'existe qu'une seule copie à faire diverger."""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Chaque valeur est EXACTEMENT la chaîne déjà utilisée en production
    dans les `HTTPException(detail={"code": ...})` existants — aucun code
    renommé par ce correctif (aurait cassé la compatibilité avec le
    frontend déjà déployé, voir DECISIONS.md sur la rétrocompatibilité par
    défaut)."""

    # ── Global (api/main.py, gestionnaires d'erreur) ────────────────────
    ERREUR_INTERNE = "ERREUR_INTERNE"
    ERREUR_HTTP = "ERREUR_HTTP"
    VALIDATION_ECHOUEE = "VALIDATION_ECHOUEE"
    AUTH_NON_AUTHENTIFIE = "AUTH_NON_AUTHENTIFIE"
    NON_TROUVE = "NON_TROUVE"
    METHODE_NON_AUTORISEE = "METHODE_NON_AUTORISEE"
    CORPS_TROP_VOLUMINEUX = "CORPS_TROP_VOLUMINEUX"
    TROP_DE_REQUETES = "TROP_DE_REQUETES"

    # ── Authentification / équipe (domains/auth) ────────────────────────
    AUTH_COMPTE_DESACTIVE = "AUTH_COMPTE_DESACTIVE"
    AUTH_EMAIL_DEJA_UTILISE = "AUTH_EMAIL_DEJA_UTILISE"
    AUTH_IDENTIFIANTS_INCORRECTS = "AUTH_IDENTIFIANTS_INCORRECTS"
    AUTH_MDP_ACTUEL_INCORRECT = "AUTH_MDP_ACTUEL_INCORRECT"
    AUTH_MDP_TROP_FAIBLE = "AUTH_MDP_TROP_FAIBLE"
    AUTH_OWNER_REQUIS = "AUTH_OWNER_REQUIS"
    AUTH_RESET_TOKEN_INVALIDE = "AUTH_RESET_TOKEN_INVALIDE"
    AUTH_TOKEN_INVALIDE = "AUTH_TOKEN_INVALIDE"
    AUTH_TROP_DE_TENTATIVES = "AUTH_TROP_DE_TENTATIVES"
    AUTH_UTILISATEUR_INTROUVABLE_OU_DESACTIVE = "AUTH_UTILISATEUR_INTROUVABLE_OU_DESACTIVE"
    ASSIGNATION_IMPOSSIBLE = "ASSIGNATION_IMPOSSIBLE"

    # ── Datasets tabulaires (domains/datasets + partagé par les jobs) ───
    DATASET_INTROUVABLE = "DATASET_INTROUVABLE"
    DATASET_NON_PRET = "DATASET_NON_PRET"
    DATASET_LECTURE_ECHEC = "DATASET_LECTURE_ECHEC"
    DATASET_FICHIER_VIDE = "DATASET_FICHIER_VIDE"
    DATASET_FORMAT_NON_SUPPORTE = "DATASET_FORMAT_NON_SUPPORTE"
    DATASET_TROP_VOLUMINEUX = "DATASET_TROP_VOLUMINEUX"
    COLONNES_INCONNUES = "COLONNES_INCONNUES"
    COLONNES_MANQUANTES = "COLONNES_MANQUANTES"
    COLONNE_CIBLE_INTROUVABLE = "COLONNE_CIBLE_INTROUVABLE"
    COLONNE_GROUPE_INTROUVABLE = "COLONNE_GROUPE_INTROUVABLE"
    COLONNE_INCONNUE = "COLONNE_INCONNUE"
    COLONNE_INTROUVABLE = "COLONNE_INTROUVABLE"
    FEATURE_NON_NUMERIQUE = "FEATURE_NON_NUMERIQUE"
    REPARTITION_INVALIDE = "REPARTITION_INVALIDE"
    TRANSFORMATION_INCONNUE = "TRANSFORMATION_INCONNUE"

    # ── Datasets vision (domains/vision/datasets) ───────────────────────
    VISION_DATASET_INTROUVABLE = "VISION_DATASET_INTROUVABLE"
    VISION_DATASET_NON_PRET = "VISION_DATASET_NON_PRET"
    VISION_DATASET_FICHIER_VIDE = "VISION_DATASET_FICHIER_VIDE"
    VISION_DATASET_FORMAT_NON_SUPPORTE = "VISION_DATASET_FORMAT_NON_SUPPORTE"
    VISION_DATASET_TROP_VOLUMINEUX = "VISION_DATASET_TROP_VOLUMINEUX"
    VISION_DATASET_STRUCTURE_INVALIDE = "VISION_DATASET_STRUCTURE_INVALIDE"
    AUCUNE_IMAGE_POUR_APERCU = "AUCUNE_IMAGE_POUR_APERCU"

    # ── Jobs — communs aux 6 domaines d'entraînement/analyse ────────────
    JOB_NON_ANNULABLE = "JOB_NON_ANNULABLE"
    ARTEFACT_INTROUVABLE = "ARTEFACT_INTROUVABLE"
    ARTEFACT_ILLISIBLE = "ARTEFACT_ILLISIBLE"
    RESULTAT_INDISPONIBLE = "RESULTAT_INDISPONIBLE"
    RESULTAT_INTROUVABLE = "RESULTAT_INTROUVABLE"
    MODELE_NON_DISPONIBLE = "MODELE_NON_DISPONIBLE"
    MODELE_INCONNU = "MODELE_INCONNU"
    MODELES_INCONNUS = "MODELES_INCONNUS"
    MODELE_NON_SELECTIONNE = "MODELE_NON_SELECTIONNE"
    ALGORITHME_INCONNU = "ALGORITHME_INCONNU"
    ALGORITHMES_INCONNUS = "ALGORITHMES_INCONNUS"
    AUCUN_MODELE_COMPATIBLE = "AUCUN_MODELE_COMPATIBLE"
    COMPARAISON_INSUFFISANTE = "COMPARAISON_INSUFFISANTE"
    COMPARATIF_MODELES_INVALIDE = "COMPARATIF_MODELES_INVALIDE"
    COMPARATIF_BACKBONES_INVALIDE = "COMPARATIF_BACKBONES_INVALIDE"
    STAGE_INVALIDE = "STAGE_INVALIDE"
    TACHE_NON_SUPPORTEE = "TACHE_NON_SUPPORTEE"
    LIMITE_INDISPONIBLE = "LIMITE_INDISPONIBLE"
    QUOTA_ENTRAINEMENTS_ATTEINT = "QUOTA_ENTRAINEMENTS_ATTEINT"
    FILE_INDISPONIBLE = "FILE_INDISPONIBLE"
    PREDICTION_IMPOSSIBLE = "PREDICTION_IMPOSSIBLE"
    PREDICTION_LOT_INTROUVABLE = "PREDICTION_LOT_INTROUVABLE"
    PREDICTION_LOT_NON_ANNULABLE = "PREDICTION_LOT_NON_ANNULABLE"
    PROJECTION_IMPOSSIBLE = "PROJECTION_IMPOSSIBLE"
    NOTATION_IMPOSSIBLE = "NOTATION_IMPOSSIBLE"
    HYPERPARAMETRE_INCONNU = "HYPERPARAMETRE_INCONNU"
    HYPERPARAMETRE_INVALIDE = "HYPERPARAMETRE_INVALIDE"
    HYPERPARAMETRE_HORS_BORNES = "HYPERPARAMETRE_HORS_BORNES"
    SEUIL_INCONNU = "SEUIL_INCONNU"

    # ── Identifiants "introuvable" par domaine (jamais fusionnés — un
    # attaquant ne doit pas pouvoir distinguer "id invalide" de "id d'un
    # autre type de ressource", mais chaque domaine reste responsable de
    # son propre message contextualisé) ──────────────────────────────────
    TRAINING_JOB_INTROUVABLE = "TRAINING_JOB_INTROUVABLE"
    CLUSTERING_JOB_INTROUVABLE = "CLUSTERING_JOB_INTROUVABLE"
    DIMENSIONALITY_JOB_INTROUVABLE = "DIMENSIONALITY_JOB_INTROUVABLE"
    ANOMALY_JOB_INTROUVABLE = "ANOMALY_JOB_INTROUVABLE"
    VISION_CLASSIFICATION_JOB_INTROUVABLE = "VISION_CLASSIFICATION_JOB_INTROUVABLE"
    VISION_ANOMALY_JOB_INTROUVABLE = "VISION_ANOMALY_JOB_INTROUVABLE"

    # ── Vision — classification/anomalies (spécifique) ──────────────────
    BACKBONE_INCONNU = "BACKBONE_INCONNU"
    CLASSE_INCONNUE = "CLASSE_INCONNUE"
    CLASSE_INTROUVABLE = "CLASSE_INTROUVABLE"
    IMAGE_INTROUVABLE = "IMAGE_INTROUVABLE"
    IMAGE_INVALIDE = "IMAGE_INVALIDE"
    AUGMENTATION_PRESET_INCONNU = "AUGMENTATION_PRESET_INCONNU"
    TAILLE_IMAGE_INCONNUE = "TAILLE_IMAGE_INCONNUE"
    AUCUN_FICHIER = "AUCUN_FICHIER"
    LOT_TROP_GRAND = "LOT_TROP_GRAND"

    # ── Notifications (domains/notifications) ───────────────────────────
    NOTIFICATION_INTROUVABLE = "NOTIFICATION_INTROUVABLE"


#: Description humaine (français), pour l'extension OpenAPI `x-error-codes`
#: (voir `api/main.py::custom_openapi`) — un client (frontend, intégrateur
#: tiers) peut lire `/openapi.json` pour savoir ce qu'un code signifie sans
#: relire le code source Python. Volontairement incomplet (pas encore 1
#: entrée par code de `ErrorCode`) : rempli au fil des besoins réels,
#: jamais un exercice de documentation exhaustive fait d'un coup.
ERROR_CODE_DESCRIPTIONS: dict[str, str] = {
    ErrorCode.ERREUR_INTERNE: "Erreur serveur inattendue — contactez le support avec le request_id fourni.",
    ErrorCode.VALIDATION_ECHOUEE: "La requête ne respecte pas le schéma attendu (voir le détail des champs).",
    ErrorCode.AUTH_NON_AUTHENTIFIE: "Jeton d'authentification absent, invalide ou expiré.",
    ErrorCode.NON_TROUVE: "Aucune route ne correspond à cette URL.",
    ErrorCode.TROP_DE_REQUETES: "Limite de fréquence atteinte — réessayez plus tard.",
    ErrorCode.QUOTA_ENTRAINEMENTS_ATTEINT: "Nombre maximal de jobs actifs simultanés atteint pour l'organisation.",
    ErrorCode.FILE_INDISPONIBLE: "Le service de traitement asynchrone (Redis/RQ) est temporairement indisponible.",
    ErrorCode.DATASET_INTROUVABLE: "Le dataset demandé n'existe pas ou n'appartient pas à votre organisation.",
    ErrorCode.DATASET_NON_PRET: "Le dataset n'a pas encore terminé son traitement d'import.",
    ErrorCode.JOB_NON_ANNULABLE: "Ce job n'est plus dans un état permettant l'annulation (déjà terminé/échoué).",
    ErrorCode.MODELE_NON_DISPONIBLE: "Ce job n'a pas encore produit de modèle exploitable.",
}
