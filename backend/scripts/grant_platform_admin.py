"""Promeut (ou rétrograde) un compte EXISTANT en administrateur de la
plateforme — le rôle qui donne accès à l'espace `/admin`, seul périmètre
autorisé à lire au-delà d'une organisation.

Usage :
    cd backend && python -m scripts.grant_platform_admin <email>
    cd backend && python -m scripts.grant_platform_admin <email> --revoke
    cd backend && python -m scripts.grant_platform_admin --list

POURQUOI UN SCRIPT, ET PAS UNE MIGRATION NI UN RÉGLAGE.

Une migration qui promeut une adresse en dur s'appliquerait aveuglément à
tous les environnements — y compris une production où ce compte n'a rien à
faire — et graverait une adresse personnelle dans l'historique git, à
jamais. Un réglage d'environnement (`PLATFORM_ADMIN_EMAIL`) promouvrait
silencieusement au démarrage : le jour où la variable est mal renseignée,
personne ne s'en aperçoit.

Ici le geste est explicite, ponctuel, et exécuté par quelqu'un qui a déjà
accès à la base.

CE SCRIPT NE TOUCHE JAMAIS AU MOT DE PASSE, et n'en demande aucun. Le
compte se crée par le parcours normal (inscription, ou invitation), avec
un mot de passe que son titulaire est seul à connaître — c'est tout
l'objet des correctifs de cette session. Un script d'administration qui
saurait fixer un mot de passe serait précisément la porte dérobée qu'on
vient de fermer.
"""
from __future__ import annotations

import argparse
import sys

from api.core.database import SessionLocal
from api.core.models import Organization, User


def _print_admins(db) -> None:
    admins = (
        db.query(User, Organization.name)
        .join(Organization, Organization.id == User.organization_id)
        .filter(User.is_platform_admin.is_(True))
        .order_by(User.created_at)
        .all()
    )
    if not admins:
        print("Aucun administrateur de plateforme.")
        return
    print(f"{len(admins)} administrateur(s) de plateforme :")
    for user, org_name in admins:
        etat = "actif" if user.actif else "ACCÈS RÉVOQUÉ"
        print(f"  - {user.email} ({user.nom}, organisation « {org_name} », {etat})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("email", nargs="?", help="adresse du compte à promouvoir (doit déjà exister)")
    parser.add_argument("--revoke", action="store_true", help="retire le rôle au lieu de l'accorder")
    parser.add_argument("--list", action="store_true", help="liste les administrateurs actuels et sort")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.list:
            _print_admins(db)
            return 0

        if not args.email:
            parser.error("indiquez une adresse e-mail, ou --list")

        user = db.query(User).filter(User.email == args.email).first()
        if user is None:
            # Volontairement explicite : ce script s'exécute en local par
            # quelqu'un qui a la main sur la base, il n'y a aucun secret à
            # protéger ici — contrairement aux endpoints publics, où l'on ne
            # révèle jamais l'existence d'un compte.
            print(f"Aucun compte avec l'adresse {args.email!r}.", file=sys.stderr)
            print("Créez-le d'abord par le parcours normal (inscription ou invitation).", file=sys.stderr)
            return 1
        if user.anonymized_at is not None:
            print("Ce compte a été anonymisé : il ne peut plus être promu.", file=sys.stderr)
            return 1

        souhaite = not args.revoke
        if user.is_platform_admin == souhaite:
            print(f"{user.email} est déjà dans cet état — rien à faire.")
            return 0

        user.is_platform_admin = souhaite
        db.commit()
        verbe = "promu administrateur de la plateforme" if souhaite else "rétrogradé simple utilisateur"
        print(f"{user.email} {verbe}.")
        if souhaite and not user.actif:
            print("ATTENTION : l'accès de ce compte est révoqué, il ne pourra pas se connecter.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
