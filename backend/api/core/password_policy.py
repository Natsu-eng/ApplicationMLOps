"""Règle de robustesse des mots de passe — un seul point de vérité côté
serveur (Phase 1B, AUDIT_BACKEND_2026-08-23.md, point 5 « ce qu'il faut
faire mieux que CIAM ») : avant ce correctif, la seule contrainte réelle
était `min_length=8` côté Pydantic, appliquée de façon éparpillée
(`RegisterRequest`, `TeamMemberCreate`, `ChangePasswordRequest`
définissaient chacun le même `Field(..., min_length=8)`). Le frontend
affiche une jauge de robustesse (`PasswordStrengthMeter.tsx`) qui reste
volontairement non bloquante — la seule règle qui *bloque* doit rester
identique des deux côtés, sans quoi la jauge devient trompeuse.

Cette fonction est appelée partout où un mot de passe est choisi :
inscription, ajout de membre, changement volontaire, confirmation de
réinitialisation (Phase 1B)."""
from __future__ import annotations

_MIN_LENGTH = 8

# Liste volontairement courte — pas un dictionnaire exhaustif de mots de
# passe compromis (hors périmètre, nécessiterait une dépendance externe ou
# un appel réseau type HaveIBeenPwned, non justifié pour ce produit B2B à
# faible volume d'utilisateurs). Vise les cas les plus évidents et les plus
# probables sur un clavier français/anglais.
_COMMON_PASSWORDS = frozenset({
    "password", "password1", "password12",
    "12345678", "123456789", "1234567890",
    "azertyuiop", "qwertyuiop",
    "motdepasse", "motdepasse01",
    "administrateur", "admin12345",
    "bienvenue1", "welcome123",
    "iloveyou1", "letmein123",
    "00000000", "11111111", "12341234",
})


def validate_password_strength(password: str, email: str | None = None) -> None:
    """Lève `ValueError` (message français, actionnable) si le mot de passe
    ne respecte pas la règle minimale — conçue pour être appelée depuis un
    `@model_validator(mode="after")` Pydantic (qui encapsule automatiquement
    un `ValueError` en erreur de validation 422) ou directement depuis un
    router (encapsuler alors dans une `HTTPException` structurée)."""
    if len(password) < _MIN_LENGTH:
        raise ValueError(f"Le mot de passe doit contenir au moins {_MIN_LENGTH} caractères.")
    if password.lower() in _COMMON_PASSWORDS:
        raise ValueError("Ce mot de passe est trop courant — choisissez-en un plus difficile à deviner.")
    if email:
        local_part = email.split("@", 1)[0].lower()
        if len(local_part) >= 3 and local_part in password.lower():
            raise ValueError("Le mot de passe ne doit pas contenir votre adresse e-mail.")
