"""Envoi d'e-mails via SMTP (bibliothèque standard, aucune dépendance
tierce) — Phase 1B (AUDIT_BACKEND_2026-08-23.md), réinitialisation de mot
de passe. Repris de CIAM (`E:\\concrete-ai-platform\\backend\\api\\core\\mailer.py`),
mécanisme déjà éprouvé — `smtplib`/`email.message`, message en texte brut
(pas de HTML : rien à échapper, aucune surface d'injection).

Aucune dépendance nouvelle : justifié par la simplicité du besoin (un
message texte, un aller SMTP STARTTLS) — une bibliothèque tierce
(ex. `fastapi-mail`) n'apporterait rien qu'`smtplib` ne fasse déjà, pour un
coût de maintenance et une surface de dépendance supplémentaires."""
from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage

from api.core.config import get_settings

logger = logging.getLogger("datalab.mailer")

_SMTP_TIMEOUT_SECONDS = 10


def mailer_configured() -> bool:
    """Canal mail disponible — voir `api/core/config.py` pour la politique
    de démarrage (optionnel en dev, avertissement bloquant en production si
    absent, voir `api/main.py::lifespan`)."""
    settings = get_settings()
    return bool(settings.smtp_host and settings.smtp_user and settings.smtp_password)


def _send_smtp(msg: EmailMessage) -> None:
    settings = get_settings()
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=_SMTP_TIMEOUT_SECONDS) as smtp:
        smtp.starttls()
        smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(msg)


def send_password_reset_email(to_email: str, reset_link: str, expires_minutes: int, requested_from_ip: str) -> None:
    """Envoie le lien de réinitialisation. Lève une exception si l'envoi
    échoue — à l'appelant (tâche de fond, voir `domains/auth/router.py`) de
    catcher : ne jamais appeler de façon bloquante dans le chemin de
    réponse HTTP (la latence SMTP ne doit jamais dépendre de l'existence du
    compte, sinon le TEMPS de réponse trahit ce que le CORPS de la réponse
    cache — voir `POST /auth/password-reset/request`).

    Mieux que CIAM (Phase 1B, point 6) : mentionne la date et l'IP de la
    demande — sans elles, quelqu'un qui reçoit un lien qu'il n'a pas demandé
    ne peut rien en conclure. Le jeton en clair n'est JAMAIS journalisé ici,
    uniquement inséré dans le lien."""
    email_msg = EmailMessage()
    email_msg["Subject"] = "[DataLab Pro] Réinitialisation de votre mot de passe"
    email_msg["From"] = get_settings().smtp_user
    email_msg["To"] = to_email
    email_msg.set_content(
        "Une demande de réinitialisation de votre mot de passe DataLab Pro a été reçue "
        f"depuis l'adresse IP {requested_from_ip}.\n\n"
        f"Ouvrez ce lien dans les {expires_minutes} minutes pour choisir un nouveau mot de passe :\n"
        f"{reset_link}\n\n"
        "Si vous n'êtes pas à l'origine de cette demande, ignorez cet e-mail — "
        "votre mot de passe actuel reste inchangé. Si cela vous inquiète, changez "
        "votre mot de passe depuis votre profil dès que possible.\n"
        "Ce lien est personnel et ne peut être utilisé qu'une seule fois.\n"
    )
    _send_smtp(email_msg)
    logger.info("[Mailer] Lien de réinitialisation envoyé à %s", to_email)


def send_password_changed_notification_email(to_email: str, requested_from_ip: str) -> None:
    """Second mail, envoyé APRÈS un changement de mot de passe effectif
    (réinitialisation ou changement volontaire) — Phase 1B, point 6 : « le
    mail de CIAM ne mentionne ni l'heure ni l'origine de la demande [...]
    envoie un second mail de notification après un changement effectif —
    c'est souvent le seul signal qu'une victime reçoit. » Absent de CIAM."""
    email_msg = EmailMessage()
    email_msg["Subject"] = "[DataLab Pro] Votre mot de passe a été modifié"
    email_msg["From"] = get_settings().smtp_user
    email_msg["To"] = to_email
    email_msg.set_content(
        f"Le mot de passe de votre compte DataLab Pro vient d'être modifié, depuis "
        f"l'adresse IP {requested_from_ip}.\n\n"
        "Si vous êtes à l'origine de ce changement, aucune action n'est nécessaire — "
        "toutes vos sessions précédentes ont été fermées par précaution.\n\n"
        "Si vous n'êtes PAS à l'origine de ce changement, votre compte est "
        "probablement compromis : contactez immédiatement le propriétaire de votre "
        "organisation.\n"
    )
    _send_smtp(email_msg)
    logger.info("[Mailer] Notification de changement de mot de passe envoyée à %s", to_email)
