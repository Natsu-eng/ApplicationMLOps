"""api/core/mailer.py — construction des messages, sans connexion SMTP
réelle (`smtplib.SMTP` mocké). Complète `test_password_reset.py`, qui ne
peut pas exercer ce module car le canal SMTP est désactivé pour toute la
suite de tests (voir conftest.py — un test ne doit jamais dépendre d'un
service tiers réel)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from api.core import mailer


def _configure_smtp(monkeypatch):
    settings = mailer.get_settings()
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_port", 587)
    monkeypatch.setattr(settings, "smtp_user", "no-reply@example.com")
    monkeypatch.setattr(settings, "smtp_password", "un-mot-de-passe-application")


def test_mailer_configured_false_without_smtp_settings(monkeypatch):
    settings = mailer.get_settings()
    monkeypatch.setattr(settings, "smtp_host", "")
    assert mailer.mailer_configured() is False


def test_mailer_configured_true_with_full_smtp_settings(monkeypatch):
    _configure_smtp(monkeypatch)
    assert mailer.mailer_configured() is True


def test_send_password_reset_email_includes_link_expiry_and_ip(monkeypatch):
    _configure_smtp(monkeypatch)
    with patch("smtplib.SMTP") as mock_smtp_cls:
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_smtp

        mailer.send_password_reset_email(
            "victime@bureau.fr", "https://app.example.com/reset-password?token=abc123", 30, "203.0.113.42"
        )

    mock_smtp.starttls.assert_called_once()
    mock_smtp.login.assert_called_once_with("no-reply@example.com", "un-mot-de-passe-application")
    assert mock_smtp.send_message.call_count == 1
    sent_message = mock_smtp.send_message.call_args[0][0]
    assert sent_message["To"] == "victime@bureau.fr"
    body = sent_message.get_content()
    assert "https://app.example.com/reset-password?token=abc123" in body
    assert "203.0.113.42" in body
    assert "30 minutes" in body


def test_send_password_changed_notification_email_includes_ip(monkeypatch):
    _configure_smtp(monkeypatch)
    with patch("smtplib.SMTP") as mock_smtp_cls:
        mock_smtp = MagicMock()
        mock_smtp_cls.return_value.__enter__.return_value = mock_smtp

        mailer.send_password_changed_notification_email("victime@bureau.fr", "198.51.100.7")

    sent_message = mock_smtp.send_message.call_args[0][0]
    assert sent_message["To"] == "victime@bureau.fr"
    body = sent_message.get_content()
    assert "198.51.100.7" in body
    assert "sessions" in body.lower()
