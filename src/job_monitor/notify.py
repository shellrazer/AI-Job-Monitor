"""Build and send the digest email over SMTP.

Thin wrapper around :mod:`smtplib` and :class:`email.message.EmailMessage`. The
caller supplies the rendered HTML/text bodies (see :mod:`job_monitor.report`);
this module assembles a ``multipart/alternative`` message and sends it via the
configured Gmail SMTP relay. Sending is gated on :attr:`EmailSettings.enabled`.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import EmailSettings

__all__ = ["build_message", "send_digest"]


def build_message(
    *,
    sender: str,
    recipients: list[str],
    subject: str,
    html_body: str,
    text_body: str,
) -> EmailMessage:
    """Assemble a ``multipart/alternative`` email (plaintext + HTML)."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    return msg


def send_digest(
    email_settings: EmailSettings,
    *,
    subject: str,
    html_body: str,
    text_body: str,
) -> bool:
    """Send the digest email; return ``True`` on success, ``False`` if disabled.

    When ``email_settings.enabled`` is false, no connection is made and the
    function returns ``False`` without sending anything. Otherwise it connects,
    upgrades to TLS, authenticates, and sends the message.
    """
    if not email_settings.enabled:
        return False

    msg = build_message(
        sender=email_settings.sender,
        recipients=email_settings.recipients,
        subject=subject,
        html_body=html_body,
        text_body=text_body,
    )

    with smtplib.SMTP(email_settings.smtp_host, email_settings.smtp_port) as smtp:
        smtp.starttls()
        smtp.login(email_settings.sender, email_settings.app_password)
        smtp.send_message(msg)
    return True
