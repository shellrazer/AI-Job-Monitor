"""Unit tests for job_monitor.notify."""

from __future__ import annotations

from job_monitor.config import EmailSettings
from job_monitor.notify import build_message, send_digest

HTML_BODY = "<p>National Quality Manager — apply at https://example.com/nqm</p>"
TEXT_BODY = "National Quality Manager\n  Apply: https://example.com/nqm\n"


def test_build_message_is_multipart_alternative_with_both_parts():
    msg = build_message(
        sender="x@gmail.com",
        recipients=["a@b.com", "c@d.com"],
        subject="Digest",
        html_body=HTML_BODY,
        text_body=TEXT_BODY,
    )
    assert msg.get_content_type() == "multipart/alternative"
    assert msg["From"] == "x@gmail.com"
    assert msg["To"] == "a@b.com, c@d.com"
    assert msg["Subject"] == "Digest"

    subtypes = {part.get_content_subtype() for part in msg.iter_parts()}
    assert "plain" in subtypes
    assert "html" in subtypes


def test_send_digest_sends_when_enabled(mock_smtp):
    settings = EmailSettings(
        enabled=True,
        sender="x@gmail.com",
        app_password="pw",
        recipients=["a@b.com"],
    )
    result = send_digest(
        settings,
        subject="Job Monitor Digest",
        html_body=HTML_BODY,
        text_body=TEXT_BODY,
    )
    assert result is True
    assert len(mock_smtp) == 1

    sender, to, raw = mock_smtp[0]
    assert sender == "x@gmail.com"
    assert to == "a@b.com"
    assert "National Quality Manager" in raw
    assert "https://example.com/nqm" in raw


def test_send_digest_noop_when_disabled(mock_smtp):
    settings = EmailSettings(
        enabled=False,
        sender="x@gmail.com",
        app_password="pw",
        recipients=["a@b.com"],
    )
    result = send_digest(
        settings,
        subject="Job Monitor Digest",
        html_body=HTML_BODY,
        text_body=TEXT_BODY,
    )
    assert result is False
    assert mock_smtp == []
