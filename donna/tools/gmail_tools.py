"""
tools/gmail_tools.py — Gmail API wrappers.

All functions return plain dicts/lists so they can be serialised directly
into Claude tool-result messages.
"""

import base64
import logging
import email as email_lib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

from donna.tools._auth import get_google_service

logger = logging.getLogger(__name__)


def _gmail():
    return get_google_service("gmail", "v1")


# ─── Read ─────────────────────────────────────────────────────────────────────

def get_emails(query: str = "is:inbox", max_results: int = 10) -> list[dict]:
    """
    Search Gmail and return a list of message summaries.

    Args:
        query:       Gmail search query (same syntax as the Gmail UI).
        max_results: Maximum number of messages to return.

    Returns:
        List of dicts with keys: id, thread_id, subject, from, date, snippet.
    """
    service = _gmail()
    result = (
        service.users()
        .messages()
        .list(userId="me", q=query, maxResults=max_results)
        .execute()
    )
    messages = result.get("messages", [])
    summaries = []
    for msg_ref in messages:
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=msg_ref["id"], format="metadata",
                 metadataHeaders=["Subject", "From", "Date"])
            .execute()
        )
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        summaries.append({
            "id": msg["id"],
            "thread_id": msg.get("threadId", ""),
            "subject": headers.get("Subject", "(no subject)"),
            "from": headers.get("From", ""),
            "date": headers.get("Date", ""),
            "snippet": msg.get("snippet", ""),
        })
    return summaries


def get_thread(thread_id: str) -> dict:
    """
    Fetch a full email thread.

    Returns:
        Dict with thread_id and a list of messages (role, from, date, body).
    """
    service = _gmail()
    thread = (
        service.users().threads().get(userId="me", id=thread_id, format="full").execute()
    )
    messages = []
    for msg in thread.get("messages", []):
        headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
        body = _extract_body(msg.get("payload", {}))
        messages.append({
            "id": msg["id"],
            "from": headers.get("From", ""),
            "date": headers.get("Date", ""),
            "subject": headers.get("Subject", ""),
            "body": body[:4000],  # cap to avoid huge context
        })
    return {"thread_id": thread_id, "messages": messages}


def _extract_body(payload: dict) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    mime_type = payload.get("mimeType", "")
    if mime_type == "text/plain":
        data = payload.get("body", {}).get("data", "")
        return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")
    if mime_type.startswith("multipart/"):
        for part in payload.get("parts", []):
            text = _extract_body(part)
            if text:
                return text
    return ""


# ─── Send ─────────────────────────────────────────────────────────────────────

def send_email(
    to: str,
    subject: str,
    body: str,
    cc: Optional[str] = None,
    reply_to_message_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> dict:
    """
    Send an email via Gmail.

    Args:
        to:                    Recipient address(es), comma-separated.
        subject:               Email subject.
        body:                  Plain-text body.
        cc:                    CC address(es), comma-separated (optional).
        reply_to_message_id:   Message-ID header value for threading (optional).
        thread_id:             Gmail thread ID to reply into (optional).

    Returns:
        Dict with id and thread_id of the sent message.
    """
    service = _gmail()
    mime = MIMEMultipart()
    mime["To"] = to
    mime["Subject"] = subject
    if cc:
        mime["Cc"] = cc
    if reply_to_message_id:
        mime["In-Reply-To"] = reply_to_message_id
        mime["References"] = reply_to_message_id
    mime.attach(MIMEText(body, "plain"))

    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    body_payload: dict = {"raw": raw}
    if thread_id:
        body_payload["threadId"] = thread_id

    sent = (
        service.users().messages().send(userId="me", body=body_payload).execute()
    )
    logger.info("Email sent, id=%s", sent["id"])
    return {"id": sent["id"], "thread_id": sent.get("threadId", "")}
