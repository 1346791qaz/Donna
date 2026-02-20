"""
tools/calendar_tools.py — Google Calendar API wrappers.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from donna.tools._auth import get_google_service

logger = logging.getLogger(__name__)


def _calendar():
    return get_google_service("calendar", "v3")


def _parse_dt(dt_str: str) -> str:
    """Normalise various date/datetime strings to RFC3339."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            continue
    # Assume already RFC3339
    return dt_str


# ─── Read ─────────────────────────────────────────────────────────────────────

def get_calendar_events(
    start_date: str,
    end_date: str,
    calendar_id: str = "primary",
    max_results: int = 20,
) -> list[dict]:
    """
    Return calendar events between start_date and end_date.

    Args:
        start_date:   ISO date or datetime, e.g. "2024-01-15" or "2024-01-15T09:00:00".
        end_date:     ISO date or datetime.
        calendar_id:  Calendar to query (default: "primary").
        max_results:  Maximum events to return.

    Returns:
        List of event dicts with id, title, start, end, attendees, description, location.
    """
    service = _calendar()
    events_result = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=_parse_dt(start_date),
            timeMax=_parse_dt(end_date),
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    items = events_result.get("items", [])
    results = []
    for e in items:
        start = e.get("start", {})
        end = e.get("end", {})
        results.append({
            "id": e.get("id"),
            "title": e.get("summary", "(no title)"),
            "start": start.get("dateTime", start.get("date", "")),
            "end": end.get("dateTime", end.get("date", "")),
            "location": e.get("location", ""),
            "description": e.get("description", "")[:500],
            "attendees": [
                a.get("email") for a in e.get("attendees", [])
            ],
            "status": e.get("status", ""),
        })
    return results


# ─── Create ───────────────────────────────────────────────────────────────────

def create_calendar_event(
    title: str,
    start: str,
    end: str,
    attendees: Optional[list[str]] = None,
    description: Optional[str] = None,
    location: Optional[str] = None,
    calendar_id: str = "primary",
) -> dict:
    """
    Create a new calendar event.

    Returns:
        Dict with id, html_link, title, start, end.
    """
    service = _calendar()
    body: dict = {
        "summary": title,
        "start": {"dateTime": _parse_dt(start), "timeZone": "UTC"},
        "end":   {"dateTime": _parse_dt(end),   "timeZone": "UTC"},
    }
    if attendees:
        body["attendees"] = [{"email": a} for a in attendees]
    if description:
        body["description"] = description
    if location:
        body["location"] = location

    event = service.events().insert(calendarId=calendar_id, body=body).execute()
    logger.info("Calendar event created: %s", event.get("id"))
    return {
        "id": event.get("id"),
        "html_link": event.get("htmlLink"),
        "title": event.get("summary"),
        "start": event.get("start", {}).get("dateTime", ""),
        "end":   event.get("end",   {}).get("dateTime", ""),
    }


# ─── Update ───────────────────────────────────────────────────────────────────

def update_calendar_event(
    event_id: str,
    changes: dict,
    calendar_id: str = "primary",
) -> dict:
    """
    Patch an existing calendar event.

    Args:
        event_id: The event's Google Calendar ID.
        changes:  Dict of fields to update, e.g. {"summary": "New title",
                  "start": {"dateTime": "..."}, "end": {"dateTime": "..."}}.

    Returns:
        Updated event dict.
    """
    service = _calendar()
    event = (
        service.events().get(calendarId=calendar_id, eventId=event_id).execute()
    )
    event.update(changes)
    updated = (
        service.events()
        .update(calendarId=calendar_id, eventId=event_id, body=event)
        .execute()
    )
    logger.info("Calendar event updated: %s", event_id)
    return {
        "id": updated.get("id"),
        "title": updated.get("summary"),
        "start": updated.get("start", {}).get("dateTime", ""),
        "end":   updated.get("end",   {}).get("dateTime", ""),
    }


# ─── Delete ───────────────────────────────────────────────────────────────────

def delete_calendar_event(
    event_id: str,
    calendar_id: str = "primary",
) -> dict:
    """
    Delete a calendar event.

    Returns:
        {"deleted": True, "event_id": event_id}
    """
    service = _calendar()
    service.events().delete(calendarId=calendar_id, eventId=event_id).execute()
    logger.info("Calendar event deleted: %s", event_id)
    return {"deleted": True, "event_id": event_id}
