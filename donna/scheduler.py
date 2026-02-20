"""
scheduler.py — Proactive behaviours via APScheduler.

Scheduled jobs:
1. Morning Brief     — once per day on first session, summarise today
2. Follow-up Tracker — on session start, surface unresolved follow-up intents
3. Meeting Prep      — 30 min before each calendar event, alert the user

All jobs dispatch through llm.chat() and route the response back to the
UI window and TTS engine via the callback registered with start().
"""

import logging
import threading
from datetime import datetime, timedelta, date, timezone
from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from donna.config import MEETING_PREP_MINUTES
from donna.db import conversation_db

logger = logging.getLogger(__name__)

# Tracks which dates the morning brief has already been delivered
_morning_brief_dates: set[date] = set()
_lock = threading.Lock()


class ProactiveScheduler:
    """
    Wraps APScheduler and owns all of Donna's proactive trigger logic.

    Args:
        on_response: Callback invoked with the LLM response text whenever a
                     proactive message is generated.  Typically wires to TTS
                     and the UI window.
    """

    def __init__(self, on_response: Callable[[str], None]):
        self._on_response = on_response
        self._scheduler = BackgroundScheduler(timezone="UTC")
        self._scheduled_event_ids: set[str] = set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        # Run follow-up tracker + calendar sync every hour
        self._scheduler.add_job(
            self._job_hourly_sync,
            trigger=CronTrigger(minute=0),
            id="hourly_sync",
            replace_existing=True,
            misfire_grace_time=120,
        )
        # Morning brief check: run at 07:00 UTC and also at startup (see below)
        self._scheduler.add_job(
            self._job_morning_brief,
            trigger=CronTrigger(hour=7, minute=0),
            id="morning_brief",
            replace_existing=True,
            misfire_grace_time=3600,
        )
        self._scheduler.start()
        logger.info("ProactiveScheduler started.")

        # Attempt morning brief immediately at startup
        threading.Thread(
            target=self._job_morning_brief, daemon=True, name="InitialBrief"
        ).start()
        # Also seed meeting-prep jobs for today
        threading.Thread(
            target=self._job_hourly_sync, daemon=True, name="InitialSync"
        ).start()

    def stop(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        logger.info("ProactiveScheduler stopped.")

    # ── Jobs ──────────────────────────────────────────────────────────────────

    def _job_morning_brief(self) -> None:
        today = date.today()
        with _lock:
            if today in _morning_brief_dates:
                return
            _morning_brief_dates.add(today)

        logger.info("Delivering morning brief for %s.", today)
        try:
            from donna.llm import morning_brief
            text = morning_brief()
            if text:
                self._on_response(text)
        except Exception:
            logger.exception("Morning brief job failed.")

    def _job_hourly_sync(self) -> None:
        """Refresh meeting-prep alerts and surface follow-ups."""
        self._schedule_meeting_prep_alerts()
        self._surface_followups()

    def _schedule_meeting_prep_alerts(self) -> None:
        """
        Fetch today's calendar events and schedule a prep alert for each one
        that starts in the future and hasn't been alerted yet.
        """
        try:
            from donna.tools.calendar_tools import get_calendar_events
            now = datetime.now(timezone.utc)
            end_of_day = now.replace(hour=23, minute=59, second=59)
            events = get_calendar_events(
                start_date=now.isoformat(),
                end_date=end_of_day.isoformat(),
                max_results=20,
            )
            for event in events:
                event_id = event.get("id")
                if not event_id or event_id in self._scheduled_event_ids:
                    continue
                start_str = event.get("start", "")
                if not start_str:
                    continue
                try:
                    # Parse RFC3339 — strip trailing Z if present
                    event_start = datetime.fromisoformat(start_str.rstrip("Z")).replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    continue
                alert_time = event_start - timedelta(minutes=MEETING_PREP_MINUTES)
                if alert_time <= now:
                    continue  # Already past the alert window
                self._scheduler.add_job(
                    self._meeting_prep_alert,
                    trigger=DateTrigger(run_date=alert_time),
                    args=[event],
                    id=f"meeting_prep_{event_id}",
                    replace_existing=True,
                )
                self._scheduled_event_ids.add(event_id)
                logger.info(
                    "Scheduled meeting prep for '%s' at %s.",
                    event.get("title"), alert_time.isoformat()
                )
        except Exception:
            logger.exception("Failed to schedule meeting prep alerts.")

    def _meeting_prep_alert(self, event: dict) -> None:
        title = event.get("title", "your next meeting")
        start = event.get("start", "")
        attendees = event.get("attendees", [])
        attendee_str = ", ".join(attendees[:3]) if attendees else "no listed attendees"

        prompt = (
            f"You have a meeting in {MEETING_PREP_MINUTES} minutes: '{title}' "
            f"starting at {start}. Attendees: {attendee_str}. "
            "Please give a brief meeting prep summary: any relevant contact notes, "
            "recent emails from attendees, and anything I should know."
        )
        try:
            from donna.llm import chat
            response = chat(prompt)
            if response:
                self._on_response(response)
        except Exception:
            logger.exception("Meeting prep alert failed for event %s.", event.get("id"))

    def _surface_followups(self) -> None:
        """
        Search conversation history for unresolved follow-up intents and
        surface them if they're from a previous session.
        """
        try:
            current_session = conversation_db.get_session_id()
            sessions = conversation_db.get_recent_sessions(n=7)
            past_sessions = [s for s in sessions if s != current_session]
            if not past_sessions:
                return

            # Look for intent keywords in past sessions
            keywords = ["i need to call", "remind me", "follow up", "i should email",
                        "i'll email", "i'll call", "don't let me forget"]
            followup_snippets: list[str] = []
            for keyword in keywords:
                hits = conversation_db.search_history(keyword, limit=3)
                for hit in hits:
                    if hit.get("session_id") != current_session:
                        followup_snippets.append(hit["content"][:200])

            if not followup_snippets:
                return

            combined = "\n- ".join(followup_snippets[:5])
            prompt = (
                "Based on the user's previous statements, there appear to be unresolved "
                f"follow-ups:\n- {combined}\n\n"
                "Briefly and naturally remind the user of these outstanding items. "
                "Keep it to 2–3 sentences."
            )
            from donna.llm import chat
            response = chat(prompt)
            if response:
                self._on_response(response)
        except Exception:
            logger.exception("Follow-up surface job failed.")
