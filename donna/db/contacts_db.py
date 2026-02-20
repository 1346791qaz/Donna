"""
db/contacts_db.py — SQLite contacts database.

Schema, initialisation, and low-level CRUD helpers used by contacts_tools.py.
The `notes` column is append-only: every update prefixes a timestamped entry.
"""

import sqlite3
import logging
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

from donna.config import CONTACTS_DB_PATH

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS contacts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name  TEXT NOT NULL,
    company    TEXT,
    title      TEXT,
    email      TEXT,
    phone      TEXT,
    notes      TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_contacts_full_name ON contacts(full_name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_contacts_email     ON contacts(email COLLATE NOCASE);
"""


@contextmanager
def _get_conn():
    conn = sqlite3.connect(CONTACTS_DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript(_DDL)
    logger.info("Contacts DB initialised at %s", CONTACTS_DB_PATH)


def add_contact(
    full_name: str,
    company: Optional[str] = None,
    title: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    """Insert a new contact and return its id."""
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")
    stamped_notes = f"[{ts}] {notes}" if notes else None
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO contacts (full_name, company, title, email, phone, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (full_name, company, title, email, phone, stamped_notes),
        )
        return cur.lastrowid


def get_contact_by_id(contact_id: int) -> Optional[dict]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM contacts WHERE id = ?", (contact_id,)
        ).fetchone()
        return dict(row) if row else None


def list_contacts(limit: int = 50, offset: int = 0) -> list[dict]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM contacts ORDER BY full_name COLLATE NOCASE LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def search_contacts_exact(query: str) -> list[dict]:
    """Case-insensitive substring search on name, email, company."""
    pat = f"%{query}%"
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM contacts
            WHERE  full_name LIKE ? COLLATE NOCASE
               OR  email     LIKE ? COLLATE NOCASE
               OR  company   LIKE ? COLLATE NOCASE
            ORDER BY full_name COLLATE NOCASE
            LIMIT 20
            """,
            (pat, pat, pat),
        ).fetchall()
        return [dict(r) for r in rows]


def update_contact(contact_id: int, fields: dict) -> bool:
    """
    Update scalar fields.  If `notes` is present it is *appended* (not replaced).
    Returns True if a row was modified.
    """
    if not fields:
        return False

    # Handle notes append separately
    note_text = fields.pop("notes", None)

    ts = datetime.now().isoformat(sep=" ", timespec="seconds")

    with _get_conn() as conn:
        if note_text:
            row = conn.execute(
                "SELECT notes FROM contacts WHERE id = ?", (contact_id,)
            ).fetchone()
            existing = row["notes"] if row and row["notes"] else ""
            sep = "\n" if existing else ""
            new_notes = f"{existing}{sep}[{ts}] {note_text}"
            conn.execute(
                "UPDATE contacts SET notes = ?, updated_at = ? WHERE id = ?",
                (new_notes, ts, contact_id),
            )

        if fields:
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            values = list(fields.values()) + [ts, contact_id]
            cur = conn.execute(
                f"UPDATE contacts SET {set_clause}, updated_at = ? WHERE id = ?",
                values,
            )
            return cur.rowcount > 0

        return True


def delete_contact(contact_id: int) -> bool:
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM contacts WHERE id = ?", (contact_id,))
        return cur.rowcount > 0


def get_all_names_and_ids() -> list[dict]:
    """Lightweight fetch for fuzzy matching."""
    with _get_conn() as conn:
        rows = conn.execute("SELECT id, full_name, company FROM contacts").fetchall()
        return [dict(r) for r in rows]
