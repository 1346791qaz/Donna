"""
db/conversation_db.py — SQLite conversation history.

Stores every LLM turn (user / assistant / tool) with its session_id.
Provides token-aware loading so we never exceed the context budget.
"""

import sqlite3
import logging
import uuid
from datetime import datetime, date
from contextlib import contextmanager
from typing import Optional

from donna.config import (
    CONVERSATION_DB_PATH,
    SESSION_MODE,
    LLM_MAX_HISTORY_MESSAGES,
    LLM_MAX_HISTORY_TOKENS,
)

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    role       TEXT     NOT NULL,
    content    TEXT     NOT NULL,
    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conv_ts      ON conversations(timestamp);
"""

# Module-level session_id — assigned once on import.
_SESSION_ID: str = ""


def _make_session_id() -> str:
    if SESSION_MODE == "daily":
        return str(date.today())
    return str(uuid.uuid4())


def get_session_id() -> str:
    global _SESSION_ID
    if not _SESSION_ID:
        _SESSION_ID = _make_session_id()
    return _SESSION_ID


@contextmanager
def _get_conn():
    conn = sqlite3.connect(
        CONVERSATION_DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES
    )
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
    with _get_conn() as conn:
        conn.executescript(_DDL)
    logger.info("Conversation DB initialised at %s", CONVERSATION_DB_PATH)


def save_message(role: str, content: str) -> None:
    """Persist a single turn to the database."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO conversations (role, content, session_id) VALUES (?, ?, ?)",
            (role, content, get_session_id()),
        )


def _rough_token_count(text: str) -> int:
    """~4 chars per token — good enough for budget management."""
    return max(1, len(text) // 4)


def load_history(
    max_messages: int = LLM_MAX_HISTORY_MESSAGES,
    max_tokens: int = LLM_MAX_HISTORY_TOKENS,
) -> list[dict]:
    """
    Return conversation history as a list of {"role": ..., "content": ...} dicts
    suitable for passing directly to the Claude API.

    Messages are loaded newest-first, then reversed so the list is chronological.
    Token budget is applied before reversing.
    """
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content FROM conversations
            ORDER BY id DESC
            LIMIT ?
            """,
            (max_messages,),
        ).fetchall()

    # Apply token budget (newest messages win)
    kept: list[dict] = []
    total_tokens = 0
    for row in rows:
        tokens = _rough_token_count(row["content"])
        if total_tokens + tokens > max_tokens:
            break
        kept.append({"role": row["role"], "content": row["content"]})
        total_tokens += tokens

    kept.reverse()  # chronological order
    return kept


def load_session_history(session_id: Optional[str] = None) -> list[dict]:
    """Load all messages for a specific session (default: current)."""
    sid = session_id or get_session_id()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT role, content FROM conversations WHERE session_id = ? ORDER BY id",
            (sid,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def search_history(keyword: str, limit: int = 10) -> list[dict]:
    """Full-text keyword search across all stored messages."""
    pat = f"%{keyword}%"
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content, timestamp, session_id
            FROM conversations
            WHERE content LIKE ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (pat, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_recent_sessions(n: int = 7) -> list[str]:
    """Return the n most recent distinct session_ids."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT session_id FROM conversations
            ORDER BY id DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
    return [r["session_id"] for r in rows]
