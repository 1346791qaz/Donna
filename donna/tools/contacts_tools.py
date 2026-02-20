"""
tools/contacts_tools.py — High-level contacts tool functions exposed to Claude.

Uses rapidfuzz for fuzzy name matching so queries like
"find John from Microsoft" work naturally.
"""

import logging
from typing import Optional

from rapidfuzz import process as fuzz_process, fuzz

from donna.db import contacts_db

logger = logging.getLogger(__name__)

_FUZZY_SCORE_CUTOFF = 60  # 0–100; lower = more permissive


def lookup_contact(query: str) -> list[dict]:
    """
    Look up contacts by name, email, or company using fuzzy matching.

    Args:
        query: Natural-language search term, e.g. "John from Microsoft".

    Returns:
        List of matching contact dicts (up to 5).
    """
    # First try exact substring search (fast path)
    results = contacts_db.search_contacts_exact(query)
    if results:
        return results[:5]

    # Fuzzy fallback against all stored names
    candidates = contacts_db.get_all_names_and_ids()
    if not candidates:
        return []

    names = [c["full_name"] for c in candidates]
    matches = fuzz_process.extract(
        query, names, scorer=fuzz.WRatio, limit=5, score_cutoff=_FUZZY_SCORE_CUTOFF
    )
    matched_ids = {candidates[m[2]]["id"] for m in matches}
    return [contacts_db.get_contact_by_id(cid) for cid in matched_ids if cid]


def add_contact(
    full_name: str,
    company: Optional[str] = None,
    title: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    notes: Optional[str] = None,
) -> dict:
    """
    Add a new contact to the database.

    Returns:
        The newly created contact dict.
    """
    contact_id = contacts_db.add_contact(
        full_name=full_name,
        company=company,
        title=title,
        email=email,
        phone=phone,
        notes=notes,
    )
    contact = contacts_db.get_contact_by_id(contact_id)
    logger.info("Contact added: id=%s name=%s", contact_id, full_name)
    return contact


def update_contact(contact_id: int, fields: dict) -> dict:
    """
    Update an existing contact.

    Args:
        contact_id: Integer ID from the contacts table.
        fields:     Dict of fields to update. Notes are appended, not replaced.

    Returns:
        Updated contact dict, or error dict if not found.
    """
    success = contacts_db.update_contact(contact_id, fields)
    if not success:
        return {"error": f"Contact {contact_id} not found or no changes made."}
    contact = contacts_db.get_contact_by_id(contact_id)
    logger.info("Contact updated: id=%s", contact_id)
    return contact


def search_contacts(query: str) -> list[dict]:
    """
    Fuzzy search contacts — alias of lookup_contact for explicit tool naming.
    """
    return lookup_contact(query)


def delete_contact(contact_id: int) -> dict:
    """
    Delete a contact by ID.

    Returns:
        {"deleted": True} or {"error": "..."}.
    """
    success = contacts_db.delete_contact(contact_id)
    if not success:
        return {"error": f"Contact {contact_id} not found."}
    logger.info("Contact deleted: id=%s", contact_id)
    return {"deleted": True, "contact_id": contact_id}


def list_contacts(limit: int = 20, offset: int = 0) -> list[dict]:
    """Return a paginated list of all contacts."""
    return contacts_db.list_contacts(limit=limit, offset=offset)
