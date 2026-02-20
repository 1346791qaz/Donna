"""
llm.py — Claude API integration with full tool-use support.

Responsibilities:
- Build system prompt (injecting datetime + user name)
- Load conversation history (token-budgeted)
- Dispatch Claude API calls
- Handle tool_use blocks: call the right local function, return tool_result
- Persist every turn to conversation_history.db
- Expose a single synchronous `chat(user_message)` → str interface
"""

import json
import logging
from datetime import datetime
from typing import Any, Callable

import anthropic

from donna.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, USER_NAME
from donna.db import conversation_db
from donna.tools import gmail_tools, calendar_tools, contacts_tools

logger = logging.getLogger(__name__)

# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """You are Donna, a highly capable British personal assistant. You are professional, warm, efficient, and proactive. You communicate in natural, conversational British English — not overly formal, not casual.

Your responsibilities:
- Manage and summarise emails and calendar events on behalf of the user
- Maintain awareness of the user's contacts and relationships
- Surface relevant context: upcoming meetings, unread priority emails, follow-ups due
- Draft emails, schedule meetings, set reminders when asked
- Remember context across conversations — reference past discussions naturally
- Anticipate needs: if the user mentions a meeting tomorrow, offer to check their calendar

Operational rules:
- Always use available tools before stating you don't have information
- Summarise email and calendar data efficiently — do not recite raw data
- Confirm with the user before creating, sending, or deleting anything
- Flag time-sensitive items proactively at session start
- Keep responses concise for simple queries; detailed when complexity warrants
- Never fabricate contact information, email content, or calendar data
- If intent is unclear, ask one clarifying question — not multiple

Response formatting:
- Plain spoken English only — no markdown whatsoever
- No asterisks, underscores, hashes, bullet symbols, dashes as list markers, or any other markdown syntax
- No emoji or special Unicode symbols
- Do not use bold, italic, headers, or horizontal rules
- Structure responses with natural spoken language: "First...", "Also...", "Finally..." rather than lists
- Write times as "6 PM" or "half past ten", not "6:00 PM" or "22:30"
- Responses will be read aloud by a TTS engine — write exactly as you would speak

Tool use:
- Available tools: Gmail (read/send), Google Calendar (read/write), Contacts DB (CRUD), Conversation History
- Use tools silently — do not narrate tool calls to the user
- Synthesise tool results into natural language responses

Current date and time: {datetime}
User's name: {user_name}"""


def _build_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(
        datetime=datetime.now().strftime("%A, %d %B %Y, %H:%M"),
        user_name=USER_NAME,
    )


# ─── Tool definitions (Claude tool-use schema) ───────────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    # ── Gmail ──
    {
        "name": "get_emails",
        "description": "Search and retrieve emails from Gmail.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Gmail search query, e.g. 'is:unread from:boss@example.com'.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of emails to return (default 10).",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email via Gmail. Always confirm with the user before calling.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to":      {"type": "string", "description": "Recipient email address(es)."},
                "subject": {"type": "string", "description": "Email subject line."},
                "body":    {"type": "string", "description": "Plain-text email body."},
                "cc":      {"type": "string", "description": "CC recipients (optional)."},
                "reply_to_message_id": {
                    "type": "string",
                    "description": "Message-ID for threading (optional).",
                },
                "thread_id": {
                    "type": "string",
                    "description": "Gmail thread ID to reply into (optional).",
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "get_thread",
        "description": "Fetch a full Gmail email thread by thread ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "thread_id": {"type": "string", "description": "Gmail thread ID."},
            },
            "required": ["thread_id"],
        },
    },
    # ── Calendar ──
    {
        "name": "get_calendar_events",
        "description": "Fetch calendar events within a date range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date/datetime in ISO format, e.g. '2024-01-15' or '2024-01-15T09:00:00'.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date/datetime in ISO format.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum events to return (default 20).",
                    "default": 20,
                },
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create a new calendar event. Always confirm with the user before calling.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title":       {"type": "string"},
                "start":       {"type": "string", "description": "ISO datetime string."},
                "end":         {"type": "string", "description": "ISO datetime string."},
                "attendees":   {"type": "array",  "items": {"type": "string"}, "description": "Email addresses."},
                "description": {"type": "string"},
                "location":    {"type": "string"},
            },
            "required": ["title", "start", "end"],
        },
    },
    {
        "name": "update_calendar_event",
        "description": "Update an existing calendar event. Always confirm with the user before calling.",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
                "changes":  {"type": "object", "description": "Fields to update."},
            },
            "required": ["event_id", "changes"],
        },
    },
    {
        "name": "delete_calendar_event",
        "description": "Delete a calendar event. Always confirm with the user before calling.",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
            },
            "required": ["event_id"],
        },
    },
    # ── Contacts ──
    {
        "name": "lookup_contact",
        "description": "Look up contacts by name, email, or company using fuzzy matching.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search term, e.g. 'John from Microsoft'."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "add_contact",
        "description": "Add a new contact to the local database.",
        "input_schema": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "company":   {"type": "string"},
                "title":     {"type": "string"},
                "email":     {"type": "string"},
                "phone":     {"type": "string"},
                "notes":     {"type": "string"},
            },
            "required": ["full_name"],
        },
    },
    {
        "name": "update_contact",
        "description": "Update an existing contact. Notes are appended with a timestamp.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contact_id": {"type": "integer"},
                "fields": {
                    "type": "object",
                    "description": "Fields to update, e.g. {\"email\": \"new@example.com\", \"notes\": \"Called today.\"}",
                },
            },
            "required": ["contact_id", "fields"],
        },
    },
    {
        "name": "search_contacts",
        "description": "Search contacts by name, company, or email.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
]

# ─── Tool dispatch ────────────────────────────────────────────────────────────

_TOOL_MAP: dict[str, Callable[..., Any]] = {
    "get_emails":             gmail_tools.get_emails,
    "send_email":             gmail_tools.send_email,
    "get_thread":             gmail_tools.get_thread,
    "get_calendar_events":    calendar_tools.get_calendar_events,
    "create_calendar_event":  calendar_tools.create_calendar_event,
    "update_calendar_event":  calendar_tools.update_calendar_event,
    "delete_calendar_event":  calendar_tools.delete_calendar_event,
    "lookup_contact":         contacts_tools.lookup_contact,
    "add_contact":            contacts_tools.add_contact,
    "update_contact":         contacts_tools.update_contact,
    "search_contacts":        contacts_tools.search_contacts,
}


def _dispatch_tool(name: str, tool_input: dict) -> Any:
    fn = _TOOL_MAP.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**tool_input)
    except Exception as exc:
        logger.exception("Tool %s raised an exception", name)
        return {"error": str(exc)}


# ─── Main chat function ───────────────────────────────────────────────────────

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def chat(user_message: str, on_tool_call: Callable[[str, dict], None] | None = None) -> str:
    """
    Send a user message to Claude, handle any tool calls, and return the
    final text response.

    Args:
        user_message:  The text typed or transcribed from the user.
        on_tool_call:  Optional callback invoked with (tool_name, tool_input)
                       each time Claude calls a tool — useful for UI status updates.

    Returns:
        Claude's final text response as a string.
    """
    # Persist user turn
    conversation_db.save_message("user", user_message)

    # Build message history
    history = conversation_db.load_history()
    # Ensure the message we just saved appears (load_history may lag behind
    # in edge cases, so we append explicitly if not already present).
    if not history or history[-1]["content"] != user_message:
        history.append({"role": "user", "content": user_message})

    messages = history.copy()

    # Agentic loop: keep calling until no more tool_use blocks
    while True:
        response = _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            system=_build_system_prompt(),
            tools=TOOL_DEFINITIONS,
            messages=messages,
            timeout=60.0,
        )

        # Collect text from this response turn
        text_parts: list[str] = []
        tool_uses: list[dict] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        # Add Claude's response to the message chain
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use" or not tool_uses:
            # Final response — no more tool calls
            final_text = " ".join(text_parts).strip()
            conversation_db.save_message("assistant", final_text)
            return final_text

        # Process tool calls and build tool_result message
        tool_results = []
        for tool_use_block in tool_uses:
            tool_name = tool_use_block.name
            tool_input = tool_use_block.input
            logger.info("Tool call: %s(%s)", tool_name, json.dumps(tool_input)[:200])

            if on_tool_call:
                on_tool_call(tool_name, tool_input)

            result = _dispatch_tool(tool_name, tool_input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_block.id,
                "content": json.dumps(result, default=str),
            })

        messages.append({"role": "user", "content": tool_results})


def morning_brief() -> str:
    """
    Generate a proactive morning briefing covering today's calendar and
    unread priority emails.
    """
    prompt = (
        "Please give me a concise morning briefing: today's calendar events "
        "and any unread priority emails I should know about. Keep it brief and "
        "highlight anything time-sensitive."
    )
    return chat(prompt)
