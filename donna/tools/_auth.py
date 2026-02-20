"""
tools/_auth.py — Google OAuth 2.0 helper shared by Gmail and Calendar tools.
"""

import logging
from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

from donna.config import GOOGLE_CREDENTIALS_PATH, GOOGLE_TOKEN_PATH, GOOGLE_SCOPES

logger = logging.getLogger(__name__)

_creds: Credentials | None = None


def _get_credentials() -> Credentials:
    global _creds
    if _creds and _creds.valid:
        return _creds

    token_path = Path(GOOGLE_TOKEN_PATH)
    creds = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), GOOGLE_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing Google OAuth token.")
            creds.refresh(Request())
        else:
            logger.info("Launching Google OAuth browser flow.")
            flow = InstalledAppFlow.from_client_secrets_file(
                GOOGLE_CREDENTIALS_PATH, GOOGLE_SCOPES
            )
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())

    _creds = creds
    return creds


def get_google_service(api_name: str, api_version: str):
    """Build and return an authenticated Google API service client."""
    creds = _get_credentials()
    return build(api_name, api_version, credentials=creds, cache_discovery=False)
