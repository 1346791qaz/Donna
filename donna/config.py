"""
config.py — Environment variable loading and application-wide constants.
Load all secrets from .env; never hardcode credentials.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Locate repo root (parent of this file's directory)
_ROOT = Path(__file__).parent.parent
_ENV_PATH = _ROOT / ".env"

load_dotenv(dotenv_path=_ENV_PATH)

# ─── Anthropic ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-opus-4-6")

# ─── Picovoice ────────────────────────────────────────────────────────────────
PICOVOICE_ACCESS_KEY: str = os.getenv("PICOVOICE_ACCESS_KEY", "")
# Path to the trained "Hey Donna" .ppn keyword model file
WAKE_WORD_MODEL_PATH: str = os.getenv(
    "WAKE_WORD_MODEL_PATH",
    str(_ROOT / "hey_donna.ppn"),
)
WAKE_WORD_SENSITIVITY: float = float(os.getenv("WAKE_WORD_SENSITIVITY", "0.5"))

# ─── Google ───────────────────────────────────────────────────────────────────
GOOGLE_CREDENTIALS_PATH: str = os.getenv(
    "GOOGLE_CREDENTIALS_PATH",
    str(_ROOT / "credentials.json"),
)
GOOGLE_TOKEN_PATH: str = os.getenv(
    "GOOGLE_TOKEN_PATH",
    str(_ROOT / "token.json"),
)
GOOGLE_SCOPES: list[str] = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
]

# ─── User identity ────────────────────────────────────────────────────────────
USER_NAME: str = os.getenv("USER_NAME", "User")

# ─── STT ─────────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "small")
# "cpu" or "cuda"
WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# ─── TTS ─────────────────────────────────────────────────────────────────────
TTS_VOICE: str = os.getenv("TTS_VOICE", "bf_emma")
TTS_SPEED: float = float(os.getenv("TTS_SPEED", "1.09"))

# ─── VAD ─────────────────────────────────────────────────────────────────────
VAD_MODE: int = int(os.getenv("VAD_MODE", "3"))          # 0–3; 3 = most aggressive
VAD_SAMPLE_RATE: int = 16000
VAD_FRAME_DURATION_MS: int = 30                           # must be 10, 20, or 30
# Silence frames needed before end-of-speech is declared
VAD_SILENCE_FRAMES: int = int(os.getenv("VAD_SILENCE_FRAMES", "25"))

# ─── Audio ────────────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE: int = 16000
AUDIO_CHANNELS: int = 1
AUDIO_CHUNK_FRAMES: int = 512                             # frames per PyAudio read

# ─── Database ─────────────────────────────────────────────────────────────────
DB_DIR: Path = _ROOT / "data"
DB_DIR.mkdir(exist_ok=True)
CONTACTS_DB_PATH: str = str(DB_DIR / "contacts.db")
CONVERSATION_DB_PATH: str = str(DB_DIR / "conversation_history.db")

# ─── LLM context ─────────────────────────────────────────────────────────────
# Maximum tokens to include from conversation history
LLM_MAX_HISTORY_TOKENS: int = int(os.getenv("LLM_MAX_HISTORY_TOKENS", "70000"))
# How many past messages to load (hard cap before token trimming)
LLM_MAX_HISTORY_MESSAGES: int = int(os.getenv("LLM_MAX_HISTORY_MESSAGES", "100"))

# ─── Session ──────────────────────────────────────────────────────────────────
# "daily"  → new session_id per calendar day
# "launch" → new session_id per application launch
SESSION_MODE: str = os.getenv("SESSION_MODE", "daily")

# ─── Scheduler ────────────────────────────────────────────────────────────────
# Minutes before a calendar event to trigger meeting-prep alert
MEETING_PREP_MINUTES: int = int(os.getenv("MEETING_PREP_MINUTES", "30"))

# ─── UI ───────────────────────────────────────────────────────────────────────
WINDOW_ALWAYS_ON_TOP: bool = os.getenv("WINDOW_ALWAYS_ON_TOP", "true").lower() == "true"
WINDOW_WIDTH: int = int(os.getenv("WINDOW_WIDTH", "480"))
WINDOW_HEIGHT: int = int(os.getenv("WINDOW_HEIGHT", "320"))
WINDOW_OPACITY: float = float(os.getenv("WINDOW_OPACITY", "0.95"))

# ─── Chime ────────────────────────────────────────────────────────────────────
CHIME_PATH: str = str(_ROOT / "assets" / "chime.wav")
