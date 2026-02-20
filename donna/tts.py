"""
tts.py — Kokoro TTS engine with bf_emma British female voice.

Features:
- Local inference — no network calls
- Interruptible: stop mid-sentence when wake word fires
- Configurable voice and speed via config.py
- Streams audio chunk-by-chunk for low latency to first word
"""

import re
import threading
import logging
import numpy as np

import sounddevice as sd

from donna.config import TTS_VOICE, TTS_SPEED


def sanitize_for_tts(text: str) -> str:
    """Strip markdown and normalise text so it reads naturally when spoken.

    Removes bold/italic markers, bullet symbols, emoji, and other formatting
    that TTS engines vocalise literally (e.g. "asterisk asterisk").
    Also normalises time strings so "6:00 PM" is spoken as "6 PM".
    """
    # ── Code blocks and inline code ──────────────────────────────────────────
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"`[^`]*`", "", text)

    # ── Markdown headers (# Heading) ─────────────────────────────────────────
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # ── Bold + italic (***text*** / ___text___) then bold then italic ─────────
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_(.+?)_", r"\1", text, flags=re.DOTALL)

    # ── Markdown links [label](url) → label ──────────────────────────────────
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # ── Horizontal rules ─────────────────────────────────────────────────────
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # ── Bullet list markers (- item / * item) ────────────────────────────────
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)

    # ── Numbered list markers (1. item) ──────────────────────────────────────
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # ── Emoji and pictographic symbols ───────────────────────────────────────
    # Covers Misc Symbols (2600-26FF), Dingbats (2700-27BF),
    # Supplemental Symbols (1F300-1F9FF), and variation selectors
    text = re.sub(r"[\u2600-\u27BF]", "", text)
    text = re.sub(r"[\U0001F000-\U0001FFFF]", "", text)
    text = re.sub(r"[\uFE00-\uFE0F]", "", text)  # variation selectors

    # ── Em/en dashes → natural pause ─────────────────────────────────────────
    text = text.replace("\u2014", ", ")  # em dash —
    text = text.replace("\u2013", ", ")  # en dash –

    # ── Times: "6:00PM" / "6:00 PM" → "6 PM"; "6:30 PM" stays "6:30 PM" ────
    def _fmt_time(m: re.Match) -> str:
        hour, mins, ampm = m.group(1), m.group(2), m.group(3)
        return f"{hour} {ampm.upper()}" if mins == "00" else f"{hour}:{mins} {ampm.upper()}"

    text = re.sub(
        r"\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)\b",
        _fmt_time,
        text,
    )

    # ── Collapse excess blank lines ───────────────────────────────────────────
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

logger = logging.getLogger(__name__)

# Lazily initialise the Kokoro pipeline to avoid slow import at startup
_pipeline = None
_pipeline_lock = threading.Lock()

# Global stop event — set to interrupt ongoing speech
_stop_event = threading.Event()


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                # Import here so startup isn't blocked by model load
                from kokoro import KPipeline  # type: ignore
                logger.info("Loading Kokoro TTS pipeline (first call)…")
                _pipeline = KPipeline(lang_code="b")
                logger.info("Kokoro TTS pipeline ready.")
    return _pipeline


def speak(
    text: str,
    voice: str | None = None,
    speed: float | None = None,
    block: bool = True,
) -> None:
    """
    Synthesise `text` and play it through the default audio output.

    Args:
        text:  Text to speak.
        voice: Kokoro voice ID (default: config TTS_VOICE).
        speed: Speech rate multiplier (default: config TTS_SPEED).
        block: If True, return only when playback is finished (or interrupted).
               If False, launch in a background thread and return immediately.
    """
    voice = voice or TTS_VOICE
    speed = speed if speed is not None else TTS_SPEED
    text = sanitize_for_tts(text)

    if not block:
        t = threading.Thread(target=_speak_blocking, args=(text, voice, speed), daemon=True)
        t.start()
        return

    _speak_blocking(text, voice, speed)


def _speak_blocking(text: str, voice: str, speed: float) -> None:
    _stop_event.clear()
    try:
        pipeline = _get_pipeline()
        # Kokoro KPipeline yields (graphemes, phonemes, audio_array) tuples.
        # Sample rate is fixed at 24 kHz.
        generator = pipeline(text, voice=voice, speed=speed)
        for _gs, _ps, audio in generator:
            if _stop_event.is_set():
                logger.debug("TTS interrupted.")
                break

            samples_np = np.array(audio, dtype=np.float32)
            peak = np.abs(samples_np).max()
            if peak > 1.0:
                samples_np /= peak

            sd.play(samples_np, samplerate=24000, blocking=True)
    except Exception:
        logger.exception("TTS playback error")


def interrupt() -> None:
    """Stop any ongoing speech immediately."""
    _stop_event.set()
    try:
        sd.stop()
    except Exception:
        pass


def is_speaking() -> bool:
    """Return True while audio is actively being played."""
    return sd.get_stream() is not None and not _stop_event.is_set()
