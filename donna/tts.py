"""
tts.py — Kokoro TTS engine with bf_emma British female voice.

Features:
- Local inference — no network calls
- Interruptible: stop mid-sentence when wake word fires
- Configurable voice and speed via config.py
- Streams audio chunk-by-chunk for low latency to first word
"""

import threading
import logging
import numpy as np

import sounddevice as sd

from donna.config import TTS_VOICE, TTS_SPEED

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
