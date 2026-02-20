"""
wake_word.py — Always-on "Hey Donna" wake word detection via Porcupine.

Design:
- Runs in a persistent background thread
- On detection: calls the registered callback, fires interrupt() on TTS,
  then the main pipeline takes over (STT → LLM → TTS)
- Low CPU: Porcupine is optimised for always-on embedded use
- Graceful shutdown via stop()

Prerequisites:
- Picovoice account + free API key (PICOVOICE_ACCESS_KEY in .env)
- Trained "Hey Donna" .ppn keyword file (WAKE_WORD_MODEL_PATH in .env)
  Train at: https://console.picovoice.ai/ → Wake Word → Create
"""

import logging
import struct
import threading
from typing import Callable

import pyaudio

from donna.config import (
    PICOVOICE_ACCESS_KEY,
    WAKE_WORD_MODEL_PATH,
    WAKE_WORD_SENSITIVITY,
    AUDIO_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

# Porcupine requires exactly 512-sample frames at 16 kHz
_PORCUPINE_FRAME_LENGTH = 512


class WakeWordEngine:
    """
    Background thread that listens for the "Hey Donna" wake word.

    Usage:
        engine = WakeWordEngine(on_wake=my_callback)
        engine.start()
        ...
        engine.stop()
    """

    def __init__(self, on_wake: Callable[[], None]):
        """
        Args:
            on_wake: Callable invoked (from the background thread) each time
                     the wake word is detected.
        """
        self._on_wake = on_wake
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._porcupine = None
        self._pa: pyaudio.PyAudio | None = None
        self._stream = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Initialise Porcupine and start the background listening thread."""
        import pvporcupine  # type: ignore

        logger.info("Initialising Porcupine wake word engine…")
        self._porcupine = pvporcupine.create(
            access_key=PICOVOICE_ACCESS_KEY,
            keyword_paths=[WAKE_WORD_MODEL_PATH],
            sensitivities=[WAKE_WORD_SENSITIVITY],
        )

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            rate=self._porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self._porcupine.frame_length,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="WakeWordEngine"
        )
        self._thread.start()
        logger.info("Wake word engine listening for 'Hey Donna'.")

    def pause_stream(self) -> None:
        """Close the mic stream AND terminate PyAudio so STT can claim the device.

        On Windows WASAPI, leaving the PyAudio host-API instance alive while
        a second instance opens the same device results in the new stream
        returning silence.  Terminating here ensures only one PortAudio
        instance owns the device at a time.

        Safe to call from within the on_wake callback (i.e. from this engine's
        own thread) because _run() is blocked inside on_wake at that point.
        """
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def resume_stream(self) -> None:
        """Reopen the mic stream after STT has released the microphone."""
        if self._porcupine is None:
            return
        try:
            if self._pa is None:
                self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                rate=self._porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self._porcupine.frame_length,
            )
        except Exception:
            logger.exception("Failed to reopen wake word audio stream after STT.")

    def stop(self) -> None:
        """Signal the background thread to stop and clean up resources."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._cleanup()

    def _cleanup(self) -> None:
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
        if self._porcupine:
            try:
                self._porcupine.delete()
            except Exception:
                pass
        self._stream = None
        self._pa = None
        self._porcupine = None

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _run(self) -> None:
        logger.debug("WakeWordEngine thread started.")
        try:
            while not self._stop_event.is_set():
                pcm = self._stream.read(
                    self._porcupine.frame_length, exception_on_overflow=False
                )
                pcm_unpacked = struct.unpack_from(
                    f"{self._porcupine.frame_length}h", pcm
                )
                result = self._porcupine.process(pcm_unpacked)
                if result >= 0:
                    logger.info("Wake word detected (keyword index %d).", result)
                    try:
                        self._on_wake()
                    except Exception:
                        logger.exception("on_wake callback raised an exception.")
        except Exception:
            logger.exception("WakeWordEngine loop terminated with error.")
        finally:
            logger.debug("WakeWordEngine thread exiting.")


# ─── Fuzzy wake word fallback ─────────────────────────────────────────────────

_WAKE_WORD_VARIANTS = {
    "hey donna", "donna", "hey don", "yo donna", "hi donna", "ok donna"
}


def is_wake_phrase(text: str) -> bool:
    """
    Return True if `text` (post-STT transcript) matches a wake word variant.
    Used as fallback when Porcupine model is unavailable.
    """
    text_lower = text.lower().strip().rstrip(".,!?")
    if text_lower in _WAKE_WORD_VARIANTS:
        return True
    # Partial prefix match (handles "Hey Donna, what's…")
    for variant in _WAKE_WORD_VARIANTS:
        if text_lower.startswith(variant):
            return True
    return False
