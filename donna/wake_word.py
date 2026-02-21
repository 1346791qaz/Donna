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
import collections

import numpy as np

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
        self._device_index: int | None = None

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
        try:
            default_info = self._pa.get_default_input_device_info()
            self._device_index = default_info.get("index")
        except Exception:
            self._device_index = None

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
        logger.debug("WakeWordEngine pausing audio stream.")
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        # Do NOT terminate the PyAudio instance here. Closing the stream is
        # sufficient on most systems. Terminating the host API can cause the
        # device to reinitialize and return silence when reopened. If a full
        # termination is required it will be handled by `_cleanup()` on stop().

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

    def capture_audio_for_stt(
        self,
        timeout_seconds: float = 15.0,
        stop_flag: threading.Event | None = None,
        silence_frames_override: int | None = None,
    ) -> bytes | None:
        """Capture microphone audio from the existing wake-word stream for STT.

        Reads raw PCM frames directly from the wake-word `PyAudio` stream and
        performs VAD to collect voiced frames, returning WAV bytes suitable
        for transcription. Safe to call from within the wake-word thread.
        """
        try:
            import webrtcvad
            import io
            import wave
            from donna.config import (
                VAD_MODE,
                VAD_SAMPLE_RATE,
                VAD_FRAME_DURATION_MS,
                VAD_SILENCE_FRAMES,
                STT_ENERGY_THRESHOLD,
                AUDIO_CHANNELS,
            )
        except Exception:
            logger.exception("Failed to import VAD dependencies for capture_audio_for_stt.")
            return None

        if self._stream is None:
            logger.warning("WakeWordEngine stream not available for capture.")
            return None

        vad = webrtcvad.Vad(VAD_MODE)
        frame_samples = int(VAD_SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)

        padding_frames = 10
        ring_buffer: collections.deque[bytes] = collections.deque(maxlen=padding_frames)
        triggered = False
        voiced_frames: list[bytes] = []
        silence_count = 0
        total_frames = 0
        max_frames = int(timeout_seconds * 1000 / VAD_FRAME_DURATION_MS)
        silence_target = silence_frames_override if silence_frames_override is not None else VAD_SILENCE_FRAMES

        try:
            while total_frames < max_frames:
                if stop_flag and stop_flag.is_set():
                    return None

                raw = self._stream.read(frame_samples, exception_on_overflow=False)
                total_frames += 1

                # Interpret raw frames as int16 or float32 and normalize to
                # int16-equivalent units so the configured energy threshold
                # (which expects values ~300) makes sense.
                raw_format = "int16"
                samples_for_rms = np.array([], dtype=np.float32)
                vad_bytes = raw
                try:
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                    raw_format = "int16"
                    samples_for_rms = samples
                    vad_bytes = raw
                except Exception:
                    try:
                        samples = np.frombuffer(raw, dtype=np.float32).astype(np.float32)
                        raw_format = "float32"
                        # If normalized floats in -1..1, scale to int16-equivalent
                        max_abs = float(np.max(np.abs(samples))) if samples.size else 0.0
                        if max_abs <= 1.1:
                            samples_for_rms = samples * 32768.0
                            scaled = True
                        else:
                            samples_for_rms = samples
                            scaled = False
                        # Convert to int16 bytes for VAD
                        try:
                            int16_bytes = (samples_for_rms.astype(np.int16)).tobytes()
                            vad_bytes = int16_bytes
                        except Exception:
                            vad_bytes = raw
                    except Exception:
                        samples_for_rms = np.array([], dtype=np.float32)
                        vad_bytes = raw

                rms = float(np.sqrt(np.mean(samples_for_rms ** 2))) if samples_for_rms.size else 0.0

                if total_frames == 1:
                    logger.info(
                        "Wake-engine STT first-frame RMS=%.3f (fmt=%s)",
                        rms,
                        raw_format,
                    )

                is_speech = rms > STT_ENERGY_THRESHOLD
                try:
                    is_speech = is_speech or vad.is_speech(vad_bytes, VAD_SAMPLE_RATE)
                except Exception:
                    pass

                if not triggered:
                    ring_buffer.append(vad_bytes)
                    if is_speech:
                        triggered = True
                        voiced_frames.extend(ring_buffer)
                        ring_buffer.clear()
                        silence_count = 0
                else:
                    voiced_frames.append(vad_bytes)
                    if is_speech:
                        silence_count = 0
                    else:
                        silence_count += 1
                        if silence_count > silence_target:
                            break

            logger.info(
                "Wake-engine VAD result: %d total frames, triggered=%s, %d voiced frames collected (silence_target=%s).",
                total_frames,
                triggered,
                len(voiced_frames),
                silence_target,
            )

            if not voiced_frames:
                return None

            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(AUDIO_CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(VAD_SAMPLE_RATE)
                wf.writeframes(b"".join(voiced_frames))
            return buf.getvalue()
        except Exception:
            logger.exception("Error capturing audio from wake-word stream.")
            return None


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
