"""
stt.py — Speech-to-Text pipeline using faster-whisper + webrtcvad.

Flow:
1. Record audio from mic using PyAudio
2. Apply VAD to detect end-of-speech (configurable silence threshold)
3. Transcribe with faster-whisper (local, offline)
4. Return transcript string

This module runs synchronously when called; the caller is responsible for
threading if non-blocking operation is needed.
"""

import logging
import collections
import struct
import wave
import io
import threading
from typing import Callable

import pyaudio
import webrtcvad
import numpy as np

from donna.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    VAD_MODE,
    VAD_SAMPLE_RATE,
    VAD_FRAME_DURATION_MS,
    VAD_SILENCE_FRAMES,
    AUDIO_CHANNELS,
    AUDIO_CHUNK_FRAMES,
)

logger = logging.getLogger(__name__)

# ─── Lazy Whisper model ───────────────────────────────────────────────────────

_whisper_model = None
_whisper_lock = threading.Lock()


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                from faster_whisper import WhisperModel  # type: ignore
                logger.info(
                    "Loading Whisper model '%s' on %s…",
                    WHISPER_MODEL_SIZE,
                    WHISPER_DEVICE,
                )
                _whisper_model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device=WHISPER_DEVICE,
                    compute_type=WHISPER_COMPUTE_TYPE,
                )
                logger.info("Whisper model loaded.")
    return _whisper_model


# ─── Audio helpers ────────────────────────────────────────────────────────────

def _frame_duration_to_samples(duration_ms: int, sample_rate: int) -> int:
    return int(sample_rate * duration_ms / 1000)


def _pcm_to_wav_bytes(pcm_frames: list[bytes], sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(AUDIO_CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(pcm_frames))
    return buf.getvalue()


# ─── VAD-gated recording ─────────────────────────────────────────────────────

def record_until_silence(
    timeout_seconds: float = 15.0,
    stop_flag: threading.Event | None = None,
) -> bytes | None:
    """
    Record from the default mic until VAD detects end-of-speech.

    Returns:
        Raw WAV bytes ready for Whisper, or None if no speech detected /
        timeout reached / stop_flag was set.
    """
    vad = webrtcvad.Vad(VAD_MODE)
    frame_samples = _frame_duration_to_samples(VAD_FRAME_DURATION_MS, VAD_SAMPLE_RATE)
    frame_bytes = frame_samples * 2  # 16-bit = 2 bytes per sample

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=AUDIO_CHANNELS,
        rate=VAD_SAMPLE_RATE,
        input=True,
        frames_per_buffer=frame_samples,
    )

    # Ring buffer for pre-speech padding (keeps ~300 ms before speech starts)
    padding_frames = 10
    ring_buffer: collections.deque[bytes] = collections.deque(maxlen=padding_frames)
    triggered = False
    voiced_frames: list[bytes] = []
    silence_count = 0
    total_frames = 0
    max_frames = int(timeout_seconds * 1000 / VAD_FRAME_DURATION_MS)

    try:
        logger.debug("VAD recording started.")
        while total_frames < max_frames:
            if stop_flag and stop_flag.is_set():
                logger.debug("STT recording cancelled by stop flag.")
                return None

            raw = stream.read(frame_samples, exception_on_overflow=False)
            total_frames += 1

            is_speech = False
            try:
                is_speech = vad.is_speech(raw, VAD_SAMPLE_RATE)
            except Exception:
                pass

            if not triggered:
                ring_buffer.append(raw)
                if is_speech:
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
                    silence_count = 0
            else:
                voiced_frames.append(raw)
                if is_speech:
                    silence_count = 0
                else:
                    silence_count += 1
                    if silence_count > VAD_SILENCE_FRAMES:
                        logger.debug(
                            "End-of-speech detected after %d voiced frames.", len(voiced_frames)
                        )
                        break

        logger.info(
            "VAD result: %d total frames, triggered=%s, %d voiced frames collected.",
            total_frames, triggered, len(voiced_frames),
        )

        if not voiced_frames:
            logger.info("No speech detected within timeout.")
            return None

        return _pcm_to_wav_bytes(voiced_frames, VAD_SAMPLE_RATE)

    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


# ─── Transcription ────────────────────────────────────────────────────────────

def transcribe(wav_bytes: bytes) -> str:
    """
    Transcribe WAV audio bytes using faster-whisper.

    Returns:
        Transcribed text, stripped and lowercased.
    """
    model = _get_whisper()
    # faster-whisper accepts a file path or a file-like object
    audio_io = io.BytesIO(wav_bytes)
    segments, _info = model.transcribe(
        audio_io,
        beam_size=5,
        language="en",
        task="transcribe",
        vad_filter=False,  # we already did VAD
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    logger.info("Transcribed: %r", text)
    return text


def listen_and_transcribe(
    timeout_seconds: float = 15.0,
    stop_flag: threading.Event | None = None,
    on_listening: Callable[[], None] | None = None,
) -> str:
    """
    High-level convenience: record until silence, then transcribe.

    Args:
        timeout_seconds: Max recording duration.
        stop_flag:       Set this event to abort recording early.
        on_listening:    Callback fired just before recording starts (for UI).

    Returns:
        Transcribed text, or "" if nothing was detected.
    """
    if on_listening:
        on_listening()
    wav = record_until_silence(timeout_seconds=timeout_seconds, stop_flag=stop_flag)
    if not wav:
        return ""
    return transcribe(wav)
