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
    STT_ENERGY_THRESHOLD,
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
    silence_frames_override: int | None = None,
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
    # Diagnostic: log default device and device list to help diagnose silent input
    try:
        default_info = pa.get_default_input_device_info()
        logger.info(
            "PyAudio default input device: index=%s name=%r",
            default_info.get("index", "n/a"),
            default_info.get("name", "n/a"),
        )
    except Exception:
        logger.debug("No default input device info available from PyAudio.")
    try:
        device_count = pa.get_device_count()
        devices = []
        for i in range(device_count):
            try:
                info = pa.get_device_info_by_index(i)
                devices.append((i, info.get("name", ""), info.get("maxInputChannels", 0)))
            except Exception:
                continue
        logger.debug("PyAudio devices: %s", devices)
    except Exception:
        logger.debug("Failed to enumerate PyAudio devices.")

    # If config specifies an input device index, use it (helps on multi-device systems)
    from donna.config import AUDIO_INPUT_DEVICE_INDEX
    stream_kwargs = dict(
        format=pyaudio.paInt16,
        channels=AUDIO_CHANNELS,
        rate=VAD_SAMPLE_RATE,
        input=True,
        frames_per_buffer=frame_samples,
    )
    if AUDIO_INPUT_DEVICE_INDEX is not None:
        stream_kwargs["input_device_index"] = AUDIO_INPUT_DEVICE_INDEX
        logger.info("Opening STT stream using configured input device index %s", AUDIO_INPUT_DEVICE_INDEX)
    else:
        logger.info("Opening STT stream using default input device")

    stream = pa.open(**stream_kwargs)

    # Warm-up reads: some host APIs return silence for the very first frames
    # after reopening the device. Read & inspect a few frames and, if they
    # appear all near-zero, attempt a single reopen to recover.
    try:
        warmup_zero = True
        warmup_count = 3
        for i in range(warmup_count):
            wraw = stream.read(frame_samples, exception_on_overflow=False)
            try:
                wsamples = np.frombuffer(wraw, dtype=np.int16).astype(np.float32)
            except Exception:
                try:
                    wsamples = np.frombuffer(wraw, dtype=np.float32).astype(np.float32)
                except Exception:
                    wsamples = np.array([], dtype=np.float32)
            if wsamples.size and float(np.max(np.abs(wsamples))) >= 1.0:
                warmup_zero = False
            logger.debug("STT warmup frame %d rms_est=%.3f", i, float(np.sqrt(np.mean((wsamples.astype(np.float32)) ** 2))) if wsamples.size else 0.0)

        if warmup_zero:
            logger.warning("STT warmup detected near-zero input frames; retrying stream open once.")
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            # Recreate Pa and reopen stream
            try:
                pa.terminate()
            except Exception:
                pass
            pa = pyaudio.PyAudio()
            stream = pa.open(**stream_kwargs)
            logger.info("STT stream reopened after warmup retry.")
            # If we still get silent frames, try enumerating devices and
            # attempt to open each available input device to find one that
            # produces non-zero audio. This is a pragmatic fallback for
            # systems where the default device index becomes inactive after
            # PyAudio/driver changes.
            try:
                # Quick check after reopen
                rraw = stream.read(frame_samples, exception_on_overflow=False)
                try:
                    r_samples = np.frombuffer(rraw, dtype=np.int16).astype(np.float32)
                except Exception:
                    try:
                        r_samples = np.frombuffer(rraw, dtype=np.float32).astype(np.float32)
                    except Exception:
                        r_samples = np.array([], dtype=np.float32)
                if not (r_samples.size and float(np.max(np.abs(r_samples))) >= 1.0):
                    logger.warning("STT stream still near-zero after reopen; scanning other input devices.")
                    try:
                        stream.stop_stream(); stream.close()
                    except Exception:
                        pass
                    pa.terminate()
                    pa = pyaudio.PyAudio()
                    device_count = pa.get_device_count()
                    found = False
                    for di in range(device_count):
                        try:
                            info = pa.get_device_info_by_index(di)
                        except Exception:
                            continue
                        if info.get("maxInputChannels", 0) <= 0:
                            continue
                        logger.info("Trying input device %d: %r", di, info.get("name", ""))
                        try:
                            test_stream = pa.open(input_device_index=di, **stream_kwargs)
                            try:
                                traw = test_stream.read(frame_samples, exception_on_overflow=False)
                                try:
                                    tsamples = np.frombuffer(traw, dtype=np.int16).astype(np.float32)
                                except Exception:
                                    try:
                                        tsamples = np.frombuffer(traw, dtype=np.float32).astype(np.float32)
                                    except Exception:
                                        tsamples = np.array([], dtype=np.float32)
                                if tsamples.size and float(np.max(np.abs(tsamples))) >= 1.0:
                                    logger.info("Selected input device %d: %r", di, info.get("name", ""))
                                    # Use this device for the main stream
                                    try:
                                        test_stream.stop_stream(); test_stream.close()
                                    except Exception:
                                        pass
                                    stream = pa.open(input_device_index=di, **stream_kwargs)
                                    found = True
                                    break
                            finally:
                                try:
                                    test_stream.stop_stream(); test_stream.close()
                                except Exception:
                                    pass
                        except Exception:
                            continue
                    if not found:
                        logger.warning("No alternative input device produced non-zero frames; continuing with current stream.")
            except Exception:
                logger.exception("Error during STT reopen/device-scan fallback.")
    except Exception:
        logger.exception("STT warmup check failed (continuing anyway).")

    # Ring buffer for pre-speech padding (keeps ~300 ms before speech starts)
    padding_frames = 10
    ring_buffer: collections.deque[bytes] = collections.deque(maxlen=padding_frames)
    triggered = False
    voiced_frames: list[bytes] = []
    silence_count = 0
    total_frames = 0
    max_frames = int(timeout_seconds * 1000 / VAD_FRAME_DURATION_MS)
    # Allow caller to override how many consecutive silence frames are
    # required before declaring end-of-speech (useful for longer post-speech
    # silence windows).
    silence_target = (
        silence_frames_override if silence_frames_override is not None else VAD_SILENCE_FRAMES
    )

    _vad_exc_logged = False

    try:
        logger.debug("VAD recording started.")
        while total_frames < max_frames:
            if stop_flag and stop_flag.is_set():
                logger.debug("STT recording cancelled by stop flag.")
                return None

            raw = stream.read(frame_samples, exception_on_overflow=False)
            total_frames += 1

            # Energy-based speech detection (works even if webrtcvad misbehaves)
            # Read raw bytes and interpret as either int16 PCM or float32 PCM
            # Some audio backends may return normalized float samples (-1..1).
            try:
                samples_raw = np.frombuffer(raw, dtype=np.int16)
                samples = samples_raw.astype(np.float32)
                raw_format = "int16"
            except Exception:
                # Fallback: try float32
                try:
                    samples = np.frombuffer(raw, dtype=np.float32).astype(np.float32)
                    raw_format = "float32"
                except Exception:
                    samples = np.array([], dtype=~np.float32)
                    raw_format = "unknown"

            # If samples are floats in -1..1, scale to int16-equivalent units so
            # STT_ENERGY_THRESHOLD continues to work (threshold expects ~300).
            if samples.size > 0 and raw_format == "float32":
                max_abs = float(np.max(np.abs(samples)))
                if max_abs <= 1.1:  # likely normalized floats
                    samples_for_rms = samples * 32768.0
                    scaled = True
                else:
                    samples_for_rms = samples
                    scaled = False
            else:
                samples_for_rms = samples
                scaled = False

            rms = float(np.sqrt(np.mean(samples_for_rms ** 2))) if samples_for_rms.size > 0 else 0.0

            if total_frames == 1:
                logger.info(
                    "STT stream first-frame RMS=%.3f  (0 → stream is silent/closed); frames=%d bytes=%d; threshold=%s; fmt=%s scaled=%s",
                    rms,
                    frame_samples,
                    len(raw),
                    STT_ENERGY_THRESHOLD,
                    raw_format,
                    scaled,
                )
                logger.debug(
                    "STT first-sample stats: dtype=%s min=%s max=%s first5=%s",
                    samples.dtype if samples.size else "none",
                    float(samples.min()) if samples.size else None,
                    float(samples.max()) if samples.size else None,
                    samples[:5].tolist() if samples.size else [],
                )
            # If samples are effectively all zero, log a warning to indicate an input problem
            if samples_for_rms.size > 0 and float(np.max(np.abs(samples_for_rms))) < 1.0:
                logger.warning(
                    "STT frame contains near-zero samples (max abs < 1). This usually indicates the selected input device is silent or wrong."
                )

            is_speech = rms > STT_ENERGY_THRESHOLD
            try:
                is_speech = is_speech or vad.is_speech(raw, VAD_SAMPLE_RATE)
            except Exception as exc:
                if not _vad_exc_logged:
                    logger.warning("webrtcvad raised on frame %d: %s", total_frames, exc)
                    _vad_exc_logged = True

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
                    if silence_count > silence_target:
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
