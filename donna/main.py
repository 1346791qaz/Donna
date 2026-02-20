"""
main.py — Donna application entry point.

Lifecycle:
  1. Initialise databases
  2. Start system tray
  3. Launch ProactiveScheduler
  4. Start WakeWordEngine (background thread)
  5. Open the floating UI window (main thread — Tkinter requirement)

The voice interaction loop runs in a dedicated thread:
  WakeWord detected → STT → LLM → TTS → back to WakeWord

Text input from the UI bypasses wake word and STT,
going directly: text → LLM → TTS.
"""

import logging
import sys
import threading
import os
from pathlib import Path

# ─── Data directory must exist before logging tries to open the log file ──────

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "data"
_DATA_DIR.mkdir(exist_ok=True)

# Ensure the repo root is on sys.path so `donna` is importable regardless
# of which directory the user launches from.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ─── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            _DATA_DIR / "donna.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("donna.main")


def _ensure_data_dir() -> None:
    _DATA_DIR.mkdir(exist_ok=True)


_ensure_data_dir()

# ─── Application imports (after logging is set) ───────────────────────────────

from donna.config import PICOVOICE_ACCESS_KEY, WAKE_WORD_MODEL_PATH, CHIME_PATH
from donna.db import contacts_db, conversation_db
from donna.ui.window import DonnaWindow
from donna.ui.tray import SystemTray
from donna import tts, stt
from donna.scheduler import ProactiveScheduler
from donna import llm

# ─── Global state ─────────────────────────────────────────────────────────────

_window: DonnaWindow | None = None
_wake_engine = None
_scheduler: ProactiveScheduler | None = None
_muted = False
_voice_stop_flag = threading.Event()
_interaction_lock = threading.Lock()  # prevent overlapping interactions


# ─── Audio helpers ────────────────────────────────────────────────────────────

def _play_chime() -> None:
    try:
        import soundfile as sf
        import sounddevice as sd
        import numpy as np
        if Path(CHIME_PATH).exists():
            data, sr = sf.read(CHIME_PATH)
            sd.play(np.array(data, dtype=np.float32), samplerate=sr, blocking=False)
    except Exception:
        pass  # chime is nice-to-have; never block on it


# ─── Interaction pipeline ─────────────────────────────────────────────────────

def _on_proactive_response(text: str) -> None:
    """Called by the scheduler when a proactive message is generated."""
    if _window:
        _window.add_message("Donna", text)
    tts.speak(text, block=False)


def _handle_llm_response(user_text: str) -> None:
    """Run LLM + TTS for a given user message.  Always called from a thread."""
    if _window:
        _window.set_thinking(True)
    try:
        def _on_tool(name: str, _input: dict) -> None:
            logger.debug("Tool: %s", name)

        response = llm.chat(user_text, on_tool_call=_on_tool)
        if _window:
            _window.add_message("Donna", response)
            _window.set_speaking(True)
        tts.speak(response, block=True)
    except Exception:
        logger.exception("LLM/TTS pipeline error")
        if _window:
            _window.add_message("Donna", "Sorry, something went wrong. Please try again.")
    finally:
        if _window:
            _window.set_thinking(False)
            _window.set_speaking(False)
            _window.set_listening(False)


def _on_wake() -> None:
    """Called by WakeWordEngine (from its background thread) on detection."""
    if _muted:
        return
    if not _interaction_lock.acquire(blocking=False):
        logger.debug("Wake word ignored — interaction already in progress.")
        return
    try:
        # Interrupt any ongoing speech
        tts.interrupt()
        _play_chime()

        if _window:
            _window.set_listening(True)

        transcript = stt.listen_and_transcribe(
            timeout_seconds=15.0,
            stop_flag=_voice_stop_flag,
        )

        if not transcript:
            logger.info("No speech detected after wake word.")
            if _window:
                _window.set_listening(False)
                _window.set_status("No speech detected", "#FFA500")
            return

        logger.info("User said: %r", transcript)
        if _window:
            _window.add_message("You", transcript)
            _window.set_listening(False)

        _handle_llm_response(transcript)
    finally:
        _interaction_lock.release()


def _on_text_input(text: str) -> None:
    """Called when the user sends text via the UI input field."""
    if not _interaction_lock.acquire(blocking=False):
        logger.debug("Text input ignored — interaction in progress.")
        return
    try:
        tts.interrupt()
        _handle_llm_response(text)
    finally:
        _interaction_lock.release()


def _on_mic_toggle() -> None:
    global _muted
    _muted = not _muted
    state = "muted" if _muted else "unmuted"
    logger.info("Microphone %s.", state)
    if _window:
        _window.set_status(f"Mic {state}", "#FFA500" if _muted else None)


# ─── Tray callbacks ───────────────────────────────────────────────────────────

def _on_tray_exit() -> None:
    logger.info("Exit requested from tray.")
    _shutdown()


def _on_show_window() -> None:
    if _window:
        _window.show()


def _on_hide_window() -> None:
    if _window:
        _window.hide()


# ─── Shutdown ─────────────────────────────────────────────────────────────────

def _shutdown() -> None:
    global _wake_engine, _scheduler
    logger.info("Donna shutting down…")
    _voice_stop_flag.set()
    tts.interrupt()
    if _wake_engine:
        _wake_engine.stop()
    if _scheduler:
        _scheduler.stop()
    if _window:
        _window.quit()
    sys.exit(0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _window, _wake_engine, _scheduler

    logger.info("Starting Donna…")

    # 1. Databases
    contacts_db.init_db()
    conversation_db.init_db()

    # 2. Build UI (must happen on main thread before mainloop)
    _window = DonnaWindow(
        on_send_text=lambda t: threading.Thread(
            target=_on_text_input, args=(t,), daemon=True
        ).start(),
        on_mic_toggle=_on_mic_toggle,
        on_close=_on_hide_window,
    )

    # 3. System tray
    tray = SystemTray(
        on_show_window=_on_show_window,
        on_hide_window=_on_hide_window,
        on_exit=_on_tray_exit,
        on_toggle_mute=_on_mic_toggle,
    )
    tray.start_threaded()

    # 4. Proactive scheduler
    _scheduler = ProactiveScheduler(on_response=_on_proactive_response)
    _scheduler.start()

    # 5. Wake word engine
    wake_model_exists = Path(WAKE_WORD_MODEL_PATH).exists()
    pico_key_set = bool(PICOVOICE_ACCESS_KEY)

    if wake_model_exists and pico_key_set:
        from donna.wake_word import WakeWordEngine
        _wake_engine = WakeWordEngine(on_wake=_on_wake)
        try:
            _wake_engine.start()
        except Exception:
            logger.exception(
                "Wake word engine failed to start. Voice activation disabled."
            )
            _wake_engine = None
    else:
        logger.warning(
            "Wake word engine NOT started. "
            "Set PICOVOICE_ACCESS_KEY and WAKE_WORD_MODEL_PATH in .env "
            "and train a 'Hey Donna' model at console.picovoice.ai."
        )

    # 6. Enter main event loop (blocks until window is destroyed)
    logger.info("Donna is ready.")
    _window.mainloop()


if __name__ == "__main__":
    main()
