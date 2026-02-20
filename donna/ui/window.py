"""
ui/window.py — Donna's floating assistant window (CustomTkinter).

Layout:
  ┌─────────────────────────────────────┐
  │ DONNA  ● Listening      [_] [×]     │  ← title bar
  ├─────────────────────────────────────┤
  │ Today: Tuesday, 20 Feb 2024         │  ← agenda widget (collapsed by default)
  ├─────────────────────────────────────┤
  │                                     │
  │  [conversation transcript scrollbox]│
  │                                     │
  ├─────────────────────────────────────┤
  │ [text input          ] [Send] [Mic] │  ← fallback input row
  └─────────────────────────────────────┘

Thread safety: all UI mutations must happen on the main Tk thread.
External code calls the public methods; the window queues them via after().
"""

import logging
import threading
from datetime import datetime
from typing import Callable

import customtkinter as ctk

from donna.config import (
    WINDOW_ALWAYS_ON_TOP,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    WINDOW_OPACITY,
)

logger = logging.getLogger(__name__)

# ─── Appearance ───────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_ACCENT      = "#7B2FBE"
_BG          = "#1A1A2E"
_TEXT_FG     = "#E0E0E0"
_STATUS_IDLE = "#6B6B6B"
_STATUS_ON   = "#4CAF50"
_STATUS_ERR  = "#F44336"


class DonnaWindow(ctk.CTk):
    """
    Main floating window.  Instantiate on the main thread; call mainloop()
    to start the event loop.
    """

    def __init__(
        self,
        on_send_text: Callable[[str], None],
        on_mic_toggle: Callable[[], None],
        on_close: Callable[[], None],
    ):
        super().__init__()

        self._on_send_text = on_send_text
        self._on_mic_toggle = on_mic_toggle
        self._on_close = on_close
        self._mic_active = False

        self._build_window()
        self._build_widgets()

    # ── Window chrome ─────────────────────────────────────────────────────────

    def _build_window(self) -> None:
        self.title("Donna")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.configure(bg_color=_BG, fg_color=_BG)
        self.attributes("-topmost", WINDOW_ALWAYS_ON_TOP)
        self.attributes("-alpha", WINDOW_OPACITY)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._handle_close)

    def _handle_close(self) -> None:
        self.withdraw()   # hide, don't destroy — tray icon keeps app alive
        self._on_close()

    # ── Widget construction ───────────────────────────────────────────────────

    def _build_widgets(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # ── Title row ──
        title_frame = ctk.CTkFrame(self, fg_color=_ACCENT, corner_radius=0, height=36)
        title_frame.grid(row=0, column=0, sticky="ew")
        title_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            title_frame, text="  DONNA", font=("Segoe UI", 13, "bold"),
            text_color="white", fg_color="transparent"
        ).grid(row=0, column=0, sticky="w", padx=4)

        self._status_label = ctk.CTkLabel(
            title_frame, text="● Idle", font=("Segoe UI", 10),
            text_color=_STATUS_IDLE, fg_color="transparent"
        )
        self._status_label.grid(row=0, column=1, sticky="w", padx=6)

        ctk.CTkButton(
            title_frame, text="✕", width=28, height=28, corner_radius=4,
            fg_color="transparent", hover_color="#9A3FDE",
            command=self._handle_close
        ).grid(row=0, column=2, padx=4, pady=4)

        # ── Agenda bar ──
        self._agenda_label = ctk.CTkLabel(
            self,
            text=f"  {datetime.now().strftime('%A, %d %B %Y')}",
            font=("Segoe UI", 9),
            text_color=_STATUS_IDLE,
            fg_color=("#111122", "#111122"),
            anchor="w",
            height=22,
        )
        self._agenda_label.grid(row=1, column=0, sticky="ew")

        # ── Transcript area ──
        self._transcript = ctk.CTkTextbox(
            self,
            font=("Segoe UI", 11),
            fg_color=_BG,
            text_color=_TEXT_FG,
            wrap="word",
            state="disabled",
        )
        self._transcript.grid(row=2, column=0, sticky="nsew", padx=6, pady=(4, 2))

        # ── Input row ──
        input_frame = ctk.CTkFrame(self, fg_color=_BG, corner_radius=0, height=44)
        input_frame.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 6))
        input_frame.grid_columnconfigure(0, weight=1)

        self._text_input = ctk.CTkEntry(
            input_frame,
            placeholder_text="Type a message…",
            font=("Segoe UI", 11),
            height=34,
        )
        self._text_input.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._text_input.bind("<Return>", self._handle_send)

        self._send_btn = ctk.CTkButton(
            input_frame, text="Send", width=52, height=34,
            fg_color=_ACCENT, hover_color="#9A3FDE",
            command=self._handle_send,
        )
        self._send_btn.grid(row=0, column=1, padx=(0, 4))

        self._mic_btn = ctk.CTkButton(
            input_frame, text="🎤", width=34, height=34,
            fg_color="#333355", hover_color=_ACCENT,
            command=self._handle_mic_toggle,
        )
        self._mic_btn.grid(row=0, column=2)

    # ── Event handlers (UI thread) ────────────────────────────────────────────

    def _handle_send(self, event=None) -> None:
        text = self._text_input.get().strip()
        if not text:
            return
        self._text_input.delete(0, "end")
        self.add_message("You", text)
        threading.Thread(
            target=self._on_send_text, args=(text,), daemon=True
        ).start()

    def _handle_mic_toggle(self) -> None:
        self._mic_active = not self._mic_active
        self._mic_btn.configure(fg_color=_ACCENT if self._mic_active else "#333355")
        self._on_mic_toggle()

    # ── Public API (thread-safe via after()) ──────────────────────────────────

    def add_message(self, speaker: str, text: str) -> None:
        """Append a message to the transcript.  Safe to call from any thread."""
        self.after(0, self._append_transcript, speaker, text)

    def _append_transcript(self, speaker: str, text: str) -> None:
        self._transcript.configure(state="normal")
        ts = datetime.now().strftime("%H:%M")
        self._transcript.insert(
            "end",
            f"\n[{ts}] {speaker}:\n{text}\n",
        )
        self._transcript.configure(state="disabled")
        self._transcript.see("end")

    def set_status(self, status: str, colour: str | None = None) -> None:
        """Update the status indicator.  Safe to call from any thread."""
        self.after(0, self._set_status_main, status, colour)

    def _set_status_main(self, status: str, colour: str | None) -> None:
        fg = colour or _STATUS_IDLE
        self._status_label.configure(text=f"● {status}", text_color=fg)

    def set_agenda(self, text: str) -> None:
        """Update the agenda bar.  Safe to call from any thread."""
        self.after(0, self._agenda_label.configure, {"text": f"  {text}"})

    def set_listening(self, active: bool) -> None:
        """Convenience: toggle listening indicator."""
        if active:
            self.set_status("Listening…", _STATUS_ON)
        else:
            self.set_status("Idle", _STATUS_IDLE)

    def set_thinking(self, active: bool) -> None:
        if active:
            self.set_status("Thinking…", "#FFA500")
        else:
            self.set_status("Idle", _STATUS_IDLE)

    def set_speaking(self, active: bool) -> None:
        if active:
            self.set_status("Speaking…", "#2196F3")
        else:
            self.set_status("Idle", _STATUS_IDLE)

    def show(self) -> None:
        self.after(0, self.deiconify)
        self.after(0, self.lift)

    def hide(self) -> None:
        self.after(0, self.withdraw)
